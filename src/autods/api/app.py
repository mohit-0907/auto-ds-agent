"""The FastAPI application: HTTP routes over RunManager. No agent logic here (invariant 9).

    POST /api/runs            upload a dataset + description -> start a run (202)
    GET  /api/runs            run history, newest first
    GET  /api/runs/{run_id}   one run: status, stage events, and results once finished
    GET  /api/runs/{run_id}/events   live stage events as Server-Sent Events
    GET  /api/runs/{run_id}/files/{name}   download one run artifact (allowlisted names)
    GET  /api/health          liveness check

Interactive documentation is generated automatically at /docs.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from autods.api.runs import (
    DOWNLOADABLE,
    RunConflictError,
    RunListItem,
    RunManager,
    RunSummary,
)
from autods.config import (
    MAX_DATASET_MB,
    RUNS_DIR,
    SSE_KEEPALIVE_S,
    SSE_POLL_INTERVAL_S,
    SUPPORTED_EXTENSIONS,
)
from autods.graph import new_run_dir

_CHUNK = 1024 * 1024  # uploads are copied to disk 1 MB at a time, never held whole in memory
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class RunCreated(BaseModel):
    """Reply to POST /api/runs: where to look for progress."""

    run_id: str
    status: str


def _safe_filename(name: str) -> str:
    """Keep only the base name and harmless characters ("../../x y.csv" -> "x_y.csv")."""
    base = Path(name).name
    cleaned = _UNSAFE_CHARS.sub("_", base).strip("._")
    return cleaned or "dataset"


def _save_upload(upload: UploadFile, destination: Path, limit_mb: float) -> None:
    """Stream the upload to disk, refusing it as soon as it passes the size limit."""
    limit, written = limit_mb * 1024 * 1024, 0
    with destination.open("wb") as out:
        while chunk := upload.file.read(_CHUNK):
            written += len(chunk)
            if written > limit:
                raise HTTPException(413, f"File is larger than the {limit_mb:g} MB limit.")
            out.write(chunk)
    if written == 0:
        raise HTTPException(400, "The uploaded file is empty.")


def _sse(event: str, data: str, event_id: int | None = None) -> str:
    """Format one Server-Sent Events message: optional id, event name, data, blank line."""
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {data}\n\n"


async def event_stream(manager: RunManager, run_id: str, after: int) -> AsyncIterator[str]:
    """Yield SSE messages: every stage event with index > after, then "end".

    Order matters on every loop: read the status FIRST, the events SECOND. If
    the status already says the run is over, the events read a moment later
    are complete (the background thread writes each event before it frees the
    run slot), so no final stage can be missed between the two reads.

    Args:
        manager: The app's RunManager.
        run_id: An existing run.
        after: Index of the last event the client already has (-1 for none);
            from the Last-Event-ID header when the browser reconnects.
    """
    sent, last_message = after, time.monotonic()
    while True:
        status = manager.status(run_id)
        for event in manager.events(run_id):
            if event.index > sent:
                yield _sse("stage", event.model_dump_json(), event.index)
                sent, last_message = event.index, time.monotonic()
        if status != "running":
            yield _sse("end", json.dumps({"status": status}))
            return
        if time.monotonic() - last_message >= SSE_KEEPALIVE_S:
            yield ": keep-alive\n\n"  # a comment line: browsers ignore it, connections stay open
            last_message = time.monotonic()
        await asyncio.sleep(SSE_POLL_INTERVAL_S)  # wait without holding a worker thread


def create_app(runs_dir: Path = RUNS_DIR) -> FastAPI:
    """Build the app. A factory, so tests and `autods serve` can choose the runs folder."""
    app = FastAPI(
        title="auto-ds-agent API",
        version="0.1.0",
        description="Upload a tabular dataset and a description; the agent does the rest.",
    )
    manager = RunManager(runs_dir)
    app.state.manager = manager

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/runs", status_code=202, response_model=RunCreated)
    def create_run(
        file: Annotated[UploadFile, File(description="CSV or Parquet dataset.")],
        description: Annotated[str, Form(description="What you want to predict, in words.")],
        target: Annotated[str | None, Form(description="Target column; inferred if empty.")] = None,
    ) -> RunCreated:
        """Save the upload into a new run folder and start the agent in the background."""
        if not description.strip():
            raise HTTPException(400, "Describe what you want to predict.")
        filename = _safe_filename(file.filename or "")
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(SUPPORTED_EXTENSIONS)
            raise HTTPException(400, f"Unsupported file type; expected one of: {allowed}.")
        if manager.busy:
            # Checked before creating anything, so a refused upload leaves no folder behind.
            raise HTTPException(409, "A run is already in progress; try again when it finishes.")

        runs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = new_run_dir(filename, runs_dir)
        dataset_path = run_dir / "input" / filename
        dataset_path.parent.mkdir()
        try:
            _save_upload(file, dataset_path, MAX_DATASET_MB)
            run_id = manager.start(run_dir, dataset_path, description.strip(), target or None)
        except HTTPException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except RunConflictError:  # another upload won the race between busy() and start()
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(409, "A run is already in progress.") from None
        return RunCreated(run_id=run_id, status="running")

    @app.get("/api/runs", response_model=list[RunListItem])
    def list_runs() -> list[RunListItem]:
        """Every run, newest first."""
        return manager.list_runs()

    @app.get("/api/runs/{run_id}", response_model=RunSummary)
    def get_run(run_id: str) -> RunSummary:
        """One run's status, stage events so far, and results once it has finished."""
        summary = manager.summary(run_id)
        if summary is None:
            raise HTTPException(404, f"No run called {run_id!r}.")
        return summary

    @app.get("/api/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        """Live stage events (Server-Sent Events): replay what happened, then push new ones.

        Browsers' EventSource sends Last-Event-ID when it reconnects; streaming
        then resumes after that event instead of starting over.
        """
        if manager.status(run_id) is None:
            raise HTTPException(404, f"No run called {run_id!r}.")
        after = int(last_event_id) if last_event_id and last_event_id.isdigit() else -1
        return StreamingResponse(
            event_stream(manager, run_id, after),
            media_type="text/event-stream",
            # no-cache: never serve a stale copy; X-Accel-Buffering: tell proxies
            # such as nginx to pass each message on at once instead of batching.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}/files/{name}")
    def download_file(run_id: str, name: str) -> FileResponse:
        """Download one of a run's artifacts. Only names on the DOWNLOADABLE allowlist."""
        run_dir = manager.run_dir(run_id)  # validates the run ID (no path traversal)
        if run_dir is None or name not in DOWNLOADABLE or not (run_dir / name).is_file():
            raise HTTPException(404, f"No file {name!r} for run {run_id!r}.")
        return FileResponse(
            run_dir / name,
            media_type=DOWNLOADABLE[name],
            # "attachment": save to disk rather than display. The run ID in the saved
            # name keeps downloads from different runs apart in the Downloads folder.
            filename=f"{run_id}_{name}",
            content_disposition_type="attachment",
        )

    return app
