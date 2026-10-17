"""Starting runs in the background and reading any run's status back.

A run is a folder under runs_dir (its name is the run ID). Everything about it
is recoverable from files in that folder, so history survives a server restart:

    request.json   what was asked: dataset, description, target, created_at
    events.jsonl   one StageEvent per finished node, appended as it happens
    error.txt      only if the run crashed outright
    run.json       written by the graph's finalize node: decisions, leaderboard, cost
    report.md      written by the reporter

Only the currently running run lives in memory, and only one at a time
(PROJECT_PLAN Phase 7: single-user tool).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel

from autods.graph import initial_state
from autods.progress import GraphRun, StageEvent
from autods.state import ModelResult

log = structlog.get_logger(__name__)

RunStatus = Literal["running", "finished", "failed", "interrupted"]

# Run IDs are folder names. Starting with a letter or digit and using only these
# characters rules out "..", "/" and "\" -- so a request for run "../../.env"
# can never read outside runs_dir (path traversal).
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RunConflictError(Exception):
    """A run is already in progress; v1 runs one at a time."""


class RunListItem(BaseModel):
    """One row of the run history."""

    run_id: str
    status: RunStatus
    dataset: str
    created_at: str | None
    task_type: str | None
    best_model_id: str | None


class RunSummary(BaseModel):
    """Everything the browser needs about one run. All of it comes from state/artifacts."""

    run_id: str
    status: RunStatus
    dataset: str
    description: str
    target_column: str | None
    created_at: str | None
    events: list[StageEvent]
    task_type: str | None
    best_model_id: str | None
    leaderboard: list[ModelResult]
    report_markdown: str | None
    token_cost_usd: float
    warnings: list[str]
    error: str | None
    files: list[str]


@dataclass
class _ActiveRun:
    run_id: str
    events: list[StageEvent] = field(default_factory=list)
    thread: threading.Thread | None = None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_events(path: Path) -> list[StageEvent]:
    events = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(StageEvent.model_validate_json(line))
            except ValueError:
                continue  # a half-written last line; the next read will have it
    return events


class RunManager:
    """Starts graph runs on a background thread and reports on any run by ID."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._lock = threading.Lock()
        self._active: _ActiveRun | None = None

    # --- starting runs ------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """True while a run is in progress."""
        with self._lock:
            return self._active is not None

    def start(self, run_dir: Path, dataset_path: Path, description: str, target: str | None) -> str:
        """Record the request and run the graph on a background thread.

        Returns immediately; the run continues after the HTTP response is sent.

        Raises:
            RunConflictError: If another run is still in progress.
        """
        run_id = run_dir.name
        with self._lock:
            if self._active is not None:
                raise RunConflictError(self._active.run_id)
            self._active = _ActiveRun(run_id=run_id)

        try:
            request = {
                "dataset": dataset_path.name,
                "description": description,
                "target_column": target,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (run_dir / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
            state = initial_state(dataset_path, description, target, run_dir)
            thread = threading.Thread(
                target=self._execute, args=(run_dir, state), name=f"run-{run_id}", daemon=True
            )
            self._active.thread = thread
            thread.start()
        except BaseException:
            # Free the slot, or one failed start would make the server "busy" forever.
            with self._lock:
                self._active = None
            raise
        log.info("api.run_started", run_id=run_id)
        return run_id

    def _execute(self, run_dir: Path, state: dict) -> None:
        """The background thread: stream the graph, recording each event as it happens."""
        try:
            with (run_dir / "events.jsonl").open("a", encoding="utf-8") as events_file:
                for event in GraphRun(state).events():
                    events_file.write(event.model_dump_json() + "\n")
                    events_file.flush()
                    with self._lock:
                        self._active.events.append(event)
        except Exception as exc:  # the graph degrades on its own; this is a last resort
            log.exception("api.run_crashed", run_id=run_dir.name)
            (run_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        finally:
            with self._lock:
                self._active = None
            log.info("api.run_ended", run_id=run_dir.name)

    def wait(self, run_id: str, timeout: float | None = None) -> None:
        """Block until the given run is no longer running (used by tests and scripts)."""
        with self._lock:
            active = self._active
        if active is not None and active.run_id == run_id and active.thread is not None:
            active.thread.join(timeout)

    # --- reading runs ---------------------------------------------------------------

    def run_dir(self, run_id: str) -> Path | None:
        """The folder for a run ID, or None if the ID is invalid or unknown."""
        if not _RUN_ID.match(run_id):
            return None
        path = self.runs_dir / run_id
        return path if path.is_dir() else None

    def _status(self, run_dir: Path) -> RunStatus:
        with self._lock:
            if self._active is not None and self._active.run_id == run_dir.name:
                return "running"
        if (run_dir / "error.txt").exists():
            return "failed"
        if (run_dir / "run.json").exists():
            return "finished"
        return "interrupted"  # started but never finished, e.g. the server was stopped

    def status(self, run_id: str) -> RunStatus | None:
        """Just the status of one run (cheap; polled by the live-events stream)."""
        run_dir = self.run_dir(run_id)
        return None if run_dir is None else self._status(run_dir)

    def events(self, run_id: str) -> list[StageEvent]:
        """The stage events so far: from memory while running, else from events.jsonl.

        _execute writes and flushes each event to the file *before* adding it to
        memory, so the file always holds at least what memory holds.
        """
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                return list(self._active.events)
        run_dir = self.run_dir(run_id)
        return [] if run_dir is None else _read_events(run_dir / "events.jsonl")

    def summary(self, run_id: str) -> RunSummary | None:
        """Everything known about one run, or None if there is no such run."""
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return None
        request = _read_json(run_dir / "request.json") or {}
        record = _read_json(run_dir / "run.json") or {}
        inputs = record.get("inputs", {})
        status = self._status(run_dir)
        events = self.events(run_id)  # after the status, for the reason in event_stream
        report = run_dir / "report.md"
        error = run_dir / "error.txt"
        decision = record.get("task_decision") or {}
        return RunSummary(
            run_id=run_id,
            status=status,
            dataset=request.get("dataset") or Path(inputs.get("dataset_path", "")).name,
            description=request.get("description") or inputs.get("user_description", ""),
            target_column=request.get("target_column", inputs.get("target_column")),
            created_at=request.get("created_at"),
            events=events,
            task_type=decision.get("task_type"),
            best_model_id=record.get("best_model_id"),
            leaderboard=[ModelResult.model_validate(r) for r in record.get("leaderboard", [])],
            report_markdown=report.read_text(encoding="utf-8") if report.exists() else None,
            token_cost_usd=record.get("token_cost_usd", 0.0),
            warnings=record.get("warnings", []),
            error=error.read_text(encoding="utf-8") if error.exists() else None,
            files=sorted(p.name for p in run_dir.iterdir() if p.is_file()),
        )

    def list_runs(self) -> list[RunListItem]:
        """Every run folder (web or CLI), newest first."""
        items = []
        if not self.runs_dir.is_dir():
            return items
        for path in sorted(self.runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if not path.is_dir() or not (
                (path / "request.json").exists() or (path / "run.json").exists()
            ):
                continue  # eval outputs and pre-Phase-6 folders have neither
            summary = self.summary(path.name)
            if summary is not None:
                items.append(
                    RunListItem(
                        run_id=summary.run_id,
                        status=summary.status,
                        dataset=summary.dataset,
                        created_at=summary.created_at,
                        task_type=summary.task_type,
                        best_model_id=summary.best_model_id,
                    )
                )
        return items
