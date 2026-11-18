"""Tests for the live-events endpoint (Server-Sent Events)."""

import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autods.api.app import create_app, event_stream
from autods.config import PROJECT_ROOT
from autods.progress import StageEvent

TITANIC = PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv"
NODES = ["profiler", "router", "sanity_check"]


def _event(index: int, node: str) -> StageEvent:
    return StageEvent(index=index, node=node, ok=True, text=f"{node} done", attempt=0, at="t")


class GatedRun:
    """A fake GraphRun that emits one event each time the test opens a gate."""

    gates: list[threading.Event] = []

    def __init__(self, state: dict) -> None:
        pass

    def events(self) -> Iterator[StageEvent]:
        for index, node in enumerate(NODES):
            GatedRun.gates[index].wait(10)
            yield _event(index, node)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    GatedRun.gates = [threading.Event() for _ in NODES]
    monkeypatch.setattr("autods.api.runs.GraphRun", GatedRun)
    monkeypatch.setattr("autods.api.app.SSE_POLL_INTERVAL_S", 0.01)
    return TestClient(create_app(tmp_path / "runs"))


def _start(client: TestClient) -> str:
    with TITANIC.open("rb") as f:
        response = client.post(
            "/api/runs", files={"file": ("t.csv", f, "text/csv")}, data={"description": "x"}
        )
    return response.json()["run_id"]


def _messages(lines: Iterator[str]) -> Iterator[dict]:
    """Group raw SSE lines into messages: {"id", "event", "data"} or {"comment"}."""
    message: dict = {}
    for line in lines:
        if line == "":
            if message:
                yield message
            message = {}
        elif line.startswith(":"):
            yield {"comment": line[1:].strip()}
        else:
            field, _, value = line.partition(": ")
            message[field] = value


def _next(stream: AsyncIterator[str]) -> Awaitable[str]:
    return asyncio.wait_for(anext(stream), timeout=5)


def test_events_arrive_while_the_run_is_still_going(client: TestClient) -> None:
    """Driven on event_stream itself: TestClient buffers a whole streaming reply,
    so it cannot show liveness (the real-network check is a curl against the server)."""
    run_id = _start(client)
    manager = client.app.state.manager

    async def scenario() -> None:
        stream = event_stream(manager, run_id, after=-1)
        GatedRun.gates[0].set()  # let only the first stage finish
        first = await _next(stream)
        assert first.startswith("id: 0\nevent: stage\n") and '"profiler"' in first
        assert manager.status(run_id) == "running"  # truly live: the run is not over

        GatedRun.gates[1].set()
        assert '"router"' in await _next(stream)
        GatedRun.gates[2].set()
        assert '"sanity_check"' in await _next(stream)
        assert (await _next(stream)).startswith("event: end\n")

    asyncio.run(scenario())


def test_a_quiet_stream_gets_keep_alive_comments(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("autods.api.app.SSE_KEEPALIVE_S", 0.05)
    run_id = _start(client)  # no gate opened: the run stays quiet

    async def scenario() -> None:
        stream = event_stream(client.app.state.manager, run_id, after=-1)
        assert await _next(stream) == ": keep-alive\n\n"

    asyncio.run(scenario())
    for gate in GatedRun.gates:  # let the background run end
        gate.set()


def test_a_finished_run_replays_everything_then_ends(client: TestClient) -> None:
    run_id = _start(client)
    for gate in GatedRun.gates:
        gate.set()
    client.app.state.manager.wait(run_id, timeout=10)

    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        messages = list(_messages(response.iter_lines()))

    assert [m.get("id") for m in messages] == ["0", "1", "2", None]
    assert messages[-1]["event"] == "end"


def test_reconnecting_resumes_after_the_last_event_id(client: TestClient) -> None:
    run_id = _start(client)
    for gate in GatedRun.gates:
        gate.set()
    client.app.state.manager.wait(run_id, timeout=10)

    headers = {"Last-Event-ID": "0"}  # "I already have event 0"
    with client.stream("GET", f"/api/runs/{run_id}/events", headers=headers) as response:
        ids = [m.get("id") for m in _messages(response.iter_lines())]

    assert ids == ["1", "2", None]  # no duplicate of event 0


def test_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/no-such-run/events").status_code == 404


def test_finished_stream_has_sse_headers(client: TestClient) -> None:
    run_id = _start(client)
    for gate in GatedRun.gates:
        gate.set()
    client.app.state.manager.wait(run_id, timeout=10)
    response = client.get(f"/api/runs/{run_id}/events")

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
