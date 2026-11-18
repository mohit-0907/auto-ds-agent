"""Tests for the web API: real HTTP requests, the agent in its background thread, LLM scripted."""

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autods.api.app import create_app
from autods.arena.candidates import candidates
from autods.config import PROJECT_ROOT
from autods.progress import StageEvent
from autods.state import FECode, ReportNarrative, TaskDecision
from tests.helpers import install_scripted_llm, ok_result

TITANIC = PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv"

DECISION = TaskDecision(
    task_type="binary_classification", target_column="Survived", reasoning="r", confidence=0.95
)
GOOD = FECode(
    explanation="Impute and encode.",
    code="""
import pandas as pd

def engineer_features(df, target_column):
    out = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    out["Age"] = out["Age"].fillna(out["Age"].median())
    out["Embarked"] = out["Embarked"].fillna("S")
    return pd.get_dummies(out, columns=["Sex", "Embarked"], dtype=bool)
""",
)
NARRATIVE = ReportNarrative(
    summary="Logistic Regression is recommended.",
    task="Binary classification.",
    features="Imputed and encoded.",
    recommendation="Use Logistic Regression.",
    caveats="- Small dataset.",
)


@pytest.fixture(autouse=True)
def quick_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "autods.nodes.arena.candidates",
        lambda task_type: [c for c in candidates(task_type) if c.model_id in {"dummy", "logreg"}],
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "runs"))


def _upload(client: TestClient, path: Path = TITANIC, **form: str):
    data = {"description": "Predict who survived", **form}
    with path.open("rb") as f:
        return client.post("/api/runs", files={"file": (path.name, f, "text/csv")}, data=data)


def _wait(client: TestClient, run_id: str) -> dict:
    client.app.state.manager.wait(run_id, timeout=120)
    return client.get(f"/api/runs/{run_id}").json()


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_upload_runs_the_agent_and_reports_results(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    install_scripted_llm(
        monkeypatch,
        {
            TaskDecision: [ok_result(DECISION)],
            FECode: [ok_result(GOOD)],
            ReportNarrative: [ok_result(NARRATIVE)],
        },
    )
    created = _upload(client)
    assert created.status_code == 202  # accepted: the run continues after this reply
    run_id = created.json()["run_id"]

    run = _wait(client, run_id)
    assert run["status"] == "finished"
    assert run["dataset"] == "Titanic-Dataset.csv"
    assert run["best_model_id"] == "logreg"
    assert run["task_type"] == "binary_classification"
    assert [r["model_id"] for r in run["leaderboard"]] == ["logreg", "dummy"]
    # The arena's selection record survives the trip through run.json.
    assert run["selection"]["chosen_model_id"] == "logreg"
    assert run["selection"]["comparisons"] == []  # no other real model in this quick arena
    assert run["selection"]["baseline"]["kind"] == "difference"
    assert "## Leaderboard" in run["report_markdown"]
    assert {"report.md", "run.json", "best_pipeline.joblib", "events.jsonl"} <= set(run["files"])

    nodes = [e["node"] for e in run["events"]]
    assert nodes[0] == "profiler" and nodes[-1] == "finalize"
    assert [e["index"] for e in run["events"]] == list(range(len(nodes)))

    # The upload lives inside the run folder: a run is self-contained.
    assert (tmp_path / "runs" / run_id / "input" / "Titanic-Dataset.csv").exists()


def test_history_lists_web_and_cli_runs(client: TestClient, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    cli_run = runs / "20261001_120000_old-cli-run"
    cli_run.mkdir(parents=True)
    (cli_run / "run.json").write_text(
        json.dumps(
            {
                "inputs": {"dataset_path": "datasets/mpg.csv", "user_description": "mpg"},
                "task_decision": {"task_type": "regression"},
                "best_model_id": "hgb",
                "leaderboard": [],
            }
        )
    )
    (runs / "eval_router_x.json").write_text("{}")  # eval outputs are files: ignored
    (runs / "20260901_000000_pre-phase-6").mkdir()  # no run.json/request.json: ignored

    [item] = client.get("/api/runs").json()
    assert item["run_id"] == "20261001_120000_old-cli-run"
    assert item["status"] == "finished"
    assert item["dataset"] == "mpg.csv"
    assert item["best_model_id"] == "hgb"


def test_a_second_upload_while_busy_is_refused(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    release = threading.Event()

    class SlowRun:
        def __init__(self, state: dict) -> None:
            pass

        def events(self):
            release.wait(30)
            return iter(())

    monkeypatch.setattr("autods.api.runs.GraphRun", SlowRun)
    first = _upload(client)
    second = _upload(client)
    release.set()

    assert first.status_code == 202
    assert second.status_code == 409
    assert len(list((tmp_path / "runs").iterdir())) == 1  # the refused upload left nothing


def test_a_crash_is_reported_as_failed(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    class CrashingRun:
        def __init__(self, state: dict) -> None:
            pass

        def events(self):
            yield StageEvent(index=0, node="profiler", ok=True, text="fine", attempt=0, at="t")
            raise RuntimeError("disk full")

    monkeypatch.setattr("autods.api.runs.GraphRun", CrashingRun)
    run_id = _upload(client).json()["run_id"]
    run = _wait(client, run_id)

    assert run["status"] == "failed"
    assert run["error"] == "RuntimeError: disk full"
    assert [e["node"] for e in run["events"]] == ["profiler"]  # progress before the crash is kept
    assert client.app.state.manager.busy is False  # the slot was released


def test_a_run_that_never_finished_is_interrupted(client: TestClient, tmp_path: Path) -> None:
    run = tmp_path / "runs" / "20261001_120000_titanic"
    run.mkdir(parents=True)
    (run / "request.json").write_text(json.dumps({"dataset": "t.csv", "description": "x"}))

    assert client.get(f"/api/runs/{run.name}").json()["status"] == "interrupted"


# --- Input validation at the door --------------------------------------------------------


def test_unsupported_file_type_is_rejected(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"not a spreadsheet")
    response = _upload(client, path)

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    assert not (tmp_path / "runs").exists()


def test_blank_description_is_rejected(client: TestClient) -> None:
    response = _upload(client, description="   ")
    assert response.status_code == 400


def test_oversized_upload_is_rejected_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    monkeypatch.setattr("autods.api.app.MAX_DATASET_MB", 0.01)  # 10 KB; Titanic is ~60 KB
    response = _upload(client)

    assert response.status_code == 413
    assert list((tmp_path / "runs").iterdir()) == []  # the half-saved run folder is removed


def test_filename_is_sanitized(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    install_scripted_llm(
        monkeypatch,
        {TaskDecision: [ok_result(DECISION)], FECode: [ok_result(GOOD)], ReportNarrative: []},
    )
    with TITANIC.open("rb") as f:
        response = client.post(
            "/api/runs",
            files={"file": ("../../evil name.csv", f, "text/csv")},
            data={"description": "x"},
        )
    run = _wait(client, response.json()["run_id"])
    assert run["dataset"] == "evil_name.csv"


@pytest.mark.parametrize("run_id", ["no-such-run", "..secrets", ".env"])
def test_unknown_or_unsafe_run_ids_are_404(client: TestClient, run_id: str) -> None:
    assert client.get(f"/api/runs/{run_id}").status_code == 404
