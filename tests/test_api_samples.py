"""Tests for sample datasets and the richer run list (Task 7.6)."""

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autods.api.app import create_app
from autods.config import PROJECT_ROOT

TITANIC = PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv"


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    folder = tmp_path / "datasets"
    folder.mkdir()
    (folder / "Titanic-Dataset.csv").write_bytes(TITANIC.read_bytes())
    (folder / "mpg.csv").write_bytes(b"mpg,cylinders\n18,8\n15,8\n")
    # penguins.csv and ames_train.csv are configured but absent here: not offered.
    return folder


class IdleRun:
    """A stand-in GraphRun that waits, so starting a run does no real work."""

    release = threading.Event()

    def __init__(self, state: dict) -> None:
        pass

    def events(self):
        IdleRun.release.wait(10)
        return iter(())


@pytest.fixture
def client(tmp_path: Path, datasets: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    IdleRun.release = threading.Event()
    monkeypatch.setattr("autods.api.runs.GraphRun", IdleRun)
    yield TestClient(create_app(tmp_path / "runs", datasets))
    IdleRun.release.set()


def test_samples_list_only_existing_files_with_row_counts(client: TestClient) -> None:
    samples = client.get("/api/samples").json()

    assert [s["name"] for s in samples] == ["Titanic-Dataset.csv", "mpg.csv"]
    titanic = samples[0]
    assert titanic["rows"] == 891
    assert titanic["kind"] == "binary"
    assert titanic["target"] == "Survived"


def test_a_run_can_start_from_a_sample(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/runs", data={"sample": "mpg.csv", "description": "Predict fuel efficiency"}
    )

    assert response.status_code == 202
    run_dir = tmp_path / "runs" / response.json()["run_id"]
    # Copied into the run folder: the run stays self-contained.
    assert (run_dir / "input" / "mpg.csv").read_bytes() == b"mpg,cylinders\n18,8\n15,8\n"


@pytest.mark.parametrize("sample", ["../.env", "penguins.csv", "nope.csv", ""])
def test_unknown_or_unsafe_samples_are_refused(client: TestClient, sample: str) -> None:
    response = client.post("/api/runs", data={"sample": sample, "description": "x"})
    assert response.status_code == 400


def test_exactly_one_of_file_or_sample(client: TestClient) -> None:
    neither = client.post("/api/runs", data={"description": "x"})
    with TITANIC.open("rb") as f:
        both = client.post(
            "/api/runs",
            files={"file": ("t.csv", f, "text/csv")},
            data={"sample": "mpg.csv", "description": "x"},
        )

    assert neither.status_code == 400
    assert both.status_code == 400
    assert "either a file or a sample" in both.json()["detail"]


# --- The richer run list ---------------------------------------------------------


def _finished_run(runs: Path, name: str, best: str | None, final_source: str | None) -> None:
    run = runs / name
    run.mkdir(parents=True)
    (run / "request.json").write_text(
        json.dumps({"dataset": "t.csv", "description": "Predict x", "target_column": None})
    )
    leaderboard = [
        {
            "model_id": "hgb",
            "model_name": "Hist Gradient Boosting",
            "status": "success",
            "metrics": {"roc_auc": 0.879},
            "primary_metric": "roc_auc",
        }
    ]
    (run / "run.json").write_text(
        json.dumps(
            {
                "best_model_id": best,
                "leaderboard": leaderboard if best else [],
                "task_decision": {"task_type": "binary_classification"},
                "feature_engineering": {"final_source": final_source},
            }
        )
    )


@pytest.mark.parametrize(
    ("best", "final_source", "outcome"),
    [
        ("hgb", "generated (attempt 1)", "model"),
        ("hgb", "fallback", "fallback"),
        (None, None, "no_model"),
    ],
)
def test_run_list_reports_outcome_best_model_and_score(
    client: TestClient, tmp_path: Path, best: str | None, final_source: str | None, outcome: str
) -> None:
    _finished_run(tmp_path / "runs", "20261030_120000_t", best, final_source)
    [item] = client.get("/api/runs").json()

    assert item["status"] == "finished"
    assert item["outcome"] == outcome
    assert item["description"] == "Predict x"
    if best:
        assert item["best_model_name"] == "Hist Gradient Boosting"
        assert (item["primary_metric"], item["best_score"]) == ("roc_auc", 0.879)
    else:
        assert item["best_model_name"] is None
    assert item["last_event"] is None  # only running runs carry one


def test_a_running_run_has_no_outcome_yet(client: TestClient) -> None:
    run_id = client.post("/api/runs", data={"sample": "mpg.csv", "description": "x"}).json()[
        "run_id"
    ]
    [item] = client.get("/api/runs").json()

    assert item["run_id"] == run_id
    assert (item["status"], item["outcome"]) == ("running", None)
