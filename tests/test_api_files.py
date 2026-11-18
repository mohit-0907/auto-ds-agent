"""Tests for downloading run artifacts: GET /api/runs/{id}/files/{name}."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autods.api.app import create_app

RUN_ID = "20261018_120000_titanic"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    run = tmp_path / "runs" / RUN_ID
    (run / "input").mkdir(parents=True)
    # Bytes, not text: on Windows write_text turns "\n" into "\r\n" on disk.
    (run / "report.md").write_bytes(b"# Run report\n")
    (run / "fe_code.py").write_text("def engineer_features(df, t):\n    return df\n")
    (run / "run.json").write_text(json.dumps({"best_model_id": "hgb", "leaderboard": []}))
    (run / "request.json").write_text(json.dumps({"dataset": "t.csv", "description": "x"}))
    (run / "input" / "titanic.csv").write_text("a,b\n1,2\n")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-secret\n")
    return TestClient(create_app(tmp_path / "runs"))


def _get(client: TestClient, name: str):
    return client.get(f"/api/runs/{RUN_ID}/files/{name}")


def test_download_an_artifact(client: TestClient) -> None:
    response = _get(client, "report.md")

    assert response.status_code == 200
    assert response.text == "# Run report\n"
    assert response.headers["content-type"].startswith("text/markdown")
    # "attachment" makes the browser save it; the run ID keeps downloads apart.
    assert response.headers["content-disposition"] == (f'attachment; filename="{RUN_ID}_report.md"')


def test_summary_lists_exactly_the_downloadable_files(client: TestClient) -> None:
    summary = client.get(f"/api/runs/{RUN_ID}").json()

    names = [d["name"] for d in summary["downloads"]]
    assert names == ["report.md", "fe_code.py", "run.json"]  # allowlist order
    assert "request.json" in summary["files"]  # present in the folder...
    assert "request.json" not in names  # ...but not offered
    # The size on disk, for the artifacts card: "# Run report\n" is 13 bytes.
    assert summary["downloads"][0] == {"name": "report.md", "size_bytes": 13}


def test_summary_carries_the_feature_code(client: TestClient, tmp_path: Path) -> None:
    summary = client.get(f"/api/runs/{RUN_ID}").json()
    assert summary["fe_code"] == "def engineer_features(df, t):\n    return df\n"

    # No model, no artifacts: a run that stopped early has no feature code to show.
    (tmp_path / "runs" / RUN_ID / "fe_code.py").unlink()
    assert client.get(f"/api/runs/{RUN_ID}").json()["fe_code"] is None


def test_a_run_saved_before_selection_existed_has_none(client: TestClient) -> None:
    # This run.json has no "selection" key, like every run from before Task 7.8.
    assert client.get(f"/api/runs/{RUN_ID}").json()["selection"] is None


@pytest.mark.parametrize(
    "name",
    [
        "request.json",  # in the folder, but not on the allowlist
        "events.jsonl",
        "leaderboard.csv",  # on the allowlist, but this run has none
        "titanic.csv",
        # (Plain ".." is not tested here: HTTP clients collapse /files/.. to the parent path
        # before sending, so the server never sees it. The encoded forms below do arrive.)
    ],
)
def test_anything_off_the_allowlist_or_missing_is_404(client: TestClient, name: str) -> None:
    assert _get(client, name).status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        f"/api/runs/{RUN_ID}/files/..%2F..%2F.env",  # encoded slashes
        f"/api/runs/{RUN_ID}/files/input%2Ftitanic.csv",
        "/api/runs/..secrets/files/report.md",  # unsafe run ID
        "/api/runs/no-such-run/files/report.md",
    ],
)
def test_traversal_attempts_never_reach_other_files(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 404
    assert "sk-secret" not in response.text
