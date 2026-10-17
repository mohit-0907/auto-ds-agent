"""Tests for the `autods` command line, end to end with a scripted LLM."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autods.arena.candidates import candidates
from autods.cli import EXIT_NO_MODEL, app
from autods.config import PROJECT_ROOT
from autods.state import FECode, ReportNarrative, TaskDecision
from tests.helpers import install_scripted_llm, ok_result

TITANIC = PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv"
runner = CliRunner()

DECISION = TaskDecision(
    task_type="binary_classification", target_column="Survived", reasoning="r", confidence=0.95
)
BROKEN = FECode(
    explanation="Use Deck.", code="def engineer_features(df, t):\n    return df['Deck']\n"
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
    task="Binary classification on Survived.",
    features="The second attempt passed after the first referenced a missing column.",
    recommendation="Use Logistic Regression.",
    caveats="- Small dataset.",
)


@pytest.fixture(autouse=True)
def quick_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "autods.nodes.arena.candidates",
        lambda task_type: [c for c in candidates(task_type) if c.model_id in {"dummy", "logreg"}],
    )


def _invoke(tmp_path: Path, *extra: str):
    args = [
        "run",
        str(TITANIC),
        "--description",
        "Predict who survived",
        "--runs-dir",
        str(tmp_path),
    ]
    return runner.invoke(app, [*args, *extra])


def _run_dir(tmp_path: Path) -> Path:
    [run_dir] = list(tmp_path.iterdir())
    return run_dir


def test_one_command_from_csv_to_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_scripted_llm(
        monkeypatch,
        {
            TaskDecision: [ok_result(DECISION)],
            FECode: [ok_result(BROKEN), ok_result(GOOD)],
            ReportNarrative: [ok_result(NARRATIVE)],
        },
    )
    result = _invoke(tmp_path, "--target", "Survived")

    assert result.exit_code == 0, result.output
    # The corrective loop is visible, stage by stage.
    for line in [
        "profiler      891 rows × 12 columns",
        "router        binary_classification → Survived",
        "sandbox       attempt 1 failed",
        "sandbox       attempt 2 passed validation",
        "arena         best: Logistic Regression",
        "reporter      report.md written",
        "finalize      run.json written",
    ]:
        assert line in result.output

    run_dir = _run_dir(tmp_path)
    for name in ["report.md", "run.json", "leaderboard.csv", "best_pipeline.joblib", "fe_code.py"]:
        assert (run_dir / name).exists(), name
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["best_model_id"] == "logreg"
    assert len(record["feature_engineering"]["attempts"]) == 2


def test_unsupported_task_exits_3_with_a_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unsupported = TaskDecision(
        task_type="unsupported",
        target_column="",
        reasoning="r",
        confidence=0.9,
        unsupported_reason="This is forecasting.",
    )
    install_scripted_llm(
        monkeypatch,
        {
            TaskDecision: [ok_result(unsupported)],
            FECode: [],
            ReportNarrative: [ok_result(NARRATIVE)],
        },
    )
    result = _invoke(tmp_path)

    assert result.exit_code == EXIT_NO_MODEL
    assert "Ended without a model" in result.output
    run_dir = _run_dir(tmp_path)
    assert (run_dir / "report.md").exists()
    assert (run_dir / "run.json").exists()
    assert not (run_dir / "best_pipeline.joblib").exists()


def test_llm_outage_still_ends_with_a_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    down = ConnectionError("API down")
    install_scripted_llm(monkeypatch, {TaskDecision: [down], FECode: [], ReportNarrative: [down]})
    result = _invoke(tmp_path)

    assert result.exit_code == EXIT_NO_MODEL
    assert "✗ router" in result.output
    assert "Narrative unavailable" in (_run_dir(tmp_path) / "report.md").read_text(encoding="utf-8")


def test_missing_file_is_a_usage_error_before_any_work(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", str(tmp_path / "nope.csv"), "--description", "x", "--runs-dir", str(tmp_path)]
    )

    assert result.exit_code == 2  # click's usage-error convention
    assert list(tmp_path.iterdir()) == []  # no run folder created for a typo


def test_help_lists_the_options() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    for option in ["--description", "--target", "--runs-dir", "--verbose"]:
        assert option in result.output
