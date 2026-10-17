"""Tests for the reporter: facts sheet, number verifier, and the node (LLM scripted)."""

from pathlib import Path

import pytest

from autods.nodes.reporter import reporter_node
from autods.reporting.facts import build_facts, fmt, leaderboard_table
from autods.reporting.verify import unverified_numbers
from autods.state import DataProfile, ModelResult, ReportNarrative, TaskDecision
from tests.helpers import install_scripted_llm, make_state, ok_result


def _result(model_id: str, name: str, folds: list[float], predict_s: float = 0.02) -> ModelResult:
    mean = sum(folds) / len(folds)
    return ModelResult(
        model_id=model_id,
        model_name=name,
        status="success",
        metrics={"roc_auc": mean, "accuracy": mean - 0.05},
        metrics_std={"roc_auc": 0.02, "accuracy": 0.03},
        primary_metric="roc_auc",
        fold_scores=folds,
        training_time_s=1.0,
        predict_time_s=predict_s,
    )


LEADERBOARD = [
    _result("hgb", "Hist Gradient Boosting", [0.90, 0.89, 0.87, 0.88, 0.87], predict_s=0.021),
    _result("logreg", "Logistic Regression", [0.86, 0.85, 0.84, 0.86, 0.85], predict_s=0.015),
    _result("dummy", "Dummy (class priors)", [0.5] * 5, predict_s=0.004),
]


@pytest.fixture
def finished(tmp_path: Path) -> dict:
    """The final state of a successful run."""
    profile = DataProfile(n_rows=891, n_cols=12, size_mb=0.1, n_duplicate_rows=0, columns=[])
    return make_state(
        dataset_path="datasets/Titanic-Dataset.csv",
        user_description="Predict who survived.",
        run_dir=str(tmp_path),
        data_profile=profile,
        task_decision=TaskDecision(
            task_type="binary_classification",
            target_column="Survived",
            reasoning="Two values.",
            confidence=0.95,
        ),
        fe_code_history=["code"],
        fe_explanations=["Imputed Age, encoded Sex."],
        processed_data_path=str(tmp_path / "fe" / "attempt_1" / "full_features.parquet"),
        leaderboard=LEADERBOARD,
        best_model_id="hgb",
        token_cost_usd=0.02,
    )


def _narrative(
    summary: str = "Hist Gradient Boosting is recommended with ROC-AUC 0.882.",
) -> ReportNarrative:
    return ReportNarrative(
        summary=summary,
        task="Binary classification on Survived.",
        features="Age was imputed and Sex encoded on the first attempt.",
        recommendation="Use Hist Gradient Boosting; it beat Logistic Regression on 5 of 5 folds.",
        caveats="- Small dataset.\n- Re-run with more data.",
    )


# --- The number verifier --------------------------------------------------------------

FACTS = "ROC-AUC 0.8815, RMSE 28,285, 891 rows, 76.4% lower, 1.4× as long"


@pytest.mark.parametrize(
    "prose",
    ["0.882", "0.88", "$28,285", "$28.3k", "76%", "1.4×", "891 passengers", "5 folds, 3 attempts"],
)
def test_honest_numbers_and_rounding_pass(prose: str) -> None:
    assert unverified_numbers(prose, FACTS) == []


@pytest.mark.parametrize(
    ("prose", "bad"),
    [("0.885", "0.885"), ("12 models", "12"), ("$30k", "$30k"), ("a 3.2× speedup", "3.2")],
)
def test_invented_numbers_are_caught(prose: str, bad: str) -> None:
    assert unverified_numbers(prose, FACTS) == [bad]


def test_names_with_digits_are_not_numbers() -> None:
    assert unverified_numbers("macro-F1, R2 and gpt-6 are names", FACTS) == []


# --- The facts sheet --------------------------------------------------------------------


def test_fmt_scales_precision_to_the_metric() -> None:
    assert fmt(0.88153) == "0.882"
    assert fmt(28284.98) == "28,285"


def test_facts_carry_python_computed_comparisons(finished: dict) -> None:
    facts = build_facts(finished)

    assert "891 rows, 12 columns" in facts
    assert (
        "Best model (chosen by the selection rule): Hist Gradient Boosting, ROC-AUC 0.882" in facts
    )
    assert "Hist Gradient Boosting was better on 5 of 5 folds" in facts
    assert "The best model is 0.382 higher" in facts  # vs the 0.500 dummy, computed here
    assert "(1.4× as long)" in facts  # 21 ms vs 15 ms, computed here
    # ...against the fastest real model, never the do-nothing dummy (4 ms here).
    assert "15.0 ms for Logistic Regression" in facts
    assert "for Dummy" not in facts


def test_facts_explain_a_simpler_winner() -> None:
    top = _result("lgbm", "LightGBM", [0.91, 0.84, 0.92, 0.84, 0.90])
    simple = _result("logreg", "Logistic Regression", [0.88, 0.86, 0.88, 0.86, 0.87])
    facts = build_facts(
        make_state(dataset_path="x.csv", leaderboard=[top, simple], best_model_id="logreg")
    )
    assert (
        "LightGBM had the highest mean ROC-AUC (0.882) but beat Logistic Regression on only 3 of 5"
        in facts
    )


def test_leaderboard_table_marks_the_best_and_failures() -> None:
    failed = ModelResult(
        model_id="xgb",
        model_name="XGBoost",
        status="failed",
        primary_metric="roc_auc",
        error="boom",
    )
    table = leaderboard_table([*LEADERBOARD, failed], "hgb")

    assert "| 1 | **Hist Gradient Boosting** ✅ | 0.882 ± 0.020 |" in table
    assert "XGBoost (failed: boom)" in table


# --- The node ------------------------------------------------------------------------------


def test_report_is_written_with_tables_and_prose(
    monkeypatch: pytest.MonkeyPatch, finished: dict
) -> None:
    calls = install_scripted_llm(monkeypatch, {ReportNarrative: [ok_result(_narrative())]})
    update = reporter_node(finished)

    report = (Path(finished["run_dir"]) / "report.md").read_text(encoding="utf-8")
    assert report == update["report"]
    for heading in [
        "## Summary",
        "## Task",
        "## Feature engineering",
        "## Leaderboard",
        "## Recommendation",
        "## Caveats and next steps",
    ]:
        assert heading in report
    assert "ROC-AUC 0.882" in report
    assert "Unverified numbers" not in report
    assert calls[ReportNarrative] == 1
    assert update["warnings"] == []


def test_invented_number_triggers_one_retry_that_fixes_it(
    monkeypatch: pytest.MonkeyPatch, finished: dict
) -> None:
    wrong = _narrative("Hist Gradient Boosting is recommended with ROC-AUC 0.913.")
    calls = install_scripted_llm(
        monkeypatch, {ReportNarrative: [ok_result(wrong), ok_result(_narrative())]}
    )
    update = reporter_node(finished)

    assert calls[ReportNarrative] == 2
    assert "0.913" not in update["report"]
    assert update["warnings"] == []


def test_a_number_still_wrong_after_the_retry_is_flagged_visibly(
    monkeypatch: pytest.MonkeyPatch, finished: dict
) -> None:
    wrong = _narrative("Hist Gradient Boosting is recommended with ROC-AUC 0.913.")
    install_scripted_llm(monkeypatch, {ReportNarrative: [ok_result(wrong), ok_result(wrong)]})
    update = reporter_node(finished)

    assert "⚠️ **Unverified numbers.**" in update["report"]
    assert "0.913" in update["warnings"][0]


def test_llm_down_still_writes_the_tables(monkeypatch: pytest.MonkeyPatch, finished: dict) -> None:
    install_scripted_llm(monkeypatch, {ReportNarrative: [ConnectionError("API down")]})
    update = reporter_node(finished)

    report = update["report"]
    assert "**Narrative unavailable.**" in report
    assert report.count("ConnectionError: API down") == 1  # said once, not in every section
    assert "checked against the run's computed results" not in report  # no false claim
    assert "| 1 | **Hist Gradient Boosting** ✅" in update["report"]
    assert "without narrative" in update["warnings"][0]


def test_a_failed_run_still_gets_a_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = make_state(
        dataset_path="missing.csv",
        run_dir=str(tmp_path),
        warnings=["Profiling failed: dataset not found: missing.csv"],
    )
    narrative = ReportNarrative(
        summary="The run stopped before training: the dataset file was not found.",
        task="No task was decided.",
        features="None.",
        recommendation="Check the file path and run again.",
        caveats="- Nothing was trained.",
    )
    install_scripted_llm(monkeypatch, {ReportNarrative: [ok_result(narrative)]})
    report = reporter_node(state)["report"]

    assert "The dataset could not be profiled." in report
    assert "_No models were trained._" in report
    assert "Profiling failed: dataset not found" in report


def test_missing_run_dir_fails_loudly(finished: dict) -> None:
    finished["run_dir"] = ""
    with pytest.raises(ValueError, match="run_dir"):
        reporter_node(finished)
