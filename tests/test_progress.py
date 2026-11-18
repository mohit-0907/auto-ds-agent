"""Tests for the stage events' expandable detail and typed header facts."""

from pathlib import Path

import pytest

from autods.arena.candidates import candidates
from autods.config import EVENT_DETAIL_MAX_CHARS, MAX_FE_ATTEMPTS, PROJECT_ROOT
from autods.graph import initial_state
from autods.progress import GraphRun, StageEvent, stage_detail, stage_facts
from autods.state import ColumnProfile, DataProfile, FECode, ReportNarrative, TaskDecision
from tests.helpers import install_scripted_llm, ok_result

TITANIC = PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv"

DECISION = TaskDecision(
    task_type="binary_classification", target_column="Survived", reasoning="r", confidence=0.95
)
BROKEN = FECode(
    explanation="Use the deck letter.",
    code="def engineer_features(df, t):\n    return df['Deck']\n",
)
GOOD = FECode(
    explanation="The Deck column does not exist.\nImpute and encode instead.",
    code="""
import pandas as pd

def engineer_features(df, target_column):
    out = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    out["Age"] = out["Age"].fillna(out["Age"].median())
    out["Embarked"] = out["Embarked"].fillna("S")
    return pd.get_dummies(out, columns=["Sex", "Embarked"], dtype=bool)
""",
)
NARRATIVE = ReportNarrative(summary="s", task="t", features="f", recommendation="r", caveats="- c")


def _column(name: str, null_fraction: float = 0.0) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="float64",
        is_categorical=False,
        null_fraction=null_fraction,
        n_unique=3,
        cardinality_ratio=0.1,
        sample_values=["1"],
    )


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[StageEvent]]:
    """A real graph run on Titanic: attempt 1 fails, attempt 2 passes. Events by node."""
    monkeypatch.setattr(
        "autods.nodes.arena.candidates",
        lambda task_type: [c for c in candidates(task_type) if c.model_id in {"dummy", "logreg"}],
    )
    install_scripted_llm(
        monkeypatch,
        {
            TaskDecision: [ok_result(DECISION)],
            FECode: [ok_result(BROKEN), ok_result(GOOD)],
            ReportNarrative: [ok_result(NARRATIVE)],
        },
    )
    state = initial_state(
        dataset_path=str(TITANIC),
        user_description="Predict who survived",
        target_column=None,
        run_dir=str(tmp_path),
    )
    by_node: dict[str, list[StageEvent]] = {}
    for event in GraphRun(state).events():
        by_node.setdefault(event.node, []).append(event)
    return by_node


def test_header_facts_ride_on_the_stage_that_learns_them(events) -> None:
    [profiler] = events["profiler"]
    assert (profiler.n_rows, profiler.n_cols) == (891, 12)
    [router] = events["router"]
    assert (router.task_type, router.target_column) == ("binary_classification", "Survived")
    # Every other stage leaves the facts empty.
    assert events["arena"][0].n_rows is None and events["arena"][0].task_type is None


def test_details_show_the_evidence_behind_each_step(events) -> None:
    profiler = events["profiler"][0]
    assert "Age" in profiler.detail and "19.9% missing" in profiler.detail
    assert "… 4 more columns" in profiler.detail  # 12 columns, 8 listed

    assert '"target_column": "Survived"' in events["router"][0].detail

    attempt1, attempt2 = events["fe_codegen"]
    assert attempt1.detail.startswith("# Use the deck letter.")
    assert "df['Deck']" in attempt1.detail  # the whole failed code, next to its error
    assert attempt2.detail.startswith("# The Deck column does not exist.\n# Impute and encode")

    failed, passed = events["sandbox"]
    assert failed.detail_kind == "error"
    assert "KeyError" in failed.detail
    assert failed.detail.endswith("→ error sent back to the LLM for attempt 2")
    assert passed.detail_kind == "code"
    assert "✓ target column unchanged" in passed.detail

    assert "2 of 2 models scored on the same 5 folds" in events["arena"][0].detail
    assert "best_pipeline.joblib" in events["artifacts"][0].detail
    assert events["finalize"][0].detail is None  # nothing more than the text


def test_last_failed_attempt_says_the_fallback_is_next() -> None:
    update = {"tracebacks": ["[runtime_error at full stage] boom"]}
    detail, kind = stage_detail("sandbox", update, MAX_FE_ATTEMPTS, ok=False)
    assert kind == "error"
    assert detail.endswith("→ last attempt: switching to the fallback recipe")


def test_detail_is_capped() -> None:
    update = {"fe_code": "x" * (EVENT_DETAIL_MAX_CHARS + 500), "fe_explanations": [""]}
    detail, _ = stage_detail("fe_codegen", update, 1, ok=True)
    assert len(detail) < EVENT_DETAIL_MAX_CHARS + 50
    assert detail.endswith("… (500 more characters)")


def test_several_warnings_become_the_detail_of_a_failed_stage() -> None:
    update = {"warnings": ["first problem", "second problem"]}
    assert stage_detail("fallback_fe", update, 3, ok=False) == (
        "• first problem\n• second problem",
        "error",
    )
    assert stage_detail("fallback_fe", {"warnings": ["only one"]}, 3, ok=False) == (None, None)


def test_profile_detail_counts_hidden_columns_from_n_cols() -> None:
    # A truncated profile lists fewer columns than the table has; the count uses n_cols.
    profile = DataProfile(
        n_rows=10,
        n_cols=300,
        size_mb=0.1,
        n_duplicate_rows=0,
        columns=[_column("a"), _column("b", null_fraction=0.5)],
        columns_truncated=True,
    )
    detail, _ = stage_detail("profiler", {"data_profile": profile}, 0, ok=True)
    assert detail.splitlines() == [
        "a  float64   3 unique · 0 missing",
        "b  float64   3 unique · 50.0% missing",
        "… 298 more columns",
    ]
    assert stage_facts("profiler", {"data_profile": profile}) == {"n_rows": 10, "n_cols": 300}


def test_a_corrected_decision_updates_the_facts() -> None:
    corrected = DECISION.model_copy(update={"task_type": "multiclass_classification"})
    facts = stage_facts("sanity_check", {"task_decision": corrected})
    assert facts == {"task_type": "multiclass_classification", "target_column": "Survived"}
    assert stage_facts("sanity_check", {}) == {}
