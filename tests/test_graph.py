"""Tests for graph wiring: routing functions alone, then whole runs with a scripted LLM.

The end-to-end tests run the real profiler, sanity check, sandbox subprocesses
and fallback on Titanic; only the two LLM calls (router, codegen) are scripted.
"""

from pathlib import Path

import pytest

from autods.arena.candidates import candidates
from autods.config import MAX_FE_ATTEMPTS, PROJECT_ROOT
from autods.graph import (
    RECURSION_LIMIT,
    build_graph,
    initial_state,
    new_run_dir,
    route_after_codegen,
    route_after_execution,
    route_after_fallback,
    route_after_sanity_check,
)
from autods.state import AgentState, FECode, TaskDecision
from tests.helpers import install_scripted_llm, make_state, ok_result

TITANIC = PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv"


@pytest.fixture(autouse=True)
def quick_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about routing, so the arena runs only dummy + logreg."""
    monkeypatch.setattr(
        "autods.nodes.arena.candidates",
        lambda task_type: [c for c in candidates(task_type) if c.model_id in {"dummy", "logreg"}],
    )


DECISION = TaskDecision(
    task_type="binary_classification", target_column="Survived", reasoning="r", confidence=0.95
)
GOOD = FECode(
    explanation="Impute, encode, drop free text.",
    code="""
import pandas as pd

def engineer_features(df, target_column):
    out = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    out["Age"] = out["Age"].fillna(out["Age"].median())
    out["Embarked"] = out["Embarked"].fillna("S")
    return pd.get_dummies(out, columns=["Sex", "Embarked"], dtype=bool)
""",
)
BROKEN = FECode(
    explanation="Use the Deck column.",
    code="def engineer_features(df, target_column):\n    return df['Deck']\n",
)


def _decision(task_type: str) -> TaskDecision:
    reason = "time series" if task_type == "unsupported" else None
    return TaskDecision(
        task_type=task_type,
        target_column="y",
        reasoning="r",
        confidence=0.9,
        unsupported_reason=reason,
    )


def _run(tmp_path: Path, dataset: Path = TITANIC) -> AgentState:
    state = initial_state(dataset, "Predict who survived", None, new_run_dir(dataset, tmp_path))
    return build_graph().invoke(state, {"recursion_limit": RECURSION_LIMIT})


# --- Routing functions: plain Python, tested with a dict ------------------------------


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (None, "report"),
        (_decision("unsupported"), "report"),
        (_decision("regression"), "fe_codegen"),
    ],
)
def test_route_after_sanity_check(decision: TaskDecision | None, expected: str) -> None:
    assert route_after_sanity_check(make_state(task_decision=decision)) == expected


@pytest.mark.parametrize(("code", "expected"), [("def f(): ...", "sandbox"), (None, "fallback_fe")])
def test_route_after_codegen(code: str | None, expected: str) -> None:
    assert route_after_codegen(make_state(fe_code=code)) == expected


@pytest.mark.parametrize(
    ("n_attempts", "processed", "expected"),
    [
        (1, "features.parquet", "arena"),
        (MAX_FE_ATTEMPTS, "features.parquet", "arena"),  # success on the last attempt still wins
        (1, None, "fe_codegen"),
        (MAX_FE_ATTEMPTS - 1, None, "fe_codegen"),
        (MAX_FE_ATTEMPTS, None, "fallback_fe"),
    ],
)
def test_route_after_execution(n_attempts: int, processed: str | None, expected: str) -> None:
    state = make_state(fe_code_history=["code"] * n_attempts, processed_data_path=processed)
    assert route_after_execution(state) == expected


@pytest.mark.parametrize(
    ("processed", "expected"), [("features.parquet", "arena"), (None, "report")]
)
def test_route_after_fallback(processed: str | None, expected: str) -> None:
    assert route_after_fallback(make_state(processed_data_path=processed)) == expected


# --- The compiled graph --------------------------------------------------------------


def test_graph_compiles_with_every_node() -> None:
    mermaid = build_graph().get_graph().draw_mermaid()
    nodes = ["profiler", "router", "sanity_check", "fe_codegen", "sandbox", "fallback_fe", "arena"]
    for node in nodes:
        assert node in mermaid


def test_readme_diagram_matches_the_graph() -> None:
    """If this fails, the graph changed: run `uv run python scripts/render_graph.py`."""
    committed = (PROJECT_ROOT / "docs" / "graph.mmd").read_text(encoding="utf-8")
    assert committed == build_graph().get_graph().draw_mermaid()


def test_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = install_scripted_llm(
        monkeypatch, {TaskDecision: [ok_result(DECISION)], FECode: [ok_result(GOOD)]}
    )
    final = _run(tmp_path)

    assert final["processed_data_path"].endswith(
        str(Path("fe", "attempt_1", "full_features.parquet"))
    )
    assert final["fe_code_history"] == [GOOD.code.strip() + "\n"]
    assert final["tracebacks"] == []
    assert calls == {TaskDecision: 1, FECode: 1}
    assert final["token_cost_usd"] > 0  # router + codegen, summed by the reducer
    # ...and on through the arena: CSV in, ranked models out.
    assert final["best_model_id"] == "logreg"
    assert [r.model_id for r in final["leaderboard"]] == ["logreg", "dummy"]
    assert (Path(final["run_dir"]) / "leaderboard.csv").exists()


def test_broken_first_attempt_is_repaired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_scripted_llm(
        monkeypatch,
        {TaskDecision: [ok_result(DECISION)], FECode: [ok_result(BROKEN), ok_result(GOOD)]},
    )
    final = _run(tmp_path)

    assert "attempt_2" in final["processed_data_path"]
    assert len(final["fe_code_history"]) == 2
    [failure] = final["tracebacks"]
    assert "KeyError: 'Deck'" in failure.splitlines()[0]
    assert final["fe_explanations"] == [BROKEN.explanation, GOOD.explanation]


def test_three_failures_fall_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = install_scripted_llm(
        monkeypatch,
        {TaskDecision: [ok_result(DECISION)], FECode: [ok_result(BROKEN)] * MAX_FE_ATTEMPTS},
    )
    final = _run(tmp_path)

    assert calls[FECode] == MAX_FE_ATTEMPTS  # the bound held: no fourth attempt
    assert final["processed_data_path"].endswith(str(Path("fe", "fallback", "features.parquet")))
    assert any(f"after {MAX_FE_ATTEMPTS} failed" in w for w in final["warnings"])
    assert final["best_model_id"] == "logreg"  # the spare tyre still reaches the arena


def test_codegen_api_down_goes_straight_to_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The case that would loop forever if attempts were counted from tracebacks."""
    calls = install_scripted_llm(
        monkeypatch,
        {TaskDecision: [ok_result(DECISION)], FECode: [ConnectionError("API down")]},
    )
    final = _run(tmp_path)

    assert calls[FECode] == 1
    assert final["fe_code_history"] == []
    assert "fallback" in final["processed_data_path"]
    assert any("could not be generated" in w for w in final["warnings"])


def test_profiler_failure_ends_the_run_without_llm_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = install_scripted_llm(monkeypatch, {TaskDecision: [], FECode: []})
    final = _run(tmp_path, dataset=tmp_path / "missing.csv")

    assert calls == {TaskDecision: 0, FECode: 0}
    assert final["task_decision"] is None
    assert final["processed_data_path"] is None
    assert any("Profiling failed" in w for w in final["warnings"])


def test_unsupported_task_ends_before_feature_engineering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unsupported = TaskDecision(
        task_type="unsupported",
        target_column="",
        reasoning="r",
        confidence=0.9,
        unsupported_reason="Forecasting.",
    )
    calls = install_scripted_llm(monkeypatch, {TaskDecision: [ok_result(unsupported)], FECode: []})
    final = _run(tmp_path)

    assert calls[FECode] == 0
    assert final["task_decision"].task_type == "unsupported"
    assert final["processed_data_path"] is None


# --- Run setup ------------------------------------------------------------------------


def test_new_run_dir_is_unique_within_the_same_second(tmp_path: Path) -> None:
    first, second = new_run_dir(TITANIC, tmp_path), new_run_dir(TITANIC, tmp_path)

    assert first != second
    assert first.is_dir() and second.is_dir()
    assert "titanic-dataset" in first.name


def test_initial_state_has_every_key() -> None:
    state = initial_state("d.csv", "desc", None, "run")
    assert set(state) == set(AgentState.__annotations__)
