"""Tests for the deterministic sanity check. Pure functions; no LLM involved.

Target profiles are built with the real profile_column, so these exercise the
statistics the check will actually see, not hand-written approximations.
"""

from typing import Any

import pandas as pd
import pytest

from autods.config import PROJECT_ROOT, SANITY_FLAGGED_CONFIDENCE
from autods.nodes.profiler import fit_to_budget, profile_column, profile_dataset
from autods.nodes.sanity_check import check_task_type, sanity_check_node
from autods.state import ColumnProfile, TaskDecision
from tests.helpers import make_state


def _target(values: list[Any], name: str = "y") -> ColumnProfile:
    return profile_column(pd.Series(values, name=name), n_rows=len(values))


def _decision(task_type: str, target: str = "y", confidence: float = 0.9) -> TaskDecision:
    return TaskDecision(
        task_type=task_type, target_column=target, reasoning="r", confidence=confidence
    )


def _check(task_type: str, values: list[Any], **kw: Any) -> tuple[TaskDecision, list[str]]:
    return check_task_type(_decision(task_type, **kw), _target(values))


# --- Phase 2 DoD: the check catches a deliberate failure -------------------------


def test_dod_regression_on_four_integer_levels_is_caught() -> None:
    """The plan's example: 'regression' on a target with 4 unique integer values."""
    decision, notes = _check("regression", [1, 2, 3, 4] * 25)

    assert decision.task_type == "regression"  # debatable, so flagged, not overridden
    assert decision.confidence == SANITY_FLAGGED_CONFIDENCE
    assert "whole-number levels" in notes[0]


def test_dod_wrong_llm_decision_on_titanic_is_overridden() -> None:
    df = pd.read_csv(PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv")
    profile = fit_to_budget(profile_dataset(df, None))
    state = make_state(
        data_profile=profile, task_decision=_decision("regression", target="Survived")
    )
    update = sanity_check_node(state)

    assert update["task_decision"].task_type == "binary_classification"
    assert "regression -> binary_classification" in update["warnings"][0]


@pytest.mark.parametrize(
    ("file", "task_type", "target"),
    [
        ("Titanic-Dataset.csv", "binary_classification", "Survived"),
        ("ames_train.csv", "regression", "SalePrice"),
    ],
)
def test_no_false_alarm_on_real_correct_decisions(file: str, task_type: str, target: str) -> None:
    """A check that fires on correct decisions trains people to ignore it."""
    df = pd.read_csv(PROJECT_ROOT / "datasets" / file)
    profile = profile_dataset(df, target)
    state = make_state(data_profile=profile, task_decision=_decision(task_type, target=target))

    assert sanity_check_node(state) == {}


# --- Certain overrides: class counts are facts ------------------------------------


@pytest.mark.parametrize("wrong", ["regression", "multiclass_classification"])
def test_two_values_is_always_binary(wrong: str) -> None:
    decision, notes = _check(wrong, [0, 1] * 30)

    assert decision.task_type == "binary_classification"
    assert "exactly 2 distinct values" in notes[0]


def test_bool_target_is_binary() -> None:
    decision, _ = _check("regression", [True, False] * 30)
    assert decision.task_type == "binary_classification"


def test_binary_with_three_values_becomes_multiclass() -> None:
    decision, notes = _check("binary_classification", ["a", "b", "c"] * 20)

    assert decision.task_type == "multiclass_classification"
    assert "3 distinct values, not 2" in notes[0]


def test_overrides_compose_binary_to_regression() -> None:
    """Binary on 200 floats: 3+ values -> multiclass, then too many levels -> regression."""
    decision, notes = _check("binary_classification", [i * 1.37 for i in range(200)])

    assert decision.task_type == "regression"
    assert len(notes) == 2


def test_numeric_multiclass_with_many_levels_becomes_regression() -> None:
    decision, notes = _check("multiclass_classification", list(range(600)))

    assert decision.task_type == "regression"
    assert "a quantity, not classes" in notes[0]


# --- Regression on text -----------------------------------------------------------


def test_numbers_stored_as_text_stay_regression_with_a_note() -> None:
    """PROJECT_PLAN's adversarial case: a regression target stored as strings."""
    decision, notes = _check("regression", [f"${i * 1000:,}" for i in range(1, 101)])

    assert decision.task_type == "regression"
    assert decision.confidence == 0.9  # not flagged: the LLM was right
    assert "must convert it" in notes[0]


def test_text_levels_cannot_be_regressed() -> None:
    decision, notes = _check("regression", ["low", "mid", "high"] * 20)

    assert decision.task_type == "multiclass_classification"
    assert "cannot be regressed" in notes[0]


def test_free_text_target_becomes_unsupported() -> None:
    decision, _ = _check("regression", [f"review number {i}" for i in range(200)])

    assert decision.task_type == "unsupported"
    assert "non-numeric text" in decision.unsupported_reason


# --- Flags: debatable, so confidence is capped instead --------------------------


def test_whole_number_floats_with_nan_are_still_flagged() -> None:
    """One NaN makes a 1-5 rating float64; it is still a 1-5 rating."""
    decision, notes = _check("regression", [1.0, 2.0, 3.0, 4.0, 5.0, None] * 10)

    assert decision.confidence == SANITY_FLAGGED_CONFIDENCE
    assert notes


def test_flag_never_raises_confidence() -> None:
    decision, _ = _check("regression", [1, 2, 3, 4] * 25, confidence=0.2)
    assert decision.confidence == 0.2


def test_many_text_classes_is_flagged_not_overridden() -> None:
    decision, notes = _check("multiclass_classification", [f"zip{i}" for i in range(80)] * 2)

    assert decision.task_type == "multiclass_classification"
    assert "not an ID" in notes[0]


# --- No false alarms --------------------------------------------------------------


@pytest.mark.parametrize(
    ("task_type", "values"),
    [
        ("binary_classification", [0, 1] * 30),
        ("multiclass_classification", ["a", "b", "c"] * 20),
        ("regression", [i * 1.37 for i in range(200)]),
        ("regression", list(range(200))),  # integers, but many levels: a quantity
    ],
)
def test_correct_decisions_pass_untouched(task_type: str, values: list[Any]) -> None:
    original = _decision(task_type)
    decision, notes = check_task_type(original, _target(values))

    assert notes == []
    assert decision == original


def test_input_decision_is_not_modified() -> None:
    original = _decision("regression")
    check_task_type(original, _target([0, 1] * 30))
    assert original.task_type == "regression"


# --- Node pass-throughs -------------------------------------------------------------


def test_node_without_a_decision_is_a_no_op() -> None:
    assert sanity_check_node(make_state()) == {}


def test_node_leaves_unsupported_alone() -> None:
    decision = TaskDecision(
        task_type="unsupported",
        target_column="",
        reasoning="r",
        confidence=0.9,
        unsupported_reason="clustering",
    )
    profile = profile_dataset(pd.DataFrame({"a": range(60)}), None)
    assert sanity_check_node(make_state(data_profile=profile, task_decision=decision)) == {}


def test_node_with_correct_decision_changes_nothing() -> None:
    profile = profile_dataset(pd.DataFrame({"y": [0, 1] * 30}), None)
    state = make_state(data_profile=profile, task_decision=_decision("binary_classification"))
    assert sanity_check_node(state) == {}


def test_node_degrades_when_target_missing_from_profile() -> None:
    profile = profile_dataset(pd.DataFrame({"a": range(60)}), None)
    state = make_state(data_profile=profile, task_decision=_decision("regression", target="gone"))

    assert "skipped" in sanity_check_node(state)["warnings"][0]
