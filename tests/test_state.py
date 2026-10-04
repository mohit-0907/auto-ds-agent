"""Contract tests for the models in autods.state.

These pin behaviour the rest of the project relies on. A later refactor that
loosens a type or drops a default should fail here, not three phases downstream
inside a node.

Fixtures and structure are scaffolding; the assertions are the exercise.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from autods.state import ColumnProfile, DataProfile, TaskDecision

# Fixtures are module-scoped and shared across tests, which is only safe because
# ColumnProfile and DataProfile are frozen -- no test can mutate one and leak the
# change into another. Do not widen this scope to the mutable models.


@pytest.fixture(scope="module")
def numeric_column() -> ColumnProfile:
    """A numeric column with every optional statistic populated."""
    return ColumnProfile(
        name="Age",
        dtype="float64",
        is_categorical=False,
        null_fraction=0.1987,
        n_unique=88,
        cardinality_ratio=0.0988,
        sample_values=["22.0", "38.0", "26.0"],
        min=0.42,
        max=80.0,
        mean=29.6991,
        std=14.5265,
        skew=0.3891,
    )


@pytest.fixture(scope="module")
def categorical_column() -> ColumnProfile:
    """A string column, leaving every numeric statistic at its default."""
    return ColumnProfile(
        name="Embarked",
        dtype="str",
        is_categorical=True,
        null_fraction=0.0022,
        n_unique=3,
        cardinality_ratio=0.0034,
        sample_values=["S", "C", "Q"],
    )


@pytest.fixture(scope="module")
def full_profile(numeric_column: ColumnProfile, categorical_column: ColumnProfile) -> DataProfile:
    """A profile exercising both column shapes and both float precisions."""
    return DataProfile(
        n_rows=891,
        n_cols=12,
        size_mb=0.1908,
        n_duplicate_rows=0,
        columns=[numeric_column, categorical_column],
        target_candidates=["Survived", "Pclass"],
        columns_truncated=False,
    )


def test_data_profile_survives_json_round_trip(full_profile: DataProfile) -> None:
    """A profile must serialize and reparse into an equal object.

    This is the contract the router depends on: the profile reaches the LLM as
    JSON text, and lands in runs/<ts>/run.json the same way. If it cannot make
    that trip losslessly, every downstream stage is reading a lie.
    """
    text = full_profile.model_dump_json()
    assert DataProfile.model_validate_json(text) == full_profile

    # None must survive the trip as None, not as the string "null". Equality
    # above would still hold if both sides were wrong in the same way.
    reparsed = DataProfile.model_validate_json(text)
    assert reparsed.columns[1].min is None
    assert reparsed.columns[1].skew is None


def test_confidence_above_one_is_rejected() -> None:
    """confidence is bounded [0, 1]; 1.5 must fail validation.

    Asserting on the error's "loc" is what stops this passing for the wrong
    reason -- without it, a typo in any other field satisfies the raises.
    """
    with pytest.raises(ValidationError) as exc_info:
        TaskDecision(
            task_type="binary_classification",
            target_column="Survived",
            reasoning="The target column holds exactly two distinct values.",
            confidence=1.5,
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("confidence",)
    assert errors[0]["type"] == "less_than_equal"


def test_task_decision_defaults_unsupported_reason_to_none() -> None:
    """A supported task must construct without mentioning unsupported_reason.

    If the field is required, .with_structured_output(TaskDecision) forces the
    model to invent an unsupported-reason for a task that is perfectly
    supported.
    """
    decision = TaskDecision(
        task_type="binary_classification",
        target_column="Survived",
        reasoning="The target column holds exactly two distinct values.",
        confidence=0.95,
    )
    assert decision.unsupported_reason is None

    # The JSON schema is what .with_structured_output() actually sends to the
    # model. Asserting the attribute proves the Python is right; asserting the
    # schema proves the prompt is.
    assert "unsupported_reason" not in TaskDecision.model_json_schema()["required"]


def test_sample_values_rejects_non_string_scalars() -> None:
    """sample_values is list[str], so numpy scalars fail at construction.

    df["Age"].head(3).tolist() yields np.float64 objects. Typed as list[Any]
    they construct fine and then raise PydanticSerializationError inside the
    reporter, far from the profiler that caused it. Typed as list[str] they are
    rejected here, at the point of the mistake.
    """
    with pytest.raises(ValidationError) as exc_info:
        ColumnProfile(
            name="Age",
            dtype="int64",
            is_categorical=False,
            null_fraction=0.0,
            n_unique=3,
            cardinality_ratio=0.1,
            sample_values=[np.int64(22)],
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("sample_values", 0)
    assert errors[0]["type"] == "string_type"
