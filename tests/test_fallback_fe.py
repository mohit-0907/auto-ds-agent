"""Tests for the deterministic fallback feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autods.config import MAX_FE_COLUMNS, PROJECT_ROOT
from autods.nodes.fallback_fe import basic_features, fallback_fe_node
from autods.sandbox.contract import validate_output
from autods.state import TaskDecision
from tests.helpers import make_state

DATASETS = PROJECT_ROOT / "datasets"


# --- It must work on every real dataset -----------------------------------------------


@pytest.mark.parametrize(
    ("file", "target"),
    [
        ("Titanic-Dataset.csv", "Survived"),
        ("ames_train.csv", "SalePrice"),
        ("iris.csv", "species"),
        ("penguins.csv", "body_mass_g"),
        ("diamonds.csv", "price"),
        ("tips.csv", "smoker"),
    ],
)
def test_passes_the_contract_on_every_real_dataset(file: str, target: str) -> None:
    df = pd.read_csv(DATASETS / file)
    features = basic_features(df, target)
    result = validate_output(df, features, target)

    assert result.ok, result.errors
    assert features.shape[1] - 1 <= MAX_FE_COLUMNS


def test_titanic_recipe() -> None:
    df = pd.read_csv(DATASETS / "Titanic-Dataset.csv")
    features = basic_features(df, "Survived")

    # Dropped: integer ID, and text with more than 20 levels.
    for gone in ["PassengerId", "Name", "Ticket", "Cabin"]:
        assert not any(c == gone or c.startswith(gone + "_") for c in features.columns)
    # Age had 177 missing values; now none, filled with the median.
    assert features["Age"].isna().sum() == 0
    assert (features["Age"] == df["Age"].median()).sum() >= 177
    # Embarked had 2 missing values: they become their own level.
    assert {"Embarked_S", "Embarked_C", "Embarked_Q", "Embarked_missing"} <= set(features.columns)
    assert features["Embarked_missing"].sum() == 2
    # The target comes through untouched.
    pd.testing.assert_series_equal(features["Survived"], df["Survived"])


# --- The recipe, one rule at a time ------------------------------------------------------


def test_infinite_values_are_treated_as_missing() -> None:
    df = pd.DataFrame({"x": [1.0, np.inf, 3.0, -np.inf, 5.0], "y": range(5)})
    assert basic_features(df, "y")["x"].tolist() == [1.0, 3.0, 3.0, 3.0, 5.0]


def test_all_missing_numeric_column_is_dropped() -> None:
    df = pd.DataFrame({"empty": [np.nan] * 4, "x": [1.0, 2.0, 3.0, 4.0], "y": range(4)})
    assert "empty" not in basic_features(df, "y").columns


def test_unique_floats_are_kept_unlike_integer_ids() -> None:
    """A unique-per-row float (a price, a measurement) is data, not an ID."""
    df = pd.DataFrame({"row_id": [10, 11, 12, 13], "weight": [1.5, 2.5, 3.5, 4.5], "y": [0, 1] * 2})
    features = basic_features(df, "y")

    assert "row_id" not in features.columns
    assert "weight" in features.columns


def test_unique_integer_measurements_are_not_mistaken_for_ids() -> None:
    """On a small dataset, square footage can be unique per row -- but it has gaps."""
    df = pd.DataFrame(
        {
            "house_id": [7, 5, 6, 4],  # consecutive 4..7, shuffled: still an ID
            "sqft": [1200, 950, 2100, 1640],  # unique, but not a gap-free run
            "y": [0, 1, 0, 1],
        }
    )
    features = basic_features(df, "y")

    assert "house_id" not in features.columns
    assert "sqft" in features.columns


def test_bool_columns_are_kept_as_is() -> None:
    df = pd.DataFrame({"flag": [True, False, True], "y": [1, 2, 3]})
    assert basic_features(df, "y")["flag"].dtype == bool


def test_colliding_one_hot_names_are_made_unique() -> None:
    df = pd.DataFrame({"a_b": ["c", "d"] * 3, "a": ["b_c", "e"] * 3, "y": range(6)})
    features = basic_features(df, "y")

    assert features.columns.is_unique
    assert validate_output(df, features, "y").ok


def test_nothing_usable_leaves_only_the_target() -> None:
    df = pd.DataFrame({"id": range(60), "note": [f"text {i}" for i in range(60)], "y": [0, 1] * 30})
    features = basic_features(df, "y")

    assert list(features.columns) == ["y"]
    assert "No feature columns" in validate_output(df, features, "y").errors[0]


def test_deterministic_and_input_untouched() -> None:
    df = pd.read_csv(DATASETS / "penguins.csv")
    before = df.copy()

    pd.testing.assert_frame_equal(
        basic_features(df, "body_mass_g"), basic_features(df, "body_mass_g")
    )
    pd.testing.assert_frame_equal(df, before)


# --- The node ------------------------------------------------------------------------------


def _state(tmp_path: Path, file: str, target: str, task_type: str, **overrides) -> dict:
    decision = TaskDecision(
        task_type=task_type, target_column=target, reasoning="r", confidence=0.9
    )
    defaults = {
        "dataset_path": str(file),
        "run_dir": str(tmp_path / "run"),
        "task_decision": decision,
    }
    return make_state(**(defaults | overrides))


def test_node_writes_features_and_says_why(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        DATASETS / "Titanic-Dataset.csv",
        "Survived",
        "binary_classification",
        fe_code_history=["a", "b", "c"],
    )
    update = fallback_fe_node(state)

    path = Path(update["processed_data_path"])
    assert path == tmp_path / "run" / "fe" / "fallback" / "features.parquet"
    assert len(pd.read_parquet(path)) == 891
    assert "after 3 failed generated-code attempts" in update["warnings"][-1]


def test_node_reports_target_conversion_only_if_the_sandbox_never_ran(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {"carat": np.linspace(0.2, 2.0, 60), "price": [f"${i * 100:,}" for i in range(60)]}
    ).to_csv(path, index=False)

    never_ran = fallback_fe_node(_state(tmp_path, path, "price", "regression"))
    ran_before = fallback_fe_node(
        _state(tmp_path, path, "price", "regression", fe_code_history=["x"])
    )

    assert any("Converted text target" in w for w in never_ran["warnings"])
    assert "could not be generated" in never_ran["warnings"][-1]
    assert not any("Converted text target" in w for w in ran_before["warnings"])


def test_node_with_no_usable_features_warns_instead_of_writing(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    pd.DataFrame({"id": range(60), "y": [0, 1] * 30}).to_csv(path, index=False)
    update = fallback_fe_node(_state(tmp_path, path, "y", "binary_classification"))

    assert "processed_data_path" not in update
    assert "Fallback FE failed validation" in update["warnings"][-1]


def test_node_without_a_decision_is_a_no_op(tmp_path: Path) -> None:
    assert fallback_fe_node(make_state(run_dir=str(tmp_path))) == {}


def test_node_missing_run_dir_fails_loudly(tmp_path: Path) -> None:
    state = _state(
        tmp_path, DATASETS / "iris.csv", "species", "multiclass_classification", run_dir=""
    )
    with pytest.raises(ValueError, match="run_dir"):
        fallback_fe_node(state)
