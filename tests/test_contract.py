"""Tests for the FE contract: prepare_target before generated code, validate_output after."""

import numpy as np
import pandas as pd
import pytest

from autods.sandbox.contract import ValidationResult, prepare_target, validate_output

N = 60


@pytest.fixture
def original() -> pd.DataFrame:
    """What the generated code receives: mixed dtypes, a missing value, a target."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.uniform(1, 80, N).round(1),
            "fare": rng.uniform(5, 500, N).round(2),
            "sex": rng.choice(["male", "female"], N),
            "survived": rng.integers(0, 2, N),
        }
    ).assign(age=lambda d: d["age"].mask(d.index % 7 == 0))


@pytest.fixture
def good(original: pd.DataFrame) -> pd.DataFrame:
    """A correct engineer_features output: imputed, encoded, target untouched."""
    return pd.DataFrame(
        {
            "age": original["age"].fillna(original["age"].median()),
            "fare": original["fare"],
            "sex_male": original["sex"].eq("male"),
            "survived": original["survived"],
        }
    )


def _errors(original: pd.DataFrame, result: object) -> list[str]:
    return validate_output(original, result, "survived").errors


# --- The happy path ---------------------------------------------------------------


def test_valid_output_passes(original: pd.DataFrame, good: pd.DataFrame) -> None:
    result = validate_output(original, good, "survived")

    assert result.ok
    assert result.errors == []
    assert result.warnings == []


def test_ok_is_serialized() -> None:
    """ok is a computed field, so it appears in logs and run.json without being stored."""
    assert ValidationResult(errors=["x"]).model_dump()["ok"] is False


def test_different_index_with_same_rows_passes(original: pd.DataFrame, good: pd.DataFrame) -> None:
    assert validate_output(original, good.set_index(good.index + 1000), "survived").ok


def test_missing_target_values_survive_unchanged() -> None:
    original = pd.DataFrame({"x": range(N), "y": [1.5, None] * (N // 2)})
    result = original.assign(x=original["x"] * 2.0)
    assert validate_output(original, result, "y").ok


# --- Structural failures ----------------------------------------------------------


@pytest.mark.parametrize("returned", [None, [1, 2, 3], pd.Series([1, 2])])
def test_must_return_a_dataframe(original: pd.DataFrame, returned: object) -> None:
    assert "must return a pandas DataFrame" in _errors(original, returned)[0]


def test_dropping_rows_is_rejected_with_a_fix(original: pd.DataFrame, good: pd.DataFrame) -> None:
    errors = _errors(original, good.dropna().iloc[:-5])

    assert "Row count changed from 60 to 55" in errors[0]
    assert "impute" in errors[0]


def test_missing_target(original: pd.DataFrame, good: pd.DataFrame) -> None:
    assert "'survived' is missing" in _errors(original, good.drop(columns="survived"))[0]


def test_modified_target_values(original: pd.DataFrame, good: pd.DataFrame) -> None:
    errors = _errors(original, good.assign(survived=1 - good["survived"]))
    assert "was modified" in errors[0]


def test_target_dtype_change_counts_as_modified(original: pd.DataFrame, good: pd.DataFrame) -> None:
    errors = _errors(original, good.assign(survived=good["survived"].astype("float64")))
    assert "was modified" in errors[0]


def test_duplicate_column_names(original: pd.DataFrame, good: pd.DataFrame) -> None:
    dup = pd.concat([good, good[["fare"]]], axis=1)
    assert "duplicated: 'fare'" in _errors(original, dup)[0]


def test_non_string_column_names(original: pd.DataFrame, good: pd.DataFrame) -> None:
    result = good.rename(columns={"fare": 0})
    assert "non-string names" in _errors(original, result)[0]


def test_target_only_has_no_features(original: pd.DataFrame, good: pd.DataFrame) -> None:
    assert "No feature columns" in _errors(original, good[["survived"]])[0]


# --- Feature checks ---------------------------------------------------------------


def test_text_column_is_rejected_with_its_dtype(original: pd.DataFrame, good: pd.DataFrame) -> None:
    errors = _errors(original, good.assign(sex=original["sex"]))

    assert "must be numeric or bool" in errors[0]
    assert "'sex'" in errors[0]


def test_nan_in_features(original: pd.DataFrame, good: pd.DataFrame) -> None:
    errors = _errors(original, good.assign(age=original["age"]))
    assert "contain NaN: 'age'" in errors[0]


def test_infinite_values(original: pd.DataFrame, good: pd.DataFrame) -> None:
    errors = _errors(original, good.assign(ratio=good["fare"] / 0))
    assert "infinite values: 'ratio'" in errors[0]


def test_all_problems_reported_together(original: pd.DataFrame, good: pd.DataFrame) -> None:
    """The retry prompt should see every problem at once, not one per attempt."""
    errors = _errors(original, good.assign(sex=original["sex"], age=original["age"]))
    assert len(errors) == 2


def test_column_explosion(
    monkeypatch: pytest.MonkeyPatch, original: pd.DataFrame, good: pd.DataFrame
) -> None:
    monkeypatch.setattr("autods.sandbox.contract.MAX_FE_COLUMNS", 2)
    assert "exceeds the limit of 2" in _errors(original, good)[0]


# --- Leakage -----------------------------------------------------------------------


def test_new_leaky_column_is_an_error(original: pd.DataFrame, good: pd.DataFrame) -> None:
    errors = _errors(original, good.assign(survived_x2=good["survived"] * 2))
    assert "'survived_x2'" in errors[0]
    assert "leak" in errors[0]


def test_original_leaky_column_is_only_a_warning() -> None:
    original = pd.DataFrame({"y": np.arange(N, dtype="float64"), "y_copy": np.arange(N) + 0.5})
    result = validate_output(original, original.copy(), "y")

    assert result.ok
    assert "'y_copy'" in result.warnings[0]


def test_leak_of_a_text_label_is_caught() -> None:
    labels = np.array(["cat", "dog"] * (N // 2))
    original = pd.DataFrame({"x": np.arange(N, dtype="float64"), "animal": labels})
    result = original.assign(is_dog=original["animal"].eq("dog"))

    assert "'is_dog'" in validate_output(original, result, "animal").errors[0]


def test_constant_feature_is_not_leakage(original: pd.DataFrame, good: pd.DataFrame) -> None:
    assert validate_output(original, good.assign(one=1.0), "survived").ok


# --- prepare_target -----------------------------------------------------------------


def test_text_price_target_is_converted() -> None:
    df = pd.DataFrame({"carat": [0.2, 0.3, 0.4], "price": ["$326", "$1,234", None]})
    out, notes = prepare_target(df, "price", "regression")

    assert out["price"].tolist()[:2] == [326.0, 1234.0]
    assert pd.isna(out["price"].iloc[2])
    assert out["price"].dtype == "float64"
    assert "Converted" in notes[0]
    assert df["price"].iloc[0] == "$326"  # input not modified


def test_partially_parseable_target_is_left_alone() -> None:
    df = pd.DataFrame({"price": ["$326", "call us", "$900"]})
    out, notes = prepare_target(df, "price", "regression")

    assert out is df
    assert "'call us'" in notes[0]


@pytest.mark.parametrize(
    ("values", "task_type"),
    [(["a", "b", "a"], "binary_classification"), ([1.5, 2.5, 3.5], "regression")],
)
def test_other_targets_pass_through(values: list, task_type: str) -> None:
    df = pd.DataFrame({"t": values})
    out, notes = prepare_target(df, "t", task_type)

    assert out is df
    assert notes == []
