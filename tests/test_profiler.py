"""Tests for the deterministic column profiler.

The interesting cases are all degenerate ones -- empty, all-null, constant.
Those are where NaN leaks into the model and where dtype checks written for
pandas 2.x quietly stop working.
"""

import math

import pandas as pd
import pytest

from autods.config import (
    CHARS_PER_TOKEN,
    MAX_DETAILED_COLUMNS,
    MAX_SAMPLE_VALUE_CHARS,
    MAX_TARGET_CANDIDATES,
    PROFILE_TOKEN_BUDGET,
)
from autods.nodes.profiler import (
    estimate_tokens,
    fit_to_budget,
    infer_target_candidates,
    profile_column,
    profile_dataset,
)
from autods.state import ColumnProfile, DataProfile


def test_numeric_column_with_nulls() -> None:
    s = pd.Series([1.0, 2.0, 3.0, None], name="Age")
    p = profile_column(s, n_rows=4)

    assert p.name == "Age"
    assert p.dtype == "float64"
    assert p.null_fraction == 0.25
    assert p.n_unique == 3
    assert p.cardinality_ratio == 0.75
    assert p.min == 1.0
    assert p.max == 3.0
    assert p.mean == 2.0
    assert p.std is not None
    assert p.skew is not None


def test_string_column_has_no_numeric_stats() -> None:
    s = pd.Series(["S", "C", "Q", "S"], name="Embarked")
    p = profile_column(s, n_rows=4)

    # Under pandas 3.0 this dtype is "str", not "object". A profiler that
    # branches on is_object_dtype misclassifies it.
    assert p.is_categorical is True
    assert p.min is None
    assert p.max is None
    assert p.mean is None
    assert p.std is None
    assert p.skew is None
    assert p.n_unique == 3


def test_sample_values_are_distinct_and_stringified() -> None:
    s = pd.Series([7, 7, 7, 8, 9, 10], name="Pclass")
    p = profile_column(s, n_rows=6)

    assert p.sample_values == ["7", "8", "9"]
    assert all(isinstance(v, str) for v in p.sample_values)


def test_constant_column_has_zero_spread_not_none() -> None:
    """A constant column of 3+ rows has a real std and skew of 0.0.

    Guarding on "is the column constant" rather than "is this value NaN" would
    wrongly null both out. Zero spread is a fact about the column, not a
    missing statistic, and the router should see the difference.
    """
    s = pd.Series([5, 5, 5, 5], name="Constant")
    p = profile_column(s, n_rows=4)

    assert p.n_unique == 1
    assert p.std == 0.0
    assert p.skew == 0.0


def test_single_row_column_nulls_out_undefined_stats() -> None:
    """std needs n>=2 and skew needs n>=3; both are NaN at n=1.

    This is the case that actually exercises the NaN -> None conversion, and it
    proves the conversion is per-statistic: mean, min, and max are all defined
    here and must survive.
    """
    s = pd.Series([5.0], name="Single")
    p = profile_column(s, n_rows=1)

    assert p.mean == 5.0
    assert p.min == 5.0
    assert p.max == 5.0
    assert p.std is None
    assert p.skew is None


def test_two_row_constant_column_nulls_out_skew_only() -> None:
    """skew needs n>=3, so at n=2 std is 0.0 but skew is NaN."""
    s = pd.Series([5.0, 5.0], name="Pair")
    p = profile_column(s, n_rows=2)

    assert p.std == 0.0
    assert p.skew is None


def test_all_null_column() -> None:
    s = pd.Series([None, None, None], name="Empty", dtype="object")
    p = profile_column(s, n_rows=3)

    assert p.null_fraction == 1.0
    assert p.n_unique == 0
    assert p.cardinality_ratio == 0.0
    assert p.sample_values == []
    assert p.mean is None


def test_all_null_float_column_takes_numeric_branch() -> None:
    """A float column of all NaN is numeric, so every stat is NaN -> None."""
    s = pd.Series([None, None], name="Missing", dtype="float64")
    p = profile_column(s, n_rows=2)

    assert p.min is None
    assert p.max is None
    assert p.mean is None
    assert p.std is None
    assert p.skew is None


def test_bool_column_is_categorical_and_keeps_its_rate() -> None:
    """is_numeric_dtype is True for bool; the mean is the positive rate."""
    s = pd.Series([True, False, True, True], name="Survived")
    p = profile_column(s, n_rows=4)

    assert p.is_categorical is True
    assert p.mean == 0.75
    assert p.n_unique == 2


def test_high_cardinality_numeric_is_not_categorical() -> None:
    s = pd.Series(range(100), name="PassengerId")
    p = profile_column(s, n_rows=100)

    assert p.is_categorical is False
    assert p.cardinality_ratio == 1.0


def test_empty_series() -> None:
    """n_rows=0 breaks on null_fraction (NaN), not on the division."""
    s = pd.Series([], name="Nothing", dtype="float64")
    p = profile_column(s, n_rows=0)

    assert p.null_fraction == 0.0
    assert p.cardinality_ratio == 0.0
    assert p.n_unique == 0
    assert p.sample_values == []


def test_unnamed_series_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be named"):
        profile_column(pd.Series([1, 2, 3]), n_rows=3)


def test_no_nan_escapes_into_json() -> None:
    """The real acceptance check: a profile of degenerate columns round-trips.

    If any NaN reached a ColumnProfile it would serialize to null and parse
    back as None, and the reparsed profile would not equal the original.
    """
    df = pd.DataFrame(
        {
            "constant": [5, 5, 5],
            "all_null": pd.Series([None, None, None], dtype="float64"),
            "text": ["a", "b", "c"],
            "flag": [True, False, True],
        }
    )
    columns: list[ColumnProfile] = [profile_column(df[c], len(df)) for c in df.columns]
    profile = DataProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        size_mb=0.001,
        n_duplicate_rows=0,
        columns=columns,
    )

    assert DataProfile.model_validate_json(profile.model_dump_json()) == profile


# --- infer_target_candidates ---------------------------------------------------


def _profiles(df: pd.DataFrame) -> list[ColumnProfile]:
    return [profile_column(df[c], len(df)) for c in df.columns]


@pytest.fixture
def titanic_like() -> list[ColumnProfile]:
    df = pd.DataFrame(
        {
            "PassengerId": range(10),
            "Survived": [0, 1] * 5,
            "Pclass": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            "Name": [f"passenger {i}" for i in range(10)],
            "Fare": [7.25 + i * 3.1 for i in range(10)],
        }
    )
    return _profiles(df)


def test_user_target_short_circuits_heuristics(titanic_like: list[ColumnProfile]) -> None:
    assert infer_target_candidates(titanic_like, "Fare") == ["Fare"]


def test_user_target_typo_suggests_real_name(titanic_like: list[ColumnProfile]) -> None:
    with pytest.raises(ValueError, match="Did you mean 'Survived'"):
        infer_target_candidates(titanic_like, "survived")


def test_name_hint_outranks_last_column() -> None:
    df = pd.DataFrame({"y": [0, 1, 0, 1], "feature": [1.5, 2.5, 3.5, 4.5]})
    assert infer_target_candidates(_profiles(df), None)[:2] == ["y", "feature"]


def test_name_hint_matches_whole_name_only() -> None:
    df = pd.DataFrame({"salary": [1.5, 2.5, 3.5, 4.5], "city": ["a", "b", "a", "b"]})
    candidates = infer_target_candidates(_profiles(df), None)

    # "city" is in the list as the last column, not because it contains "y".
    assert candidates == ["city"]


def test_high_cardinality_last_column_is_kept(titanic_like: list[ColumnProfile]) -> None:
    """Fare is continuous with a ratio of 1.0 -- a valid regression target."""
    assert "Fare" in infer_target_candidates(titanic_like, None)


def test_identifier_is_not_a_candidate(titanic_like: list[ColumnProfile]) -> None:
    candidates = infer_target_candidates(titanic_like, None)

    assert "PassengerId" not in candidates
    assert "Name" not in candidates


def test_constant_last_column_is_excluded() -> None:
    df = pd.DataFrame({"label": [0, 1, 0, 1], "source": ["web"] * 4})
    assert infer_target_candidates(_profiles(df), None) == ["label"]


def test_candidates_are_capped_after_dedupe() -> None:
    n_cols = MAX_TARGET_CANDIDATES + 3
    df = pd.DataFrame({f"cat_{i}": ["a", "b"] * 5 for i in range(n_cols)})
    candidates = infer_target_candidates(_profiles(df), None)

    assert len(candidates) == MAX_TARGET_CANDIDATES
    assert len(set(candidates)) == len(candidates)


def test_ordering_is_deterministic(titanic_like: list[ColumnProfile]) -> None:
    assert infer_target_candidates(titanic_like, None) == infer_target_candidates(
        titanic_like, None
    )


def test_no_usable_columns_returns_empty() -> None:
    df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "x"]})
    assert infer_target_candidates(_profiles(df), None) == []


# --- profile_dataset -----------------------------------------------------------


def test_dataset_counts_duplicates_after_first_occurrence() -> None:
    df = pd.DataFrame({"a": [1, 2, 1, 1], "b": ["x", "y", "x", "x"]})
    profile = profile_dataset(df, None)

    assert profile.n_rows == 4
    assert profile.n_cols == 2
    assert profile.n_duplicate_rows == 2
    assert isinstance(profile.n_duplicate_rows, int)


def test_dataset_preserves_column_order() -> None:
    df = pd.DataFrame({"z": [1, 2], "a": [3, 4], "m": [5, 6]})
    assert [c.name for c in profile_dataset(df, None).columns] == ["z", "a", "m"]


def test_dataset_lists_constant_and_all_null_columns() -> None:
    df = pd.DataFrame(
        {
            "varies": [1, 2, 3],
            "constant": ["k", "k", "k"],
            "all_null": pd.Series([None, None, None], dtype="float64"),
        }
    )
    assert profile_dataset(df, None).constant_columns == ["constant", "all_null"]


def test_dataset_size_counts_string_payloads() -> None:
    """deep=True must be used: long strings dwarf their 8-byte pointers."""
    df = pd.DataFrame({"text": ["x" * 10_000] * 100})
    assert profile_dataset(df, None).size_mb > 0.5


def test_dataset_rejects_duplicate_column_names() -> None:
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(ValueError, match="unique"):
        profile_dataset(df, None)


def test_dataset_rejects_names_that_collide_once_stringified() -> None:
    df = pd.DataFrame([[1, 2], [3, 4]], columns=[1, "1"])
    with pytest.raises(ValueError, match="unique"):
        profile_dataset(df, None)


def test_dataset_handles_integer_column_names() -> None:
    df = pd.DataFrame([[1, 0], [2, 1], [3, 0]])  # headers are 0 and 1
    profile = profile_dataset(df, "1")

    assert [c.name for c in profile.columns] == ["0", "1"]
    assert profile.target_candidates == ["1"]


def test_dataset_handles_zero_rows() -> None:
    df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="str")})
    profile = profile_dataset(df, None)

    assert profile.n_rows == 0
    assert profile.n_duplicate_rows == 0
    assert profile.constant_columns == ["a", "b"]


def test_dataset_propagates_bad_target() -> None:
    df = pd.DataFrame({"Survived": [0, 1], "Age": [20.0, 30.0]})
    with pytest.raises(ValueError, match="Did you mean"):
        profile_dataset(df, "survived")


def test_dataset_profile_round_trips_through_json() -> None:
    df = pd.DataFrame(
        {
            "id": range(5),
            "flag": [True, False, True, True, False],
            "city": ["a", "b", None, "a", "b"],
            "price": [1.5, None, 3.5, 4.5, 5.5],
            "constant": [7] * 5,
        }
    )
    profile = profile_dataset(df, None)
    assert DataProfile.model_validate_json(profile.model_dump_json()) == profile


def test_dataset_does_not_modify_input() -> None:
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    before = df.copy()
    profile_dataset(df, None)
    pd.testing.assert_frame_equal(df, before)


# --- estimate_tokens / fit_to_budget -------------------------------------------


def _wide_frame(n_cols: int, n_rows: int = 50) -> pd.DataFrame:
    """Mixed-dtype synthetic frame: floats, ints, and strings in rotation."""
    data: dict[str, list[object]] = {}
    for i in range(n_cols):
        if i % 3 == 0:
            data[f"f{i}"] = [r * 1.37 + i for r in range(n_rows)]
        elif i % 3 == 1:
            data[f"i{i}"] = [(r * 7 + i) % 13 for r in range(n_rows)]
        else:
            data[f"s{i}"] = [f"level_{(r + i) % 4}" for r in range(n_rows)]
    return pd.DataFrame(data)


def test_estimate_tokens_rounds_up() -> None:
    """Written against the ratio, so recalibrating CHARS_PER_TOKEN needs no edit here."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    n = math.ceil(CHARS_PER_TOKEN * 10)
    assert estimate_tokens("x" * n) == math.ceil(n / CHARS_PER_TOKEN)
    assert estimate_tokens("x" * (n + 1)) >= estimate_tokens("x" * n)


def test_small_profile_is_returned_unchanged() -> None:
    profile = profile_dataset(_wide_frame(5), None)
    fitted = fit_to_budget(profile)

    assert fitted == profile
    assert fitted.columns_truncated is False
    assert fitted.dtype_summary == {}


def test_wide_profile_is_truncated_to_budget() -> None:
    profile = profile_dataset(_wide_frame(200), None)
    fitted = fit_to_budget(profile)

    assert fitted.columns_truncated is True
    assert estimate_tokens(fitted.model_dump_json()) <= PROFILE_TOKEN_BUDGET
    assert len(fitted.columns) <= MAX_DETAILED_COLUMNS


def test_truncated_profile_keeps_whole_dataset_facts() -> None:
    profile = profile_dataset(_wide_frame(200), None)
    fitted = fit_to_budget(profile)

    assert fitted.n_cols == 200
    assert sum(fitted.dtype_summary.values()) == 200
    assert fitted.target_candidates == profile.target_candidates
    assert fitted.constant_columns == profile.constant_columns


def test_deep_target_candidate_survives_truncation() -> None:
    df = _wide_frame(200)
    df.insert(150, "target", [0, 1] * 25)
    fitted = fit_to_budget(profile_dataset(df, None))

    assert "target" in [c.name for c in fitted.columns]


def test_tight_budget_trims_below_column_cap_but_keeps_candidates() -> None:
    profile = profile_dataset(_wide_frame(60), "f30")
    fitted = fit_to_budget(profile, max_tokens=500)

    names = [c.name for c in fitted.columns]
    assert "f30" in names
    assert len(names) < MAX_DETAILED_COLUMNS
    assert estimate_tokens(fitted.model_dump_json()) <= 500


def test_candidates_are_kept_even_over_budget() -> None:
    profile = profile_dataset(_wide_frame(60), "f30")
    fitted = fit_to_budget(profile, max_tokens=1)

    assert [c.name for c in fitted.columns] == ["f30"]


def test_kept_columns_stay_in_dataset_order() -> None:
    df = _wide_frame(100)
    df.insert(90, "label", [0, 1] * 25)
    names = [c.name for c in fit_to_budget(profile_dataset(df, None)).columns]

    # "label" is ranked first by relevance (a name hint), but must appear at its
    # dataset position: after the leading columns, before f99 (the last column).
    assert names[0] != "label"
    assert names.index("label") < names.index("f99")
    assert names == sorted(names, key=list(df).index)


def test_fit_to_budget_does_not_modify_input() -> None:
    profile = profile_dataset(_wide_frame(200), None)
    before = profile.model_copy(deep=True)
    fit_to_budget(profile)

    assert profile == before
    assert profile.columns_truncated is False


def test_long_text_samples_are_capped() -> None:
    s = pd.Series([f"{i} " + "x" * 10_000 for i in range(5)], name="review")
    p = profile_column(s, n_rows=5)

    assert all(len(v) <= MAX_SAMPLE_VALUE_CHARS for v in p.sample_values)
