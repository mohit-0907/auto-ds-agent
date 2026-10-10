"""The contract LLM-generated feature engineering code must satisfy.

The generated code defines exactly one function, ENTRYPOINT, with this shape:

    def engineer_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame

Constraining the shape is what makes repair tractable: a failure is always
"this function, on this input, broke this rule", never "somewhere in a script".

Everything here is deterministic Python. prepare_target runs before the
generated code (so the LLM never touches the label), and validate_output runs
after it (so nothing the LLM returns reaches the arena unchecked).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field

from autods.config import LEAKAGE_CORRELATION_THRESHOLD, MAX_FE_COLUMNS

ENTRYPOINT = "engineer_features"

_NUMBER_DECORATIONS = str.maketrans("", "", "$€£,% ")
_MAX_NAMES_IN_MESSAGE = 10


class ValidationResult(BaseModel):
    """Outcome of checking generated code's output against the contract.

    Error messages are written to be fed back to the codegen LLM on retry, so
    each one names the column and says what to change.
    """

    errors: list[str] = Field(default_factory=list, description="Contract violations.")
    warnings: list[str] = Field(
        default_factory=list, description="Problems that do not block training."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """True when there are no errors; warnings do not block."""
        return not self.errors


def strip_number_decorations(text: str) -> str:
    """Remove currency symbols, thousands separators, % and spaces."""
    return text.translate(_NUMBER_DECORATIONS)


def prepare_target(
    df: pd.DataFrame, target_column: str, task_type: str
) -> tuple[pd.DataFrame, list[str]]:
    """Convert a regression target stored as text ("$1,234") to numbers.

    Runs before the generated code, so the LLM receives a clean label and the
    contract can require it back unchanged. Converts only when every non-null
    value parses; a partial conversion would silently turn bad values into NaN.

    Args:
        df: The loaded dataset. Not modified.
        target_column: The target chosen by the router.
        task_type: The router's task type; only "regression" is converted.

    Returns:
        The frame (a copy if converted) and notes for state["warnings"].
    """
    target = df[target_column]
    if task_type != "regression" or pd.api.types.is_numeric_dtype(target):
        return df, []

    present = target.dropna()
    converted = pd.to_numeric(present.astype(str).map(strip_number_decorations), errors="coerce")
    n_failed = int(converted.isna().sum())
    if n_failed:
        examples = present[converted.isna()].astype(str).unique()[:3].tolist()
        return df, [
            f"Target {target_column!r} is text and {n_failed} values do not parse as "
            f"numbers (e.g. {examples}); left unconverted."
        ]

    out = df.copy()
    # Every non-null value parsed above; missing targets stay NaN (the arena
    # drops those rows deterministically).
    out[target_column] = converted.reindex(df.index).astype("float64")
    return out, [f"Converted text target {target_column!r} to numbers before feature engineering."]


def _names(columns: list[str]) -> str:
    shown = ", ".join(repr(c) for c in columns[:_MAX_NAMES_IN_MESSAGE])
    extra = len(columns) - _MAX_NAMES_IN_MESSAGE
    return shown + (f" and {extra} more" if extra > 0 else "")


def find_leaks(
    original: pd.DataFrame, result: pd.DataFrame, target_column: str, features: list[str]
) -> tuple[list[str], list[str]]:
    """Find feature columns that near-perfectly track the target.

    Shared by validate_output and the fallback FE, so both use one definition
    of "leaks the target".

    Args:
        original: The frame before feature engineering.
        result: The engineered frame, target included.
        target_column: The target column name.
        features: Numeric feature columns of result to check.

    Returns:
        (new_columns, original_columns) correlating above the threshold. A new
        one was created by feature engineering, so it can be fixed there; an
        original one is leakage in the dataset itself, which only a human can
        judge.
    """
    target = result[target_column]
    mask = target.notna()
    if mask.sum() < 3 or not features:
        return [], []
    y = target[mask]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y)[0], index=y.index, dtype="float64")
    # Constant columns correlate as NaN (zero std): "no signal", not leakage.
    # errstate silences numpy's divide warning for exactly that expected case.
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = result.loc[mask, features].astype("float64").corrwith(y.astype("float64"))
    leaky = correlations[correlations.abs() > LEAKAGE_CORRELATION_THRESHOLD].index.tolist()
    new = [c for c in leaky if c not in original.columns]
    old = [c for c in leaky if c in original.columns]
    return new, old


def validate_output(original: pd.DataFrame, result: object, target_column: str) -> ValidationResult:
    """Check what the generated code returned against the contract.

    Structural failures (not a frame, wrong row count, missing target) return
    early: every later check assumes rows line up with the input.

    Args:
        original: The frame the generated code received (after prepare_target).
        result: Whatever engineer_features returned.
        target_column: The target column name.

    Returns:
        A ValidationResult; ok is True when there are no errors.
    """
    errors: list[str] = []
    warnings: list[str] = []

    def done() -> ValidationResult:
        return ValidationResult(errors=errors, warnings=warnings)

    if not isinstance(result, pd.DataFrame):
        errors.append(f"{ENTRYPOINT} must return a pandas DataFrame, got {type(result).__name__}.")
        return done()

    non_string = [str(c) for c in result.columns if not isinstance(c, str)]
    if non_string:
        errors.append(f"Column names must be strings; got non-string names {_names(non_string)}.")
    duplicated = sorted({str(c) for c in result.columns[result.columns.duplicated()]})
    if duplicated:
        errors.append(f"Column names must be unique; duplicated: {_names(duplicated)}.")
        return done()

    if len(result) != len(original):
        errors.append(
            f"Row count changed from {len(original)} to {len(result)}. Do not drop or add "
            "rows: impute missing values instead of dropping them."
        )
        return done()

    if target_column not in result.columns:
        errors.append(f"Target column {target_column!r} is missing; return it unchanged.")
        return done()
    expected = original[target_column].reset_index(drop=True)
    if not result[target_column].reset_index(drop=True).equals(expected):
        errors.append(
            f"Target column {target_column!r} was modified (values or dtype). "
            "Return it exactly as received."
        )

    features = [c for c in result.columns if c != target_column]
    if not features:
        errors.append("No feature columns were returned besides the target.")
        return done()
    if len(features) > MAX_FE_COLUMNS:
        errors.append(
            f"{len(features)} feature columns exceeds the limit of {MAX_FE_COLUMNS}; "
            "one-hot encode only low-cardinality columns."
        )

    non_numeric = [
        c
        for c in features
        if not (pd.api.types.is_numeric_dtype(result[c]) or pd.api.types.is_bool_dtype(result[c]))
    ]
    if non_numeric:
        dtypes = ", ".join(f"{c!r}: {result[c].dtype}" for c in non_numeric[:_MAX_NAMES_IN_MESSAGE])
        errors.append(
            f"Every column except the target must be numeric or bool. Non-numeric: {dtypes}"
            f"{' ...' if len(non_numeric) > _MAX_NAMES_IN_MESSAGE else ''}. Encode or drop them."
        )

    numeric = [c for c in features if c not in non_numeric]
    values = result[numeric].astype("float64") if numeric else pd.DataFrame()
    with_nan = [c for c in numeric if values[c].isna().any()]
    if with_nan:
        errors.append(f"Feature columns contain NaN: {_names(with_nan)}. Impute them.")
    with_inf = [c for c in numeric if values[c].abs().eq(float("inf")).any()]
    if with_inf:
        errors.append(
            f"Feature columns contain infinite values: {_names(with_inf)}. "
            "Guard divisions and logs against zero."
        )

    if numeric and not with_nan and not with_inf:
        new_leaks, original_leaks = find_leaks(original, result, target_column, numeric)
        if new_leaks:
            errors.append(
                f"New columns {_names(new_leaks)} correlate above "
                f"{LEAKAGE_CORRELATION_THRESHOLD} with the target: they leak it. Do not "
                "derive features from the target."
            )
        if original_leaks:
            warnings.append(
                f"Original columns {_names(original_leaks)} correlate above "
                f"{LEAKAGE_CORRELATION_THRESHOLD} with the target; possible leakage in the "
                "dataset itself."
            )

    return done()
