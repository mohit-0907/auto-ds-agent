"""Deterministic pandas profiling.

No LLM calls live in this module (invariant 1): every number the router sees
about a dataset is computed here, in Python, and is exact.
"""

from __future__ import annotations

import difflib
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from autods.config import (
    CHARS_PER_TOKEN,
    HIGH_CARDINALITY_RATIO,
    MAX_CATEGORICAL_UNIQUE,
    MAX_DATASET_MB,
    MAX_DETAILED_COLUMNS,
    MAX_SAMPLE_VALUE_CHARS,
    MAX_TARGET_CANDIDATES,
    MIN_ROWS,
    N_SAMPLE_VALUES,
    PROFILE_TOKEN_BUDGET,
    SUPPORTED_EXTENSIONS,
    TARGET_NAME_HINTS,
)
from autods.state import AgentState, ColumnProfile, DataProfile

log = structlog.get_logger(__name__)


def _to_float_or_none(value: object) -> float | None:
    """Normalize a pandas aggregation result into a JSON-safe Python float.

    NaN must not reach ColumnProfile. Pydantic accepts it (NaN is a valid
    float), serializes it to JSON null, and parses it back as None -- so a
    profile containing one no longer compares equal to itself after a round
    trip, and the router silently reads a missing statistic as absent rather
    than as uncomputable.

    Args:
        value: A scalar produced by a pandas aggregation such as .mean().

    Returns:
        The value as a Python float, or None if it is null or not convertible.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sample_values(series: pd.Series) -> list[str]:
    """Take up to N_SAMPLE_VALUES distinct non-null values, as strings.

    Distinct rather than head(n): three identical values tell the router
    nothing about the column. Stringified because .unique() yields numpy
    scalars, which are not JSON-serializable. Each value is cut to
    MAX_SAMPLE_VALUE_CHARS, so a column of free-text reviews cannot consume the
    prompt budget on its own.

    Args:
        series: The column to sample.

    Returns:
        Up to N_SAMPLE_VALUES string representations of distinct values.
    """
    distinct = series.dropna().unique()
    return [str(value)[:MAX_SAMPLE_VALUE_CHARS] for value in distinct[:N_SAMPLE_VALUES]]


def _is_categorical(series: pd.Series, n_unique: int, cardinality_ratio: float) -> bool:
    """Decide whether a model would treat this column as discrete levels.

    The rule is hybrid rather than purely dtype-based: Pclass and Survived are
    stored as integers but are categories to any estimator, and a dtype-only
    rule would miss them. High-cardinality text (a Name column) still counts as
    categorical -- it is nominal, just unhelpful, and cardinality_ratio is the
    field that tells the router so.

    A numeric column needs all three of: integer dtype, few distinct values,
    and a low cardinality ratio. Dropping any one of them misclassifies
    something -- floats are measurements rather than levels, and an integer key
    with one value per row is an identifier no matter how short the frame is.

    Args:
        series: The column to classify.
        n_unique: Count of distinct non-null values, already computed.
        cardinality_ratio: n_unique over the dataset row count.

    Returns:
        True if the column reads as categorical.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    # Checked before the numeric branch: is_numeric_dtype is True for bool.
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.api.types.is_integer_dtype(series)
            and n_unique <= MAX_CATEGORICAL_UNIQUE
            and cardinality_ratio < HIGH_CARDINALITY_RATIO
        )
    # Strings, objects, and anything else nominal. Under pandas 3.0 a text
    # column's dtype is `str`, not `object`, so an is_object_dtype check here
    # would miss every string column in the dataset.
    return True


def profile_column(series: pd.Series, n_rows: int) -> ColumnProfile:
    """Summarize one column into a ColumnProfile.

    Args:
        series: The column to profile. Not modified.
        n_rows: Row count of the parent dataset, used as the denominator for
            cardinality_ratio. Passed explicitly rather than derived from
            len(series) so a sampled series still reports a ratio against the
            true dataset size.

    Returns:
        A ColumnProfile. Numeric statistics are None for non-numeric columns,
        and for statistics pandas cannot compute on the given data.

    Raises:
        ValueError: If the series has no name, or n_rows is negative.
    """
    if series.name is None:
        raise ValueError("series must be named; profile_column uses .name as the column name")
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative, got {n_rows}")

    n_unique = int(series.nunique())

    # On an empty series isna().mean() is NaN, which fails null_fraction's
    # [0, 1] bound with a message that points nowhere useful. Guard here rather
    # than at the division, which is the failure you would expect first.
    null_fraction = float(series.isna().mean()) if len(series) > 0 else 0.0
    cardinality_ratio = n_unique / n_rows if n_rows > 0 else 0.0

    stats: dict[str, float | None] = {}
    if pd.api.types.is_numeric_dtype(series):
        # bool takes this branch deliberately: the mean of a boolean column is
        # its positive rate, which is exactly what the router wants to know
        # about a candidate binary target.
        stats = {
            "min": _to_float_or_none(series.min()),
            "max": _to_float_or_none(series.max()),
            "mean": _to_float_or_none(series.mean()),
            "std": _to_float_or_none(series.std()),
            "skew": _to_float_or_none(series.skew()),
        }

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        is_categorical=_is_categorical(series, n_unique, cardinality_ratio),
        null_fraction=null_fraction,
        n_unique=n_unique,
        cardinality_ratio=min(cardinality_ratio, 1.0),
        sample_values=_sample_values(series),
        **stats,
    )


def infer_target_candidates(
    columns: list[ColumnProfile],
    target_column: str | None,
) -> list[str]:
    """Rank columns that plausibly hold the prediction target.

    Proposes rather than decides: the router makes the final call. Ranking is
    by strength of evidence -- an explicit name like "target" beats the
    last-column convention, which beats merely being low-cardinality.

    Args:
        columns: Per-column profiles, in the dataset's original column order.
        target_column: The user-supplied target, or None if not given.

    Returns:
        Candidate column names, most plausible first, no duplicates, at most
        MAX_TARGET_CANDIDATES long. Exactly [target_column] when one is given.

    Raises:
        ValueError: If target_column is given but is not a column in the dataset.
    """
    names = [column.name for column in columns]

    if target_column is not None:
        if target_column in names:
            # The user's intent wins, even over a constant column -- whether it
            # is usable is for the router's sanity check to flag, not to hide.
            return [target_column]
        close = difflib.get_close_matches(target_column, names, n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(f"target column {target_column!r} is not in the dataset.{hint}")

    # A column with at most one distinct value cannot be predicted.
    usable = [column for column in columns if column.n_unique > 1]

    by_name = [column.name for column in usable if column.name.lower() in TARGET_NAME_HINTS]
    last = [usable[-1].name] if usable and usable[-1].name == names[-1] else []
    # Both bounds: n_unique alone lets a Name column through on a short frame,
    # where every passenger is one of only a handful of rows.
    low_cardinality = [
        column.name
        for column in usable
        if column.is_categorical
        and column.n_unique <= MAX_CATEGORICAL_UNIQUE
        and column.cardinality_ratio < HIGH_CARDINALITY_RATIO
    ]

    # dict.fromkeys dedupes while keeping first-seen order; a set would not,
    # and the order is the signal. Dedupe before capping, or duplicates eat slots.
    ranked = list(dict.fromkeys(by_name + last + low_cardinality))
    return ranked[:MAX_TARGET_CANDIDATES]


def profile_dataset(df: pd.DataFrame, target_column: str | None) -> DataProfile:
    """Build the full deterministic profile of a dataset.

    Args:
        df: The dataset. Not modified.
        target_column: The user-supplied target, or None to infer candidates.

    Returns:
        A DataProfile with every column profiled in original order,
        columns_truncated=False (budgeting is a separate step).

    Raises:
        ValueError: If column names are not unique, or target_column is
            given but not present.
    """
    # Checked on the stringified names, since that is what ColumnProfile stores:
    # a frame with both 1 and "1" as headers would otherwise collide silently.
    # Raw duplicates are worse -- df["a"] returns a DataFrame, not a Series.
    names = [str(name) for name in df.columns]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"column names must be unique; duplicated: {duplicates}")

    n_rows = len(df)
    # Positional access sidesteps label lookup entirely, so non-string headers
    # (0, 1, 2 from a headerless CSV) need no special handling.
    columns = [profile_column(df.iloc[:, i], n_rows) for i in range(df.shape[1])]

    return DataProfile(
        n_rows=n_rows,
        n_cols=df.shape[1],
        # deep=True counts the actual string payloads; without it every text
        # value is an 8-byte pointer and the size is badly understated.
        size_mb=float(df.memory_usage(deep=True).sum()) / 1024**2,
        n_duplicate_rows=int(df.duplicated().sum()),
        columns=columns,
        target_candidates=infer_target_candidates(columns, target_column),
        constant_columns=[column.name for column in columns if column.n_unique <= 1],
    )


def estimate_tokens(text: str) -> int:
    """Approximate the token count of a string, without a tokenizer.

    A real count needs either Anthropic's count-tokens endpoint (a network
    call, which tests may not make) or a third-party tokenizer (an extra
    dependency that is still only an approximation of Claude's). Characters
    over CHARS_PER_TOKEN is free and deterministic. It is calibrated for prose;
    digit-heavy JSON can tokenize denser, so the budget is checked against real
    datasets rather than trusted blind.

    Args:
        text: The text that will be sent to the model.

    Returns:
        The estimated token count, rounded up.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def fit_to_budget(profile: DataProfile, max_tokens: int = PROFILE_TOKEN_BUDGET) -> DataProfile:
    """Return a profile whose JSON fits within max_tokens.

    Column detail is kept by relevance: target candidates first, since the
    router must choose among them, then the rest in original column order.
    Detail is dropped from the least relevant end until the profile fits. The
    dataset-level fields (n_cols, constant_columns, target_candidates) are left
    untouched, so they still describe the whole dataset.

    Target candidates are never dropped. If they alone exceed max_tokens, the
    candidates-only profile is returned over budget -- a profile the router
    cannot choose a target from is worse than one that costs more.

    Args:
        profile: A full profile from profile_dataset. Not modified.
        max_tokens: Token ceiling for profile.model_dump_json().

    Returns:
        The input unchanged if it has at most MAX_DETAILED_COLUMNS columns and
        fits. Otherwise a copy with columns_truncated=True, dtype_summary
        filled, and per-column detail kept only for the most relevant columns.
    """
    if len(profile.columns) <= MAX_DETAILED_COLUMNS and (
        estimate_tokens(profile.model_dump_json()) <= max_tokens
    ):
        return profile

    candidates = set(profile.target_candidates)
    candidate_names = [c.name for c in profile.columns if c.name in candidates]
    other_names = [c.name for c in profile.columns if c.name not in candidates]
    by_relevance = candidate_names + other_names
    n_candidates = len(candidate_names)
    dtype_summary = dict(Counter(column.dtype for column in profile.columns))

    # Start at the column cap rather than at every column: on a 300-column
    # frame, trimming one at a time from the top would serialize ~260 times.
    n_keep = min(MAX_DETAILED_COLUMNS, len(by_relevance))
    while True:
        kept_names = set(by_relevance[: max(n_keep, n_candidates)])
        trimmed = profile.model_copy(
            update={
                # Kept columns stay in dataset order, which is easier to read
                # than relevance order and is what the router sees untruncated.
                "columns": [c for c in profile.columns if c.name in kept_names],
                "columns_truncated": True,
                "dtype_summary": dtype_summary,
            }
        )
        if n_keep <= n_candidates or estimate_tokens(trimmed.model_dump_json()) <= max_tokens:
            return trimmed
        n_keep -= 1


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read a dataset file, refusing anything the agent cannot handle.

    Every refusal raises with a message written for the end user, because the
    node passes it through to state["warnings"] verbatim.

    Args:
        path: Path to a CSV or Parquet file.

    Returns:
        The loaded frame.

    Raises:
        FileNotFoundError: If the path does not exist or is not a file.
        ValueError: If the extension is unsupported, the file exceeds
            MAX_DATASET_MB on disk, it cannot be parsed, or it has fewer than
            MIN_ROWS rows.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported file type {suffix!r}; expected one of {list(SUPPORTED_EXTENSIONS)}"
        )

    # Checked on disk, before reading: the point is to refuse a 5 GB file
    # without first trying to fit it in memory. In-memory size is usually
    # larger than on-disk size for CSV, so this is a floor, not an exact cap.
    size_mb = path.stat().st_size / 1024**2
    if size_mb > MAX_DATASET_MB:
        raise ValueError(f"dataset is {size_mb:.0f} MB on disk; the limit is {MAX_DATASET_MB} MB")

    df = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"dataset has {len(df)} rows; at least {MIN_ROWS} are needed for "
            "cross-validated scores to mean anything"
        )
    return df


def profiler_node(state: AgentState) -> dict[str, Any]:
    """Graph node: load the dataset and write a budgeted profile to state.

    Expected failures -- a missing or oversized file, a parse error, too few
    rows, a target column that does not exist -- become a warning and
    data_profile=None, which the graph routes to an early exit (invariant 6).
    Anything else is a bug in this code and is allowed to raise.

    Args:
        state: Current graph state. Reads dataset_path and target_column.

    Returns:
        A partial state update: data_profile, plus warnings when the profile
        was truncated or profiling failed.
    """
    path = state["dataset_path"]
    try:
        df = load_dataset(path)
        full = profile_dataset(df, state["target_column"])
    # OSError covers missing and unreadable files; ValueError covers our own
    # refusals and pandas/pyarrow parse failures, which all subclass it.
    except (OSError, ValueError) as exc:
        log.warning("profiler.failed", dataset_path=path, error=str(exc))
        return {"data_profile": None, "warnings": [f"Profiling failed: {exc}"]}

    profile = fit_to_budget(full)
    warnings: list[str] = []
    if profile.columns_truncated:
        warnings.append(
            f"Profile truncated to fit the prompt budget: {len(profile.columns)} of "
            f"{profile.n_cols} columns shown in detail."
        )

    log.info(
        "profiler.done",
        dataset_path=path,
        n_rows=profile.n_rows,
        n_cols=profile.n_cols,
        detailed_columns=len(profile.columns),
        est_tokens=estimate_tokens(profile.model_dump_json()),
    )
    return {"data_profile": profile, "warnings": warnings}
