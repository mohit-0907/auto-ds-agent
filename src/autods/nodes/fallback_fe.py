"""Fallback feature engineering: the spare tyre for when codegen gives up.

Used when generated code fails MAX_FE_ATTEMPTS times, or the codegen LLM
cannot be reached at all, so the run degrades instead of dying (invariant 6).
Plain deterministic pandas, no LLM: the same input always gives the same
features.

It runs in-process rather than in the sandbox -- this is our own tested code,
not untrusted LLM output -- but its output still goes through validate_output.
The fallback obeys the same contract as generated code, and if it ever broke
it we want to know before the arena trains on it.

No scaling, deliberately, matching the codegen prompt: the arena scales inside
each model's pipeline, so both FE paths hand it the same kind of input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from autods.config import MAX_CATEGORICAL_UNIQUE
from autods.nodes.profiler import load_dataset
from autods.sandbox.contract import prepare_target, validate_output
from autods.state import AgentState

log = structlog.get_logger(__name__)

MISSING_LEVEL = "missing"


def _looks_like_row_id(series: pd.Series, n_rows: int) -> bool:
    """An integer that is unique per row AND a gap-free run, like 1..891.

    Uniqueness alone is not enough: on a small dataset a real measurement
    (square footage, a price in cents) can easily be unique on every row.
    Row IDs are also consecutive; measurements almost never are.
    """
    if not pd.api.types.is_integer_dtype(series) or n_rows < 2:
        return False
    if series.isna().any() or series.nunique() != n_rows:
        return False
    return int(series.max()) - int(series.min()) + 1 == n_rows


def _numeric_feature(series: pd.Series, n_rows: int) -> pd.Series | None:
    """Median-impute a numeric column, or None to drop it."""
    if _looks_like_row_id(series, n_rows):
        return None
    values = series.replace([np.inf, -np.inf], np.nan)
    if values.isna().all():
        return None  # nothing to impute from
    return values.fillna(values.median()) if values.isna().any() else values


def _one_hot(series: pd.Series, name: str) -> pd.DataFrame | None:
    """One-hot encode a low-cardinality text column, or None to drop it."""
    if series.nunique(dropna=True) > MAX_CATEGORICAL_UNIQUE:
        return None  # names, tickets, free text: too many levels to encode
    # astype("string") first: fillna on a categorical rejects unseen levels.
    levels = series.astype("string").fillna(MISSING_LEVEL)
    return pd.get_dummies(levels, prefix=name, prefix_sep="_", dtype=bool)


def _unique_names(columns: list[str]) -> list[str]:
    """Suffix repeats: column 'a_b' level 'c' and column 'a' level 'b_c' both give 'a_b_c'."""
    seen: dict[str, int] = {}
    out = []
    for name in columns:
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}__{count}")
    return out


def basic_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Turn a raw frame into model-ready features with a fixed, simple recipe.

    numeric          -> keep, median-impute (inf treated as missing)
    integer ID       -> drop (one distinct value per row)
    text, <= 20 lvl  -> one-hot, missing becomes its own level
    text, > 20 lvl   -> drop
    bool             -> keep
    datetime, other  -> drop
    target           -> unchanged, appended last

    Args:
        df: The prepared frame (after prepare_target). Not modified.
        target_column: The target column name.

    Returns:
        Features plus the untouched target, same rows and index as df.
    """
    pieces: list[pd.Series | pd.DataFrame] = []
    for name in df.columns:
        if name == target_column:
            continue
        series = df[name]
        if pd.api.types.is_bool_dtype(series):
            pieces.append(series)
        elif pd.api.types.is_numeric_dtype(series):
            feature = _numeric_feature(series, len(df))
            if feature is not None:
                pieces.append(feature)
        elif pd.api.types.is_datetime64_any_dtype(series):
            continue
        else:
            encoded = _one_hot(series, str(name))
            if encoded is not None:
                pieces.append(encoded)

    features = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)
    features.columns = _unique_names([str(c) for c in features.columns])
    features[target_column] = df[target_column]
    return features


def fallback_fe_node(state: AgentState) -> dict[str, Any]:
    """Graph node: build features with basic_features and write them to run_dir.

    Args:
        state: Current graph state. Reads task_decision, dataset_path, run_dir,
            fe_code_history (for the warning and to avoid repeating notes).

    Returns:
        processed_data_path plus a warning saying the fallback was used; or,
        if even the fallback fails, only a warning (the graph then ends the run).

    Raises:
        ValueError: If run_dir is not set (a wiring bug, as in the sandbox node).
    """
    decision = state["task_decision"]
    if decision is None or decision.task_type == "unsupported":
        return {}
    if not state["run_dir"]:
        raise ValueError("state['run_dir'] must be set before the fallback FE node runs")

    n_attempts = len(state["fe_code_history"])
    reason = (
        f"after {n_attempts} failed generated-code attempts"
        if n_attempts
        else "because feature engineering code could not be generated"
    )

    try:
        df = load_dataset(state["dataset_path"])
    except (OSError, ValueError) as exc:
        return {"warnings": [f"Fallback FE could not load the dataset: {exc}"]}

    df, notes = prepare_target(df, decision.target_column, decision.task_type)
    # If any attempt reached the sandbox, it already reported the conversion.
    warnings = [] if n_attempts else list(notes)

    features = basic_features(df, decision.target_column)
    validation = validate_output(df, features, decision.target_column)
    if not validation.ok:
        log.error("fallback_fe.invalid", errors=validation.errors)
        return {
            "warnings": warnings
            + [f"Fallback FE failed validation {reason}: " + " ".join(validation.errors)]
        }

    out_dir = Path(state["run_dir"]) / "fe" / "fallback"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    features.to_parquet(out_path)

    log.info("fallback_fe.done", n_features=features.shape[1] - 1, reason=reason)
    return {
        "processed_data_path": str(out_path),
        "warnings": warnings
        + validation.warnings
        + [f"Used the fallback feature engineering {reason}."],
    }
