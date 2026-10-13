"""Arena node: cross-validate every candidate, rank them, pick the best. No LLM.

Every number on the leaderboard comes from scikit-learn (invariant 1); the
reporter later explains these numbers, it never produces them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from autods.arena.candidates import candidates
from autods.arena.scoring import evaluate_candidate, make_cv, prepare_xy
from autods.arena.selection import rank, select_best
from autods.state import AgentState, ModelResult

log = structlog.get_logger(__name__)


def leaderboard_frame(ranked: list[ModelResult], best_model_id: str | None) -> pd.DataFrame:
    """Flatten ranked results into one row per model, for leaderboard.csv.

    Args:
        ranked: Results in leaderboard order.
        best_model_id: The chosen model, marked in an is_best column.

    Returns:
        One row per model: rank, identity, every metric's mean and std, times, error.
    """
    rows = []
    for position, result in enumerate(ranked, start=1):
        row: dict[str, Any] = {
            "rank": position if result.status == "success" else None,
            "model_id": result.model_id,
            "model_name": result.model_name,
            "status": result.status,
            "is_best": result.model_id == best_model_id,
            "primary_metric": result.primary_metric,
        }
        for name, value in result.metrics.items():
            row[name] = value
            row[f"{name}_std"] = result.metrics_std.get(name)
        row["training_time_s"] = result.training_time_s
        row["predict_time_s"] = result.predict_time_s
        row["error"] = result.error
        rows.append(row)
    return pd.DataFrame(rows)


def arena_node(state: AgentState) -> dict[str, Any]:
    """Graph node: benchmark every candidate on the processed features.

    Models and their CV folds run one after another, single-threaded (see
    config.CV_N_JOBS and RF_N_JOBS for the measured reasons). A failing model
    is recorded and the arena goes on.

    Args:
        state: Current graph state. Reads processed_data_path, task_decision,
            run_dir.

    Returns:
        leaderboard (ranked ModelResults), best_model_id, and warnings; writes
        run_dir/leaderboard.csv. An empty dict when there is nothing to train on.

    Raises:
        ValueError: If run_dir is not set (a wiring bug, as in the other file-writing nodes).
    """
    path, decision = state["processed_data_path"], state["task_decision"]
    if path is None or decision is None or decision.task_type == "unsupported":
        return {}
    if not state["run_dir"]:
        raise ValueError("state['run_dir'] must be set before the arena node runs")

    task_type = decision.task_type
    X, y, _, warnings = prepare_xy(pd.read_parquet(path), decision.target_column, task_type)
    try:
        cv = make_cv(task_type, y)
    except ValueError as exc:
        return {
            "leaderboard": [],
            "best_model_id": None,
            "warnings": [*warnings, f"Arena could not cross-validate: {exc}"],
        }

    results = []
    for candidate in candidates(task_type):
        result = evaluate_candidate(candidate, X, y, task_type, cv)
        log.info(
            "arena.scored",
            model_id=result.model_id,
            status=result.status,
            score=result.metrics.get(result.primary_metric),
        )
        results.append(result)

    ranked = rank(results)
    best_model_id, notes = select_best(results)
    failed = [f"{r.model_name} failed in the arena: {r.error}" for r in ranked if r.error]

    out_path = Path(state["run_dir"]) / "leaderboard.csv"
    leaderboard_frame(ranked, best_model_id).to_csv(out_path, index=False)

    log.info("arena.done", best_model_id=best_model_id, leaderboard=str(out_path))
    return {
        "leaderboard": ranked,
        "best_model_id": best_model_id,
        "warnings": warnings + failed + notes,
    }
