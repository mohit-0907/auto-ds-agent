"""Sandbox node: runs the latest generated FE code and records the outcome in state.

The bridge between codegen and the executor. It never decides what happens
next -- that is the graph's job (Phase 4), reading what this node wrote:

    processed_data_path set      -> the code worked; go to the arena
    a new entry in tracebacks    -> it failed; retry codegen, or fall back
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from autods.nodes.profiler import load_dataset
from autods.sandbox.contract import prepare_target
from autods.sandbox.executor import execute_fe_code
from autods.state import AgentState

log = structlog.get_logger(__name__)


def sandbox_node(state: AgentState) -> dict[str, Any]:
    """Graph node: execute state["fe_code"] on the full dataset.

    Each attempt gets its own folder, run_dir/fe/attempt_<n>/, kept after the
    run so failed attempts can be inspected and reported.

    Args:
        state: Current graph state. Reads fe_code, fe_code_history (for the
            attempt number), task_decision, dataset_path, run_dir.

    Returns:
        On success: processed_data_path plus any validation warnings.
        On failure: one entry appended to tracebacks -- the
        ExecutionResult.feedback() that codegen's next attempt will read.
        An empty dict when there is no code to run.

    Raises:
        ValueError: If run_dir is not set. That is a wiring bug in whoever
            started the run, not a failure of the generated code.
    """
    code, decision = state["fe_code"], state["task_decision"]
    if code is None or decision is None:
        return {}
    if not state["run_dir"]:
        raise ValueError("state['run_dir'] must be set before the sandbox node runs")

    attempt = max(len(state["fe_code_history"]), 1)
    work_dir = Path(state["run_dir"]) / "fe" / f"attempt_{attempt}"

    try:
        df = load_dataset(state["dataset_path"])
    except (OSError, ValueError) as exc:
        # The profiler loaded this file moments ago, so this is near-impossible;
        # degrade rather than crash (invariant 6).
        return {
            "tracebacks": [f"[load_error at setup stage] Could not load the dataset: {exc}"],
            "warnings": [f"Sandbox could not load the dataset: {exc}"],
        }

    df, notes = prepare_target(df, decision.target_column, decision.task_type)
    # The conversion happens identically on every attempt; report it once.
    warnings = notes if attempt == 1 else []

    result = execute_fe_code(code, df, decision.target_column, work_dir)
    log.info(
        "sandbox.done",
        attempt=attempt,
        status=result.status,
        stage=result.stage,
        duration_s=round(result.duration_s, 2),
    )
    if result.ok:
        validation_warnings = result.validation.warnings if result.validation else []
        return {
            "processed_data_path": result.output_path,
            "warnings": warnings + validation_warnings,
        }
    return {"tracebacks": [result.feedback()], "warnings": warnings}
