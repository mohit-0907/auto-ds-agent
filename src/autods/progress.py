"""Running the graph as a stream of human-readable stage events.

Shared by the CLI and the web API so both describe a run identically: one
place drives graph.stream(), counts feature-engineering attempts, and turns
each node's state update into a StageEvent ("sandbox: attempt 1 failed: ...").
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from autods.graph import RECURSION_LIMIT, build_graph
from autods.reporting.facts import METRIC_LABELS, fmt
from autods.state import AgentState


class StageEvent(BaseModel):
    """One finished graph node, described for people. Sent to the browser as JSON."""

    index: int  # 0, 1, 2, ... in the order nodes finished
    node: str
    ok: bool
    text: str
    attempt: int  # feature-engineering attempt in progress when this node ran (0 = none yet)
    at: str  # ISO timestamp


def _first_warning(update: dict[str, Any]) -> str:
    warnings = update.get("warnings") or []
    return warnings[0] if warnings else "no details"


def describe(node: str, update: dict[str, Any], attempt: int) -> tuple[bool, str]:
    """One progress line for a node's update.

    Args:
        node: The node that just finished.
        update: What it wrote to state (may be empty).
        attempt: The current feature-engineering attempt number.

    Returns:
        (succeeded, text).
    """
    if node == "profiler":
        profile = update.get("data_profile")
        if profile is None:
            return False, _first_warning(update)
        return True, f"{profile.n_rows:,} rows × {profile.n_cols} columns"
    if node == "router":
        decision = update.get("task_decision")
        if decision is None:
            return False, _first_warning(update)
        target = decision.target_column or "-"
        return True, f"{decision.task_type} → {target} (confidence {decision.confidence:.2f})"
    if node == "sanity_check":
        return True, _first_warning(update) if update.get("warnings") else "no changes"
    if node == "fe_codegen":
        if update.get("fe_code") is None:
            return False, "could not generate code → fallback"
        return True, f"attempt {attempt} written"
    if node == "sandbox":
        if update.get("processed_data_path"):
            return True, f"attempt {attempt} passed validation"
        failure = (update.get("tracebacks") or ["no details"])[0].splitlines()[0]
        return False, f"attempt {attempt} failed: {failure[:110]}"
    if node == "fallback_fe":
        if update.get("processed_data_path"):
            return True, "fallback features ready"
        return False, _first_warning(update)
    if node == "arena":
        best_id = update.get("best_model_id")
        best = next((r for r in update.get("leaderboard", []) if r.model_id == best_id), None)
        if best is None:
            return False, _first_warning(update)
        metric = best.primary_metric
        return True, (
            f"best: {best.model_name} "
            f"({METRIC_LABELS.get(metric, metric)} {fmt(best.metrics[metric])})"
        )
    if node == "artifacts":
        return (False, _first_warning(update)) if update.get("warnings") else (True, "saved")
    if node == "reporter":
        if update.get("warnings"):
            return False, f"report.md written, but: {_first_warning(update)}"
        return True, "report.md written"
    if node == "finalize":
        return True, "run.json written"
    return True, ""


class GraphRun:
    """One execution of the graph, consumed as a stream of StageEvents.

    Usage:
        run = GraphRun(initial_state(...))
        for event in run.events():
            ...                     # show it, store it, send it to a browser
        run.final["best_model_id"]  # complete state once events() is exhausted
    """

    def __init__(self, state: AgentState) -> None:
        self.final: AgentState = state

    def events(self) -> Iterator[StageEvent]:
        """Run the graph, yielding one event per finished node; sets self.final as it goes."""
        attempt, index = 0, 0
        for mode, chunk in build_graph().stream(
            self.final, {"recursion_limit": RECURSION_LIMIT}, stream_mode=["updates", "values"]
        ):
            if mode == "values":
                self.final = chunk
                continue
            for node, update in chunk.items():
                update = update or {}
                if node == "fe_codegen" and update.get("fe_code"):
                    attempt += 1
                ok, text = describe(node, update, attempt)
                yield StageEvent(
                    index=index,
                    node=node,
                    ok=ok,
                    text=text,
                    attempt=attempt,
                    at=datetime.now().isoformat(timespec="seconds"),
                )
                index += 1
