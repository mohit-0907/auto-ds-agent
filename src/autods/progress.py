"""Running the graph as a stream of human-readable stage events.

Shared by the CLI and the web API so both describe a run identically: one
place drives graph.stream(), counts feature-engineering attempts, and turns
each node's state update into a StageEvent ("sandbox: attempt 1 failed: ...").
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from autods.config import (
    EVENT_DETAIL_MAX_CHARS,
    EVENT_DETAIL_PROFILE_COLUMNS,
    MAX_FE_ATTEMPTS,
    MAX_FE_COLUMNS,
    SANDBOX_SAMPLE_ROWS,
)
from autods.graph import RECURSION_LIMIT, build_graph
from autods.nodes.artifacts import BUNDLE_FILE, FE_CODE_FILE
from autods.reporting.facts import METRIC_LABELS, fmt
from autods.state import AgentState

DetailKind = Literal["code", "error"]


class StageEvent(BaseModel):
    """One finished graph node, described for people. Sent to the browser as JSON."""

    index: int  # 0, 1, 2, ... in the order nodes finished
    node: str
    ok: bool
    text: str
    attempt: int  # feature-engineering attempt in progress when this node ran (0 = none yet)
    at: str  # ISO timestamp
    # What an expanded step shows: code, a decision, a traceback. None = nothing to expand.
    detail: str | None = None
    detail_kind: DetailKind | None = None
    # Typed facts for the run header, set only by the stage that learns them
    # (profiler: shape; router / sanity check: task and target), so the browser
    # never has to parse them out of `text` or `detail`.
    n_rows: int | None = None
    n_cols: int | None = None
    task_type: str | None = None
    target_column: str | None = None


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


def _cap(text: str) -> str:
    if len(text) <= EVENT_DETAIL_MAX_CHARS:
        return text
    hidden = len(text) - EVENT_DETAIL_MAX_CHARS
    return f"{text[:EVENT_DETAIL_MAX_CHARS]}\n… ({hidden:,} more characters)"


def _profile_detail(update: dict[str, Any]) -> str | None:
    profile = update.get("data_profile")
    if profile is None:
        return None
    shown = profile.columns[:EVENT_DETAIL_PROFILE_COLUMNS]
    width = max(len(c.name) for c in shown) if shown else 0
    lines = [
        f"{c.name:<{width}}  {c.dtype:<8}  {c.n_unique:,} unique · "
        + (f"{c.null_fraction:.1%} missing" if c.null_fraction else "0 missing")
        for c in shown
    ]
    if profile.n_cols > len(shown):
        lines.append(f"… {profile.n_cols - len(shown)} more columns")
    return "\n".join(lines)


def _code_detail(update: dict[str, Any]) -> str | None:
    code = update.get("fe_code")
    if code is None:
        return None
    # The LLM explains its features (and, on a retry, its diagnosis) before the code;
    # shown as a comment block above it.
    explanations = update.get("fe_explanations") or []
    if not explanations or not explanations[0]:
        return code
    comment = "\n".join(f"# {line}".rstrip() for line in explanations[0].splitlines())
    return f"{comment}\n\n{code}"


def _sandbox_detail(update: dict[str, Any], attempt: int) -> tuple[str, DetailKind]:
    if update.get("processed_data_path"):
        checks = [
            "✓ passed the static check (allowed imports, no blocked builtins)",
            # The update does not say how many rows there were, so this line must hold
            # either way (tables of SANDBOX_SAMPLE_ROWS or fewer skip the sample run).
            f"✓ ran on the full table (after a {SANDBOX_SAMPLE_ROWS:,}-row sample run, "
            "if the table is larger)",
            "✓ same number of rows as the input",
            "✓ target column unchanged",
            "✓ every feature numeric, no NaN or infinity",
            f"✓ at most {MAX_FE_COLUMNS} feature columns",
            "✓ no feature leaks the target",
        ]
        warnings = [f"! {w}" for w in update.get("warnings") or []]
        return "\n".join(checks + warnings), "code"
    feedback = (update.get("tracebacks") or ["no details"])[0]
    next_step = (
        f"→ error sent back to the LLM for attempt {attempt + 1}"
        if attempt < MAX_FE_ATTEMPTS
        else "→ last attempt: switching to the fallback recipe"
    )
    return f"{feedback}\n\n{next_step}", "error"


def _arena_detail(update: dict[str, Any]) -> str | None:
    board = update.get("leaderboard") or []
    scored = [r for r in board if r.status == "success"]
    if not board:
        return None
    folds = max((len(r.fold_scores) for r in scored), default=0)
    lines = [f"{len(scored)} of {len(board)} models scored on the same {folds} folds"]
    lines += [f"• {w}" for w in update.get("warnings") or []]  # failed models, selection notes
    return "\n".join(lines)


def stage_detail(
    node: str, update: dict[str, Any], attempt: int, ok: bool
) -> tuple[str | None, DetailKind | None]:
    """What an expanded step shows in the web app: the evidence behind the one-line text.

    Built only from the node's own state update, like `describe`.

    Args:
        node: The node that just finished.
        update: What it wrote to state (may be empty).
        attempt: The current feature-engineering attempt number.
        ok: Whether `describe` judged the stage a success; failures are shown as errors.

    Returns:
        (detail, kind), capped at EVENT_DETAIL_MAX_CHARS; (None, None) when there is
        nothing more to show than the one-line text.
    """
    warnings = update.get("warnings") or []
    detail: str | None = None
    kind: DetailKind = "code"
    if node == "profiler":
        detail = _profile_detail(update)
    elif node in ("router", "sanity_check") and update.get("task_decision") is not None:
        detail = update["task_decision"].model_dump_json(indent=2)
        if warnings:
            detail = "\n".join(warnings) + "\n\n" + detail
    elif node == "fe_codegen":
        detail = _code_detail(update)
    elif node == "sandbox":
        detail, kind = _sandbox_detail(update, attempt)
    elif node == "arena":
        detail = _arena_detail(update)
    elif node == "artifacts" and not warnings:
        detail = f"{BUNDLE_FILE} · {FE_CODE_FILE} · leaderboard.csv"
    elif node == "reporter" and not warnings:
        detail = "Every number in the narrative matched the computed results."
    if detail is None and len(warnings) > 1:
        # One warning is already the text; several are worth listing.
        detail = "\n".join(f"• {w}" for w in warnings)
    if detail is None:
        return None, None
    return _cap(detail), (kind if ok else "error")


def stage_facts(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """The typed header facts a node's update reveals (shape, task, target), else {}."""
    if node == "profiler" and update.get("data_profile") is not None:
        profile = update["data_profile"]
        return {"n_rows": profile.n_rows, "n_cols": profile.n_cols}
    if node in ("router", "sanity_check") and update.get("task_decision") is not None:
        decision = update["task_decision"]
        return {"task_type": decision.task_type, "target_column": decision.target_column}
    return {}


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
                detail, detail_kind = stage_detail(node, update, attempt, ok)
                yield StageEvent(
                    index=index,
                    node=node,
                    ok=ok,
                    text=text,
                    attempt=attempt,
                    at=datetime.now().isoformat(timespec="seconds"),
                    detail=detail,
                    detail_kind=detail_kind,
                    **stage_facts(node, update),
                )
                index += 1
