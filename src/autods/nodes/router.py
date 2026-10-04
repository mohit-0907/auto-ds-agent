"""Task router: the first LLM node.

The LLM proposes a TaskDecision; this module checks it against the profile
before it reaches state. Only contract checks live here -- does the target
exist, is the decision internally consistent. Statistical checks on the target
column belong to the deterministic sanity check (Task 2.3).
"""

from __future__ import annotations

from typing import Any

import structlog

from autods.config import ROUTER_MODEL, estimate_cost_usd
from autods.llm import get_llm, load_prompt
from autods.state import AgentState, DataProfile, TaskDecision

log = structlog.get_logger(__name__)


def build_router_messages(
    profile: DataProfile, description: str, target_column: str | None
) -> list[tuple[str, str]]:
    """Render the router prompt files into chat messages.

    Args:
        profile: The budgeted profile from the profiler node.
        description: The user's natural-language description of their goal.
        target_column: The user-supplied target, or None.

    Returns:
        [("system", ...), ("user", ...)] for a LangChain chat model.
    """
    user = load_prompt(
        "router_user",
        description=description.strip() or "(no description given)",
        target=target_column or "(none -- choose one from the profile)",
        # Compact, not indented: this must be byte-for-byte the text that
        # fit_to_budget measured. indent=2 costs ~1.2k extra tokens on Titanic.
        profile=profile.model_dump_json(),
    )
    return [("system", load_prompt("router_system")), ("user", user)]


def validate_decision(
    decision: TaskDecision, profile: DataProfile, target_column: str | None
) -> TaskDecision:
    """Check an LLM decision against the profile it was made from.

    Violations that make the decision unusable raise. A stray
    unsupported_reason on a supported task is harmless, so it is cleared
    rather than failing the run.

    Args:
        decision: The parsed LLM output.
        profile: The profile the LLM was shown.
        target_column: The user-supplied target, or None.

    Returns:
        The decision, normalized.

    Raises:
        ValueError: If the target is not a column in the profile, contradicts
            the user's target, or an unsupported decision gives no reason.
    """
    if decision.task_type == "unsupported":
        if not decision.unsupported_reason:
            raise ValueError("router said 'unsupported' without an unsupported_reason")
        return decision

    if target_column is not None and decision.target_column != target_column:
        raise ValueError(
            f"router chose target {decision.target_column!r}, "
            f"but the user specified {target_column!r}"
        )
    names = {column.name for column in profile.columns}
    if decision.target_column not in names:
        raise ValueError(f"router chose target {decision.target_column!r}, not in the profile")
    if decision.target_column in profile.constant_columns:
        raise ValueError(f"router chose constant column {decision.target_column!r} as target")

    if decision.unsupported_reason is not None:
        return decision.model_copy(update={"unsupported_reason": None})
    return decision


def router_node(state: AgentState) -> dict[str, Any]:
    """Graph node: ask the LLM for a TaskDecision and validate it.

    Every failure -- no profile, an API error, unparseable output, a decision
    that fails validation -- becomes task_decision=None plus a warning, which
    the graph routes to an early exit (invariant 6). The cost of a call is
    recorded even when its output is then rejected: the tokens were billed.

    Args:
        state: Current graph state. Reads data_profile, user_description,
            target_column.

    Returns:
        A partial state update: task_decision, token_cost_usd (this call's cost
        only; the state key accumulates), and warnings on failure.
    """
    profile = state["data_profile"]
    if profile is None:
        return {"task_decision": None, "warnings": ["Routing skipped: no data profile."]}

    target_column = state["target_column"]
    messages = build_router_messages(profile, state["user_description"], target_column)
    # Outside the try: a malformed ROUTER_MODEL is a configuration bug, and
    # should fail loudly rather than read as "the LLM had a bad day".
    llm = get_llm(ROUTER_MODEL).with_structured_output(TaskDecision, include_raw=True)

    try:
        result = llm.invoke(messages)
    # Broad on purpose, and scoped to the one line that talks to the network:
    # each provider SDK raises its own exception hierarchy, and none of them
    # should crash the run. Our own code is outside this try.
    except Exception as exc:
        log.warning("router.llm_failed", error=f"{type(exc).__name__}: {exc}")
        return {
            "task_decision": None,
            "warnings": [f"Routing failed: LLM call raised {type(exc).__name__}: {exc}"],
        }

    usage = getattr(result["raw"], "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = estimate_cost_usd(ROUTER_MODEL, input_tokens, output_tokens)

    decision = result["parsed"]
    if result["parsing_error"] is not None or decision is None:
        log.warning("router.parse_failed", error=str(result["parsing_error"]))
        return {
            "task_decision": None,
            "token_cost_usd": cost,
            "warnings": [f"Routing failed: could not parse LLM output: {result['parsing_error']}"],
        }

    try:
        decision = validate_decision(decision, profile, target_column)
    except ValueError as exc:
        log.warning("router.invalid_decision", error=str(exc), decision=decision.model_dump())
        return {
            "task_decision": None,
            "token_cost_usd": cost,
            "warnings": [f"Routing failed: {exc}"],
        }

    log.info(
        "router.done",
        task_type=decision.task_type,
        target_column=decision.target_column,
        confidence=decision.confidence,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
    )
    return {"task_decision": decision, "token_cost_usd": cost}
