"""Deterministic sanity check on the router's decision: LLM proposes, code verifies.

The policy is tiered. Where the target's statistics make the LLM's task type
impossible or very likely wrong, it is overridden. Where they only make it
debatable, the decision stands but is flagged and its confidence capped. Code
corrects facts; judgment calls stay with the LLM, visibly marked.

Rules apply in sequence to the progressively corrected decision, so they
compose: "binary" on 600 float values becomes multiclass (3+ values), then
regression (too many levels to be classes).
"""

from __future__ import annotations

from typing import Any

import structlog

from autods.config import (
    SANITY_FEW_INTEGER_LEVELS,
    SANITY_FLAGGED_CONFIDENCE,
    SANITY_MAX_CLASSES,
)
from autods.sandbox.contract import strip_number_decorations
from autods.state import AgentState, ColumnProfile, TaskDecision

log = structlog.get_logger(__name__)


def _is_numeric(target: ColumnProfile) -> bool:
    # The profiler computes min/max only for numeric dtypes (bool included).
    return target.min is not None


def _is_whole_numbered(target: ColumnProfile) -> bool:
    """Integer dtype, or a float column holding only whole numbers.

    The float case matters: one missing value turns an int column into
    float64 in pandas, and a 1-5 rating with a NaN is still a 1-5 rating.
    Judged from min, max, and the samples -- a heuristic, which is why the
    rule that uses it only flags and never overrides.
    """
    if target.dtype.lower().startswith(("int", "uint")):
        return True
    if not _is_numeric(target) or target.dtype == "bool":
        return False
    values = [target.min, target.max]
    try:
        values += [float(v) for v in target.sample_values]
    except ValueError:
        return False
    return all(v is not None and float(v).is_integer() for v in values)


def _parses_as_numbers(samples: list[str]) -> bool:
    """True if every sample is a number once currency, comma, % and spaces go.

    Shares strip_number_decorations with prepare_target, so the check never
    approves a text target that the conversion step would then refuse.
    """
    if not samples:
        return False
    try:
        for sample in samples:
            float(strip_number_decorations(sample))
    except ValueError:
        return False
    return True


def check_task_type(
    decision: TaskDecision, target: ColumnProfile
) -> tuple[TaskDecision, list[str]]:
    """Verify the router's task type against the target column's statistics.

    Confidence is the LLM's self-report and is not recomputed on override; it
    is only capped when a decision is flagged as debatable.

    Args:
        decision: A validated, supported (not 'unsupported') router decision.
        target: The profile of decision.target_column.

    Returns:
        The possibly corrected decision (a copy; the input is not modified),
        and human-readable notes describing every override and flag.
    """
    notes: list[str] = []
    n = target.n_unique
    name = target.name

    def override(new_type: str, why: str, **extra: Any) -> None:
        nonlocal decision
        notes.append(f"Sanity check overrode task type {decision.task_type} -> {new_type}: {why}")
        decision = decision.model_copy(update={"task_type": new_type, **extra})

    def flag(why: str) -> None:
        nonlocal decision
        notes.append(f"Sanity check flagged {decision.task_type} on {name!r}: {why}")
        capped = min(decision.confidence, SANITY_FLAGGED_CONFIDENCE)
        decision = decision.model_copy(update={"confidence": capped})

    # Certain: the number of classes is a fact, not a judgment.
    if n == 2 and decision.task_type != "binary_classification":
        override("binary_classification", f"{name!r} has exactly 2 distinct values")
    if decision.task_type == "binary_classification" and n >= 3:
        override("multiclass_classification", f"{name!r} has {n} distinct values, not 2")

    if decision.task_type == "multiclass_classification" and n > SANITY_MAX_CLASSES:
        if _is_numeric(target):
            override(
                "regression",
                f"{name!r} is numeric with {n} distinct values -- a quantity, not classes",
            )
        else:
            flag(f"{n} distinct text classes is unusually many; check this is not an ID")

    if decision.task_type == "regression":
        if not _is_numeric(target):
            if _parses_as_numbers(target.sample_values):
                notes.append(
                    f"Target {name!r} holds numbers stored as text "
                    f"(e.g. {target.sample_values[0]!r}); feature engineering must "
                    "convert it before training."
                )
            elif n <= SANITY_MAX_CLASSES:
                override(
                    "multiclass_classification",
                    f"{name!r} is non-numeric text with {n} levels, which cannot be regressed",
                )
            else:
                override(
                    "unsupported",
                    f"{name!r} is free text with {n} distinct values",
                    unsupported_reason=(
                        f"Target {name!r} is non-numeric text with {n} distinct values: "
                        "neither a regression nor a classification target."
                    ),
                )
        elif _is_whole_numbered(target) and n <= SANITY_FEW_INTEGER_LEVELS:
            flag(
                f"only {n} whole-number levels -- could be ordered classes "
                "(e.g. a rating) rather than a quantity"
            )

    return decision, notes


def sanity_check_node(state: AgentState) -> dict[str, Any]:
    """Graph node: verify the router's decision against the profile.

    Pass-through when there is nothing to check: no decision (the router
    already warned) or an 'unsupported' one (no task type to verify).

    Args:
        state: Current graph state. Reads task_decision and data_profile.

    Returns:
        A partial state update: the corrected task_decision and notes in
        warnings, or an empty dict when nothing changed.
    """
    decision = state["task_decision"]
    profile = state["data_profile"]
    if decision is None or profile is None or decision.task_type == "unsupported":
        return {}

    target = next((c for c in profile.columns if c.name == decision.target_column), None)
    if target is None:
        # The router validates the target against the profile, so this is a
        # broken upstream contract. Degrade rather than crash (invariant 6).
        return {
            "warnings": [f"Sanity check skipped: target {decision.target_column!r} not in profile."]
        }

    checked, notes = check_task_type(decision, target)
    if not notes:
        return {}

    log.info(
        "sanity_check.changed",
        before=decision.task_type,
        after=checked.task_type,
        confidence=checked.confidence,
        notes=notes,
    )
    return {"task_decision": checked, "warnings": notes}
