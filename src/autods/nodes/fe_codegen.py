"""Feature engineering codegen: the second LLM node.

Writes engineer_features(df, target_column) source for the sandbox to run. It
never runs or judges the code itself -- the executor is the single place code
is checked (invariant 3).

On a retry the prompt carries the last attempt in full (code + executor
feedback) and a one-line summary of each earlier failure, so the model sees
what just broke and what it already tried, at a prompt size that stays roughly
flat across attempts.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from autods.config import (
    FE_CODEGEN_MODEL,
    MAX_CATEGORICAL_UNIQUE,
    MAX_FE_ATTEMPTS,
    MAX_FE_COLUMNS,
    SANDBOX_ALLOWED_IMPORTS,
    SANDBOX_BLOCKED_CALLS,
)
from autods.llm import invoke_structured, load_prompt
from autods.sandbox.contract import ENTRYPOINT
from autods.state import AgentState, DataProfile, FECode, TaskDecision

log = structlog.get_logger(__name__)

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def strip_code_fences(code: str) -> str:
    """Remove a surrounding ```python fence, if the model added one anyway."""
    match = _FENCE.match(code)
    return (match.group(1) if match else code).strip() + "\n"


def build_fe_messages(
    profile: DataProfile,
    decision: TaskDecision,
    description: str,
    attempts: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Render the codegen prompt files into chat messages.

    The system and task messages are identical on every attempt and the retry
    context comes last, so a provider's prompt cache can reuse the prefix.

    Args:
        profile: The budgeted profile from the profiler node.
        decision: The (sanity-checked) task decision.
        description: The user's description of their goal.
        attempts: (code, feedback) for each previous failed attempt, oldest
            first. Empty on the first attempt.

    Returns:
        [system, user] on a first attempt; [system, user, retry] afterwards.
    """
    system = load_prompt(
        "fe_system",
        entrypoint=ENTRYPOINT,
        max_columns=str(MAX_FE_COLUMNS),
        max_onehot=str(MAX_CATEGORICAL_UNIQUE),
        allowed_imports=", ".join(SANDBOX_ALLOWED_IMPORTS),
        blocked_calls=", ".join(f"{name}()" for name in SANDBOX_BLOCKED_CALLS),
    )
    user = load_prompt(
        "fe_user",
        task_type=decision.task_type,
        target_column=decision.target_column,
        description=description.strip() or "(no description given)",
        profile=profile.model_dump_json(),
    )
    messages = [("system", system), ("user", user)]
    if not attempts:
        return messages

    *earlier, (previous_code, feedback) = attempts
    earlier_text = ""
    if earlier:
        lines = "\n".join(
            f"- Attempt {i}: {fb.splitlines()[0]}" for i, (_, fb) in enumerate(earlier, start=1)
        )
        earlier_text = f"\n\nEarlier attempts also failed:\n{lines}"
    retry = load_prompt(
        "fe_retry",
        attempt=str(len(attempts) + 1),
        max_attempts=str(MAX_FE_ATTEMPTS),
        earlier=earlier_text,
        previous_attempt=str(len(attempts)),
        previous_code=previous_code.strip(),
        feedback=feedback.strip(),
    )
    return [*messages, ("user", retry)]


def fe_codegen_node(state: AgentState) -> dict[str, Any]:
    """Graph node: ask the LLM for feature engineering code.

    The attempt number is len(fe_code_history); the bound on attempts is
    enforced by the graph (invariant 5), not here. A failed LLM call returns
    fe_code=None plus a warning, which the graph routes to the fallback FE.

    Args:
        state: Current graph state. Reads data_profile, task_decision,
            user_description, fe_code_history, tracebacks.

    Returns:
        A partial state update: fe_code, and on success one appended entry each
        in fe_code_history and fe_explanations; token_cost_usd; warnings on
        failure.
    """
    profile, decision = state["data_profile"], state["task_decision"]
    if profile is None or decision is None or decision.task_type == "unsupported":
        return {"fe_code": None, "warnings": ["FE codegen skipped: no supported task decision."]}

    attempts = list(zip(state["fe_code_history"], state["tracebacks"], strict=False))
    messages = build_fe_messages(profile, decision, state["user_description"], attempts)
    call = invoke_structured(FE_CODEGEN_MODEL, FECode, messages)
    if call.parsed is None:
        return {
            "fe_code": None,
            "token_cost_usd": call.cost_usd,
            "warnings": [f"FE codegen failed on attempt {len(attempts) + 1}: {call.error}"],
        }

    code = strip_code_fences(call.parsed.code)
    log.info(
        "fe_codegen.done",
        attempt=len(attempts) + 1,
        code_lines=code.count("\n"),
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        cost_usd=round(call.cost_usd, 6),
    )
    return {
        "fe_code": code,
        "fe_code_history": [code],
        "fe_explanations": [call.parsed.explanation.strip()],
        "token_cost_usd": call.cost_usd,
    }
