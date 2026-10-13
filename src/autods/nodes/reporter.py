"""Reporter: the third and last LLM node. Prose from the LLM, every number checked by Python.

    build_facts (Python)  ->  LLM writes ReportNarrative from the facts
                          ->  verify every number against the facts (Python)
                          ->  one retry if any fail, then a visible warning
                          ->  render Markdown: Python tables + LLM prose

The report is always written, even for failed runs and even when the LLM is
unreachable: the tables and warnings are Python's, so only the prose is lost
(invariant 6).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from autods.config import REPORTER_MODEL
from autods.llm import invoke_structured, load_prompt
from autods.reporting.facts import build_facts, leaderboard_table
from autods.reporting.verify import unverified_numbers
from autods.state import AgentState, ReportNarrative

log = structlog.get_logger(__name__)

NARRATIVE_SECTIONS = ("summary", "task", "features", "recommendation", "caveats")


def _prose(narrative: ReportNarrative) -> str:
    return "\n\n".join(getattr(narrative, s) for s in NARRATIVE_SECTIONS)


def write_narrative(facts: str) -> tuple[ReportNarrative | None, list[str], float, str | None]:
    """Ask the LLM for the narrative, verify its numbers, retry once if needed.

    Args:
        facts: The facts sheet from build_facts.

    Returns:
        (narrative, unverified numbers still present, total cost, error).
        narrative is None only when the LLM call itself failed.
    """
    messages = [
        ("system", load_prompt("reporter_system")),
        ("user", load_prompt("reporter_user", facts=facts)),
    ]
    call = invoke_structured(REPORTER_MODEL, ReportNarrative, messages)
    cost = call.cost_usd
    if call.parsed is None:
        return None, [], cost, call.error

    narrative = call.parsed
    bad = unverified_numbers(_prose(narrative), facts)
    if bad:
        log.warning("reporter.unverified_numbers", attempt=1, numbers=bad)
        retry = load_prompt("reporter_retry", unverified=", ".join(bad), previous=_prose(narrative))
        second = invoke_structured(REPORTER_MODEL, ReportNarrative, [*messages, ("user", retry)])
        cost += second.cost_usd
        if second.parsed is not None:
            narrative = second.parsed
            bad = unverified_numbers(_prose(narrative), facts)
    return narrative, bad, cost, None


def render_report(
    state: AgentState, narrative: ReportNarrative | None, unverified: list[str], note: str | None
) -> str:
    """Assemble the Markdown report: Python's tables and facts around the LLM's prose.

    Args:
        state: The run's final state.
        narrative: The verified prose, or None if it could not be written.
        unverified: Numbers in the prose that no fact accounts for.
        note: Why the narrative is missing, if it is.

    Returns:
        The full report as Markdown.
    """

    def section(field: str) -> str:
        if narrative is None:
            return "_No narrative (see the note at the top)._"
        return getattr(narrative, field).strip()

    decision, profile = state["task_decision"], state["data_profile"]
    lines = [f"# Run report: {Path(state['dataset_path']).name}", ""]
    if narrative is None:
        # Said once, here -- not repeated in every section the prose would have filled.
        lines += [
            f"_Generated {datetime.now():%Y-%m-%d %H:%M} by auto-ds-agent._",
            "",
            f"> **Narrative unavailable.** The report LLM could not be used ({note}). "
            "The tables and warnings below are complete; only the written explanation "
            "is missing.",
            "",
        ]
    else:
        lines += [
            f"_Generated {datetime.now():%Y-%m-%d %H:%M} by auto-ds-agent. "
            f"Narrative by {REPORTER_MODEL}; every number in it was checked against the "
            "run's computed results._",
            "",
        ]
    if unverified:
        lines += [
            "> ⚠️ **Unverified numbers.** These numbers in the narrative do not match any "
            f"computed result and should not be trusted: {', '.join(unverified)}.",
            "",
        ]
    lines += ["## Summary", "", section("summary"), "", "## Dataset", ""]
    if profile is None:
        lines.append("The dataset could not be profiled.")
    else:
        lines.append(
            f"{profile.n_rows:,} rows × {profile.n_cols} columns; "
            f"{profile.n_duplicate_rows:,} duplicate rows."
        )
    lines += ["", "## Task", ""]
    if decision is not None:
        lines += [
            f"**{decision.task_type}** · target `{decision.target_column or '–'}` · "
            f"router confidence {decision.confidence:.2f}",
            "",
        ]
    lines += [section("task"), "", "## Feature engineering", "", section("features"), ""]
    lines += [
        "## Leaderboard",
        "",
        leaderboard_table(state["leaderboard"], state["best_model_id"]),
        "",
        "## Recommendation",
        "",
        section("recommendation"),
        "",
        "## Caveats and next steps",
        "",
        section("caveats"),
        "",
    ]
    if state["warnings"]:
        lines += ["## Warnings recorded during the run", ""]
        lines += [f"- {w}" for w in state["warnings"]]
        lines.append("")
    return "\n".join(lines)


def reporter_node(state: AgentState) -> dict[str, Any]:
    """Graph node: write report.md into run_dir and put the Markdown in state.

    Args:
        state: The run's final state; any stage may be missing.

    Returns:
        report, this node's token_cost_usd, and warnings if the narrative is
        missing or contains unverified numbers.

    Raises:
        ValueError: If run_dir is not set (a wiring bug, as in the other file-writing nodes).
    """
    if not state["run_dir"]:
        raise ValueError("state['run_dir'] must be set before the reporter node runs")

    facts = build_facts(state)
    narrative, unverified, cost, error = write_narrative(facts)
    warnings = []
    if narrative is None:
        warnings.append(f"Report written without narrative: {error}")
    if unverified:
        warnings.append(f"Report narrative contains unverified numbers: {', '.join(unverified)}")

    report = render_report(state, narrative, unverified, error)
    Path(state["run_dir"], "report.md").write_text(report, encoding="utf-8")
    log.info("reporter.done", unverified=unverified, cost_usd=round(cost, 6))
    return {"report": report, "token_cost_usd": cost, "warnings": warnings}
