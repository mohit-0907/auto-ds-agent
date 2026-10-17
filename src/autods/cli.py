"""Command-line entry point: `autods run data.csv --description "..." [--target col]`.

Streams the graph so each stage prints as it finishes -- the corrective loop's
retries are visible live. The only module allowed to print() (CLAUDE.md).

Exit codes, so scripts and CI can tell outcomes apart:
    0  a model was recommended
    1  unexpected crash (Python's default for an uncaught exception)
    2  bad command line (typer/click's convention, e.g. the file does not exist)
    3  the run finished but ended without a model; report.md explains why
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import structlog
import typer

from autods.config import RUNS_DIR
from autods.graph import RECURSION_LIMIT, build_graph, initial_state, new_run_dir
from autods.reporting.facts import METRIC_LABELS, fmt

EXIT_NO_MODEL = 3

app = typer.Typer(
    help="Autonomous data science agent: CSV + description in, ranked models and a report out.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Autonomous data science agent."""
    # A callback keeps `run` a named subcommand (autods run ...) even while it is
    # the only one, so adding commands later does not change how it is called.


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
        (succeeded, text) for the CLI to print.
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


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))


@app.command()
def run(
    dataset: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="CSV or Parquet file to analyse."),
    ],
    description: Annotated[
        str, typer.Option("--description", "-d", help="What you want to predict, in words.")
    ],
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Target column; inferred if omitted."),
    ] = None,
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Where run folders are created.")
    ] = RUNS_DIR,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Profile a dataset, pick the task, engineer features, benchmark models, write a report."""
    _configure_logging(verbose)
    run_dir = new_run_dir(dataset, runs_dir)
    typer.echo(f"Run folder: {run_dir}\n")

    state = initial_state(dataset, description, target, run_dir)
    final, attempt = state, 0
    for mode, chunk in build_graph().stream(
        state, {"recursion_limit": RECURSION_LIMIT}, stream_mode=["updates", "values"]
    ):
        if mode == "values":
            final = chunk
            continue
        for node, update in chunk.items():
            if node == "fe_codegen" and (update or {}).get("fe_code"):
                attempt += 1
            ok, text = describe(node, update or {}, attempt)
            mark = typer.style("✓", fg="green") if ok else typer.style("✗", fg="red")
            typer.echo(f"{mark} {node:<13} {text}")

    report = run_dir / "report.md"
    typer.echo(f"\nReport: {report}   ·   LLM cost ${final['token_cost_usd']:.4f}")
    if final["best_model_id"] is None:
        typer.echo("Ended without a model; the report explains why.")
        raise typer.Exit(code=EXIT_NO_MODEL)
