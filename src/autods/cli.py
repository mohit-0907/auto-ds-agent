"""Command-line entry points: `autods run ...` and `autods serve`.

`run` streams the graph so each stage prints as it finishes -- the corrective
loop's retries are visible live. `serve` starts the web API. The only module
allowed to print() (CLAUDE.md).

Exit codes, so scripts and CI can tell outcomes apart:
    0  a model was recommended
    1  unexpected crash (Python's default for an uncaught exception)
    2  bad command line (typer/click's convention, e.g. the file does not exist), or
       the API key the configured models need is not set
    3  the run finished but ended without a model; report.md explains why
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import structlog
import typer

from autods import llm
from autods.config import RUNS_DIR, WEB_DIST_DIR
from autods.graph import initial_state, new_run_dir
from autods.progress import GraphRun

EXIT_USAGE = 2  # the same code typer uses for a bad command line
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
    missing = llm.missing_api_keys()
    if missing:  # before the run folder exists: a run that can only fail leaves nothing
        typer.echo(llm.missing_key_message(missing), err=True)
        raise typer.Exit(code=EXIT_USAGE)
    run_dir = new_run_dir(dataset, runs_dir)
    typer.echo(f"Run folder: {run_dir}\n")

    graph_run = GraphRun(initial_state(dataset, description, target, run_dir))
    for event in graph_run.events():
        mark = typer.style("✓", fg="green") if event.ok else typer.style("✗", fg="red")
        typer.echo(f"{mark} {event.node:<13} {event.text}")
    final = graph_run.final

    report = run_dir / "report.md"
    typer.echo(f"\nReport: {report}   ·   LLM cost ${final['token_cost_usd']:.4f}")
    if final["best_model_id"] is None:
        typer.echo("Ended without a model; the report explains why.")
        raise typer.Exit(code=EXIT_NO_MODEL)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Where run folders are created.")
    ] = RUNS_DIR,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Start the web app: the React frontend and the API it calls, on one port."""
    import uvicorn

    from autods.api.app import create_app

    _configure_logging(verbose)
    url = f"http://{host}:{port}"
    if (WEB_DIST_DIR / "index.html").is_file():
        typer.echo(f"autods web app on {url}  ·  API docs at {url}/docs")
    else:
        typer.echo(
            f"autods API on {url}/docs  ·  the web app is not built yet: "
            "run `cd web && npm install && npm run build`, then restart."
        )
    missing = llm.missing_api_keys()
    if missing:  # the pages still work (past runs, downloads); new runs are refused
        typer.echo(f"Warning: {llm.missing_key_message(missing)}", err=True)
    uvicorn.run(
        create_app(runs_dir), host=host, port=port, log_level="info" if verbose else "warning"
    )
