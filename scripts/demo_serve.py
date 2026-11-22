"""The web app with attempt 1 deliberately broken, to record the corrective loop.

    uv run python scripts/demo_serve.py            # then open http://127.0.0.1:8000

For demos only. Generated feature code usually passes on the first attempt (10/10 in
tests/eval_fe.py), so an ordinary run rarely shows the retry. This starts the normal
web app with one change: in every run, after the LLM writes attempt 1, a wrapper is
appended that silently drops the last row. The sandbox then reports a real
contract violation ("Row count changed"), and the LLM reads that error and writes
attempt 2 itself. The same fault tests/eval_graph.py injects; nothing in src/ changes.

The fault is not hidden: it is visible in attempt 1's code in the Steps tab (with a
docstring naming this script), and the README's demo caption says it was injected.
Needs OPENAI_API_KEY like any real run.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
import uvicorn

import autods.graph
from autods.api.app import create_app
from autods.config import RUNS_DIR
from autods.nodes.fe_codegen import fe_codegen_node
from autods.state import AgentState

# Appended to attempt 1 only. The LLM's function still runs; the wrapper drops one row.
SABOTAGE = '''

_llm_engineer_features = engineer_features


def engineer_features(df, target_column):
    """DEMO: fault injected by scripts/demo_serve.py to break attempt 1 on purpose."""
    return _llm_engineer_features(df, target_column).iloc[:-1]
'''


def sabotaged_codegen(state: AgentState) -> dict[str, Any]:
    """The real fe_codegen node, with SABOTAGE appended to the first attempt's code."""
    update = fe_codegen_node(state)
    if not state["fe_code_history"] and update.get("fe_code"):
        code = update["fe_code"] + SABOTAGE
        update = {**update, "fe_code": code, "fe_code_history": [code]}
    return update


def main(
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Start the web app in demo mode on 127.0.0.1."""
    # build_graph() runs once per run and reads this name then, so every run gets it.
    autods.graph.fe_codegen_node = sabotaged_codegen
    typer.echo(f"DEMO MODE: attempt 1 of every run is broken on purpose.  http://127.0.0.1:{port}")
    uvicorn.run(create_app(RUNS_DIR), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    typer.run(main)
