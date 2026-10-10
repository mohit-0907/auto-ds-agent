"""LangGraph wiring: which node runs, and what runs next.

Nodes do the work and write results into state; the routing functions below
read state and pick the next node. Routing is plain Python on purpose -- each
function is a few lines, testable with a dict, no LLM or graph needed.

    profiler -> router -> sanity_check -+- no usable decision --> END
                                        +-> fe_codegen
    fe_codegen -+- LLM call failed ------------------------> fallback_fe -> END
                +-> sandbox -+- passed ---------------------> arena (END for now)
                             +- failed, attempts left ------> fe_codegen (retry)
                             +- failed MAX_FE_ATTEMPTS -----> fallback_fe

Early exits go to END until the reporter exists (Phase 6), and "arena" is a
named route pointing at END until the arena exists (Phase 5); each is a
one-line change in build_graph when its node arrives.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from autods.config import MAX_FE_ATTEMPTS, RUNS_DIR
from autods.nodes.fallback_fe import fallback_fe_node
from autods.nodes.fe_codegen import fe_codegen_node
from autods.nodes.profiler import profiler_node
from autods.nodes.router import router_node
from autods.nodes.sandbox import sandbox_node
from autods.nodes.sanity_check import sanity_check_node
from autods.state import AgentState

# Each node runs once except fe_codegen and sandbox, which run up to
# MAX_FE_ATTEMPTS times each. The routing functions enforce the real bound
# (invariant 5); this is a second net that turns a future routing bug into a
# clean GraphRecursionError instead of an endless loop.
RECURSION_LIMIT = 10 + 2 * MAX_FE_ATTEMPTS


def route_after_sanity_check(state: AgentState) -> Literal["fe_codegen", "end"]:
    """Continue only with a supported task decision.

    No decision means profiling or routing failed (each already left a
    warning); "unsupported" may come from the router or a sanity-check override.
    """
    decision = state["task_decision"]
    if decision is None or decision.task_type == "unsupported":
        return "end"
    return "fe_codegen"


def route_after_codegen(state: AgentState) -> Literal["sandbox", "fallback_fe"]:
    """No code means the LLM call itself failed: nothing to run, use the fallback.

    Checked here rather than after the sandbox, because a failed LLM call adds
    nothing to fe_code_history -- an attempt counter would never reach the
    limit, and routing back to codegen could loop for as long as the API is down.
    """
    return "sandbox" if state["fe_code"] is not None else "fallback_fe"


def route_after_execution(state: AgentState) -> Literal["arena", "fe_codegen", "fallback_fe"]:
    """The corrective loop's decision (invariant 5: the bound lives here, in code).

    Success is read from processed_data_path, which the sandbox sets only when
    generated code passed validation. Attempts are counted with
    fe_code_history, which grows by exactly one per generated piece of code.
    """
    if state["processed_data_path"] is not None:
        return "arena"
    if len(state["fe_code_history"]) < MAX_FE_ATTEMPTS:
        return "fe_codegen"
    return "fallback_fe"


def build_graph() -> CompiledStateGraph:
    """Wire every node and edge, and compile.

    Returns:
        The compiled graph. Run it with
        graph.invoke(initial_state(...), {"recursion_limit": RECURSION_LIMIT}).
    """
    graph = StateGraph(AgentState)

    graph.add_node("profiler", profiler_node)
    graph.add_node("router", router_node)
    graph.add_node("sanity_check", sanity_check_node)
    graph.add_node("fe_codegen", fe_codegen_node)
    graph.add_node("sandbox", sandbox_node)
    graph.add_node("fallback_fe", fallback_fe_node)

    graph.add_edge(START, "profiler")
    # A failed profile needs no special edge: the router skips when there is no
    # profile, and route_after_sanity_check then ends the run.
    graph.add_edge("profiler", "router")
    graph.add_edge("router", "sanity_check")
    graph.add_conditional_edges(
        "sanity_check", route_after_sanity_check, {"fe_codegen": "fe_codegen", "end": END}
    )
    graph.add_conditional_edges(
        "fe_codegen", route_after_codegen, {"sandbox": "sandbox", "fallback_fe": "fallback_fe"}
    )
    graph.add_conditional_edges(
        "sandbox",
        route_after_execution,
        {"arena": END, "fe_codegen": "fe_codegen", "fallback_fe": "fallback_fe"},
    )
    graph.add_edge("fallback_fe", END)

    return graph.compile()


def new_run_dir(dataset_path: str | Path, root: Path = RUNS_DIR) -> Path:
    """Create this run's artifact folder, e.g. runs/20261010_153000_titanic-dataset/.

    Args:
        dataset_path: The dataset being run; its file stem names the folder.
        root: Parent folder for all runs.

    Returns:
        The created folder. Unique per second; a clash adds a numeric suffix.
    """
    stem = Path(dataset_path).stem.lower().replace(" ", "-")
    base = root / f"{datetime.now():%Y%m%d_%H%M%S}_{stem}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}_{n}")
    path.mkdir(parents=True)
    return path


def initial_state(
    dataset_path: str | Path,
    user_description: str,
    target_column: str | None,
    run_dir: str | Path,
) -> AgentState:
    """Build the starting state: the inputs, and every other key empty.

    AgentState promises every key is always present, so nodes can read any key
    without guarding against it being absent.
    """
    return AgentState(
        dataset_path=str(dataset_path),
        user_description=user_description,
        target_column=target_column,
        run_dir=str(run_dir),
        data_profile=None,
        task_decision=None,
        fe_code=None,
        fe_code_history=[],
        fe_explanations=[],
        tracebacks=[],
        processed_data_path=None,
        leaderboard=[],
        best_model_id=None,
        report=None,
        warnings=[],
        token_cost_usd=0.0,
    )
