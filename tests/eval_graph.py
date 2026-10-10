"""Phase 4 DoD, live: the whole graph with real LLM calls.

    uv run pytest tests/eval_graph.py -s

Not collected by `uv run pytest` or CI (costs money, needs the network). The
offline equivalents, with a scripted LLM, are in tests/test_graph.py.

1. Injected fault: Sol writes attempt 1 for real; this file adds one bug before
   it runs (it drops a row); Sol must read the real error and repair it.
   Reliable every run -- and honest: we caused the failure, on purpose.
2. Natural trap: Titanic plus a text column that secretly copies the target.
   Sol tends to one-hot it on attempt 1, the validator flags the leak, and the
   retry drops it. Probabilistic, so it is reported as a rate over 3 runs.
3. Impossible case: a 0.01 s sandbox timeout no code can meet, so all attempts
   time out and the fallback (in-process, no timeout) saves the run.

Run artifacts land in runs/ for inspection.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from autods.config import MAX_FE_ATTEMPTS, PROJECT_ROOT, RUNS_DIR
from autods.graph import RECURSION_LIMIT, build_graph, initial_state, new_run_dir
from autods.nodes.fe_codegen import fe_codegen_node
from autods.state import AgentState

pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")

TITANIC = PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv"
NATURAL_TRAP_RUNS = 3

# Appended to attempt 1 only. Wraps the LLM's function so it silently drops the
# last row -- a contract violation the sandbox reports as "Row count changed".
SABOTAGE = '''

_llm_engineer_features = engineer_features


def engineer_features(df, target_column):
    """Fault injected by tests/eval_graph.py to break attempt 1."""
    return _llm_engineer_features(df, target_column).iloc[:-1]
'''

results: dict[str, Any] = {}


def _run(dataset: Path, target: str | None = "Survived") -> AgentState:
    state = initial_state(
        dataset, "Predict which passengers survived.", target, new_run_dir(dataset)
    )
    return build_graph().invoke(state, {"recursion_limit": RECURSION_LIMIT})


def _summary(final: AgentState) -> dict[str, Any]:
    return {
        "run_dir": final["run_dir"],
        "attempts": len(final["fe_code_history"]),
        "failures": [t.splitlines()[0] for t in final["tracebacks"]],
        "processed_data_path": final["processed_data_path"],
        "used_fallback": "fallback" in (final["processed_data_path"] or ""),
        "cost_usd": round(final["token_cost_usd"], 4),
    }


def test_injected_fault_is_repaired_on_attempt_2(monkeypatch: pytest.MonkeyPatch) -> None:
    def sabotage_first_attempt(state: AgentState) -> dict[str, Any]:
        update = fe_codegen_node(state)
        if not state["fe_code_history"] and update.get("fe_code"):
            code = update["fe_code"] + SABOTAGE
            update = {**update, "fe_code": code, "fe_code_history": [code]}
        return update

    # build_graph looks the node function up at call time, so this swaps it in.
    monkeypatch.setattr("autods.graph.fe_codegen_node", sabotage_first_attempt)
    final = _run(TITANIC)
    results["injected_fault"] = summary = _summary(final)

    assert summary["attempts"] == 2
    assert "Row count changed" in summary["failures"][0]
    assert "attempt_2" in summary["processed_data_path"]
    assert "_llm_engineer_features" not in final["fe_code_history"][1]  # Sol removed the bug


def test_natural_leakage_trap(tmp_path: Path) -> None:
    df = pd.read_csv(TITANIC)
    # Mid-file and innocently named: a post-hoc label that copies the target.
    df.insert(3, "cohort", df["Survived"].map({1: "K", 0: "M"}))
    dataset = tmp_path / "titanic_with_cohort.csv"
    df.to_csv(dataset, index=False)

    runs = [_summary(_run(dataset)) for _ in range(NATURAL_TRAP_RUNS)]
    trapped = sum(bool(r["failures"]) and "leak" in r["failures"][0] for r in runs)
    results["natural_trap"] = {"trapped_on_attempt_1": trapped, "runs": runs}

    # Whatever Sol does, every run must end with features -- via repair or fallback.
    assert all(r["processed_data_path"] for r in runs)


def test_impossible_case_exits_via_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autods.sandbox.executor.SANDBOX_TIMEOUT_S", 0.01)
    final = _run(TITANIC)
    results["impossible"] = summary = _summary(final)

    assert summary["attempts"] == MAX_FE_ATTEMPTS
    assert all(f.startswith("[timeout") for f in summary["failures"])
    assert summary["used_fallback"]


def teardown_module() -> None:
    if not results:
        return
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"eval_graph_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n\nPhase 4 live DoD -> {out}")
    for name, value in results.items():
        if name == "natural_trap":
            print(
                f"  natural_trap: trapped on attempt 1 in "
                f"{value['trapped_on_attempt_1']}/{NATURAL_TRAP_RUNS} runs"
            )
            for r in value["runs"]:
                print(
                    f"    attempts={r['attempts']} fallback={r['used_fallback']} "
                    f"cost=${r['cost_usd']} failures={r['failures']}"
                )
        else:
            print(
                f"  {name}: attempts={value['attempts']} fallback={value['used_fallback']} "
                f"cost=${value['cost_usd']} failures={value['failures']}"
            )
