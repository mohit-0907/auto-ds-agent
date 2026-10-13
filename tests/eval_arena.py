"""Phase 5 DoD, live: the real graph from CSV to leaderboard on five datasets.

    uv run pytest tests/eval_arena.py -s

Not collected by `uv run pytest` or CI (costs money, needs the network).

For each dataset: run the whole graph (real router, real generated features,
real sandbox, full six-model arena), then check
  1. the best model clearly beats the dummy (the selection rule's own test:
     better on >= 4 of 5 folds and on average), and
  2. re-running the arena on the very same features file gives an identical
     leaderboard and winner.

Reproducibility is claimed for the arena only. The generated feature code can
differ between runs (current models reject temperature=0), so two full runs
can train on slightly different features; the arena itself, given the same
features, must not vary at all.

Targets are passed explicitly so this measures the arena, not the router (the
router has its own eval). tips predicts the `tip` amount, not `smoker`: whether
a diner smokes carries almost no signal in that data, so failing to beat the
dummy there would be a fact about the data, not the agent.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pytest

from autods.arena.selection import BASELINE_ID, clearly_beats
from autods.config import PROJECT_ROOT, RUNS_DIR
from autods.graph import RECURSION_LIMIT, build_graph, initial_state, new_run_dir
from autods.nodes.arena import arena_node
from autods.state import AgentState
from tests.test_arena_node import comparable

pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")

DATASETS = PROJECT_ROOT / "datasets"
CASES = [
    ("Titanic-Dataset.csv", "Predict which passengers survived.", "Survived"),
    ("ames_train.csv", "Estimate the sale price of a house.", "SalePrice"),
    ("iris.csv", "Identify the species of an iris flower.", "species"),
    ("penguins.csv", "Predict a penguin's body mass.", "body_mass_g"),
    ("tips.csv", "Predict the tip a restaurant customer leaves.", "tip"),
]
DOD = {"Titanic-Dataset.csv", "ames_train.csv"}

results: list[dict[str, Any]] = []


def _run(file: str, description: str, target: str) -> AgentState:
    dataset = DATASETS / file
    state = initial_state(dataset, description, target, new_run_dir(dataset))
    return build_graph().invoke(state, {"recursion_limit": RECURSION_LIMIT})


@pytest.mark.parametrize(("file", "description", "target"), CASES, ids=[c[0] for c in CASES])
def test_live_graph_to_leaderboard(file: str, description: str, target: str) -> None:
    final = _run(file, description, target)
    board = {r.model_id: r for r in final["leaderboard"]}
    best = board.get(final["best_model_id"])
    beats_dummy = best is not None and clearly_beats(best, board[BASELINE_ID])
    rerun = arena_node(final)  # same features file, same run_dir
    reproducible = comparable(rerun) == comparable(final)

    results.append(
        {
            "dataset": file,
            "task_type": final["task_decision"].task_type,
            "best_model_id": final["best_model_id"],
            "beats_dummy": beats_dummy,
            "reproducible": reproducible,
            "fe_attempts": len(final["fe_code_history"]),
            "used_fallback": "fallback" in (final["processed_data_path"] or ""),
            "cost_usd": round(final["token_cost_usd"], 4),
            "leaderboard": [
                {
                    "model_id": r.model_id,
                    "status": r.status,
                    "metric": r.primary_metric,
                    "mean": r.metrics.get(r.primary_metric),
                    "std": r.metrics_std.get(r.primary_metric),
                    "train_s": round(r.training_time_s, 2),
                }
                for r in final["leaderboard"]
            ],
            "run_dir": final["run_dir"],
        }
    )

    assert reproducible, "the arena gave a different answer on the same features"
    assert beats_dummy, f"{final['best_model_id']} does not clearly beat the dummy on {file}"


def teardown_module() -> None:
    if not results:
        return
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"eval_arena_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\n\nPhase 5 live eval -> {out}\n")
    for r in results:
        dod = "  (DoD)" if r["dataset"] in DOD else ""
        print(
            f"{r['dataset']}{dod}: {r['task_type']}, best={r['best_model_id']}, "
            f"beats dummy={r['beats_dummy']}, reproducible={r['reproducible']}, "
            f"FE attempts={r['fe_attempts']}, fallback={r['used_fallback']}, ${r['cost_usd']}"
        )
        for row in r["leaderboard"]:
            score = "failed" if row["mean"] is None else f"{row['mean']:.4f} ± {row['std']:.4f}"
            print(f"    {row['model_id']:7} {row['metric']:9} {score}   train {row['train_s']}s")
    beats = sum(r["beats_dummy"] for r in results)
    first_try = sum(r["fe_attempts"] == 1 and not r["used_fallback"] for r in results)
    print(f"\nBeats dummy: {beats}/{len(results)}")
    print(f"Generated FE passed on the first attempt: {first_try}/{len(results)}")
    print(f"Total cost: ${sum(r['cost_usd'] for r in results):.4f}")
