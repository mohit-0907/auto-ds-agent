"""Phase 8 evaluation, live: feature-engineering success on ten datasets.

    uv run pytest tests/eval_fe.py -s

Not collected by `uv run pytest` or CI (costs money, needs the network).

For each dataset the real graph runs (real router, real generated code, real
sandbox, the graph's own MAX_FE_ATTEMPTS bound) and is stopped as soon as
feature engineering is over: either an attempt passed the sandbox, or the
fallback recipe ran. LangGraph only runs the next node when the stream is read
again, so stopping there means the arena and the reporter never run.

It measures PROJECT_PLAN section 7's two FE rows:
  * first-attempt success (target >= 60%): the first generated code passed.
  * success within MAX_FE_ATTEMPTS (target >= 90%): some generated attempt
    passed. Falling back to the fixed recipe counts as a failure here, even
    though the run itself carries on.

Targets are passed explicitly so this measures feature engineering, not the
router (which has its own eval). One pass per dataset: generated code differs
between runs, so these numbers are one sample, not a stable rate.

It also reports the mean LLM cost of every full run already saved in runs/
(each run.json records token_cost_usd), at no extra cost.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from statistics import mean
from typing import Any

import pytest

from autods.config import MAX_FE_ATTEMPTS, PROJECT_ROOT, RUNS_DIR
from autods.graph import RECURSION_LIMIT, build_graph, initial_state, new_run_dir

pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")

DATASETS = PROJECT_ROOT / "datasets"
CASES = [
    ("Titanic-Dataset.csv", "Predict which passengers survived.", "Survived"),
    ("ames_train.csv", "Estimate the sale price of a house.", "SalePrice"),
    ("diamonds.csv", "Predict the price of a diamond.", "price"),
    ("iris.csv", "Identify the species of an iris flower.", "species"),
    ("mpg.csv", "Predict a car's fuel efficiency in miles per gallon.", "mpg"),
    ("penguins.csv", "Identify the species of a penguin.", "species"),
    ("taxis.csv", "Predict the tip for a taxi ride.", "tip"),
    ("tips.csv", "Predict the tip a restaurant customer leaves.", "tip"),
    ("healthexp.csv", "Predict a country's life expectancy.", "Life_Expectancy"),
    ("planets.csv", "Predict how an exoplanet was discovered.", "method"),
]

results: list[dict[str, Any]] = []


def _feature_engineering(file: str, description: str, target: str) -> dict[str, Any]:
    """Run the graph until feature engineering is over; describe what happened."""
    dataset = DATASETS / file
    state = initial_state(dataset, description, target, new_run_dir(dataset))
    attempts: list[str] = []  # one outcome line per generated attempt
    source, cost, start = None, 0.0, time.monotonic()

    # Updates only: each is what one node just wrote, so nothing depends on how
    # LangGraph orders "updates" and "values" chunks within a step.
    for chunk in build_graph().stream(
        state, {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"
    ):
        node, update = next(iter(chunk.items()))
        update = update or {}
        cost += update.get("token_cost_usd", 0.0)  # each node returns only its own cost
        if node == "sandbox":
            if update.get("processed_data_path"):
                attempts.append("passed")
                source = "generated"
            else:
                attempts.append((update.get("tracebacks") or ["?"])[0].splitlines()[0][:160])
        if node == "fe_codegen" and update.get("fe_code") is None:
            attempts.append("code generation failed (LLM error)")
        if node == "fallback_fe":
            source = "fallback"
        if source is not None or node in {"arena", "reporter"}:
            break  # FE is over (or the run left the FE loop early): stop before the arena

    return {
        "dataset": file,
        "target": target,
        "attempts": attempts,
        "n_attempts": len(attempts),
        "first_attempt": attempts[:1] == ["passed"],
        "within_limit": source == "generated",
        "source": source,
        "cost_usd": round(cost, 4),
        "seconds": round(time.monotonic() - start, 1),
        "run_dir": state["run_dir"],  # each attempt's code and output, under fe/
    }


@pytest.mark.parametrize(("file", "description", "target"), CASES, ids=[c[0] for c in CASES])
def test_live_feature_engineering(file: str, description: str, target: str) -> None:
    result = _feature_engineering(file, description, target)
    results.append(result)
    # Recorded, not asserted: a fallback is a measured outcome, not a broken test.
    assert result["source"] in {"generated", "fallback"}, f"FE never finished on {file}"


def _full_run_costs() -> list[float]:
    """token_cost_usd of every completed run saved in runs/ (CLI or web)."""
    costs = []
    for record in RUNS_DIR.glob("*/run.json"):
        data = json.loads(record.read_text(encoding="utf-8"))
        if data.get("best_model_id") and data.get("token_cost_usd"):
            costs.append(float(data["token_cost_usd"]))
    return costs


def teardown_module() -> None:
    if not results:
        return
    costs = _full_run_costs()
    summary = {
        "datasets": len(results),
        "first_attempt": sum(r["first_attempt"] for r in results),
        "within_limit": sum(r["within_limit"] for r in results),
        "max_fe_attempts": MAX_FE_ATTEMPTS,
        "eval_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "full_runs": {
            "n": len(costs),
            "mean_cost_usd": round(mean(costs), 4) if costs else None,
            "min_cost_usd": round(min(costs), 4) if costs else None,
            "max_cost_usd": round(max(costs), 4) if costs else None,
        },
    }
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"eval_fe_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print(f"\n\nFE eval -> {out}\n")
    for r in results:
        print(
            f"{r['dataset']:<20} {r['source']:<9} attempts={r['n_attempts']} "
            f"${r['cost_usd']:.4f} {r['seconds']:>5}s"
        )
        for i, outcome in enumerate(r["attempts"], start=1):
            print(f"    attempt {i}: {outcome}")
    n = summary["datasets"]
    print(f"\nFirst attempt:        {summary['first_attempt']}/{n}")
    print(f"Within {MAX_FE_ATTEMPTS} attempts:     {summary['within_limit']}/{n}")
    print(f"Eval cost:            ${summary['eval_cost_usd']}")
    full = summary["full_runs"]
    print(
        f"Full runs in runs/:   n={full['n']}, mean ${full['mean_cost_usd']} "
        f"(min ${full['min_cost_usd']}, max ${full['max_cost_usd']})"
    )
