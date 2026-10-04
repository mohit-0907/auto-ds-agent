"""Router eval: real LLM calls against labelled datasets (Phase 2 DoD: >= 8/10).

Run explicitly -- it costs money and needs the network:

    uv run pytest tests/eval_router.py -s

The filename does not match pytest's test_*.py pattern, so `uv run pytest` and
CI never collect it. Each case runs the real profiler -> router -> sanity check
pipeline EVAL_RUNS times. A case passes on majority vote; "stable" means every
run produced the same final decision. Scores are reported both for the raw LLM
decision and after the sanity check, so the check's contribution is visible.

Datasets are the public seaborn-data CSVs in datasets/ (mwaskom/seaborn-data),
plus Titanic and Ames. Adversarial and unlabelled variants are derived from them
at run time into a temp directory, so only original files are committed.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from autods.config import PROJECT_ROOT, RANDOM_STATE, ROUTER_MODEL, RUNS_DIR
from autods.nodes.profiler import profiler_node
from autods.nodes.router import router_node
from autods.nodes.sanity_check import sanity_check_node
from tests.helpers import make_state

EVAL_RUNS = 3
PASS_THRESHOLD = 8  # of 10 cases, on majority vote, after the sanity check
DATASETS = PROJECT_ROOT / "datasets"


class EvalCase(BaseModel):
    name: str
    file: str  # a file in datasets/, or a derived file built by _derive()
    description: str
    expected_task_type: str
    expected_target: str | None  # None: the target is not scored (unsupported cases)
    note: str = ""


CASES = [
    EvalCase(
        name="titanic",
        file="Titanic-Dataset.csv",
        description="Predict which passengers survived the sinking.",
        expected_task_type="binary_classification",
        expected_target="Survived",
    ),
    EvalCase(
        name="ames",
        file="ames_train.csv",
        description="Estimate the sale price of a house.",
        expected_task_type="regression",
        expected_target="SalePrice",
        note="81 columns: truncated profile",
    ),
    EvalCase(
        name="iris",
        file="iris.csv",
        description="Identify the species of an iris flower from its measurements.",
        expected_task_type="multiclass_classification",
        expected_target="species",
    ),
    EvalCase(
        name="penguins",
        file="penguins.csv",
        description="Predict a penguin's body mass.",
        expected_task_type="regression",
        expected_target="body_mass_g",
        note="target is not the last column",
    ),
    EvalCase(
        name="diamonds",
        file="diamonds.csv",
        description="Predict the price of a diamond.",
        expected_task_type="regression",
        expected_target="price",
        note="target mid-file",
    ),
    EvalCase(
        name="tips_smoker",
        file="tips.csv",
        description="Predict whether a restaurant customer is a smoker.",
        expected_task_type="binary_classification",
        expected_target="smoker",
        note="text yes/no target",
    ),
    EvalCase(
        name="flights_forecast",
        file="flights.csv",
        description="Forecast the number of airline passengers for each month of next year.",
        expected_task_type="unsupported",
        expected_target=None,
        note="time series",
    ),
    EvalCase(
        name="customers_segment",
        file="derived_customers.csv",
        description="Segment these restaurant customers into groups. We have no labels.",
        expected_task_type="unsupported",
        expected_target=None,
        note="unlabelled; derived from tips",
    ),
    EvalCase(
        name="diamonds_price_text",
        file="derived_diamonds_price_text.csv",
        description="Predict the price of a diamond.",
        expected_task_type="regression",
        expected_target="price",
        note="ADVERSARIAL: price stored as '$1,234' strings",
    ),
    EvalCase(
        name="loans_risk_grade",
        file="derived_loans.csv",
        description="Assign each loan application to its risk grade.",
        expected_task_type="multiclass_classification",
        expected_target="risk_grade",
        note="ADVERSARIAL: 5 classes stored as integers 1-5",
    ),
]


def _derive(out: Path) -> None:
    """Build the derived datasets. Deterministic: same files every run."""
    diamonds = pd.read_csv(DATASETS / "diamonds.csv")
    diamonds["price"] = diamonds["price"].map(lambda p: f"${p:,}")
    diamonds.to_csv(out / "derived_diamonds_price_text.csv", index=False)

    tips = pd.read_csv(DATASETS / "tips.csv")
    tips[["total_bill", "day", "time", "size"]].to_csv(out / "derived_customers.csv", index=False)

    rng = np.random.default_rng(RANDOM_STATE)
    n = 400
    loans = pd.DataFrame(
        {
            "annual_income": rng.lognormal(10.8, 0.4, n).round(-2),
            "loan_amount": rng.lognormal(9.5, 0.6, n).round(-2),
            "debt_to_income": rng.uniform(0.05, 0.6, n).round(3),
            "credit_history_years": rng.integers(0, 30, n),
        }
    )
    score = (
        loans["debt_to_income"] * 5
        - loans["credit_history_years"] / 10
        + loans["loan_amount"] / loans["annual_income"] * 3
        + rng.normal(0, 0.3, n)
    )
    loans["risk_grade"] = pd.qcut(score, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    loans.to_csv(out / "derived_loans.csv", index=False)


def _key(decision: object) -> tuple[str, str] | None:
    return None if decision is None else (decision.task_type, decision.target_column)


def _correct(case: EvalCase, key: tuple[str, str] | None) -> bool:
    if key is None or key[0] != case.expected_task_type:
        return False
    return case.expected_target is None or key[1] == case.expected_target


def _run_once(case: EvalCase, path: Path) -> dict:
    state = make_state(dataset_path=str(path), user_description=case.description)
    state.update(profiler_node(state))
    routed = router_node(state)
    raw = routed["task_decision"]
    checked = sanity_check_node({**state, "task_decision": raw})
    final = checked.get("task_decision", raw)
    return {
        "raw": _key(raw),
        "final": _key(final),
        "raw_correct": _correct(case, _key(raw)),
        "final_correct": _correct(case, _key(final)),
        "confidence": final.confidence if final else None,
        "cost_usd": routed.get("token_cost_usd", 0.0),
        "warnings": routed.get("warnings", []) + checked.get("warnings", []),
    }


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
def test_router_eval(tmp_path: Path) -> None:
    _derive(tmp_path)
    results = []
    for case in CASES:
        path = tmp_path / case.file if case.file.startswith("derived_") else DATASETS / case.file
        runs = [_run_once(case, path) for _ in range(EVAL_RUNS)]
        finals = Counter(r["final"] for r in runs)
        results.append(
            {
                "case": case.model_dump(),
                "runs": runs,
                "raw_passed": sum(r["raw_correct"] for r in runs) > EVAL_RUNS / 2,
                "final_passed": sum(r["final_correct"] for r in runs) > EVAL_RUNS / 2,
                "stable": len(finals) == 1,
                "majority": finals.most_common(1)[0][0],
            }
        )

    n_calls = len(CASES) * EVAL_RUNS
    summary = {
        "model": ROUTER_MODEL,
        "runs_per_case": EVAL_RUNS,
        "cases_passed_raw": sum(r["raw_passed"] for r in results),
        "cases_passed_final": sum(r["final_passed"] for r in results),
        "run_accuracy_raw": sum(x["raw_correct"] for r in results for x in r["runs"]) / n_calls,
        "run_accuracy_final": sum(x["final_correct"] for r in results for x in r["runs"]) / n_calls,
        "stable_cases": sum(r["stable"] for r in results),
        "total_cost_usd": sum(x["cost_usd"] for r in results for x in r["runs"]),
    }
    summary["mean_cost_per_call_usd"] = summary["total_cost_usd"] / n_calls

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"eval_router_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str))

    print(f"\n\nRouter eval -- {ROUTER_MODEL}, {EVAL_RUNS} runs per case\n")
    print(f"{'case':22} {'expected':38} {'majority final':38} raw  final stable")
    for r in results:
        case = r["case"]
        expected = f"{case['expected_task_type']} / {case['expected_target'] or '-'}"
        majority = " / ".join(r["majority"]) if r["majority"] else "(no decision)"
        raw_n = sum(x["raw_correct"] for x in r["runs"])
        final_n = sum(x["final_correct"] for x in r["runs"])
        mark = "PASS" if r["final_passed"] else "FAIL"
        stable = "yes" if r["stable"] else "NO"
        print(
            f"{case['name']:22} {expected:38} {majority:38} "
            f"{raw_n}/{EVAL_RUNS}  {final_n}/{EVAL_RUNS}   {stable:6} {mark}"
        )
    print(
        f"\nCases passed: raw {summary['cases_passed_raw']}/{len(CASES)}, "
        f"after sanity check {summary['cases_passed_final']}/{len(CASES)}"
    )
    print(
        f"Run accuracy: raw {summary['run_accuracy_raw']:.0%}, "
        f"final {summary['run_accuracy_final']:.0%}"
    )
    print(f"Stable cases: {summary['stable_cases']}/{len(CASES)}")
    print(
        f"Cost: ${summary['total_cost_usd']:.3f} total, "
        f"${summary['mean_cost_per_call_usd']:.4f} per call"
    )
    print(f"Saved: {out}")

    assert summary["cases_passed_final"] >= PASS_THRESHOLD
