"""scripts/demo_serve.py: attempt 1 is broken on purpose, attempt 2 repairs it, visibly."""

import importlib.util
from pathlib import Path

import pytest

from autods.arena.candidates import candidates
from autods.config import PROJECT_ROOT
from autods.graph import initial_state
from autods.progress import GraphRun
from autods.state import FECode, ReportNarrative, TaskDecision
from tests.helpers import install_scripted_llm, ok_result

_spec = importlib.util.spec_from_file_location(
    "demo_serve", PROJECT_ROOT / "scripts" / "demo_serve.py"
)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

TITANIC = PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv"
DECISION = TaskDecision(
    task_type="binary_classification", target_column="Survived", reasoning="r", confidence=0.95
)
# Valid code: it only fails because the demo appends its fault to attempt 1.
GOOD = FECode(
    explanation="Impute and encode.",
    code="""
import pandas as pd

def engineer_features(df, target_column):
    out = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    out["Age"] = out["Age"].fillna(out["Age"].median())
    out["Embarked"] = out["Embarked"].fillna("S")
    return pd.get_dummies(out, columns=["Sex", "Embarked"], dtype=bool)
""",
)
NARRATIVE = ReportNarrative(summary="s", task="t", features="f", recommendation="r", caveats="- c")


def test_attempt_one_fails_visibly_and_attempt_two_repairs_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "autods.nodes.arena.candidates",
        lambda task_type: [c for c in candidates(task_type) if c.model_id in {"dummy", "logreg"}],
    )
    calls = install_scripted_llm(
        monkeypatch,
        {
            TaskDecision: [ok_result(DECISION)],
            FECode: [ok_result(GOOD), ok_result(GOOD)],
            ReportNarrative: [ok_result(NARRATIVE)],
        },
    )
    # What demo_serve.main() does before serving.
    monkeypatch.setattr("autods.graph.fe_codegen_node", demo.sabotaged_codegen)

    run = GraphRun(initial_state(TITANIC, "Predict who survived", None, tmp_path / "run"))
    events = list(run.events())

    sandbox = [e for e in events if e.node == "sandbox"]
    assert [e.ok for e in sandbox] == [False, True]
    assert "Row count changed" in sandbox[0].detail
    codegen = [e for e in events if e.node == "fe_codegen"]
    assert "DEMO: fault injected" in codegen[0].detail  # the viewer can see it was staged
    assert "DEMO" not in codegen[1].detail  # attempt 2 is the LLM's own code
    assert calls[FECode] == 2
    assert run.final["best_model_id"] is not None
