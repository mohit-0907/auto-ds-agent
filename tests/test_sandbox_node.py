"""Tests for the sandbox node, including the Phase 3 Definition of Done.

PROJECT_PLAN Phase 3 DoD: valid code runs and passes validation; deliberately
broken code produces a clean structured error object, not a crash; an infinite
loop is killed by the timeout. These start real subprocesses.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autods.nodes.fe_codegen import fe_codegen_node
from autods.nodes.profiler import profile_dataset
from autods.nodes.sandbox import sandbox_node
from autods.state import AgentState, FECode, TaskDecision
from tests.helpers import install_fake_llm, make_state, ok_result

GOOD_CODE = """
def engineer_features(df, target_column):
    out = df.copy()
    out["age"] = out["age"].fillna(out["age"].median())
    out["is_male"] = out["sex"].eq("male")
    return out.drop(columns=["sex"])
"""
BROKEN_CODE = "def engineer_features(df, target_column):\n    return df['no_such_column']\n"
LOOP_CODE = "def engineer_features(df, target_column):\n    while True:\n        pass\n"


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "age": rng.uniform(1, 80, n).round(1),
            "sex": rng.choice(["male", "female"], n),
            "survived": rng.integers(0, 2, n),
        }
    )
    df.loc[df.index % 9 == 0, "age"] = np.nan
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _state(dataset: Path, code: str | None, attempt: int = 1, **overrides) -> AgentState:
    decision = TaskDecision(
        task_type="binary_classification", target_column="survived", reasoning="r", confidence=0.9
    )
    history = [f"earlier attempt {i}" for i in range(1, attempt)] + ([code] if code else [])
    defaults = {
        "dataset_path": str(dataset),
        "run_dir": str(dataset.parent / "run"),
        "task_decision": decision,
        "fe_code": code,
        "fe_code_history": history,
    }
    return make_state(**(defaults | overrides))


# --- Phase 3 Definition of Done --------------------------------------------------------


def test_dod_valid_code_runs_and_passes_validation(dataset: Path) -> None:
    update = sandbox_node(_state(dataset, GOOD_CODE))

    assert "tracebacks" not in update
    features = pd.read_parquet(update["processed_data_path"])
    assert len(features) == 120
    assert features.drop(columns="survived").map(np.isreal).all().all()


def test_dod_broken_code_gives_a_structured_error_not_a_crash(dataset: Path) -> None:
    update = sandbox_node(_state(dataset, BROKEN_CODE))  # must not raise

    assert "processed_data_path" not in update
    [feedback] = update["tracebacks"]
    assert feedback.startswith("[runtime_error at full stage]")
    assert "KeyError" in feedback.splitlines()[0]
    assert 'File "generated_fe.py", line 2' in feedback  # the traceback, below the summary


def test_dod_infinite_loop_is_killed_by_the_timeout(
    monkeypatch: pytest.MonkeyPatch, dataset: Path
) -> None:
    monkeypatch.setattr("autods.sandbox.executor.SANDBOX_TIMEOUT_S", 3)
    update = sandbox_node(_state(dataset, LOOP_CODE))

    assert update["tracebacks"][0].startswith("[timeout at full stage] Timed out after 3s")


# --- The loop closes: codegen reads what the sandbox wrote ----------------------------


def test_codegen_retry_sees_the_sandbox_error(
    monkeypatch: pytest.MonkeyPatch, dataset: Path
) -> None:
    state = _state(dataset, BROKEN_CODE)
    state["data_profile"] = profile_dataset(pd.read_csv(dataset), "survived")
    state["tracebacks"] = state["tracebacks"] + sandbox_node(state)["tracebacks"]

    fake = install_fake_llm(monkeypatch, ok_result(FECode(explanation="Fixed.", code=GOOD_CODE)))
    fe_codegen_node(state)

    retry_message = fake.messages[2][1]
    assert BROKEN_CODE.strip() in retry_message
    assert "KeyError" in retry_message


# --- Attempts, warnings, and edge cases ----------------------------------------------


def test_each_attempt_gets_its_own_folder(dataset: Path) -> None:
    sandbox_node(_state(dataset, BROKEN_CODE, attempt=1))
    update = sandbox_node(_state(dataset, GOOD_CODE, attempt=2))

    run = dataset.parent / "run" / "fe"
    assert (run / "attempt_1" / "generated_fe.py").read_text() == BROKEN_CODE
    assert Path(update["processed_data_path"]).parent == run / "attempt_2"


@pytest.mark.parametrize(("attempt", "expect_note"), [(1, True), (2, False)])
def test_target_conversion_is_reported_once(
    tmp_path: Path, attempt: int, expect_note: bool
) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {"carat": np.linspace(0.2, 2.0, 60), "price": [f"${i * 100:,}" for i in range(60)]}
    ).to_csv(path, index=False)
    decision = TaskDecision(
        task_type="regression", target_column="price", reasoning="r", confidence=0.9
    )
    code = "def engineer_features(df, target_column):\n    return df\n"
    state = make_state(
        dataset_path=str(path),
        run_dir=str(tmp_path / "run"),
        task_decision=decision,
        fe_code=code,
        fe_code_history=[code] * attempt,
    )
    update = sandbox_node(state)

    assert "processed_data_path" in update  # the code ran on the converted, numeric target
    assert any("Converted text target" in w for w in update["warnings"]) is expect_note


def test_rejected_code_is_recorded_as_a_failed_attempt(dataset: Path) -> None:
    update = sandbox_node(_state(dataset, "import os\n" + GOOD_CODE))
    assert update["tracebacks"][0].startswith("[rejected at static stage]")


def test_no_code_is_a_no_op(dataset: Path) -> None:
    assert sandbox_node(_state(dataset, None)) == {}


def test_missing_run_dir_fails_loudly(dataset: Path) -> None:
    with pytest.raises(ValueError, match="run_dir"):
        sandbox_node(_state(dataset, GOOD_CODE, run_dir=""))


def test_unloadable_dataset_degrades(tmp_path: Path) -> None:
    state = _state(tmp_path / "gone.csv", GOOD_CODE)
    update = sandbox_node(state)

    assert update["tracebacks"][0].startswith("[load_error at setup stage]")
