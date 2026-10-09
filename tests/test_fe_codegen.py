"""Tests for the FE codegen node. The LLM is faked (tests/helpers.py); no network calls."""

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from autods.config import (
    FE_CODEGEN_MODEL,
    MAX_FE_ATTEMPTS,
    MAX_FE_COLUMNS,
    PROJECT_ROOT,
    SANDBOX_ALLOWED_IMPORTS,
    estimate_cost_usd,
)
from autods.nodes.fe_codegen import build_fe_messages, fe_codegen_node, strip_code_fences
from autods.nodes.profiler import fit_to_budget, profile_dataset
from autods.sandbox.executor import ExecutionResult
from autods.state import DataProfile, FECode, TaskDecision
from tests.helpers import USAGE, install_fake_llm, make_state, ok_result

CODE = "def engineer_features(df, target_column):\n    return df\n"
CALL_COST = estimate_cost_usd(FE_CODEGEN_MODEL, USAGE["input_tokens"], USAGE["output_tokens"])


@pytest.fixture(scope="module")
def titanic() -> DataProfile:
    df = pd.read_csv(PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv")
    return fit_to_budget(profile_dataset(df, None))


@pytest.fixture
def decision() -> TaskDecision:
    return TaskDecision(
        task_type="binary_classification",
        target_column="Survived",
        reasoning="r",
        confidence=0.9,
    )


def _feedback(message: str, traceback: str | None = None) -> str:
    return ExecutionResult(
        status="runtime_error", stage="sample", message=message, traceback=traceback
    ).feedback()


# --- Prompt construction --------------------------------------------------------------


def test_first_attempt_is_system_plus_task(titanic: DataProfile, decision: TaskDecision) -> None:
    (s_role, system), (u_role, user) = build_fe_messages(titanic, decision, "who survived", [])

    assert (s_role, u_role) == ("system", "user")
    assert "def engineer_features(df: pd.DataFrame, target_column: str)" in system
    assert "binary_classification" in user
    assert "`Survived`" in user
    assert titanic.model_dump_json() in user  # the exact text the token budget measured


def test_system_prompt_is_rendered_from_config(
    titanic: DataProfile, decision: TaskDecision
) -> None:
    """The prompt and the static check must not drift apart: both read config."""
    system = build_fe_messages(titanic, decision, "", [])[0][1]

    assert ", ".join(SANDBOX_ALLOWED_IMPORTS) in system
    assert "open()" in system
    assert f"At most {MAX_FE_COLUMNS} feature columns" in system
    assert '"$1,234"' in system  # $$ escape rendered as a literal dollar


def test_retry_adds_previous_code_and_full_feedback(
    titanic: DataProfile, decision: TaskDecision
) -> None:
    feedback = _feedback("engineer_features raised: KeyError: 'Cabin'", "Traceback ...\nKeyError")
    messages = build_fe_messages(titanic, decision, "", [(CODE, feedback)])

    assert len(messages) == 3
    retry = messages[2][1]
    assert f"attempt 2 of {MAX_FE_ATTEMPTS}" in retry
    assert CODE.strip() in retry
    assert "Traceback ...\nKeyError" in retry
    assert "Earlier attempts" not in retry


def test_retry_summarizes_earlier_attempts_in_one_line(
    titanic: DataProfile, decision: TaskDecision
) -> None:
    first = _feedback("raised: KeyError: 'Cabin'", "LONG TRACEBACK ONE")
    second = _feedback("raised: ValueError: could not convert", "LONG TRACEBACK TWO")
    retry = build_fe_messages(titanic, decision, "", [("code one", first), ("code two", second)])[
        2
    ][1]

    assert "Attempt 1: [runtime_error at sample stage] raised: KeyError: 'Cabin'" in retry
    assert "LONG TRACEBACK ONE" not in retry  # earlier attempts: summary only
    assert "code one" not in retry
    assert "LONG TRACEBACK TWO" in retry  # latest attempt: in full
    assert "code two" in retry


def test_shared_prefix_is_identical_across_attempts(
    titanic: DataProfile, decision: TaskDecision
) -> None:
    """Retry context goes last so the provider's prompt cache can reuse the prefix."""
    first = build_fe_messages(titanic, decision, "x", [])
    retry = build_fe_messages(titanic, decision, "x", [(CODE, _feedback("boom"))])
    assert retry[:2] == first


def test_feedback_first_line_is_a_one_line_summary() -> None:
    feedback = _feedback("line one\nline two", "Traceback (most recent call last):\n  ...")
    assert feedback.splitlines()[0] == "[runtime_error at sample stage] line one line two"


# --- The node --------------------------------------------------------------------------


def test_node_returns_code_history_explanation_and_cost(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile, decision: TaskDecision
) -> None:
    fake = install_fake_llm(monkeypatch, ok_result(FECode(explanation="Drop Name.", code=CODE)))
    update = fe_codegen_node(make_state(data_profile=titanic, task_decision=decision))

    assert fake.schema is FECode
    assert update["fe_code"] == CODE
    assert update["fe_code_history"] == [CODE]
    assert update["fe_explanations"] == ["Drop Name."]
    assert update["token_cost_usd"] == pytest.approx(CALL_COST)


def test_node_uses_history_and_tracebacks_for_retry(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile, decision: TaskDecision
) -> None:
    fake = install_fake_llm(monkeypatch, ok_result(FECode(explanation="Fixed.", code=CODE)))
    state = make_state(
        data_profile=titanic,
        task_decision=decision,
        fe_code_history=["old code"],
        tracebacks=[_feedback("raised: KeyError")],
    )
    fe_codegen_node(state)

    assert len(fake.messages) == 3
    assert "old code" in fake.messages[2][1]


def test_markdown_fences_are_stripped(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile, decision: TaskDecision
) -> None:
    fenced = f"```python\n{CODE}```"
    install_fake_llm(monkeypatch, ok_result(FECode(explanation="e", code=fenced)))
    update = fe_codegen_node(make_state(data_profile=titanic, task_decision=decision))

    assert update["fe_code"] == CODE


@pytest.mark.parametrize("code", [CODE, f"```\n{CODE}```", f"  ```py\n{CODE}\n```  "])
def test_strip_code_fences(code: str) -> None:
    assert strip_code_fences(code) == CODE


@pytest.mark.parametrize("task_type", ["unsupported", None])
def test_skipped_without_a_supported_decision(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile, task_type: str | None
) -> None:
    def explode(model: str) -> None:
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr("autods.llm.get_llm", explode)
    decision = (
        None
        if task_type is None
        else TaskDecision(
            task_type=task_type,
            target_column="",
            reasoning="r",
            confidence=0.9,
            unsupported_reason="time series",
        )
    )
    update = fe_codegen_node(make_state(data_profile=titanic, task_decision=decision))

    assert update["fe_code"] is None
    assert "skipped" in update["warnings"][0]


def test_llm_error_degrades_to_warning(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile, decision: TaskDecision
) -> None:
    install_fake_llm(monkeypatch, exc=TimeoutError("slow"))
    update = fe_codegen_node(make_state(data_profile=titanic, task_decision=decision))

    assert update["fe_code"] is None
    assert "fe_code_history" not in update  # a failed call is not an attempt
    assert "TimeoutError: slow" in update["warnings"][0]


def test_parse_failure_still_records_cost(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile, decision: TaskDecision
) -> None:
    result = {
        "raw": AIMessage(content="garbage", usage_metadata=USAGE),
        "parsed": None,
        "parsing_error": ValueError("bad"),
    }
    install_fake_llm(monkeypatch, result)
    update = fe_codegen_node(make_state(data_profile=titanic, task_decision=decision))

    assert update["fe_code"] is None
    assert update["token_cost_usd"] == pytest.approx(CALL_COST)
