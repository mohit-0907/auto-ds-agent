"""Tests for the router node. The LLM is faked (tests/helpers.py); no network calls."""

import operator
from typing import Any, get_type_hints

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from autods.config import MODEL_PRICING_PER_MTOK, PROJECT_ROOT, ROUTER_MODEL
from autods.llm import load_prompt
from autods.nodes.profiler import fit_to_budget, profile_dataset
from autods.nodes.router import build_router_messages, router_node
from autods.state import AgentState, DataProfile, TaskDecision
from tests.helpers import USAGE, FakeStructuredLLM, install_fake_llm, make_state, ok_result

# Derived from the configured router model's rates, so swapping ROUTER_MODEL does
# not break these tests; the rates themselves are tested in test_llm.py.
_IN_RATE, _OUT_RATE = MODEL_PRICING_PER_MTOK[ROUTER_MODEL]
CALL_COST = (USAGE["input_tokens"] * _IN_RATE + USAGE["output_tokens"] * _OUT_RATE) / 1_000_000


@pytest.fixture(scope="module")
def titanic() -> DataProfile:
    df = pd.read_csv(PROJECT_ROOT / "datasets" / "Titanic-Dataset.csv")
    return fit_to_budget(profile_dataset(df, None))


def _decision(**overrides: Any) -> TaskDecision:
    fields = {
        "task_type": "binary_classification",
        "target_column": "Survived",
        "reasoning": "Survived has two values, 0 and 1.",
        "confidence": 0.95,
    }
    return TaskDecision(**(fields | overrides))


_ok = ok_result


def _run(
    monkeypatch: pytest.MonkeyPatch,
    state: AgentState,
    result: dict | None = None,
    exc: Exception | None = None,
) -> tuple[dict, FakeStructuredLLM]:
    fake = install_fake_llm(monkeypatch, result, exc)
    update = router_node(state)
    if fake.schema is not None:
        assert fake.schema is TaskDecision
    return update, fake


# --- Happy path -------------------------------------------------------------------


def test_valid_decision_reaches_state_with_its_cost(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile
) -> None:
    update, _ = _run(monkeypatch, make_state(data_profile=titanic), _ok(_decision()))

    assert update["task_decision"] == _decision()
    assert update["token_cost_usd"] == pytest.approx(CALL_COST)
    assert "warnings" not in update


def test_messages_carry_description_target_and_profile(titanic: DataProfile) -> None:
    (system_role, system), (user_role, user) = build_router_messages(
        titanic, "predict who survived", None
    )

    assert (system_role, user_role) == ("system", "user")
    assert "binary_classification" in system
    assert "predict who survived" in user
    assert "(none -- choose one from the profile)" in user
    # The whole profile arrives intact, braces and all -- and in exactly the
    # serialization fit_to_budget measured, or the token budget means nothing.
    assert titanic.model_dump_json() in user


def test_user_target_is_rendered(titanic: DataProfile) -> None:
    _, (_, user) = build_router_messages(titanic, "", "Survived")
    assert "Survived" in user
    assert "(no description given)" in user


def test_system_prompt_dollar_sign_is_escaped() -> None:
    """string.Template needs $$ for a literal $; a bare "$1,200" raises ValueError."""
    assert '"$1,200"' in load_prompt("router_system")


def test_missing_placeholder_value_raises() -> None:
    with pytest.raises(KeyError):
        load_prompt("router_user", description="x", target="y")


# --- Failure paths degrade to warnings --------------------------------------------


def test_no_profile_skips_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(model: str) -> None:
        raise AssertionError("the LLM must not be called without a profile")

    monkeypatch.setattr("autods.llm.get_llm", explode)
    update = router_node(make_state())

    assert update["task_decision"] is None
    assert "no data profile" in update["warnings"][0]


def test_api_error_becomes_a_warning(monkeypatch: pytest.MonkeyPatch, titanic: DataProfile) -> None:
    update, _ = _run(
        monkeypatch, make_state(data_profile=titanic), exc=ConnectionError("network down")
    )

    assert update["task_decision"] is None
    assert "ConnectionError: network down" in update["warnings"][0]


def test_parse_failure_still_records_cost(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile
) -> None:
    result = {
        "raw": AIMessage(content="not json", usage_metadata=USAGE),
        "parsed": None,
        "parsing_error": ValueError("bad json"),
    }
    update, _ = _run(monkeypatch, make_state(data_profile=titanic), result)

    assert update["task_decision"] is None
    assert update["token_cost_usd"] == pytest.approx(CALL_COST)
    assert "could not parse" in update["warnings"][0]


def test_bad_model_config_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile
) -> None:
    """A malformed ROUTER_MODEL is our bug, not a flaky LLM; it must not become a warning."""
    monkeypatch.setattr("autods.nodes.router.ROUTER_MODEL", "no-provider-prefix")
    with pytest.raises(ValueError, match="<provider>:<model-id>"):
        router_node(make_state(data_profile=titanic))


# --- Decision validation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("decision", "user_target", "message"),
    [
        (_decision(target_column="Lifeboat"), None, "not in the profile"),
        (_decision(target_column="Pclass"), "Survived", "user specified 'Survived'"),
        (_decision(task_type="unsupported"), None, "without an unsupported_reason"),
    ],
)
def test_invalid_decisions_are_rejected_but_billed(
    monkeypatch: pytest.MonkeyPatch,
    titanic: DataProfile,
    decision: TaskDecision,
    user_target: str | None,
    message: str,
) -> None:
    state = make_state(data_profile=titanic, target_column=user_target)
    update, _ = _run(monkeypatch, state, _ok(decision))

    assert update["task_decision"] is None
    assert message in update["warnings"][0]
    assert update["token_cost_usd"] == pytest.approx(CALL_COST)


def test_constant_column_is_rejected_as_target(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"x": range(60), "always_one": [1] * 60})
    profile = profile_dataset(df, None)
    update, _ = _run(
        monkeypatch,
        make_state(data_profile=profile),
        _ok(_decision(task_type="regression", target_column="always_one")),
    )

    assert update["task_decision"] is None
    assert "constant column" in update["warnings"][0]


def test_unsupported_with_reason_is_accepted_even_without_a_target(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile
) -> None:
    decision = _decision(
        task_type="unsupported", target_column="", unsupported_reason="User wants clustering."
    )
    update, _ = _run(monkeypatch, make_state(data_profile=titanic), _ok(decision))

    assert update["task_decision"] == decision


def test_stray_unsupported_reason_is_cleared(
    monkeypatch: pytest.MonkeyPatch, titanic: DataProfile
) -> None:
    decision = _decision(unsupported_reason="leftover text")
    update, _ = _run(monkeypatch, make_state(data_profile=titanic), _ok(decision))

    assert update["task_decision"].unsupported_reason is None


# --- State contract ----------------------------------------------------------------


def test_token_cost_accumulates_across_nodes() -> None:
    hint = get_type_hints(AgentState, include_extras=True)["token_cost_usd"]
    assert hint.__metadata__ == (operator.add,)
