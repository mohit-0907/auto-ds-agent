"""Tests for model construction and cost accounting. No network calls.

Constructing a ChatOpenAI makes no request; it only needs an API key to be
present, so a dummy one is set for each test.
"""

import sys

import pytest
from langchain_openai import ChatOpenAI

from autods.config import (
    FE_CODEGEN_MODEL,
    MODEL_PRICING_PER_MTOK,
    REPORTER_MODEL,
    ROUTER_MODEL,
    estimate_cost_usd,
)
from autods.llm import (
    PROVIDER_KEY_ENV,
    get_llm,
    missing_api_keys,
    missing_key_message,
    parse_model,
)

CONFIGURED_MODELS = (ROUTER_MODEL, FE_CODEGEN_MODEL, REPORTER_MODEL)


@pytest.fixture(autouse=True)
def _dummy_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


def test_parse_model_splits_on_first_colon() -> None:
    assert parse_model("openai:gpt-6.1-sol") == ("openai", "gpt-6.1-sol")


@pytest.mark.parametrize("bad", ["gpt-6.1-sol", ":gpt-6.1-sol", "openai:", ""])
def test_parse_model_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="<provider>:<model-id>"):
        parse_model(bad)


def test_parse_model_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unsupported provider 'mistral'"):
        parse_model("mistral:large")


def test_get_llm_builds_openai_model_without_temperature() -> None:
    llm = get_llm("openai:gpt-6.1-sol")

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-6.1-sol"
    # gpt-6.1-sol returns a 400 for any explicit temperature.
    assert llm.temperature is None


def test_get_llm_explains_missing_anthropic_package(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes the import raise, whether or not the
    # package happens to be installed.
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    with pytest.raises(ImportError, match="uv add langchain-anthropic"):
        get_llm("anthropic:claude-opus-5-5")


@pytest.mark.parametrize("model", [ROUTER_MODEL, FE_CODEGEN_MODEL, REPORTER_MODEL])
def test_configured_models_are_priced(model: str) -> None:
    """A configured model without a price would report its calls as free."""
    assert model in MODEL_PRICING_PER_MTOK


def test_every_priced_model_parses() -> None:
    for model in MODEL_PRICING_PER_MTOK:
        parse_model(model)


def test_estimate_cost_usd() -> None:
    # gpt-6.1-sol: $2 in / $10 out per 1M tokens.
    assert estimate_cost_usd("openai:gpt-6.1-sol", 1_000_000, 0) == pytest.approx(2.0)
    assert estimate_cost_usd("openai:gpt-6.1-sol", 1491, 80) == pytest.approx(0.003782)


def test_unpriced_model_costs_zero() -> None:
    assert estimate_cost_usd("openai:not-in-the-table", 1000, 1000) == 0.0


# --- The API-key check before a run (the real function: imported above, before the
# --- autouse stub in conftest.py replaced the module attribute) ------------------------


def test_no_missing_keys_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert missing_api_keys(["openai:gpt-6-luna", "openai:gpt-6.1-sol"]) == []


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_key_is_missing(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    # Blank is what a freshly copied .env.example gives: "OPENAI_API_KEY=".
    if value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", value)
    assert missing_api_keys(["openai:gpt-6-luna"]) == ["OPENAI_API_KEY"]


def test_each_provider_needs_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    models = ["openai:gpt-6-luna", "anthropic:claude-opus-5-5", "anthropic:claude-sonnet-5-5"]
    assert missing_api_keys(models) == ["ANTHROPIC_API_KEY"]  # once, though two models


def test_default_checks_the_three_configured_models(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROVIDER_KEY_ENV.values():
        monkeypatch.delenv(name, raising=False)
    needed = {PROVIDER_KEY_ENV[parse_model(m)[0]] for m in CONFIGURED_MODELS}
    assert missing_api_keys() == sorted(needed)


def test_missing_key_message_names_the_key_and_the_fix() -> None:
    one = missing_key_message(["OPENAI_API_KEY"])
    assert one.startswith("OPENAI_API_KEY is not set") and "--env-file .env" in one
    two = missing_key_message(["ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
    assert two.startswith("ANTHROPIC_API_KEY and OPENAI_API_KEY are not set")
