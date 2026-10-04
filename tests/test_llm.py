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
from autods.llm import get_llm, parse_model


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
