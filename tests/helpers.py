"""Shared test builders and the fake LLM."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from autods.state import AgentState

USAGE = {"input_tokens": 1500, "output_tokens": 80, "total_tokens": 1580}


class FakeStructuredLLM:
    """Stands in for get_llm(...).with_structured_output(schema, include_raw=True).

    Returns the same {"raw", "parsed", "parsing_error"} dict LangChain does, and
    records what it was called with.
    """

    def __init__(self, result: dict | None = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.schema: type | None = None
        self.messages: list | None = None

    def invoke(self, messages: list) -> dict:
        self.messages = messages
        if self.exc is not None:
            raise self.exc
        return self.result


class _FakeLLM:
    def __init__(self, structured: FakeStructuredLLM) -> None:
        self.structured = structured

    def with_structured_output(self, schema: type, include_raw: bool = False) -> FakeStructuredLLM:
        assert include_raw, "include_raw=True is needed to read token usage"
        self.structured.schema = schema
        return self.structured


def install_fake_llm(
    monkeypatch: pytest.MonkeyPatch, result: dict | None = None, exc: Exception | None = None
) -> FakeStructuredLLM:
    """Make every get_llm() call return a fake; returns it for inspection."""
    fake = FakeStructuredLLM(result, exc)
    monkeypatch.setattr("autods.llm.get_llm", lambda model: _FakeLLM(fake))
    return fake


def ok_result(parsed: BaseModel, content: str = "") -> dict:
    """A successful structured-output response carrying USAGE."""
    return {
        "raw": AIMessage(content=content, usage_metadata=USAGE),
        "parsed": parsed,
        "parsing_error": None,
    }


def make_state(**overrides: Any) -> AgentState:
    """Build a complete AgentState; pass keyword overrides for the keys under test."""
    state = AgentState(
        dataset_path="",
        user_description="",
        target_column=None,
        data_profile=None,
        task_decision=None,
        fe_code=None,
        fe_code_history=[],
        fe_explanations=[],
        tracebacks=[],
        processed_data_path=None,
        n_retries=0,
        leaderboard=[],
        best_model_id=None,
        report=None,
        warnings=[],
        token_cost_usd=0.0,
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state
