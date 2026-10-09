"""The single place chat models are constructed, called, and prompts loaded.

Nodes call invoke_structured(config.ROUTER_MODEL, Schema, messages) and never
import a provider class, so the provider is a config value. Provider packages
are imported lazily: only the one in use needs to be installed, and a missing
one fails with an install hint instead of an ImportError at module import time.
"""

from __future__ import annotations

from functools import cache
from string import Template
from typing import TYPE_CHECKING, Generic, TypeVar

import structlog
from pydantic import BaseModel

from autods.config import (
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_S,
    PROMPTS_DIR,
    estimate_cost_usd,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

SUPPORTED_PROVIDERS = ("openai", "anthropic")

log = structlog.get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredCall(BaseModel, Generic[SchemaT]):
    """Outcome of one structured LLM call.

    cost_usd is set whenever tokens were billed -- including when the output
    then failed to parse. A failed call that reported $0 would make failing
    runs look cheaper than they were.
    """

    parsed: SchemaT | None = None
    error: str | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@cache
def _read_prompt(name: str) -> Template:
    return Template((PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8"))


def load_prompt(name: str, **values: str) -> str:
    """Render a prompt file from src/autods/prompts/ (invariant 7).

    Uses string.Template ($name placeholders) rather than str.format, because
    the values include profile JSON: every brace in it would be read as a
    format field.

    Args:
        name: File stem, e.g. "router_system" for prompts/router_system.md.
        **values: Substitutions for the template's $placeholders.

    Returns:
        The rendered prompt.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        KeyError: If the template uses a placeholder not given in values.
    """
    return _read_prompt(name).substitute(values)


def parse_model(model: str) -> tuple[str, str]:
    """Split a "<provider>:<model-id>" string.

    Args:
        model: A model string such as "openai:gpt-6.1-sol".

    Returns:
        (provider, model_id).

    Raises:
        ValueError: If the string has no provider prefix, an empty part, or an
            unsupported provider.
    """
    provider, sep, model_id = model.partition(":")
    if not sep or not provider or not model_id:
        raise ValueError(
            f"model must look like '<provider>:<model-id>', e.g. 'openai:gpt-6.1-sol'; "
            f"got {model!r}"
        )
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported provider {provider!r}; expected one of {SUPPORTED_PROVIDERS}"
        )
    return provider, model_id


def get_llm(model: str) -> BaseChatModel:
    """Build a chat model for the given model string.

    No temperature is passed: current models from both providers reject any
    non-default value (see the note in config.py).

    Args:
        model: A "<provider>:<model-id>" string, normally from config.

    Returns:
        A LangChain chat model, ready for .with_structured_output().

    Raises:
        ValueError: If the model string is malformed or the provider unknown.
        ImportError: If the provider's LangChain package is not installed.
    """
    provider, model_id = parse_model(model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_id,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT_S,
            max_retries=LLM_MAX_RETRIES,
        )

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "provider 'anthropic' needs langchain-anthropic: run `uv add langchain-anthropic`"
        ) from exc
    return ChatAnthropic(
        model=model_id,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
    )


def invoke_structured(
    model: str, schema: type[SchemaT], messages: list[tuple[str, str]]
) -> StructuredCall[SchemaT]:
    """Call a model with structured output (invariant 2) and account for its cost.

    API and parse failures come back as StructuredCall.error rather than
    raising, so every LLM node degrades the same way (invariant 6). Cost is
    computed here, in Python, from the provider's reported usage (invariant 1).

    Args:
        model: A "<provider>:<model-id>" string from config.
        schema: The Pydantic model the output must parse into.
        messages: (role, content) chat messages.

    Returns:
        The parsed output or an error, with token counts and cost.

    Raises:
        ValueError, ImportError: From get_llm, for a misconfigured model. These
            are deliberately not caught: a config bug should fail loudly, not
            read as "the LLM had a bad day".
    """
    llm = get_llm(model).with_structured_output(schema, include_raw=True)

    try:
        result = llm.invoke(messages)
    # Broad on purpose, and scoped to the one line that talks to the network:
    # each provider SDK raises its own exception hierarchy, and none of them
    # should crash the run.
    except Exception as exc:
        error = f"LLM call raised {type(exc).__name__}: {exc}"
        log.warning("llm.call_failed", model=model, error=error)
        return StructuredCall(error=error)

    usage = getattr(result["raw"], "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    billed = {
        "cost_usd": estimate_cost_usd(model, input_tokens, output_tokens),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    if result["parsing_error"] is not None or result["parsed"] is None:
        error = f"could not parse LLM output: {result['parsing_error']}"
        log.warning("llm.parse_failed", model=model, error=error)
        return StructuredCall(error=error, **billed)
    return StructuredCall(parsed=result["parsed"], **billed)
