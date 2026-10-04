"""The single place chat models are constructed and prompts are loaded.

Nodes call get_llm(config.ROUTER_MODEL) and never import a provider class, so
the provider is a config value. Provider packages are imported lazily: only the
one in use needs to be installed, and a missing one fails with an install hint
instead of an ImportError at module import time.
"""

from __future__ import annotations

from functools import cache
from string import Template
from typing import TYPE_CHECKING

from autods.config import LLM_MAX_RETRIES, LLM_MAX_TOKENS, LLM_TIMEOUT_S, PROMPTS_DIR

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

SUPPORTED_PROVIDERS = ("openai", "anthropic")


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
