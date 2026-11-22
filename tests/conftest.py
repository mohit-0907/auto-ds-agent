"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _api_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the "is the API key set?" check in every test, whatever the environment.

    Tests script every LLM answer (tests/helpers.py), so they need no key, and CI has
    none. The check itself is replaced, not the environment: a fake OPENAI_API_KEY would
    let a test that forgot to script the LLM make a real network call (rejected with
    401, then quietly degraded), where today it fails loudly. Tests of the check
    re-patch it, or call the function they imported before this fixture ran.
    """
    monkeypatch.setattr("autods.llm.missing_api_keys", lambda models=(): [])
