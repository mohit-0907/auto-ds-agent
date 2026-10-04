"""Shared test builders."""

from typing import Any

from autods.state import AgentState


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
