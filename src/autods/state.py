"""AgentState and every Pydantic model passed between graph nodes.

State is the only channel between nodes: no globals, no side-channels.

Every model here is produced either by deterministic Python (``DataProfile``,
``ModelResult``) or by a single structured LLM call (``TaskDecision``).
``TaskDecision`` is the only model whose ``Field(description=...)`` text reaches
a model as JSON schema; elsewhere the descriptions are documentation.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class ColumnProfile(BaseModel):
    """Deterministic summary of one column, sized to fit inside a prompt.

    Produced by the profiler node. The numeric statistics are ``None`` for
    non-numeric columns. ``sample_values`` is typed ``list[str]`` deliberately:
    it forces the profiler to stringify pandas/numpy scalars at construction
    time, which is what keeps the whole profile JSON-serializable.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Name of the column.")
    dtype: str = Field(description="Pandas dtype of the column, as a string.")
    is_categorical: bool = Field(
        description="Whether the profiler treats this column as categorical."
    )
    null_fraction: float = Field(
        ge=0, le=1, description="Fraction of values in the column that are null."
    )
    n_unique: int = Field(ge=0, description="Number of distinct non-null values.")
    cardinality_ratio: float = Field(
        ge=0, le=1, description="n_unique divided by the dataset row count."
    )
    sample_values: list[str] = Field(
        description="Up to three example values from the column, stringified."
    )
    min: float | None = Field(default=None, description="Minimum value; None if non-numeric.")
    max: float | None = Field(default=None, description="Maximum value; None if non-numeric.")
    mean: float | None = Field(default=None, description="Mean value; None if non-numeric.")
    std: float | None = Field(default=None, description="Standard deviation; None if non-numeric.")
    skew: float | None = Field(default=None, description="Skewness; None if non-numeric.")


class DataProfile(BaseModel):
    """Dataset-level facts plus per-column profiles. Output of the profiler node.

    Serialized into the router prompt, so it must stay under the Phase 1 token
    budget. When it does not fit, the profiler drops per-column detail and sets
    ``columns_truncated`` so the router knows it is seeing a partial view.
    """

    model_config = ConfigDict(frozen=True)

    n_rows: int = Field(ge=0, description="Number of rows in the dataset.")
    n_cols: int = Field(ge=0, description="Number of columns in the dataset.")
    size_mb: float = Field(ge=0, description="In-memory size of the dataset in megabytes.")
    n_duplicate_rows: int = Field(ge=0, description="Number of fully duplicated rows.")
    columns: list[ColumnProfile] = Field(description="Per-column profiles.")
    target_candidates: list[str] = Field(
        default_factory=list,
        description="Column names that plausibly hold the target, for the router to choose from.",
    )
    columns_truncated: bool = Field(
        default=False,
        description="True if per-column detail was dropped to fit the prompt token budget.",
    )
    constant_columns: list[str] = Field(
        default_factory=list,
        description="Columns with at most one distinct non-null value.",
    )
    dtype_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Column count per dtype, covering all columns. Set when columns_truncated.",
    )


class TaskDecision(BaseModel):
    """The router's judgment about which ML problem this dataset represents.

    The only LLM-authored model in the project, produced via
    ``.with_structured_output(TaskDecision)``. Mutable, so the deterministic
    sanity check that follows the LLM call can correct ``task_type`` in place.
    """

    task_type: Literal[
        "binary_classification", "multiclass_classification", "regression", "unsupported"
    ] = Field(
        description=(
            "The supervised learning task this dataset supports. Use "
            "'binary_classification' when the target has exactly two distinct values, "
            "'multiclass_classification' for three or more discrete classes, "
            "'regression' for a continuous numeric target, and 'unsupported' when no "
            "usable target exists or the problem is out of scope (time series "
            "forecasting, free text, image paths, or unlabeled data)."
        )
    )
    target_column: str = Field(
        description="Name of the column to predict. Must appear in the profile."
    )
    reasoning: str = Field(
        description="Two or three sentences justifying the task type and target choice."
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence in this decision, where 1.0 is certain."
    )
    unsupported_reason: str | None = Field(
        default=None,
        description="Why the dataset is unsupported. Required when task_type is "
        "'unsupported', otherwise null.",
    )


class ModelResult(BaseModel):
    """One candidate model's cross-validated performance in the arena.

    Never LLM-authored -- every number here comes from sklearn. A model that
    fails to fit is recorded with ``status='failed'`` and an ``error`` rather
    than being dropped, so the report can surface what was attempted.
    """

    model_id: str = Field(description="Stable slug identifying the candidate, e.g. 'hgb'.")
    model_name: str = Field(description="Human-readable estimator name.")
    status: Literal["success", "failed"] = Field(description="Whether the model fit and scored.")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Mean cross-validated score per metric name."
    )
    metrics_std: dict[str, float] = Field(
        default_factory=dict, description="Standard deviation across folds, same keys as metrics."
    )
    primary_metric: str = Field(
        description="Key in metrics that the leaderboard ranks on. Determined by the "
        "task type before any model is fitted, so it is set even when status is 'failed'."
    )
    training_time_s: float = Field(ge=0, default=0.0, description="Total fit time in seconds.")
    error: str | None = Field(default=None, description="Failure message when status is 'failed'.")


class AgentState(TypedDict):
    """The single channel between graph nodes.

    Every key is always present; optional values are None rather than absent.
    Paths cross node boundaries, never DataFrames -- frames are not serializable
    and would make each checkpoint cost hundreds of megabytes.

    The four ``Annotated[..., operator.add]`` keys are written by more than one
    node (or more than once by the same node) and accumulate. Everything else is
    last-write-wins, which is what keeps a replayed or resumed run idempotent.

    ``token_cost_usd`` accumulates so that each LLM node returns only the cost of
    its own calls; no node can reset the running total by forgetting to add to it.
    """

    # Inputs
    dataset_path: str
    user_description: str
    target_column: str | None

    # Stage outputs
    data_profile: DataProfile | None
    task_decision: TaskDecision | None
    fe_code: str | None
    fe_code_history: Annotated[list[str], operator.add]
    tracebacks: Annotated[list[str], operator.add]
    processed_data_path: str | None
    n_retries: int
    leaderboard: list[ModelResult]
    best_model_id: str | None
    report: str | None

    # Bookkeeping
    warnings: Annotated[list[str], operator.add]
    token_cost_usd: Annotated[float, operator.add]
