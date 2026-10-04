# auto-ds-agent

An agent that ingests a tabular dataset, infers the ML task, writes and self-repairs
feature engineering code in a sandbox, benchmarks candidate models, and recommends the
best pipeline with a written justification.

**Core principle: the LLM makes judgments, Python does the computation.** Every metric,
statistic, and training run is deterministic. The LLM is confined to three nodes —
task routing, feature-engineering codegen, and the final report.

Status: **Phase 0 — contracts & scaffolding.** Not yet runnable.

See `PROJECT_PLAN.md` for the full architecture and phase breakdown.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src/ --fix
```

## Out of scope (v1)

Deep learning · NLP · computer vision · time series forecasting · multi-table joins ·
distributed training · exhaustive hyperparameter optimization.
