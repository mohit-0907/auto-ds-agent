# auto-ds-agent

An agent that ingests a tabular dataset, infers the ML task, writes and self-repairs
feature engineering code in a sandbox, benchmarks candidate models, and recommends the
best pipeline with a written justification.

**Core principle: the LLM makes judgments, Python does the computation.** Every metric,
statistic, and training run is deterministic. The LLM is confined to three nodes —
task routing, feature-engineering codegen, and the final report.

Status: **Phase 5 of 8 — the agent runs end to end from a CSV to a ranked model
leaderboard** (profiling, task routing, self-repairing feature engineering, fallback,
cross-validated model arena). The written report and CLI (Phase 6) are next.

See `PROJECT_PLAN.md` for the full architecture and phase breakdown.

## Architecture

![Agent graph](docs/graph.png)

Dashed arrows are decisions made in code by reading the shared state:

- **sanity_check → end** when there is no supported task (profiling failed, or the task is
  out of scope, e.g. time-series forecasting).
- **fe_codegen → sandbox → fe_codegen** is the corrective loop: the LLM writes feature
  engineering code, a sandboxed subprocess runs and validates it, and on failure the error
  goes back to the LLM. At most 3 attempts, enforced by the graph, not the prompt.
- **→ fallback_fe** after 3 failures, or if the LLM cannot be reached: a deterministic
  pandas recipe, so the run degrades instead of failing.
- **→ arena** once features exist: six models (a dummy baseline, a linear model, random
  forest, HistGradientBoosting, XGBoost, LightGBM) are cross-validated on the same five
  folds. The best mean wins only if it beats every simpler model on at least 4 of 5 folds;
  otherwise the simpler model wins. Given the same features, the leaderboard is identical
  on every run. (The generated feature code itself can vary between runs: current LLMs do
  not accept temperature 0.)

The diagram is generated from the code (`uv run python scripts/render_graph.py`); a test
fails if `docs/graph.mmd` no longer matches the graph.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src/ --fix
```

## Out of scope (v1)

Deep learning · NLP · computer vision · time series forecasting · multi-table joins ·
distributed training · exhaustive hyperparameter optimization.
