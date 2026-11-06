# auto-ds-agent

An agent that ingests a tabular dataset, infers the ML task, writes and self-repairs
feature engineering code in a sandbox, benchmarks candidate models, and recommends the
best pipeline with a written justification.

**Core principle: the LLM makes judgments, Python does the computation.** Every metric,
statistic, and training run is deterministic. The LLM is confined to three nodes —
task routing, feature-engineering codegen, and the final report.

Status: **Phase 6 of 8 — one command takes a CSV to a written, number-verified report**
(profiling, task routing, self-repairing feature engineering, fallback, cross-validated
model arena, report, saved model). A web app (FastAPI + React, Phase 7) is next.

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

## Usage

```bash
uv sync
cp .env.example .env          # then set OPENAI_API_KEY
uv run autods run datasets/Titanic-Dataset.csv --description "Predict who survived"
```

**Example output:** [`docs/example_run/report.md`](docs/example_run/report.md), produced by
one command on a dataset the agent had never seen (car fuel efficiency, no target given),
with the generated [`fe_code.py`](docs/example_run/fe_code.py) and
[`leaderboard.csv`](docs/example_run/leaderboard.csv).

Each run gets its own folder under `runs/` with `report.md`, `leaderboard.csv`,
`best_pipeline.joblib`, `fe_code.py` and `run.json`. `--target` names the column to predict
(otherwise it is inferred); `--verbose` shows detailed logs. Exit codes: `0` a model was
recommended, `2` bad command line, `3` the run ended without a model (the report says why).

## Web app

The same agent with a browser interface: upload a table, watch every stage live
(including the feature-code retries), then read the leaderboard and report and download
the artifacts.

```bash
uv sync
cp .env.example .env                       # then set OPENAI_API_KEY
cd web && npm install && npm run build && cd ..   # once; needs Node.js 22.12+
uv run autods serve                        # then open http://127.0.0.1:8000
```

1. **New run:** drop a CSV or Parquet file (or pick a sample dataset), say in words what
   to predict, and optionally name the target column. Click **Start run**.
2. **Steps:** each stage appears as it finishes. Click a step to see what it produced:
   the router's decision, the generated feature code, a sandbox error and the retry.
3. **Results:** the recommended model, how far it beats the baseline, the fold-by-fold
   evidence for choosing it, the full leaderboard, and the files to download.
4. **Report** and **Feature code** tabs hold the written report and the code the model's
   features came from. **All runs** lists every past run, from the web app or the CLI.

One run at a time: a second upload while a run is in progress is refused. For frontend
development, run `npm run dev` in `web/` instead of building; it proxies `/api` to
`autods serve` on port 8000.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src/ --fix
```

## Known limitations

- **The sandbox is isolation, not security.** Generated code runs in a separate process with
  a timeout, a stripped environment (no API keys), an import allowlist and, on Linux/macOS, a
  memory cap. It still shares the filesystem and network with the host; real security needs a
  container. The import check stops accidental harm, not a determined adversary.
- **Generated feature code is batch-stateless.** It computes medians and categories from
  whatever frame it receives, so `best_pipeline.joblib` is for re-scoring batches (run
  `fe_code.py`, then align to the saved `feature_columns`), not single-row serving.
- **Only load `best_pipeline.joblib` files you created.** Like any pickle-based format,
  loading one executes code.
- **Feature code can differ between runs.** Current LLMs do not accept temperature 0. Given
  the same features, the model arena's results are identical on every run.

## Out of scope (v1)

Deep learning · NLP · computer vision · time series forecasting · multi-table joins ·
distributed training · exhaustive hyperparameter optimization.
