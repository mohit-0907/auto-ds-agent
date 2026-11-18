# auto-ds-agent

An agent that ingests a tabular dataset, infers the ML task, writes and self-repairs
feature engineering code in a sandbox, benchmarks candidate models, and recommends the
best pipeline with a written justification.

**Core principle: the LLM makes judgments, Python does the computation.** Every metric,
statistic, and training run is deterministic. The LLM is confined to three nodes —
task routing, feature-engineering codegen, and the final report.

Status: **Phase 8 of 8 (hardening)** — a CLI and a web app (FastAPI + React, live progress
over Server-Sent Events) take a CSV to a number-verified report, a leaderboard and a saved
model. CI runs both test suites on every push.

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

## Evaluation

Live runs against the real LLMs (`uv run pytest tests/eval_<name>.py -s`; not part of the
test suite or CI, since they cost money and need the network).

| What | How | Target | Result |
|---|---|---|---|
| Router picks task type and target | 10 labelled datasets, 3 repeats (`eval_router`) | ≥ 80% | **10/10**, all 3 repeats |
| Generated FE passes on the first attempt | 10 datasets, targets given (`eval_fe`) | ≥ 60% | **10/10** |
| Generated FE passes within 3 attempts | same run | ≥ 90% | **10/10** |
| Chosen model clearly beats the dummy baseline | 5 datasets, full pipeline (`eval_arena`) | 5/5 | **5/5** |
| Same features → identical leaderboard | same 5, arena re-run on the saved features | 5/5 | **5/5** |
| LLM cost per full run | `token_cost_usd` of 14 real CLI and web runs | report it | **$0.023** mean ($0.019–$0.027) |

How to read these numbers, including the parts that flatter the agent:

- **The FE result is a ceiling, not a rate.** 10/10 on the first attempt means this eval set
  is too easy to exercise the corrective loop. These are well-known public datasets
  (Titanic, iris, diamonds…), probably familiar to the model, and it is a single pass:
  generated code differs between runs. The loop is shown working by `eval_graph`, which
  injects faults: a dropped row was repaired on attempt 2; a 0.01 s timeout failed all 3
  attempts and the fallback recipe took over. One natural failure was repaired on Titanic
  in development (attempt 1 wrote into a read-only pandas 3 array; attempt 2 passed).
- **The router descriptions are explicit** ("predict the sale price…"), and the sanity check
  never had to correct a decision in the eval; its value is shown by unit tests. A harder
  set with vague descriptions and ambiguous targets is the obvious next eval.
- **A leakage trap did not fire**: a text copy of the target was planted in a dataset, and
  the model excluded it itself in 3 of 3 runs, so the validator's leakage check (tested
  offline) was never needed live.
- **A good score is not always a good model.** On `healthexp` (life expectancy by country and
  year) the arena reports an RMSE of 0.32 years, because random K-fold lets a model fill in
  a gap between a country's neighbouring years. Forecasting future years would be far
  less accurate. The report flagged this as a caveat; the agent cannot detect it.

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
uv run pytest                      # ~3 min; every LLM call is scripted, no API key needed
uv run pytest --cov                # with coverage (fails below 60%)
uv run ruff check src/ tests/ --fix
cd web && npm run lint && npm run test && npm run build
```

CI (`.github/workflows/ci.yml`) runs the same checks on every push: a Python job (ruff and
pytest with coverage) and a web job (lint, test, build), with no secrets.

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
