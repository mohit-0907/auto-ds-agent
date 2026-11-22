# auto-ds-agent

An agent that ingests a tabular dataset, infers the ML task, writes and self-repairs
feature engineering code in a sandbox, benchmarks candidate models, and recommends the
best pipeline with a written justification.

**Core principle: the LLM makes judgments, Python does the computation.** Every metric,
statistic, and training run is deterministic. The LLM is confined to three nodes —
task routing, feature-engineering codegen, and the final report.

Status: **complete (v1)**: a CLI and a web app (FastAPI + React, live progress over
Server-Sent Events) take a CSV to a number-verified report, a leaderboard and a saved
model; a Docker image runs it all. CI runs the test suites and builds the image on every
push. `PROJECT_PLAN.md` has the full phase breakdown.

<!-- Demo GIF: docs/demo.gif, recorded with scripts/demo_serve.py (see "Web app"). -->

## Quick start

With Docker only (nothing else to install):

```bash
git clone <this repository> && cd auto-ds-agent
cp .env.example .env                  # then set OPENAI_API_KEY=sk-... (no quotes)
docker build -t autods .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env -v autods-runs:/app/runs autods
```

Open http://127.0.0.1:8000, pick a sample dataset, say what to predict, and watch the run.
A run takes about a minute and costs about $0.02 in LLM calls. Without Docker, see
[Web app](#web-app) (uv and Node.js) or [Usage](#usage) (command line only).

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

## Why the LLM only judges

An LLM is good at reading a description and a data profile and making a call: *which*
column is the target, *what* features make sense for this table, *how* to explain a
result. It is bad at being the source of a number: it produces plausible figures that
nobody can check, and different ones on every run. So the work is split:

| The LLM decides (3 nodes) | Python computes and checks |
|---|---|
| **Router:** the task type and target column | **Sanity check:** the target's real statistics can overrule it (e.g. "binary" on 600 distinct floats becomes regression) |
| **Feature codegen:** pandas code for this dataset | **Sandbox + validator:** the code runs in a subprocess and is accepted only if the row count, target, column types and leakage checks pass |
| **Reporter:** five prose sections | **Facts sheet + number check:** every metric, lift and table is computed in Python; every number in the prose must match a fact (rounding allowed), or the reporter gets one retry naming the wrong numbers, then the report carries a visible warning |

Every LLM answer arrives as a typed Pydantic object (structured output), never parsed out
of free text. What this buys, measured:

- **Reproducible results.** Given the same features, the leaderboard is identical on
  every run, on Windows and in the Linux container alike (Titanic: HistGradientBoosting,
  ROC-AUC 0.884, both).
- **No invented numbers.** A metric in the report either matches the arena's own
  calculation or is flagged.
- **Cheap and bounded.** Usually three calls per run; more only when a feature attempt
  fails (at most 3 attempts, enforced by the graph) or the report's number check asks for
  its one retry. $0.023 per full run on average.
- **It degrades instead of failing.** If the feature-code LLM fails, a fixed pandas recipe
  makes the features; if the reporter fails, the report keeps its tables with a note; if
  the router fails, the run ends with a report saying why. None of these crash the run.

## Evaluation

Live runs against the real LLMs (`uv run pytest tests/eval_<name>.py -s`; not part of the
test suite or CI, since they cost money and need the network). The rows come from three
separate evals, named in the "How" column: the router and feature evals stop before any
model is trained, and the model rows come from the full-pipeline arena eval.

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
recommended, `2` bad command line or the API key is not set, `3` the run ended without a
model (the report says why).

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

**Demo mode.** Generated feature code usually passes on the first attempt, so an ordinary
run rarely shows the repair loop. `uv run python scripts/demo_serve.py` starts the same app
but breaks attempt 1 of every run on purpose: it appends a wrapper that drops one row, the
sandbox rejects the result ("Row count changed"), and the LLM reads that error and writes
attempt 2 itself. The injected code is visible in attempt 1's step, so a recording made
this way shows openly that the first failure was staged.

## Docker

Only Docker is needed: no Python, uv or Node.js on the host.

```bash
docker build -t autods .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env -v autods-runs:/app/runs autods
# then open http://127.0.0.1:8000
```

- **`--env-file .env`** passes the API key when the container starts; the image never
  contains it (`.dockerignore` keeps `.env` out of the build). Docker reads the file
  literally, so write `OPENAI_API_KEY=sk-...` without quotes. Or pass it from the shell:
  `-e OPENAI_API_KEY`. Forget it, and the app still opens but refuses to start a run,
  saying which key is missing.
- **`-p 127.0.0.1:8000:8000`** publishes the app on this machine only. The app has no
  login, and plain `-p 8000:8000` would let anyone on the network start runs on your key.
  The container's log says `http://0.0.0.0:8000`: that is the address inside the
  container, not one to open.
- **`-v autods-runs:/app/runs`** keeps the run history in a named volume, so it survives
  the container being removed. Without it, every run is lost with the container.

The image (about 1.3 GB, mostly scikit-learn, XGBoost, LightGBM and pandas) is built in
two stages: Node builds the frontend, and only the built files are copied into a slim
Python image, which runs as an ordinary user that can write to `runs/` and nowhere else.
CI builds it and checks that it starts on every push.

## Development

```bash
uv sync
uv run pytest                      # ~3 min; every LLM call is scripted, no API key needed
uv run pytest --cov                # with coverage (fails below 60%)
uv run ruff check src/ tests/ --fix
cd web && npm run lint && npm run test && npm run build
```

CI (`.github/workflows/ci.yml`) runs the same checks on every push, with no secrets: a
Python job (ruff and pytest with coverage), a web job (lint, test, build), and a Docker
job that builds the image and checks that it starts and serves the app.

## Known limitations

- **The sandbox is isolation, not security.** Generated code runs in a separate process with
  a timeout, a stripped environment (no API keys), an import allowlist and, on Linux/macOS, a
  memory cap. Run directly, it shares the filesystem and network with the host. The Docker
  image keeps generated code away from the host's files, but inside the container the code
  still runs as the same user as the web server, whose environment holds the API key: the
  stripped environment stops accidents, not code written to go looking for the key (a
  separate sandbox user would). The import check likewise stops accidental harm, not a
  determined adversary.
- **Generated feature code is batch-stateless.** It computes medians and categories from
  whatever frame it receives, so `best_pipeline.joblib` is for re-scoring batches (run
  `fe_code.py`, then align to the saved `feature_columns`), not single-row serving.
- **Only load `best_pipeline.joblib` files you created.** Like any pickle-based format,
  loading one executes code.
- **Feature code can differ between runs.** Current LLMs do not accept temperature 0. Given
  the same features, the model arena's results are identical on every run.
- **Cross-validation assumes independent rows.** The arena uses random (stratified)
  K-fold. On grouped or time-ordered data, such as countries measured year after year, a
  model is scored on years between ones it trained on, so the score is optimistic for
  forecasting (see `healthexp` under Evaluation). Grouped or time-ordered folds would fix
  this; the agent does not detect such data.
- **The selection rule tests only the top model.** The best mean must beat every simpler
  model on at least 4 of 5 folds, or the simpler one wins. Models ranked below the top are
  never tested against the winner, so a lower-ranked model can sometimes clearly beat the
  chosen one; the Results tab marks such comparisons "untested".
- **The agent sees a profile, not how the data was collected.** In the NYC taxi data, cash
  tips are recorded as $0, so "predict the tip" partly learns the payment method. Nothing
  in the column statistics says so, and the agent cannot know it.
- **Single user, local use.** One run at a time, no login. Keep it on your own machine
  (the Docker command publishes it on 127.0.0.1 only).
- **The narrow-screen drawer does not move keyboard focus** into the run list when it
  opens.

## Out of scope (v1)

Deep learning · NLP · computer vision · time series forecasting · multi-table joins ·
distributed training · exhaustive hyperparameter optimization.
