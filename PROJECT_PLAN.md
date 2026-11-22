# Autonomous Data Science Agent — Project Plan

**Repo name suggestion:** `auto-ds-agent`
**One-liner:** An agent that ingests a tabular dataset, infers the ML task, writes and self-repairs feature engineering code, benchmarks candidate models, and recommends the best pipeline with a written justification.

---

## 1. Core design principle

> **The LLM makes judgments. Python does the computation.**

This is the single decision that separates this project from the hundreds of "I wrapped AutoML in a chatbot" repos. Be able to defend it in an interview.

| Stage | Who does it | Why |
|---|---|---|
| Data profiling | Deterministic pandas | Stats must be exact, not hallucinated |
| Task type decision | LLM (structured output) | Genuine judgment: needs data + user intent |
| Feature engineering plan + code | LLM (codegen) | Open-ended, context-dependent |
| Code execution | Sandboxed subprocess | LLM output is untrusted input |
| Error repair | LLM (sees traceback) | This is the "corrective loop" |
| Model training & CV | Deterministic sklearn | Reproducibility, no hallucinated metrics |
| Final recommendation | LLM, over deterministic metrics | Narrative reasoning, not number generation |

If you ever find yourself asking an LLM to compute a mean or report an F1 score, you have made a mistake.

---

## 2. Architecture

```
CSV + user description (+ optional target column)
   │
   ▼
[1] PROFILER            deterministic  →  DataProfile
   │
   ▼
[2] TASK ROUTER         LLM            →  TaskDecision {task_type, target, confidence, reasoning}
   │
   ▼
[3] FE CODEGEN          LLM            →  Python source implementing a fixed contract
   │
   ▼
[4] SANDBOX EXECUTOR    subprocess     →  success | traceback
   │        │
   │        └──► [5] CORRECTIVE LOOP ──► back to [3] with traceback  (max 3 attempts)
   │                                     after 3 failures → fallback baseline FE
   ▼
[6] MODEL ARENA         deterministic  →  Leaderboard[ModelResult]
   │
   ▼
[7] REPORTER            LLM            →  report.md
   │
   ▼
Artifacts: leaderboard.csv · best_pipeline.joblib · fe_code.py · report.md · run.json
```

**Graph state** (single source of truth, defined in Phase 0):

```python
class AgentState(TypedDict):
    # inputs
    dataset_path: str
    user_description: str
    target_column: str | None
    # stage outputs
    profile: DataProfile | None
    task_decision: TaskDecision | None
    fe_code: str | None
    fe_attempts: int
    last_traceback: str | None
    processed_data_path: str | None
    leaderboard: list[ModelResult]
    best_model_id: str | None
    report: str | None
    # bookkeeping
    warnings: list[str]
    token_cost_usd: float
```

---

## 3. Scope

### In scope (v1)
- Tabular CSV / Parquet, single file, fits in memory (< ~200 MB)
- Task types: `binary_classification`, `multiclass_classification`, `regression`
- A refusal path: `unsupported` (time series, text-heavy, image paths, unlabeled)
- 5–6 candidate models with cross-validation, no heavy HPO
- CLI first, then a web app (FastAPI backend + React frontend) that streams the agent live

### Explicitly out of scope (v1) — write this in your README
- Deep learning, NLP, computer vision
- Time series forecasting
- Multi-table / SQL joins
- Distributed training, cloud deployment
- Exhaustive hyperparameter optimization

Scope discipline is a CV signal. An honest "here's what it doesn't do" section reads as seniority.

---

## 4. Tech stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph (`StateGraph`, conditional edges for the retry loop) |
| LLM interface | LangChain `ChatAnthropic` / `ChatOpenAI` with `.with_structured_output()` |
| Schemas | Pydantic v2 |
| ML | scikit-learn, xgboost, lightgbm |
| Data | pandas, pyarrow |
| Sandbox | `subprocess` + `resource` limits + timeout |
| Web backend | FastAPI, Server-Sent Events for live progress, uvicorn |
| Web frontend | React + TypeScript, built with Vite |
| Tests | pytest (Python); Vitest + React Testing Library (frontend) |
| Env | `uv` (or poetry), `.env` via python-dotenv |
| Logging | `structlog` or stdlib logging with JSON formatter |

---

## 5. Repo layout

```
auto-ds-agent/
├── CLAUDE.md
├── PROJECT_PLAN.md
├── README.md
├── pyproject.toml
├── .env.example
├── src/autods/
│   ├── state.py            # AgentState + all Pydantic models  [Phase 0]
│   ├── config.py           # settings, model names, limits      [Phase 0]
│   ├── graph.py            # LangGraph wiring                   [Phase 4]
│   ├── nodes/
│   │   ├── profiler.py     # [Phase 1]
│   │   ├── router.py       # [Phase 2]
│   │   ├── fe_codegen.py   # [Phase 3]
│   │   ├── arena.py        # [Phase 5]
│   │   └── reporter.py     # [Phase 6]
│   ├── sandbox/
│   │   ├── executor.py     # [Phase 3]
│   │   └── contract.py     # the function signature LLM code must satisfy
│   ├── prompts/            # .md or .txt files, NOT inline strings
│   ├── cli.py              # [Phase 6]
│   └── api/                # FastAPI app: thin HTTP layer over the graph  [Phase 7]
├── web/                    # React + TypeScript frontend (Vite)            [Phase 7]
├── tests/
├── datasets/               # small public CSVs for eval
└── runs/                   # gitignored artifacts
```

---

## 6. Phases

Each phase has a **Definition of Done**. Do not start phase N+1 until phase N's DoD is green. Ask Claude Code to hold you to this.

**Working mode (from Phase 1.6 on):** Claude Code implements each task and its tests, then explains the design and trade-offs; the developer makes the design decisions and must be able to defend every line in an interview. See `CLAUDE.md` → *Operating mode*.

### Phase 0 — Contracts & scaffolding · ~2h
Define every Pydantic model and the `AgentState` before writing any logic. Everything downstream is typed against these.

- `DataProfile`, `ColumnProfile`, `TaskDecision`, `ModelResult`, `AgentState`
- `config.py` with model names, `MAX_FE_ATTEMPTS=3`, `SANDBOX_TIMEOUT_S=120`, CV folds
- `pyproject.toml`, `.env.example`, `.gitignore`, empty module tree

**DoD:** `python -c "from autods.state import AgentState"` works; `pytest` collects zero tests without error.

---

### Phase 1 — Profiler · ~3h
Pure pandas, no LLM. Produces a compact JSON summary small enough to fit in a prompt.

Per column: dtype, null %, n_unique, cardinality ratio, sample values (3), and for numerics min/max/mean/std/skew. Dataset level: shape, memory, duplicate row count, constant columns.

Two critical details:
- **Token budget.** For a 300-column dataset, a naive profile blows your context. Cap it — full detail for ≤ 40 columns, otherwise summarize by dtype group and detail only the top-N by relevance.
- **Target inference.** If `target_column` is None, produce *candidates* (last column, columns named `target`/`label`/`y`, low-cardinality columns) and let the router decide.

**DoD:** profiles Titanic, Ames Housing, and a 200-column synthetic frame; JSON output under 4k tokens in all three cases; unit tests for null/constant/high-cardinality edge cases.

---

### Phase 2 — Task router · ~3h
First LLM node. Input: profile JSON + user description. Output: `TaskDecision` via `.with_structured_output(TaskDecision)`.

```python
class TaskDecision(BaseModel):
    task_type: Literal["binary_classification", "multiclass_classification",
                       "regression", "unsupported"]
    target_column: str
    reasoning: str
    confidence: float = Field(ge=0, le=1)
    unsupported_reason: str | None = None
```

Add a **deterministic sanity check** after the LLM: if it says regression but the target has 4 unique integer values, override or flag. LLM proposes, code verifies. This is a strong talking point.

Build a tiny eval: `tests/eval_router.py` with 8–10 public datasets and expected labels. Report accuracy. Include at least two adversarial cases (a regression target stored as strings; a 5-class target that looks numeric).

**DoD:** ≥ 8/10 on the eval set; the sanity check catches at least one deliberate failure.

---

### Phase 3 — FE codegen + sandbox · ~6h · *the hard one*

**The contract.** The LLM writes exactly one function, always this signature:

```python
def engineer_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Returns a frame with the target intact and all other columns numeric."""
```

Constraining the shape of generated code is what makes repair tractable. Free-form scripts are unrepairable.

**The sandbox.** Non-negotiable:
- `subprocess.run([sys.executable, "-c", ...], timeout=SANDBOX_TIMEOUT_S)`
- Run on a **sample first** (1000 rows) to fail fast, then the full frame
- Capture stdout, stderr, and full traceback separately
- Post-execution validation: target still present, no NaNs, all non-target columns numeric, row count sane, no target leakage (a new column correlating > 0.99 with the target is a red flag)

Note the honest limitation for your README: a subprocess is isolation, not security. Real security means a container. Say so rather than overclaiming.

**DoD:** valid code runs and passes validation; deliberately broken code produces a clean structured error object, not a crash; an infinite loop is killed by the timeout.

---

### Phase 4 — LangGraph wiring + corrective loop · ~3h
Now assemble the graph. The conditional edge is the centerpiece:

```
fe_codegen ──► sandbox ──► [route_after_execution]
                              ├─ ok                      → arena
                              ├─ error & attempts < 3    → fe_codegen (with traceback)
                              └─ error & attempts >= 3   → fallback_fe → arena
```

The fallback path matters more than it looks: an agent that degrades gracefully instead of dying beats one that only works on the happy path. The fallback is a plain sklearn `ColumnTransformer` (median/mode impute, one-hot ≤ 20 categories, standard-scale numerics).

When retrying, feed back the failed code *and* the traceback, and instruct the model to explain the fix before rewriting. Keep prior attempts in state so it doesn't loop on the same mistake.

**DoD:** graph compiles and renders (`get_graph().draw_mermaid_png()` — put it in the README); a dataset engineered to break FE on attempt 1 succeeds on attempt 2; a genuinely impossible case exits via fallback.

---

### Phase 5 — Model arena · ~4h
Fully deterministic. No LLM.

Candidates by task type — always include a **dummy baseline**:

- Classification: `DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, `XGBClassifier`, `LGBMClassifier`
- Regression: `DummyRegressor`, `Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, `XGBRegressor`

Metrics: classification → ROC-AUC (binary) / macro-F1 (multi), accuracy, precision, recall; regression → RMSE, MAE, R². Cross-validated with `StratifiedKFold` / `KFold`, report mean ± std, per-model fit time.

Two things people forget: a model that fails to fit should be recorded as `status="failed"` and not kill the run; and "best" should incorporate the std — a 0.001 improvement with 3× the variance is not an improvement.

Small, defensible bonus: 20-trial Optuna search on the top-2 models only.

**DoD:** full leaderboard on Titanic and Ames; best model beats the dummy baseline; results reproducible across runs with a fixed seed.

---

### Phase 6 — Reporter, artifacts, CLI · ~3h
LLM writes the narrative; every number in it comes from `state["leaderboard"]`, passed in as text. Report sections: dataset summary, task decision + why, FE applied (including retries — surface the failures, they're interesting), leaderboard table, recommendation with trade-offs (accuracy vs. inference cost vs. interpretability), caveats and next steps.

Persist `runs/<timestamp>/` with the leaderboard, fitted best pipeline, generated FE code, report, and a `run.json` of decisions and token cost.

CLI: `autods run data.csv --description "..." --target price`

**DoD:** one command, cold start to `report.md`, on a dataset the system has never seen.

---

### Phase 7 — Web app: FastAPI backend + React frontend · ~22h (incl. redesign)
*Changed from Streamlit on 2026-10-17: a full-stack app (agent + REST API + live streaming + React UI) is a stronger project than a Streamlit wrapper. Cost: ~4× the time, a second toolchain (Node), and a more involved Phase 8.*

Upload → description → live stage progress → leaderboard → report → downloads, as a React app talking to a FastAPI backend.

```
Browser (React, web/)                    Python (FastAPI, src/autods/api/)        Agent
New-run form --POST /api/runs---------->  save upload, new_run_dir, start graph -> build_graph().stream()
              (file, description, target)  in a background thread                  in a worker thread
Live view   <-GET /api/runs/{id}/events--  one Server-Sent Event per finished node  <- node updates
Results     --GET /api/runs/{id}-------->  leaderboard, best model, report.md
Downloads   --GET /api/runs/{id}/files/{name}--> report.md, fe_code.py, best_pipeline.joblib, ...
History     --GET /api/runs------------->  list of past run folders
```

Rules:
- **The web layer is a thin caller of the graph**, exactly like the CLI: no agent logic in API routes, and the stage descriptions are shared with the CLI (one module, not two copies).
- **The browser never talks to an LLM and never computes a metric.** Everything it shows comes from state or run artifacts.
- **Server-Sent Events, not WebSockets:** progress only flows server → browser, SSE is plain HTTP with built-in reconnect, and it is easy to test.
- **One process in production:** `uv run autods serve` serves the API and the built React app together. In development, Vite's dev server proxies `/api` to FastAPI.
- **Uploads are validated at the API boundary** (extension, size) as well as by the profiler.
- **One run at a time per server** (a second submission gets `409 Conflict`): v1 is a single-user tool, stated in the README's limitations.

Watching the corrective loop retry live (attempt 1 ✗ → attempt 2 ✓) is the demo moment — make sure it is visible.

Tasks:
- **7.1** API skeleton: FastAPI app, upload + start a run in the background, run status and run list, shared progress module; API tests with the scripted LLM.
- **7.2** Live events: the SSE endpoint fed by `graph.stream()`; tests.
- **7.3** Frontend scaffold: Vite + React + TypeScript in `web/`, a typed API client, the new-run page (file drop, description, optional target).
- **7.4** Live run view: consume the event stream; a stage timeline that shows retries.
- **7.5** Results view: leaderboard table, rendered report, downloads, run history.

*Redesign added on 2026-10-30 from `docs/design_handoff_auto_ds_agent_ui/` (high-fidelity, "Organic" design system: app shell with a run sidebar, light/dark theme, pipeline graph, tabs). Decisions: lift and fold verdicts are computed by the server, not the browser (invariant 9); fonts are bundled via @fontsource; a real samples endpoint lets a run start without an upload.*
- **7.6** Design foundation + app shell + New run: tokens (light/dark) mapped into Tailwind, bundled fonts, `lucide-react` icons, theme toggle, sidebar run list, top bar, status-tag mapping, restyled New run screen with the "What the agent does" aside and sample datasets (`GET /api/samples`, start a run from a sample).
- **7.7** Run screen: header chips, pipeline graph (pure, tested graph-state function), tabs, Steps tab with expandable rows and attempt groups; backend: per-event `detail`.
- **7.8** (split: 7.8a backend, 7.8b UI) Results tab + failure states: recommendation card, "why this model" fold squares, artifacts with sizes, leaderboard with fold dots, warnings strip, no-model and fallback states; backend: `selection` + lift + file sizes in `RunSummary`.
- **7.9** Report tab, Feature code tab, All runs page (search + status filter), narrow-screen sidebar drawer; backend: richer `RunListItem`.
- **7.10** Production serve + DoD: `autods serve` serves the built frontend (SPA fallback); walkthrough on a never-seen CSV.

New dependencies: `fastapi`, `uvicorn`, `python-multipart` (Python); `react`, `react-dom`, `react-router`, `vite`, `typescript`, Tailwind, `react-markdown` + `remark-gfm` + `@tailwindcss/typography`, `vitest` + `@testing-library/react` (Node); for the redesign `lucide-react` and `@fontsource/*`.

The developer is new to React: frontend tasks explain each concept as it appears (components, props, state, hooks, effects) and trace every request from button click to the agent and back.

**DoD:** `uv run autods serve` → browser → upload a never-seen CSV → watch every stage live, including a retry when one happens → leaderboard and report → download the artifacts. A stranger can do this without reading the code. API tests and frontend tests green; `npm run build` clean.

---

### Phase 8 — Hardening & presentation · ~5h
- pytest suite with LLM calls mocked; ≥ 60% coverage on non-LLM modules
- GitHub Actions with two jobs: Python (ruff + pytest) and web (`npm ci`, lint, test, build)
- README: architecture diagram, 60-second demo GIF of the React app running a live run, honest limitations section, "why LLM-for-judgment-only"
- Dockerfile, multi-stage: a Node stage builds `web/`, a Python stage runs `autods serve` (and is what makes the sandbox claim real)

**DoD:** clone → `docker run` (or `uv sync && autods serve`) → working web app, no hand-holding.

---

## 7. Evaluation

Do not skip this. "I evaluated my agent" is worth more on a CV than one more feature.

| What | How | Target |
|---|---|---|
| Router accuracy | 10 labeled datasets | ≥ 80% |
| FE first-attempt success | 10 datasets | ≥ 60% |
| FE success within 3 attempts | same | ≥ 90% |
| Beats dummy baseline | 5 datasets | 5/5 |
| Mean cost per run | token accounting | track it, report it |

Put this table in the README with your real numbers. Including the ones that disappoint you.

---

## 8. Suggested schedule (~40h)

| Week | Phases |
|---|---|
| 1 | 0, 1, 2 — foundations and first LLM node |
| 2 | 3, 4 — codegen, sandbox, corrective loop |
| 3 | 5, 6 — arena and reporting; end-to-end works |
| 4 | 7 — web app (FastAPI + React) |
| 5 | 8 — CI, Docker, README, demo |

---

## 9. Stretch goals (only after Phase 8 is green)

Time-series support · SHAP explanations in the report · Optuna HPO · cloud deploy of the Phase 7–8 web app · multi-user job queue · human-in-the-loop approval before FE runs (LangGraph `interrupt`) · caching profiles/decisions by dataset hash

---

## 10. CV framing

> **Autonomous Data Science Agent** — LangGraph agent that profiles a tabular dataset, infers the ML task type, generates and self-repairs feature engineering code in a sandboxed executor (3-attempt corrective loop with graceful fallback), benchmarks 6 model families under cross-validation, and produces a justified, number-verified recommendation — via a CLI and a React + FastAPI web app that streams the agent's progress live over Server-Sent Events. Architected so the LLM handles judgment while all metrics and training remain deterministic. Router 10/10 on a labelled eval set (stable over 3 repeats); generated feature code passed on the first attempt for 10/10 datasets; chosen model beat the baseline 5/5; ~$0.02 of LLM calls per run. Dockerized; 400+ tests and a 3-job CI pipeline.

*(Numbers from the README's Evaluation section, updated 2026-11-21 from the planning placeholders "9/10" and "92%". In an interview, say why 10/10 FE is a ceiling, not a rate: easy public datasets, single pass; the repair loop's evidence is the injected-fault eval.)*

Interview questions this project earns you, and which you must be able to answer:
1. Why keep the LLM out of the training loop?
2. How do you stop generated code from destroying the host?
3. What does the retry loop do when the model keeps failing the same way?
4. How do you know the agent is any good? *(← the eval table)*
5. What breaks at 10 GB instead of 100 MB?