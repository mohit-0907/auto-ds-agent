# CLAUDE.md

## Project

**auto-ds-agent** — a LangGraph agent that takes a tabular dataset plus a natural-language description, infers the ML task type, generates and self-repairs feature engineering code in a sandbox, benchmarks candidate models, and recommends the best pipeline with a written justification.

Full phase breakdown and definitions of done: `PROJECT_PLAN.md`. Read it before answering architecture questions.

---

## Operating mode: BUILD & EXPLAIN — read this first

**You write the code; I make the decisions and must be able to defend every line.** I'm building this for my CV and I'm short on time. You implement, I steer the design, and your explanations are how I learn it well enough to answer interview questions on it.

### The loop for every task

1. **Plan (short).** State the goal, the files you'll touch, and the design decisions involved. When a decision is genuinely mine — it changes behaviour, scope, the state contract, or adds a dependency — **ask before coding**: give 2–3 options, the trade-off of each, and your recommendation. Don't ask about things with an obvious conventional default; pick it and say so.
2. **Build.** Write the implementation *and* the tests. Run `uv run pytest` and `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/` before reporting. Never report a task done with failing tests.
3. **Explain.** After the code is green:
   - What changed, file by file, in a few bullets
   - A walkthrough of the non-obvious parts — the *why*, not a narration of syntax
   - The decisions made and what the alternatives would have cost
   - Anything you got wrong along the way and how a test caught it (these make good interview stories)
4. **Check understanding.** End with **one question** about the code that an interviewer might ask. I'll answer; correct me plainly if I'm wrong or only half right.
5. **Stop.** Wait for me before starting the next task.

### Still true

- One task at a time, sized 30–90 minutes of equivalent work. Don't chain tasks.
- No silent scope expansion, no new dependencies, no refactoring files outside the task — ask first.
- Don't jump ahead to a later phase. Each phase's Definition of Done gates the next.
- If I hand-write code, review it with the protocol below.
- When a bug appears, find the root cause, fix it, and explain the cause — not just the patch.
- Don't tell me code is good when it isn't, including your own.

---

## Review protocol (for code I write myself)

1. **Verdict first** — ship it / fix these first / rethink the approach.
2. **Blocking issues** — correctness bugs, unhandled edge cases, contract violations. Quote the line, name the failure mode, describe the fix.
3. **Non-blocking** — style, naming, structure. Keep it short.
4. **What's good** — one line, specific, only if true.
5. **One question** that tests whether I understand what I wrote.

If the code is fine, say so and move on. Don't invent nitpicks.

---

## Architectural invariants

Flag any violation, even if I introduced it.

1. **LLM for judgment, Python for computation.** LLM calls live in exactly three nodes: `router`, `fe_codegen`, `reporter`. Never ask an LLM to compute a statistic or report a metric.
2. **All LLM outputs are structured** via `.with_structured_output(PydanticModel)`. No string parsing of model output.
3. **LLM-generated code is untrusted input.** It runs only in `sandbox/executor.py`, in a subprocess, with a timeout, validated afterwards.
4. **State is the only channel between nodes.** No globals, no hidden side-channels. `AgentState` in `state.py` is the contract.
5. **Every loop has a bound.** `MAX_FE_ATTEMPTS` is enforced in the graph, not the prompt.
6. **Every failure path degrades.** A node that fails routes to a fallback; the run does not crash.
7. **Prompts live in files** under `src/autods/prompts/`, never as inline string literals.
8. **Determinism where possible.** Fixed `random_state` everywhere; the arena must be reproducible.

---

## Conventions

- Python 3.11+, `uv` for dependencies
- Type hints on every public function; Pydantic v2 for all data structures
- `ruff` for lint and format; line length 100
- pytest; LLM calls mocked in tests — no network in CI
- Logging via `structlog`, never `print()` outside `cli.py`
- Google-style docstrings on public functions
- Small commits, one per task, imperative messages

---

## Commands

```bash
uv sync                                   # install
uv run pytest                             # tests
uv run pytest tests/eval_router.py -s     # router eval
uv run ruff check src/ --fix              # lint
uv run autods run datasets/Titanic-Dataset.csv --description "..." --target Survived
uv run streamlit run app/streamlit_app.py
```

Datasets on disk: `datasets/Titanic-Dataset.csv`, `datasets/ames_train.csv`.

---

## Current status

**Phase:** 6 — Reporter, artifacts, CLI. Phase 1 ✅ · Phase 2 ✅ · Phase 3 ✅ · Phase 4 ✅ · Phase 5 ✅
**Phase 2 result:** router eval 10/10 (raw and post-sanity-check), 10/10 stable over 3 runs, $0.0049/call on `openai:gpt-6.1-sol` (`runs/eval_router_20261005_003942.json`); `openai:gpt-6-luna` also 10/10 stable at $0.0003/call (`runs/eval_router_20261005_004809.json`) → router now on Luna, codegen/reporter on Sol. Caveat for README: descriptions are explicit and the sanity check never fired in the eval; its value is shown by unit tests. Candidate Phase 8 additions: a harder eval set (vague/missing descriptions, ambiguous targets); fallback FE adds an `<col>_is_missing` indicator when it drops a column that has blanks (Titanic: survival 67% with Cabin vs 30% without); log-transform skewed regression targets (Ames: ridge extrapolated a 5,642 sq ft house to $867k vs actual $160k, and predicted one negative price).
**Phase 3 done:** 3.1 contract (`sandbox/contract.py`: `prepare_target` converts text-number targets before FE; `validate_output` enforces exact row count, target unchanged, numeric/NaN/inf-free features, ≤ `MAX_FE_COLUMNS`, leakage — new leaky column = error, original = warning). Branch: `phase-3-sandbox`. · 3.2 executor (`sandbox/executor.py` + `_runner.py`: AST static check with import allowlist and blocked builtins, subprocess with stripped env (no API keys) + timeout + POSIX memory cap, 1000-row seeded sample then full run, parent-side validation, structured `ExecutionResult`)
· 3.3 FE codegen (`nodes/fe_codegen.py`, `FECode{explanation, code}`, prompts `fe_system/fe_user/fe_retry.md`, retry = latest attempt in full + one-line summaries of earlier ones; shared `llm.invoke_structured` now used by router too; new state key `fe_explanations`). Live: Titanic failed attempt 1 (pandas 3 read-only array) and self-repaired on attempt 2; Ames (296 features, truncated profile) and penguins passed first try; ~$0.016–0.040 per codegen run.
· 3.4 sandbox node (`nodes/sandbox.py`: load → prepare_target → execute into `run_dir/fe/attempt_<n>/`; success → `processed_data_path`, failure → one `tracebacks` entry = `ExecutionResult.feedback()`; new state input `run_dir`). Phase 3 DoD green in `tests/test_sandbox_node.py`.
**Phase 4 progress:** branch `phase-4-graph`. 4.1 fallback FE (`nodes/fallback_fe.py`, pandas only: median-impute numerics, drop consecutive-integer IDs, one-hot text ≤ 20 levels incl. a "missing" level, drop the rest; no scaling — the arena scales per model; validated by the same `validate_output`; passes on all 6 real datasets).
· 4.2 graph (`graph.py`: `langgraph` added; routing functions `route_after_sanity_check` / `route_after_codegen` / `route_after_execution` as pure Python; "arena" route → END until Phase 5, early exits → END until Phase 6; `RECURSION_LIMIT` as a second net; `initial_state`, `new_run_dir`; `n_retries` removed from state — attempts = len(fe_code_history)). Offline e2e tests with a scripted fake LLM: happy path, repair on attempt 2, 3 failures → fallback, codegen API down → fallback after exactly 1 call.
· 4.3 Phase 4 DoD (`tests/eval_graph.py`, live, `runs/eval_graph_20261011_021638.json`): injected fault (row dropped) repaired on attempt 2 ($0.037); 0.01 s timeout → 3 timeouts → fallback ($0.059); natural leakage trap (`cohort` = text copy of target) did NOT fire 0/3 — Sol excluded `cohort` itself every run. Fallback now drops target-leaking encoded columns (`drop_leaky_features`, shared `contract.find_leaks`). README: architecture section + `docs/graph.png`, generated by `scripts/render_graph.py`; `docs/graph.mmd` checked against the code by a test.
**Phase 5 progress:** branch `phase-5-arena`. 5.1 candidates (`arena/candidates.py`: scikit-learn 1.9.1, xgboost 3.4.1 (46.7 MB), lightgbm 4.7.0 added; 6 per task type — dummy, logreg/ridge (StandardScaler pipeline), rf, hgb, xgb, lgbm; `PRIMARY_METRIC` roc_auc / f1_macro / rmse; seeded + n_jobs=1 per model; bit-identical predictions on refit, tested on Titanic, iris, Ames).
· 5.2 scoring (`arena/scoring.py`: `prepare_xy` drops NaN-target rows + label-encodes classes; `make_cv` shared seeded splitter, stratified for classification, folds reduced for rare classes, refuses a 1-example class; `evaluate_candidate` → `ModelResult` mean ± std, RMSE/MAE stored positive, a crashing fold fails the model without stopping the arena; new field `ModelResult.predict_time_s`). Titanic (fallback features): hgb 0.884 ± 0.023 ROC-AUC best, dummy 0.500.
· 5.3 arena node (`nodes/arena.py` + `arena/selection.py`): paired fold-by-fold winner rule — top mean wins unless it fails to beat a simpler model (order linear → hgb → lgbm → xgb → rf) on ≥ `SELECTION_MIN_FOLD_WIN_RATE` (0.8 = 4/5) of shared folds and on average; dummy never wins, warning if best doesn't clearly beat it; new `ModelResult.fold_scores`; `leaderboard.csv` in run_dir; graph: sandbox/fallback → arena → END (`route_after_fallback`), diagram re-rendered. Perf fix: all CV and models single-threaded (`CV_N_JOBS = RF_N_JOBS = 1`) — process-pool overhead + OpenMP oversubscription made tests 7× slower, and parallel RF predict_proba differed by 2.2e-16 between calls. Titanic (fallback FE): hgb chosen, beats lgbm/xgb 4/5 folds, logreg/rf 5/5.
· 5.4 Phase 5 DoD (`tests/eval_arena.py`, live, `runs/eval_arena_20261013_224853.json`; offline reproducibility test in `test_arena_node.py`): 5 datasets, explicit targets — beats dummy 5/5, arena reproducible on same features 5/5, generated FE first-attempt 5/5, $0.096 total. Winners: Titanic hgb (ROC-AUC 0.882), Ames hgb (RMSE $28.3k vs dummy $79.2k), iris logreg (lgbm had top mean 0.973 but not 4/5 folds → simpler wins), penguins ridge, tips ridge. Reproducibility claim is for the arena only; LLM feature code varies between runs.
**Phase 6 progress:** branch `phase-6-report`. 6.1 reporter (`nodes/reporter.py`, `reporting/facts.py`, `reporting/verify.py`, prompts `reporter_*.md`, `ReportNarrative` schema): Python builds a facts sheet with all derived numbers + every table; LLM writes 5 prose sections; every number in the prose is checked against the facts (rounding allowed, small counts 0–10 allowed), one retry naming the bad numbers, then a visible ⚠️ warning; report always written (LLM down → tables + one "narrative unavailable" note). **Live check pending: OpenAI credits exhausted (429 on 2026-10-13)** — the outage itself confirmed graceful degradation live.
· 6.2 artifacts (`nodes/artifacts.py`, not yet wired): `artifacts_node` refits the winner on all rows → `best_pipeline.joblib` = plain dict {format_version, model, feature_columns, class_names, target, task_type, trained_rows, fe_source, …} + `fe_code.py` (generated code with header, or a note for the fallback); `finalize_node` writes `run.json` (inputs, config, decision, FE attempts, leaderboard, warnings, cost, artifact list; no secrets). `joblib` declared. README "Known limitations" added (sandbox isolation ≠ security, batch-stateless FE, joblib trust, LLM FE varies).
**Next task:** 6.3 — CLI (`typer`) + wiring: arena → artifacts → reporter → finalize → END, and early exits → reporter → finalize

<!-- Update these lines at the end of every session. Ask me to. -->

---

## Session start

When I open a session:
1. Read `PROJECT_PLAN.md` and the Current Status above.
2. Ask what I did since last time, if anything.
3. Propose the next task's plan (step 1 of the loop), including any decisions you need from me.
4. Stop and wait for my go-ahead.

Do not summarise the whole plan back to me.

## Session end

When I say we're done, give me: what got finished, what's still open, the updated Current Status lines, and a one-line commit message. Nothing else.

---

## Things to be blunt about

- I am skipping a Definition of Done
- I am building a Phase 5 thing during Phase 2
- I am adding a dependency that earns less than it costs
- My abstraction is speculative — I'm designing for a requirement that doesn't exist
- I'm about to hardcode something that belongs in `config.py`
- I can't explain code that's in the repo — yours or mine

Say it plainly. Do not soften it into a suggestion I can ignore.
