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

**Phase:** 3 — FE codegen + sandbox. Phase 1 ✅ · Phase 2 ✅
**Phase 2 result:** router eval 10/10 (raw and post-sanity-check), 10/10 stable over 3 runs, $0.0049/call on `openai:gpt-6.1-sol` (`runs/eval_router_20261005_003942.json`); `openai:gpt-6-luna` also 10/10 stable at $0.0003/call (`runs/eval_router_20261005_004809.json`) → router now on Luna, codegen/reporter on Sol. Caveat for README: descriptions are explicit and the sanity check never fired in the eval; its value is shown by unit tests. Candidate Phase 8 addition: a harder eval set (vague/missing descriptions, ambiguous targets).
**Phase 3 done:** 3.1 contract (`sandbox/contract.py`: `prepare_target` converts text-number targets before FE; `validate_output` enforces exact row count, target unchanged, numeric/NaN/inf-free features, ≤ `MAX_FE_COLUMNS`, leakage — new leaky column = error, original = warning). Branch: `phase-3-sandbox`. · 3.2 executor (`sandbox/executor.py` + `_runner.py`: AST static check with import allowlist and blocked builtins, subprocess with stripped env (no API keys) + timeout + POSIX memory cap, 1000-row seeded sample then full run, parent-side validation, structured `ExecutionResult`)
· 3.3 FE codegen (`nodes/fe_codegen.py`, `FECode{explanation, code}`, prompts `fe_system/fe_user/fe_retry.md`, retry = latest attempt in full + one-line summaries of earlier ones; shared `llm.invoke_structured` now used by router too; new state key `fe_explanations`). Live: Titanic failed attempt 1 (pandas 3 read-only array) and self-repaired on attempt 2; Ames (296 features, truncated profile) and penguins passed first try; ~$0.016–0.040 per codegen run.
**Next task:** 3.4 — sandbox graph node (load → prepare_target → execute → state) + Phase 3 DoD tests

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
