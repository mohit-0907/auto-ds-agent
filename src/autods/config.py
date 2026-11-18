"""Settings, model names, and hard limits.

Anything that would otherwise be hardcoded inside a node belongs here. Loop
bounds in particular are enforced by the graph reading these constants, never by
asking a prompt nicely (invariant 5).

Values that vary by environment are read from .env; values that define the
agent's behaviour are constants, so a run is reproducible from the source alone.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
RUNS_DIR = PROJECT_ROOT / "runs"
# The built React app (`cd web && npm run build`). `autods serve` serves it next to the API.
WEB_DIST_DIR = PROJECT_ROOT / "web" / "dist"

# --- Models ----------------------------------------------------------------
# Model strings are "<provider>:<model-id>"; autods.llm.get_llm dispatches on the
# prefix, so switching provider is a config change, never a node change.
#
# Three named seams rather than one constant: the router is a classification
# call, the reporter is a writing call, and codegen is the expensive one.
# AUTODS_MODEL, when set, overrides all three for cheap end-to-end testing.
#
# The router runs on the efficient tier: on the Phase 2 eval gpt-6-luna matched
# gpt-6.1-sol (10/10, stable over 3 runs) at 1/16 the cost ($0.0003 vs $0.0049
# per call; runs/eval_router_20261005_*.json). That eval has explicit
# descriptions, so re-check this choice when a harder eval set exists.

_MODEL_OVERRIDE = os.getenv("AUTODS_MODEL")
DEFAULT_MODEL = _MODEL_OVERRIDE or "openai:gpt-6.1-sol"

ROUTER_MODEL = _MODEL_OVERRIDE or "openai:gpt-6-luna"
FE_CODEGEN_MODEL = DEFAULT_MODEL
REPORTER_MODEL = DEFAULT_MODEL

# No temperature setting, deliberately. Current models from both providers reject
# any non-default temperature with a 400 (verified live against gpt-6.1-sol;
# documented for Claude Opus 5.5 / Sonnet 5.5). Invariant 8 is "determinism where
# possible" -- here it is not, so routing stability comes from the deterministic
# sanity check after the LLM call and is measured by repeated eval runs instead.
LLM_MAX_TOKENS = 8_000
LLM_TIMEOUT_S = 120
LLM_MAX_RETRIES = 2  # SDK-level retries on 429/5xx/connection errors.

# USD per 1,000,000 tokens, as (input, output), keyed by the full model string.
# Used to populate state["token_cost_usd"] -- Python computes the cost, never
# the LLM. Standard-tier list prices, checked 2026-10-04 against
# developers.openai.com/api/docs/pricing and Anthropic's published rates.
MODEL_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai:gpt-6-astra": (10.00, 50.00),
    "openai:gpt-6.1-sol": (2.00, 10.00),
    "openai:gpt-6-luna": (0.10, 0.50),
    "anthropic:claude-opus-5-5": (4.00, 20.00),
    "anthropic:claude-sonnet-5-5": (2.00, 10.00),
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
}

# --- Loop bounds -----------------------------------------------------------

MAX_FE_ATTEMPTS = 3  # After this many failures the graph routes to fallback FE.

# --- Sandbox ---------------------------------------------------------------

SANDBOX_TIMEOUT_S = 120  # Per run (sample and full each get this long).
SANDBOX_SAMPLE_ROWS = 1_000  # Run generated code on a sample first, to fail fast.
# Address-space cap for the child process. POSIX only: Windows has no
# `resource` module, so there the timeout is the only resource limit.
SANDBOX_MAX_MEMORY_MB = 4_096
SANDBOX_OUTPUT_TAIL_CHARS = 3_000  # stdout/stderr kept per run, from the end.
# Guardrails against accidental harm, checked statically before execution.
# Not security: a determined adversary can get around an AST check.
SANDBOX_ALLOWED_IMPORTS: tuple[str, ...] = (
    "pandas",
    "numpy",
    "math",
    "re",
    "datetime",
    "string",
    "itertools",
    "collections",
    "functools",
    "statistics",
    "typing",
)
SANDBOX_BLOCKED_CALLS: tuple[str, ...] = (
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
)
LEAKAGE_CORRELATION_THRESHOLD = 0.99  # A new column this correlated with the target is suspect.
# Ceiling on feature columns the generated code may return; catches one-hot
# encoding of a high-cardinality column exploding the frame.
MAX_FE_COLUMNS = 500

# --- Web API ---------------------------------------------------------------

DATASETS_DIR = PROJECT_ROOT / "datasets"
# Datasets the web app offers as one-click samples, so someone without a CSV at hand can
# try the agent. Only these names can be started as a sample (an allowlist, like
# downloads). An empty target means the agent infers it -- shown on purpose for penguins.
SAMPLE_DATASETS: tuple[dict[str, str], ...] = (
    {
        "file": "Titanic-Dataset.csv",
        "kind": "binary",
        "description": "Predict which passengers survived",
        "target": "Survived",
    },
    {
        "file": "penguins.csv",
        "kind": "multiclass",
        "description": "Classify the penguin species",
        "target": "",
    },
    {
        "file": "mpg.csv",
        "kind": "regression",
        "description": "Predict a car's fuel efficiency",
        "target": "mpg",
    },
    {
        "file": "ames_train.csv",
        "kind": "regression",
        "description": "Predict the house sale price",
        "target": "SalePrice",
    },
)

# How often an open live-events stream checks for new stage events. Stages take
# seconds, so a quarter-second delay is invisible.
SSE_POLL_INTERVAL_S = 0.25
# A comment line sent on a quiet stream, so proxies and browsers do not drop
# a connection that waits a long time (the arena can be quiet for a while).
SSE_KEEPALIVE_S = 15.0

# Each stage event carries an expandable detail (generated code, a traceback, the
# router's decision). Capped so one event stays small in events.jsonl and on the wire;
# 12k characters fits a full generated feature file with room to spare.
EVENT_DETAIL_MAX_CHARS = 12_000
EVENT_DETAIL_PROFILE_COLUMNS = 8  # columns listed in the profiler step's detail

# --- Profiler --------------------------------------------------------------

MAX_DATASET_MB = 200  # Checked against file size on disk, before loading.
# With CV_FOLDS=5 this leaves 10 rows per test fold -- about the least where a
# cross-validated score means anything.
MIN_ROWS = 50
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".csv", ".parquet", ".pq")
MAX_DETAILED_COLUMNS = 40  # Above this, the profile summarizes by dtype group.
N_SAMPLE_VALUES = 3
MAX_SAMPLE_VALUE_CHARS = 50  # One free-text column must not blow the budget alone.
PROFILE_TOKEN_BUDGET = 4_000  # Ceiling for the profile JSON sent to the router.
# Measured, not assumed: profile JSON tokenizes at 2.60-2.98 chars/token under
# tiktoken o200k/cl100k (digits and punctuation split finely), versus ~4 for
# prose. 2.5 sits below the lowest observation, so estimates err high.
CHARS_PER_TOKEN = 2.5
HIGH_CARDINALITY_RATIO = 0.5  # Above this ratio, a column is likely an identifier.
MAX_CATEGORICAL_UNIQUE = 20  # A numeric column with at most this many levels reads as categorical.
# Lowercase, matched against the whole column name -- never as a substring, or
# "y" would match "salary" and "city".
TARGET_NAME_HINTS: tuple[str, ...] = ("target", "label", "y", "class", "outcome")
MAX_TARGET_CANDIDATES = 5

# --- Sanity check (router output vs target statistics) --------------------

# A regression target with at most this many whole-number levels is ambiguous
# (ratings 1-5 are classes; a count of children 0-5 is a quantity): flagged.
SANITY_FEW_INTEGER_LEVELS = 10
# Above this many distinct values, a numeric target is a quantity, not classes:
# a classification decision is overridden to regression.
SANITY_MAX_CLASSES = 50
# Confidence ceiling for a decision the check flags but does not override.
SANITY_FLAGGED_CONFIDENCE = 0.5

# --- Arena -----------------------------------------------------------------

RANDOM_STATE = 42  # Invariant 8: fixed everywhere, the arena must be reproducible.
CV_FOLDS = 5
# Folds run one after another. Measured on Windows: ridge CV on Ames took 0.19 s
# sequential vs 3.4 s with n_jobs=-1 -- each worker is a new process that must
# import pandas/sklearn/xgboost/lightgbm and receive a copy of the data, and
# hgb's own OpenMP threads inside every worker oversubscribed the CPU (one test
# went from 17 s to 122 s).
CV_N_JOBS = 1
# Every model is single-threaded too, random forest included. Its trees are
# seeded one by one, so training is identical with any n_jobs -- but with
# n_jobs=-1 predict_proba sums the trees' votes across threads in whatever
# order they finish, and float addition is order-dependent: the same fitted
# forest predicted twice differed by 2.2e-16. Tiny, but invariant 8 asks for
# the same answer every run, and the determinism test caught it.
RF_N_JOBS = 1
# Sensible library-style defaults, not tuned (no heavy HPO in v1). Shared by
# the forest and boosting candidates so they compete on similar budgets.
ARENA_N_ESTIMATORS = 300
ARENA_LEARNING_RATE = 0.05
# The top model must beat a simpler one on at least this share of the shared CV
# folds to be chosen over it. If two models were truly equal, each fold would be
# a coin flip: winning >= 3 of 5 happens 50% of the time by luck, >= 4 of 5
# about 19%, 5 of 5 about 3%. So a bare majority is no evidence; 0.8 is.
SELECTION_MIN_FOLD_WIN_RATE = 0.8
MODEL_FIT_TIMEOUT_S = 300

# --- Logging ---------------------------------------------------------------

DEBUG_LLM = os.getenv("AUTODS_DEBUG_LLM", "0") == "1"


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute the USD cost of one LLM call.

    Args:
        model: Full model string, "<provider>:<model-id>".
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.

    Returns:
        Cost in USD, or 0.0 if the model has no pricing entry. The miss is
        logged, because a silent $0.00 would read as "free" in the eval table.
    """
    if model not in MODEL_PRICING_PER_MTOK:
        structlog.get_logger(__name__).warning("cost.unpriced_model", model=model)
        return 0.0
    input_rate, output_rate = MODEL_PRICING_PER_MTOK[model]
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
