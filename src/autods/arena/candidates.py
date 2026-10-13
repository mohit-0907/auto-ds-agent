"""The candidate models the arena benchmarks, per task type.

Every list starts with a dummy baseline: a model that ignores the features
entirely. A candidate that cannot beat it has learned nothing, whatever its
score looks like in isolation.

Determinism (invariant 8): every estimator with randomness gets RANDOM_STATE,
and every model runs single-threaded: multithreaded sums are added in whatever
order threads finish, and float addition is order-dependent (see RF_N_JOBS and
CV_N_JOBS in config for the measured cases). LightGBM also gets
deterministic=True with row-wise histograms, its documented recipe for
bit-identical results.

Scaling: linear models get StandardScaler inside their own pipeline; tree
models get raw features. Neither feature engineering path scales, so this is
the one place it happens, and only where it helps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
from pydantic import BaseModel, ConfigDict
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from autods.config import ARENA_LEARNING_RATE, ARENA_N_ESTIMATORS, RANDOM_STATE, RF_N_JOBS

# The metric each task type's leaderboard ranks on. Chosen before any model is
# fitted, so a failed model still reports which metric it was competing on.
PRIMARY_METRIC: dict[str, str] = {
    "binary_classification": "roc_auc",  # threshold-free; robust to class imbalance
    "multiclass_classification": "f1_macro",  # every class counts equally
    "regression": "rmse",  # in the target's own units; lower is better
}


class Candidate(BaseModel):
    """One model entered into the arena: a fresh, unfitted estimator."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model_id: str
    model_name: str
    estimator: Any  # an unfitted scikit-learn-compatible estimator


_SEED = {"random_state": RANDOM_STATE}
_BOOSTING = {"n_estimators": ARENA_N_ESTIMATORS, "learning_rate": ARENA_LEARNING_RATE}
_LGBM_DETERMINISM = {"deterministic": True, "force_row_wise": True, "verbose": -1}


def _classifiers() -> list[Candidate]:
    return [
        Candidate(
            model_id="dummy",
            model_name="Dummy (class priors)",
            estimator=DummyClassifier(strategy="prior"),
        ),
        Candidate(
            model_id="logreg",
            model_name="Logistic Regression",
            estimator=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **_SEED)),
        ),
        Candidate(
            model_id="rf",
            model_name="Random Forest",
            estimator=RandomForestClassifier(
                n_estimators=ARENA_N_ESTIMATORS, n_jobs=RF_N_JOBS, **_SEED
            ),
        ),
        Candidate(
            model_id="hgb",
            model_name="Hist Gradient Boosting",
            estimator=HistGradientBoostingClassifier(learning_rate=ARENA_LEARNING_RATE, **_SEED),
        ),
        Candidate(
            model_id="xgb",
            model_name="XGBoost",
            estimator=XGBClassifier(**_BOOSTING, n_jobs=1, verbosity=0, **_SEED),
        ),
        Candidate(
            model_id="lgbm",
            model_name="LightGBM",
            estimator=LGBMClassifier(**_BOOSTING, n_jobs=1, **_LGBM_DETERMINISM, **_SEED),
        ),
    ]


def _regressors() -> list[Candidate]:
    return [
        Candidate(
            model_id="dummy",
            model_name="Dummy (mean)",
            estimator=DummyRegressor(strategy="mean"),
        ),
        Candidate(
            model_id="ridge",
            model_name="Ridge Regression",
            estimator=make_pipeline(StandardScaler(), Ridge(**_SEED)),
        ),
        Candidate(
            model_id="rf",
            model_name="Random Forest",
            estimator=RandomForestRegressor(
                n_estimators=ARENA_N_ESTIMATORS, n_jobs=RF_N_JOBS, **_SEED
            ),
        ),
        Candidate(
            model_id="hgb",
            model_name="Hist Gradient Boosting",
            estimator=HistGradientBoostingRegressor(learning_rate=ARENA_LEARNING_RATE, **_SEED),
        ),
        Candidate(
            model_id="xgb",
            model_name="XGBoost",
            estimator=XGBRegressor(**_BOOSTING, n_jobs=1, verbosity=0, **_SEED),
        ),
        Candidate(
            model_id="lgbm",
            model_name="LightGBM",
            estimator=LGBMRegressor(**_BOOSTING, n_jobs=1, **_LGBM_DETERMINISM, **_SEED),
        ),
    ]


_BUILDERS: dict[str, Callable[[], list[Candidate]]] = {
    "binary_classification": _classifiers,
    "multiclass_classification": _classifiers,
    "regression": _regressors,
}


def candidates(task_type: str) -> list[Candidate]:
    """Fresh, unfitted candidate models for a task type, dummy baseline first.

    A new list of new estimators on every call, so one run's fitted state can
    never leak into another's.

    Args:
        task_type: A supported TaskDecision.task_type.

    Returns:
        The candidates, in a fixed order (the leaderboard's tie-break order).

    Raises:
        ValueError: For "unsupported" or an unknown task type -- the graph
            should never route those to the arena.
    """
    if task_type not in _BUILDERS:
        raise ValueError(f"no arena candidates for task type {task_type!r}")
    return _BUILDERS[task_type]()
