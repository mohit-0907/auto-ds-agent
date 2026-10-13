"""Cross-validate one candidate into a ModelResult. Deterministic; no LLM.

Every candidate in a run is scored with the same splitter object built by
make_cv, so all models face exactly the same five train/test splits -- the
comparison is between models, never between lucky and unlucky splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

from autods.arena.candidates import PRIMARY_METRIC, Candidate
from autods.config import CV_FOLDS, CV_N_JOBS, RANDOM_STATE
from autods.state import ModelResult

log = structlog.get_logger(__name__)

# Metric name -> scikit-learn scorer, per task type. Scorers are "higher is
# better", so error metrics come back negated; NEGATED lists the ones whose
# sign evaluate_candidate flips back, so RMSE is stored as positive dollars.
# zero_division=0: a model that never predicts the positive class (the dummy)
# has precision 0 by definition, not a warning per fold.
_METRICS: dict[str, dict[str, object]] = {
    "binary_classification": {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
    },
    "multiclass_classification": {
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    },
    "regression": {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    },
}
NEGATED = frozenset({"rmse", "mae"})
LOWER_IS_BETTER = NEGATED  # after the flip, these read as errors: smaller wins


def prepare_xy(
    features: pd.DataFrame, target_column: str, task_type: str
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Split features from the target, ready for scikit-learn.

    Rows with a missing target are dropped: there is no answer to learn from.
    Classification targets are always encoded to 0..k-1 -- XGBoost requires it,
    and doing it for every model keeps the contest identical.

    Args:
        features: Output of feature engineering, target included.
        target_column: The target column name.
        task_type: A supported task type.

    Returns:
        (X, y, class_names, notes): class_names[i] is the original label of
        encoded class i (empty for regression); notes are for state warnings.
    """
    notes: list[str] = []
    missing = features[target_column].isna()
    if missing.any():
        notes.append(f"Dropped {int(missing.sum())} rows with a missing target before training.")
        features = features[~missing]

    X = features.drop(columns=target_column)
    target = features[target_column]
    if task_type == "regression":
        return X, target.to_numpy(dtype="float64"), [], notes

    encoder = LabelEncoder().fit(target)
    return X, encoder.transform(target), [str(c) for c in encoder.classes_], notes


def make_cv(task_type: str, y: np.ndarray) -> KFold | StratifiedKFold:
    """The splitter every candidate in this run is scored with.

    Classification is stratified, so every fold keeps the dataset's class mix.
    Shuffled with a fixed seed: rows are often sorted (by date, by class), and
    unshuffled folds would test on systematically different rows.

    Args:
        task_type: A supported task type.
        y: The encoded target.

    Returns:
        A splitter with CV_FOLDS folds, or fewer if the rarest class has fewer
        members than that.

    Raises:
        ValueError: If some class has a single example -- it cannot appear in
            both a training and a test fold.
    """
    if task_type == "regression":
        return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rarest = int(np.bincount(y).min())
    if rarest < 2:
        raise ValueError("a class has only 1 example; cross-validation needs at least 2 per class")
    folds = min(CV_FOLDS, rarest)
    if folds < CV_FOLDS:
        log.warning("scoring.fewer_folds", folds=folds, rarest_class_count=rarest)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)


def evaluate_candidate(
    candidate: Candidate,
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: str,
    cv: KFold | StratifiedKFold,
) -> ModelResult:
    """Cross-validate one candidate and summarize it as a ModelResult.

    A fold that raises fails the whole model (error_score="raise"): an average
    over only the folds that worked would flatter it. The failure is recorded,
    not raised, so the other candidates still compete (invariant 6).

    Args:
        candidate: The model to score.
        X: Features.
        y: Encoded target.
        task_type: A supported task type.
        cv: The run's shared splitter, from make_cv.

    Returns:
        A ModelResult with mean and std per metric, or status="failed".
    """
    primary = PRIMARY_METRIC[task_type]
    try:
        scores = cross_validate(
            candidate.estimator,
            X,
            y,
            cv=cv,
            scoring=_METRICS[task_type],
            n_jobs=CV_N_JOBS,
            error_score="raise",
        )
    except Exception as exc:  # any estimator error; this model loses, the arena goes on
        error = f"{type(exc).__name__}: {exc}"
        log.warning("scoring.failed", model_id=candidate.model_id, error=error)
        return ModelResult(
            model_id=candidate.model_id,
            model_name=candidate.model_name,
            status="failed",
            primary_metric=primary,
            error=error,
        )

    metrics, metrics_std, fold_scores = {}, {}, []
    for name in _METRICS[task_type]:
        values = scores[f"test_{name}"]
        if name in NEGATED:
            values = -values
        metrics[name] = float(np.mean(values))
        metrics_std[name] = float(np.std(values))
        if name == primary:
            fold_scores = [float(v) for v in values]

    return ModelResult(
        model_id=candidate.model_id,
        model_name=candidate.model_name,
        status="success",
        metrics=metrics,
        metrics_std=metrics_std,
        primary_metric=primary,
        fold_scores=fold_scores,
        training_time_s=float(np.sum(scores["fit_time"])),
        predict_time_s=float(np.mean(scores["score_time"])),
    )
