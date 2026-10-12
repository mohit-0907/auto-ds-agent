"""Tests for the arena's candidate models.

Features come from the fallback recipe (basic_features), so these run on the
same kind of frame the arena will receive: numeric and bool columns, no NaN.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from autods.arena.candidates import PRIMARY_METRIC, candidates
from autods.config import PROJECT_ROOT
from autods.nodes.fallback_fe import basic_features

DATASETS = PROJECT_ROOT / "datasets"


def _xy(file: str, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    features = basic_features(pd.read_csv(DATASETS / file), target)
    X = features.drop(columns=target)
    y = features[target]
    # XGBoost requires class labels 0..k-1; the arena encodes them the same way.
    if not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y)
    return X, np.asarray(y)


DATA = {
    "binary_classification": ("Titanic-Dataset.csv", "Survived"),
    "multiclass_classification": ("iris.csv", "species"),
    "regression": ("ames_train.csv", "SalePrice"),
}


@pytest.fixture(scope="module", params=list(DATA))
def task(request: pytest.FixtureRequest) -> tuple[str, pd.DataFrame, np.ndarray]:
    X, y = _xy(*DATA[request.param])
    return request.param, X, y


def _predict(estimator: object, X: pd.DataFrame, task_type: str) -> np.ndarray:
    if task_type == "regression":
        return estimator.predict(X)
    return estimator.predict_proba(X)


# --- The lists -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task_type", "ids"),
    [
        ("binary_classification", ["dummy", "logreg", "rf", "hgb", "xgb", "lgbm"]),
        ("multiclass_classification", ["dummy", "logreg", "rf", "hgb", "xgb", "lgbm"]),
        ("regression", ["dummy", "ridge", "rf", "hgb", "xgb", "lgbm"]),
    ],
)
def test_six_candidates_dummy_first(task_type: str, ids: list[str]) -> None:
    assert [c.model_id for c in candidates(task_type)] == ids


@pytest.mark.parametrize("bad", ["unsupported", "clustering", ""])
def test_no_candidates_for_unsupported_tasks(bad: str) -> None:
    with pytest.raises(ValueError, match="no arena candidates"):
        candidates(bad)


def test_every_call_builds_fresh_estimators() -> None:
    first, second = candidates("regression"), candidates("regression")
    assert all(a.estimator is not b.estimator for a, b in zip(first, second, strict=True))


def test_primary_metric_per_task_type() -> None:
    assert PRIMARY_METRIC == {
        "binary_classification": "roc_auc",
        "multiclass_classification": "f1_macro",
        "regression": "rmse",
    }


def test_only_linear_models_are_scaled() -> None:
    for task_type in DATA:
        for c in candidates(task_type):
            scaled = isinstance(c.estimator, Pipeline) and isinstance(
                c.estimator.steps[0][1], StandardScaler
            )
            assert scaled == (c.model_id in {"logreg", "ridge"}), c.model_id


# --- Every model fits real data, reproducibly -----------------------------------------


def test_every_candidate_fits_and_is_deterministic(task: tuple) -> None:
    """Every model trains on real features -- and twice gives the same answer to the last bit.

    One test, not two: a separate "it fits" test would repeat half of this one's work.
    """
    task_type, X, y = task
    for first, second in zip(candidates(task_type), candidates(task_type), strict=True):
        a = _predict(first.estimator.fit(X, y), X, task_type)
        b = _predict(second.estimator.fit(X, y), X, task_type)
        assert len(a) == len(X) and np.isfinite(a).all(), first.model_id
        assert np.array_equal(a, b), f"{first.model_id} is not deterministic"
