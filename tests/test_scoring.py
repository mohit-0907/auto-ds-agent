"""Tests for arena scoring: preparing X/y, the shared splitter, one model's CV."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

from autods.arena.candidates import Candidate, candidates
from autods.arena.scoring import evaluate_candidate, make_cv, prepare_xy
from autods.config import CV_FOLDS, PROJECT_ROOT
from autods.nodes.fallback_fe import basic_features

DATASETS = PROJECT_ROOT / "datasets"


def _by_id(task_type: str, model_id: str) -> Candidate:
    return next(c for c in candidates(task_type) if c.model_id == model_id)


@pytest.fixture(scope="module")
def titanic() -> tuple[pd.DataFrame, np.ndarray]:
    features = basic_features(pd.read_csv(DATASETS / "Titanic-Dataset.csv"), "Survived")
    X, y, _, _ = prepare_xy(features, "Survived", "binary_classification")
    return X, y


# --- prepare_xy -------------------------------------------------------------------------


def test_text_classes_become_0_to_k_minus_1() -> None:
    features = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "species": ["b", "a", "c", "a"]})
    X, y, classes, notes = prepare_xy(features, "species", "multiclass_classification")

    assert list(X.columns) == ["x"]
    assert y.tolist() == [1, 0, 2, 0]
    assert classes == ["a", "b", "c"]  # classes[i] is the original label of code i
    assert notes == []


def test_rows_with_missing_target_are_dropped_with_a_note() -> None:
    features = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, None, 30.0, None]})
    X, y, classes, notes = prepare_xy(features, "y", "regression")

    assert y.tolist() == [10.0, 30.0]
    assert len(X) == 2
    assert classes == []
    assert notes == ["Dropped 2 rows with a missing target before training."]


# --- make_cv ---------------------------------------------------------------------------


def test_classification_folds_are_stratified() -> None:
    """Every test fold keeps Titanic's ~38% survival rate."""
    features = basic_features(pd.read_csv(DATASETS / "Titanic-Dataset.csv"), "Survived")
    X, y, _, _ = prepare_xy(features, "Survived", "binary_classification")
    cv = make_cv("binary_classification", y)

    assert isinstance(cv, StratifiedKFold)
    rates = [y[test].mean() for _, test in cv.split(X, y)]
    assert max(rates) - min(rates) < 0.01


def test_same_splits_every_time(titanic: tuple) -> None:
    """Fair contest: every model, every run, faces the same five splits."""
    X, y = titanic
    first = [test.tolist() for _, test in make_cv("binary_classification", y).split(X, y)]
    second = [test.tolist() for _, test in make_cv("binary_classification", y).split(X, y)]
    assert first == second


def test_rare_class_reduces_folds() -> None:
    y = np.array([0] * 20 + [1] * 20 + [2] * 3)
    assert make_cv("multiclass_classification", y).get_n_splits() == 3 < CV_FOLDS


def test_single_example_class_is_refused() -> None:
    y = np.array([0] * 20 + [1] * 20 + [2])
    with pytest.raises(ValueError, match="only 1 example"):
        make_cv("multiclass_classification", y)


# --- evaluate_candidate ------------------------------------------------------------------


def test_scorecard_for_a_real_model(titanic: tuple) -> None:
    X, y = titanic
    result = evaluate_candidate(
        _by_id("binary_classification", "logreg"),
        X,
        y,
        "binary_classification",
        make_cv("binary_classification", y),
    )

    assert result.status == "success"
    assert result.primary_metric == "roc_auc"
    assert set(result.metrics) == {"roc_auc", "accuracy", "precision", "recall"}
    assert set(result.metrics_std) == set(result.metrics)
    assert 0.8 < result.metrics["roc_auc"] < 0.9
    assert result.training_time_s > 0 and result.predict_time_s > 0


def test_dummy_scores_exactly_chance(titanic: tuple) -> None:
    X, y = titanic
    result = evaluate_candidate(
        _by_id("binary_classification", "dummy"),
        X,
        y,
        "binary_classification",
        make_cv("binary_classification", y),
    )

    assert result.metrics["roc_auc"] == 0.5
    assert result.metrics["precision"] == 0.0  # never predicts a survivor; no warning spam


def test_regression_errors_are_stored_as_positive_units() -> None:
    features = basic_features(pd.read_csv(DATASETS / "ames_train.csv"), "SalePrice")
    X, y, _, _ = prepare_xy(features, "SalePrice", "regression")
    result = evaluate_candidate(
        _by_id("regression", "dummy"), X, y, "regression", make_cv("regression", y)
    )

    # The mean-price dummy is off by about one standard deviation of price.
    assert result.metrics["rmse"] == pytest.approx(np.std(y), rel=0.1)
    assert result.metrics["mae"] > 0
    assert result.metrics["r2"] < 0.01  # predicting the mean explains nothing


def test_a_crashing_model_is_recorded_not_raised(titanic: tuple) -> None:
    class Broken(ClassifierMixin, BaseEstimator):
        """A valid scikit-learn estimator whose fit fails, like a real model on bad data."""

        def fit(self, X: object, y: object) -> "Broken":
            raise ValueError("Input contains infinity")

    X, y = titanic
    candidate = Candidate(model_id="broken", model_name="Broken", estimator=Broken())
    result = evaluate_candidate(
        candidate, X, y, "binary_classification", make_cv("binary_classification", y)
    )

    assert result.status == "failed"
    assert result.primary_metric == "roc_auc"  # still says what it was competing on
    assert "ValueError: Input contains infinity" in result.error
    assert result.metrics == {}


def test_scoring_is_reproducible(titanic: tuple) -> None:
    X, y = titanic
    runs = [
        evaluate_candidate(
            _by_id("binary_classification", "rf"),
            X,
            y,
            "binary_classification",
            make_cv("binary_classification", y),
        )
        for _ in range(2)
    ]
    assert runs[0].metrics == runs[1].metrics
    assert runs[0].metrics_std == runs[1].metrics_std
