"""Tests for run artifacts: the saved best model, fe_code.py, and run.json."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from autods.arena.scoring import prepare_xy
from autods.config import PROJECT_ROOT
from autods.nodes.artifacts import artifacts_node, finalize_node
from autods.nodes.fallback_fe import basic_features
from autods.state import AgentState, ModelResult, TaskDecision
from tests.helpers import make_state

DATASETS = PROJECT_ROOT / "datasets"
GENERATED = "def engineer_features(df, target_column):\n    return df\n"


def _state(
    tmp_path: Path, file: str, target: str, task_type: str, best: str, fallback: bool = False
) -> AgentState:
    sub = "fallback" if fallback else "attempt_1"
    features_dir = tmp_path / "fe" / sub
    features_dir.mkdir(parents=True)
    path = features_dir / "features.parquet"
    basic_features(pd.read_csv(DATASETS / file), target).to_parquet(path)
    return make_state(
        dataset_path=str(DATASETS / file),
        run_dir=str(tmp_path),
        processed_data_path=str(path),
        task_decision=TaskDecision(
            task_type=task_type, target_column=target, reasoning="r", confidence=0.9
        ),
        best_model_id=best,
        fe_code_history=[] if fallback else [GENERATED],
        fe_explanations=[] if fallback else ["Kept the frame as is."],
    )


# --- best_pipeline.joblib -------------------------------------------------------------


def test_bundle_holds_a_usable_model_and_its_metadata(tmp_path: Path) -> None:
    state = _state(tmp_path, "Titanic-Dataset.csv", "Survived", "binary_classification", "logreg")
    assert artifacts_node(state) == {}

    bundle = joblib.load(tmp_path / "best_pipeline.joblib")
    assert bundle["format_version"] == 1
    assert bundle["model_id"] == "logreg"
    assert bundle["trained_rows"] == 891  # all rows, not one CV fold's 80%
    assert bundle["fe_source"] == "generated (attempt 1)"

    # Usable: align features to the saved columns and predict.
    X, _, _, _ = prepare_xy(
        pd.read_parquet(state["processed_data_path"]), "Survived", "binary_classification"
    )
    predictions = bundle["model"].predict(X[bundle["feature_columns"]])
    assert set(np.unique(predictions)) <= {0, 1}


def test_class_names_decode_the_models_numbers(tmp_path: Path) -> None:
    state = _state(tmp_path, "iris.csv", "species", "multiclass_classification", "logreg")
    artifacts_node(state)
    bundle = joblib.load(tmp_path / "best_pipeline.joblib")

    assert bundle["class_names"] == ["setosa", "versicolor", "virginica"]
    X = pd.read_parquet(state["processed_data_path"]).drop(columns="species")
    first = bundle["model"].predict(X[bundle["feature_columns"]].iloc[:1])[0]
    assert bundle["class_names"][first] == "setosa"  # row 0 of iris is a setosa


def test_refit_is_deterministic(tmp_path: Path) -> None:
    state = _state(tmp_path, "Titanic-Dataset.csv", "Survived", "binary_classification", "hgb")
    artifacts_node(state)
    first = joblib.load(tmp_path / "best_pipeline.joblib")["model"]
    artifacts_node(state)
    second = joblib.load(tmp_path / "best_pipeline.joblib")["model"]

    X = pd.read_parquet(state["processed_data_path"]).drop(columns="Survived")
    assert np.array_equal(first.predict_proba(X), second.predict_proba(X))


def test_failed_refit_degrades_to_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _state(tmp_path, "Titanic-Dataset.csv", "Survived", "binary_classification", "logreg")

    class _Broken:
        def fit(self, X: object, y: object) -> None:
            raise ValueError("singular matrix")

    class _Candidate:
        model_id, model_name, estimator = "logreg", "Logistic Regression", _Broken()

    monkeypatch.setattr("autods.nodes.artifacts.candidates", lambda task_type: [_Candidate()])
    update = artifacts_node(state)

    assert "Could not retrain Logistic Regression" in update["warnings"][0]
    assert not (tmp_path / "best_pipeline.joblib").exists()


def test_nothing_to_save_without_a_best_model(tmp_path: Path) -> None:
    assert artifacts_node(make_state(run_dir=str(tmp_path))) == {}
    assert list(tmp_path.iterdir()) == []


# --- fe_code.py ------------------------------------------------------------------------


def test_fe_code_is_the_generated_code_with_a_header(tmp_path: Path) -> None:
    artifacts_node(_state(tmp_path, "iris.csv", "species", "multiclass_classification", "logreg"))
    code = (tmp_path / "fe_code.py").read_text(encoding="utf-8")

    assert code.startswith("# Feature engineering code for iris.csv")
    assert code.endswith(GENERATED)
    compile(code, "fe_code.py", "exec")  # still valid Python with the header


def test_fallback_runs_get_an_explanation_instead_of_code(tmp_path: Path) -> None:
    state = _state(
        tmp_path, "iris.csv", "species", "multiclass_classification", "logreg", fallback=True
    )
    artifacts_node(state)
    code = (tmp_path / "fe_code.py").read_text(encoding="utf-8")

    assert "deterministic fallback recipe" in code
    assert joblib.load(tmp_path / "best_pipeline.joblib")["fe_source"] == "fallback"


# --- run.json --------------------------------------------------------------------------


def test_run_json_records_decisions_and_cost_without_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-very-secret")
    state = _state(tmp_path, "iris.csv", "species", "multiclass_classification", "logreg")
    state["leaderboard"] = [
        ModelResult(model_id="logreg", model_name="LR", status="success", primary_metric="f1_macro")
    ]
    state["token_cost_usd"] = 0.0423
    state["warnings"] = ["Something worth knowing."]
    artifacts_node(state)
    finalize_node(state)

    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["task_decision"]["target_column"] == "species"
    assert record["best_model_id"] == "logreg"
    assert record["token_cost_usd"] == 0.0423
    assert record["feature_engineering"]["final_source"] == "generated (attempt 1)"
    assert record["feature_engineering"]["attempts"][0]["outcome"] == "passed validation"
    assert {"best_pipeline.joblib", "fe_code.py"} <= set(record["artifacts"])
    assert record["config"]["random_state"] == 42
    assert "sk-test-very-secret" not in text


def test_run_json_is_written_for_a_failed_run_too(tmp_path: Path) -> None:
    state = make_state(
        dataset_path="missing.csv", run_dir=str(tmp_path), warnings=["Profiling failed."]
    )
    finalize_node(state)
    record = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))

    assert record["task_decision"] is None
    assert record["leaderboard"] == []
    assert record["warnings"] == ["Profiling failed."]


def test_finalize_without_run_dir_fails_loudly() -> None:
    with pytest.raises(ValueError, match="run_dir"):
        finalize_node(make_state())
