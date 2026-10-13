"""Tests for the arena node: real cross-validation on real features."""

from pathlib import Path

import pandas as pd
import pytest

from autods.arena.candidates import candidates
from autods.config import PROJECT_ROOT
from autods.nodes.arena import arena_node
from autods.nodes.fallback_fe import basic_features
from autods.state import AgentState, TaskDecision
from tests.helpers import make_state

DATASETS = PROJECT_ROOT / "datasets"


def _features(tmp_path: Path, file: str, target: str) -> Path:
    path = tmp_path / "features.parquet"
    basic_features(pd.read_csv(DATASETS / file), target).to_parquet(path)
    return path


def _state(tmp_path: Path, file: str, target: str, task_type: str) -> AgentState:
    decision = TaskDecision(
        task_type=task_type, target_column=target, reasoning="r", confidence=0.9
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return make_state(
        processed_data_path=str(_features(tmp_path, file, target)),
        task_decision=decision,
        run_dir=str(run_dir),
    )


@pytest.fixture
def quick_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy + the linear model only: same code path, a fraction of the time."""
    monkeypatch.setattr(
        "autods.nodes.arena.candidates",
        lambda task_type: [
            c for c in candidates(task_type) if c.model_id in {"dummy", "logreg", "ridge"}
        ],
    )


def test_full_arena_on_titanic(tmp_path: Path) -> None:
    update = arena_node(
        _state(tmp_path, "Titanic-Dataset.csv", "Survived", "binary_classification")
    )

    leaderboard = update["leaderboard"]
    assert {r.model_id for r in leaderboard} == {"dummy", "logreg", "rf", "hgb", "xgb", "lgbm"}
    assert all(r.status == "success" for r in leaderboard)
    assert leaderboard[-1].model_id == "dummy"  # ranked last: everything beats chance
    assert update["best_model_id"] in {"logreg", "hgb", "lgbm", "xgb", "rf"}

    csv = pd.read_csv(tmp_path / "run" / "leaderboard.csv")
    assert csv["is_best"].sum() == 1
    assert csv.loc[csv["is_best"], "model_id"].item() == update["best_model_id"]
    assert {"roc_auc", "roc_auc_std", "training_time_s", "predict_time_s"} <= set(csv.columns)


def comparable(update: dict) -> tuple:
    """Everything the arena decides, minus wall-clock timings (which always vary)."""
    rows = [
        (r.model_id, r.status, r.metrics, r.metrics_std, r.fold_scores)
        for r in update["leaderboard"]
    ]
    return update["best_model_id"], rows


def test_dod_same_features_give_an_identical_leaderboard(tmp_path: Path) -> None:
    """Phase 5 DoD, offline: fixed seeds make the full 6-model arena reproducible."""
    state = _state(tmp_path, "Titanic-Dataset.csv", "Survived", "binary_classification")
    first, second = arena_node(state), arena_node(state)

    assert comparable(first) == comparable(second)
    assert first["best_model_id"] != "dummy"


def test_regression_leaderboard_is_ranked_lowest_rmse_first(
    tmp_path: Path, quick_arena: None
) -> None:
    update = arena_node(_state(tmp_path, "ames_train.csv", "SalePrice", "regression"))

    assert [r.model_id for r in update["leaderboard"]] == ["ridge", "dummy"]
    assert update["best_model_id"] == "ridge"


def test_multiclass_text_labels(tmp_path: Path, quick_arena: None) -> None:
    update = arena_node(_state(tmp_path, "iris.csv", "species", "multiclass_classification"))

    assert update["best_model_id"] == "logreg"
    assert update["leaderboard"][0].primary_metric == "f1_macro"


def test_missing_targets_are_dropped_and_reported(tmp_path: Path, quick_arena: None) -> None:
    state = _state(tmp_path, "Titanic-Dataset.csv", "Survived", "binary_classification")
    df = pd.read_csv(DATASETS / "Titanic-Dataset.csv")
    df.loc[:9, "Survived"] = None
    # Overwrite the clean features _state just wrote -- after it, not before.
    basic_features(df, "Survived").to_parquet(state["processed_data_path"])

    update = arena_node(state)
    assert "Dropped 10 rows with a missing target" in update["warnings"][0]


def test_a_class_with_one_example_degrades(tmp_path: Path, quick_arena: None) -> None:
    state = _state(tmp_path, "iris.csv", "species", "multiclass_classification")
    df = pd.read_csv(DATASETS / "iris.csv")
    df.loc[0, "species"] = "unicorn"
    basic_features(df, "species").to_parquet(state["processed_data_path"])

    update = arena_node(state)
    assert update["best_model_id"] is None
    assert update["leaderboard"] == []
    assert "only 1 example" in update["warnings"][-1]


def test_nothing_to_train_on_is_a_no_op(tmp_path: Path) -> None:
    assert arena_node(make_state(run_dir=str(tmp_path))) == {}


def test_missing_run_dir_fails_loudly(tmp_path: Path) -> None:
    state = _state(tmp_path, "iris.csv", "species", "multiclass_classification")
    state["run_dir"] = ""
    with pytest.raises(ValueError, match="run_dir"):
        arena_node(state)
