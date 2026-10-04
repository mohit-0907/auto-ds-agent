"""Tests for the profiler node: file loading, refusals, and the Phase 1 DoD.

The DoD tests run against the real CSVs in datasets/. They are small and
committed, so these run in CI like any other test.
"""

from pathlib import Path

import pandas as pd
import pytest

from autods.config import MAX_DETAILED_COLUMNS, MIN_ROWS, PROFILE_TOKEN_BUDGET, PROJECT_ROOT
from autods.nodes.profiler import estimate_tokens, load_dataset, profiler_node
from autods.state import AgentState, DataProfile
from tests.helpers import make_state

DATASETS = PROJECT_ROOT / "datasets"
TITANIC = DATASETS / "Titanic-Dataset.csv"
AMES = DATASETS / "ames_train.csv"


def _state(path: Path | str, target: str | None = None) -> AgentState:
    return make_state(dataset_path=str(path), target_column=target)


def _profile(update: dict) -> DataProfile:
    profile = update["data_profile"]
    assert isinstance(profile, DataProfile), update["warnings"]
    return profile


def _frame(n_rows: int, n_cols: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {f"c{i}": [(r * (i + 3)) % 7 for r in range(n_rows)] for i in range(n_cols)}
    )


# --- Phase 1 Definition of Done -------------------------------------------------


def test_dod_titanic() -> None:
    profile = _profile(profiler_node(_state(TITANIC)))

    assert (profile.n_rows, profile.n_cols) == (891, 12)
    assert profile.columns_truncated is False
    assert "Survived" in profile.target_candidates
    assert estimate_tokens(profile.model_dump_json()) <= PROFILE_TOKEN_BUDGET


def test_dod_ames() -> None:
    update = profiler_node(_state(AMES))
    profile = _profile(update)

    assert (profile.n_rows, profile.n_cols) == (1460, 81)
    assert profile.columns_truncated is True
    assert len(profile.columns) <= MAX_DETAILED_COLUMNS
    assert "SalePrice" in [c.name for c in profile.columns]
    assert sum(profile.dtype_summary.values()) == 81
    assert estimate_tokens(profile.model_dump_json()) <= PROFILE_TOKEN_BUDGET
    assert any("truncated" in w for w in update["warnings"])


def test_dod_synthetic_200_columns(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    _frame(n_rows=500, n_cols=200).to_csv(path, index=False)
    profile = _profile(profiler_node(_state(path)))

    assert profile.n_cols == 200
    assert profile.columns_truncated is True
    assert estimate_tokens(profile.model_dump_json()) <= PROFILE_TOKEN_BUDGET


def test_user_target_is_respected() -> None:
    profile = _profile(profiler_node(_state(AMES, target="SalePrice")))
    assert profile.target_candidates == ["SalePrice"]


# --- Refusals become warnings, not exceptions -----------------------------------


def test_bad_target_fails_fast_with_suggestion() -> None:
    update = profiler_node(_state(TITANIC, target="survived"))

    assert update["data_profile"] is None
    assert "Did you mean 'Survived'" in update["warnings"][0]


def test_missing_file(tmp_path: Path) -> None:
    update = profiler_node(_state(tmp_path / "nope.csv"))

    assert update["data_profile"] is None
    assert "not found" in update["warnings"][0]


def test_directory_is_not_a_dataset(tmp_path: Path) -> None:
    update = profiler_node(_state(tmp_path))
    assert update["data_profile"] is None


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"not really excel")
    update = profiler_node(_state(path))

    assert update["data_profile"] is None
    assert "unsupported file type" in update["warnings"][0]


def test_too_few_rows(tmp_path: Path) -> None:
    path = tmp_path / "tiny.csv"
    _frame(n_rows=MIN_ROWS - 1).to_csv(path, index=False)
    update = profiler_node(_state(path))

    assert update["data_profile"] is None
    assert f"at least {MIN_ROWS}" in update["warnings"][0]


def test_exactly_min_rows_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "edge.csv"
    _frame(n_rows=MIN_ROWS).to_csv(path, index=False)
    assert profiler_node(_state(path))["data_profile"] is not None


def test_empty_csv(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert profiler_node(_state(path))["data_profile"] is None


def test_oversized_file_is_refused_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patched where it is used: profiler imported the name, not the module.
    monkeypatch.setattr("autods.nodes.profiler.MAX_DATASET_MB", 0)
    path = tmp_path / "big.csv"
    _frame(n_rows=MIN_ROWS).to_csv(path, index=False)
    update = profiler_node(_state(path))

    assert update["data_profile"] is None
    assert "limit is 0 MB" in update["warnings"][0]


def test_corrupt_parquet(tmp_path: Path) -> None:
    path = tmp_path / "broken.parquet"
    path.write_text("this is not parquet")
    assert profiler_node(_state(path))["data_profile"] is None


# --- load_dataset ---------------------------------------------------------------


def test_parquet_round_trip(tmp_path: Path) -> None:
    df = _frame(n_rows=MIN_ROWS)
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    pd.testing.assert_frame_equal(load_dataset(path), df)


def test_extension_check_is_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "DATA.CSV"
    _frame(n_rows=MIN_ROWS).to_csv(path, index=False)
    assert len(load_dataset(path)) == MIN_ROWS
