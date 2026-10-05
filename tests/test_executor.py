"""Tests for the sandbox executor. These start real subprocesses (~1s each on Windows)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autods.config import SANDBOX_SAMPLE_ROWS
from autods.sandbox.executor import _child_env, execute_fe_code, static_check

GOOD_CODE = """
import pandas as pd

def engineer_features(df, target_column):
    out = df.copy()
    out["age"] = out["age"].fillna(out["age"].median())
    out["is_male"] = out["sex"].eq("male")
    return out.drop(columns=["sex"])
"""


def _frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.uniform(1, 80, n).round(1),
            "sex": rng.choice(["male", "female"], n),
            "survived": rng.integers(0, 2, n),
        }
    )
    df.loc[df.index % 9 == 0, "age"] = np.nan
    return df


@pytest.fixture
def small() -> pd.DataFrame:
    return _frame(200)  # below SANDBOX_SAMPLE_ROWS: full run only


@pytest.fixture
def large() -> pd.DataFrame:
    return _frame(SANDBOX_SAMPLE_ROWS * 2)  # sample run, then full run


def _run(code: str, df: pd.DataFrame, tmp_path: Path):
    return execute_fe_code(code, df, "survived", tmp_path / "attempt")


# --- Static checks: no process started ----------------------------------------------


def test_syntax_error_is_rejected_with_line() -> None:
    assert static_check("def engineer_features(df, t)\n    return df") == [
        "SyntaxError on line 1: expected ':'"
    ]


def test_missing_entrypoint() -> None:
    assert "No top-level function" in static_check("def features(df, t):\n    return df")[0]


def test_nested_entrypoint_does_not_count() -> None:
    code = "class X:\n    def engineer_features(self, df, t):\n        return df"
    assert "No top-level function" in static_check(code)[0]


@pytest.mark.parametrize(
    "line",
    ["import os", "import subprocess", "from shutil import rmtree", "import urllib.request"],
)
def test_disallowed_imports(line: str) -> None:
    problems = static_check(f"{line}\ndef engineer_features(df, t):\n    return df")
    assert "is not allowed (line 1)" in problems[0]


def test_relative_import_is_rejected() -> None:
    problems = static_check("from . import x\ndef engineer_features(df, t):\n    return df")
    assert "Relative imports" in problems[0]


@pytest.mark.parametrize("call", ["open('x.txt', 'w')", "eval('1')", "__import__('os')"])
def test_blocked_builtin_calls(call: str) -> None:
    problems = static_check(f"def engineer_features(df, t):\n    {call}\n    return df")
    assert "is not allowed (line 2)" in problems[0]


def test_allowed_imports_pass() -> None:
    code = "import numpy as np\nfrom datetime import date\nimport pandas.api.types\n" + GOOD_CODE
    assert static_check(code) == []


def test_all_static_problems_are_collected() -> None:
    code = "import os\nimport socket\ndef nope(df, t):\n    return df"
    assert len(static_check(code)) == 3


def test_rejected_code_reports_static_stage(small: pd.DataFrame, tmp_path: Path) -> None:
    result = _run("import os\n" + GOOD_CODE, small, tmp_path)

    assert (result.status, result.stage) == ("rejected", "static")
    assert not (tmp_path / "attempt").exists()  # no files, no process


# --- Execution ------------------------------------------------------------------------


def test_valid_code_runs_and_passes(small: pd.DataFrame, tmp_path: Path) -> None:
    result = _run(GOOD_CODE, small, tmp_path)

    assert result.ok, result.message
    assert result.stage == "full"
    features = pd.read_parquet(result.output_path)
    assert list(features.columns) == ["age", "survived", "is_male"]
    assert len(features) == len(small)


def test_large_frame_runs_sample_then_full(large: pd.DataFrame, tmp_path: Path) -> None:
    result = _run(GOOD_CODE, large, tmp_path)

    assert result.ok, result.message
    assert (tmp_path / "attempt" / "sample_features.parquet").exists()
    assert len(pd.read_parquet(result.output_path)) == len(large)


def test_runtime_error_returns_traceback_into_generated_code(
    small: pd.DataFrame, tmp_path: Path
) -> None:
    code = "def engineer_features(df, target_column):\n    return df['no_such_column']\n"
    result = _run(code, small, tmp_path)

    assert result.status == "runtime_error"
    assert "KeyError" in result.message
    assert 'File "generated_fe.py", line 2' in result.traceback


def test_error_on_sample_stops_before_full_run(large: pd.DataFrame, tmp_path: Path) -> None:
    code = "def engineer_features(df, target_column):\n    raise ValueError('boom')\n"
    result = _run(code, large, tmp_path)

    assert (result.status, result.stage) == ("runtime_error", "sample")
    assert not (tmp_path / "attempt" / "full_input.parquet").exists()


def test_error_only_at_full_scale_is_caught_at_full_stage(
    large: pd.DataFrame, tmp_path: Path
) -> None:
    code = (
        "def engineer_features(df, target_column):\n"
        f"    if len(df) > {SANDBOX_SAMPLE_ROWS}:\n"
        "        raise MemoryError('too big')\n"
        "    return df.drop(columns=['sex']).fillna(0)\n"
    )
    result = _run(code, large, tmp_path)

    assert (result.status, result.stage) == ("runtime_error", "full")


def test_contract_violation_is_invalid_output(small: pd.DataFrame, tmp_path: Path) -> None:
    """Runs fine, returns a DataFrame -- but drops rows, which only the parent's validation sees."""
    code = (
        "def engineer_features(df, target_column):\n    return df.drop(columns=['sex']).dropna()\n"
    )
    result = _run(code, small, tmp_path)

    assert result.status == "invalid_output"
    assert "Row count changed" in result.validation.errors[0]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("return df['age']", "must return a pandas DataFrame, got Series"),
        ("return df.rename(columns={'age': 0})", "Column names must be strings"),
    ],
)
def test_unrepresentable_output_is_reported_in_contract_terms(
    small: pd.DataFrame, tmp_path: Path, body: str, expected: str
) -> None:
    code = f"def engineer_features(df, target_column):\n    {body}\n"
    result = _run(code, small, tmp_path)

    assert result.status == "invalid_output"
    assert expected in result.message


def test_infinite_loop_is_killed(
    monkeypatch: pytest.MonkeyPatch, small: pd.DataFrame, tmp_path: Path
) -> None:
    monkeypatch.setattr("autods.sandbox.executor.SANDBOX_TIMEOUT_S", 3)
    code = "def engineer_features(df, target_column):\n    while True:\n        pass\n"
    result = _run(code, small, tmp_path)

    assert result.status == "timeout"
    assert "Timed out after 3s" in result.message
    assert result.duration_s < 30


def test_stdout_is_captured(small: pd.DataFrame, tmp_path: Path) -> None:
    code = GOOD_CODE.replace("out = df.copy()", "print('hello from FE')\n    out = df.copy()")
    assert "hello from FE" in _run(code, small, tmp_path).stdout


def test_input_frame_is_not_modified(small: pd.DataFrame, tmp_path: Path) -> None:
    before = small.copy()
    _run(GOOD_CODE, small, tmp_path)
    pd.testing.assert_frame_equal(small, before)


def test_ok_is_serialized(small: pd.DataFrame, tmp_path: Path) -> None:
    assert _run(GOOD_CODE, small, tmp_path).model_dump()["ok"] is True


# --- Isolation --------------------------------------------------------------------------


def test_child_environment_has_no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    env = _child_env()

    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env["PYTHONHASHSEED"] == "0"
