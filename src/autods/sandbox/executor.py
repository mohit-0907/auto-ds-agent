"""Run LLM-generated feature engineering code in an isolated subprocess.

Invariant 3: generated code is untrusted input. It runs only here, in a child
process, with a timeout, a stripped environment, and (on POSIX) a memory cap,
and its output is validated afterwards by the parent -- never by the child.

This is isolation, not security. A subprocess shares the filesystem and network
with the host; real security needs a container. The static checks below stop
accidental harm, not a determined adversary.

Flow: static check (no process) -> run on a SANDBOX_SAMPLE_ROWS sample ->
validate -> run on the full frame -> validate. The sample run makes most
failures cost a second instead of a full-data run.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import structlog
from pydantic import BaseModel, computed_field

from autods.config import (
    RANDOM_STATE,
    SANDBOX_ALLOWED_IMPORTS,
    SANDBOX_BLOCKED_CALLS,
    SANDBOX_MAX_MEMORY_MB,
    SANDBOX_OUTPUT_TAIL_CHARS,
    SANDBOX_SAMPLE_ROWS,
    SANDBOX_TIMEOUT_S,
)
from autods.sandbox import _runner
from autods.sandbox.contract import ENTRYPOINT, ValidationResult, validate_output

log = structlog.get_logger(__name__)

RUNNER_PATH = Path(_runner.__file__)
# Environment variables the child keeps. Everything else -- API keys first of
# all -- is dropped. SYSTEMROOT is required for Python to start on Windows.
_ENV_PASSTHROUGH = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE")

Status = Literal["ok", "rejected", "runtime_error", "timeout", "invalid_output"]
Stage = Literal["static", "sample", "full"]


class ExecutionResult(BaseModel):
    """Structured outcome of one execute_fe_code call.

    `message` is a short, self-contained summary written for the codegen LLM's
    retry prompt; `traceback` and `validation` carry the detail.
    """

    status: Status
    stage: Stage
    message: str
    traceback: str | None = None
    stdout: str = ""
    stderr: str = ""
    validation: ValidationResult | None = None
    duration_s: float = 0.0
    output_path: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """True when the code ran on the full frame and passed validation."""
        return self.status == "ok"


def static_check(code: str) -> list[str]:
    """Reject code that cannot or should not run, without starting a process.

    Args:
        code: The generated source.

    Returns:
        Every problem found (empty when the code may run). All of them, not
        the first, so a retry can fix them together.
    """
    try:
        tree = ast.parse(code, filename="generated_fe.py")
    except SyntaxError as exc:
        return [f"SyntaxError on line {exc.lineno}: {exc.msg}"]

    problems: list[str] = []
    if not any(isinstance(node, ast.FunctionDef) and node.name == ENTRYPOINT for node in tree.body):
        problems.append(f"No top-level function named {ENTRYPOINT!r} is defined.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append("Relative imports are not allowed.")
                continue
            modules = [node.module or ""]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in SANDBOX_BLOCKED_CALLS
        ):
            problems.append(f"Calling {node.func.id}() is not allowed (line {node.lineno}).")
            continue
        else:
            continue
        for module in modules:
            if module.split(".")[0] not in SANDBOX_ALLOWED_IMPORTS:
                problems.append(
                    f"Import of {module!r} is not allowed (line {node.lineno}); "
                    f"allowed: {', '.join(SANDBOX_ALLOWED_IMPORTS)}."
                )
    return problems


def _child_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ}
    env["PYTHONHASHSEED"] = "0"  # Invariant 8: hash-dependent code behaves the same each run.
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _limit_memory() -> None:  # pragma: no cover -- runs inside the POSIX child
    import resource

    limit = SANDBOX_MAX_MEMORY_MB * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _tail(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-SANDBOX_OUTPUT_TAIL_CHARS:]


def _run_stage(
    stage: Stage, frame: pd.DataFrame, target_column: str, work_dir: Path, code_path: Path
) -> ExecutionResult:
    """Run the code on one frame in a child process, then validate its output."""
    in_path = work_dir / f"{stage}_input.parquet"
    out_path = work_dir / f"{stage}_features.parquet"
    out_path.unlink(missing_ok=True)  # Never validate a stale file from an earlier attempt.
    frame.to_parquet(in_path)

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(RUNNER_PATH),
                str(code_path),
                str(in_path),
                str(out_path),
                target_column,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SANDBOX_TIMEOUT_S,
            cwd=work_dir,
            env=_child_env(),
            preexec_fn=_limit_memory if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(
            status="timeout",
            stage=stage,
            message=(
                f"Timed out after {SANDBOX_TIMEOUT_S}s on the {stage} run ({len(frame)} rows). "
                "Look for infinite loops or row-by-row Python loops; use vectorized pandas."
            ),
            stdout=_tail(exc.stdout),
            stderr=_tail(exc.stderr),
            duration_s=time.perf_counter() - start,
        )
    duration = time.perf_counter() - start
    stdout, stderr = _tail(proc.stdout), _tail(proc.stderr)

    if proc.returncode == 3 and _runner.CONTRACT_PREFIX in stderr:
        message = stderr.split(_runner.CONTRACT_PREFIX, 1)[1].strip()
        return ExecutionResult(
            status="invalid_output",
            stage=stage,
            message=message,
            stdout=stdout,
            stderr=stderr,
            validation=ValidationResult(errors=[message]),
            duration_s=duration,
        )
    if proc.returncode != 0:
        last_line = stderr.strip().splitlines()[-1] if stderr.strip() else ""
        message = (
            f"{ENTRYPOINT} raised on the {stage} run ({len(frame)} rows): {last_line}"
            if last_line
            else f"The {stage} run exited with code {proc.returncode} and no traceback "
            "(possibly killed for exceeding the memory limit)."
        )
        return ExecutionResult(
            status="runtime_error",
            stage=stage,
            message=message,
            traceback=stderr or None,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
        )

    # The parent validates; the child's exit code 0 only means "it returned".
    validation = validate_output(frame, pd.read_parquet(out_path), target_column)
    if not validation.ok:
        return ExecutionResult(
            status="invalid_output",
            stage=stage,
            message=f"Output of the {stage} run breaks the contract: "
            + " ".join(validation.errors),
            stdout=stdout,
            stderr=stderr,
            validation=validation,
            duration_s=duration,
        )
    return ExecutionResult(
        status="ok",
        stage=stage,
        message=f"{stage} run passed validation.",
        stdout=stdout,
        stderr=stderr,
        validation=validation,
        duration_s=duration,
        output_path=str(out_path),
    )


def execute_fe_code(
    code: str, df: pd.DataFrame, target_column: str, work_dir: Path
) -> ExecutionResult:
    """Statically check, then run generated FE code on a sample and on the full frame.

    Args:
        code: Source defining engineer_features(df, target_column).
        df: The prepared frame (after prepare_target). Not modified; the child
            gets its own copy through a Parquet file.
        target_column: The target column name.
        work_dir: Directory for this attempt's files. Created if missing.

    Returns:
        The first failing stage's result, or the full run's result with
        output_path set to the validated features Parquet file.
    """
    start = time.perf_counter()
    problems = static_check(code)
    if problems:
        log.info("executor.rejected", problems=problems)
        return ExecutionResult(
            status="rejected",
            stage="static",
            message="Code rejected before running: " + " ".join(problems),
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    code_path = work_dir / "generated_fe.py"
    code_path.write_text(code, encoding="utf-8")

    # Random, not head(): the first rows of a file are often unrepresentative
    # (sorted by date, grouped by class). Fixed seed keeps the sample stable.
    if len(df) > SANDBOX_SAMPLE_ROWS:
        sample = df.sample(n=SANDBOX_SAMPLE_ROWS, random_state=RANDOM_STATE)
        result = _run_stage("sample", sample, target_column, work_dir, code_path)
        log.info(
            "executor.stage", stage="sample", status=result.status, duration_s=result.duration_s
        )
        if not result.ok:
            return result

    result = _run_stage("full", df, target_column, work_dir, code_path)
    result = result.model_copy(update={"duration_s": time.perf_counter() - start})
    log.info("executor.stage", stage="full", status=result.status, duration_s=result.duration_s)
    return result
