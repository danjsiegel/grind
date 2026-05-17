from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class ValidationExecutionResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _display_command(argv: Sequence[str]) -> str:
    return " ".join(argv)


def run_validation_commands(
    cwd: Path,
    commands: list[Sequence[str]],
    *,
    stop_on_failure: bool = True,
    timeout_seconds: int | None = None,
) -> list[ValidationExecutionResult]:
    results: list[ValidationExecutionResult] = []
    for argv in commands:
        if not argv:
            raise ValueError("validation command cannot be empty")
        try:
            completed = subprocess.run(
                list(argv),
                cwd=str(cwd),
                shell=False,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            result = ValidationExecutionResult(
                command=_display_command(argv),
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as error:
            result = ValidationExecutionResult(
                command=_display_command(argv),
                returncode=124,
                stdout=error.stdout or "",
                stderr=(error.stderr or "") + f"validation command timed out after {timeout_seconds} seconds",
                timed_out=True,
            )
        results.append(result)
        if stop_on_failure and (result.returncode != 0 or result.timed_out):
            break
    return results