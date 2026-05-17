from __future__ import annotations

from pathlib import Path

from grind.validation.runner import run_validation_commands


def test_timeout_enforcement(tmp_path: Path) -> None:
    python_path = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    results = run_validation_commands(
        tmp_path,
        [[str(python_path), "-c", "import time; time.sleep(0.2)"]],
        timeout_seconds=0.05,
    )

    assert len(results) == 1
    assert results[0].timed_out is True
    assert results[0].returncode == 124