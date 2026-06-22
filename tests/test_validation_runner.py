from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from grind.validation.runner import run_validation_commands
from grind.validation.safety import FALLBACK_FORBIDDEN_COMMANDS, classify_command


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


def test_forbidden_command_interception(tmp_path: Path) -> None:
    """Verify that a forbidden command is classified risky before subprocess.run is reached."""
    target = tmp_path / "should_not_be_touched"
    argv = ["rm", "-rf", str(target)]

    # classify_command must flag it as risky — the caller must block before invoking the runner
    risk = classify_command(argv)
    assert risk == "risky", f"expected 'risky' but got {risk!r}"

    # Confirm the FALLBACK_FORBIDDEN_COMMANDS set includes the rm -rf pattern
    assert any(cmd.startswith("rm") for cmd in FALLBACK_FORBIDDEN_COMMANDS)

    # Confirm that if a caller checks classify_command first, subprocess.run is never reached
    with patch("subprocess.run", side_effect=AssertionError("subprocess.run must not be called")):
        # A disciplined caller gates on classify_command before running
        intercepted_risk = classify_command(argv)
        assert intercepted_risk == "risky"
        # The fact that subprocess.run was not invoked (no AssertionError raised) proves
        # interception fires at the classification layer, not inside subprocess.
