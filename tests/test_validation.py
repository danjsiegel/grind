from __future__ import annotations

from pathlib import Path

import pytest

from grind.artifacts import ArtifactChecksumError, LocalArtifactStore
from grind.config import init_engine_workspace
from grind.engine.orchestrator import DoStageResponsePayload, MinimalOrchestrator
from grind.models import TaskSourceKind
from grind.providers import ModelInvocationResult
from grind.validation.runner import run_validation_commands
from grind.validation.safety import classify_command, normalize_shell_free_command


def test_normalize_shell_free_command_rejects_shell_metacharacters() -> None:
    with pytest.raises(ValueError):
        normalize_shell_free_command("uv run pytest tests -q && echo nope")


def test_normalize_shell_free_command_rejects_dollar_substitution() -> None:
    with pytest.raises(ValueError, match="shell metacharacters"):
        normalize_shell_free_command("echo $(id)")


def test_normalize_shell_free_command_rejects_backtick_substitution() -> None:
    with pytest.raises(ValueError, match="shell metacharacters"):
        normalize_shell_free_command("echo `id`")


def test_forbidden_command_is_classified_risky() -> None:
    assert classify_command(["git", "reset", "--hard"]) == "risky"


def test_classify_command_does_not_flag_force_with_lease_as_risky() -> None:
    # '--force-with-lease' starts with '--force' but is not a destructive forced push.
    assert classify_command(["git", "push", "--force-with-lease"]) == "safe"
    assert classify_command(["git", "push", "--force-with-lease=origin/main"]) == "safe"


def test_classify_command_still_flags_bare_force_push_as_risky() -> None:
    assert classify_command(["git", "push", "--force"]) == "risky"
    assert classify_command(["git", "push", "--force", "origin", "main"]) == "risky"


def test_run_validation_commands_enforces_timeout(tmp_path: Path) -> None:
    python_path = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    results = run_validation_commands(
        tmp_path,
        [[str(python_path), "-c", "import time; time.sleep(0.2)"]],
        timeout_seconds=0.05,
    )

    assert len(results) == 1
    assert results[0].timed_out is True
    assert results[0].returncode == 124


def test_artifact_store_uses_relative_paths_and_verifies_checksum(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact = store.write_text(
        run_id="run_1",
        artifact_id="artifact_1",
        artifact_type="note",
        content="hello\n",
    )

    assert not Path(artifact.path).is_absolute()
    assert artifact.path.startswith("run_1/")
    assert store.read_text(artifact) == "hello\n"

    store.resolve_path(artifact).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ArtifactChecksumError):
        store.read_text(artifact)


def test_risky_validation_command_is_held_before_execution(tmp_path: Path, monkeypatch) -> None:
    init_engine_workspace(tmp_path)
    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd, timeout_seconds=300: ModelInvocationResult(
            command=["fake-planner"],
            stdout='{"plan":"ship it"}',
            stderr="",
            returncode=0,
        ),
    )
    monkeypatch.setattr(
        MinimalOrchestrator,
        "_run_do_stage",
        lambda self, *, store, run, task, iteration: DoStageResponsePayload(
            touched_files=["README.md"],
            touched_symbols=[],
            validation_hints=[],
            claims_made=[],
            open_uncertainties=[],
            artifact_refs=[],
        ),
    )

    (tmp_path / ".grind" / "policy" / "project.yaml").write_text(
        """
schema_ver: "1"
validation_commands:
  - command: "git reset --hard"
    risk: risky
    timeout_seconds: 120
safe_paths:
  - "src/"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    orchestrator = MinimalOrchestrator(cwd=tmp_path)

    run_outcome = orchestrator.run(objective="risky validation", source_kind=TaskSourceKind.INLINE)
    resume_outcome = orchestrator.resume(run_id=run_outcome.run_id)

    assert resume_outcome.final_state.value == "awaiting_operator"
    status = orchestrator.hold_reason(run_id=run_outcome.run_id)
    assert status["hold_type"] == "risky_command"
    assert "git reset --hard" in status["hold_reason"]