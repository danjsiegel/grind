from __future__ import annotations

from pathlib import Path

import pytest

from grind.config import init_engine_workspace
from grind.engine.difference_surface_builder import build_difference_surface
from grind.engine.orchestrator import DoStageResponsePayload, MinimalOrchestrator
from grind.models import OperatorStatus, Run, RunState, Task, TaskSourceKind
from grind.policy import PolicyLoader
from grind.providers import ModelInvocationResult
from grind.validation import ValidationExecutionResult


def test_policy_loader_normalizes_template_commands() -> None:
    template_path = Path(__file__).resolve().parents[1] / "templates" / "policy" / "project.yaml"

    policy_pack = PolicyLoader.load(template_path)

    assert policy_pack.schema_ver == "1"
    assert policy_pack.validation_commands[0].argv == ["uv", "run", "pytest", "tests", "-q"]


def test_policy_loader_accepts_constructor_load_style() -> None:
    template_path = Path(__file__).resolve().parents[1] / "templates" / "policy" / "project.yaml"

    policy_pack = PolicyLoader(template_path).load()

    assert policy_pack.schema_ver == "1"


def test_scope_rules_exclude_paths_from_difference_surface(tmp_path: Path) -> None:
    (tmp_path / "included.txt").write_text("keep\n", encoding="utf-8")
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated" / "note.txt").write_text("exclude\n", encoding="utf-8")

    run = Run(
        run_id="run_scope_rules",
        repo_path=str(tmp_path),
        policy_pack_path=str(tmp_path / ".grind" / "policy"),
        policy_schema_ver="1",
        requested_objective="scope rule test",
        state=RunState.AWAITING_VALIDATION,
        operator_status=OperatorStatus.NONE,
    )
    task = Task(
        task_id="task_scope_rules",
        run_id=run.run_id,
        sequence=0,
        source_kind=TaskSourceKind.INLINE,
        raw_input="scope rule test",
    )

    surface = build_difference_surface(
        cwd=tmp_path,
        run=run,
        task=task,
        iteration=1,
        observed_delta={"reported_touched_files": []},
        validation_results=[],
        open_findings=[],
        baseline_snapshot_path=None,
        stop_on_failure=True,
        scope_excludes=["generated/"],
    )

    assert "included.txt" in surface.authoritative_touched_files
    assert "generated/note.txt" not in surface.authoritative_touched_files


def test_missing_engine_config_with_valid_policy_pack_starts(tmp_path: Path, monkeypatch) -> None:
    policy_dir = tmp_path / ".grind" / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "project.yaml").write_text(
        """
schema_ver: \"1\"
validation_commands:
  - command: \"uv run pytest tests -q\"
    risk: safe
    timeout_seconds: 120
safe_paths:
  - \"src/\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner"],
            stdout='{"plan":"ship it"}',
            stderr="",
            returncode=0,
        ),
    )

    orchestrator = MinimalOrchestrator(cwd=tmp_path)
    outcome = orchestrator.run(objective="no config", source_kind=TaskSourceKind.INLINE)

    assert outcome.final_state.value == "awaiting_operator"


def test_engine_prefers_policy_pack_validation_commands(tmp_path: Path, monkeypatch) -> None:
    init_engine_workspace(tmp_path)
    policy_dir = tmp_path / ".grind" / "policy"
    (policy_dir / "project.yaml").write_text(
        """
schema_ver: \"1\"
validation_commands:
  - command: \"python -c 'print(123)'\"
    risk: safe
    timeout_seconds: 10
safe_paths:
  - \"src/\"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
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
    def fake_validation_runner(cwd, commands, *, stop_on_failure, timeout_seconds):
        captured["commands"] = [list(command) for command in commands]
        return [ValidationExecutionResult(command="python -c print(123)", returncode=0, stdout="", stderr="")]

    monkeypatch.setattr("grind.engine.orchestrator.run_validation_commands", fake_validation_runner)
    monkeypatch.setattr(
        MinimalOrchestrator,
        "_run_semantic_audit_stage",
        lambda self, *, store, run, task, iteration, observed_delta, validation_results: (_ for _ in ()).throw(RuntimeError("stop after validation selection")),
    )

    orchestrator = MinimalOrchestrator(cwd=tmp_path)
    run_outcome = orchestrator.run(objective="policy pack", source_kind=TaskSourceKind.INLINE)

    with pytest.raises(RuntimeError, match="stop after validation selection"):
        orchestrator.resume(run_id=run_outcome.run_id)

    assert captured["commands"] == [["python", "-c", "print(123)"]]


def test_explicit_missing_policy_pack_fails_actionably(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="policy pack not found"):
        MinimalOrchestrator(cwd=tmp_path, policy_pack_path=tmp_path / "missing-policy")