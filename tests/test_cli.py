from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import yaml

from grind.cli import main
from grind.config import load_engine_config
from grind.models import OperatorActionType
from grind.providers import ModelInvocationResult
from grind.retrieval import LanceDBRetrievalService
from grind.retrieval.embeddings import EmbeddingBatchResult
from grind.state import open_state_store
from grind.validation import ValidationExecutionResult
from grind.verification.models import (
    ProbeKind,
    ProbeResult,
    ProbeStatus,
    ResolvedIdentity,
    VerificationOverallStatus,
    VerificationReport,
)
from grind.verification.service import DefaultBackendVerifier


def _model_response_for_prompt(prompt: str) -> ModelInvocationResult:
    if prompt.startswith("You are planning a grind task"):
        return ModelInvocationResult(
            command=["fake-planner"],
            stdout='{"plan":"ship it"}',
            stderr="",
            returncode=0,
        )
    if prompt.startswith("You are the do stage implementer"):
        return ModelInvocationResult(
            command=["fake-implementer", "do"],
            stdout=(
                '{"touched_files":["src/grind/engine/orchestrator.py"],'
                '"touched_symbols":["MinimalOrchestrator.resume"],'
                '"validation_hints":[{"command":"uv run pytest tests -q","reason":"workflow changed"}],'
                '"claims_made":[{"claim":"resume now dispatches the implementer","evidence":"doing stage executed"}],'
                '"open_uncertainties":[],"artifact_refs":[]}'
            ),
            stderr="",
            returncode=0,
        )
    if prompt.startswith("You are the act stage implementer"):
        finding_id = prompt.split("finding_id: ", 1)[1].split("\n", 1)[0]
        return ModelInvocationResult(
            command=["fake-implementer", "act"],
            stdout=(
                '{"triage":[{' 
                f'"finding_id":"{finding_id}","action":"fixed","justification":"Applied the requested fix.",'
                '"fix_artifact_id":null,"requested_validation_ids":[]}],'
                '"remaining_open_issues":[],"new_uncertainties":[]}'
            ),
            stderr="",
            returncode=0,
        )
    if prompt.startswith("You are the checker stage"):
        return ModelInvocationResult(
            command=["fake-checker"],
            stdout='{"summary":"no issues","findings":[]}',
            stderr="",
            returncode=0,
        )
    if prompt.startswith("You are the adjudicator stage"):
        return ModelInvocationResult(
            command=["fake-adjudicator"],
            stdout='{"summary":"no dispositions needed","dispositions":[]}',
            stderr="",
            returncode=0,
        )
    raise AssertionError(f"unexpected prompt: {prompt}")


def _validation_result(
    commands: list[list[str]],
    *,
    returncode: int,
    stdout: str,
    stderr: str,
) -> ValidationExecutionResult:
    return ValidationExecutionResult(
        command=" ".join(commands[0]),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_verify_backend_json_output(monkeypatch, capsys) -> None:
    report = VerificationReport(
        backend="github_cli",
        role="planner",
        resolved_identity=ResolvedIdentity(provider="github_cli", model="gpt-5.4"),
        overall_status=VerificationOverallStatus.PASSED,
        probes=[
            ProbeResult(
                probe_id="github_cli.auth_status",
                backend="github_cli",
                kind=ProbeKind.AUTH,
                required=True,
                status=ProbeStatus.PASSED,
                status_reason="active host entry present",
            )
        ],
    )

    monkeypatch.setattr(DefaultBackendVerifier, "verify", lambda self, request: report)

    exit_code = main([
        "verify-backend",
        "--backend",
        "github_cli",
        "--model",
        "gpt-5.4",
        "--json",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["overall_status"] == "passed"
    assert payload["probes"][0]["probe_id"] == "github_cli.auth_status"


def test_init_writes_default_engine_yaml(tmp_path: Path, capsys) -> None:
    exit_code = main(["init", "--cwd", str(tmp_path)])

    config_path = tmp_path / ".grind" / "engine.yaml"
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    content = config_path.read_text(encoding="utf-8")
    captured = capsys.readouterr()

    assert exit_code == 0
    assert config_path.exists()
    assert database_path.exists()
    assert "path: .grind/state/grind.duckdb" in content
    assert "root: .grind/artifacts" in content
    assert "enabled: true" in content
    assert "path: .grind/state/lancedb" in content
    assert "keep_last_terminal_runs:" in content
    assert "validation:" in content
    assert "schema version: 6" in captured.out


def test_init_refuses_to_overwrite_existing_config(tmp_path: Path, capsys) -> None:
    config_dir = tmp_path / ".grind"
    config_dir.mkdir(parents=True)
    (config_dir / "engine.yaml").write_text("models: {}\n", encoding="utf-8")

    exit_code = main(["init", "--cwd", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "configuration already exists" in captured.err


def test_run_uses_default_engine_config_when_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner", "--json"],
            stdout='{"plan":"review this objective"}',
            stderr="",
            returncode=0,
        ),
    )

    exit_code = main(["run", "ship it", "--cwd", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (tmp_path / ".grind" / "state" / "grind.duckdb").exists()
    assert "run_id:" in captured.out
    assert "status: awaiting_operator" in captured.out


def test_run_plain_text_surfaces_plan_review_guidance(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner", "--json"],
            stdout='{"plan":"review this objective"}',
            stderr="",
            returncode=0,
        ),
    )

    exit_code = main(["run", "ship it", "--cwd", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "status: awaiting_operator" in captured.out
    assert "hold: plan_review" in captured.out
    assert "reason: awaiting operator review of planner output" in captured.out
    assert "review plan: " in captured.out
    assert ".grind/artifacts/" in captured.out
    assert "approve: grind approve" in captured.out
    assert "reject: grind reject" in captured.out
    assert "resume after approval: grind resume" in captured.out


def test_run_json_includes_plan_review_hold_context(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner", "--json"],
            stdout='{"plan":"review this objective"}',
            stderr="",
            returncode=0,
        ),
    )

    exit_code = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["hold_type"] == "plan_review"
    assert payload["hold_reason"] == "awaiting operator review of planner output"
    assert payload["hold_context"]["plan_artifact_id"] is not None
    assert payload["hold_context"]["response_artifact_id"] is not None
    assert payload["review_paths"]["plan"].endswith(".md")
    assert payload["review_paths"]["planner_response"].endswith(".md")

    plan_review = Path(payload["review_paths"]["plan"]).read_text()
    planner_response = Path(payload["review_paths"]["planner_response"]).read_text()

    assert "# Plan Review" in plan_review
    assert "## Objective" in plan_review
    assert "## Proposed Plan" in plan_review
    assert "review this objective" in plan_review
    assert planner_response.strip() == "review this objective"


def test_run_json_strips_transcript_noise_from_plan_review(tmp_path: Path, monkeypatch, capsys) -> None:
    noisy_response = """
You are planning a grind task. Produce a concise actionable plan for the operator review stage.

Let me inspect the repo.
Here is the operator review plan:

## Actual Plan

1. Run the focused phase-2 tests.
2. If green, run the full suite.
3. Stop on the first failing validation.

## Operator Actions

- Approve to continue.

Now I have all the context I need.
Could you provide the next thinking that needs to be rewritten?
""".strip()

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner", "--json"],
            stdout=noisy_response,
            stderr="",
            returncode=0,
        ),
    )

    exit_code = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0

    plan_review = Path(payload["review_paths"]["plan"]).read_text()
    planner_response = Path(payload["review_paths"]["planner_response"]).read_text()

    assert "## Actual Plan" in plan_review
    assert "Run the focused phase-2 tests" in plan_review
    assert "Could you provide the next thinking" not in plan_review
    assert "Here is the operator review plan:" not in plan_review
    assert planner_response.strip().startswith("## Actual Plan")
    assert "Could you provide the next thinking" not in planner_response


def test_run_json_prefers_embedded_plan_payload_over_transcript(tmp_path: Path, monkeypatch, capsys) -> None:
    plan = """## Phase 2 Implementation Plan

### Step 1
Verify the Phase 2 seams against the live code.

```bash
uv run pytest tests/test_phase2_seams.py -q
```

### Step 2
Only continue if validation stays green.
"""
    noisy_response = (
        "# Kepler Context\n\n"
        "Use this skill when the task needs more than the always-on repo instructions.\n\n"
        "Let me inspect the live repo first.\n\n"
        "```json\n"
        f"{json.dumps({'plan': plan})}\n"
        "```\n\n"
        "Now I have enough context to keep thinking out loud.\n"
    )

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner", "--json"],
            stdout=noisy_response,
            stderr="",
            returncode=0,
        ),
    )

    exit_code = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0

    plan_review = Path(payload["review_paths"]["plan"]).read_text()
    planner_response = Path(payload["review_paths"]["planner_response"]).read_text()

    assert "## Phase 2 Implementation Plan" in plan_review
    assert "uv run pytest tests/test_phase2_seams.py -q" in plan_review
    assert "# Kepler Context" not in plan_review
    assert "Now I have enough context" not in plan_review
    assert planner_response.strip() == plan.strip()


def test_prune_removes_old_terminal_runs_and_artifacts(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner", "--json"],
            stdout='{"plan":"review this objective"}',
            stderr="",
            returncode=0,
        ),
    )

    run_ids: list[str] = []
    for _ in range(3):
        run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
        run_payload = json.loads(capsys.readouterr().out)
        assert run_exit == 0
        run_ids.append(run_payload["run_id"])
        abort_exit = main(["abort", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
        assert abort_exit == 0
        capsys.readouterr()

    prune_exit = main(["prune", "--cwd", str(tmp_path), "--keep-last", "1", "--json"])
    prune_payload = json.loads(capsys.readouterr().out)

    assert prune_exit == 0
    assert prune_payload["runs_pruned"] == 2
    assert len(prune_payload["run_ids"]) == 2
    assert prune_payload["retrieval_documents_pruned"] >= 0

    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    with open_state_store(database_path) as store:
        assert store.runs.get(run_ids[2]) is not None
        assert store.runs.get(run_ids[0]) is None
        assert store.runs.get(run_ids[1]) is None

    assert not (tmp_path / ".grind" / "artifacts" / run_ids[0]).exists()
    assert not (tmp_path / ".grind" / "artifacts" / run_ids[1]).exists()
    assert (tmp_path / ".grind" / "artifacts" / run_ids[2]).exists()


def test_auto_prune_keeps_only_configured_terminal_runs(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    config_path = tmp_path / ".grind" / "engine.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data["retention"]["mode"] = "auto"
    config_data["retention"]["keep_last_terminal_runs"] = 1
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner", "--json"],
            stdout='{"plan":"review this objective"}',
            stderr="",
            returncode=0,
        ),
    )

    run_ids: list[str] = []
    last_abort_payload: dict[str, object] | None = None
    for _ in range(3):
        run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
        run_payload = json.loads(capsys.readouterr().out)
        assert run_exit == 0
        run_ids.append(run_payload["run_id"])

        abort_exit = main(["abort", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
        last_abort_payload = json.loads(capsys.readouterr().out)
        assert abort_exit == 0

    assert last_abort_payload is not None
    assert last_abort_payload["auto_prune"]["runs_pruned"] == 1

    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    with open_state_store(database_path) as store:
        assert store.runs.get(run_ids[2]) is not None
        assert store.runs.get(run_ids[0]) is None
        assert store.runs.get(run_ids[1]) is None

    assert not (tmp_path / ".grind" / "artifacts" / run_ids[0]).exists()
    assert not (tmp_path / ".grind" / "artifacts" / run_ids[1]).exists()
    assert (tmp_path / ".grind" / "artifacts" / run_ids[2]).exists()


def test_prune_removes_retrieval_documents_for_pruned_runs(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner", "--json"],
            stdout='{"plan":"review this objective"}',
            stderr="",
            returncode=0,
        ),
    )

    run_ids: list[str] = []
    for _ in range(2):
        run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
        run_payload = json.loads(capsys.readouterr().out)
        assert run_exit == 0
        run_ids.append(run_payload["run_id"])

        abort_exit = main(["abort", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
        assert abort_exit == 0
        capsys.readouterr()

    config = load_engine_config(tmp_path / ".grind" / "engine.yaml")
    retrieval_service = LanceDBRetrievalService(cwd=tmp_path, config=config)

    before = retrieval_service.collection_stats(run_id=run_ids[0])
    assert before.get("run_summaries", 0) > 0

    prune_exit = main(["prune", "--cwd", str(tmp_path), "--keep-last", "1", "--json"])
    prune_payload = json.loads(capsys.readouterr().out)

    assert prune_exit == 0
    assert prune_payload["retrieval_documents_pruned"] > 0

    after_pruned = retrieval_service.collection_stats(run_id=run_ids[0])
    after_kept = retrieval_service.collection_stats(run_id=run_ids[1])

    assert after_pruned.get("run_summaries", 0) == 0
    assert after_kept.get("run_summaries", 0) > 0


def test_run_creates_stored_run_with_real_planner_adapter(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()
    objective_file = tmp_path / "handoff.md"
    objective_file.write_text("Kepler v1.1 handoff objective\n", encoding="utf-8")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner", "--json"],
            stdout='{"plan":"review this objective"}',
            stderr="",
            returncode=0,
        ),
    )

    exit_code = main([
        "run",
        "--cwd",
        str(tmp_path),
        "--objective-file",
        str(objective_file),
        "--json",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["final_state"] == "awaiting_operator"
    assert Path(payload["database_path"]).exists()

    with open_state_store(Path(payload["database_path"])) as store:
        run = store.runs.get(payload["run_id"])
        tasks = store.tasks.list_by_run(payload["run_id"])
        stages = store.stages.list_by_run(payload["run_id"])
        transitions = store.transitions.list_by_run(payload["run_id"])
        artifacts = store.artifacts.list_by_run(payload["run_id"])
        checkpoints = store.checkpoints.list_by_run(payload["run_id"])

    assert run is not None
    assert run.state.value == "awaiting_operator"
    assert len(tasks) == 1
    assert len(stages) == 1
    assert len(transitions) == 3
    assert len(artifacts) == 4
    assert len(checkpoints) == 1


def test_resume_status_findings_and_restore_checkpoint(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()
    tracked_file = tmp_path / "tracked.txt"
    tracked_file.write_text("baseline\n", encoding="utf-8")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    tracked_file.write_text("changed\n", encoding="utf-8")

    restore_exit = main([
        "restore-checkpoint",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    restore_payload = json.loads(capsys.readouterr().out)

    assert restore_exit == 0
    assert restore_payload["checkpoint_id"] == run_payload["checkpoint_id"]
    assert tracked_file.read_text(encoding="utf-8") == "baseline\n"

    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=1, stdout="", stderr="tests failed")
        ],
    )

    resume_exit = main([
        "resume",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    resume_payload = json.loads(capsys.readouterr().out)

    assert resume_exit == 0
    assert resume_payload["final_state"] == "awaiting_operator"

    status_exit = main(["status", "--run-id", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    status_payload = json.loads(capsys.readouterr().out)

    assert status_exit == 0
    assert status_payload["state"] == "awaiting_operator"
    assert status_payload["finding_count"] == 1
    assert status_payload["validation_count"] == 1

    with open_state_store(Path(run_payload["database_path"])) as store:
        stages = store.stages.list_by_run(run_payload["run_id"])

    assert [stage.stage_name for stage in stages] == ["planning", "doing", "validation", "semantic_auditing"]

    findings_exit = main(["findings", "--run-id", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    findings_payload = json.loads(capsys.readouterr().out)

    assert findings_exit == 0
    assert len(findings_payload) == 1
    assert findings_payload[0]["severity"] == "high"
    assert "Validation failed" in findings_payload[0]["title"]


def test_resume_can_complete_when_validation_passes(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    resume_payload = json.loads(capsys.readouterr().out)

    assert resume_exit == 0
    assert resume_payload["final_state"] == "completed"

    with open_state_store(Path(run_payload["database_path"])) as store:
        run = store.runs.get(run_payload["run_id"])
        validations = store.validations.list_by_run(run_payload["run_id"])
        actions = store.operator_actions.list_by_run(run_payload["run_id"])
        stages = store.stages.list_by_run(run_payload["run_id"])
        artifacts = store.artifacts.list_by_run(run_payload["run_id"])

    assert run is not None
    assert run.state.value == "completed"
    assert len(validations) == 1
    assert [stage.stage_name for stage in stages] == [
        "planning",
        "doing",
        "validation",
        "semantic_auditing",
        "checking",
        "adjudicating",
    ]
    assert "difference_surface" in [artifact.artifact_type for artifact in artifacts]
    assert "semantic_audit_report" in [artifact.artifact_type for artifact in artifacts]
    assert [action.action_type for action in actions] == [OperatorActionType.RESUME]


def test_retrieval_index_search_and_report(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "README.md").write_text("Workspace docs mention ship it architecture.\n", encoding="utf-8")
    (tmp_path / ".local" / "specs").mkdir(parents=True)
    (tmp_path / ".local" / "specs" / "product.md").write_text(
        "Specification guidance keeps ship it rollout deterministic.\n",
        encoding="utf-8",
    )
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    assert resume_exit == 0
    capsys.readouterr()

    inspect_exit = main([
        "inspect",
        "retrieval_queue",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    inspect_payload = json.loads(capsys.readouterr().out)

    assert inspect_exit == 0
    assert inspect_payload["retrieval_queue"]
    assert inspect_payload["retrieval_queue"][0]["queue_status"] == "completed"

    search_exit = main([
        "retrieval-search",
        "ship it architecture",
        "--collection",
        "docs_chunks",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    search_payload = json.loads(capsys.readouterr().out)

    assert search_exit == 0
    assert search_payload["results"]
    assert search_payload["results"][0]["collection"] == "docs_chunks"
    assert search_payload["collection_readiness"]["docs_chunks"]["state"] == "ready"

    spec_search_exit = main([
        "retrieval-search",
        "deterministic rollout guidance",
        "--collection",
        "spec_chunks",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    spec_search_payload = json.loads(capsys.readouterr().out)

    assert spec_search_exit == 0
    assert spec_search_payload["results"]
    assert spec_search_payload["results"][0]["collection"] == "spec_chunks"

    report_exit = main([
        "report",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    report_payload = json.loads(capsys.readouterr().out)

    assert report_exit == 0
    assert report_payload["retrieval"]["documents_by_collection"]
    assert report_payload["retrieval"]["documents_by_collection"]["docs_chunks"] > 0
    assert report_payload["retrieval"]["documents_by_collection"]["spec_chunks"] > 0
    assert report_payload["retrieval"]["readiness_by_collection"]["docs_chunks"]["state"] == "ready"
    assert report_payload["model_calls"]["total"] == 4

    index_exit = main([
        "retrieval-index",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    index_payload = json.loads(capsys.readouterr().out)

    assert index_exit == 0
    assert index_payload["indexed_collections"]["docs_chunks"] > 0


def test_retrieval_search_uses_hybrid_local_fallback_without_model(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "README.md").write_text("Kepler rollout review notes live here.\n", encoding="utf-8")
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(
        "grind.retrieval.embeddings.ProviderEmbeddingAdapter.embed_texts",
        lambda self, texts: EmbeddingBatchResult(
            vectors=[[0.0] * self.config.embedding_dimensions for _ in texts],
            backend="hash-fallback",
            model=self.config.embedding_model,
        ),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    assert resume_exit == 0
    capsys.readouterr()

    search_exit = main([
        "retrieval-search",
        "rollout review notes",
        "--collection",
        "docs_chunks",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    search_payload = json.loads(capsys.readouterr().out)

    assert search_exit == 0
    assert search_payload["search_strategy"] == "hybrid_hash_lexical"
    assert search_payload["results"]
    assert search_payload["results"][0]["collection"] == "docs_chunks"
    assert "rollout review notes" in search_payload["results"][0]["chunk_text"].lower()


def test_retrieval_search_falls_back_to_lexical_when_collection_embeddings_are_incompatible(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "README.md").write_text("Rollout notes for lexical fallback live here.\n", encoding="utf-8")
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    monkeypatch.setattr(
        "grind.retrieval.embeddings.ProviderEmbeddingAdapter.embed_texts",
        lambda self, texts: EmbeddingBatchResult(
            vectors=[[1.0] * self.config.embedding_dimensions for _ in texts],
            backend="openai",
            model=self.config.embedding_model,
        ),
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    assert resume_exit == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "grind.retrieval.embeddings.ProviderEmbeddingAdapter.embed_texts",
        lambda self, texts: EmbeddingBatchResult(
            vectors=[[0.0] * self.config.embedding_dimensions for _ in texts],
            backend="hash-fallback",
            model=self.config.embedding_model,
        ),
    )

    search_exit = main([
        "retrieval-search",
        "lexical fallback",
        "--collection",
        "docs_chunks",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    search_payload = json.loads(capsys.readouterr().out)

    assert search_exit == 0
    assert search_payload["collection_readiness"]["docs_chunks"]["state"] == "incompatible"
    assert search_payload["collection_readiness"]["docs_chunks"]["search_strategy"] == "lexical"
    assert search_payload["results"]
    assert "lexical fallback" in search_payload["results"][0]["chunk_text"].lower()


def test_resume_runs_act_stage_before_returning_to_hold_on_persistent_blockers(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    checker_calls = {"count": 0}

    def model_response(prompt: str) -> ModelInvocationResult:
        if prompt.startswith("You are planning a grind task"):
            return ModelInvocationResult(
                command=["fake-planner"],
                stdout='{"plan":"ship it"}',
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the do stage implementer"):
            return ModelInvocationResult(
                command=["fake-implementer", "do"],
                stdout=(
                    '{"touched_files":["tests/test_cli.py"],"touched_symbols":["test"],'
                    '"validation_hints":[],"claims_made":[],"open_uncertainties":[],"artifact_refs":[]}'
                ),
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the act stage implementer"):
            finding_id = prompt.split("finding_id: ", 1)[1].split("\n", 1)[0]
            return ModelInvocationResult(
                command=["fake-implementer", "act"],
                stdout=(
                    '{"triage":[{' 
                    f'"finding_id":"{finding_id}","action":"fixed","justification":"Added the requested coverage.",'
                    '"fix_artifact_id":null,"requested_validation_ids":[]}],'
                    '"remaining_open_issues":[],"new_uncertainties":[]}'
                ),
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the checker stage"):
            checker_calls["count"] += 1
            return ModelInvocationResult(
                command=["fake-checker"],
                stdout=(
                    '{"summary":"one blocker","findings":[{"title":"Missing regression coverage",'
                    '"severity":"high","confidence":"proven","category":"test_coverage",'
                    '"rationale":"Critical regression path has no test coverage.",'
                    '"exact_fix_action":"Add a regression test for the failing path.",'
                    '"file_path":"tests/test_cli.py","primary_symbol":"test_resume_can_complete_when_validation_passes",'
                    '"line_range":"1-10"}]}'
                ),
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the adjudicator stage"):
            stable_id = prompt.split("stable_id: ", 1)[1].split("\n", 1)[0]
            return ModelInvocationResult(
                command=["fake-adjudicator"],
                stdout=(
                    '{"summary":"blocker confirmed","dispositions":[{' 
                    f'"stable_id":"{stable_id}","decision":"open","justification":"Blocking until coverage is added."' 
                    '}]}'
                ),
                stderr="",
                returncode=0,
            )
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: model_response(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    resume_payload = json.loads(capsys.readouterr().out)

    assert resume_exit == 0
    assert resume_payload["final_state"] == "awaiting_operator"

    with open_state_store(Path(run_payload["database_path"])) as store:
        run = store.runs.get(run_payload["run_id"])
        findings = store.findings.list_by_run(run_payload["run_id"])
        dispositions = store.dispositions.list_by_run(run_payload["run_id"])
        checkpoints = store.checkpoints.list_by_run(run_payload["run_id"])
        stages = store.stages.list_by_run(run_payload["run_id"])

    assert run is not None
    assert run.state.value == "awaiting_operator"
    assert checker_calls["count"] == 2
    assert len(findings) == 2
    assert any(finding.status.value == "open" for finding in findings)
    assert all(finding.adjudicated is True for finding in findings)
    assert len(dispositions) == 2
    assert checkpoints[0].checkpoint_kind.value == "task_baseline"
    assert checkpoints[1].checkpoint_kind.value == "pre_act"
    assert [stage.stage_name for stage in stages] == [
        "planning",
        "doing",
        "validation",
        "semantic_auditing",
        "checking",
        "adjudicating",
        "acting",
        "validation",
        "semantic_auditing",
        "checking",
        "adjudicating",
    ]


def test_resume_persists_evidence_verification_for_checker_citations(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    def model_response(prompt: str) -> ModelInvocationResult:
        if prompt.startswith("You are planning a grind task"):
            return ModelInvocationResult(
                command=["fake-planner"],
                stdout='{"plan":"ship it"}',
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the do stage implementer"):
            return ModelInvocationResult(
                command=["fake-implementer", "do"],
                stdout=(
                    '{"touched_files":["src/grind/engine/orchestrator.py"],'
                    '"touched_symbols":["MinimalOrchestrator.resume"],'
                    '"validation_hints":[],"claims_made":[],"open_uncertainties":[],"artifact_refs":[]}'
                ),
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the checker stage"):
            return ModelInvocationResult(
                command=["fake-checker"],
                stdout=(
                    '{"summary":"bad citation","findings":[{"title":"Bogus file citation",'
                    '"severity":"high","confidence":"likely","category":"unsupported_claim",'
                    '"rationale":"Checker cited a file that does not exist.",'
                    '"exact_fix_action":"Verify the cited file before acting.",'
                    '"file_path":"src/does_not_exist.py","primary_symbol":"ghost_symbol",'
                    '"line_range":"99-100"}]}'
                ),
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the adjudicator stage"):
            assert '"status": "failed"' in prompt
            assert 'src/does_not_exist.py' in prompt
            stable_id = prompt.split("stable_id: ", 1)[1].split("\n", 1)[0]
            return ModelInvocationResult(
                command=["fake-adjudicator"],
                stdout=(
                    '{"summary":"citation failure confirmed","dispositions":[{' 
                    f'"stable_id":"{stable_id}","decision":"rejected","justification":"The verification report shows the citation failed."'
                    '}]}'
                ),
                stderr="",
                returncode=0,
            )
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: model_response(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    resume_payload = json.loads(capsys.readouterr().out)

    assert resume_exit == 0
    assert resume_payload["final_state"] == "completed"

    with open_state_store(Path(run_payload["database_path"])) as store:
        artifacts = store.artifacts.list_by_run(run_payload["run_id"])

    verification_artifact = next(
        artifact for artifact in artifacts if artifact.artifact_type == "evidence_verification_report"
    )
    verification_payload = json.loads(
        (Path(run_payload["artifacts_root"]) / verification_artifact.path).read_text(encoding="utf-8")
    )

    assert verification_payload["findings"][0]["status"] == "failed"
    assert verification_payload["findings"][0]["checks"][0]["check"] == "file_path"


def test_approve_sets_operator_status_and_patches_limits(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    approve_exit = main([
        "approve",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--max-iterations",
        "5",
        "--budget-limit-usd",
        "12.50",
        "--json",
    ])
    approve_payload = json.loads(capsys.readouterr().out)

    assert approve_exit == 0
    assert approve_payload["operator_status"] == "approved"
    assert approve_payload["max_iterations"] == 5
    assert Decimal(approve_payload["budget_limit_usd"]) == Decimal("12.50")

    with open_state_store(Path(run_payload["database_path"])) as store:
        run = store.runs.get(run_payload["run_id"])
        actions = store.operator_actions.list_by_run(run_payload["run_id"])

    assert run is not None
    assert run.operator_status.value == "approved"
    assert run.max_iterations == 5
    assert run.budget_limit_usd == Decimal("12.50")
    assert [action.action_type for action in actions] == [OperatorActionType.APPROVE]


def test_abort_transitions_run_to_aborted(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    abort_exit = main(["abort", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    abort_payload = json.loads(capsys.readouterr().out)

    assert abort_exit == 0
    assert abort_payload["state"] == "aborted"

    with open_state_store(Path(run_payload["database_path"])) as store:
        run = store.runs.get(run_payload["run_id"])
        actions = store.operator_actions.list_by_run(run_payload["run_id"])

    assert run is not None
    assert run.state.value == "aborted"
    assert [action.action_type for action in actions] == [OperatorActionType.ABORT]


def test_hold_reason_returns_current_operator_hold_reason(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)

    assert run_exit == 0

    hold_reason_exit = main([
        "hold-reason",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    hold_reason_payload = json.loads(capsys.readouterr().out)

    assert hold_reason_exit == 0
    assert hold_reason_payload["state"] == "awaiting_operator"
    assert hold_reason_payload["hold_type"] == "plan_review"
    assert hold_reason_payload["hold_reason"] == "awaiting operator review of planner output"
    assert hold_reason_payload["hold_context"]["planning_stage_id"] is not None
    assert hold_reason_payload["review_paths"]["plan"].endswith(".md")
    assert hold_reason_payload["review_paths"]["planner_response"].endswith(".md")


def test_hold_reason_plain_text_surfaces_review_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)

    assert run_exit == 0

    hold_reason_exit = main([
        "hold-reason",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
    ])
    captured = capsys.readouterr()

    assert hold_reason_exit == 0
    assert "run_id: " in captured.out
    assert "status: awaiting_operator" in captured.out
    assert "hold: plan_review" in captured.out
    assert "reason: awaiting operator review of planner output" in captured.out
    assert "review plan: " in captured.out
    assert ".grind/artifacts/" in captured.out
    assert "approve: grind approve" in captured.out
    assert "reject: grind reject" in captured.out
    assert "resume after approval: grind resume" in captured.out


def test_reject_replans_and_returns_to_plan_review_hold(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    calls = {"planner": 0}

    def model_response(prompt: str) -> ModelInvocationResult:
        if prompt.startswith("You are planning a grind task"):
            calls["planner"] += 1
            return ModelInvocationResult(
                command=["fake-planner", str(calls["planner"])],
                stdout=json.dumps({"plan": f"ship it {calls['planner']}"}),
                stderr="",
                returncode=0,
            )
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: model_response(prompt),
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    reject_exit = main(["reject", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    reject_payload = json.loads(capsys.readouterr().out)

    assert reject_exit == 0
    assert reject_payload["state"] == "awaiting_operator"
    assert reject_payload["hold_type"] == "plan_review"
    assert reject_payload["hold_context"]["replan"] is True

    with open_state_store(Path(run_payload["database_path"])) as store:
        stages = store.stages.list_by_run(run_payload["run_id"])
        actions = store.operator_actions.list_by_run(run_payload["run_id"])

    assert len([stage for stage in stages if stage.stage_name == "planning"]) == 2
    assert [action.action_type for action in actions] == [OperatorActionType.REJECT]


def test_resume_persists_model_calls_and_exposes_them_via_inspect(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    def model_response(prompt: str) -> ModelInvocationResult:
        response = _model_response_for_prompt(prompt)
        return ModelInvocationResult(
            command=response.command,
            stdout=response.stdout,
            stderr=response.stderr,
            returncode=response.returncode,
            estimated_cost_usd=Decimal("0.05"),
            input_tokens=100,
            output_tokens=20,
            provider_metadata={"reported_model": "fake", "input_tokens": 100, "output_tokens": 20, "estimated_cost_usd": Decimal("0.05")},
        )

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: model_response(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    resume_payload = json.loads(capsys.readouterr().out)
    assert resume_exit == 0
    assert resume_payload["final_state"] == "completed"

    inspect_exit = main([
        "inspect",
        "model_calls",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    inspect_payload = json.loads(capsys.readouterr().out)

    assert inspect_exit == 0
    assert len(inspect_payload["model_calls"]) == 4
    assert inspect_payload["model_calls"][0]["input_tokens"] == 100
    assert Decimal(inspect_payload["model_calls"][0]["estimated_cost_usd"]) == Decimal("0.05")


def test_semantic_hard_fail_persists_records_before_checker(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    def model_response(prompt: str) -> ModelInvocationResult:
        if prompt.startswith("You are planning a grind task"):
            return ModelInvocationResult(command=["fake-planner"], stdout='{"plan":"ship it"}', stderr="", returncode=0)
        if prompt.startswith("You are the do stage implementer"):
            (tmp_path / "surprise.txt").write_text("semantic drift\n", encoding="utf-8")
            return ModelInvocationResult(
                command=["fake-implementer", "do"],
                stdout='{"touched_files":[],"touched_symbols":[],"validation_hints":[],"claims_made":[],"open_uncertainties":[],"artifact_refs":[]}',
                stderr="",
                returncode=0,
            )
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: model_response(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    approve_exit = main([
        "approve",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--max-iterations",
        "1",
        "--json",
    ])
    assert approve_exit == 0
    capsys.readouterr()

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    resume_payload = json.loads(capsys.readouterr().out)

    assert resume_exit == 0
    assert resume_payload["final_state"] == "awaiting_operator"

    with open_state_store(Path(run_payload["database_path"])) as store:
        semantic_audits = store.semantic_audits.list_by_run(run_payload["run_id"])
        findings = store.findings.list_by_run(run_payload["run_id"])
        dispositions = store.dispositions.list_by_run(run_payload["run_id"])
        stages = store.stages.list_by_run(run_payload["run_id"])

    assert semantic_audits[0].hard_fail is True
    assert any(disposition.decided_by.value == "engine" for disposition in dispositions)
    assert any(finding.category.value == "scope_violation" for finding in findings)
    assert "checking" not in [stage.stage_name for stage in stages]


def test_consensus_split_persists_panel_votes_and_holds_run(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()
    engine_config = tmp_path / ".grind" / "engine.yaml"
    engine_config.write_text(
        engine_config.read_text(encoding="utf-8").replace("consensus_enabled: false", "consensus_enabled: true"),
        encoding="utf-8",
    )

    def model_response(prompt: str) -> ModelInvocationResult:
        if prompt.startswith("You are planning a grind task"):
            return ModelInvocationResult(command=["fake-planner"], stdout='{"plan":"ship it"}', stderr="", returncode=0)
        if prompt.startswith("You are the do stage implementer"):
            return ModelInvocationResult(
                command=["fake-implementer", "do"],
                stdout='{"touched_files":["tests/test_cli.py"],"touched_symbols":[],"validation_hints":[],"claims_made":[],"open_uncertainties":[],"artifact_refs":[]}',
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the checker stage"):
            return ModelInvocationResult(
                command=["fake-checker"],
                stdout=(
                    '{"summary":"one blocker","findings":[{"title":"Missing regression coverage",'
                    '"severity":"high","confidence":"proven","category":"test_coverage",'
                    '"rationale":"Critical regression path has no test coverage.",'
                    '"exact_fix_action":"Add a regression test for the failing path.",'
                    '"file_path":"tests/test_cli.py","primary_symbol":"test_consensus_split_persists_panel_votes_and_holds_run",'
                    '"line_range":"1-10"}]}'
                ),
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the adjudicator stage"):
            stable_id = prompt.split("stable_id: ", 1)[1].split("\n", 1)[0]
            decision = "open"
            if "testing_specialist" in prompt:
                decision = "rejected"
            return ModelInvocationResult(
                command=["fake-adjudicator"],
                stdout=(
                    '{"summary":"vote","dispositions":[{'
                    f'"stable_id":"{stable_id}","decision":"{decision}","justification":"panel vote"'
                    '}]}'
                ),
                stderr="",
                returncode=0,
            )
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: model_response(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    resume_payload = json.loads(capsys.readouterr().out)

    assert resume_exit == 0
    assert resume_payload["final_state"] == "awaiting_operator"

    status_exit = main(["status", "--run-id", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    status_payload = json.loads(capsys.readouterr().out)

    assert status_exit == 0
    assert status_payload["hold_type"] == "critical_disagreement"

    with open_state_store(Path(run_payload["database_path"])) as store:
        panels = store.adjudication_panels.list_by_run(run_payload["run_id"])
        votes = store.adjudication_votes.list_by_run(run_payload["run_id"])

    assert panels[0].status == "split"
    assert len(votes) == 3


def test_patch_policy_updates_validation_override_and_resume_uses_it(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    observed_commands: list[list[str]] = []

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )

    def fake_validation_runner(
        cwd: Path,
        commands: list[list[str]],
        *,
        stop_on_failure: bool,
        timeout_seconds: int | None,
    ) -> list[ValidationExecutionResult]:
        observed_commands.append([" ".join(command) for command in commands])
        return [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ]

    monkeypatch.setattr("grind.engine.orchestrator.run_validation_commands", fake_validation_runner)

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    patch_exit = main([
        "patch-policy",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--validation-command",
        "uv run pytest tests/test_cli.py -q",
        "--validation-command",
        "uv run pytest tests/test_state_store.py -q",
        "--json",
    ])
    patch_payload = json.loads(capsys.readouterr().out)

    assert patch_exit == 0
    assert patch_payload["validation_commands_override"] == [
        "uv run pytest tests/test_cli.py -q",
        "uv run pytest tests/test_state_store.py -q",
    ]

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    resume_payload = json.loads(capsys.readouterr().out)

    assert resume_exit == 0
    assert resume_payload["final_state"] == "completed"
    assert observed_commands == [
        ["uv run pytest tests/test_cli.py -q"],
        ["uv run pytest tests/test_state_store.py -q"],
    ]

    with open_state_store(Path(run_payload["database_path"])) as store:
        run = store.runs.get(run_payload["run_id"])
        actions = store.operator_actions.list_by_run(run_payload["run_id"])

    assert run is not None
    assert run.validation_commands_override == [
        "uv run pytest tests/test_cli.py -q",
        "uv run pytest tests/test_state_store.py -q",
    ]
    assert [action.action_type for action in actions] == [
        OperatorActionType.PATCH_POLICY,
        OperatorActionType.RESUME,
    ]
    assert actions[0].payload is not None
    assert actions[0].payload["effective"]["validation_commands_override"] == [
        "uv run pytest tests/test_cli.py -q",
        "uv run pytest tests/test_state_store.py -q",
    ]


def test_inspect_lists_and_reads_artifacts(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: _model_response_for_prompt(prompt),
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    inspect_exit = main([
        "inspect",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    inspect_payload = json.loads(capsys.readouterr().out)

    assert inspect_exit == 0
    assert len(inspect_payload["artifacts"]) == 4
    assert inspect_payload["artifacts"][0]["artifact_type"] == "planning_prompt"

    inspect_prompt_exit = main([
        "inspect",
        "planning_prompt",
        "--run-id",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    inspect_prompt_payload = json.loads(capsys.readouterr().out)

    assert inspect_prompt_exit == 0
    assert "Objective: ship it" in inspect_prompt_payload["content"]


def test_resume_holds_when_max_iterations_reached(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    def model_response(prompt: str) -> ModelInvocationResult:
        if prompt.startswith("You are planning a grind task"):
            return ModelInvocationResult(command=["fake-planner"], stdout='{"plan":"ship it"}', stderr="", returncode=0)
        if prompt.startswith("You are the do stage implementer"):
            return ModelInvocationResult(
                command=["fake-implementer", "do"],
                stdout=(
                    '{"touched_files":["tests/test_cli.py"],"touched_symbols":["test"],'
                    '"validation_hints":[],"claims_made":[],"open_uncertainties":[],"artifact_refs":[]}'
                ),
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the checker stage"):
            return ModelInvocationResult(
                command=["fake-checker"],
                stdout=(
                    '{"summary":"one blocker","findings":[{"title":"Missing regression coverage",'
                    '"severity":"high","confidence":"proven","category":"test_coverage",'
                    '"rationale":"Critical regression path has no test coverage.",'
                    '"exact_fix_action":"Add a regression test for the failing path.",'
                    '"file_path":"tests/test_cli.py","primary_symbol":"test_resume_can_complete_when_validation_passes",'
                    '"line_range":"1-10"}]}'
                ),
                stderr="",
                returncode=0,
            )
        if prompt.startswith("You are the adjudicator stage"):
            stable_id = prompt.split("stable_id: ", 1)[1].split("\n", 1)[0]
            return ModelInvocationResult(
                command=["fake-adjudicator"],
                stdout=(
                    '{"summary":"blocker confirmed","dispositions":[{'
                    f'"stable_id":"{stable_id}","decision":"open","justification":"Blocking until coverage is added."'
                    '}]}'
                ),
                stderr="",
                returncode=0,
            )
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: model_response(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    approve_exit = main([
        "approve",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--max-iterations",
        "1",
        "--json",
    ])
    assert approve_exit == 0
    capsys.readouterr()

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    resume_payload = json.loads(capsys.readouterr().out)

    assert resume_exit == 0
    assert resume_payload["final_state"] == "awaiting_operator"

    status_exit = main(["status", "--run-id", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    status_payload = json.loads(capsys.readouterr().out)

    assert status_exit == 0
    assert status_payload["hold_reason"].startswith("max_iterations:")
    assert status_payload["iteration_count"] == 1

    with open_state_store(Path(run_payload["database_path"])) as store:
        stages = store.stages.list_by_run(run_payload["run_id"])

    assert "acting" not in [stage.stage_name for stage in stages]


def test_resume_holds_when_budget_limit_reached(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()

    def model_response(prompt: str) -> ModelInvocationResult:
        if prompt.startswith("You are planning a grind task"):
            return ModelInvocationResult(command=["fake-planner"], stdout='{"plan":"ship it"}', stderr="", returncode=0)
        if prompt.startswith("You are the do stage implementer"):
            return ModelInvocationResult(
                command=["fake-implementer", "do"],
                stdout=(
                    '{"touched_files":["tests/test_cli.py"],"touched_symbols":["test"],'
                    '"validation_hints":[],"claims_made":[],"open_uncertainties":[],"artifact_refs":[]}'
                ),
                stderr="",
                returncode=0,
                estimated_cost_usd=Decimal("0.40"),
            )
        if prompt.startswith("You are the checker stage"):
            return ModelInvocationResult(
                command=["fake-checker"],
                stdout=(
                    '{"summary":"one blocker","findings":[{"title":"Missing regression coverage",'
                    '"severity":"high","confidence":"proven","category":"test_coverage",'
                    '"rationale":"Critical regression path has no test coverage.",'
                    '"exact_fix_action":"Add a regression test for the failing path.",'
                    '"file_path":"tests/test_cli.py","primary_symbol":"test_resume_can_complete_when_validation_passes",'
                    '"line_range":"1-10"}]}'
                ),
                stderr="",
                returncode=0,
                estimated_cost_usd=Decimal("0.40"),
            )
        if prompt.startswith("You are the adjudicator stage"):
            stable_id = prompt.split("stable_id: ", 1)[1].split("\n", 1)[0]
            return ModelInvocationResult(
                command=["fake-adjudicator"],
                stdout=(
                    '{"summary":"blocker confirmed","dispositions":[{'
                    f'"stable_id":"{stable_id}","decision":"open","justification":"Blocking until coverage is added."'
                    '}]}'
                ),
                stderr="",
                returncode=0,
                estimated_cost_usd=Decimal("0.30"),
            )
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: model_response(prompt),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            _validation_result(commands, returncode=0, stdout="passed", stderr="")
        ],
    )

    run_exit = main(["run", "ship it", "--cwd", str(tmp_path), "--json"])
    run_payload = json.loads(capsys.readouterr().out)
    assert run_exit == 0

    approve_exit = main([
        "approve",
        run_payload["run_id"],
        "--cwd",
        str(tmp_path),
        "--budget-limit-usd",
        "1.00",
        "--json",
    ])
    assert approve_exit == 0
    capsys.readouterr()

    resume_exit = main(["resume", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    resume_payload = json.loads(capsys.readouterr().out)

    assert resume_exit == 0
    assert resume_payload["final_state"] == "awaiting_operator"

    status_exit = main(["status", "--run-id", run_payload["run_id"], "--cwd", str(tmp_path), "--json"])
    status_payload = json.loads(capsys.readouterr().out)

    assert status_exit == 0
    assert status_payload["hold_reason"].startswith("budget_exceeded:")

    with open_state_store(Path(run_payload["database_path"])) as store:
        run = store.runs.get(run_payload["run_id"])
        stages = store.stages.list_by_run(run_payload["run_id"])

    assert run is not None
    assert run.total_cost_usd >= Decimal("1.00")
    assert "acting" not in [stage.stage_name for stage in stages]
