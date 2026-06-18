from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import duckdb

from grind.models import (
    AdjudicationPanelRecord,
    AdjudicationVoteRecord,
    CaptureMode,
    CheckpointKind,
    Finding,
    FindingCategory,
    FindingConfidence,
    FindingSeverity,
    HoldType,
    ModelCallRecord,
    ModelRole,
    OperatorActionRecord,
    OperatorActionType,
    OperatorStatus,
    RetrievalQueueRecord,
    Run,
    RunState,
    SemanticAuditRecord,
    Stage,
    StageStatus,
    Task,
    TaskSourceKind,
    ValidationRecord,
    WorkspaceCheckpoint,
)
from grind.models.artifact import ArtifactRecord
from grind.models.transition import TransitionRecord
from grind.state import bootstrap_state_store, current_schema_version, open_state_store


def test_bootstrap_state_store_creates_database_and_schema(tmp_path: Path) -> None:
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"

    bootstrap_state_store(database_path)

    assert database_path.exists()
    assert current_schema_version(database_path) == 7

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
    finally:
        connection.close()

    assert "runs" in tables
    assert "tasks" in tables
    assert "stages" in tables
    assert "artifacts" in tables
    assert "transitions" in tables
    assert "findings" in tables
    assert "retrieval_index_queue" in tables
    assert "workspace_checkpoints" in tables
    assert "validations" in tables
    assert "operator_actions" in tables
    assert "model_calls" in tables
    assert "semantic_audits" in tables
    assert "adjudication_panels" in tables
    assert "adjudication_votes" in tables
    assert "workers" in tables
    assert "run_leases" in tables


def test_repositories_round_trip_core_ledger(tmp_path: Path) -> None:
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    bootstrap_state_store(database_path)

    run = Run(
        run_id="run_20260516_120000_deadbeef",
        repo_path=str(tmp_path),
        policy_pack_path=str(tmp_path / ".grind"),
        policy_schema_ver="0.1",
        requested_objective="execute the task",
        state=RunState.CREATED,
        operator_status=OperatorStatus.NONE,
        current_hold_type=HoldType.PLAN_REVIEW,
        current_hold_reason="awaiting operator review of planner output",
        current_hold_context={"planning_stage_id": "stage_1"},
        validation_commands_override=["uv run pytest tests -q"],
        total_cost_usd=Decimal("0"),
    )
    task = Task(
        task_id="task_1",
        run_id=run.run_id,
        sequence=0,
        source_kind=TaskSourceKind.INLINE,
        raw_input="execute the task",
    )
    stage = Stage(
        stage_id="stage_1",
        run_id=run.run_id,
        task_id=task.task_id,
        stage_name="planning",
        status=StageStatus.COMPLETED,
        model_role=ModelRole.PLANNER,
    )
    artifact = ArtifactRecord(
        artifact_id="artifact_1",
        run_id=run.run_id,
        artifact_type="plan",
        path=str(tmp_path / "plan.json"),
    )
    transition = TransitionRecord(
        transition_id="transition_1",
        run_id=run.run_id,
        from_state=RunState.CREATED,
        to_state=RunState.PLANNING,
        reason="started",
        actor="engine",
    )
    finding = Finding(
        finding_id="finding_1",
        run_id=run.run_id,
        stage_id=stage.stage_id,
        stable_id="1234567890abcdef",
        title="Missing validation",
        severity=FindingSeverity.HIGH,
        confidence=FindingConfidence.PROVEN,
        category=FindingCategory.MISSING_VALIDATION,
        rationale="validation was skipped",
        exact_fix_action="run the validation",
    )
    checkpoint = WorkspaceCheckpoint(
        checkpoint_id="checkpoint_1",
        run_id=run.run_id,
        task_id=task.task_id,
        iteration=0,
        checkpoint_kind=CheckpointKind.TASK_BASELINE,
        capture_mode=CaptureMode.SAFE_PATH_SNAPSHOT,
        scope_paths=["."],
        artifact_id=artifact.artifact_id,
    )
    validation = ValidationRecord(
        validation_id="validation_1",
        run_id=run.run_id,
        task_id=task.task_id,
        stage_id=stage.stage_id,
        command="uv run pytest tests -q",
        status="passed",
        exit_code=0,
        stdout_artifact_id=artifact.artifact_id,
        summary="passed",
    )
    action = OperatorActionRecord(
        action_id="action_1",
        run_id=run.run_id,
        action_type=OperatorActionType.RESUME,
        note="operator approved",
        checkpoint_id=checkpoint.checkpoint_id,
        payload={"source": "test"},
    )
    model_call = ModelCallRecord(
        model_call_id="model_call_1",
        run_id=run.run_id,
        stage_id=stage.stage_id,
        model_role=ModelRole.PLANNER,
        provider="github_cli",
        model_name="gpt-5.4",
        command=["gh", "copilot"],
        status="completed",
        input_tokens=123,
        output_tokens=45,
        estimated_cost_usd=Decimal("0.12"),
        metadata={"reported_model": "gpt-5.4"},
    )
    semantic_audit = SemanticAuditRecord(
        semantic_audit_id="semantic_audit_1",
        run_id=run.run_id,
        task_id=task.task_id,
        stage_id=stage.stage_id,
        iteration=1,
        capability_level="filesystem",
        hard_fail=True,
        blocking_findings=[{"title": "Missing path"}],
        report_artifact_id=artifact.artifact_id,
        difference_surface_artifact_id=artifact.artifact_id,
    )
    panel = AdjudicationPanelRecord(
        panel_id="panel_1",
        run_id=run.run_id,
        task_id=task.task_id,
        stage_id=stage.stage_id,
        iteration=1,
        mode="consensus",
        status="running",
    )
    vote = AdjudicationVoteRecord(
        vote_id="vote_1",
        panel_id=panel.panel_id,
        run_id=run.run_id,
        stage_id=stage.stage_id,
        member_label="security_auditor",
        provider="github_cli",
        model_name="gpt-5.4",
        payload={"summary": "ship it"},
    )
    retrieval_queue = RetrievalQueueRecord(
        queue_id="queue_1",
        run_id=run.run_id,
        artifact_id=artifact.artifact_id,
        collection="run_summaries",
    )

    with open_state_store(database_path) as store:
        store.runs.create(run)
        store.tasks.create(task)
        store.stages.create(stage)
        store.artifacts.create(artifact)
        store.transitions.create(transition)
        store.findings.create(finding)
        store.checkpoints.create(checkpoint)
        store.validations.create(validation)
        store.operator_actions.create(action)
        store.model_calls.create(model_call)
        store.semantic_audits.create(semantic_audit)
        store.adjudication_panels.create(panel)
        store.adjudication_votes.create(vote)
        store.retrieval_queue.create(retrieval_queue)
        store.checkpoints.mark_restored(checkpoint.checkpoint_id)
        store.adjudication_panels.complete(panel.panel_id, status="unanimous")
        store.retrieval_queue.mark_running(retrieval_queue.queue_id)
        store.retrieval_queue.mark_completed(retrieval_queue.queue_id)

        stored_run = store.runs.get(run.run_id)
        recent_runs = store.runs.list_recent(limit=1)
        stored_tasks = store.tasks.list_by_run(run.run_id)
        stored_stages = store.stages.list_by_run(run.run_id)
        stored_artifact = store.artifacts.get(artifact.artifact_id)
        stored_transitions = store.transitions.list_by_run(run.run_id)
        stored_findings = store.findings.list_by_run(run.run_id)
        stored_checkpoints = store.checkpoints.list_by_run(run.run_id)
        restored_checkpoint = store.checkpoints.get(checkpoint.checkpoint_id)
        stored_validations = store.validations.list_by_run(run.run_id)
        stored_actions = store.operator_actions.list_by_run(run.run_id)
        stored_model_calls = store.model_calls.list_by_run(run.run_id)
        stored_semantic_audits = store.semantic_audits.list_by_run(run.run_id)
        stored_panels = store.adjudication_panels.list_by_run(run.run_id)
        stored_votes = store.adjudication_votes.list_by_run(run.run_id)
        stored_retrieval_queue = store.retrieval_queue.list_by_run(run.run_id)

    assert stored_run is not None
    assert stored_run.run_id == run.run_id
    assert stored_run.current_hold_type == HoldType.PLAN_REVIEW
    assert stored_run.current_hold_context == {"planning_stage_id": "stage_1"}
    assert stored_run.validation_commands_override == ["uv run pytest tests -q"]
    assert recent_runs[0].run_id == run.run_id
    assert stored_tasks[0].task_id == task.task_id
    assert stored_stages[0].stage_id == stage.stage_id
    assert stored_artifact is not None
    assert stored_artifact.artifact_id == artifact.artifact_id
    assert stored_transitions[0].to_state == RunState.PLANNING
    assert stored_findings[0].finding_id == finding.finding_id
    assert stored_checkpoints[0].checkpoint_id == checkpoint.checkpoint_id
    assert restored_checkpoint is not None
    assert restored_checkpoint.status.value == "restored"
    assert stored_validations[0].validation_id == validation.validation_id
    assert stored_actions[0].action_id == action.action_id
    assert stored_actions[0].action_type == OperatorActionType.RESUME
    assert stored_actions[0].payload == {"source": "test"}
    assert stored_model_calls[0].input_tokens == 123
    assert stored_semantic_audits[0].hard_fail is True
    assert stored_panels[0].status == "unanimous"
    assert stored_votes[0].member_label == "security_auditor"
    assert stored_retrieval_queue[0].queue_status == "completed"


def test_bootstrap_state_store_can_route_through_quack_connection(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    remote_path = tmp_path / "remote.duckdb"

    monkeypatch.setenv("GRIND_DB_URI", "quack:localhost")
    monkeypatch.setenv("GRIND_DB_TOKEN", "test-token")
    monkeypatch.setattr(
        "grind.state.store.quack_connect",
        lambda uri, token: duckdb.connect(str(remote_path)),
    )

    bootstrap_state_store(database_path)

    assert current_schema_version(database_path) == 7

    with open_state_store(database_path) as store:
        row = store.connection.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()

    assert row == (7,)


def test_open_state_store_auto_starts_local_quack_when_token_missing(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    remote_path = tmp_path / "remote.duckdb"

    monkeypatch.setenv("GRIND_DB_URI", "quack:localhost")
    monkeypatch.delenv("GRIND_DB_TOKEN", raising=False)
    def fake_ensure_local_quack_server(path, uri):
        bootstrap_state_store(remote_path, db_uri=str(remote_path))
        return "auto-token"

    monkeypatch.setattr("grind.state.store.ensure_local_quack_server", fake_ensure_local_quack_server)
    monkeypatch.setattr(
        "grind.state.store.quack_connect",
        lambda uri, token: duckdb.connect(str(remote_path)),
    )

    bootstrap_state_store(database_path)

    with open_state_store(database_path) as store:
        row = store.connection.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()

    assert row == (7,)
