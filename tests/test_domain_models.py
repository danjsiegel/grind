from __future__ import annotations

from decimal import Decimal

import pytest

from grind.models import (
    CaptureMode,
    CheckpointKind,
    CheckpointStatus,
    DecidedBy,
    DifferenceSurface,
    Disposition,
    EnforcementMode,
    EvidenceType,
    Finding,
    FindingCategory,
    FindingConfidence,
    FindingEvidence,
    FindingSeverity,
    FindingStatus,
    InvariantContract,
    InvariantKind,
    InvariantSourceKind,
    InvariantStatus,
    ModelRole,
    OperatorStatus,
    Run,
    RunState,
    Stage,
    StageStatus,
    Task,
    TaskSourceKind,
    TaskStatus,
    WorkspaceCheckpoint,
)


# ── helpers ──────────────────────────────────────────────────────────────────

RUN_ID = "run_20260516_120000_abcdef01"


def make_run(**kw: object) -> Run:
    return Run(
        run_id=RUN_ID,
        repo_path="/repos/myproject",
        policy_pack_path="/repos/myproject/.grind/policy",
        policy_schema_ver="1",
        requested_objective="implement feature X",
        **kw,
    )


def make_finding(
    finding_id: str = "f-001",
    stable_id: str = "a" * 16,
    severity: FindingSeverity = FindingSeverity.HIGH,
) -> Finding:
    return Finding(
        finding_id=finding_id,
        run_id=RUN_ID,
        stage_id="stage-001",
        stable_id=stable_id,
        title="Missing null check",
        severity=severity,
        confidence=FindingConfidence.PROVEN,
        category=FindingCategory.CORRECTNESS,
        rationale="The function does not handle null input",
        exact_fix_action="Add null check at line 42",
    )


def make_disposition(finding_id: str = "f-001", iteration: int = 1) -> Disposition:
    return Disposition(
        disposition_id=f"d-{finding_id}",
        finding_id=finding_id,
        stage_id="stage-002",
        iteration=iteration,
        decided_by=DecidedBy.ADJUDICATOR,
        decision=FindingStatus.OPEN,
        justification="Evidence confirmed at src/foo.py:42",
    )


# ── Run ───────────────────────────────────────────────────────────────────────

def test_run_defaults() -> None:
    run = make_run()
    assert run.state == RunState.CREATED
    assert run.operator_status == OperatorStatus.NONE
    assert run.iteration_count == 0
    assert run.max_iterations == 3
    assert run.total_cost_usd == Decimal("0")
    assert run.budget_limit_usd is None
    assert run.current_worker_id is None


def test_run_with_budget() -> None:
    run = make_run(budget_limit_usd=Decimal("10.50"))
    assert run.budget_limit_usd == Decimal("10.50")


def test_run_iteration_count_non_negative() -> None:
    with pytest.raises(Exception):
        make_run(iteration_count=-1)


def test_run_max_iterations_at_least_one() -> None:
    with pytest.raises(Exception):
        make_run(max_iterations=0)


# ── Task ──────────────────────────────────────────────────────────────────────

def test_task_defaults() -> None:
    task = Task(
        task_id="task-001",
        run_id=RUN_ID,
        sequence=0,
        source_kind=TaskSourceKind.INLINE,
        raw_input="implement feature X",
    )
    assert task.status == TaskStatus.PENDING
    assert task.acceptance_checks == []
    assert task.normalized_scope is None
    assert task.phase_label is None


def test_task_with_acceptance_checks() -> None:
    task = Task(
        task_id="task-001",
        run_id=RUN_ID,
        sequence=0,
        source_kind=TaskSourceKind.FILE,
        raw_input="implement feature X",
        acceptance_checks=["all tests pass", "no new linter errors"],
    )
    assert len(task.acceptance_checks) == 2


# ── Stage ─────────────────────────────────────────────────────────────────────

def test_stage_defaults() -> None:
    stage = Stage(
        stage_id="stage-001",
        run_id=RUN_ID,
        task_id="task-001",
        stage_name="planning",
        model_role=ModelRole.PLANNER,
    )
    assert stage.status == StageStatus.PENDING
    assert stage.iteration == 1
    assert stage.ended_at is None
    assert stage.latency_ms is None


def test_stage_iteration_at_least_one() -> None:
    with pytest.raises(Exception):
        Stage(
            stage_id="stage-001",
            run_id=RUN_ID,
            task_id="task-001",
            stage_name="planning",
            iteration=0,
        )


# ── Finding ───────────────────────────────────────────────────────────────────

def test_finding_stable_id_must_be_16_chars() -> None:
    with pytest.raises(Exception):
        make_finding(stable_id="too-short")


def test_finding_stable_id_too_long() -> None:
    with pytest.raises(Exception):
        make_finding(stable_id="a" * 17)


def test_finding_defaults() -> None:
    f = make_finding()
    assert f.status == FindingStatus.OPEN
    assert f.adjudicated is False


def test_finding_severity_values() -> None:
    for sev in FindingSeverity:
        f = make_finding(severity=sev)
        assert f.severity == sev


# ── FindingEvidence ───────────────────────────────────────────────────────────

def test_finding_evidence_snippet_max_length() -> None:
    with pytest.raises(Exception):
        FindingEvidence(
            evidence_id="ev-001",
            finding_id="f-001",
            evidence_type=EvidenceType.FILE_REF,
            snippet="x" * 2001,
        )


def test_finding_evidence_minimal() -> None:
    ev = FindingEvidence(
        evidence_id="ev-001",
        finding_id="f-001",
        evidence_type=EvidenceType.DIFF,
    )
    assert ev.snippet is None
    assert ev.artifact_id is None
    assert ev.source_ref is None


def test_finding_evidence_at_max_snippet() -> None:
    ev = FindingEvidence(
        evidence_id="ev-001",
        finding_id="f-001",
        evidence_type=EvidenceType.TEST_RESULT,
        snippet="x" * 2000,
    )
    assert len(ev.snippet) == 2000  # type: ignore[arg-type]


# ── Disposition ───────────────────────────────────────────────────────────────

def test_disposition_empty_justification_rejected() -> None:
    with pytest.raises(Exception):
        Disposition(
            disposition_id="d-001",
            finding_id="f-001",
            stage_id="stage-001",
            iteration=1,
            decided_by=DecidedBy.ADJUDICATOR,
            decision=FindingStatus.OPEN,
            justification="",
        )


def test_disposition_serialization_round_trip() -> None:
    d = make_disposition()
    data = d.model_dump()
    restored = Disposition.model_validate(data)
    assert restored.disposition_id == d.disposition_id
    assert restored.decided_by == DecidedBy.ADJUDICATOR
    assert restored.decision == FindingStatus.OPEN


# ── InvariantContract ─────────────────────────────────────────────────────────

def test_invariant_contract_defaults() -> None:
    inv = InvariantContract(
        invariant_id="inv-001",
        run_id=RUN_ID,
        task_id="task-001",
        scope_ref="src/foo.py::MyClass.method",
        invariant_kind=InvariantKind.SIGNATURE,
        statement="method(x: int) -> str must remain stable",
        source_kind=InvariantSourceKind.SPEC,
        source_artifact_id="artifact-001",
        enforcement_mode=EnforcementMode.BLOCKING,
    )
    assert inv.status == InvariantStatus.ACTIVE
    assert inv.retired_at is None


def test_invariant_contract_round_trip() -> None:
    inv = InvariantContract(
        invariant_id="inv-001",
        run_id=RUN_ID,
        task_id="task-001",
        scope_ref="src/foo.py::MyClass.method",
        invariant_kind=InvariantKind.DEPENDENCY,
        statement="no circular imports in src/",
        source_kind=InvariantSourceKind.SEMANTIC_AUDIT,
        source_artifact_id="artifact-002",
        enforcement_mode=EnforcementMode.ADVISORY,
    )
    data = inv.model_dump()
    restored = InvariantContract.model_validate(data)
    assert restored.invariant_id == inv.invariant_id
    assert restored.enforcement_mode == EnforcementMode.ADVISORY
    assert restored.source_kind == InvariantSourceKind.SEMANTIC_AUDIT


def test_invariant_contract_all_kinds() -> None:
    for kind in InvariantKind:
        inv = InvariantContract(
            invariant_id="inv-x",
            run_id=RUN_ID,
            task_id="task-001",
            scope_ref="src/",
            invariant_kind=kind,
            statement="some invariant",
            source_kind=InvariantSourceKind.POLICY,
            source_artifact_id="artifact-001",
            enforcement_mode=EnforcementMode.ADVISORY,
        )
        assert inv.invariant_kind == kind


# ── WorkspaceCheckpoint ───────────────────────────────────────────────────────

def test_checkpoint_defaults() -> None:
    cp = WorkspaceCheckpoint(
        checkpoint_id="cp-001",
        run_id=RUN_ID,
        iteration=0,
        checkpoint_kind=CheckpointKind.TASK_BASELINE,
        capture_mode=CaptureMode.GIT_PATCH_BUNDLE,
        scope_paths=["src/", "tests/"],
        artifact_id="artifact-001",
    )
    assert cp.status == CheckpointStatus.AVAILABLE
    assert cp.created_by == "engine"
    assert cp.restored_at is None
    assert cp.task_id is None
    assert cp.stage_id is None


def test_checkpoint_round_trip() -> None:
    cp = WorkspaceCheckpoint(
        checkpoint_id="cp-001",
        run_id=RUN_ID,
        task_id="task-001",
        stage_id="stage-005",
        iteration=1,
        checkpoint_kind=CheckpointKind.PRE_ACT,
        capture_mode=CaptureMode.SAFE_PATH_SNAPSHOT,
        scope_paths=["src/feature/"],
        artifact_id="artifact-002",
    )
    data = cp.model_dump()
    restored = WorkspaceCheckpoint.model_validate(data)
    assert restored.checkpoint_id == cp.checkpoint_id
    assert restored.scope_paths == ["src/feature/"]
    assert restored.checkpoint_kind == CheckpointKind.PRE_ACT
    assert restored.created_by == "engine"


def test_checkpoint_scope_paths_required() -> None:
    with pytest.raises(Exception):
        WorkspaceCheckpoint(
            checkpoint_id="cp-001",
            run_id=RUN_ID,
            iteration=0,
            checkpoint_kind=CheckpointKind.TASK_BASELINE,
            capture_mode=CaptureMode.GIT_PATCH_BUNDLE,
            scope_paths=[],  # min_length=1 should fail
            artifact_id="artifact-001",
        )


# ── DifferenceSurface ─────────────────────────────────────────────────────────

def test_difference_surface_minimal() -> None:
    ds = DifferenceSurface(
        surface_id="ds-001",
        run_id=RUN_ID,
        task_id="task-001",
        iteration=1,
    )
    assert ds.policy_delta == {}
    assert ds.findings_delta == {}
    assert ds.iteration == 1


def test_difference_surface_with_deltas() -> None:
    ds = DifferenceSurface(
        surface_id="ds-001",
        run_id=RUN_ID,
        task_id="task-001",
        iteration=1,
        requested_delta={"task_summary": "implement X", "acceptance_checks": ["tests pass"]},
        observed_delta={"touched_files": ["src/foo.py"], "touched_symbols": ["MyClass.method"]},
        policy_delta={"max_iterations": {"prior": 3, "effective": 5}},
    )
    assert ds.observed_delta["touched_files"] == ["src/foo.py"]
    assert ds.policy_delta["max_iterations"]["effective"] == 5


def test_difference_surface_round_trip() -> None:
    ds = DifferenceSurface(
        surface_id="ds-001",
        run_id=RUN_ID,
        task_id="task-001",
        iteration=2,
        risk_delta={"scope_expansions": ["src/internal/"]},
    )
    data = ds.model_dump()
    restored = DifferenceSurface.model_validate(data)
    assert restored.surface_id == ds.surface_id
    assert restored.iteration == 2
    assert restored.risk_delta == {"scope_expansions": ["src/internal/"]}


def test_difference_surface_iteration_must_be_positive() -> None:
    with pytest.raises(Exception):
        DifferenceSurface(
            surface_id="ds-002",
            run_id=RUN_ID,
            task_id="task-001",
            iteration=0,
        )
