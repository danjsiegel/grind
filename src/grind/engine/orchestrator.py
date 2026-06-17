from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import secrets
import socket

from pydantic import BaseModel, Field

from grind.artifacts import LocalArtifactStore
from grind.config import ModelProfileConfig, default_engine_config_path, load_engine_config
from grind.engine.checkpoints import capture_workspace_snapshot, restore_workspace_snapshot
from grind.engine.difference_surface_builder import build_difference_surface
from grind.engine.evidence_verifier import (
    CandidateFindingForVerification,
    EvidenceVerificationReport,
    verify_candidate_findings,
)
from grind.engine.leases import BackgroundHeartbeat, LeaseConflictError, acquire_lease, heartbeat_worker, release_lease
from grind.engine.state_machine import finalized_actionable_finding_set, is_terminal
from grind.models import (
    AdjudicationPanelRecord,
    AdjudicationVoteRecord,
    CaptureMode,
    CheckpointKind,
    DecidedBy,
    DifferenceSurface,
    Disposition,
    EvidenceType,
    Finding,
    FindingEvidence,
    FindingCategory,
    FindingConfidence,
    FindingSeverity,
    FindingStatus,
    HoldType,
    ModelCallRecord,
    ModelRole,
    OperatorActionRecord,
    OperatorActionType,
    OperatorStatus,
    Run,
    RunState,
    SemanticAuditRecord,
    Stage,
    StageStatus,
    Task,
    TaskSourceKind,
    TaskStatus,
    TransitionRecord,
    ValidationRecord,
    Worker,
    WorkspaceCheckpoint,
    stable_id,
)
from grind.policy import PolicyLoader
from grind.policy.models import PolicyPack, ValidationCommandSpec
from grind.providers import ModelInvocationError, ModelInvocationResult, extract_json_output, extract_text_output, invoke_text_prompt
from grind.retrieval import LanceDBRetrievalService
from grind.state import bootstrap_state_store, open_state_store
from grind.state.quack import QuackConnectionError
from grind.validation import ValidationExecutionResult, run_validation_commands
from grind.validation.safety import ValidationCommandError, classify_command, normalize_shell_free_command


POLICY_SCHEMA_VERSION = "0.1"


class RiskyValidationCommandError(RuntimeError):
    def __init__(self, *, stage_id: str, command: str, reason: str):
        super().__init__(reason)
        self.stage_id = stage_id
        self.command = command
        self.reason = reason


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _prefixed_id(prefix: str) -> str:
    return f"{prefix}_{_utc_now():%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"


def generate_run_id() -> str:
    return _prefixed_id("run")


def _sanitize_planning_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    for marker in ("Here is the operator review plan:", "Here is the plan:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[1].strip()

    lines = cleaned.splitlines()
    first_heading = next((index for index, line in enumerate(lines) if line.startswith("#")), None)
    if first_heading is not None:
        cleaned = "\n".join(lines[first_heading:]).strip()

    cut_markers = (
        "\n## Operator Actions",
        "\nNow I have all the context I need.",
        "\nI notice the current rewritten thinking",
        "\nCould you provide the next thinking",
    )
    cut_at = len(cleaned)
    for marker in cut_markers:
        index = cleaned.find(marker)
        if index != -1:
            cut_at = min(cut_at, index)
    return cleaned[:cut_at].strip()


def _planning_response_text(result: ModelInvocationResult | None, fallback: str) -> str:
    if result is not None:
        extracted = extract_text_output(result.stdout)
        if extracted.strip():
            stripped_stdout = result.stdout.strip()
            if extracted.strip() != stripped_stdout:
                sanitized = _sanitize_planning_text(extracted)
                if sanitized:
                    return sanitized
                return extracted.strip()

        stripped_stdout = result.stdout.strip()
        if stripped_stdout:
            try:
                parsed = json.loads(stripped_stdout)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                for key in ("plan", "summary", "proposed_plan"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        sanitized = _sanitize_planning_text(value)
                        return sanitized or value.strip()
                    if isinstance(value, list):
                        parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
                        if parts:
                            sanitized = _sanitize_planning_text("\n".join(parts))
                            return sanitized or "\n".join(parts)

            if extracted.strip():
                sanitized = _sanitize_planning_text(extracted)
                if sanitized:
                    return sanitized
                return extracted.strip()
        if result.stderr.strip():
            sanitized = _sanitize_planning_text(result.stderr)
            return sanitized or result.stderr.strip()

    fallback = _sanitize_planning_text(fallback)
    if fallback:
        return fallback
    return "Planner returned no reviewable text."


def _render_plan_review_markdown(
    *,
    objective: str,
    response_text: str,
    replan: bool = False,
) -> str:
    lines = [
        "# Plan Review",
        "",
        "This run is waiting for operator approval before implementation continues.",
        "",
        "Approve if the plan is scoped to the objective, concrete, and includes validation.",
        "Reject if the plan is vague, off-scope, unsafe, or missing a clear validation path.",
        "",
        "## Objective",
        "",
        objective.strip(),
        "",
    ]
    if replan:
        lines.extend([
            "## Context",
            "",
            "This is a replanned proposal after an operator rejection.",
            "",
        ])
    lines.extend([
        "## Proposed Plan",
        "",
        response_text.strip() or "Planner returned no reviewable text. Reject and request replanning.",
        "",
        "## Operator Actions",
        "",
        "- Approve to let grind continue into implementation.",
        "- Reject if the plan is not good enough and request replanning.",
    ])
    return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    task_id: str
    planning_stage_id: str
    checkpoint_id: str
    final_state: RunState
    operator_status: OperatorStatus
    hold_type: str | None
    hold_reason: str | None
    hold_context: dict[str, object] | None
    database_path: Path
    artifacts_root: Path


@dataclass(frozen=True)
class ResumeOutcome:
    run_id: str
    task_id: str
    validation_stage_id: str
    final_state: RunState
    operator_status: OperatorStatus
    hold_type: str | None
    hold_reason: str | None
    hold_context: dict[str, object] | None
    restored_checkpoint_id: str | None
    database_path: Path
    artifacts_root: Path


class ImplementerValidationHintPayload(BaseModel):
    command: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ImplementerClaimPayload(BaseModel):
    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class DoStageResponsePayload(BaseModel):
    touched_files: list[str] = Field(default_factory=list)
    touched_symbols: list[str] = Field(default_factory=list)
    validation_hints: list[ImplementerValidationHintPayload] = Field(default_factory=list)
    claims_made: list[ImplementerClaimPayload] = Field(default_factory=list)
    open_uncertainties: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class ActTriagePayload(BaseModel):
    finding_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    fix_artifact_id: str | None = None
    requested_validation_ids: list[str] = Field(default_factory=list)


class ActStageResponsePayload(BaseModel):
    triage: list[ActTriagePayload] = Field(default_factory=list)
    remaining_open_issues: list[str] = Field(default_factory=list)
    new_uncertainties: list[str] = Field(default_factory=list)


class CheckerFindingPayload(BaseModel):
    title: str = Field(min_length=1)
    severity: FindingSeverity
    confidence: FindingConfidence = FindingConfidence.LIKELY
    category: FindingCategory
    rationale: str = Field(min_length=1)
    exact_fix_action: str = Field(min_length=1)
    file_path: str | None = None
    primary_symbol: str | None = None
    line_range: str | None = None


class CheckerResponsePayload(BaseModel):
    summary: str | None = None
    findings: list[CheckerFindingPayload] = Field(default_factory=list)


class AdjudicatorDispositionPayload(BaseModel):
    stable_id: str = Field(min_length=16, max_length=16)
    decision: FindingStatus
    justification: str = Field(min_length=1)


class AdjudicatorResponsePayload(BaseModel):
    summary: str | None = None
    dispositions: list[AdjudicatorDispositionPayload] = Field(default_factory=list)


class SemanticAuditFindingPayload(BaseModel):
    title: str = Field(min_length=1)
    severity: FindingSeverity
    category: FindingCategory
    rationale: str = Field(min_length=1)
    exact_fix_action: str = Field(min_length=1)
    source_ref: str | None = None


class SemanticAuditResponsePayload(BaseModel):
    audit_id: str = Field(min_length=1)
    capability_level: str = Field(min_length=1)
    hard_fail: bool = False
    blocking_findings: list[SemanticAuditFindingPayload] = Field(default_factory=list)
    advisory_findings: list[SemanticAuditFindingPayload] = Field(default_factory=list)
    invariant_violations: list[str] = Field(default_factory=list)
    dependency_impacts: list[str] = Field(default_factory=list)
    unsupported_checks: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SemanticAuditStageResult:
    stage_id: str
    surface: DifferenceSurface
    surface_artifact_id: str
    report: SemanticAuditResponsePayload
    report_artifact_id: str


@dataclass(frozen=True)
class CheckerStageResult:
    findings: list[Finding]
    evidence_verification: EvidenceVerificationReport
    evidence_verification_artifact_id: str


@dataclass(frozen=True)
class AdjudicationStageResult:
    findings: list[Finding]
    dispositions: list[Disposition]
    held_for_consensus: bool = False


class MinimalOrchestrator:
    def __init__(self, *, cwd: Path, config_path: Path | None = None, policy_pack_path: Path | None = None):
        self.cwd = cwd
        self.config_path = config_path or default_engine_config_path(cwd)
        self.config = load_engine_config(self.config_path)
        self.database_path = self.config.state_path(cwd)
        self.db_uri = self.config.state_db_uri()
        if self.config.state.require_quack and not (self.db_uri and self.db_uri.startswith("quack:")):
            raise QuackConnectionError(
                "Quack is required by configuration for this workspace; set state.db_uri or GRIND_DB_URI to a quack: URI"
            )
        self.artifacts_root = self.config.artifacts_root(cwd)
        self.worker_id = f"worker_{socket.gethostname()}_{os.getpid()}_{secrets.token_hex(4)}"
        self.lease_heartbeat_interval_seconds = 1.0
        self._policy_cache: dict[Path, PolicyPack] = {}
        self.policy_pack = self._load_policy_pack(self._resolve_policy_pack_path(policy_pack_path))

        bootstrap_state_store(self.database_path, db_uri=self.db_uri)
        self.artifact_store = LocalArtifactStore(self.artifacts_root)
        self.retrieval_service = (
            LanceDBRetrievalService(cwd=cwd, config=self.config)
            if self.config.retrieval.enabled
            else None
        )
        self._register_worker()

    def _open_store(self):
        return open_state_store(self.database_path, db_uri=self.db_uri)

    def _resolve_policy_pack_path(self, policy_pack_path: Path | None) -> Path | None:
        if policy_pack_path is not None:
            return policy_pack_path
        candidate = self.cwd / ".grind" / "policy"
        if (candidate / "project.yaml").exists():
            return candidate
        return None

    def _load_policy_pack(self, policy_pack_path: Path | None) -> PolicyPack | None:
        if policy_pack_path is None:
            return None
        resolved_path = policy_pack_path.resolve()
        cached = self._policy_cache.get(resolved_path)
        if cached is not None:
            return cached
        policy_pack = PolicyLoader.load(resolved_path)
        self._policy_cache[resolved_path] = policy_pack
        return policy_pack

    def _policy_pack_for_run(self, run: Run) -> PolicyPack | None:
        if run.policy_pack_path:
            policy_dir = Path(run.policy_pack_path)
            if (policy_dir / "project.yaml").exists():
                return self._load_policy_pack(policy_dir)
        return self.policy_pack

    def _register_worker(self, store: object | None = None) -> None:
        worker = Worker(
            worker_id=self.worker_id,
            hostname=socket.gethostname(),
            pid=os.getpid(),
        )
        if store is None:
            with self._open_store() as worker_store:
                worker_store.workers.register(worker)
            return
        store.workers.register(worker)

    def _heartbeat_worker_once(self) -> None:
        with self._open_store() as store:
            heartbeat_worker(store.connection, self.worker_id)

    @contextmanager
    def _lease_guard(self, store: object, run_id: str):
        self._register_worker(store)
        lease = acquire_lease(store.connection, run_id, self.worker_id)
        heartbeat = BackgroundHeartbeat(
            heartbeat=self._heartbeat_worker_once,
            interval_seconds=self.lease_heartbeat_interval_seconds,
        )
        heartbeat.start()
        try:
            yield lease
        finally:
            heartbeat.stop()
            release_lease(store.connection, lease.lease_id)

    def _sync_retrieval_for_run(self, run_id: str) -> None:
        if self.retrieval_service is None:
            return
        try:
            self.retrieval_service.index_run(run_id=run_id)
        except Exception:
            return

    def _validation_specs_for_run(self, run: Run) -> tuple[list[ValidationCommandSpec], list[str]]:
        if run.validation_commands_override:
            return (
                [
                    ValidationCommandSpec(
                        command=command,
                        argv=normalize_shell_free_command(command),
                        timeout_seconds=self.config.validation.timeout_seconds,
                    )
                    for command in run.validation_commands_override
                ],
                [],
            )

        policy_pack = self._policy_pack_for_run(run)
        if policy_pack is not None:
            return policy_pack.validation_commands, policy_pack.forbidden_commands

        return (
            [
                ValidationCommandSpec(
                    command=command,
                    argv=normalize_shell_free_command(command),
                    timeout_seconds=self.config.validation.timeout_seconds,
                )
                for command in self.config.validation.commands
            ],
            [],
        )

    def run(self, *, objective: str, source_kind: TaskSourceKind) -> RunOutcome:
        planner = self._required_model("planner")
        run_id = generate_run_id()
        task_id = _prefixed_id("task")
        planning_stage_id = _prefixed_id("stage")
        checkpoint_id = _prefixed_id("checkpoint")
        prompt_artifact_id = _prefixed_id("artifact")
        response_artifact_id = _prefixed_id("artifact")
        plan_artifact_id = _prefixed_id("artifact")
        checkpoint_artifact_id = _prefixed_id("artifact")

        planning_prompt = self._planning_prompt(objective)
        prompt_artifact = self.artifact_store.write_text(
            run_id=run_id,
            artifact_id=prompt_artifact_id,
            artifact_type="planning_prompt",
            content=planning_prompt,
            suffix=".md",
            metadata={"stage_name": "planning", "role": ModelRole.PLANNER.value},
        )
        checkpoint_artifact = capture_workspace_snapshot(
            self.cwd,
            run_id=run_id,
            artifact_id=checkpoint_artifact_id,
            artifact_store=self.artifact_store,
        )
        checkpoint = WorkspaceCheckpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            task_id=task_id,
            iteration=0,
            checkpoint_kind=CheckpointKind.TASK_BASELINE,
            capture_mode=CaptureMode.SAFE_PATH_SNAPSHOT,
            scope_paths=["."],
            artifact_id=checkpoint_artifact.artifact_id,
        )
        run = Run(
            run_id=run_id,
            repo_path=str(self.cwd),
            policy_pack_path=str(self.policy_pack.directory if self.policy_pack is not None else self.config_path.parent),
            policy_schema_ver=self.policy_pack.schema_ver if self.policy_pack is not None else POLICY_SCHEMA_VERSION,
            requested_objective=objective,
            state=RunState.CREATED,
            operator_status=OperatorStatus.NONE,
            max_iterations=self.config.execution.max_iterations,
            budget_limit_usd=self.config.execution.budget_limit_usd,
        )
        task = Task(
            task_id=task_id,
            run_id=run_id,
            sequence=0,
            source_kind=source_kind,
            raw_input=objective,
            status=TaskStatus.IN_PROGRESS,
        )
        planning_stage = Stage(
            stage_id=planning_stage_id,
            run_id=run_id,
            task_id=task_id,
            stage_name="planning",
            status=StageStatus.RUNNING,
            model_role=ModelRole.PLANNER,
            model_name=planner.model,
            provider=planner.provider,
            runtime_agent=planner.agent,
            runtime_variant=planner.variant,
            prompt_artifact_id=prompt_artifact_id,
            response_artifact_id=response_artifact_id,
            output_artifact_id=plan_artifact_id,
            iteration=1,
        )

        final_state = RunState.FAILED
        final_operator_status = OperatorStatus.NONE
        final_hold_type: str | None = None
        final_hold_reason: str | None = None
        final_hold_context: dict[str, object] | None = None

        with self._open_store() as store:
            store.runs.create(run)
            store.tasks.create(task)
            store.stages.create(planning_stage)
            store.artifacts.create(checkpoint_artifact)
            store.checkpoints.create(checkpoint)
            store.artifacts.create(prompt_artifact)

            with self._lease_guard(store, run_id):
                try:
                    planning_result = invoke_text_prompt(planner, prompt=planning_prompt, cwd=self.cwd)
                    planning_status = StageStatus.COMPLETED
                    planning_summary = "Planner response persisted for operator review."
                except ModelInvocationError as error:
                    planning_result = error.result
                    planning_status = StageStatus.FAILED
                    planning_summary = str(error)

                response_content = _planning_response_text(planning_result, planning_summary)

                response_artifact = self.artifact_store.write_text(
                    run_id=run_id,
                    artifact_id=response_artifact_id,
                    artifact_type="planning_response",
                    content=response_content + "\n",
                    suffix=".md",
                    metadata={
                        "stage_name": "planning",
                        "provider": planner.provider,
                        "model": planner.model,
                    },
                )
                plan_artifact = self.artifact_store.write_text(
                    run_id=run_id,
                    artifact_id=plan_artifact_id,
                    artifact_type="plan_review",
                    content=_render_plan_review_markdown(
                        objective=objective,
                        response_text=response_content,
                    ),
                    suffix=".md",
                    metadata={"stage_name": "planning"},
                )
                store.artifacts.create(response_artifact)
                store.artifacts.create(plan_artifact)

                if planning_result is not None:
                    if planning_result.estimated_cost_usd is not None:
                        run = run.model_copy(update={"total_cost_usd": planning_result.estimated_cost_usd})
                        store.runs.add_total_cost(
                            run_id,
                            delta_cost_usd=planning_result.estimated_cost_usd,
                        )
                    self._record_model_call(
                        store=store,
                        run_id=run_id,
                        stage=planning_stage,
                        profile=planner,
                        role=ModelRole.PLANNER,
                        result=planning_result,
                        status="completed" if planning_status == StageStatus.COMPLETED else "failed",
                        error_reason=planning_summary if planning_status == StageStatus.FAILED else None,
                    )

                self._transition(
                    store=store,
                    run_id=run_id,
                    from_state=RunState.CREATED,
                    to_state=RunState.PLANNING,
                    reason="run started",
                )

                store.stages.complete(
                    planning_stage_id,
                    status=planning_status.value,
                    output_artifact_id=plan_artifact.artifact_id,
                    summary=planning_summary,
                )

                if planning_status == StageStatus.FAILED:
                    self._transition(
                        store=store,
                        run_id=run_id,
                        from_state=RunState.PLANNING,
                        to_state=RunState.FAILED,
                        reason=planning_summary,
                    )
                    store.tasks.update_status(task_id, status=TaskStatus.FAILED.value)
                    final_state = RunState.FAILED
                    final_operator_status = OperatorStatus.NONE
                else:
                    self._transition(
                        store=store,
                        run_id=run_id,
                        from_state=RunState.PLANNING,
                        to_state=RunState.PLAN_REVIEW,
                        reason="planner response persisted",
                    )
                    self._transition(
                        store=store,
                        run_id=run_id,
                        from_state=RunState.PLAN_REVIEW,
                        to_state=RunState.AWAITING_OPERATOR,
                        reason="awaiting operator review of planner output",
                        operator_status=OperatorStatus.HOLD,
                        hold_type=HoldType.PLAN_REVIEW,
                        hold_context={
                            "planning_stage_id": planning_stage_id,
                            "response_artifact_id": response_artifact.artifact_id,
                            "plan_artifact_id": plan_artifact.artifact_id,
                        },
                    )
                    final_state = RunState.AWAITING_OPERATOR
                    final_operator_status = OperatorStatus.HOLD
                    final_hold_type = HoldType.PLAN_REVIEW.value
                    final_hold_reason = "awaiting operator review of planner output"
                    final_hold_context = {
                        "planning_stage_id": planning_stage_id,
                        "response_artifact_id": response_artifact.artifact_id,
                        "plan_artifact_id": plan_artifact.artifact_id,
                    }

        outcome = RunOutcome(
            run_id=run_id,
            task_id=task_id,
            planning_stage_id=planning_stage_id,
            checkpoint_id=checkpoint_id,
            final_state=final_state,
            operator_status=final_operator_status,
            hold_type=final_hold_type,
            hold_reason=final_hold_reason,
            hold_context=final_hold_context,
            database_path=self.database_path,
            artifacts_root=self.artifacts_root,
        )
        self._sync_retrieval_for_run(run_id)
        return outcome

    def resume(
        self,
        *,
        run_id: str,
        checkpoint_id: str | None = None,
        restore_checkpoint: bool = False,
    ) -> ResumeOutcome:
        def finalize(outcome: ResumeOutcome) -> ResumeOutcome:
            self._sync_retrieval_for_run(run_id)
            return outcome

        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            if run.state == RunState.DOING and store.run_leases.get_active_by_run(run_id) is None:
                return self._resume_interrupted_doing_run(
                    store=store,
                    run=run,
                    checkpoint_id=checkpoint_id,
                    restore_checkpoint=restore_checkpoint,
                    finalize=finalize,
                )
            if run.state != RunState.AWAITING_OPERATOR:
                raise ValueError(f"run is not awaiting operator input: {run.state.value}")

            tasks = store.tasks.list_by_run(run_id)
            if not tasks:
                raise ValueError(f"run has no tasks: {run_id}")
            task = tasks[-1]
            restored_checkpoint_id: str | None = None
            if restore_checkpoint:
                restored_checkpoint_id = self._restore_checkpoint_in_store(
                    store=store,
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                )
            hold_transition = self._latest_hold_transition(store, run_id)
            resuming_from_plan_review = (
                run.current_hold_type == HoldType.PLAN_REVIEW
                or (hold_transition is None or hold_transition.from_state == RunState.PLAN_REVIEW)
            )

            def make_resume_outcome(
                *,
                validation_stage_id: str,
                final_state: RunState,
                operator_status: OperatorStatus,
            ) -> ResumeOutcome:
                updated_run = self._resolve_run(store, run_id)
                return finalize(ResumeOutcome(
                    run_id=run_id,
                    task_id=task.task_id,
                    validation_stage_id=validation_stage_id,
                    final_state=final_state,
                    operator_status=operator_status,
                    hold_type=updated_run.current_hold_type.value if updated_run.current_hold_type else None,
                    hold_reason=self._current_hold_reason(store, updated_run.run_id),
                    hold_context=updated_run.current_hold_context,
                    restored_checkpoint_id=restored_checkpoint_id,
                    database_path=self.database_path,
                    artifacts_root=self.artifacts_root,
                ))

            try:
                with self._lease_guard(store, run_id):
                    store.operator_actions.create(
                        OperatorActionRecord(
                            action_id=_prefixed_id("action"),
                            run_id=run_id,
                            action_type=OperatorActionType.RESUME,
                            note="Operator resumed the held run.",
                            checkpoint_id=restored_checkpoint_id,
                        )
                    )
                    previous_actionable_ids = self._actionable_stable_ids_for_iteration(
                        store=store,
                        run_id=run_id,
                        iteration=run.iteration_count,
                    )
                    iteration = max(run.iteration_count + 1, 1)

                    if resuming_from_plan_review:
                        self._transition(
                            store=store,
                            run_id=run_id,
                            from_state=RunState.AWAITING_OPERATOR,
                            to_state=RunState.PLAN_READY,
                            reason="operator approved plan review",
                            operator_status=OperatorStatus.NONE,
                        )
                        self._transition(
                            store=store,
                            run_id=run_id,
                            from_state=RunState.PLAN_READY,
                            to_state=RunState.DOING,
                            reason="resume entered implementer do stage",
                            operator_status=OperatorStatus.NONE,
                        )
                        try:
                            do_output = self._run_do_stage(
                                store=store,
                                run=run,
                                task=task,
                                iteration=iteration,
                            )
                        except ValueError as error:
                            self._transition(
                                store=store,
                                run_id=run_id,
                                from_state=RunState.DOING,
                                to_state=RunState.FAILED,
                                reason=str(error),
                            )
                            store.tasks.update_status(task.task_id, status=TaskStatus.FAILED.value)
                            raise

                        self._transition(
                            store=store,
                            run_id=run_id,
                            from_state=RunState.DOING,
                            to_state=RunState.AWAITING_VALIDATION,
                            reason=f"do stage completed with {len(do_output.touched_files)} touched files",
                        )
                        observed_delta = self._observed_delta_from_do_output(do_output)
                    else:
                        self._transition(
                            store=store,
                            run_id=run_id,
                            from_state=RunState.AWAITING_OPERATOR,
                            to_state=RunState.AWAITING_VALIDATION,
                            reason="operator resumed held run for revalidation",
                            operator_status=OperatorStatus.NONE,
                        )
                        observed_delta = {
                            "source_stage": "operator_resume",
                            "resumed_from_state": hold_transition.from_state.value if hold_transition else None,
                            "hold_reason": hold_transition.reason if hold_transition else None,
                            "artifact_refs": [],
                        }

                    while True:
                        store.runs.set_iteration_count(run_id, iteration_count=iteration)
                        run = self._resolve_run(store, run_id)
                        cycle_status, validation_stage_id, actionable_findings = self._run_validation_review_cycle(
                            store=store,
                            run=run,
                            task=task,
                            iteration=iteration,
                            observed_delta=observed_delta,
                        )
                        if cycle_status == "hold":
                            return make_resume_outcome(
                                validation_stage_id=validation_stage_id,
                                final_state=RunState.AWAITING_OPERATOR,
                                operator_status=OperatorStatus.HOLD,
                            )
                        if cycle_status == "passed":
                            self._transition(
                                store=store,
                                run_id=run_id,
                                from_state=RunState.CHECK_PASSED,
                                to_state=RunState.COMPLETED,
                                reason="review cycle completed without actionable findings",
                                operator_status=OperatorStatus.NONE,
                            )
                            store.tasks.update_status(task.task_id, status=TaskStatus.COMPLETED.value)
                            return make_resume_outcome(
                                validation_stage_id=validation_stage_id,
                                final_state=RunState.COMPLETED,
                                operator_status=OperatorStatus.NONE,
                            )

                        actionable_ids = {finding.stable_id for finding in actionable_findings}
                        run = self._resolve_run(store, run_id)
                        if self._should_hold_for_diminishing_returns(previous_actionable_ids, actionable_ids):
                            self._transition(
                                store=store,
                                run_id=run_id,
                                from_state=RunState.CHECK_FAILED,
                                to_state=RunState.AWAITING_OPERATOR,
                                reason=(
                                    f"{HoldType.DIMINISHING_RETURNS.value}: actionable findings repeated across iterations; "
                                    "operator review required"
                                ),
                                operator_status=OperatorStatus.HOLD,
                                hold_type=HoldType.DIMINISHING_RETURNS,
                                hold_context={
                                    "iteration": iteration,
                                    "stable_ids": sorted(actionable_ids),
                                },
                            )
                            store.tasks.update_status(task.task_id, status=TaskStatus.IN_PROGRESS.value)
                            return make_resume_outcome(
                                validation_stage_id=validation_stage_id,
                                final_state=RunState.AWAITING_OPERATOR,
                                operator_status=OperatorStatus.HOLD,
                            )
                        if iteration >= run.max_iterations:
                            self._transition(
                                store=store,
                                run_id=run_id,
                                from_state=RunState.CHECK_FAILED,
                                to_state=RunState.AWAITING_OPERATOR,
                                reason=(
                                    f"{HoldType.MAX_ITERATIONS.value}: hit max iterations ({run.max_iterations}); "
                                    "operator review required"
                                ),
                                operator_status=OperatorStatus.HOLD,
                                hold_type=HoldType.MAX_ITERATIONS,
                                hold_context={
                                    "iteration": iteration,
                                    "max_iterations": run.max_iterations,
                                    "stable_ids": sorted(actionable_ids),
                                },
                            )
                            store.tasks.update_status(task.task_id, status=TaskStatus.IN_PROGRESS.value)
                            return make_resume_outcome(
                                validation_stage_id=validation_stage_id,
                                final_state=RunState.AWAITING_OPERATOR,
                                operator_status=OperatorStatus.HOLD,
                            )
                        if run.budget_limit_usd is not None and run.total_cost_usd >= run.budget_limit_usd:
                            self._transition(
                                store=store,
                                run_id=run_id,
                                from_state=RunState.CHECK_FAILED,
                                to_state=RunState.AWAITING_OPERATOR,
                                reason=(
                                    f"{HoldType.BUDGET_EXCEEDED.value}: run exceeded budget limit; operator review required"
                                ),
                                operator_status=OperatorStatus.HOLD,
                                hold_type=HoldType.BUDGET_EXCEEDED,
                                hold_context={
                                    "iteration": iteration,
                                    "total_cost_usd": str(run.total_cost_usd),
                                    "budget_limit_usd": str(run.budget_limit_usd),
                                },
                            )
                            store.tasks.update_status(task.task_id, status=TaskStatus.IN_PROGRESS.value)
                            return make_resume_outcome(
                                validation_stage_id=validation_stage_id,
                                final_state=RunState.AWAITING_OPERATOR,
                                operator_status=OperatorStatus.HOLD,
                            )

                        act_stage_id = _prefixed_id("stage")
                        self._capture_pre_act_checkpoint(
                            store=store,
                            run_id=run_id,
                            task_id=task.task_id,
                            stage_id=act_stage_id,
                            iteration=iteration,
                        )
                        self._transition(
                            store=store,
                            run_id=run_id,
                            from_state=RunState.CHECK_FAILED,
                            to_state=RunState.ACTING,
                            reason=f"entering act stage for {len(actionable_findings)} actionable findings",
                        )
                        try:
                            act_output = self._run_act_stage(
                                store=store,
                                run=run,
                                task=task,
                                stage_id=act_stage_id,
                                iteration=iteration,
                                findings=actionable_findings,
                            )
                        except ValueError as error:
                            self._transition(
                                store=store,
                                run_id=run_id,
                                from_state=RunState.ACTING,
                                to_state=RunState.FAILED,
                                reason=str(error),
                            )
                            store.tasks.update_status(task.task_id, status=TaskStatus.FAILED.value)
                            raise

                        self._transition(
                            store=store,
                            run_id=run_id,
                            from_state=RunState.ACTING,
                            to_state=RunState.AWAITING_VALIDATION,
                            reason="act stage completed; rerunning validation",
                        )
                        previous_actionable_ids = actionable_ids
                        observed_delta = self._observed_delta_from_act_output(act_output)
                        iteration += 1
            except LeaseConflictError as error:
                store.runs.set_operator_status(run_id, operator_status=OperatorStatus.HOLD.value)
                store.runs.set_hold_context(
                    run_id,
                    current_hold_type=HoldType.LEASE_CONFLICT.value,
                    current_hold_reason=str(error),
                    current_hold_context={"worker_id": self.worker_id},
                )
                return make_resume_outcome(
                    validation_stage_id="",
                    final_state=RunState.AWAITING_OPERATOR,
                    operator_status=OperatorStatus.HOLD,
                )

        raise RuntimeError(f"resume terminated unexpectedly for run {run_id}")

    def _resume_interrupted_doing_run(
        self,
        *,
        store: object,
        run: Run,
        checkpoint_id: str | None,
        restore_checkpoint: bool,
        finalize,
    ) -> ResumeOutcome:
        tasks = store.tasks.list_by_run(run.run_id)
        if not tasks:
            raise ValueError(f"run has no tasks: {run.run_id}")
        task = tasks[-1]
        restored_checkpoint_id: str | None = None
        if restore_checkpoint:
            restored_checkpoint_id = self._restore_checkpoint_in_store(
                store=store,
                run_id=run.run_id,
                checkpoint_id=checkpoint_id,
            )

        def make_resume_outcome(
            *,
            validation_stage_id: str,
            final_state: RunState,
            operator_status: OperatorStatus,
        ) -> ResumeOutcome:
            updated_run = self._resolve_run(store, run.run_id)
            return finalize(ResumeOutcome(
                run_id=run.run_id,
                task_id=task.task_id,
                validation_stage_id=validation_stage_id,
                final_state=final_state,
                operator_status=operator_status,
                hold_type=updated_run.current_hold_type.value if updated_run.current_hold_type else None,
                hold_reason=self._current_hold_reason(store, updated_run.run_id),
                hold_context=updated_run.current_hold_context,
                restored_checkpoint_id=restored_checkpoint_id,
                database_path=self.database_path,
                artifacts_root=self.artifacts_root,
            ))

        with self._lease_guard(store, run.run_id):
            store.operator_actions.create(
                OperatorActionRecord(
                    action_id=_prefixed_id("action"),
                    run_id=run.run_id,
                    action_type=OperatorActionType.RESUME,
                    note="Recovered interrupted do stage.",
                    checkpoint_id=restored_checkpoint_id,
                )
            )

            interrupted_stage = self._latest_stage_by_name(store=store, run_id=run.run_id, stage_name="doing")
            if interrupted_stage is not None and interrupted_stage.status == StageStatus.RUNNING:
                store.stages.complete(
                    interrupted_stage.stage_id,
                    status=StageStatus.FAILED.value,
                    summary="interrupted do stage recovered by resume",
                )

            previous_actionable_ids = self._actionable_stable_ids_for_iteration(
                store=store,
                run_id=run.run_id,
                iteration=run.iteration_count,
            )
            iteration = max(run.iteration_count + 1, 1)
            try:
                do_output = self._run_do_stage(
                    store=store,
                    run=run,
                    task=task,
                    iteration=iteration,
                )
            except ValueError as error:
                self._transition(
                    store=store,
                    run_id=run.run_id,
                    from_state=RunState.DOING,
                    to_state=RunState.FAILED,
                    reason=str(error),
                )
                store.tasks.update_status(task.task_id, status=TaskStatus.FAILED.value)
                raise

            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.DOING,
                to_state=RunState.AWAITING_VALIDATION,
                reason="recovered interrupted do stage and resumed validation cycle",
            )
            observed_delta = self._observed_delta_from_do_output(do_output)

            while True:
                store.runs.set_iteration_count(run.run_id, iteration_count=iteration)
                updated_run = self._resolve_run(store, run.run_id)
                cycle_status, validation_stage_id, actionable_findings = self._run_validation_review_cycle(
                    store=store,
                    run=updated_run,
                    task=task,
                    iteration=iteration,
                    observed_delta=observed_delta,
                )
                if cycle_status == "hold":
                    return make_resume_outcome(
                        validation_stage_id=validation_stage_id,
                        final_state=RunState.AWAITING_OPERATOR,
                        operator_status=OperatorStatus.HOLD,
                    )
                if cycle_status == "passed":
                    self._transition(
                        store=store,
                        run_id=run.run_id,
                        from_state=RunState.CHECK_PASSED,
                        to_state=RunState.COMPLETED,
                        reason="review cycle completed without actionable findings",
                        operator_status=OperatorStatus.NONE,
                    )
                    store.tasks.update_status(task.task_id, status=TaskStatus.COMPLETED.value)
                    return make_resume_outcome(
                        validation_stage_id=validation_stage_id,
                        final_state=RunState.COMPLETED,
                        operator_status=OperatorStatus.NONE,
                    )

                actionable_ids = {finding.stable_id for finding in actionable_findings}
                updated_run = self._resolve_run(store, run.run_id)
                if self._should_hold_for_diminishing_returns(previous_actionable_ids, actionable_ids):
                    self._transition(
                        store=store,
                        run_id=run.run_id,
                        from_state=RunState.CHECK_FAILED,
                        to_state=RunState.AWAITING_OPERATOR,
                        reason=(
                            f"{HoldType.DIMINISHING_RETURNS.value}: actionable findings repeated across iterations; "
                            "operator review required"
                        ),
                        operator_status=OperatorStatus.HOLD,
                        hold_type=HoldType.DIMINISHING_RETURNS,
                        hold_context={
                            "iteration": iteration,
                            "stable_ids": sorted(actionable_ids),
                        },
                    )
                    store.tasks.update_status(task.task_id, status=TaskStatus.IN_PROGRESS.value)
                    return make_resume_outcome(
                        validation_stage_id=validation_stage_id,
                        final_state=RunState.AWAITING_OPERATOR,
                        operator_status=OperatorStatus.HOLD,
                    )
                if iteration >= updated_run.max_iterations:
                    self._transition(
                        store=store,
                        run_id=run.run_id,
                        from_state=RunState.CHECK_FAILED,
                        to_state=RunState.AWAITING_OPERATOR,
                        reason=(
                            f"{HoldType.MAX_ITERATIONS.value}: hit max iterations ({updated_run.max_iterations}); "
                            "operator review required"
                        ),
                        operator_status=OperatorStatus.HOLD,
                        hold_type=HoldType.MAX_ITERATIONS,
                        hold_context={
                            "iteration": iteration,
                            "max_iterations": updated_run.max_iterations,
                            "stable_ids": sorted(actionable_ids),
                        },
                    )
                    store.tasks.update_status(task.task_id, status=TaskStatus.IN_PROGRESS.value)
                    return make_resume_outcome(
                        validation_stage_id=validation_stage_id,
                        final_state=RunState.AWAITING_OPERATOR,
                        operator_status=OperatorStatus.HOLD,
                    )
                if updated_run.budget_limit_usd is not None and updated_run.total_cost_usd >= updated_run.budget_limit_usd:
                    self._transition(
                        store=store,
                        run_id=run.run_id,
                        from_state=RunState.CHECK_FAILED,
                        to_state=RunState.AWAITING_OPERATOR,
                        reason=(
                            f"{HoldType.BUDGET_EXCEEDED.value}: run exceeded budget limit; operator review required"
                        ),
                        operator_status=OperatorStatus.HOLD,
                        hold_type=HoldType.BUDGET_EXCEEDED,
                        hold_context={
                            "iteration": iteration,
                            "total_cost_usd": str(updated_run.total_cost_usd),
                            "budget_limit_usd": str(updated_run.budget_limit_usd),
                        },
                    )
                    store.tasks.update_status(task.task_id, status=TaskStatus.IN_PROGRESS.value)
                    return make_resume_outcome(
                        validation_stage_id=validation_stage_id,
                        final_state=RunState.AWAITING_OPERATOR,
                        operator_status=OperatorStatus.HOLD,
                    )

                act_stage_id = _prefixed_id("stage")
                self._capture_pre_act_checkpoint(
                    store=store,
                    run_id=run.run_id,
                    task_id=task.task_id,
                    stage_id=act_stage_id,
                    iteration=iteration,
                )
                self._transition(
                    store=store,
                    run_id=run.run_id,
                    from_state=RunState.CHECK_FAILED,
                    to_state=RunState.ACTING,
                    reason=f"entering act stage for {len(actionable_findings)} actionable findings",
                )
                try:
                    act_output = self._run_act_stage(
                        store=store,
                        run=updated_run,
                        task=task,
                        stage_id=act_stage_id,
                        iteration=iteration,
                        findings=actionable_findings,
                    )
                except ValueError as error:
                    self._transition(
                        store=store,
                        run_id=run.run_id,
                        from_state=RunState.ACTING,
                        to_state=RunState.FAILED,
                        reason=str(error),
                    )
                    store.tasks.update_status(task.task_id, status=TaskStatus.FAILED.value)
                    raise

                self._transition(
                    store=store,
                    run_id=run.run_id,
                    from_state=RunState.ACTING,
                    to_state=RunState.AWAITING_VALIDATION,
                    reason="act stage completed; rerunning validation",
                )
                previous_actionable_ids = actionable_ids
                observed_delta = self._observed_delta_from_act_output(act_output)
                iteration += 1

    def restore_checkpoint(self, *, run_id: str, checkpoint_id: str | None = None) -> WorkspaceCheckpoint:
        with self._open_store() as store:
            restored_checkpoint_id = self._restore_checkpoint_in_store(
                store=store,
                run_id=run_id,
                checkpoint_id=checkpoint_id,
            )
            checkpoint = store.checkpoints.get(restored_checkpoint_id)
            if checkpoint is None:
                raise ValueError(f"checkpoint not found after restore: {restored_checkpoint_id}")
            return checkpoint

    def status(self, *, run_id: str | None = None) -> dict[str, object]:
        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            tasks = store.tasks.list_by_run(run.run_id)
            stages = store.stages.list_by_run(run.run_id)
            findings = store.findings.list_by_run(run.run_id)
            validations = store.validations.list_by_run(run.run_id)
            checkpoints = store.checkpoints.list_by_run(run.run_id)
            retrieval_queue = store.retrieval_queue.list_by_run(run.run_id)
            hold = self._current_hold_snapshot(store, run.run_id)
            return {
                "run_id": run.run_id,
                "state": run.state.value,
                "operator_status": run.operator_status.value,
                "hold_type": hold["hold_type"],
                "hold_reason": hold["hold_reason"],
                "hold_context": hold["hold_context"],
                "validation_commands_override": run.validation_commands_override,
                "effective_validation_commands": [spec.command for spec in self._validation_specs_for_run(run)[0]],
                "objective": run.requested_objective,
                "iteration_count": run.iteration_count,
                "max_iterations": run.max_iterations,
                "budget_limit_usd": str(run.budget_limit_usd) if run.budget_limit_usd is not None else None,
                "total_cost_usd": str(run.total_cost_usd),
                "model_call_count": len(store.model_calls.list_by_run(run.run_id)),
                "semantic_audit_count": len(store.semantic_audits.list_by_run(run.run_id)),
                "adjudication_panel_count": len(store.adjudication_panels.list_by_run(run.run_id)),
                "retrieval_queue_count": len(retrieval_queue),
                "retrieval_pending_count": sum(1 for record in retrieval_queue if record.queue_status == "pending"),
                "task_count": len(tasks),
                "stage_count": len(stages),
                "finding_count": len(findings),
                "open_finding_count": sum(1 for finding in findings if finding.status == FindingStatus.OPEN),
                "validation_count": len(validations),
                "checkpoint_count": len(checkpoints),
                "latest_checkpoint_id": checkpoints[-1].checkpoint_id if checkpoints else None,
                "updated_at": run.updated_at.isoformat(),
            }

    def report(self, *, run_id: str | None = None) -> dict[str, object]:
        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            model_calls = store.model_calls.list_by_run(run.run_id)
            semantic_audits = store.semantic_audits.list_by_run(run.run_id)
            panels = store.adjudication_panels.list_by_run(run.run_id)
            votes = store.adjudication_votes.list_by_run(run.run_id)
            retrieval_queue = store.retrieval_queue.list_by_run(run.run_id)

        cost_by_role: dict[str, dict[str, object]] = defaultdict(lambda: {"count": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": Decimal("0")})
        cost_by_model: dict[str, dict[str, object]] = defaultdict(lambda: {"count": 0, "estimated_cost_usd": Decimal("0")})
        for record in model_calls:
            role_key = record.model_role.value
            role_entry = cost_by_role[role_key]
            role_entry["count"] += 1
            role_entry["input_tokens"] += record.input_tokens or 0
            role_entry["output_tokens"] += record.output_tokens or 0
            role_entry["estimated_cost_usd"] += record.estimated_cost_usd or Decimal("0")

            model_key = f"{record.provider}:{record.model_name}"
            model_entry = cost_by_model[model_key]
            model_entry["count"] += 1
            model_entry["estimated_cost_usd"] += record.estimated_cost_usd or Decimal("0")

        retrieval_documents = self.retrieval_service.collection_stats(run_id=run.run_id) if self.retrieval_service else {}
        retrieval_readiness = self.retrieval_service.collection_readiness(run_id=run.run_id) if self.retrieval_service else {}
        return {
            "run_id": run.run_id,
            "state": run.state.value,
            "model_calls": {
                "total": len(model_calls),
                "by_role": {
                    key: {
                        "count": value["count"],
                        "input_tokens": value["input_tokens"],
                        "output_tokens": value["output_tokens"],
                        "estimated_cost_usd": str(value["estimated_cost_usd"]),
                    }
                    for key, value in cost_by_role.items()
                },
                "by_model": {
                    key: {
                        "count": value["count"],
                        "estimated_cost_usd": str(value["estimated_cost_usd"]),
                    }
                    for key, value in cost_by_model.items()
                },
            },
            "semantic_audits": {
                "total": len(semantic_audits),
                "hard_fail_count": sum(1 for record in semantic_audits if record.hard_fail),
                "blocking_finding_count": sum(len(record.blocking_findings) for record in semantic_audits),
            },
            "adjudication": {
                "panel_count": len(panels),
                "vote_count": len(votes),
                "split_panel_count": sum(1 for panel in panels if panel.status == "split"),
            },
            "retrieval": {
                "enabled": self.retrieval_service is not None,
                "queue_total": len(retrieval_queue),
                "queue_pending": sum(1 for record in retrieval_queue if record.queue_status == "pending"),
                "queue_failed": sum(1 for record in retrieval_queue if record.queue_status == "failed"),
                "documents_by_collection": retrieval_documents,
                "readiness_by_collection": retrieval_readiness,
            },
        }

    def findings(self, *, run_id: str | None = None) -> list[dict[str, object]]:
        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            findings = store.findings.list_by_run(run.run_id)
            return [
                {
                    "finding_id": finding.finding_id,
                    "stage_id": finding.stage_id,
                    "severity": finding.severity.value,
                    "status": finding.status.value,
                    "category": finding.category.value,
                    "title": finding.title,
                    "rationale": finding.rationale,
                    "exact_fix_action": finding.exact_fix_action,
                }
                for finding in findings
            ]

    def approve(
        self,
        *,
        run_id: str,
        note: str | None = None,
        max_iterations: int | None = None,
        budget_limit_usd: Decimal | None = None,
    ) -> dict[str, object]:
        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            if run.state != RunState.AWAITING_OPERATOR:
                raise ValueError(f"run is not awaiting operator input: {run.state.value}")
            if max_iterations is not None and max_iterations < max(run.iteration_count, 1):
                raise ValueError("max_iterations cannot be set below the current iteration progress")

            store.operator_actions.create(
                OperatorActionRecord(
                    action_id=_prefixed_id("action"),
                    run_id=run_id,
                    action_type=OperatorActionType.APPROVE,
                    note=note or "Operator approved the current hold state.",
                    payload={
                        "max_iterations": max_iterations,
                        "budget_limit_usd": str(budget_limit_usd) if budget_limit_usd is not None else None,
                    },
                )
            )
            store.runs.set_operator_status(run_id, operator_status=OperatorStatus.APPROVED.value)
            store.runs.patch_limits(
                run_id,
                max_iterations=max_iterations,
                budget_limit_usd=budget_limit_usd,
            )
            updated_run = self._resolve_run(store, run_id)
            return {
                "run_id": updated_run.run_id,
                "state": updated_run.state.value,
                "operator_status": updated_run.operator_status.value,
                "hold_type": updated_run.current_hold_type.value if updated_run.current_hold_type else None,
                "hold_reason": self._current_hold_reason(store, updated_run.run_id),
                "hold_context": updated_run.current_hold_context,
                "iteration_count": updated_run.iteration_count,
                "max_iterations": updated_run.max_iterations,
                "budget_limit_usd": str(updated_run.budget_limit_usd) if updated_run.budget_limit_usd is not None else None,
                "total_cost_usd": str(updated_run.total_cost_usd),
            }

    def reject(self, *, run_id: str, note: str | None = None) -> dict[str, object]:
        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            if run.state != RunState.AWAITING_OPERATOR:
                raise ValueError(f"run is not awaiting operator input: {run.state.value}")
            if run.current_hold_type != HoldType.PLAN_REVIEW:
                raise ValueError("reject is only valid for plan-review holds")

            tasks = store.tasks.list_by_run(run_id)
            if not tasks:
                raise ValueError(f"run has no tasks: {run_id}")
            task = tasks[-1]
            planner = self._required_model("planner")
            planning_stage_id = _prefixed_id("stage")
            prompt_artifact_id = _prefixed_id("artifact")
            response_artifact_id = _prefixed_id("artifact")
            plan_artifact_id = _prefixed_id("artifact")
            planning_prompt = self._planning_prompt(task.raw_input)

            store.operator_actions.create(
                OperatorActionRecord(
                    action_id=_prefixed_id("action"),
                    run_id=run_id,
                    action_type=OperatorActionType.REJECT,
                    note=note or "Operator rejected planner output and requested replanning.",
                )
            )

            prompt_artifact = self.artifact_store.write_text(
                run_id=run_id,
                artifact_id=prompt_artifact_id,
                artifact_type="planning_prompt",
                content=planning_prompt,
                suffix=".md",
                metadata={"stage_name": "planning", "role": ModelRole.PLANNER.value, "replan": True},
            )
            planning_stage = Stage(
                stage_id=planning_stage_id,
                run_id=run_id,
                task_id=task.task_id,
                stage_name="planning",
                status=StageStatus.RUNNING,
                model_role=ModelRole.PLANNER,
                model_name=planner.model,
                provider=planner.provider,
                runtime_agent=planner.agent,
                runtime_variant=planner.variant,
                prompt_artifact_id=prompt_artifact_id,
                response_artifact_id=response_artifact_id,
                output_artifact_id=plan_artifact_id,
                iteration=max(run.iteration_count, 1),
            )
            store.artifacts.create(prompt_artifact)
            store.stages.create(planning_stage)
            self._transition(
                store=store,
                run_id=run_id,
                from_state=RunState.AWAITING_OPERATOR,
                to_state=RunState.PLANNING,
                reason=note or "operator rejected plan review; replanning requested",
                operator_status=OperatorStatus.NONE,
            )

            try:
                planning_result = invoke_text_prompt(planner, prompt=planning_prompt, cwd=self.cwd)
                planning_status = StageStatus.COMPLETED
                planning_summary = "Planner response persisted for operator review."
            except ModelInvocationError as error:
                planning_result = error.result
                planning_status = StageStatus.FAILED
                planning_summary = str(error)

            response_content = _planning_response_text(planning_result, planning_summary)

            response_artifact = self.artifact_store.write_text(
                run_id=run_id,
                artifact_id=response_artifact_id,
                artifact_type="planning_response",
                content=response_content + "\n",
                suffix=".md",
                metadata={"stage_name": "planning", "provider": planner.provider, "model": planner.model, "replan": True},
            )
            plan_artifact = self.artifact_store.write_text(
                run_id=run_id,
                artifact_id=plan_artifact_id,
                artifact_type="plan_review",
                content=_render_plan_review_markdown(
                    objective=task.raw_input,
                    response_text=response_content,
                    replan=True,
                ),
                suffix=".md",
                metadata={"stage_name": "planning", "replan": True},
            )
            store.artifacts.create(response_artifact)
            store.artifacts.create(plan_artifact)
            if planning_result is not None:
                self._record_model_call(
                    store=store,
                    run_id=run_id,
                    stage=planning_stage,
                    profile=planner,
                    role=ModelRole.PLANNER,
                    result=planning_result,
                    status="completed" if planning_status == StageStatus.COMPLETED else "failed",
                    error_reason=planning_summary if planning_status == StageStatus.FAILED else None,
                )
                self._record_invocation_cost(store=store, run_id=run_id, result=planning_result)

            store.stages.complete(
                planning_stage_id,
                status=planning_status.value,
                output_artifact_id=plan_artifact.artifact_id,
                summary=planning_summary,
            )
            if planning_status == StageStatus.FAILED:
                self._transition(
                    store=store,
                    run_id=run_id,
                    from_state=RunState.PLANNING,
                    to_state=RunState.FAILED,
                    reason=planning_summary,
                )
                store.tasks.update_status(task.task_id, status=TaskStatus.FAILED.value)
            else:
                self._transition(
                    store=store,
                    run_id=run_id,
                    from_state=RunState.PLANNING,
                    to_state=RunState.PLAN_REVIEW,
                    reason="replanned response persisted",
                )
                self._transition(
                    store=store,
                    run_id=run_id,
                    from_state=RunState.PLAN_REVIEW,
                    to_state=RunState.AWAITING_OPERATOR,
                    reason="awaiting operator review of replanned output",
                    operator_status=OperatorStatus.HOLD,
                    hold_type=HoldType.PLAN_REVIEW,
                    hold_context={
                        "planning_stage_id": planning_stage_id,
                        "response_artifact_id": response_artifact.artifact_id,
                        "plan_artifact_id": plan_artifact.artifact_id,
                        "replan": True,
                    },
                )

            updated_run = self._resolve_run(store, run_id)
            return {
                "run_id": updated_run.run_id,
                "state": updated_run.state.value,
                "operator_status": updated_run.operator_status.value,
                "hold_type": updated_run.current_hold_type.value if updated_run.current_hold_type else None,
                "hold_reason": updated_run.current_hold_reason,
                "hold_context": updated_run.current_hold_context,
            }

    def hold_reason(self, *, run_id: str) -> dict[str, object]:
        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            return {
                "run_id": run.run_id,
                "state": run.state.value,
                "operator_status": run.operator_status.value,
                "hold_type": run.current_hold_type.value if run.current_hold_type else None,
                "hold_reason": self._current_hold_reason(store, run.run_id),
                "hold_context": run.current_hold_context,
            }

    def patch_policy(
        self,
        *,
        run_id: str,
        note: str | None = None,
        max_iterations: int | None = None,
        budget_limit_usd: Decimal | None = None,
        validation_commands_override: list[str] | None = None,
    ) -> dict[str, object]:
        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            if run.state != RunState.AWAITING_OPERATOR:
                raise ValueError(f"run is not awaiting operator input: {run.state.value}")
            if (
                max_iterations is None
                and budget_limit_usd is None
                and validation_commands_override is None
            ):
                raise ValueError("patch-policy requires at least one override")
            if max_iterations is not None and max_iterations < max(run.iteration_count, 1):
                raise ValueError("max_iterations cannot be set below the current iteration progress")

            if max_iterations is not None or budget_limit_usd is not None:
                store.runs.patch_limits(
                    run_id,
                    max_iterations=max_iterations,
                    budget_limit_usd=budget_limit_usd,
                )
            if validation_commands_override is not None:
                store.runs.set_validation_commands_override(
                    run_id,
                    validation_commands_override=validation_commands_override,
                )

            updated_run = self._resolve_run(store, run_id)
            store.operator_actions.create(
                OperatorActionRecord(
                    action_id=_prefixed_id("action"),
                    run_id=run_id,
                    action_type=OperatorActionType.PATCH_POLICY,
                    note=note or "Operator patched run policy while held.",
                    payload={
                        "previous": {
                            "max_iterations": run.max_iterations,
                            "budget_limit_usd": str(run.budget_limit_usd) if run.budget_limit_usd is not None else None,
                            "validation_commands_override": run.validation_commands_override,
                        },
                        "effective": {
                            "max_iterations": updated_run.max_iterations,
                            "budget_limit_usd": str(updated_run.budget_limit_usd) if updated_run.budget_limit_usd is not None else None,
                            "validation_commands_override": updated_run.validation_commands_override,
                        },
                    },
                )
            )
            return {
                "run_id": updated_run.run_id,
                "state": updated_run.state.value,
                "operator_status": updated_run.operator_status.value,
                "hold_type": updated_run.current_hold_type.value if updated_run.current_hold_type else None,
                "hold_reason": self._current_hold_reason(store, updated_run.run_id),
                "hold_context": updated_run.current_hold_context,
                "validation_commands_override": updated_run.validation_commands_override,
                "effective_validation_commands": [spec.command for spec in self._validation_specs_for_run(updated_run)[0]],
                "iteration_count": updated_run.iteration_count,
                "max_iterations": updated_run.max_iterations,
                "budget_limit_usd": str(updated_run.budget_limit_usd) if updated_run.budget_limit_usd is not None else None,
                "total_cost_usd": str(updated_run.total_cost_usd),
            }

    def abort(self, *, run_id: str, note: str | None = None) -> dict[str, object]:
        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            if is_terminal(run.state):
                raise ValueError(f"run is already terminal: {run.state.value}")

            store.operator_actions.create(
                OperatorActionRecord(
                    action_id=_prefixed_id("action"),
                    run_id=run_id,
                    action_type=OperatorActionType.ABORT,
                    note=note or "Operator aborted the run.",
                )
            )
            self._transition(
                store=store,
                run_id=run_id,
                from_state=run.state,
                to_state=RunState.ABORTED,
                reason=note or "operator aborted run",
                operator_status=OperatorStatus.NONE,
            )
            tasks = store.tasks.list_by_run(run_id)
            if tasks:
                store.tasks.update_status(tasks[-1].task_id, status=TaskStatus.FAILED.value)
            aborted_run = self._resolve_run(store, run_id)
            return {
                "run_id": aborted_run.run_id,
                "state": aborted_run.state.value,
                "operator_status": aborted_run.operator_status.value,
            }

    def inspect(self, *, run_id: str | None = None, selector: str | None = None) -> dict[str, object]:
        with self._open_store() as store:
            run = self._resolve_run(store, run_id)
            artifacts = store.artifacts.list_by_run(run.run_id)
            if selector is None:
                return {
                    "run_id": run.run_id,
                    "artifacts": [self._artifact_summary(artifact) for artifact in artifacts],
                }

            if selector == "model_calls":
                return {
                    "run_id": run.run_id,
                    "model_calls": [record.model_dump(mode="json") for record in store.model_calls.list_by_run(run.run_id)],
                }
            if selector == "semantic_audits":
                return {
                    "run_id": run.run_id,
                    "semantic_audits": [record.model_dump(mode="json") for record in store.semantic_audits.list_by_run(run.run_id)],
                }
            if selector == "adjudication_panels":
                return {
                    "run_id": run.run_id,
                    "adjudication_panels": [record.model_dump(mode="json") for record in store.adjudication_panels.list_by_run(run.run_id)],
                }
            if selector == "adjudication_votes":
                return {
                    "run_id": run.run_id,
                    "adjudication_votes": [record.model_dump(mode="json") for record in store.adjudication_votes.list_by_run(run.run_id)],
                }
            if selector == "retrieval_queue":
                return {
                    "run_id": run.run_id,
                    "retrieval_queue": [record.model_dump(mode="json") for record in store.retrieval_queue.list_by_run(run.run_id)],
                }

            selected = next((artifact for artifact in artifacts if artifact.artifact_id == selector), None)
            if selected is None:
                matching = [artifact for artifact in artifacts if artifact.artifact_type == selector]
                if not matching:
                    raise ValueError(f"artifact not found for selector: {selector}")
                selected = matching[-1]
            return {
                "run_id": run.run_id,
                "artifact": self._artifact_summary(selected),
                "content": self._load_artifact_content(self.artifact_store.resolve_path(selected)),
            }

    def _run_do_stage(
        self,
        *,
        store: object,
        run: Run,
        task: Task,
        iteration: int,
    ) -> DoStageResponsePayload:
        implementer = self._required_model("implementer")
        stage_id = _prefixed_id("stage")
        prompt_artifact_id = _prefixed_id("artifact")
        response_artifact_id = _prefixed_id("artifact")
        output_artifact_id = _prefixed_id("artifact")
        prompt = self._do_prompt(objective=task.raw_input)
        prompt_artifact = self.artifact_store.write_text(
            run_id=run.run_id,
            artifact_id=prompt_artifact_id,
            artifact_type="do_prompt",
            content=prompt,
            suffix=".md",
            metadata={"stage_name": "doing", "role": ModelRole.IMPLEMENTER.value},
        )
        stage = Stage(
            stage_id=stage_id,
            run_id=run.run_id,
            task_id=task.task_id,
            stage_name="doing",
            status=StageStatus.RUNNING,
            model_role=ModelRole.IMPLEMENTER,
            model_name=implementer.model,
            provider=implementer.provider,
            runtime_agent=implementer.agent,
            runtime_variant=implementer.variant,
            prompt_artifact_id=prompt_artifact_id,
            response_artifact_id=response_artifact_id,
            output_artifact_id=output_artifact_id,
            iteration=iteration,
        )
        store.artifacts.create(prompt_artifact)
        store.stages.create(stage)

        try:
            result = invoke_text_prompt(implementer, prompt=prompt, cwd=self.cwd)
            self._record_model_call(
                store=store,
                run_id=run.run_id,
                stage=stage,
                profile=implementer,
                role=ModelRole.IMPLEMENTER,
                result=result,
                status="completed",
            )
            self._record_invocation_cost(store=store, run_id=run.run_id, result=result)
            response_artifact = self.artifact_store.write_text(
                run_id=run.run_id,
                artifact_id=response_artifact_id,
                artifact_type="do_response",
                content=result.stdout + "\n",
                suffix=".json",
                metadata={"stage_name": "doing", "provider": implementer.provider, "model": implementer.model},
            )
            payload = DoStageResponsePayload.model_validate(extract_json_output(result.stdout))
            output_artifact = self.artifact_store.write_json(
                run_id=run.run_id,
                artifact_id=output_artifact_id,
                artifact_type="do_output",
                payload=payload.model_dump(mode="json"),
                metadata={"stage_name": "doing"},
            )
            store.artifacts.create(response_artifact)
            store.artifacts.create(output_artifact)
            store.stages.complete(
                stage_id,
                status=StageStatus.COMPLETED.value,
                output_artifact_id=output_artifact.artifact_id,
                summary=f"Implementer reported {len(payload.touched_files)} touched files.",
            )
            return payload
        except Exception as error:
            if isinstance(error, ModelInvocationError) and error.result is not None:
                self._record_model_call(
                    store=store,
                    run_id=run.run_id,
                    stage=stage,
                    profile=implementer,
                    role=ModelRole.IMPLEMENTER,
                    result=error.result,
                    status="failed",
                    error_reason=str(error),
                )
            store.stages.complete(stage_id, status=StageStatus.FAILED.value, summary=str(error))
            raise ValueError(f"do stage failed: {error}") from error

    def _run_act_stage(
        self,
        *,
        store: object,
        run: Run,
        task: Task,
        stage_id: str,
        iteration: int,
        findings: list[Finding],
    ) -> ActStageResponsePayload:
        implementer = self._required_model("implementer")
        prompt_artifact_id = _prefixed_id("artifact")
        response_artifact_id = _prefixed_id("artifact")
        output_artifact_id = _prefixed_id("artifact")
        prompt = self._act_prompt(objective=task.raw_input, findings=findings)
        prompt_artifact = self.artifact_store.write_text(
            run_id=run.run_id,
            artifact_id=prompt_artifact_id,
            artifact_type="act_prompt",
            content=prompt,
            suffix=".md",
            metadata={"stage_name": "acting", "role": ModelRole.IMPLEMENTER.value},
        )
        stage = Stage(
            stage_id=stage_id,
            run_id=run.run_id,
            task_id=task.task_id,
            stage_name="acting",
            status=StageStatus.RUNNING,
            model_role=ModelRole.IMPLEMENTER,
            model_name=implementer.model,
            provider=implementer.provider,
            runtime_agent=implementer.agent,
            runtime_variant=implementer.variant,
            prompt_artifact_id=prompt_artifact_id,
            response_artifact_id=response_artifact_id,
            output_artifact_id=output_artifact_id,
            iteration=iteration,
        )
        store.artifacts.create(prompt_artifact)
        store.stages.create(stage)

        try:
            result = invoke_text_prompt(implementer, prompt=prompt, cwd=self.cwd)
            self._record_model_call(
                store=store,
                run_id=run.run_id,
                stage=stage,
                profile=implementer,
                role=ModelRole.IMPLEMENTER,
                result=result,
                status="completed",
            )
            self._record_invocation_cost(store=store, run_id=run.run_id, result=result)
            response_artifact = self.artifact_store.write_text(
                run_id=run.run_id,
                artifact_id=response_artifact_id,
                artifact_type="act_response",
                content=result.stdout + "\n",
                suffix=".json",
                metadata={"stage_name": "acting", "provider": implementer.provider, "model": implementer.model},
            )
            payload = ActStageResponsePayload.model_validate(extract_json_output(result.stdout))
            output_artifact = self.artifact_store.write_json(
                run_id=run.run_id,
                artifact_id=output_artifact_id,
                artifact_type="act_output",
                payload=payload.model_dump(mode="json"),
                metadata={"stage_name": "acting"},
            )
            store.artifacts.create(response_artifact)
            store.artifacts.create(output_artifact)
            store.stages.complete(
                stage_id,
                status=StageStatus.COMPLETED.value,
                output_artifact_id=output_artifact.artifact_id,
                summary=f"Act stage returned triage for {len(payload.triage)} findings.",
            )
            return payload
        except Exception as error:
            if isinstance(error, ModelInvocationError) and error.result is not None:
                self._record_model_call(
                    store=store,
                    run_id=run.run_id,
                    stage=stage,
                    profile=implementer,
                    role=ModelRole.IMPLEMENTER,
                    result=error.result,
                    status="failed",
                    error_reason=str(error),
                )
            store.stages.complete(stage_id, status=StageStatus.FAILED.value, summary=str(error))
            raise ValueError(f"act stage failed: {error}") from error

    def _run_validation_review_cycle(
        self,
        *,
        store: object,
        run: Run,
        task: Task,
        iteration: int,
        observed_delta: dict[str, object],
    ) -> tuple[str, str, list[Finding]]:
        self._transition(
            store=store,
            run_id=run.run_id,
            from_state=RunState.AWAITING_VALIDATION,
            to_state=RunState.VALIDATING,
            reason="running configured validation commands",
        )
        try:
            validation_stage_id, validation_results, any_failed = self._run_validation_stage(
                store=store,
                run=run,
                task=task,
                iteration=iteration,
            )
        except RiskyValidationCommandError as error:
            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.VALIDATING,
                to_state=RunState.AWAITING_OPERATOR,
                reason=error.reason,
                operator_status=OperatorStatus.HOLD,
                hold_type=HoldType.RISKY_COMMAND,
                hold_context={
                    "validation_stage_id": error.stage_id,
                    "command": error.command,
                },
            )
            store.tasks.update_status(task.task_id, status=TaskStatus.IN_PROGRESS.value)
            return "hold", error.stage_id, []
        self._transition(
            store=store,
            run_id=run.run_id,
            from_state=RunState.VALIDATING,
            to_state=RunState.SEMANTIC_AUDITING,
            reason="validation results persisted",
        )
        semantic_audit = self._run_semantic_audit_stage(
            store=store,
            run=run,
            task=task,
            iteration=iteration,
            observed_delta=observed_delta,
            validation_results=validation_results,
        )

        if any_failed:
            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.SEMANTIC_AUDITING,
                to_state=RunState.CHECK_FAILED,
                reason="validation failure synthesized as blocking finding",
            )
            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.CHECK_FAILED,
                to_state=RunState.AWAITING_OPERATOR,
                reason="awaiting operator after validation failure",
                operator_status=OperatorStatus.HOLD,
                hold_type=HoldType.VALIDATION_BLOCKED,
                hold_context={
                    "validation_stage_id": validation_stage_id,
                    "failed_commands": [result.command for result in validation_results if result.returncode != 0],
                },
            )
            store.tasks.update_status(task.task_id, status=TaskStatus.FAILED.value)
            return "hold", validation_stage_id, []

        semantic_findings = self._persist_semantic_audit_findings(
            store=store,
            run=run,
            semantic_audit=semantic_audit,
        )
        if semantic_audit.report.hard_fail and not self.config.adjudication.require_model_review_on_semantic_hard_fail:
            dispositions = self._persist_engine_dispositions(
                store=store,
                findings=semantic_findings,
                stage_id=semantic_audit.stage_id,
                iteration=iteration,
                justification="semantic audit hard-failed on authoritative workspace contradictions",
            )
            actionable_findings = finalized_actionable_finding_set(
                semantic_findings,
                dispositions,
                iteration=iteration,
            )
            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.SEMANTIC_AUDITING,
                to_state=RunState.CHECK_FAILED,
                reason=f"semantic audit hard-failed with {len(actionable_findings)} blocking findings",
            )
            return "actionable", validation_stage_id, actionable_findings

        if semantic_audit.report.hard_fail and semantic_findings:
            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.SEMANTIC_AUDITING,
                to_state=RunState.ADJUDICATING,
                reason=f"semantic audit hard-failed; routing {len(semantic_findings)} findings for adjudicator review",
            )
            adjudication_result = self._run_adjudicator_stage(
                store=store,
                run=run,
                task=task,
                findings=semantic_findings,
                iteration=iteration,
                semantic_audit=semantic_audit,
                evidence_verification=EvidenceVerificationReport(report_id=_prefixed_id("evidence_report")),
                evidence_verification_artifact_id=semantic_audit.report_artifact_id,
            )
            if adjudication_result.held_for_consensus:
                store.tasks.update_status(task.task_id, status=TaskStatus.IN_PROGRESS.value)
                return "hold", validation_stage_id, []
            actionable_findings = finalized_actionable_finding_set(
                adjudication_result.findings,
                adjudication_result.dispositions,
                iteration=iteration,
            )
            if actionable_findings:
                self._transition(
                    store=store,
                    run_id=run.run_id,
                    from_state=RunState.ADJUDICATING,
                    to_state=RunState.CHECK_FAILED,
                    reason=f"semantic audit adjudication left {len(actionable_findings)} actionable findings open",
                )
                return "actionable", validation_stage_id, actionable_findings
            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.ADJUDICATING,
                to_state=RunState.CHECK_PASSED,
                reason="semantic audit adjudication cleared all blocking findings",
            )
            return "passed", validation_stage_id, []

        self._transition(
            store=store,
            run_id=run.run_id,
            from_state=RunState.SEMANTIC_AUDITING,
            to_state=RunState.CHECKING,
            reason="invoking checker adapter",
        )
        try:
            checker_result = self._run_checker_stage(
                store=store,
                run=run,
                task=task,
                validation_results=validation_results,
                iteration=iteration,
                semantic_audit=semantic_audit,
            )
        except ValueError as error:
            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.CHECKING,
                to_state=RunState.FAILED,
                reason=str(error),
            )
            store.tasks.update_status(task.task_id, status=TaskStatus.FAILED.value)
            raise

        self._transition(
            store=store,
            run_id=run.run_id,
            from_state=RunState.CHECKING,
            to_state=RunState.ADJUDICATING,
            reason=f"checker persisted {len(checker_result.findings)} candidate findings",
        )
        try:
            adjudication_result = self._run_adjudicator_stage(
                store=store,
                run=run,
                task=task,
                findings=checker_result.findings,
                iteration=iteration,
                semantic_audit=semantic_audit,
                evidence_verification=checker_result.evidence_verification,
                evidence_verification_artifact_id=checker_result.evidence_verification_artifact_id,
            )
        except ValueError as error:
            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.ADJUDICATING,
                to_state=RunState.FAILED,
                reason=str(error),
            )
            store.tasks.update_status(task.task_id, status=TaskStatus.FAILED.value)
            raise

        if adjudication_result.held_for_consensus:
            store.tasks.update_status(task.task_id, status=TaskStatus.IN_PROGRESS.value)
            return "hold", validation_stage_id, []

        actionable_findings = finalized_actionable_finding_set(
            adjudication_result.findings,
            adjudication_result.dispositions,
            iteration=iteration,
        )
        if actionable_findings:
            self._transition(
                store=store,
                run_id=run.run_id,
                from_state=RunState.ADJUDICATING,
                to_state=RunState.CHECK_FAILED,
                reason=f"adjudicator left {len(actionable_findings)} actionable findings open",
            )
            return "actionable", validation_stage_id, actionable_findings

        self._transition(
            store=store,
            run_id=run.run_id,
            from_state=RunState.ADJUDICATING,
            to_state=RunState.CHECK_PASSED,
            reason="checker/adjudicator completed without actionable findings",
        )
        return "passed", validation_stage_id, []

    def _run_validation_stage(
        self,
        *,
        store: object,
        run: Run,
        task: Task,
        iteration: int,
    ) -> tuple[str, list[ValidationExecutionResult], bool]:
        validation_stage_id = _prefixed_id("stage")
        validation_stage = Stage(
            stage_id=validation_stage_id,
            run_id=run.run_id,
            task_id=task.task_id,
            stage_name="validation",
            status=StageStatus.RUNNING,
            iteration=iteration,
        )
        store.stages.create(validation_stage)

        try:
            validation_specs, forbidden_commands = self._validation_specs_for_run(run)
        except ValidationCommandError as error:
            store.stages.complete(
                validation_stage_id,
                status=StageStatus.FAILED.value,
                summary=str(error),
            )
            raise RiskyValidationCommandError(
                stage_id=validation_stage_id,
                command="",
                reason=str(error),
            ) from error

        validation_results: list[ValidationExecutionResult] = []
        any_failed = False
        for command_spec in validation_specs:
            if classify_command(command_spec.argv, forbidden_commands=forbidden_commands) == "risky":
                rendered_command = command_spec.command
                store.stages.complete(
                    validation_stage_id,
                    status=StageStatus.FAILED.value,
                    summary=f"blocked risky validation command: {rendered_command}",
                )
                raise RiskyValidationCommandError(
                    stage_id=validation_stage_id,
                    command=rendered_command,
                    reason=f"blocked risky validation command: {rendered_command}",
                )
            results = run_validation_commands(
                self.cwd,
                [command_spec.argv],
                stop_on_failure=self.config.validation.stop_on_failure,
                timeout_seconds=command_spec.timeout_seconds,
            )
            result = results[0]
            validation_results.extend(results)
            stdout_artifact = self.artifact_store.write_text(
                run_id=run.run_id,
                artifact_id=_prefixed_id("artifact"),
                artifact_type="validation_stdout",
                content=result.stdout,
                suffix=".log",
                metadata={"command": result.command},
            )
            stderr_artifact = self.artifact_store.write_text(
                run_id=run.run_id,
                artifact_id=_prefixed_id("artifact"),
                artifact_type="validation_stderr",
                content=result.stderr,
                suffix=".log",
                metadata={"command": result.command},
            )
            store.artifacts.create(stdout_artifact)
            store.artifacts.create(stderr_artifact)
            store.validations.create(
                ValidationRecord(
                    validation_id=_prefixed_id("validation"),
                    run_id=run.run_id,
                    task_id=task.task_id,
                    stage_id=validation_stage_id,
                    command=result.command,
                    status="passed" if result.returncode == 0 else "failed",
                    exit_code=result.returncode,
                    stdout_artifact_id=stdout_artifact.artifact_id,
                    stderr_artifact_id=stderr_artifact.artifact_id,
                    summary=f"command exited {result.returncode}",
                    completed_at=_utc_now(),
                )
            )
            if result.returncode != 0:
                any_failed = True
                store.findings.create(
                    Finding(
                        finding_id=_prefixed_id("finding"),
                        run_id=run.run_id,
                        stage_id=validation_stage_id,
                        stable_id=stable_id(
                            run_id=run.run_id,
                            category=FindingCategory.MISSING_VALIDATION.value,
                            stage_id=validation_stage_id,
                        ),
                        title=f"Validation failed: {result.command}",
                        severity=FindingSeverity.HIGH,
                        confidence=FindingConfidence.PROVEN,
                        category=FindingCategory.MISSING_VALIDATION,
                        rationale=result.stderr or result.stdout or "validation command failed",
                        exact_fix_action="Investigate the failing validation output and resume again.",
                        status=FindingStatus.OPEN,
                    )
                )
            if self.config.validation.stop_on_failure and any_failed:
                break

        store.stages.complete(
            validation_stage_id,
            status=StageStatus.FAILED.value if any_failed else StageStatus.COMPLETED.value,
            summary="One or more validation commands failed." if any_failed else "Validation commands passed.",
        )
        return validation_stage_id, validation_results, any_failed

    def _run_semantic_audit_stage(
        self,
        *,
        store: object,
        run: Run,
        task: Task,
        iteration: int,
        observed_delta: dict[str, object],
        validation_results: list[ValidationExecutionResult],
    ) -> SemanticAuditStageResult:
        stage_id = _prefixed_id("stage")
        surface_artifact_id = _prefixed_id("artifact")
        report_artifact_id = _prefixed_id("artifact")
        stage = Stage(
            stage_id=stage_id,
            run_id=run.run_id,
            task_id=task.task_id,
            stage_name="semantic_auditing",
            status=StageStatus.RUNNING,
            iteration=iteration,
            output_artifact_id=report_artifact_id,
        )
        store.stages.create(stage)
        open_findings = store.findings.list_by_run(run.run_id)
        policy_pack = self._policy_pack_for_run(run)
        surface_result = build_difference_surface(
            cwd=self.cwd,
            run=run,
            task=task,
            iteration=iteration,
            observed_delta=observed_delta,
            validation_results=validation_results,
            open_findings=open_findings,
            baseline_snapshot_path=self._resolve_baseline_snapshot_path(store, run.run_id),
            stop_on_failure=self.config.validation.stop_on_failure,
            scope_excludes=policy_pack.scope_rules.exclude if policy_pack is not None else (),
        )
        advisory_findings = [
            SemanticAuditFindingPayload(
                title=f"Implementer reported missing path: {path}",
                severity=FindingSeverity.MEDIUM,
                category=FindingCategory.UNSUPPORTED_CLAIM,
                rationale="The implementer reported a touched file that is not present in the workspace.",
                exact_fix_action="Reconcile the claimed workspace changes with actual files before trusting the report.",
                source_ref=path,
            )
            for path in surface_result.missing_reported_files
        ]
        blocking_findings = [
            SemanticAuditFindingPayload(
                title=f"Unreported workspace delta: {path}",
                severity=FindingSeverity.HIGH,
                category=FindingCategory.SCOPE_VIOLATION,
                rationale="The engine observed a changed path that was not reported by the implementer stage.",
                exact_fix_action="Inspect the workspace delta and confirm whether the change is in scope.",
                source_ref=path,
            )
            for path in surface_result.unreported_changed_files
        ]
        advisory_findings.extend(
            SemanticAuditFindingPayload(
                title=f"Reported but unchanged path: {path}",
                severity=FindingSeverity.LOW,
                category=FindingCategory.PROCESS_ARTIFACT,
                rationale="The implementer reported a path as changed, but the authoritative workspace delta did not change it.",
                exact_fix_action="Verify the implementation report and remove unsupported claims.",
                source_ref=path,
            )
            for path in surface_result.reported_but_unchanged_files
        )
        surface_artifact = self.artifact_store.write_json(
            run_id=run.run_id,
            artifact_id=surface_artifact_id,
            artifact_type="difference_surface",
            payload=surface_result.surface.model_dump(mode="json"),
            metadata={
                "stage_name": "semantic_auditing",
                "iteration": iteration,
                "capability_level": surface_result.capability_level,
            },
        )
        report = SemanticAuditResponsePayload(
            audit_id=stage_id,
            capability_level=surface_result.capability_level,
            hard_fail=bool(blocking_findings),
            blocking_findings=blocking_findings,
            advisory_findings=advisory_findings,
            unsupported_checks=surface_result.unsupported_checks,
            artifact_refs=[surface_artifact.artifact_id],
        )
        report_artifact = self.artifact_store.write_json(
            run_id=run.run_id,
            artifact_id=report_artifact_id,
            artifact_type="semantic_audit_report",
            payload=report.model_dump(mode="json"),
            metadata={
                "stage_name": "semantic_auditing",
                "difference_surface_artifact_id": surface_artifact.artifact_id,
                "iteration": iteration,
            },
        )
        store.artifacts.create(surface_artifact)
        store.artifacts.create(report_artifact)
        store.semantic_audits.create(
            SemanticAuditRecord(
                semantic_audit_id=stage_id,
                run_id=run.run_id,
                task_id=task.task_id,
                stage_id=stage_id,
                iteration=iteration,
                capability_level=report.capability_level,
                hard_fail=report.hard_fail,
                blocking_findings=[item.model_dump(mode="json") for item in report.blocking_findings],
                advisory_findings=[item.model_dump(mode="json") for item in report.advisory_findings],
                unsupported_checks=report.unsupported_checks,
                report_artifact_id=report_artifact.artifact_id,
                difference_surface_artifact_id=surface_artifact.artifact_id,
                summary=(
                    f"Semantic audit hard-failed with {len(report.blocking_findings)} blocking findings"
                    if report.hard_fail
                    else f"Semantic audit produced {len(report.advisory_findings)} advisory findings"
                ),
            )
        )
        store.stages.complete(
            stage_id,
            status=StageStatus.COMPLETED.value,
            output_artifact_id=report_artifact.artifact_id,
            summary=(
                f"Semantic audit hard-failed with {len(blocking_findings)} blocking findings."
                if blocking_findings
                else f"Semantic audit recorded filesystem-level context with {len(advisory_findings)} advisory findings."
            ),
        )
        return SemanticAuditStageResult(
            stage_id=stage_id,
            surface=surface_result.surface,
            surface_artifact_id=surface_artifact.artifact_id,
            report=report,
            report_artifact_id=report_artifact.artifact_id,
        )

    def _run_checker_stage(
        self,
        *,
        store: object,
        run: Run,
        task: Task,
        validation_results: list[ValidationExecutionResult],
        iteration: int,
        semantic_audit: SemanticAuditStageResult,
    ) -> CheckerStageResult:
        checker = self._required_model("checker")
        stage_id = _prefixed_id("stage")
        prompt_artifact_id = _prefixed_id("artifact")
        response_artifact_id = _prefixed_id("artifact")
        output_artifact_id = _prefixed_id("artifact")
        prompt = self._checker_prompt(
            objective=task.raw_input,
            difference_surface=semantic_audit.surface,
            semantic_audit=semantic_audit.report,
        )
        prompt_artifact = self.artifact_store.write_text(
            run_id=run.run_id,
            artifact_id=prompt_artifact_id,
            artifact_type="checker_prompt",
            content=prompt,
            suffix=".md",
            metadata={
                "stage_name": "checking",
                "role": ModelRole.CHECKER.value,
                "difference_surface_artifact_id": semantic_audit.surface_artifact_id,
                "semantic_audit_report_artifact_id": semantic_audit.report_artifact_id,
            },
        )
        stage = Stage(
            stage_id=stage_id,
            run_id=run.run_id,
            task_id=task.task_id,
            stage_name="checking",
            status=StageStatus.RUNNING,
            model_role=ModelRole.CHECKER,
            model_name=checker.model,
            provider=checker.provider,
            runtime_agent=checker.agent,
            runtime_variant=checker.variant,
            prompt_artifact_id=prompt_artifact_id,
            response_artifact_id=response_artifact_id,
            output_artifact_id=output_artifact_id,
            iteration=iteration,
        )
        store.artifacts.create(prompt_artifact)
        store.stages.create(stage)

        try:
            result = invoke_text_prompt(checker, prompt=prompt, cwd=self.cwd)
            self._record_model_call(
                store=store,
                run_id=run.run_id,
                stage=stage,
                profile=checker,
                role=ModelRole.CHECKER,
                result=result,
                status="completed",
            )
            self._record_invocation_cost(store=store, run_id=run.run_id, result=result)
            response_artifact = self.artifact_store.write_text(
                run_id=run.run_id,
                artifact_id=response_artifact_id,
                artifact_type="checker_response",
                content=result.stdout + "\n",
                suffix=".json",
                metadata={
                    "stage_name": "checking",
                    "provider": checker.provider,
                    "model": checker.model,
                    "difference_surface_artifact_id": semantic_audit.surface_artifact_id,
                    "semantic_audit_report_artifact_id": semantic_audit.report_artifact_id,
                },
            )
            payload = CheckerResponsePayload.model_validate(extract_json_output(result.stdout))
            output_artifact = self.artifact_store.write_json(
                run_id=run.run_id,
                artifact_id=output_artifact_id,
                artifact_type="checker_findings",
                payload=payload.model_dump(mode="json"),
                metadata={"stage_name": "checking"},
            )
            store.artifacts.create(response_artifact)
            store.artifacts.create(output_artifact)
            store.stages.complete(
                stage_id,
                status=StageStatus.COMPLETED.value,
                output_artifact_id=output_artifact.artifact_id,
                summary=payload.summary or f"Checker emitted {len(payload.findings)} findings.",
            )
            findings = self._persist_checker_findings(
                store=store,
                run=run,
                stage_id=stage_id,
                response_artifact_id=response_artifact.artifact_id,
                payload=payload,
            )
            evidence_verification_artifact_id = _prefixed_id("artifact")
            evidence_verification = self._verify_checker_findings(
                run_id=run.run_id,
                checker_stage_id=stage_id,
                evidence_verification_artifact_id=evidence_verification_artifact_id,
                payload=payload,
            )
            evidence_verification_artifact = self.artifact_store.write_json(
                run_id=run.run_id,
                artifact_id=evidence_verification_artifact_id,
                artifact_type="evidence_verification_report",
                payload=evidence_verification.model_dump(mode="json"),
                metadata={
                    "stage_name": "checking",
                    "checker_stage_id": stage_id,
                    "response_artifact_id": response_artifact.artifact_id,
                },
            )
            store.artifacts.create(evidence_verification_artifact)
            return CheckerStageResult(
                findings=findings,
                evidence_verification=evidence_verification,
                evidence_verification_artifact_id=evidence_verification_artifact.artifact_id,
            )
        except Exception as error:
            if isinstance(error, ModelInvocationError) and error.result is not None:
                self._record_model_call(
                    store=store,
                    run_id=run.run_id,
                    stage=stage,
                    profile=checker,
                    role=ModelRole.CHECKER,
                    result=error.result,
                    status="failed",
                    error_reason=str(error),
                )
            store.stages.complete(stage_id, status=StageStatus.FAILED.value, summary=str(error))
            raise ValueError(f"checker stage failed: {error}") from error

    def _run_adjudicator_stage(
        self,
        *,
        store: object,
        run: Run,
        task: Task,
        findings: list[Finding],
        iteration: int,
        semantic_audit: SemanticAuditStageResult,
        evidence_verification: EvidenceVerificationReport,
        evidence_verification_artifact_id: str,
    ) -> AdjudicationStageResult:
        adjudicator = self._required_model("adjudicator")
        if self._should_use_consensus(findings=findings, semantic_audit=semantic_audit):
            return self._run_consensus_adjudication_stage(
                store=store,
                run=run,
                task=task,
                findings=findings,
                iteration=iteration,
                semantic_audit=semantic_audit,
                evidence_verification=evidence_verification,
                evidence_verification_artifact_id=evidence_verification_artifact_id,
            )
        stage_id = _prefixed_id("stage")
        prompt_artifact_id = _prefixed_id("artifact")
        response_artifact_id = _prefixed_id("artifact")
        output_artifact_id = _prefixed_id("artifact")
        prompt = self._adjudicator_prompt(
            objective=task.raw_input,
            findings=findings,
            difference_surface=semantic_audit.surface,
            semantic_audit=semantic_audit.report,
            evidence_verification=evidence_verification,
        )
        prompt_artifact = self.artifact_store.write_text(
            run_id=run.run_id,
            artifact_id=prompt_artifact_id,
            artifact_type="adjudicator_prompt",
            content=prompt,
            suffix=".md",
            metadata={
                "stage_name": "adjudicating",
                "role": ModelRole.ADJUDICATOR.value,
                "difference_surface_artifact_id": semantic_audit.surface_artifact_id,
                "semantic_audit_report_artifact_id": semantic_audit.report_artifact_id,
                "evidence_verification_artifact_id": evidence_verification_artifact_id,
            },
        )
        stage = Stage(
            stage_id=stage_id,
            run_id=run.run_id,
            task_id=task.task_id,
            stage_name="adjudicating",
            status=StageStatus.RUNNING,
            model_role=ModelRole.ADJUDICATOR,
            model_name=adjudicator.model,
            provider=adjudicator.provider,
            runtime_agent=adjudicator.agent,
            runtime_variant=adjudicator.variant,
            prompt_artifact_id=prompt_artifact_id,
            response_artifact_id=response_artifact_id,
            output_artifact_id=output_artifact_id,
            iteration=iteration,
        )
        store.artifacts.create(prompt_artifact)
        store.stages.create(stage)

        try:
            result = invoke_text_prompt(adjudicator, prompt=prompt, cwd=self.cwd)
            self._record_model_call(
                store=store,
                run_id=run.run_id,
                stage=stage,
                profile=adjudicator,
                role=ModelRole.ADJUDICATOR,
                result=result,
                status="completed",
            )
            self._record_invocation_cost(store=store, run_id=run.run_id, result=result)
            response_artifact = self.artifact_store.write_text(
                run_id=run.run_id,
                artifact_id=response_artifact_id,
                artifact_type="adjudicator_response",
                content=result.stdout + "\n",
                suffix=".json",
                metadata={
                    "stage_name": "adjudicating",
                    "provider": adjudicator.provider,
                    "model": adjudicator.model,
                    "difference_surface_artifact_id": semantic_audit.surface_artifact_id,
                    "semantic_audit_report_artifact_id": semantic_audit.report_artifact_id,
                    "evidence_verification_artifact_id": evidence_verification_artifact_id,
                },
            )
            payload = AdjudicatorResponsePayload.model_validate(extract_json_output(result.stdout))
            output_artifact = self.artifact_store.write_json(
                run_id=run.run_id,
                artifact_id=output_artifact_id,
                artifact_type="adjudicator_dispositions",
                payload=payload.model_dump(mode="json"),
                metadata={"stage_name": "adjudicating"},
            )
            store.artifacts.create(response_artifact)
            store.artifacts.create(output_artifact)
            store.stages.complete(
                stage_id,
                status=StageStatus.COMPLETED.value,
                output_artifact_id=output_artifact.artifact_id,
                summary=payload.summary or f"Adjudicator returned {len(payload.dispositions)} dispositions.",
            )
            persisted_findings, dispositions = self._persist_dispositions(
                store=store,
                findings=findings,
                stage_id=stage_id,
                iteration=iteration,
                payload=payload,
            )
            return AdjudicationStageResult(findings=persisted_findings, dispositions=dispositions)
        except Exception as error:
            if isinstance(error, ModelInvocationError) and error.result is not None:
                self._record_model_call(
                    store=store,
                    run_id=run.run_id,
                    stage=stage,
                    profile=adjudicator,
                    role=ModelRole.ADJUDICATOR,
                    result=error.result,
                    status="failed",
                    error_reason=str(error),
                )
            store.stages.complete(stage_id, status=StageStatus.FAILED.value, summary=str(error))
            raise ValueError(f"adjudicator stage failed: {error}") from error

    def _run_consensus_adjudication_stage(
        self,
        *,
        store: object,
        run: Run,
        task: Task,
        findings: list[Finding],
        iteration: int,
        semantic_audit: SemanticAuditStageResult,
        evidence_verification: EvidenceVerificationReport,
        evidence_verification_artifact_id: str,
    ) -> AdjudicationStageResult:
        adjudicator = self._required_model("adjudicator")
        stage_id = _prefixed_id("stage")
        panel_id = _prefixed_id("panel")
        stage = Stage(
            stage_id=stage_id,
            run_id=run.run_id,
            task_id=task.task_id,
            stage_name="adjudicating",
            status=StageStatus.RUNNING,
            model_role=ModelRole.ADJUDICATOR,
            model_name=adjudicator.model,
            provider=adjudicator.provider,
            runtime_agent=adjudicator.agent,
            runtime_variant=adjudicator.variant,
            iteration=iteration,
        )
        store.stages.create(stage)
        store.adjudication_panels.create(
            AdjudicationPanelRecord(
                panel_id=panel_id,
                run_id=run.run_id,
                task_id=task.task_id,
                stage_id=stage_id,
                iteration=iteration,
                mode="consensus",
                primary_reason="semantic_hard_fail" if semantic_audit.report.hard_fail else "high_severity_findings",
                status="running",
            )
        )
        vote_payloads: list[tuple[AdjudicatorResponsePayload, str]] = []
        for member_label in self.config.adjudication.consensus_member_labels:
            prompt_artifact_id = _prefixed_id("artifact")
            response_artifact_id = _prefixed_id("artifact")
            output_artifact_id = _prefixed_id("artifact")
            prompt = (
                self._adjudicator_prompt(
                    objective=task.raw_input,
                    findings=findings,
                    difference_surface=semantic_audit.surface,
                    semantic_audit=semantic_audit.report,
                    evidence_verification=evidence_verification,
                )
                + f"\n\nConsensus member label: {member_label}. Return an independent decision."
            )
            prompt_artifact = self.artifact_store.write_text(
                run_id=run.run_id,
                artifact_id=prompt_artifact_id,
                artifact_type="adjudicator_prompt",
                content=prompt,
                suffix=".md",
                metadata={
                    "stage_name": "adjudicating",
                    "role": ModelRole.ADJUDICATOR.value,
                    "panel_id": panel_id,
                    "member_label": member_label,
                    "difference_surface_artifact_id": semantic_audit.surface_artifact_id,
                    "semantic_audit_report_artifact_id": semantic_audit.report_artifact_id,
                    "evidence_verification_artifact_id": evidence_verification_artifact_id,
                },
            )
            store.artifacts.create(prompt_artifact)
            try:
                result = invoke_text_prompt(adjudicator, prompt=prompt, cwd=self.cwd)
                self._record_model_call(
                    store=store,
                    run_id=run.run_id,
                    stage=stage,
                    profile=adjudicator,
                    role=ModelRole.ADJUDICATOR,
                    result=result,
                    status="completed",
                )
                self._record_invocation_cost(store=store, run_id=run.run_id, result=result)
                response_artifact = self.artifact_store.write_text(
                    run_id=run.run_id,
                    artifact_id=response_artifact_id,
                    artifact_type="adjudicator_response",
                    content=result.stdout + "\n",
                    suffix=".json",
                    metadata={
                        "stage_name": "adjudicating",
                        "provider": adjudicator.provider,
                        "model": adjudicator.model,
                        "panel_id": panel_id,
                        "member_label": member_label,
                    },
                )
                payload = AdjudicatorResponsePayload.model_validate(extract_json_output(result.stdout))
                output_artifact = self.artifact_store.write_json(
                    run_id=run.run_id,
                    artifact_id=output_artifact_id,
                    artifact_type="adjudicator_dispositions",
                    payload=payload.model_dump(mode="json"),
                    metadata={"stage_name": "adjudicating", "panel_id": panel_id, "member_label": member_label},
                )
                store.artifacts.create(response_artifact)
                store.artifacts.create(output_artifact)
                store.adjudication_votes.create(
                    AdjudicationVoteRecord(
                        vote_id=_prefixed_id("vote"),
                        panel_id=panel_id,
                        run_id=run.run_id,
                        stage_id=stage_id,
                        member_label=member_label,
                        provider=adjudicator.provider,
                        model_name=adjudicator.model,
                        runtime_agent=adjudicator.agent,
                        runtime_variant=adjudicator.variant,
                        response_artifact_id=response_artifact.artifact_id,
                        output_artifact_id=output_artifact.artifact_id,
                        payload=payload.model_dump(mode="json"),
                        summary=payload.summary,
                    )
                )
                vote_payloads.append((payload, member_label))
            except Exception as error:
                if isinstance(error, ModelInvocationError) and error.result is not None:
                    self._record_model_call(
                        store=store,
                        run_id=run.run_id,
                        stage=stage,
                        profile=adjudicator,
                        role=ModelRole.ADJUDICATOR,
                        result=error.result,
                        status="failed",
                        error_reason=str(error),
                    )
                store.stages.complete(stage_id, status=StageStatus.FAILED.value, summary=str(error))
                store.adjudication_panels.complete(panel_id, status="failed", summary=str(error))
                raise ValueError(f"consensus adjudication failed: {error}") from error

        signatures = {self._normalize_disposition_signature(payload) for payload, _ in vote_payloads}
        if len(signatures) == 1 and vote_payloads:
            persisted_findings, dispositions = self._persist_dispositions(
                store=store,
                findings=findings,
                stage_id=stage_id,
                iteration=iteration,
                payload=vote_payloads[0][0],
            )
            store.stages.complete(
                stage_id,
                status=StageStatus.COMPLETED.value,
                summary=f"Consensus adjudication reached unanimity across {len(vote_payloads)} votes.",
            )
            store.adjudication_panels.complete(
                panel_id,
                status="unanimous",
                summary=f"Consensus adjudication reached unanimity across {len(vote_payloads)} votes.",
            )
            return AdjudicationStageResult(findings=persisted_findings, dispositions=dispositions)

        disagreement_artifact = self.artifact_store.write_json(
            run_id=run.run_id,
            artifact_id=_prefixed_id("artifact"),
            artifact_type="adjudication_disagreement_report",
            payload={
                "panel_id": panel_id,
                "member_positions": [
                    {"member_label": member_label, "payload": payload.model_dump(mode="json")}
                    for payload, member_label in vote_payloads
                ],
            },
            metadata={"stage_name": "adjudicating", "panel_id": panel_id},
        )
        store.artifacts.create(disagreement_artifact)
        store.stages.complete(
            stage_id,
            status=StageStatus.COMPLETED.value,
            summary=f"Consensus adjudication split across {len(vote_payloads)} votes.",
        )
        store.adjudication_panels.complete(
            panel_id,
            status="split",
            summary=f"Consensus adjudication split across {len(vote_payloads)} votes.",
            disagreement_artifact_id=disagreement_artifact.artifact_id,
        )
        self._transition(
            store=store,
            run_id=run.run_id,
            from_state=RunState.ADJUDICATING,
            to_state=RunState.AWAITING_OPERATOR,
            reason=f"{HoldType.CRITICAL_DISAGREEMENT.value}: adjudication panel split; operator review required",
            operator_status=OperatorStatus.HOLD,
            hold_type=HoldType.CRITICAL_DISAGREEMENT,
            hold_context={
                "panel_id": panel_id,
                "vote_count": len(vote_payloads),
                "disagreement_artifact_id": disagreement_artifact.artifact_id,
            },
        )
        return AdjudicationStageResult(findings=[], dispositions=[], held_for_consensus=True)

    def _persist_checker_findings(
        self,
        *,
        store: object,
        run: Run,
        stage_id: str,
        response_artifact_id: str,
        payload: CheckerResponsePayload,
    ) -> list[Finding]:
        findings: list[Finding] = []
        for item in payload.findings:
            finding = Finding(
                finding_id=_prefixed_id("finding"),
                run_id=run.run_id,
                stage_id=stage_id,
                stable_id=stable_id(
                    run_id=run.run_id,
                    category=item.category.value,
                    file_path=item.file_path,
                    primary_symbol=item.primary_symbol,
                    line_range=item.line_range,
                    stage_id=stage_id,
                ),
                title=item.title,
                severity=item.severity,
                confidence=item.confidence,
                category=item.category,
                rationale=item.rationale,
                exact_fix_action=item.exact_fix_action,
            )
            store.findings.create(finding)
            store.findings.add_evidence(
                FindingEvidence(
                    evidence_id=_prefixed_id("evidence"),
                    finding_id=finding.finding_id,
                    evidence_type=EvidenceType.MODEL_OUTPUT,
                    artifact_id=response_artifact_id,
                    snippet=item.rationale[:2000],
                    source_ref=item.file_path or f"stage:{stage_id}",
                )
            )
            findings.append(finding)
        return findings

    def _persist_dispositions(
        self,
        *,
        store: object,
        findings: list[Finding],
        stage_id: str,
        iteration: int,
        payload: AdjudicatorResponsePayload,
    ) -> tuple[list[Finding], list[Disposition]]:
        findings_by_stable_id = {finding.stable_id: finding for finding in findings}
        unmatched_payloads = {item.stable_id for item in payload.dispositions}
        dispositions: list[Disposition] = []
        updated_findings: list[Finding] = []
        for finding in findings:
            matching_payload = next(
                (item for item in payload.dispositions if item.stable_id == finding.stable_id),
                None,
            )
            if matching_payload is None:
                raise ValueError(f"adjudicator omitted disposition for stable_id: {finding.stable_id}")
            matched_finding = findings_by_stable_id.get(matching_payload.stable_id)
            if matched_finding is None:
                raise ValueError(f"adjudicator returned unknown stable_id: {matching_payload.stable_id}")
            unmatched_payloads.discard(matching_payload.stable_id)
            disposition = Disposition(
                disposition_id=_prefixed_id("disposition"),
                finding_id=matched_finding.finding_id,
                stage_id=stage_id,
                iteration=iteration,
                decided_by=DecidedBy.ADJUDICATOR,
                decision=matching_payload.decision,
                justification=matching_payload.justification,
            )
            store.dispositions.create(disposition)
            store.findings.update_status(
                matched_finding.finding_id,
                status=matching_payload.decision.value,
                adjudicated=True,
            )
            dispositions.append(disposition)
            updated_findings.append(
                matched_finding.model_copy(
                    update={"status": matching_payload.decision, "adjudicated": True}
                )
            )
        if unmatched_payloads:
            raise ValueError(f"adjudicator returned unknown stable_ids: {sorted(unmatched_payloads)}")
        return updated_findings, dispositions

    def _persist_semantic_audit_findings(
        self,
        *,
        store: object,
        run: Run,
        semantic_audit: SemanticAuditStageResult,
    ) -> list[Finding]:
        existing_stable_ids = {finding.stable_id for finding in store.findings.list_by_run(run.run_id)}
        findings: list[Finding] = []
        for item in semantic_audit.report.blocking_findings:
            finding = Finding(
                finding_id=_prefixed_id("finding"),
                run_id=run.run_id,
                stage_id=semantic_audit.stage_id,
                stable_id=stable_id(
                    run_id=run.run_id,
                    category=item.category.value,
                    file_path=item.source_ref,
                    stage_id=semantic_audit.stage_id,
                ),
                title=item.title,
                severity=item.severity,
                confidence=FindingConfidence.PROVEN,
                category=item.category,
                rationale=item.rationale,
                exact_fix_action=item.exact_fix_action,
            )
            if finding.stable_id in existing_stable_ids:
                continue
            store.findings.create(finding)
            store.findings.add_evidence(
                FindingEvidence(
                    evidence_id=_prefixed_id("evidence"),
                    finding_id=finding.finding_id,
                    evidence_type=EvidenceType.MODEL_OUTPUT,
                    artifact_id=semantic_audit.report_artifact_id,
                    snippet=item.rationale[:2000],
                    source_ref=item.source_ref or f"stage:{semantic_audit.stage_id}",
                )
            )
            findings.append(finding)
        return findings

    def _persist_engine_dispositions(
        self,
        *,
        store: object,
        findings: list[Finding],
        stage_id: str,
        iteration: int,
        justification: str,
    ) -> list[Disposition]:
        dispositions: list[Disposition] = []
        for finding in findings:
            disposition = Disposition(
                disposition_id=_prefixed_id("disposition"),
                finding_id=finding.finding_id,
                stage_id=stage_id,
                iteration=iteration,
                decided_by=DecidedBy.ENGINE,
                decision=FindingStatus.OPEN,
                justification=justification,
            )
            store.dispositions.create(disposition)
            store.findings.update_status(
                finding.finding_id,
                status=FindingStatus.OPEN.value,
                adjudicated=True,
            )
            dispositions.append(disposition)
        return dispositions

    def _should_use_consensus(
        self,
        *,
        findings: list[Finding],
        semantic_audit: SemanticAuditStageResult,
    ) -> bool:
        return self.config.adjudication.consensus_enabled and (
            semantic_audit.report.hard_fail
            or any(finding.severity in {FindingSeverity.HIGH, FindingSeverity.CRITICAL} for finding in findings)
        )

    def _normalize_disposition_signature(self, payload: AdjudicatorResponsePayload) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((item.stable_id, item.decision.value) for item in payload.dispositions))

    def _capture_pre_act_checkpoint(
        self,
        *,
        store: object,
        run_id: str,
        task_id: str,
        stage_id: str,
        iteration: int,
    ) -> WorkspaceCheckpoint:
        checkpoint_artifact = capture_workspace_snapshot(
            self.cwd,
            run_id=run_id,
            artifact_id=_prefixed_id("artifact"),
            artifact_store=self.artifact_store,
        )
        checkpoint = WorkspaceCheckpoint(
            checkpoint_id=_prefixed_id("checkpoint"),
            run_id=run_id,
            task_id=task_id,
            stage_id=stage_id,
            iteration=iteration,
            checkpoint_kind=CheckpointKind.PRE_ACT,
            capture_mode=CaptureMode.SAFE_PATH_SNAPSHOT,
            scope_paths=["."],
            artifact_id=checkpoint_artifact.artifact_id,
        )
        store.artifacts.create(checkpoint_artifact)
        store.checkpoints.create(checkpoint)
        return checkpoint

    def _resolve_run(self, store: object, run_id: str | None) -> Run:
        if run_id is not None:
            run = store.runs.get(run_id)
            if run is None:
                raise ValueError(f"run not found: {run_id}")
            return run
        recent_runs = store.runs.list_recent(limit=1)
        if not recent_runs:
            raise ValueError("no runs found")
        return recent_runs[0]

    def _restore_checkpoint_in_store(self, *, store: object, run_id: str, checkpoint_id: str | None) -> str:
        checkpoint = self._resolve_checkpoint(store, run_id=run_id, checkpoint_id=checkpoint_id)
        artifact = store.artifacts.get(checkpoint.artifact_id)
        if artifact is None:
            raise ValueError(f"checkpoint artifact not found: {checkpoint.artifact_id}")
        restore_workspace_snapshot(
            self.cwd,
            artifact_store=self.artifact_store,
            snapshot_artifact=artifact,
        )
        store.checkpoints.mark_restored(checkpoint.checkpoint_id)
        store.operator_actions.create(
            OperatorActionRecord(
                action_id=_prefixed_id("action"),
                run_id=run_id,
                action_type=OperatorActionType.RESTORE_CHECKPOINT,
                note="Workspace restored from stored checkpoint artifact.",
                checkpoint_id=checkpoint.checkpoint_id,
            )
        )
        return checkpoint.checkpoint_id

    def _resolve_checkpoint(self, store: object, *, run_id: str, checkpoint_id: str | None) -> WorkspaceCheckpoint:
        if checkpoint_id is not None:
            checkpoint = store.checkpoints.get(checkpoint_id)
            if checkpoint is None or checkpoint.run_id != run_id:
                raise ValueError(f"checkpoint not found for run: {checkpoint_id}")
            return checkpoint
        checkpoints = store.checkpoints.list_by_run(run_id)
        if not checkpoints:
            raise ValueError(f"no checkpoints found for run: {run_id}")
        return checkpoints[-1]

    def _required_model(self, role: str) -> ModelProfileConfig:
        profile = self.config.models.get(role)
        if profile is None:
            raise ValueError(f"model profile not configured: {role}")
        return profile

    def _planning_prompt(self, objective: str) -> str:
        return (
            "You are planning a grind task. Produce a concise actionable implementation plan for the current repository.\n\n"
            f"Repository: {self.cwd}\n"
            f"Objective: {objective.strip()}\n\n"
            "Constraints:\n"
            "- Use the live workspace as the source of truth. Inspect the current code before claiming anything is complete.\n"
            "- If the objective references prior run notes, results files, or earlier grind artifacts, treat them as advisory hints only; verify against the current repository and spec instead of trusting those notes.\n"
            "- Keep the plan scoped to the requested objective and the next implementation slice that should be executed now.\n"
            "- This is a planning step, not a verdict step. Do not declare the objective complete, verified, deferred, or out-of-scope in the plan.\n"
            "- If completion is uncertain, the first plan steps must inspect the specific current files, tests, and spec clauses needed to prove or falsify the suspected gap.\n"
            "- Do not trust summary notes like 'phase complete' without fresh evidence from the live repository gathered in this run.\n"
            "- Include concrete implementation steps and the focused validation to run after the change.\n"
            "- Do not produce an operator-review-only, check-only, or closeout-only plan unless the objective explicitly asks for review or audit work.\n"
            "- Return exactly one JSON object with this shape and nothing else: {\"plan\":\"step-by-step implementation plan including focused validation\"}.\n"
            "- The value of \"plan\" must contain only the final plan, not chain-of-thought, exploration notes, or tool transcript.\n"
            "- Prefer concrete steps over narrative.\n"
        )

    def _do_prompt(self, *, objective: str) -> str:
        return (
            "You are the do stage implementer for a grind run. Apply the approved plan in the workspace and return JSON only.\n\n"
            f"Repository: {self.cwd}\n"
            f"Objective: {objective.strip()}\n\n"
            "Return exactly this JSON shape:\n"
            '{"touched_files":["src/foo.py"],"touched_symbols":["Foo.bar"],"validation_hints":[{"command":"uv run pytest tests -q","reason":"changed behavior"}],"claims_made":[{"claim":"...","evidence":"..."}],"open_uncertainties":[],"artifact_refs":[]}\n\n'
            "Only include JSON in your response."
        )

    def _checker_prompt(
        self,
        *,
        objective: str,
        difference_surface: DifferenceSurface,
        semantic_audit: SemanticAuditResponsePayload,
    ) -> str:
        categories = ", ".join(category.value for category in FindingCategory)
        severities = ", ".join(severity.value for severity in FindingSeverity)
        confidences = ", ".join(confidence.value for confidence in FindingConfidence)
        return (
            "You are the checker stage for a grind run. Review the engine-authored difference surface and return JSON only.\n\n"
            f"Objective: {objective.strip()}\n\n"
            "Difference surface:\n"
            f"{json.dumps(difference_surface.model_dump(mode='json'), indent=2)}\n\n"
            "Semantic audit:\n"
            f"{json.dumps(semantic_audit.model_dump(mode='json'), indent=2)}\n\n"
            "Return exactly this JSON shape:\n"
            '{"summary":"...","findings":[{"title":"...","severity":"...","confidence":"...","category":"...","rationale":"...","exact_fix_action":"...","file_path":null,"primary_symbol":null,"line_range":null}]}\n\n'
            f"Allowed severity values: {severities}.\n"
            f"Allowed confidence values: {confidences}.\n"
            f"Allowed category values: {categories}.\n"
            "Use an empty findings array when there are no candidate findings."
        )

    def _adjudicator_prompt(
        self,
        *,
        objective: str,
        findings: list[Finding],
        difference_surface: DifferenceSurface,
        semantic_audit: SemanticAuditResponsePayload,
        evidence_verification: EvidenceVerificationReport,
    ) -> str:
        findings_summary = "\n".join(
            f"- stable_id: {finding.stable_id}\n  severity: {finding.severity.value}\n  category: {finding.category.value}\n  title: {finding.title}\n  rationale: {finding.rationale}\n  exact_fix_action: {finding.exact_fix_action}"
            for finding in findings
        )
        decisions = ", ".join(status.value for status in FindingStatus)
        return (
            "You are the adjudicator stage for a grind run. Review the checker findings and return JSON only.\n\n"
            f"Objective: {objective.strip()}\n\n"
            "Difference surface:\n"
            f"{json.dumps(difference_surface.model_dump(mode='json'), indent=2)}\n\n"
            "Semantic audit:\n"
            f"{json.dumps(semantic_audit.model_dump(mode='json'), indent=2)}\n\n"
            "Evidence verification:\n"
            f"{json.dumps(evidence_verification.model_dump(mode='json'), indent=2)}\n\n"
            "Candidate findings:\n"
            f"{findings_summary or '- no candidate findings'}\n\n"
            "Return exactly this JSON shape:\n"
            '{"summary":"...","dispositions":[{"stable_id":"16hexchars","decision":"open","justification":"..."}]}\n\n'
            f"Allowed decision values: {decisions}.\n"
            "Reference only the provided stable_id values."
        )

    def _verify_checker_findings(
        self,
        *,
        run_id: str,
        checker_stage_id: str,
        evidence_verification_artifact_id: str,
        payload: CheckerResponsePayload,
    ) -> EvidenceVerificationReport:
        candidates = [
            CandidateFindingForVerification(
                stable_id=stable_id(
                    run_id=run_id,
                    category=item.category.value,
                    file_path=item.file_path,
                    primary_symbol=item.primary_symbol,
                    line_range=item.line_range,
                    stage_id=checker_stage_id,
                ),
                title=item.title,
                file_path=item.file_path,
                primary_symbol=item.primary_symbol,
                line_range=item.line_range,
            )
            for item in payload.findings
        ]
        return verify_candidate_findings(
            report_id=evidence_verification_artifact_id,
            candidates=candidates,
            cwd=self.cwd,
        )

    def _act_prompt(self, *, objective: str, findings: list[Finding]) -> str:
        findings_summary = "\n".join(
            f"- finding_id: {finding.finding_id}\n  stable_id: {finding.stable_id}\n  severity: {finding.severity.value}\n  title: {finding.title}\n  rationale: {finding.rationale}\n  exact_fix_action: {finding.exact_fix_action}"
            for finding in findings
        )
        return (
            "You are the act stage implementer for a grind run. Fix only the adjudicated actionable findings and return JSON only.\n\n"
            f"Repository: {self.cwd}\n"
            f"Objective: {objective.strip()}\n\n"
            "Actionable findings:\n"
            f"{findings_summary}\n\n"
            "Return exactly this JSON shape:\n"
            '{"triage":[{"finding_id":"...","action":"fixed","justification":"...","fix_artifact_id":null,"requested_validation_ids":[]}],"remaining_open_issues":[],"new_uncertainties":[]}\n\n'
            "Keep fixes minimal and only include JSON in your response."
        )

    def _observed_delta_from_do_output(self, payload: DoStageResponsePayload) -> dict[str, object]:
        return {
            "source_stage": "doing",
            "reported_touched_files": payload.touched_files,
            "reported_touched_symbols": payload.touched_symbols,
            "validation_hints": [hint.model_dump(mode="json") for hint in payload.validation_hints],
            "claims_made": [claim.model_dump(mode="json") for claim in payload.claims_made],
            "open_uncertainties": payload.open_uncertainties,
            "artifact_refs": payload.artifact_refs,
        }

    def _observed_delta_from_act_output(self, payload: ActStageResponsePayload) -> dict[str, object]:
        return {
            "source_stage": "acting",
            "triage": [item.model_dump(mode="json") for item in payload.triage],
            "remaining_open_issues": payload.remaining_open_issues,
            "new_uncertainties": payload.new_uncertainties,
        }

    def _latest_stage_by_name(self, *, store: object, run_id: str, stage_name: str) -> Stage | None:
        stages = [stage for stage in store.stages.list_by_run(run_id) if stage.stage_name == stage_name]
        if not stages:
            return None
        return stages[-1]

    def _record_invocation_cost(
        self,
        *,
        store: object,
        run_id: str,
        result: ModelInvocationResult | None,
    ) -> None:
        if result is None or result.estimated_cost_usd is None:
            return
        store.runs.add_total_cost(run_id, delta_cost_usd=result.estimated_cost_usd)

    def _budget_limit_reached(self, run: Run) -> bool:
        return run.budget_limit_usd is not None and run.total_cost_usd >= run.budget_limit_usd

    def _should_hold_for_diminishing_returns(
        self,
        previous_actionable_ids: set[str],
        current_actionable_ids: set[str],
    ) -> bool:
        return bool(previous_actionable_ids) and current_actionable_ids == previous_actionable_ids

    def _actionable_stable_ids_for_iteration(
        self,
        *,
        store: object,
        run_id: str,
        iteration: int,
    ) -> set[str]:
        if iteration <= 0:
            return set()
        stages_by_id = {stage.stage_id: stage for stage in store.stages.list_by_run(run_id)}
        findings = [
            finding
            for finding in store.findings.list_by_run(run_id)
            if (stage := stages_by_id.get(finding.stage_id)) is not None and stage.iteration == iteration
        ]
        dispositions = [
            disposition
            for disposition in store.dispositions.list_by_run(run_id)
            if disposition.iteration == iteration
        ]
        actionable = finalized_actionable_finding_set(findings, dispositions, iteration=iteration)
        return {finding.stable_id for finding in actionable}

    def _latest_hold_transition(self, store: object, run_id: str) -> TransitionRecord | None:
        transitions = store.transitions.list_by_run(run_id)
        hold_transitions = [
            transition for transition in transitions if transition.to_state == RunState.AWAITING_OPERATOR
        ]
        return hold_transitions[-1] if hold_transitions else None

    def _current_hold_reason(self, store: object, run_id: str) -> str | None:
        run = store.runs.get(run_id)
        if run is None or run.state != RunState.AWAITING_OPERATOR:
            return None
        return run.current_hold_reason

    def _current_hold_snapshot(self, store: object, run_id: str) -> dict[str, object]:
        run = store.runs.get(run_id)
        if run is None or run.state != RunState.AWAITING_OPERATOR:
            return {"hold_type": None, "hold_reason": None, "hold_context": None}
        return {
            "hold_type": run.current_hold_type.value if run.current_hold_type else None,
            "hold_reason": run.current_hold_reason,
            "hold_context": run.current_hold_context,
        }

    def _record_model_call(
        self,
        *,
        store: object,
        run_id: str,
        stage: Stage,
        profile: ModelProfileConfig,
        role: ModelRole,
        result: ModelInvocationResult,
        status: str,
        error_reason: str | None = None,
    ) -> None:
        store.model_calls.create(
            ModelCallRecord(
                model_call_id=_prefixed_id("modelcall"),
                run_id=run_id,
                stage_id=stage.stage_id,
                model_role=role,
                provider=profile.provider,
                model_name=profile.model,
                runtime_agent=profile.agent,
                runtime_variant=profile.variant,
                command=result.command,
                status=status,
                completed_at=_utc_now(),
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                estimated_cost_usd=result.estimated_cost_usd,
                metadata=result.provider_metadata,
                error_reason=error_reason,
            )
        )

    def _artifact_summary(self, artifact: object) -> dict[str, object]:
        return {
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type,
            "path": artifact.path,
            "created_at": artifact.created_at.isoformat(),
            "metadata": artifact.metadata,
        }

    def _load_artifact_content(self, path: Path) -> object:
        if not path.exists():
            return None
        if path.suffixes[-2:] == [".tar", ".gz"]:
            return {"kind": "binary", "path": str(path)}
        content = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".json":
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return content

    def _resolve_baseline_snapshot_path(self, store: object, run_id: str) -> Path | None:
        checkpoints = store.checkpoints.list_by_run(run_id)
        baseline = next(
            (
                checkpoint
                for checkpoint in checkpoints
                if checkpoint.checkpoint_kind == CheckpointKind.TASK_BASELINE
            ),
            None,
        )
        if baseline is None:
            return None
        artifact = store.artifacts.get(baseline.artifact_id)
        if artifact is None:
            return None
        return self.artifact_store.resolve_path(artifact)

    def _transition(
        self,
        *,
        store: object,
        run_id: str,
        from_state: RunState,
        to_state: RunState,
        reason: str,
        operator_status: OperatorStatus | None = None,
        hold_type: HoldType | None = None,
        hold_context: dict[str, object] | None = None,
    ) -> None:
        transition = TransitionRecord(
            transition_id=_prefixed_id("transition"),
            run_id=run_id,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            actor="engine",
        )
        store.transitions.create(transition)
        store.runs.update_state(
            run_id,
            state=to_state.value,
            operator_status=operator_status.value if operator_status is not None else None,
        )
        if to_state == RunState.AWAITING_OPERATOR:
            store.runs.set_hold_context(
                run_id,
                current_hold_type=hold_type.value if hold_type is not None else None,
                current_hold_reason=reason,
                current_hold_context=hold_context,
            )
        else:
            store.runs.clear_hold_context(run_id)
