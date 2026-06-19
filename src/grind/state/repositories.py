from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from grind.models.artifact import ArtifactRecord
from grind.models.adjudication import AdjudicationPanelRecord, AdjudicationVoteRecord
from grind.models.checkpoint import WorkspaceCheckpoint
from grind.models.disposition import Disposition
from grind.models.finding import Finding, FindingEvidence
from grind.models.model_call import ModelCallRecord
from grind.models.operator_action import OperatorActionRecord
from grind.models.retrieval import RetrievalQueueRecord
from grind.models.run_lease import RunLease
from grind.models.run import Run
from grind.models.semantic_audit import SemanticAuditRecord
from grind.models.stage import Stage
from grind.models.task import Task
from grind.models.transition import TransitionRecord
from grind.models.validation import ValidationRecord
from grind.models.worker import Worker


def _json_encode(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _json_decode(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, run: Run) -> Run:
        self.connection.execute(
            """
            INSERT INTO runs (
              run_id, repo_path, policy_pack_path, policy_schema_ver,
              created_at, updated_at, state, requested_objective, normalized_scope,
                            operator_status, current_worker_id, current_hold_type, current_hold_reason,
                            current_hold_context, validation_commands_override, iteration_count,
                            max_iterations, budget_limit_usd, total_cost_usd
                                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run.run_id,
                run.repo_path,
                run.policy_pack_path,
                run.policy_schema_ver,
                run.created_at,
                run.updated_at,
                run.state.value,
                run.requested_objective,
                _json_encode(run.normalized_scope),
                run.operator_status.value,
                run.current_worker_id,
                run.current_hold_type.value if run.current_hold_type else None,
                run.current_hold_reason,
                _json_encode(run.current_hold_context),
                _json_encode(run.validation_commands_override),
                run.iteration_count,
                run.max_iterations,
                run.budget_limit_usd,
                run.total_cost_usd,
            ],
        )
        return run

    def get(self, run_id: str) -> Run | None:
        row = self.connection.execute(
            """
            SELECT run_id, repo_path, policy_pack_path, policy_schema_ver,
                   created_at, updated_at, state, requested_objective, normalized_scope,
                     operator_status, current_worker_id, current_hold_type, current_hold_reason,
                     current_hold_context, validation_commands_override, iteration_count,
                     max_iterations, budget_limit_usd, total_cost_usd
            FROM runs WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if row is None:
            return None
        return _run_from_row(row)

    def list_recent(self, limit: int = 10) -> list[Run]:
        rows = self.connection.execute(
            """
            SELECT run_id, repo_path, policy_pack_path, policy_schema_ver,
                   created_at, updated_at, state, requested_objective, normalized_scope,
                     operator_status, current_worker_id, current_hold_type, current_hold_reason,
                     current_hold_context, validation_commands_override, iteration_count,
                     max_iterations, budget_limit_usd, total_cost_usd
            FROM runs ORDER BY created_at DESC LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [_run_from_row(row) for row in rows]

    def update_state(
        self,
        run_id: str,
        *,
        state: str,
        operator_status: str | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        timestamp = updated_at or _utc_now()
        if operator_status is None:
            self.connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
                [state, timestamp, run_id],
            )
            return
        self.connection.execute(
            "UPDATE runs SET state = ?, operator_status = ?, updated_at = ? WHERE run_id = ?",
            [state, operator_status, timestamp, run_id],
        )

    def set_operator_status(
        self,
        run_id: str,
        *,
        operator_status: str,
        updated_at: datetime | None = None,
    ) -> None:
        self.connection.execute(
            "UPDATE runs SET operator_status = ?, updated_at = ? WHERE run_id = ?",
            [operator_status, updated_at or _utc_now(), run_id],
        )

    def set_iteration_count(
        self,
        run_id: str,
        *,
        iteration_count: int,
        updated_at: datetime | None = None,
    ) -> None:
        self.connection.execute(
            "UPDATE runs SET iteration_count = ?, updated_at = ? WHERE run_id = ?",
            [iteration_count, updated_at or _utc_now(), run_id],
        )

    def patch_limits(
        self,
        run_id: str,
        *,
        max_iterations: int | None = None,
        budget_limit_usd: Decimal | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[Any] = []
        if max_iterations is not None:
            assignments.append("max_iterations = ?")
            values.append(max_iterations)
        if budget_limit_usd is not None:
            assignments.append("budget_limit_usd = ?")
            values.append(budget_limit_usd)
        if not assignments:
            return
        assignments.append("updated_at = ?")
        values.append(updated_at or _utc_now())
        values.append(run_id)
        self.connection.execute(
            f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?",
            values,
        )

    def set_validation_commands_override(
        self,
        run_id: str,
        *,
        validation_commands_override: list[str] | None,
        updated_at: datetime | None = None,
    ) -> None:
        self.connection.execute(
            "UPDATE runs SET validation_commands_override = ?, updated_at = ? WHERE run_id = ?",
            [
                _json_encode(validation_commands_override),
                updated_at or _utc_now(),
                run_id,
            ],
        )

    def set_hold_context(
        self,
        run_id: str,
        *,
        current_hold_type: str | None,
        current_hold_reason: str | None,
        current_hold_context: dict[str, Any] | None,
        updated_at: datetime | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE runs
            SET current_hold_type = ?, current_hold_reason = ?, current_hold_context = ?, updated_at = ?
            WHERE run_id = ?
            """,
            [
                current_hold_type,
                current_hold_reason,
                _json_encode(current_hold_context),
                updated_at or _utc_now(),
                run_id,
            ],
        )

    def clear_hold_context(self, run_id: str, *, updated_at: datetime | None = None) -> None:
        self.set_hold_context(
            run_id,
            current_hold_type=None,
            current_hold_reason=None,
            current_hold_context=None,
            updated_at=updated_at,
        )

    def set_current_worker(
        self,
        run_id: str,
        *,
        current_worker_id: str | None,
        updated_at: datetime | None = None,
    ) -> None:
        self.connection.execute(
            "UPDATE runs SET current_worker_id = ?, updated_at = ? WHERE run_id = ?",
            [current_worker_id, updated_at or _utc_now(), run_id],
        )

    def add_total_cost(
        self,
        run_id: str,
        *,
        delta_cost_usd: Decimal,
        updated_at: datetime | None = None,
    ) -> None:
        self.connection.execute(
            "UPDATE runs SET total_cost_usd = total_cost_usd + ?, updated_at = ? WHERE run_id = ?",
            [delta_cost_usd, updated_at or _utc_now(), run_id],
        )


class TaskRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, task: Task) -> Task:
        self.connection.execute(
            """
            INSERT INTO tasks (
              task_id, run_id, sequence, source_kind, raw_input, normalized_scope,
              phase_label, acceptance_checks, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                task.task_id,
                task.run_id,
                task.sequence,
                task.source_kind.value,
                task.raw_input,
                _json_encode(task.normalized_scope),
                task.phase_label,
                _json_encode(task.acceptance_checks),
                task.status.value,
                task.created_at,
                task.updated_at,
            ],
        )
        return task

    def list_by_run(self, run_id: str) -> list[Task]:
        rows = self.connection.execute(
            """
            SELECT task_id, run_id, sequence, source_kind, raw_input, normalized_scope,
                   phase_label, acceptance_checks, status, created_at, updated_at
            FROM tasks WHERE run_id = ? ORDER BY sequence ASC
            """,
            [run_id],
        ).fetchall()
        return [
            Task(
                task_id=row[0],
                run_id=row[1],
                sequence=row[2],
                source_kind=row[3],
                raw_input=row[4],
                normalized_scope=_json_decode(row[5]),
                phase_label=row[6],
                acceptance_checks=_json_decode(row[7]) or [],
                status=row[8],
                created_at=row[9],
                updated_at=row[10],
            )
            for row in rows
        ]

    def update_status(self, task_id: str, *, status: str, updated_at: datetime | None = None) -> None:
        self.connection.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?",
            [status, updated_at or _utc_now(), task_id],
        )


class StageRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, stage: Stage) -> Stage:
        self.connection.execute(
            """
            INSERT INTO stages (
              stage_id, run_id, task_id, stage_name, started_at, ended_at, status,
              model_role, model_name, provider, runtime_agent, runtime_variant,
              prompt_artifact_id, response_artifact_id, output_artifact_id,
              summary, iteration, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                stage.stage_id,
                stage.run_id,
                stage.task_id,
                stage.stage_name,
                stage.started_at,
                stage.ended_at,
                stage.status.value,
                stage.model_role.value if stage.model_role else None,
                stage.model_name,
                stage.provider,
                stage.runtime_agent,
                stage.runtime_variant,
                stage.prompt_artifact_id,
                stage.response_artifact_id,
                stage.output_artifact_id,
                stage.summary,
                stage.iteration,
                stage.latency_ms,
            ],
        )
        return stage

    def list_by_run(self, run_id: str) -> list[Stage]:
        rows = self.connection.execute(
            """
            SELECT stage_id, run_id, task_id, stage_name, started_at, ended_at, status,
                   model_role, model_name, provider, runtime_agent, runtime_variant,
                   prompt_artifact_id, response_artifact_id, output_artifact_id,
                   summary, iteration, latency_ms
            FROM stages WHERE run_id = ? ORDER BY started_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            Stage(
                stage_id=row[0],
                run_id=row[1],
                task_id=row[2],
                stage_name=row[3],
                started_at=row[4],
                ended_at=row[5],
                status=row[6],
                model_role=row[7],
                model_name=row[8],
                provider=row[9],
                runtime_agent=row[10],
                runtime_variant=row[11],
                prompt_artifact_id=row[12],
                response_artifact_id=row[13],
                output_artifact_id=row[14],
                summary=row[15],
                iteration=row[16],
                latency_ms=row[17],
            )
            for row in rows
        ]

    def complete(
        self,
        stage_id: str,
        *,
        status: str,
        ended_at: datetime | None = None,
        output_artifact_id: str | None = None,
        summary: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE stages
            SET status = ?, ended_at = ?, output_artifact_id = COALESCE(?, output_artifact_id),
                summary = COALESCE(?, summary)
            WHERE stage_id = ?
            """,
            [status, ended_at or _utc_now(), output_artifact_id, summary, stage_id],
        )


class ArtifactRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, artifact: ArtifactRecord) -> ArtifactRecord:
        self.connection.execute(
            """
            INSERT INTO artifacts (
              artifact_id, run_id, artifact_type, path, storage_kind, checksum,
              size_bytes, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                artifact.artifact_id,
                artifact.run_id,
                artifact.artifact_type,
                artifact.path,
                artifact.storage_kind,
                artifact.checksum,
                artifact.size_bytes,
                artifact.created_at,
                _json_encode(artifact.metadata),
            ],
        )
        return artifact

    def list_by_run(self, run_id: str) -> list[ArtifactRecord]:
        rows = self.connection.execute(
            """
            SELECT artifact_id, run_id, artifact_type, path, storage_kind, checksum,
                   size_bytes, created_at, metadata
            FROM artifacts WHERE run_id = ? ORDER BY created_at ASC
            """,
            [run_id],
        ).fetchall()
        return [_artifact_from_row(row) for row in rows]

    def list_all(self) -> list[ArtifactRecord]:
        rows = self.connection.execute(
            """
            SELECT artifact_id, run_id, artifact_type, path, storage_kind, checksum,
                   size_bytes, created_at, metadata
            FROM artifacts ORDER BY created_at ASC
            """
        ).fetchall()
        return [_artifact_from_row(row) for row in rows]

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        row = self.connection.execute(
            """
            SELECT artifact_id, run_id, artifact_type, path, storage_kind, checksum,
                   size_bytes, created_at, metadata
            FROM artifacts WHERE artifact_id = ?
            """,
            [artifact_id],
        ).fetchone()
        if row is None:
            return None
        return _artifact_from_row(row)

    def update_path(self, artifact_id: str, *, path: str) -> None:
        self.connection.execute(
            "UPDATE artifacts SET path = ? WHERE artifact_id = ?",
            [path, artifact_id],
        )


class TransitionRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, transition: TransitionRecord) -> TransitionRecord:
        self.connection.execute(
            """
            INSERT INTO transitions (
              transition_id, run_id, from_state, to_state, reason, actor, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                transition.transition_id,
                transition.run_id,
                transition.from_state.value,
                transition.to_state.value,
                transition.reason,
                transition.actor,
                transition.created_at,
            ],
        )
        return transition

    def list_by_run(self, run_id: str) -> list[TransitionRecord]:
        rows = self.connection.execute(
            """
            SELECT transition_id, run_id, from_state, to_state, reason, actor, created_at
            FROM transitions WHERE run_id = ? ORDER BY created_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            TransitionRecord(
                transition_id=row[0],
                run_id=row[1],
                from_state=row[2],
                to_state=row[3],
                reason=row[4],
                actor=row[5],
                created_at=row[6],
            )
            for row in rows
        ]


class FindingRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, finding: Finding) -> Finding:
        self.connection.execute(
            """
            INSERT INTO findings (
              finding_id, run_id, stage_id, stable_id, title, severity, confidence,
              category, rationale, exact_fix_action, status, first_seen_at,
              last_updated_at, adjudicated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                finding.finding_id,
                finding.run_id,
                finding.stage_id,
                finding.stable_id,
                finding.title,
                finding.severity.value,
                finding.confidence.value,
                finding.category.value,
                finding.rationale,
                finding.exact_fix_action,
                finding.status.value,
                finding.first_seen_at,
                finding.last_updated_at,
                finding.adjudicated,
            ],
        )
        return finding

    def add_evidence(self, evidence: FindingEvidence) -> FindingEvidence:
        self.connection.execute(
            """
            INSERT INTO finding_evidence (
              evidence_id, finding_id, evidence_type, artifact_id, snippet, source_ref, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                evidence.evidence_id,
                evidence.finding_id,
                evidence.evidence_type.value,
                evidence.artifact_id,
                evidence.snippet,
                evidence.source_ref,
                evidence.created_at,
            ],
        )
        return evidence

    def list_by_run(self, run_id: str) -> list[Finding]:
        rows = self.connection.execute(
            """
            SELECT finding_id, run_id, stage_id, stable_id, title, severity, confidence,
                   category, rationale, exact_fix_action, status, first_seen_at,
                   last_updated_at, adjudicated
            FROM findings WHERE run_id = ? ORDER BY first_seen_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            Finding(
                finding_id=row[0],
                run_id=row[1],
                stage_id=row[2],
                stable_id=row[3],
                title=row[4],
                severity=row[5],
                confidence=row[6],
                category=row[7],
                rationale=row[8],
                exact_fix_action=row[9],
                status=row[10],
                first_seen_at=row[11],
                last_updated_at=row[12],
                adjudicated=row[13],
            )
            for row in rows
        ]

    def update_status(
        self,
        finding_id: str,
        *,
        status: str,
        adjudicated: bool,
        updated_at: datetime | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE findings
            SET status = ?, adjudicated = ?, last_updated_at = ?
            WHERE finding_id = ?
            """,
            [status, adjudicated, updated_at or _utc_now(), finding_id],
        )


class DispositionRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, disposition: Disposition) -> Disposition:
        self.connection.execute(
            """
            INSERT INTO dispositions (
              disposition_id, finding_id, stage_id, iteration, decided_by, decision,
              justification, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                disposition.disposition_id,
                disposition.finding_id,
                disposition.stage_id,
                disposition.iteration,
                disposition.decided_by.value,
                disposition.decision.value,
                disposition.justification,
                disposition.created_at,
            ],
        )
        return disposition

    def list_by_run(self, run_id: str) -> list[Disposition]:
        rows = self.connection.execute(
            """
            SELECT d.disposition_id, d.finding_id, d.stage_id, d.iteration, d.decided_by,
                   d.decision, d.justification, d.created_at
            FROM dispositions d
            JOIN findings f ON f.finding_id = d.finding_id
            WHERE f.run_id = ?
            ORDER BY d.created_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            Disposition(
                disposition_id=row[0],
                finding_id=row[1],
                stage_id=row[2],
                iteration=row[3],
                decided_by=row[4],
                decision=row[5],
                justification=row[6],
                created_at=row[7],
            )
            for row in rows
        ]


class CheckpointRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, checkpoint: WorkspaceCheckpoint) -> WorkspaceCheckpoint:
        self.connection.execute(
            """
            INSERT INTO workspace_checkpoints (
              checkpoint_id, run_id, task_id, stage_id, iteration, checkpoint_kind,
              capture_mode, scope_paths, artifact_id, status, created_by, created_at, restored_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                checkpoint.checkpoint_id,
                checkpoint.run_id,
                checkpoint.task_id,
                checkpoint.stage_id,
                checkpoint.iteration,
                checkpoint.checkpoint_kind.value,
                checkpoint.capture_mode.value,
                _json_encode(checkpoint.scope_paths),
                checkpoint.artifact_id,
                checkpoint.status.value,
                checkpoint.created_by,
                checkpoint.created_at,
                checkpoint.restored_at,
            ],
        )
        return checkpoint

    def list_by_run(self, run_id: str) -> list[WorkspaceCheckpoint]:
        rows = self.connection.execute(
            """
            SELECT checkpoint_id, run_id, task_id, stage_id, iteration, checkpoint_kind,
                   capture_mode, scope_paths, artifact_id, status, created_by,
                   created_at, restored_at
            FROM workspace_checkpoints WHERE run_id = ? ORDER BY created_at ASC
            """,
            [run_id],
        ).fetchall()
        return [_checkpoint_from_row(row) for row in rows]

    def get(self, checkpoint_id: str) -> WorkspaceCheckpoint | None:
        row = self.connection.execute(
            """
            SELECT checkpoint_id, run_id, task_id, stage_id, iteration, checkpoint_kind,
                   capture_mode, scope_paths, artifact_id, status, created_by,
                   created_at, restored_at
            FROM workspace_checkpoints WHERE checkpoint_id = ?
            """,
            [checkpoint_id],
        ).fetchone()
        if row is None:
            return None
        return _checkpoint_from_row(row)

    def mark_restored(self, checkpoint_id: str, *, restored_at: datetime | None = None) -> None:
        self.connection.execute(
            "UPDATE workspace_checkpoints SET status = 'restored', restored_at = ? WHERE checkpoint_id = ?",
            [restored_at or _utc_now(), checkpoint_id],
        )


class ValidationRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, validation: ValidationRecord) -> ValidationRecord:
        self.connection.execute(
            """
            INSERT INTO validations (
              validation_id, run_id, task_id, stage_id, command, status, required,
              exit_code, stdout_artifact_id, stderr_artifact_id, summary, created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                validation.validation_id,
                validation.run_id,
                validation.task_id,
                validation.stage_id,
                validation.command,
                validation.status,
                validation.required,
                validation.exit_code,
                validation.stdout_artifact_id,
                validation.stderr_artifact_id,
                validation.summary,
                validation.created_at,
                validation.completed_at,
            ],
        )
        return validation

    def list_by_run(self, run_id: str) -> list[ValidationRecord]:
        rows = self.connection.execute(
            """
            SELECT validation_id, run_id, task_id, stage_id, command, status, required,
                   exit_code, stdout_artifact_id, stderr_artifact_id, summary, created_at, completed_at
            FROM validations WHERE run_id = ? ORDER BY created_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            ValidationRecord(
                validation_id=row[0],
                run_id=row[1],
                task_id=row[2],
                stage_id=row[3],
                command=row[4],
                status=row[5],
                required=row[6],
                exit_code=row[7],
                stdout_artifact_id=row[8],
                stderr_artifact_id=row[9],
                summary=row[10],
                created_at=row[11],
                completed_at=row[12],
            )
            for row in rows
        ]


class OperatorActionRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, action: OperatorActionRecord) -> OperatorActionRecord:
        self.connection.execute(
            """
            INSERT INTO operator_actions (
                            action_id, run_id, action_type, note, checkpoint_id, payload, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                action.action_id,
                action.run_id,
                                action.action_type.value,
                action.note,
                action.checkpoint_id,
                                _json_encode(action.payload),
                action.created_at,
            ],
        )
        return action

    def list_by_run(self, run_id: str) -> list[OperatorActionRecord]:
        rows = self.connection.execute(
            """
            SELECT action_id, run_id, action_type, note, checkpoint_id, payload, created_at
            FROM operator_actions WHERE run_id = ? ORDER BY created_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            OperatorActionRecord(
                action_id=row[0],
                run_id=row[1],
                action_type=row[2],
                note=row[3],
                checkpoint_id=row[4],
                payload=_json_decode(row[5]),
                created_at=row[6],
            )
            for row in rows
        ]


class ModelCallRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, record: ModelCallRecord) -> ModelCallRecord:
        self.connection.execute(
            """
            INSERT INTO model_calls (
              model_call_id, run_id, stage_id, model_role, provider, model_name,
              runtime_agent, runtime_variant, command, status, started_at, completed_at,
              latency_ms, input_tokens, output_tokens, estimated_cost_usd, metadata, error_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.model_call_id,
                record.run_id,
                record.stage_id,
                record.model_role.value,
                record.provider,
                record.model_name,
                record.runtime_agent,
                record.runtime_variant,
                _json_encode(record.command),
                record.status,
                record.started_at,
                record.completed_at,
                record.latency_ms,
                record.input_tokens,
                record.output_tokens,
                record.estimated_cost_usd,
                _json_encode(record.metadata),
                record.error_reason,
            ],
        )
        return record

    def list_by_run(self, run_id: str) -> list[ModelCallRecord]:
        rows = self.connection.execute(
            """
            SELECT model_call_id, run_id, stage_id, model_role, provider, model_name,
                   runtime_agent, runtime_variant, command, status, started_at, completed_at,
                   latency_ms, input_tokens, output_tokens, estimated_cost_usd, metadata, error_reason
            FROM model_calls WHERE run_id = ? ORDER BY started_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            ModelCallRecord(
                model_call_id=row[0],
                run_id=row[1],
                stage_id=row[2],
                model_role=row[3],
                provider=row[4],
                model_name=row[5],
                runtime_agent=row[6],
                runtime_variant=row[7],
                command=_json_decode(row[8]) or [],
                status=row[9],
                started_at=row[10],
                completed_at=row[11],
                latency_ms=row[12],
                input_tokens=row[13],
                output_tokens=row[14],
                estimated_cost_usd=row[15],
                metadata=_json_decode(row[16]),
                error_reason=row[17],
            )
            for row in rows
        ]


class SemanticAuditRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, record: SemanticAuditRecord) -> SemanticAuditRecord:
        self.connection.execute(
            """
            INSERT INTO semantic_audits (
              semantic_audit_id, run_id, task_id, stage_id, iteration, capability_level,
              hard_fail, blocking_findings, advisory_findings, unsupported_checks,
              report_artifact_id, difference_surface_artifact_id, summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.semantic_audit_id,
                record.run_id,
                record.task_id,
                record.stage_id,
                record.iteration,
                record.capability_level,
                record.hard_fail,
                _json_encode(record.blocking_findings),
                _json_encode(record.advisory_findings),
                _json_encode(record.unsupported_checks),
                record.report_artifact_id,
                record.difference_surface_artifact_id,
                record.summary,
                record.created_at,
            ],
        )
        return record

    def list_by_run(self, run_id: str) -> list[SemanticAuditRecord]:
        rows = self.connection.execute(
            """
            SELECT semantic_audit_id, run_id, task_id, stage_id, iteration, capability_level,
                   hard_fail, blocking_findings, advisory_findings, unsupported_checks,
                   report_artifact_id, difference_surface_artifact_id, summary, created_at
            FROM semantic_audits WHERE run_id = ? ORDER BY created_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            SemanticAuditRecord(
                semantic_audit_id=row[0],
                run_id=row[1],
                task_id=row[2],
                stage_id=row[3],
                iteration=row[4],
                capability_level=row[5],
                hard_fail=row[6],
                blocking_findings=_json_decode(row[7]) or [],
                advisory_findings=_json_decode(row[8]) or [],
                unsupported_checks=_json_decode(row[9]) or [],
                report_artifact_id=row[10],
                difference_surface_artifact_id=row[11],
                summary=row[12],
                created_at=row[13],
            )
            for row in rows
        ]


class AdjudicationPanelRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, record: AdjudicationPanelRecord) -> AdjudicationPanelRecord:
        self.connection.execute(
            """
            INSERT INTO adjudication_panels (
              panel_id, run_id, task_id, stage_id, iteration, mode, primary_reason,
              status, disagreement_artifact_id, summary, created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.panel_id,
                record.run_id,
                record.task_id,
                record.stage_id,
                record.iteration,
                record.mode,
                record.primary_reason,
                record.status,
                record.disagreement_artifact_id,
                record.summary,
                record.created_at,
                record.completed_at,
            ],
        )
        return record

    def complete(
        self,
        panel_id: str,
        *,
        status: str,
        summary: str | None = None,
        disagreement_artifact_id: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE adjudication_panels
            SET status = ?, summary = COALESCE(?, summary),
                disagreement_artifact_id = COALESCE(?, disagreement_artifact_id),
                completed_at = ?
            WHERE panel_id = ?
            """,
            [status, summary, disagreement_artifact_id, completed_at or _utc_now(), panel_id],
        )

    def list_by_run(self, run_id: str) -> list[AdjudicationPanelRecord]:
        rows = self.connection.execute(
            """
            SELECT panel_id, run_id, task_id, stage_id, iteration, mode, primary_reason,
                   status, disagreement_artifact_id, summary, created_at, completed_at
            FROM adjudication_panels WHERE run_id = ? ORDER BY created_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            AdjudicationPanelRecord(
                panel_id=row[0],
                run_id=row[1],
                task_id=row[2],
                stage_id=row[3],
                iteration=row[4],
                mode=row[5],
                primary_reason=row[6],
                status=row[7],
                disagreement_artifact_id=row[8],
                summary=row[9],
                created_at=row[10],
                completed_at=row[11],
            )
            for row in rows
        ]


class AdjudicationVoteRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, record: AdjudicationVoteRecord) -> AdjudicationVoteRecord:
        self.connection.execute(
            """
            INSERT INTO adjudication_votes (
              vote_id, panel_id, run_id, stage_id, member_label, provider, model_name,
              runtime_agent, runtime_variant, response_artifact_id, output_artifact_id,
              payload, summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.vote_id,
                record.panel_id,
                record.run_id,
                record.stage_id,
                record.member_label,
                record.provider,
                record.model_name,
                record.runtime_agent,
                record.runtime_variant,
                record.response_artifact_id,
                record.output_artifact_id,
                _json_encode(record.payload),
                record.summary,
                record.created_at,
            ],
        )
        return record

    def list_by_run(self, run_id: str) -> list[AdjudicationVoteRecord]:
        rows = self.connection.execute(
            """
            SELECT vote_id, panel_id, run_id, stage_id, member_label, provider, model_name,
                   runtime_agent, runtime_variant, response_artifact_id, output_artifact_id,
                   payload, summary, created_at
            FROM adjudication_votes WHERE run_id = ? ORDER BY created_at ASC
            """,
            [run_id],
        ).fetchall()
        return [
            AdjudicationVoteRecord(
                vote_id=row[0],
                panel_id=row[1],
                run_id=row[2],
                stage_id=row[3],
                member_label=row[4],
                provider=row[5],
                model_name=row[6],
                runtime_agent=row[7],
                runtime_variant=row[8],
                response_artifact_id=row[9],
                output_artifact_id=row[10],
                payload=_json_decode(row[11]) or {},
                summary=row[12],
                created_at=row[13],
            )
            for row in rows
        ]


class RetrievalQueueRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, record: RetrievalQueueRecord) -> RetrievalQueueRecord:
        self.connection.execute(
            """
            INSERT INTO retrieval_index_queue (
              queue_id, run_id, artifact_id, collection, queue_status,
              attempts, last_error, queued_at, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.queue_id,
                record.run_id,
                record.artifact_id,
                record.collection,
                record.queue_status,
                record.attempts,
                record.last_error,
                record.queued_at,
                record.started_at,
                record.completed_at,
            ],
        )
        return record

    def get_existing(self, *, run_id: str, artifact_id: str, collection: str) -> RetrievalQueueRecord | None:
        row = self.connection.execute(
            """
            SELECT queue_id, run_id, artifact_id, collection, queue_status,
                   attempts, last_error, queued_at, started_at, completed_at
            FROM retrieval_index_queue
            WHERE run_id = ? AND artifact_id = ? AND collection = ?
            ORDER BY queued_at DESC
            LIMIT 1
            """,
            [run_id, artifact_id, collection],
        ).fetchone()
        if row is None:
            return None
        return _retrieval_queue_from_row(row)

    def list_by_run(self, run_id: str) -> list[RetrievalQueueRecord]:
        rows = self.connection.execute(
            """
            SELECT queue_id, run_id, artifact_id, collection, queue_status,
                   attempts, last_error, queued_at, started_at, completed_at
            FROM retrieval_index_queue
            WHERE run_id = ?
            ORDER BY queued_at ASC
            """,
            [run_id],
        ).fetchall()
        return [_retrieval_queue_from_row(row) for row in rows]

    def list_pending(self, *, run_id: str | None = None) -> list[RetrievalQueueRecord]:
        if run_id is None:
            rows = self.connection.execute(
                """
                SELECT queue_id, run_id, artifact_id, collection, queue_status,
                       attempts, last_error, queued_at, started_at, completed_at
                FROM retrieval_index_queue
                WHERE queue_status = 'pending'
                ORDER BY queued_at ASC
                """
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT queue_id, run_id, artifact_id, collection, queue_status,
                       attempts, last_error, queued_at, started_at, completed_at
                FROM retrieval_index_queue
                WHERE run_id = ? AND queue_status = 'pending'
                ORDER BY queued_at ASC
                """,
                [run_id],
            ).fetchall()
        return [_retrieval_queue_from_row(row) for row in rows]

    def mark_running(self, queue_id: str) -> None:
        self.connection.execute(
            """
            UPDATE retrieval_index_queue
            SET queue_status = 'running', attempts = attempts + 1, started_at = ?, last_error = NULL
            WHERE queue_id = ?
            """,
            [_utc_now(), queue_id],
        )

    def mark_completed(self, queue_id: str) -> None:
        self.connection.execute(
            """
            UPDATE retrieval_index_queue
            SET queue_status = 'completed', completed_at = ?
            WHERE queue_id = ?
            """,
            [_utc_now(), queue_id],
        )

    def mark_failed(self, queue_id: str, *, last_error: str) -> None:
        self.connection.execute(
            """
            UPDATE retrieval_index_queue
            SET queue_status = 'failed', last_error = ?, completed_at = ?
            WHERE queue_id = ?
            """,
            [last_error, _utc_now(), queue_id],
        )


class WorkerRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def register(self, worker: Worker) -> Worker:
        existing = self.get(worker.worker_id)
        if existing is not None:
            return existing
        self.connection.execute(
            """
            INSERT INTO workers (worker_id, hostname, pid, registered_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                worker.worker_id,
                worker.hostname,
                worker.pid,
                worker.registered_at,
                worker.last_seen_at,
            ],
        )
        return worker

    def get(self, worker_id: str) -> Worker | None:
        row = self.connection.execute(
            "SELECT worker_id, hostname, pid, registered_at, last_seen_at FROM workers WHERE worker_id = ?",
            [worker_id],
        ).fetchone()
        if row is None:
            return None
        return _worker_from_row(row)

    def heartbeat(self, worker_id: str, *, last_seen_at: datetime | None = None) -> None:
        self.connection.execute(
            "UPDATE workers SET last_seen_at = ? WHERE worker_id = ?",
            [last_seen_at or _utc_now(), worker_id],
        )

    def list_recent(self, limit: int = 25) -> list[Worker]:
        rows = self.connection.execute(
            "SELECT worker_id, hostname, pid, registered_at, last_seen_at FROM workers ORDER BY registered_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [_worker_from_row(row) for row in rows]


class RunLeaseRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def create(self, lease: RunLease) -> RunLease:
        self.connection.execute(
            """
            INSERT INTO run_leases (
              lease_id, run_id, worker_id, acquired_at, released_at, active_run_key, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                lease.lease_id,
                lease.run_id,
                lease.worker_id,
                lease.acquired_at,
                lease.released_at,
                lease.run_id if lease.status == "active" else None,
                lease.status,
            ],
        )
        return lease

    def get(self, lease_id: str) -> RunLease | None:
        row = self.connection.execute(
            "SELECT lease_id, run_id, worker_id, acquired_at, released_at, status FROM run_leases WHERE lease_id = ?",
            [lease_id],
        ).fetchone()
        if row is None:
            return None
        return _run_lease_from_row(row)

    def get_active_by_run(self, run_id: str) -> RunLease | None:
        row = self.connection.execute(
            """
            SELECT lease_id, run_id, worker_id, acquired_at, released_at, status
            FROM run_leases
            WHERE run_id = ? AND status = 'active'
            ORDER BY acquired_at DESC
            LIMIT 1
            """,
            [run_id],
        ).fetchone()
        if row is None:
            return None
        return _run_lease_from_row(row)

    def list_by_run(self, run_id: str) -> list[RunLease]:
        rows = self.connection.execute(
            """
            SELECT lease_id, run_id, worker_id, acquired_at, released_at, status
            FROM run_leases
            WHERE run_id = ?
            ORDER BY acquired_at
            """,
            [run_id],
        ).fetchall()
        return [_run_lease_from_row(row) for row in rows]

    def list_active(self) -> list[RunLease]:
        rows = self.connection.execute(
            """
            SELECT lease_id, run_id, worker_id, acquired_at, released_at, status
            FROM run_leases
            WHERE status = 'active'
            ORDER BY acquired_at
            """
        ).fetchall()
        return [_run_lease_from_row(row) for row in rows]

    def release(self, lease_id: str, *, released_at: datetime | None = None) -> None:
        self.connection.execute(
            "UPDATE run_leases SET status = 'released', released_at = ?, active_run_key = NULL WHERE lease_id = ?",
            [released_at or _utc_now(), lease_id],
        )

    def release_active_for_run(self, run_id: str, *, released_at: datetime | None = None) -> None:
        self.connection.execute(
            """
            UPDATE run_leases
            SET status = 'released', released_at = ?, active_run_key = NULL
            WHERE run_id = ? AND status = 'active'
            """,
            [released_at or _utc_now(), run_id],
        )

    def expire(self, lease_id: str, *, released_at: datetime | None = None) -> None:
        self.connection.execute(
            "UPDATE run_leases SET status = 'expired', released_at = ?, active_run_key = NULL WHERE lease_id = ?",
            [released_at or _utc_now(), lease_id],
        )


class DuckDBStateStore:
    def __init__(
        self,
        *,
        connection: duckdb.DuckDBPyConnection,
        database_path: Path | None = None,
    ):
        self.database_path = database_path
        self.connection = connection
        self.runs = RunRepository(self.connection)
        self.workers = WorkerRepository(self.connection)
        self.run_leases = RunLeaseRepository(self.connection)
        self.tasks = TaskRepository(self.connection)
        self.stages = StageRepository(self.connection)
        self.artifacts = ArtifactRepository(self.connection)
        self.transitions = TransitionRepository(self.connection)
        self.findings = FindingRepository(self.connection)
        self.dispositions = DispositionRepository(self.connection)
        self.checkpoints = CheckpointRepository(self.connection)
        self.validations = ValidationRepository(self.connection)
        self.operator_actions = OperatorActionRepository(self.connection)
        self.model_calls = ModelCallRepository(self.connection)
        self.semantic_audits = SemanticAuditRepository(self.connection)
        self.adjudication_panels = AdjudicationPanelRepository(self.connection)
        self.adjudication_votes = AdjudicationVoteRepository(self.connection)
        self.retrieval_queue = RetrievalQueueRepository(self.connection)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> DuckDBStateStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _run_from_row(row: tuple[Any, ...]) -> Run:
    return Run(
        run_id=row[0],
        repo_path=row[1],
        policy_pack_path=row[2],
        policy_schema_ver=row[3],
        created_at=row[4],
        updated_at=row[5],
        state=row[6],
        requested_objective=row[7],
        normalized_scope=_json_decode(row[8]),
        operator_status=row[9],
        current_worker_id=row[10],
        current_hold_type=row[11],
        current_hold_reason=row[12],
        current_hold_context=_json_decode(row[13]),
        validation_commands_override=_json_decode(row[14]),
        iteration_count=row[15],
        max_iterations=row[16],
        budget_limit_usd=row[17],
        total_cost_usd=row[18] if isinstance(row[18], Decimal) else Decimal(str(row[18])),
    )


def _artifact_from_row(row: tuple[Any, ...]) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=row[0],
        run_id=row[1],
        artifact_type=row[2],
        path=row[3],
        storage_kind=row[4],
        checksum=row[5],
        size_bytes=row[6],
        created_at=row[7],
        metadata=_json_decode(row[8]),
    )


def _worker_from_row(row: tuple[Any, ...]) -> Worker:
    return Worker(
        worker_id=row[0],
        hostname=row[1],
        pid=row[2],
        registered_at=row[3],
        last_seen_at=row[4],
    )


def _run_lease_from_row(row: tuple[Any, ...]) -> RunLease:
    return RunLease(
        lease_id=row[0],
        run_id=row[1],
        worker_id=row[2],
        acquired_at=row[3],
        released_at=row[4],
        status=row[5],
    )


def _checkpoint_from_row(row: tuple[Any, ...]) -> WorkspaceCheckpoint:
    return WorkspaceCheckpoint(
        checkpoint_id=row[0],
        run_id=row[1],
        task_id=row[2],
        stage_id=row[3],
        iteration=row[4],
        checkpoint_kind=row[5],
        capture_mode=row[6],
        scope_paths=_json_decode(row[7]) or [],
        artifact_id=row[8],
        status=row[9],
        created_by=row[10],
        created_at=row[11],
        restored_at=row[12],
    )


def _retrieval_queue_from_row(row: tuple[Any, ...]) -> RetrievalQueueRecord:
    return RetrievalQueueRecord(
        queue_id=row[0],
        run_id=row[1],
        artifact_id=row[2],
        collection=row[3],
        queue_status=row[4],
        attempts=row[5],
        last_error=row[6],
        queued_at=row[7],
        started_at=row[8],
        completed_at=row[9],
    )
