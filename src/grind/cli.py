from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
import secrets
import socket
import shlex
import sys
from pathlib import Path

import shutil
from grind.config import EngineConfig, default_engine_config_path, init_engine_workspace, load_engine_config
from grind.engine.leases import acquire_lease, release_lease
from grind.engine.orchestrator import MinimalOrchestrator
from grind.models import OperatorStatus, Run, RunState, Worker
from grind.models.enums import TaskSourceKind
from grind.providers import extract_text_output
from grind.retrieval import LanceDBRetrievalService
from grind.state.quack import QuackConnectionError, ensure_local_quack_server, is_local_quack_uri
from grind.state import bootstrap_state_store, current_schema_version, open_state_store
from grind.validation.safety import ValidationCommandError, classify_command, normalize_shell_free_command
from grind.verification.models import VerificationOverallStatus, VerificationRequest
from grind.verification.service import DefaultBackendVerifier, VerificationConfigError


def _shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _base_command(command: str, *, cwd: Path | None, config_path: Path | None) -> list[str]:
    parts = ["grind", command]
    if cwd is not None:
        parts.extend(["--cwd", str(cwd)])
    if config_path is not None:
        parts.extend(["--config", str(config_path)])
    return parts


def _inspect_command(
    *,
    run_id: str,
    cwd: Path | None,
    config_path: Path | None,
    artifact_id: str | None = None,
) -> str:
    parts = _base_command("inspect", cwd=cwd, config_path=config_path)
    if artifact_id is not None:
        parts.append(artifact_id)
    parts.extend(["--run-id", run_id])
    return _shell_command(parts)


def _resume_command(*, run_id: str, cwd: Path | None, config_path: Path | None) -> str:
    parts = _base_command("resume", cwd=cwd, config_path=config_path)
    parts.append(run_id)
    return _shell_command(parts)


def _approve_command(*, run_id: str, cwd: Path | None, config_path: Path | None) -> str:
    parts = _base_command("approve", cwd=cwd, config_path=config_path)
    parts.append(run_id)
    return _shell_command(parts)


def _reject_command(*, run_id: str, cwd: Path | None, config_path: Path | None) -> str:
    parts = _base_command("reject", cwd=cwd, config_path=config_path)
    parts.append(run_id)
    return _shell_command(parts)


def _hold_reason_command(*, run_id: str, cwd: Path | None, config_path: Path | None) -> str:
    parts = _base_command("hold-reason", cwd=cwd, config_path=config_path)
    parts.append(run_id)
    return _shell_command(parts)


def _resolve_artifact_path(
    *,
    database_path: Path,
    artifacts_root: Path,
    artifact_id: str,
    db_uri: str | None,
) -> str | None:
    with open_state_store(database_path, db_uri=db_uri) as store:
        artifact = store.artifacts.get(artifact_id)
    if artifact is None:
        return None
    candidate = Path(artifact.path)
    if not candidate.is_absolute():
        candidate = artifacts_root / candidate
    return str(candidate)


def _plan_review_paths(
    *,
    database_path: Path,
    artifacts_root: Path,
    hold_context: dict[str, object] | None,
    db_uri: str | None,
) -> dict[str, str]:
    context = hold_context if isinstance(hold_context, dict) else {}
    resolved: dict[str, str] = {}
    artifact_ids = {
        "plan": context.get("plan_artifact_id"),
        "planner_response": context.get("response_artifact_id"),
    }
    for label, artifact_id in artifact_ids.items():
        if not isinstance(artifact_id, str):
            continue
        path = _resolve_artifact_path(
            database_path=database_path,
            artifacts_root=artifacts_root,
            artifact_id=artifact_id,
            db_uri=db_uri,
        )
        if path is not None:
            resolved[label] = path
    if "plan" in resolved:
        synthesized = _synthesize_legacy_plan_review(
            plan_path=Path(resolved["plan"]),
            planner_response_path=Path(resolved["planner_response"]) if "planner_response" in resolved else None,
        )
        if synthesized is not None:
            resolved["plan"] = str(synthesized)
    return resolved


def _extract_planner_review_text(payload: str) -> str:
    extracted = extract_text_output(payload)
    stripped = payload.strip()
    if extracted.strip() and extracted.strip() != stripped:
        sanitized = _sanitize_plan_text(extracted)
        return sanitized or extracted.strip()

    if stripped:
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            for key in ("plan", "summary", "proposed_plan"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    sanitized = _sanitize_plan_text(value)
                    return sanitized or value.strip()
                if isinstance(value, list):
                    parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
                    if parts:
                        combined = "\n".join(parts)
                        sanitized = _sanitize_plan_text(combined)
                        return sanitized or combined

    sanitized = _sanitize_plan_text(extracted or stripped)
    return sanitized or extracted.strip() or stripped


def _sanitize_plan_text(text: str) -> str:
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


def _looks_like_planning_prompt(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("You are planning a grind task.")


def _extract_markdown_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        return ""

    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## ") and lines[index].strip() != heading:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _synthesize_legacy_plan_review(*, plan_path: Path, planner_response_path: Path | None) -> Path | None:
    if not plan_path.exists():
        return None

    if plan_path.suffix.lower() == ".md":
        try:
            plan_text = plan_path.read_text(encoding="utf-8")
        except OSError:
            return None

        objective = _extract_markdown_section(plan_text, "## Objective") or "Objective unavailable."
        response_text = ""
        if planner_response_path is not None and planner_response_path.exists():
            try:
                response_text = _extract_planner_review_text(planner_response_path.read_text(encoding="utf-8"))
            except OSError:
                response_text = ""
        if not response_text:
            response_text = _sanitize_plan_text(_extract_markdown_section(plan_text, "## Proposed Plan"))

        if not response_text or _looks_like_planning_prompt(response_text):
            return None

        proposed_section = _extract_markdown_section(plan_text, "## Proposed Plan")
        if response_text == proposed_section.strip() and "Here is the operator review plan:" not in proposed_section:
            return None

        plan_path.write_text(
            "\n".join(
                [
                    "# Plan Review",
                    "",
                    "This review brief was cleaned from a noisy planner transcript.",
                    "",
                    "## Objective",
                    "",
                    objective.strip(),
                    "",
                    "## Proposed Plan",
                    "",
                    response_text.strip(),
                    "",
                    "## Operator Actions",
                    "",
                    "- Approve only if this plan is specific, scoped, and reviewable.",
                    "- Reject and replan if the proposal is still vague or mismatched to your goal.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return plan_path

    if plan_path.suffix.lower() != ".json":
        return None

    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    objective = payload.get("objective") if isinstance(payload.get("objective"), str) else "Objective unavailable."
    response_text = ""
    if planner_response_path is not None and planner_response_path.exists():
        try:
            response_text = _extract_planner_review_text(planner_response_path.read_text(encoding="utf-8"))
        except OSError:
            response_text = ""

    if not response_text or _looks_like_planning_prompt(response_text):
        response_text = "Stored planner output is not a reviewable plan. Reject this hold and replan to generate a proper review brief."

    return None


def _print_hold_guidance(
    *,
    run_id: str,
    hold_type: str | None,
    hold_reason: str | None,
    hold_context: dict[str, object] | None,
    database_path: Path,
    artifacts_root: Path,
    db_uri: str | None,
    cwd: Path | None,
    config_path: Path | None,
) -> None:
    if hold_type:
        print(f"hold: {hold_type}")
    if hold_reason:
        print(f"reason: {hold_reason}")

    context = hold_context if isinstance(hold_context, dict) else {}
    if hold_type == "plan_review":
        review_paths = _plan_review_paths(
            database_path=database_path,
            artifacts_root=artifacts_root,
            hold_context=context,
            db_uri=db_uri,
        )
        if review_paths.get("plan"):
            print(f"review plan: {review_paths['plan']}")
        print(f"approve: {_approve_command(run_id=run_id, cwd=cwd, config_path=config_path)}")
        print(
            "reject: "
            + _reject_command(run_id=run_id, cwd=cwd, config_path=config_path)
            + " --reason 'needs changes'"
        )
        print(
            "resume after approval: "
            + _resume_command(run_id=run_id, cwd=cwd, config_path=config_path)
        )
        return

    if hold_type or hold_reason:
        print(f"details: {_hold_reason_command(run_id=run_id, cwd=cwd, config_path=config_path)}")


def _terminal_run_ids_to_prune(*, database_path: Path, db_uri: str | None, keep_last: int) -> list[str]:
    with open_state_store(database_path, db_uri=db_uri) as store:
        rows = store.connection.execute(
            """
            SELECT run_id
            FROM runs
            WHERE state IN ('completed', 'failed', 'aborted')
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [row[0] for row in rows[keep_last:]]


def _prune_run_records(*, database_path: Path, db_uri: str | None, run_id: str) -> dict[str, int]:
    with open_state_store(database_path, db_uri=db_uri) as store:
        artifact_rows = store.artifacts.list_by_run(run_id)
        artifact_count = len(artifact_rows)
        artifact_bytes = sum(int(artifact.size_bytes or 0) for artifact in artifact_rows)
        artifact_ids = [artifact.artifact_id for artifact in artifact_rows]
        stage_ids = [row[0] for row in store.connection.execute("SELECT stage_id FROM stages WHERE run_id = ?", [run_id]).fetchall()]
        task_ids = [row[0] for row in store.connection.execute("SELECT task_id FROM tasks WHERE run_id = ?", [run_id]).fetchall()]

        store.connection.execute("DELETE FROM adjudication_votes WHERE run_id = ?", [run_id])
        for stage_id in stage_ids:
            store.connection.execute("DELETE FROM adjudication_votes WHERE stage_id = ?", [stage_id])
        store.connection.execute(
            "DELETE FROM dispositions WHERE finding_id IN (SELECT finding_id FROM findings WHERE run_id = ?)",
            [run_id],
        )
        for stage_id in stage_ids:
            store.connection.execute("DELETE FROM dispositions WHERE stage_id = ?", [stage_id])
        store.connection.execute(
            "DELETE FROM finding_evidence WHERE finding_id IN (SELECT finding_id FROM findings WHERE run_id = ?)",
            [run_id],
        )
        store.connection.execute("DELETE FROM validations WHERE run_id = ?", [run_id])
        for stage_id in stage_ids:
            store.connection.execute("DELETE FROM validations WHERE stage_id = ?", [stage_id])
        for task_id in task_ids:
            store.connection.execute("DELETE FROM validations WHERE task_id = ?", [task_id])
        store.connection.execute("DELETE FROM model_calls WHERE run_id = ?", [run_id])
        for stage_id in stage_ids:
            store.connection.execute("DELETE FROM model_calls WHERE stage_id = ?", [stage_id])
        store.connection.execute("DELETE FROM semantic_audits WHERE run_id = ?", [run_id])
        for stage_id in stage_ids:
            store.connection.execute("DELETE FROM semantic_audits WHERE stage_id = ?", [stage_id])
        for task_id in task_ids:
            store.connection.execute("DELETE FROM semantic_audits WHERE task_id = ?", [task_id])
        store.connection.execute("DELETE FROM adjudication_panels WHERE run_id = ?", [run_id])
        for stage_id in stage_ids:
            store.connection.execute("DELETE FROM adjudication_panels WHERE stage_id = ?", [stage_id])
        for task_id in task_ids:
            store.connection.execute("DELETE FROM adjudication_panels WHERE task_id = ?", [task_id])
        store.connection.execute("DELETE FROM retrieval_index_queue WHERE run_id = ?", [run_id])
        store.connection.execute("DELETE FROM operator_actions WHERE run_id = ?", [run_id])
        store.connection.execute("DELETE FROM run_leases WHERE run_id = ?", [run_id])
        store.connection.execute("DELETE FROM transitions WHERE run_id = ?", [run_id])
        store.connection.execute("DELETE FROM workspace_checkpoints WHERE run_id = ?", [run_id])
        store.connection.execute("DELETE FROM findings WHERE run_id = ?", [run_id])
        store.connection.execute("DELETE FROM stages WHERE run_id = ?", [run_id])
        store.connection.execute("DELETE FROM tasks WHERE run_id = ?", [run_id])
        for artifact_id in artifact_ids:
            store.connection.execute("DELETE FROM artifacts WHERE artifact_id = ?", [artifact_id])
        store.connection.execute("DELETE FROM runs WHERE run_id = ?", [run_id])

    return {
        "runs_pruned": 1,
        "artifacts_pruned": artifact_count,
        "artifact_bytes_pruned": artifact_bytes,
    }


def _delete_run_artifacts(*, artifacts_root: Path, run_id: str) -> None:
    run_dir = artifacts_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)


def _resolve_quack_token(*, database_path: Path, db_uri: str | None) -> str | None:
    if not db_uri or not db_uri.startswith("quack:") or not is_local_quack_uri(db_uri):
        return None
    return ensure_local_quack_server(database_path, db_uri)


def _cleanup_artifact_layout(
    *,
    database_path: Path,
    db_uri: str | None,
    artifacts_root: Path,
    quack_token: str | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    orphan_files_deleted = 0
    orphan_bytes_deleted = 0
    directories_removed = 0

    with open_state_store(database_path, db_uri=db_uri, quack_token=quack_token) as store:
        artifacts = store.artifacts.list_all()
        referenced_paths = {artifacts_root / artifact.path for artifact in artifacts}

    for file_path in sorted(path for path in artifacts_root.rglob("*") if path.is_file()):
        if file_path not in referenced_paths:
            orphan_files_deleted += 1
            orphan_bytes_deleted += file_path.stat().st_size
            if not dry_run:
                file_path.unlink()

    for dir_path in sorted((path for path in artifacts_root.rglob("*") if path.is_dir()), reverse=True):
        if any(dir_path.iterdir()):
            continue
        directories_removed += 1
        if not dry_run:
            dir_path.rmdir()

    return {
        "orphan_files_deleted": orphan_files_deleted,
        "orphan_bytes_deleted": orphan_bytes_deleted,
        "directories_removed": directories_removed,
        "dry_run": dry_run,
    }


def _vacuum_database(*, database_path: Path, db_uri: str | None) -> None:
    with open_state_store(database_path, db_uri=db_uri) as store:
        store.connection.execute("VACUUM")


def _prune_run_ids(
    *,
    run_ids: list[str],
    database_path: Path,
    db_uri: str | None,
    artifacts_root: Path,
    retrieval_service: LanceDBRetrievalService | None,
) -> dict[str, object]:
    payload = {
        "runs_pruned": 0,
        "artifacts_pruned": 0,
        "artifact_bytes_pruned": 0,
        "retrieval_documents_pruned": 0,
        "retrieval_collections_pruned": {},
        "run_ids": run_ids,
    }
    for run_id in run_ids:
        summary = _prune_run_records(
            database_path=database_path,
            db_uri=db_uri,
            run_id=run_id,
        )
        _delete_run_artifacts(artifacts_root=artifacts_root, run_id=run_id)
        payload["runs_pruned"] += summary["runs_pruned"]
        payload["artifacts_pruned"] += summary["artifacts_pruned"]
        payload["artifact_bytes_pruned"] += summary["artifact_bytes_pruned"]

        if retrieval_service is not None:
            retrieval_summary = retrieval_service.delete_run(run_id=run_id)
            payload["retrieval_documents_pruned"] += retrieval_summary["documents_deleted"]
            collections = payload["retrieval_collections_pruned"]
            for collection, count in retrieval_summary["collections"].items():
                collections[collection] = collections.get(collection, 0) + count

    if run_ids:
        _vacuum_database(database_path=database_path, db_uri=db_uri)
    return payload


def _auto_prune_if_configured(*, cwd: Path, config: EngineConfig, retrieval_service: LanceDBRetrievalService) -> dict[str, object] | None:
    if config.retention.mode != "auto":
        return None
    keep_last = config.retention.keep_last_terminal_runs
    if keep_last is None:
        return None

    run_ids = _terminal_run_ids_to_prune(
        database_path=config.state_path(cwd),
        db_uri=config.state_db_uri(),
        keep_last=keep_last,
    )
    try:
        return _prune_run_ids(
            run_ids=run_ids,
            database_path=config.state_path(cwd),
            db_uri=config.state_db_uri(),
            artifacts_root=config.artifacts_root(cwd),
            retrieval_service=retrieval_service,
        )
    except Exception as error:
        db_uri = config.state_db_uri()
        if not db_uri or not db_uri.startswith("quack:"):
            raise
        return {
            "runs_pruned": 0,
            "artifacts_pruned": 0,
            "artifact_bytes_pruned": 0,
            "retrieval_documents_pruned": 0,
            "retrieval_collections_pruned": {},
            "run_ids": run_ids,
            "skipped": True,
            "reason": f"auto-prune skipped for Quack transport: {error}",
        }


def _self_host_quack_probe(*, cwd: Path, config: EngineConfig) -> dict[str, object]:
    db_uri = config.state_db_uri()
    if not db_uri or not db_uri.startswith("quack:"):
        return {
            "status": "failed",
            "reason": "self-hosting requires state.db_uri to point at a quack: URI",
            "db_uri": db_uri,
        }

    probe_suffix = secrets.token_hex(4)
    probe_run_id = f"self_host_probe_run_{probe_suffix}"
    probe_worker_id = f"self_host_probe_worker_{socket.gethostname()}_{probe_suffix}"
    database_path = config.state_path(cwd)
    quack_token: str | None = None
    if db_uri.startswith("quack:"):
        quack_token = ensure_local_quack_server(database_path, db_uri, force_restart=True)

    with open_state_store(database_path, db_uri=db_uri, quack_token=quack_token) as store:
        worker = Worker(worker_id=probe_worker_id, hostname=socket.gethostname(), pid=1)
        store.workers.register(worker)
        if store.workers.get(probe_worker_id) is None:
            return {"status": "failed", "reason": "worker registration did not persist", "db_uri": db_uri}
        store.workers.heartbeat(probe_worker_id)

        run = Run(
            run_id=probe_run_id,
            repo_path=str(cwd),
            policy_pack_path=str((cwd / ".grind").resolve()),
            policy_schema_ver="1",
            created_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
            requested_objective="self-host readiness probe",
            state=RunState.CREATED,
            operator_status=OperatorStatus.NONE,
            max_iterations=config.execution.max_iterations,
            budget_limit_usd=config.execution.budget_limit_usd,
            total_cost_usd=Decimal("0"),
        )
        store.runs.create(run)
        lease = acquire_lease(store.connection, probe_run_id, probe_worker_id)
        release_lease(store.connection, lease.lease_id)

    return {"status": "passed", "reason": "Quack read/write probe succeeded", "db_uri": db_uri}


def _self_host_validation_probe(*, config: EngineConfig) -> dict[str, object]:
    command_reports: list[dict[str, object]] = []
    overall = "passed"
    for command in config.validation.commands:
        try:
            argv = normalize_shell_free_command(command)
            risk = classify_command(argv)
            status = "passed" if risk != "risky" else "failed"
            if status != "passed":
                overall = "failed"
            command_reports.append({
                "command": command,
                "argv": argv,
                "risk": risk,
                "status": status,
            })
        except ValidationCommandError as error:
            overall = "failed"
            command_reports.append({
                "command": command,
                "status": "failed",
                "reason": str(error),
            })
    return {"status": overall, "commands": command_reports}


def _self_host_backend_probe(*, cwd: Path, config_path: Path, strict: bool, smoke: bool) -> dict[str, object]:
    verifier = DefaultBackendVerifier()
    config = load_engine_config(config_path)
    reports: dict[str, object] = {}
    overall = VerificationOverallStatus.PASSED.value
    for role in ("planner", "implementer", "checker", "adjudicator"):
        request = VerificationRequest(
            backend=config.models[role].provider,
            role=role,
            cwd=cwd,
            config_path=config_path,
            smoke=smoke,
            strict=strict,
        )
        report = verifier.verify(request)
        reports[role] = report.model_dump(mode="json")
        if report.overall_status == VerificationOverallStatus.FAILED:
            overall = VerificationOverallStatus.FAILED.value
        elif report.overall_status != VerificationOverallStatus.PASSED and overall == VerificationOverallStatus.PASSED.value:
            overall = VerificationOverallStatus.INCONCLUSIVE.value
    return {"status": overall, "roles": reports}


def _verify_self_host(*, cwd: Path, config_path: Path, strict: bool, smoke: bool) -> dict[str, object]:
    config = load_engine_config(config_path)
    quack = _self_host_quack_probe(cwd=cwd, config=config)
    backends = _self_host_backend_probe(cwd=cwd, config_path=config_path, strict=strict, smoke=smoke)
    validation = _self_host_validation_probe(config=config)
    statuses = {quack["status"], backends["status"], validation["status"]}
    if "failed" in statuses:
        overall = VerificationOverallStatus.FAILED.value
    elif VerificationOverallStatus.INCONCLUSIVE.value in statuses:
        overall = VerificationOverallStatus.INCONCLUSIVE.value
    else:
        overall = VerificationOverallStatus.PASSED.value
    return {
        "overall_status": overall,
        "quack": quack,
        "backends": backends,
        "validation": validation,
    }
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grind")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--cwd", type=Path)
    init.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("objective", nargs="?")
    run.add_argument("--objective-file", type=Path)
    run.add_argument("--cwd", type=Path)
    run.add_argument("--config", dest="config_path", type=Path)
    run.add_argument("--policy-pack", dest="policy_pack_path", type=Path)
    run.add_argument("--json", action="store_true")

    resume = subparsers.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--cwd", type=Path)
    resume.add_argument("--config", dest="config_path", type=Path)
    resume.add_argument("--checkpoint-id")
    resume.add_argument("--restore-checkpoint", action="store_true")
    resume.add_argument("--json", action="store_true")

    approve = subparsers.add_parser("approve")
    approve.add_argument("run_id")
    approve.add_argument("--reason")
    approve.add_argument("--max-iterations", type=int)
    approve.add_argument("--budget-limit-usd", type=Decimal)
    approve.add_argument("--cwd", type=Path)
    approve.add_argument("--config", dest="config_path", type=Path)
    approve.add_argument("--json", action="store_true")

    reject = subparsers.add_parser("reject")
    reject.add_argument("run_id")
    reject.add_argument("--reason")
    reject.add_argument("--cwd", type=Path)
    reject.add_argument("--config", dest="config_path", type=Path)
    reject.add_argument("--json", action="store_true")

    abort = subparsers.add_parser("abort")
    abort.add_argument("run_id")
    abort.add_argument("--reason")
    abort.add_argument("--cwd", type=Path)
    abort.add_argument("--config", dest="config_path", type=Path)
    abort.add_argument("--json", action="store_true")

    hold_reason = subparsers.add_parser("hold-reason")
    hold_reason.add_argument("run_id")
    hold_reason.add_argument("--cwd", type=Path)
    hold_reason.add_argument("--config", dest="config_path", type=Path)
    hold_reason.add_argument("--json", action="store_true")

    patch_policy = subparsers.add_parser("patch-policy")
    patch_policy.add_argument("run_id")
    patch_policy.add_argument("--reason")
    patch_policy.add_argument("--max-iterations", type=int)
    patch_policy.add_argument("--budget-limit-usd", type=Decimal)
    patch_policy.add_argument("--validation-command", action="append", dest="validation_commands")
    patch_policy.add_argument("--cwd", type=Path)
    patch_policy.add_argument("--config", dest="config_path", type=Path)
    patch_policy.add_argument("--json", action="store_true")

    restore_checkpoint = subparsers.add_parser("restore-checkpoint")
    restore_checkpoint.add_argument("run_id")
    restore_checkpoint.add_argument("--cwd", type=Path)
    restore_checkpoint.add_argument("--config", dest="config_path", type=Path)
    restore_checkpoint.add_argument("--checkpoint-id")
    restore_checkpoint.add_argument("--json", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--run-id")
    status.add_argument("--cwd", type=Path)
    status.add_argument("--config", dest="config_path", type=Path)
    status.add_argument("--json", action="store_true")

    findings = subparsers.add_parser("findings")
    findings.add_argument("--run-id")
    findings.add_argument("--cwd", type=Path)
    findings.add_argument("--config", dest="config_path", type=Path)
    findings.add_argument("--json", action="store_true")

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("target", nargs="?")
    inspect.add_argument("--run-id")
    inspect.add_argument("--cwd", type=Path)
    inspect.add_argument("--config", dest="config_path", type=Path)
    inspect.add_argument("--json", action="store_true")

    report = subparsers.add_parser("report")
    report.add_argument("--run-id")
    report.add_argument("--cwd", type=Path)
    report.add_argument("--config", dest="config_path", type=Path)
    report.add_argument("--json", action="store_true")

    retrieval_index = subparsers.add_parser("retrieval-index")
    retrieval_index.add_argument("--run-id")
    retrieval_index.add_argument("--cwd", type=Path)
    retrieval_index.add_argument("--config", dest="config_path", type=Path)
    retrieval_index.add_argument("--json", action="store_true")

    retrieval_search = subparsers.add_parser("retrieval-search")
    retrieval_search.add_argument("query")
    retrieval_search.add_argument("--run-id")
    retrieval_search.add_argument("--collection")
    retrieval_search.add_argument("--limit", type=int)
    retrieval_search.add_argument("--cwd", type=Path)
    retrieval_search.add_argument("--config", dest="config_path", type=Path)
    retrieval_search.add_argument("--json", action="store_true")

    prune = subparsers.add_parser("prune")
    prune.add_argument("--keep-last", type=int, required=True)
    prune.add_argument("--cwd", type=Path)
    prune.add_argument("--config", dest="config_path", type=Path)
    prune.add_argument("--dry-run", action="store_true")
    prune.add_argument("--json", action="store_true")

    cleanup_artifacts = subparsers.add_parser("cleanup-artifacts")
    cleanup_artifacts.add_argument("--cwd", type=Path)
    cleanup_artifacts.add_argument("--config", dest="config_path", type=Path)
    cleanup_artifacts.add_argument("--dry-run", action="store_true")
    cleanup_artifacts.add_argument("--json", action="store_true")

    verify_backend = subparsers.add_parser("verify-backend")
    verify_backend.add_argument(
        "--backend",
        choices=["github_cli", "kilo_cli"],
        help="backend to verify; omit to verify all configured backends",
    )
    verify_backend.add_argument(
        "--role",
        choices=["planner", "implementer", "checker", "adjudicator"],
    )
    verify_backend.add_argument("--model")
    verify_backend.add_argument("--agent")
    verify_backend.add_argument("--variant")
    verify_backend.add_argument("--cwd", type=Path)
    verify_backend.add_argument("--config", dest="config_path", type=Path)

    smoke_group = verify_backend.add_mutually_exclusive_group()
    smoke_group.add_argument("--smoke", dest="smoke", action="store_true", default=True)
    smoke_group.add_argument("--no-smoke", dest="smoke", action="store_false")

    verify_backend.add_argument("--strict", action="store_true")
    verify_backend.add_argument("--json", action="store_true")

    verify_self_host = subparsers.add_parser("verify-self-host")
    verify_self_host.add_argument("--cwd", type=Path)
    verify_self_host.add_argument("--config", dest="config_path", type=Path)
    verify_self_host.add_argument("--strict", action="store_true")
    verify_self_host.add_argument("--json", action="store_true")
    verify_self_host_smoke = verify_self_host.add_mutually_exclusive_group()
    verify_self_host_smoke.add_argument("--smoke", dest="smoke", action="store_true", default=True)
    verify_self_host_smoke.add_argument("--no-smoke", dest="smoke", action="store_false")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        cwd = args.cwd or Path.cwd()
        try:
            config_path = init_engine_workspace(cwd, force=args.force)
        except FileExistsError as error:
            print(str(error), file=sys.stderr)
            return 2

        config = load_engine_config(config_path)
        bootstrap_state_store(config.state_path(cwd), db_uri=config.state_db_uri())
        print(f"initialized: {config_path}")
        print(f"state database: {config.state_path(cwd)}")
        print(f"artifacts root: {config.artifacts_root(cwd)}")
        print(f"retention mode: {config.retention.mode}")
        print(
            f"schema version: {current_schema_version(config.state_path(cwd), db_uri=config.state_db_uri())}"
        )
        return 0

    if args.command == "verify-self-host":
        cwd = args.cwd or Path.cwd()
        config_path = args.config_path or default_engine_config_path(cwd)
        try:
            payload = _verify_self_host(cwd=cwd, config_path=config_path, strict=args.strict, smoke=args.smoke)
        except (VerificationConfigError, QuackConnectionError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"overall_status: {payload['overall_status']}")
            print(f"quack: {payload['quack']['status']} ({payload['quack'].get('reason', '-')})")
            print(f"backends: {payload['backends']['status']}")
            print(f"validation: {payload['validation']['status']}")

        if payload["overall_status"] == VerificationOverallStatus.FAILED.value:
            return 1
        if payload["overall_status"] != VerificationOverallStatus.PASSED.value:
            return 2
        return 0

    if args.command == "cleanup-artifacts":
        cwd = args.cwd or Path.cwd()
        config_path = args.config_path or default_engine_config_path(cwd)
        config = load_engine_config(config_path)
        payload = _cleanup_artifact_layout(
            database_path=config.state_path(cwd),
            db_uri=config.state_db_uri(),
            artifacts_root=config.artifacts_root(cwd),
            quack_token=_resolve_quack_token(database_path=config.state_path(cwd), db_uri=config.state_db_uri()),
            dry_run=args.dry_run,
        )

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"orphan_files_deleted: {payload['orphan_files_deleted']}")
            print(f"orphan_bytes_deleted: {payload['orphan_bytes_deleted']}")
            print(f"directories_removed: {payload['directories_removed']}")
        return 0

    if args.command in {"run", "resume", "approve", "reject", "abort", "hold-reason", "patch-policy", "restore-checkpoint", "status", "findings", "inspect", "report", "retrieval-index", "retrieval-search", "prune"}:
        cwd = args.cwd or Path.cwd()
        config_path = args.config_path or default_engine_config_path(cwd)
        config = load_engine_config(config_path)
        orchestrator = MinimalOrchestrator(
            cwd=cwd,
            config_path=config_path,
            policy_pack_path=getattr(args, "policy_pack_path", None),
        )
        retrieval_service = LanceDBRetrievalService(cwd=cwd, config=config)

    if args.command == "run":
        if args.objective and args.objective_file:
            print("provide either an inline objective or --objective-file, not both", file=sys.stderr)
            return 2
        if not args.objective and not args.objective_file:
            print("an inline objective or --objective-file is required", file=sys.stderr)
            return 2

        if args.objective_file:
            objective = args.objective_file.read_text(encoding="utf-8")
            source_kind = TaskSourceKind.FILE
        else:
            objective = args.objective
            source_kind = TaskSourceKind.INLINE

        outcome = orchestrator.run(objective=objective, source_kind=source_kind)

        payload = {
            "run_id": outcome.run_id,
            "task_id": outcome.task_id,
            "planning_stage_id": outcome.planning_stage_id,
            "checkpoint_id": outcome.checkpoint_id,
            "final_state": outcome.final_state.value,
            "operator_status": outcome.operator_status.value,
            "hold_type": outcome.hold_type,
            "hold_reason": outcome.hold_reason,
            "hold_context": outcome.hold_context,
            "database_path": str(outcome.database_path),
            "artifacts_root": str(outcome.artifacts_root),
        }
        if outcome.hold_type == "plan_review":
            payload["review_paths"] = _plan_review_paths(
                database_path=outcome.database_path,
                artifacts_root=outcome.artifacts_root,
                hold_context=outcome.hold_context,
                db_uri=config.state_db_uri(),
            )
        auto_prune = _auto_prune_if_configured(cwd=cwd, config=config, retrieval_service=retrieval_service)
        if auto_prune is not None:
            payload["auto_prune"] = auto_prune
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"run_id: {outcome.run_id}")
            print(f"status: {outcome.final_state.value}")
            _print_hold_guidance(
                run_id=outcome.run_id,
                hold_type=outcome.hold_type,
                hold_reason=outcome.hold_reason,
                hold_context=outcome.hold_context,
                database_path=outcome.database_path,
                artifacts_root=outcome.artifacts_root,
                db_uri=config.state_db_uri(),
                cwd=args.cwd,
                config_path=args.config_path,
            )
        return 0

    if args.command == "resume":
        try:
            outcome = orchestrator.resume(
                run_id=args.run_id,
                checkpoint_id=args.checkpoint_id,
                restore_checkpoint=args.restore_checkpoint,
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        payload = {
            "run_id": outcome.run_id,
            "task_id": outcome.task_id,
            "validation_stage_id": outcome.validation_stage_id,
            "final_state": outcome.final_state.value,
            "operator_status": outcome.operator_status.value,
            "hold_type": outcome.hold_type,
            "hold_reason": outcome.hold_reason,
            "hold_context": outcome.hold_context,
            "restored_checkpoint_id": outcome.restored_checkpoint_id,
            "database_path": str(outcome.database_path),
            "artifacts_root": str(outcome.artifacts_root),
        }
        if outcome.hold_type == "plan_review":
            payload["review_paths"] = _plan_review_paths(
                database_path=outcome.database_path,
                artifacts_root=outcome.artifacts_root,
                hold_context=outcome.hold_context,
                db_uri=config.state_db_uri(),
            )
        auto_prune = _auto_prune_if_configured(cwd=cwd, config=config, retrieval_service=retrieval_service)
        if auto_prune is not None:
            payload["auto_prune"] = auto_prune
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"run_id: {outcome.run_id}")
            print(f"status: {outcome.final_state.value}")
            if outcome.restored_checkpoint_id:
                print(f"restored_checkpoint_id: {outcome.restored_checkpoint_id}")
            _print_hold_guidance(
                run_id=outcome.run_id,
                hold_type=outcome.hold_type,
                hold_reason=outcome.hold_reason,
                hold_context=outcome.hold_context,
                database_path=outcome.database_path,
                artifacts_root=outcome.artifacts_root,
                db_uri=config.state_db_uri(),
                cwd=args.cwd,
                config_path=args.config_path,
            )
        return 0

    if args.command == "prune":
        if args.keep_last < 0:
            print("--keep-last must be >= 0", file=sys.stderr)
            return 2

        run_ids = _terminal_run_ids_to_prune(
            database_path=config.state_path(cwd),
            db_uri=config.state_db_uri(),
            keep_last=args.keep_last,
        )
        payload = {
            "keep_last": args.keep_last,
            "dry_run": args.dry_run,
            "run_ids": run_ids,
        }
        if not args.dry_run:
            payload.update(
                _prune_run_ids(
                    run_ids=run_ids,
                    database_path=config.state_path(cwd),
                    db_uri=config.state_db_uri(),
                    artifacts_root=config.artifacts_root(cwd),
                    retrieval_service=retrieval_service,
                )
            )
        else:
            payload.update(
                {
                    "runs_pruned": 0,
                    "artifacts_pruned": 0,
                    "artifact_bytes_pruned": 0,
                    "retrieval_documents_pruned": 0,
                    "retrieval_collections_pruned": {},
                }
            )

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"runs selected: {len(run_ids)}")
            print(f"keep last terminal runs: {args.keep_last}")
            if args.dry_run:
                print("mode: dry_run")
            else:
                print(f"runs pruned: {payload['runs_pruned']}")
                print(f"artifacts pruned: {payload['artifacts_pruned']}")
                print(f"artifact bytes pruned: {payload['artifact_bytes_pruned']}")
                print(f"retrieval documents pruned: {payload['retrieval_documents_pruned']}")
            for run_id in run_ids:
                print(f"run_id: {run_id}")
        return 0

    if args.command == "approve":
        try:
            payload = orchestrator.approve(
                run_id=args.run_id,
                note=args.reason,
                max_iterations=args.max_iterations,
                budget_limit_usd=args.budget_limit_usd,
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        auto_prune = _auto_prune_if_configured(cwd=cwd, config=config, retrieval_service=retrieval_service)
        if auto_prune is not None:
            payload["auto_prune"] = auto_prune

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if args.command == "reject":
        try:
            payload = orchestrator.reject(run_id=args.run_id, note=args.reason)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        auto_prune = _auto_prune_if_configured(cwd=cwd, config=config, retrieval_service=retrieval_service)
        if auto_prune is not None:
            payload["auto_prune"] = auto_prune

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if args.command == "abort":
        try:
            payload = orchestrator.abort(run_id=args.run_id, note=args.reason)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        auto_prune = _auto_prune_if_configured(cwd=cwd, config=config, retrieval_service=retrieval_service)
        if auto_prune is not None:
            payload["auto_prune"] = auto_prune

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if args.command == "hold-reason":
        try:
            payload = orchestrator.hold_reason(run_id=args.run_id)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        if payload.get("hold_type") == "plan_review":
            payload["review_paths"] = _plan_review_paths(
                database_path=config.state_path(cwd),
                artifacts_root=config.artifacts_root(cwd),
                hold_context=payload.get("hold_context"),
                db_uri=config.state_db_uri(),
            )

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"run_id: {payload['run_id']}")
            print(f"status: {payload['state']}")
            if payload.get("operator_status") is not None:
                print(f"operator_status: {payload['operator_status']}")
            _print_hold_guidance(
                run_id=payload["run_id"],
                hold_type=payload.get("hold_type"),
                hold_reason=payload.get("hold_reason"),
                hold_context=payload.get("hold_context"),
                database_path=config.state_path(cwd),
                artifacts_root=config.artifacts_root(cwd),
                db_uri=config.state_db_uri(),
                cwd=args.cwd,
                config_path=args.config_path,
            )
        return 0

    if args.command == "patch-policy":
        try:
            payload = orchestrator.patch_policy(
                run_id=args.run_id,
                note=args.reason,
                max_iterations=args.max_iterations,
                budget_limit_usd=args.budget_limit_usd,
                validation_commands_override=args.validation_commands,
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if args.command == "restore-checkpoint":
        try:
            checkpoint = orchestrator.restore_checkpoint(
                run_id=args.run_id,
                checkpoint_id=args.checkpoint_id,
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        payload = {
            "run_id": checkpoint.run_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "status": checkpoint.status.value,
            "artifact_id": checkpoint.artifact_id,
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"run_id: {checkpoint.run_id}")
            print(f"checkpoint_id: {checkpoint.checkpoint_id}")
            print(f"status: {checkpoint.status}")
            print(f"artifact_id: {checkpoint.artifact_id}")
        return 0

    if args.command == "status":
        try:
            payload = orchestrator.status(run_id=args.run_id)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if args.command == "findings":
        try:
            payload = orchestrator.findings(run_id=args.run_id)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            if not payload:
                print("no findings")
            for finding in payload:
                print(f"{finding['severity']} {finding['status']} {finding['title']}")
        return 0

    if args.command == "inspect":
        try:
            payload = orchestrator.inspect(run_id=args.run_id, selector=args.target)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            artifacts = payload.get("artifacts")
            if isinstance(artifacts, list):
                for artifact in artifacts:
                    print(f"{artifact['artifact_id']} {artifact['artifact_type']} {artifact['path']}")
            else:
                artifact = payload.get("artifact")
                if artifact is not None:
                    print(f"artifact_id: {artifact['artifact_id']}")
                    print(f"artifact_type: {artifact['artifact_type']}")
                    print(f"path: {artifact['path']}")
                content = payload.get("content")
                if content is not None:
                    if isinstance(content, (dict, list)):
                        print(json.dumps(content, indent=2))
                    else:
                        print(content)
        return 0

    if args.command == "report":
        try:
            payload = orchestrator.report(run_id=args.run_id)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload, indent=2))
        return 0

    if args.command == "retrieval-index":
        try:
            run_id = args.run_id or orchestrator.status()["run_id"]
            payload = retrieval_service.index_run(run_id=run_id)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if args.command == "retrieval-search":
        try:
            payload = retrieval_service.search(
                query=args.query,
                run_id=args.run_id,
                collection=args.collection,
                limit=args.limit,
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for result in payload["results"]:
                print(f"{result['collection']} {result['chunk_id']} {result['score']}")
                print(result["chunk_text"])
        return 0

    if args.command != "verify-backend":
        parser.error(f"unsupported command: {args.command}")

    backends: list[str] = [args.backend] if args.backend else ["github_cli", "kilo_cli"]

    verifier = DefaultBackendVerifier()
    worst_exit = 0

    for backend in backends:
        request = VerificationRequest(
            backend=backend,
            role=args.role,
            model=args.model,
            agent=args.agent,
            variant=args.variant,
            cwd=args.cwd,
            config_path=args.config_path,
            smoke=args.smoke,
            strict=args.strict,
        )

        try:
            report = verifier.verify(request)
        except VerificationConfigError as error:
            print(str(error), file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(report.model_dump(mode="json"), indent=2))
        else:
            print(f"backend: {report.backend}")
            print(f"role: {report.role or '-'}")
            print(f"overall_status: {report.overall_status}")
            for probe in report.probes:
                print(f"- {probe.probe_id}: {probe.status} ({probe.status_reason})")

        if report.overall_status == VerificationOverallStatus.FAILED:
            worst_exit = max(worst_exit, 1)
        elif report.overall_status != VerificationOverallStatus.PASSED:
            worst_exit = max(worst_exit, 2)

        if len(backends) > 1:
            print()

    return worst_exit
