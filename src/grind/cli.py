from __future__ import annotations

import argparse
from decimal import Decimal
import json
import sys
from pathlib import Path

from grind.config import default_engine_config_path, init_engine_workspace, load_engine_config
from grind.engine.orchestrator import MinimalOrchestrator
from grind.models.enums import TaskSourceKind
from grind.retrieval import LanceDBRetrievalService
from grind.state import bootstrap_state_store, current_schema_version
from grind.verification.models import VerificationOverallStatus, VerificationRequest
from grind.verification.service import DefaultBackendVerifier, VerificationConfigError


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

    if args.command in {"run", "resume", "approve", "reject", "abort", "hold-reason", "patch-policy", "restore-checkpoint", "status", "findings", "inspect", "report", "retrieval-index", "retrieval-search"}:
        cwd = args.cwd or Path.cwd()
        config_path = args.config_path or default_engine_config_path(cwd)
        orchestrator = MinimalOrchestrator(
            cwd=cwd,
            config_path=config_path,
            policy_pack_path=getattr(args, "policy_pack_path", None),
        )
        retrieval_service = LanceDBRetrievalService(cwd=cwd, config=load_engine_config(config_path))

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
            "database_path": str(outcome.database_path),
            "artifacts_root": str(outcome.artifacts_root),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"run_id: {outcome.run_id}")
            print(f"task_id: {outcome.task_id}")
            print(f"final_state: {outcome.final_state}")
            print(f"operator_status: {outcome.operator_status}")
            print(f"state database: {outcome.database_path}")
            print(f"artifacts root: {outcome.artifacts_root}")
            print("note: planner output is stored for operator review; use `grind resume` to continue")
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
            "restored_checkpoint_id": outcome.restored_checkpoint_id,
            "database_path": str(outcome.database_path),
            "artifacts_root": str(outcome.artifacts_root),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"run_id: {outcome.run_id}")
            print(f"validation_stage_id: {outcome.validation_stage_id}")
            print(f"final_state: {outcome.final_state}")
            print(f"operator_status: {outcome.operator_status}")
            if outcome.restored_checkpoint_id:
                print(f"restored_checkpoint_id: {outcome.restored_checkpoint_id}")
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

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
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
