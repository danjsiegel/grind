from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from hashlib import sha256
from io import BytesIO
import tarfile
from pathlib import Path
import subprocess

from grind.engine.checkpoints import EXCLUDED_PATHS
from grind.models import DifferenceSurface, Finding, FindingStatus, Run, Task
from grind.validation import ValidationExecutionResult


@dataclass(frozen=True)
class DifferenceSurfaceBuildResult:
    surface: DifferenceSurface
    capability_level: str
    unsupported_checks: list[str]
    authoritative_touched_files: list[str]
    added_files: list[str]
    modified_files: list[str]
    removed_files: list[str]
    existing_reported_files: list[str]
    missing_reported_files: list[str]
    reported_but_unchanged_files: list[str]
    unreported_changed_files: list[str]
    git_touched_files: list[str]


def build_difference_surface(
    *,
    cwd: Path,
    run: Run,
    task: Task,
    iteration: int,
    observed_delta: dict[str, object],
    validation_results: list[ValidationExecutionResult],
    open_findings: list[Finding],
    baseline_snapshot_path: Path | None,
    stop_on_failure: bool,
    scope_excludes: Sequence[str] = (),
) -> DifferenceSurfaceBuildResult:
    reported_touched_files = [str(path) for path in observed_delta.get("reported_touched_files", [])]
    current_snapshot = _snapshot_workspace(cwd, scope_excludes=scope_excludes)
    baseline_snapshot = (
        _snapshot_tarball(baseline_snapshot_path, scope_excludes=scope_excludes)
        if baseline_snapshot_path
        else {}
    )

    added_files = sorted(path for path in current_snapshot if path not in baseline_snapshot)
    removed_files = sorted(path for path in baseline_snapshot if path not in current_snapshot)
    modified_files = sorted(
        path
        for path, digest in current_snapshot.items()
        if path in baseline_snapshot and baseline_snapshot[path] != digest
    )
    git_touched_files = _git_touched_files(cwd, scope_excludes=scope_excludes)
    authoritative_touched_files = sorted(set(added_files + modified_files + removed_files + git_touched_files))

    existing_reported_files = sorted(path for path in reported_touched_files if path in current_snapshot)
    missing_reported_files = sorted(path for path in reported_touched_files if path not in current_snapshot)
    reported_but_unchanged_files = sorted(
        path for path in existing_reported_files if path not in authoritative_touched_files
    )
    unreported_changed_files = sorted(
        path for path in authoritative_touched_files if path not in reported_touched_files
    )

    validation_summary = [
        {
            "command": result.command,
            "exit_code": result.returncode,
            "status": "passed" if result.returncode == 0 else "failed",
        }
        for result in validation_results
    ]
    claims = [claim for claim in observed_delta.get("claims_made", []) if isinstance(claim, dict)]
    capability_parts = ["filesystem"]
    if baseline_snapshot_path is not None:
        capability_parts.append("baseline_snapshot")
    if git_touched_files:
        capability_parts.append("git")
    capability_level = "+".join(capability_parts)
    unsupported_checks = ["symbol_graph", "dependency_graph", "invariant_enforcement"]

    surface = DifferenceSurface(
        surface_id=_surface_id(run.run_id, iteration),
        run_id=run.run_id,
        task_id=task.task_id,
        iteration=iteration,
        requested_delta={
            "objective": task.raw_input.strip(),
            "phase": observed_delta.get("source_stage"),
        },
        observed_delta={
            **observed_delta,
            "authoritative_touched_files": authoritative_touched_files,
            "added_files": added_files,
            "modified_files": modified_files,
            "removed_files": removed_files,
            "existing_reported_files": existing_reported_files,
            "missing_reported_files": missing_reported_files,
            "reported_but_unchanged_files": reported_but_unchanged_files,
            "unreported_changed_files": unreported_changed_files,
            "git_touched_files": git_touched_files,
        },
        evidence_delta={
            "baseline_snapshot_present": baseline_snapshot_path is not None,
            "baseline_file_count": len(baseline_snapshot),
            "current_file_count": len(current_snapshot),
            "validation_result_count": len(validation_results),
            "open_findings_before_check": len(open_findings),
            "claim_count": len(claims),
            "claims_with_evidence": sum(1 for claim in claims if claim.get("evidence")),
        },
        risk_delta={
            "missing_reported_files": missing_reported_files,
            "reported_but_unchanged_files": reported_but_unchanged_files,
            "unreported_changed_files": unreported_changed_files,
            "removed_files": removed_files,
            "validation_failed": any(result.returncode != 0 for result in validation_results),
        },
        findings_delta={
            "open_findings": [finding.stable_id for finding in open_findings if finding.status == FindingStatus.OPEN],
        },
        validation_delta={
            "results": validation_summary,
            "stop_on_failure": stop_on_failure,
            "hinted_commands": [
                hint.get("command")
                for hint in observed_delta.get("validation_hints", [])
                if isinstance(hint, dict) and hint.get("command")
            ],
        },
        semantic_audit_delta={
            "capability_level": capability_level,
            "blocking_finding_count": 0,
            "advisory_finding_count": len(missing_reported_files) + len(unreported_changed_files),
            "unsupported_checks": unsupported_checks,
        },
        invariant_delta={},
        policy_delta={
            "iteration_count": run.iteration_count,
            "max_iterations": run.max_iterations,
            "remaining_iterations": max(run.max_iterations - iteration, 0),
            "budget_limit_usd": str(run.budget_limit_usd) if run.budget_limit_usd is not None else None,
            "total_cost_usd": str(run.total_cost_usd),
        },
    )

    return DifferenceSurfaceBuildResult(
        surface=surface,
        capability_level=capability_level,
        unsupported_checks=unsupported_checks,
        authoritative_touched_files=authoritative_touched_files,
        added_files=added_files,
        modified_files=modified_files,
        removed_files=removed_files,
        existing_reported_files=existing_reported_files,
        missing_reported_files=missing_reported_files,
        reported_but_unchanged_files=reported_but_unchanged_files,
        unreported_changed_files=unreported_changed_files,
        git_touched_files=git_touched_files,
    )


def _surface_id(run_id: str, iteration: int) -> str:
    return f"surface_{run_id}_{iteration}"


def _snapshot_workspace(cwd: Path, *, scope_excludes: Sequence[str] = ()) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(cwd.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(cwd)
        if _is_excluded(relative_path, scope_excludes=scope_excludes):
            continue
        snapshot[str(relative_path)] = _digest_bytes(path.read_bytes())
    return snapshot


def _snapshot_tarball(snapshot_path: Path, *, scope_excludes: Sequence[str] = ()) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    with tarfile.open(snapshot_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_path = Path(member.name)
            if _is_excluded(member_path, scope_excludes=scope_excludes):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            snapshot[str(member_path)] = _digest_bytes(extracted.read())
    return snapshot


def _git_touched_files(cwd: Path, *, scope_excludes: Sequence[str] = ()) -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    touched_files: list[str] = []
    for line in completed.stdout.splitlines():
        if len(line) < 4:
            continue
        candidate = line[3:].strip()
        if not candidate:
            continue
        path = Path(candidate)
        if _is_excluded(path, scope_excludes=scope_excludes):
            continue
        touched_files.append(candidate)
    return sorted(set(touched_files))


def _digest_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _is_excluded(relative_path: Path, *, scope_excludes: Sequence[str] = ()) -> bool:
    if any(part in EXCLUDED_PATHS for part in relative_path.parts):
        return True

    path_text = relative_path.as_posix()
    return any(_matches_scope_rule(path_text, pattern) for pattern in scope_excludes)


def _matches_scope_rule(path_text: str, pattern: str) -> bool:
    normalized = pattern.strip().replace("\\", "/")
    if not normalized:
        return False
    if any(char in normalized for char in "*?["):
        return fnmatch(path_text, normalized)
    prefix = normalized.rstrip("/")
    return path_text == prefix or path_text.startswith(f"{prefix}/")