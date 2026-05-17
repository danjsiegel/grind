from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from grind.verification.models import (
    ProbeKind,
    ProbeResult,
    ProbeStatus,
    ResolvedIdentity,
    VerificationRequest,
)


@dataclass
class CommandProbe:
    probe_id: str
    backend: str
    kind: ProbeKind
    required_by_default: bool
    runner: callable

    def applies(self, request: VerificationRequest) -> bool:
        if self.kind == ProbeKind.SMOKE:
            return request.smoke
        if self.probe_id == "kilo_cli.agent_presence":
            return bool(request.agent)
        return True

    def execute(self, request: VerificationRequest) -> ProbeResult:
        return self.runner(request)


def github_cli_probe_pack() -> list[CommandProbe]:
    return [
        CommandProbe("github_cli.auth_status", "github_cli", ProbeKind.AUTH, True, github_auth_status_probe),
        CommandProbe("github_cli.cli_help", "github_cli", ProbeKind.AVAILABILITY, True, github_cli_help_probe),
        CommandProbe("github_cli.permission_flags", "github_cli", ProbeKind.PERMISSION, True, github_permission_flags_probe),
        CommandProbe("github_cli.model_smoke", "github_cli", ProbeKind.SMOKE, True, github_model_smoke_probe),
    ]


def kilo_cli_probe_pack() -> list[CommandProbe]:
    return [
        CommandProbe("kilo_cli.auth_list", "kilo_cli", ProbeKind.AUTH, True, kilo_auth_list_probe),
        CommandProbe("kilo_cli.model_catalog", "kilo_cli", ProbeKind.IDENTITY, True, kilo_model_catalog_probe),
        CommandProbe("kilo_cli.run_help", "kilo_cli", ProbeKind.AVAILABILITY, True, kilo_run_help_probe),
        CommandProbe("kilo_cli.agent_presence", "kilo_cli", ProbeKind.IDENTITY, False, kilo_agent_presence_probe),
        CommandProbe("kilo_cli.json_smoke", "kilo_cli", ProbeKind.SMOKE, True, kilo_json_smoke_probe),
        CommandProbe("kilo_cli.identity_source", "kilo_cli", ProbeKind.IDENTITY, True, kilo_identity_source_probe),
        CommandProbe("kilo_cli.permission_audit", "kilo_cli", ProbeKind.PERMISSION, True, kilo_permission_audit_probe),
    ]


def github_auth_status_probe(request: VerificationRequest) -> ProbeResult:
    completed = run_command(request, "github_cli", "github_cli.auth_status", ["gh", "auth", "status", "--json", "hosts"])
    status = ProbeStatus.PASSED if completed.returncode == 0 and "hosts" in completed.stdout else ProbeStatus.FAILED
    reason = "active host entry present" if status == ProbeStatus.PASSED else "gh auth status did not report an authenticated host"
    return build_probe_result(request, "github_cli.auth_status", ProbeKind.AUTH, status, reason, completed)


def github_cli_help_probe(request: VerificationRequest) -> ProbeResult:
    completed = run_command(request, "github_cli", "github_cli.cli_help", ["gh", "copilot", "--", "--help"])
    required_flags = ["--prompt", "--model", "--output-format"]
    status = ProbeStatus.PASSED if completed.returncode == 0 and all(flag in completed.stdout for flag in required_flags) else ProbeStatus.FAILED
    reason = "copilot CLI help exposes required automation flags" if status == ProbeStatus.PASSED else "copilot CLI help is missing required automation flags"
    return build_probe_result(request, "github_cli.cli_help", ProbeKind.AVAILABILITY, status, reason, completed)


def github_permission_flags_probe(request: VerificationRequest) -> ProbeResult:
    completed = run_command(request, "github_cli", "github_cli.permission_flags", ["gh", "copilot", "--", "--help"])
    has_permissions = "--allow-tool" in completed.stdout or "--allow-all-tools" in completed.stdout
    status = ProbeStatus.PASSED if completed.returncode == 0 and has_permissions else ProbeStatus.FAILED
    reason = "permission flags are present" if status == ProbeStatus.PASSED else "permission flags were not found in copilot CLI help"
    return build_probe_result(request, "github_cli.permission_flags", ProbeKind.PERMISSION, status, reason, completed)


def github_model_smoke_probe(request: VerificationRequest) -> ProbeResult:
    if not request.model:
        return ProbeResult(
            probe_id="github_cli.model_smoke",
            backend="github_cli",
            kind=ProbeKind.SMOKE,
            required=True,
            status=ProbeStatus.ERROR,
            status_reason="model is required for github_cli smoke verification",
        )
    completed = run_command(
        request,
        "github_cli",
        "github_cli.model_smoke",
        [
            "gh",
            "copilot",
            "--",
            "--prompt",
            "Reply with exactly OK",
            "--model",
            request.model,
            "--output-format",
            "json",
            "--allow-all-tools",
        ],
        timeout_seconds=60,
    )
    status = ProbeStatus.PASSED if completed.returncode == 0 and "OK" in completed.stdout else ProbeStatus.FAILED
    reason = "model-bound smoke probe returned OK" if status == ProbeStatus.PASSED else "model-bound smoke probe did not return OK"
    return build_probe_result(
        request,
        "github_cli.model_smoke",
        ProbeKind.SMOKE,
        status,
        reason,
        completed,
        observed_identity=ResolvedIdentity(provider="github_cli", model=request.model, agent=request.agent, variant=request.variant),
    )


def kilo_auth_list_probe(request: VerificationRequest) -> ProbeResult:
    completed = run_command(request, "kilo_cli", "kilo_cli.auth_list", ["kilo", "auth", "list"])
    status = ProbeStatus.PASSED if completed.returncode == 0 and completed.stdout.strip() else ProbeStatus.FAILED
    reason = "credential sources were listed" if status == ProbeStatus.PASSED else "kilo auth list returned no credential sources"
    return build_probe_result(request, "kilo_cli.auth_list", ProbeKind.AUTH, status, reason, completed)


def kilo_model_catalog_probe(request: VerificationRequest) -> ProbeResult:
    if not request.model:
        return ProbeResult(
            probe_id="kilo_cli.model_catalog",
            backend="kilo_cli",
            kind=ProbeKind.IDENTITY,
            required=True,
            status=ProbeStatus.ERROR,
            status_reason="model is required for kilo_cli model verification",
        )
    completed = run_command(request, "kilo_cli", "kilo_cli.model_catalog", ["kilo", "models"])
    lines = {line.strip() for line in completed.stdout.splitlines()}
    status = ProbeStatus.PASSED if completed.returncode == 0 and request.model in lines else ProbeStatus.FAILED
    reason = "configured model exists in kilo models output" if status == ProbeStatus.PASSED else "configured model was not found in kilo models output"
    return build_probe_result(
        request,
        "kilo_cli.model_catalog",
        ProbeKind.IDENTITY,
        status,
        reason,
        completed,
        observed_identity=ResolvedIdentity(provider="kilo_cli", model=request.model, agent=request.agent, variant=request.variant),
    )


def kilo_run_help_probe(request: VerificationRequest) -> ProbeResult:
    completed = run_command(request, "kilo_cli", "kilo_cli.run_help", ["kilo", "run", "--help"])
    required_flags = ["--model", "--agent", "--format", "--dir", "--variant", "--auto"]
    # kilo run --help writes to stderr, not stdout — check both streams
    output = completed.stdout + completed.stderr
    status = ProbeStatus.PASSED if completed.returncode == 0 and all(flag in output for flag in required_flags) else ProbeStatus.FAILED
    reason = "kilo run help exposes required runtime flags" if status == ProbeStatus.PASSED else "kilo run help is missing one or more required runtime flags"
    return build_probe_result(request, "kilo_cli.run_help", ProbeKind.AVAILABILITY, status, reason, completed)


def kilo_agent_presence_probe(request: VerificationRequest) -> ProbeResult:
    if not request.agent:
        return ProbeResult(
            probe_id="kilo_cli.agent_presence",
            backend="kilo_cli",
            kind=ProbeKind.IDENTITY,
            required=False,
            status=ProbeStatus.SKIPPED,
            status_reason="agent verification is not required when no agent is configured",
        )
    completed = run_command(request, "kilo_cli", "kilo_cli.agent_presence", ["kilo", "agent", "list"])
    status = ProbeStatus.PASSED if completed.returncode == 0 and request.agent in completed.stdout else ProbeStatus.FAILED
    reason = "configured agent is present in local agent list" if status == ProbeStatus.PASSED else "configured agent was not found in local agent list"
    return build_probe_result(
        request,
        "kilo_cli.agent_presence",
        ProbeKind.IDENTITY,
        status,
        reason,
        completed,
        required=False,
        observed_identity=ResolvedIdentity(provider="kilo_cli", model=request.model, agent=request.agent, variant=request.variant),
    )


def kilo_json_smoke_probe(request: VerificationRequest) -> ProbeResult:
    if not request.model:
        return ProbeResult(
            probe_id="kilo_cli.json_smoke",
            backend="kilo_cli",
            kind=ProbeKind.SMOKE,
            required=True,
            status=ProbeStatus.ERROR,
            status_reason="model is required for kilo_cli smoke verification",
        )
    command = ["kilo", "run", "--auto", "--format", "json", "--model", request.model]
    if request.agent:
        command.extend(["--agent", request.agent])
    command.append("Reply with exactly OK")
    completed = run_command(request, "kilo_cli", "kilo_cli.json_smoke", command, timeout_seconds=60)
    status = ProbeStatus.PASSED if completed.returncode == 0 and "OK" in completed.stdout and "step_finish" in completed.stdout else ProbeStatus.FAILED
    reason = "JSON smoke probe produced expected event stream and OK output" if status == ProbeStatus.PASSED else "JSON smoke probe did not produce the expected event stream"
    return build_probe_result(
        request,
        "kilo_cli.json_smoke",
        ProbeKind.SMOKE,
        status,
        reason,
        completed,
        observed_identity=ResolvedIdentity(provider="kilo_cli", model=request.model, agent=request.agent, variant=request.variant),
    )


def kilo_identity_source_probe(request: VerificationRequest) -> ProbeResult:
    if not request.model:
        return ProbeResult(
            probe_id="kilo_cli.identity_source",
            backend="kilo_cli",
            kind=ProbeKind.IDENTITY,
            required=True,
            status=ProbeStatus.ERROR,
            status_reason="model launch binding is required to establish identity source",
        )
    completed = run_command(request, "kilo_cli", "kilo_cli.identity_source", ["kilo", "debug", "config"])
    status = ProbeStatus.PASSED if completed.returncode == 0 else ProbeStatus.ERROR
    reason = "launch args remain the authoritative identity source for Kilo verification" if status == ProbeStatus.PASSED else "kilo debug config could not be inspected"
    return build_probe_result(
        request,
        "kilo_cli.identity_source",
        ProbeKind.IDENTITY,
        status,
        reason,
        completed,
        observed_identity=ResolvedIdentity(provider="kilo_cli", model=request.model, agent=request.agent, variant=request.variant),
    )


def kilo_permission_audit_probe(request: VerificationRequest) -> ProbeResult:
    completed = run_command(request, "kilo_cli", "kilo_cli.permission_audit", ["kilo", "agent", "list"])
    if request.role in {"planner", "checker", "adjudicator"}:
        if not request.agent:
            return build_probe_result(
                request,
                "kilo_cli.permission_audit",
                ProbeKind.PERMISSION,
                ProbeStatus.ERROR,
                "agent is required for read-only permission auditing",
                completed,
            )
        if request.agent in completed.stdout and "* -> allow" not in completed.stdout:
            return build_probe_result(
                request,
                "kilo_cli.permission_audit",
                ProbeKind.PERMISSION,
                ProbeStatus.PASSED,
                "read-only role agent does not expose broad wildcard permissions",
                completed,
            )
        return build_probe_result(
            request,
            "kilo_cli.permission_audit",
            ProbeKind.PERMISSION,
            ProbeStatus.INCONCLUSIVE if False else ProbeStatus.ERROR,
            "could not deterministically prove a read-only permission ceiling from kilo agent list output",
            completed,
        )
    return build_probe_result(
        request,
        "kilo_cli.permission_audit",
        ProbeKind.PERMISSION,
        ProbeStatus.PASSED,
        "write-capable role does not require a read-only permission ceiling audit",
        completed,
    )


def run_command(
    request: VerificationRequest,
    backend: str,
    probe_id: str,
    command: list[str],
    *,
    timeout_seconds: int = 15,
) -> subprocess.CompletedProcess[str]:
    cwd = str(request.cwd) if request.cwd else None
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        artifact_path = write_evidence_file(request, backend, probe_id, command, "", str(error), error=True)
        return subprocess.CompletedProcess(command, 1, "", f"{error}\nartifact_ref={artifact_path}")

    write_evidence_file(request, backend, probe_id, command, completed.stdout, completed.stderr)
    return completed


def build_probe_result(
    request: VerificationRequest,
    probe_id: str,
    kind: ProbeKind,
    status: ProbeStatus,
    reason: str,
    completed: subprocess.CompletedProcess[str],
    *,
    required: bool = True,
    observed_identity: ResolvedIdentity | None = None,
) -> ProbeResult:
    artifact_ref = evidence_ref(request, probe_id)
    return ProbeResult(
        probe_id=probe_id,
        backend=request.backend,
        kind=kind,
        required=required,
        status=status,
        status_reason=reason,
        command=" ".join(str(part) for part in completed.args) if completed.args else None,
        artifact_refs=[artifact_ref],
        observed_identity=observed_identity,
    )


def evidence_root(request: VerificationRequest) -> Path:
    base_dir = request.cwd or Path.cwd()
    return base_dir / ".grind" / "verify"


def evidence_ref(request: VerificationRequest, probe_id: str) -> str:
    return str(evidence_root(request) / f"{probe_id}.json")


def write_evidence_file(
    request: VerificationRequest,
    backend: str,
    probe_id: str,
    command: list[str],
    stdout: str,
    stderr: str,
    *,
    error: bool = False,
) -> Path:
    artifact_path = evidence_root(request) / f"{probe_id}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend,
        "probe_id": probe_id,
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
    }
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path