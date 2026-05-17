from pathlib import Path

import pytest
from pydantic import ValidationError

from grind.verification.models import (
    ProbeKind,
    ProbeResult,
    ProbeStatus,
    ResolvedIdentity,
    VerificationOverallStatus,
    VerificationReport,
    VerificationRequest,
    summarize_probe_results,
)


def test_probe_result_requires_evidence_for_executed_probe() -> None:
    with pytest.raises(ValidationError):
        ProbeResult(
            probe_id="github_cli.auth_status",
            backend="github_cli",
            kind=ProbeKind.AUTH,
            status=ProbeStatus.PASSED,
            status_reason="active host entry present",
            command="gh auth status --json hosts",
        )


def test_report_is_failed_when_required_probe_fails() -> None:
    probes = [
        ProbeResult(
            probe_id="github_cli.auth_status",
            backend="github_cli",
            kind=ProbeKind.AUTH,
            required=True,
            status=ProbeStatus.FAILED,
            status_reason="not authenticated",
            command="gh auth status --json hosts",
            artifact_refs=["verify/github_cli/auth_status.json"],
        )
    ]

    report = VerificationReport.from_probe_results(
        backend="github_cli",
        role="planner",
        resolved_identity=ResolvedIdentity(provider="github_cli", model="gpt-5.4"),
        probes=probes,
    )

    assert report.overall_status == VerificationOverallStatus.FAILED


def test_report_is_inconclusive_when_required_probe_errors() -> None:
    probes = [
        ProbeResult(
            probe_id="kilo_cli.identity_source",
            backend="kilo_cli",
            kind=ProbeKind.IDENTITY,
            required=True,
            status=ProbeStatus.ERROR,
            status_reason="config inspection failed",
        )
    ]

    assert summarize_probe_results(probes) == VerificationOverallStatus.INCONCLUSIVE


def test_report_is_inconclusive_when_required_probe_is_skipped() -> None:
    probes = [
        ProbeResult(
            probe_id="github_cli.model_smoke",
            backend="github_cli",
            kind=ProbeKind.SMOKE,
            required=True,
            status=ProbeStatus.SKIPPED,
            status_reason="smoke disabled",
        )
    ]

    assert summarize_probe_results(probes) == VerificationOverallStatus.INCONCLUSIVE


def test_request_accepts_paths() -> None:
    request = VerificationRequest(
        backend="kilo_cli",
        role="planner",
        cwd=Path("."),
        config_path=Path(".grind/engine.yaml"),
    )

    assert request.cwd == Path(".")
    assert request.config_path == Path(".grind/engine.yaml")
