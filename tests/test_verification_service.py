from __future__ import annotations

from pathlib import Path

from grind.verification.models import ProbeKind, ProbeStatus, VerificationRequest
from grind.verification.service import DefaultBackendVerifier


class SkippedProbe:
    probe_id = "github_cli.optional_probe"
    backend = "github_cli"
    kind = ProbeKind.OUTPUT
    required_by_default = False

    def applies(self, request: VerificationRequest) -> bool:
        return False

    def execute(self, request: VerificationRequest):  # pragma: no cover
        raise AssertionError("skipped probe should not execute")


def test_resolve_request_from_engine_config(tmp_path: Path) -> None:
    config_dir = tmp_path / ".grind"
    config_dir.mkdir()
    config_path = config_dir / "engine.yaml"
    config_path.write_text(
        "models:\n"
        "  planner:\n"
        "    provider: github_cli\n"
        "    model: gpt-5.4\n"
        "    agent: null\n"
        "    variant: null\n",
        encoding="utf-8",
    )

    verifier = DefaultBackendVerifier()
    resolved = verifier.resolve_request(
        VerificationRequest(
            backend="github_cli",
            role="planner",
            cwd=tmp_path,
        )
    )

    assert resolved.model == "gpt-5.4"
    assert resolved.config_path == config_path


def test_strict_mode_upgrades_skipped_probe_to_failed() -> None:
    verifier = DefaultBackendVerifier()
    verifier.probes_for = lambda backend: [SkippedProbe()]  # type: ignore[method-assign]

    report = verifier.verify(
        VerificationRequest(
            backend="github_cli",
            model="gpt-5.4",
            strict=True,
        )
    )

    assert report.overall_status.value == "failed"
    assert report.probes[0].status == ProbeStatus.FAILED
    assert report.probes[0].status_reason.startswith("strict mode:")