from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ProbeKind(StrEnum):
    AVAILABILITY = "availability"
    AUTH = "auth"
    IDENTITY = "identity"
    PERMISSION = "permission"
    OUTPUT = "output"
    SMOKE = "smoke"


class ProbeStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class VerificationOverallStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class ResolvedIdentity(BaseModel):
    provider: str
    model: str | None = None
    agent: str | None = None
    variant: str | None = None


class ProbeResult(BaseModel):
    probe_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    kind: ProbeKind
    required: bool = True
    status: ProbeStatus
    status_reason: str = Field(min_length=1)
    command: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    observed_identity: ResolvedIdentity | None = None

    @model_validator(mode="after")
    def require_evidence_for_execution(self) -> "ProbeResult":
        if self.command and not self.artifact_refs:
            raise ValueError("executed probes must include at least one evidence ref")
        return self


class VerificationRequest(BaseModel):
    backend: str = Field(min_length=1)
    role: str | None = None
    model: str | None = None
    agent: str | None = None
    variant: str | None = None
    cwd: Path | None = None
    config_path: Path | None = None
    smoke: bool = True
    strict: bool = False


class VerificationReport(BaseModel):
    backend: str = Field(min_length=1)
    role: str | None = None
    resolved_identity: ResolvedIdentity
    overall_status: VerificationOverallStatus
    probes: list[ProbeResult] = Field(default_factory=list)
    maker_owned_prerequisites: list[str] = Field(default_factory=list)

    @classmethod
    def from_probe_results(
        cls,
        *,
        backend: str,
        resolved_identity: ResolvedIdentity,
        probes: list[ProbeResult],
        role: str | None = None,
        maker_owned_prerequisites: list[str] | None = None,
    ) -> "VerificationReport":
        overall_status = summarize_probe_results(probes)
        return cls(
            backend=backend,
            role=role,
            resolved_identity=resolved_identity,
            overall_status=overall_status,
            probes=probes,
            maker_owned_prerequisites=maker_owned_prerequisites or [],
        )


def summarize_probe_results(probes: list[ProbeResult]) -> VerificationOverallStatus:
    required_probes = [probe for probe in probes if probe.required]
    if any(probe.status == ProbeStatus.FAILED for probe in required_probes):
        return VerificationOverallStatus.FAILED
    if any(probe.status == ProbeStatus.ERROR for probe in required_probes):
        return VerificationOverallStatus.INCONCLUSIVE
    if any(probe.status == ProbeStatus.SKIPPED for probe in required_probes):
        return VerificationOverallStatus.INCONCLUSIVE
    return VerificationOverallStatus.PASSED
