from __future__ import annotations

from typing import Protocol

from grind.verification.models import ProbeResult, VerificationReport, VerificationRequest


class BackendProbe(Protocol):
    probe_id: str
    backend: str
    required_by_default: bool

    def applies(self, request: VerificationRequest) -> bool: ...
    def execute(self, request: VerificationRequest) -> ProbeResult: ...


class BackendVerifier(Protocol):
    def resolve_request(
        self,
        request: VerificationRequest,
        config: object | None = None,
    ) -> VerificationRequest: ...

    def probes_for(self, backend: str) -> list[BackendProbe]: ...
    def verify(self, request: VerificationRequest) -> VerificationReport: ...
