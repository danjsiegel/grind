from __future__ import annotations

from pathlib import Path

import yaml

from grind.config import default_engine_config_path
from grind.verification.models import (
    ProbeResult,
    ProbeStatus,
    ResolvedIdentity,
    VerificationReport,
    VerificationRequest,
)
from grind.verification.protocols import BackendProbe, BackendVerifier
from grind.verification.probes import github_cli_probe_pack, kilo_cli_probe_pack


class VerificationConfigError(ValueError):
    pass


class DefaultBackendVerifier(BackendVerifier):
    def resolve_request(
        self,
        request: VerificationRequest,
        config: object | None = None,
    ) -> VerificationRequest:
        if request.backend not in {"github_cli", "kilo_cli"}:
            raise VerificationConfigError(f"unsupported backend: {request.backend}")

        if request.model or not request.role:
            return request

        resolved_from_config = self._resolve_role_profile(request)
        return request.model_copy(update=resolved_from_config)

    def probes_for(self, backend: str) -> list[BackendProbe]:
        if backend == "github_cli":
            return github_cli_probe_pack()
        if backend == "kilo_cli":
            return kilo_cli_probe_pack()
        raise VerificationConfigError(f"unsupported backend: {backend}")

    def verify(self, request: VerificationRequest) -> VerificationReport:
        resolved_request = self.resolve_request(request)
        resolved_identity = ResolvedIdentity(
            provider=resolved_request.backend,
            model=resolved_request.model,
            agent=resolved_request.agent,
            variant=resolved_request.variant,
        )

        probe_results: list[ProbeResult] = []
        for probe in self.probes_for(resolved_request.backend):
            if probe.applies(resolved_request):
                result = probe.execute(resolved_request)
            else:
                result = ProbeResult(
                    probe_id=probe.probe_id,
                    backend=probe.backend,
                    kind=probe.kind,
                    required=probe.required_by_default,
                    status=ProbeStatus.SKIPPED,
                    status_reason="probe not applicable to this request",
                )

            if resolved_request.strict and result.status == ProbeStatus.SKIPPED:
                result = result.model_copy(
                    update={
                        "required": True,
                        "status": ProbeStatus.FAILED,
                        "status_reason": f"strict mode: {result.status_reason}",
                    }
                )
            probe_results.append(result)

        prerequisites = maker_owned_prerequisites(resolved_request.backend)
        return VerificationReport.from_probe_results(
            backend=resolved_request.backend,
            role=resolved_request.role,
            resolved_identity=resolved_identity,
            probes=probe_results,
            maker_owned_prerequisites=prerequisites,
        )

    def _resolve_role_profile(self, request: VerificationRequest) -> dict[str, str | None]:
        config_path = request.config_path or default_config_path(request.cwd)
        if not config_path.exists():
            raise VerificationConfigError(
                "model was not provided and no config file was found to resolve the role profile"
            )

        with config_path.open("r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}

        models = config_data.get("models") or {}
        role_config = models.get(request.role or "")
        if not isinstance(role_config, dict):
            raise VerificationConfigError(
                f"role {request.role!r} was not found in config {config_path}"
            )

        provider = role_config.get("provider")
        if provider and provider != request.backend:
            raise VerificationConfigError(
                f"role {request.role!r} resolves to provider {provider!r}, not backend {request.backend!r}"
            )

        return {
            "model": role_config.get("model"),
            "agent": request.agent or role_config.get("agent"),
            "variant": request.variant or role_config.get("variant"),
            "config_path": config_path,
        }


def default_config_path(cwd: Path | None) -> Path:
    return default_engine_config_path(cwd)


def maker_owned_prerequisites(backend: str) -> list[str]:
    if backend == "github_cli":
        return [
            "GitHub account, Copilot entitlement, and login state remain authoritative upstream",
        ]
    if backend == "kilo_cli":
        return [
            "Kilo auth configuration, model catalog semantics, and agent meanings remain authoritative upstream",
        ]
    return []