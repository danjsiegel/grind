from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DifferenceSurface(BaseModel):
    """Formally constructed checking substrate built by the engine before checker invocation.

    Not a raw diff. Contextualizes changes against task scope, evidence, risk,
    findings history, validation state, semantic audit results, invariant contracts,
    and any effective policy overrides for this iteration (§6.7).
    """

    surface_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    iteration: int = Field(ge=1)

    # Tier 1 — always included: what the task explicitly required
    requested_delta: dict[str, Any] = Field(default_factory=dict)

    # Engine-authoritative touched files, symbols, commands executed, artifact IDs
    # (model self-report is advisory only; this is derived from filesystem + audit logs)
    observed_delta: dict[str, Any] = Field(default_factory=dict)

    # Which claims have supporting evidence vs. asserted without proof
    evidence_delta: dict[str, Any] = Field(default_factory=dict)

    # Scope expansions, missing validations, unproven assumptions, risky commands
    risk_delta: dict[str, Any] = Field(default_factory=dict)

    # Per-finding status changes: new, fixed, rejected, deferred, still-open
    findings_delta: dict[str, Any] = Field(default_factory=dict)

    # Planned / selected / run / skipped / failed / required validation results
    validation_delta: dict[str, Any] = Field(default_factory=dict)

    # Deterministic symbol graph / interface / dependency / invariant audit results
    semantic_audit_delta: dict[str, Any] = Field(default_factory=dict)

    # Active invariant contracts relevant to touched scope and lifecycle events
    invariant_delta: dict[str, Any] = Field(default_factory=dict)

    # Run-scoped policy overrides or operator approvals effective for this iteration
    # Prospective only; does not retroactively rewrite prior DifferenceSurface artifacts
    policy_delta: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=_utc_now)
