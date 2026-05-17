from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SemanticAuditRecord(BaseModel):
    semantic_audit_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    iteration: int = Field(ge=1)
    capability_level: str = Field(min_length=1)
    hard_fail: bool = False
    blocking_findings: list[dict[str, Any]] = Field(default_factory=list)
    advisory_findings: list[dict[str, Any]] = Field(default_factory=list)
    unsupported_checks: list[str] = Field(default_factory=list)
    report_artifact_id: str = Field(min_length=1)
    difference_surface_artifact_id: str = Field(min_length=1)
    summary: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)