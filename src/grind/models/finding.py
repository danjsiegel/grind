from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from grind.models.enums import (
    EvidenceType,
    FindingCategory,
    FindingConfidence,
    FindingSeverity,
    FindingStatus,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FindingEvidence(BaseModel):
    evidence_id: str = Field(min_length=1)
    finding_id: str = Field(min_length=1)
    evidence_type: EvidenceType
    artifact_id: str | None = None
    snippet: str | None = Field(default=None, max_length=2000)
    source_ref: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)


class Finding(BaseModel):
    finding_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    stable_id: str = Field(min_length=16, max_length=16)
    title: str = Field(min_length=1)
    severity: FindingSeverity
    confidence: FindingConfidence
    category: FindingCategory
    rationale: str = Field(min_length=1)
    exact_fix_action: str = Field(min_length=1)
    status: FindingStatus = FindingStatus.OPEN
    first_seen_at: datetime = Field(default_factory=_utc_now)
    last_updated_at: datetime = Field(default_factory=_utc_now)
    adjudicated: bool = False
