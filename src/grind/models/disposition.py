from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from grind.models.enums import DecidedBy, FindingStatus


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Disposition(BaseModel):
    disposition_id: str = Field(min_length=1)
    finding_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    iteration: int = Field(ge=1)
    decided_by: DecidedBy
    decision: FindingStatus
    justification: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=_utc_now)
