from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from grind.models.enums import (
    EnforcementMode,
    InvariantKind,
    InvariantSourceKind,
    InvariantStatus,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InvariantContract(BaseModel):
    invariant_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    scope_ref: str = Field(min_length=1)
    invariant_kind: InvariantKind
    statement: str = Field(min_length=1)
    source_kind: InvariantSourceKind
    source_artifact_id: str = Field(min_length=1)
    enforcement_mode: EnforcementMode
    status: InvariantStatus = InvariantStatus.ACTIVE
    created_at: datetime = Field(default_factory=_utc_now)
    retired_at: datetime | None = None
