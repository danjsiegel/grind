from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from grind.models.enums import RunState


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TransitionRecord(BaseModel):
    transition_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    from_state: RunState
    to_state: RunState
    reason: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=_utc_now)