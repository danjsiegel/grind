from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RetrievalQueueRecord(BaseModel):
    queue_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    queue_status: str = Field(default="pending", min_length=1)
    attempts: int = Field(default=0, ge=0)
    last_error: str | None = None
    queued_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None