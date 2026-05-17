from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Worker(BaseModel):
    worker_id: str = Field(min_length=1)
    hostname: str = Field(min_length=1)
    pid: int = Field(ge=1)
    registered_at: datetime = Field(default_factory=_utc_now)
    last_seen_at: datetime = Field(default_factory=_utc_now)