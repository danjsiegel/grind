from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunLease(BaseModel):
    lease_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    acquired_at: datetime = Field(default_factory=_utc_now)
    released_at: datetime | None = None
    status: Literal["active", "released", "expired"] = "active"