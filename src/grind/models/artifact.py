from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactRecord(BaseModel):
    artifact_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    path: str = Field(min_length=1)
    storage_kind: str = Field(default="local", min_length=1)
    checksum: str | None = None
    size_bytes: int | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] | None = None