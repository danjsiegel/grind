from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ValidationRecord(BaseModel):
    validation_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    command: str = Field(min_length=1)
    status: str = Field(min_length=1)
    required: bool = True
    exit_code: int | None = None
    stdout_artifact_id: str | None = None
    stderr_artifact_id: str | None = None
    summary: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None