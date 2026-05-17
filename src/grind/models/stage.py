from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from grind.models.enums import ModelRole, StageStatus


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Stage(BaseModel):
    stage_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    stage_name: str = Field(min_length=1)
    started_at: datetime = Field(default_factory=_utc_now)
    ended_at: datetime | None = None
    status: StageStatus = StageStatus.PENDING
    model_role: ModelRole | None = None
    model_name: str | None = None
    provider: str | None = None
    runtime_agent: str | None = None
    runtime_variant: str | None = None
    prompt_artifact_id: str | None = None
    response_artifact_id: str | None = None
    output_artifact_id: str | None = None
    summary: str | None = None
    iteration: int = Field(default=1, ge=1)
    latency_ms: int | None = None
