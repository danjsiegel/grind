from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AdjudicationPanelRecord(BaseModel):
    panel_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    iteration: int = Field(ge=1)
    mode: str = Field(min_length=1)
    primary_reason: str | None = None
    status: str = Field(min_length=1)
    disagreement_artifact_id: str | None = None
    summary: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None


class AdjudicationVoteRecord(BaseModel):
    vote_id: str = Field(min_length=1)
    panel_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    member_label: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    runtime_agent: str | None = None
    runtime_variant: str | None = None
    response_artifact_id: str | None = None
    output_artifact_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)