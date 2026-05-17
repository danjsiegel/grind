from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from grind.models.enums import CaptureMode, CheckpointKind, CheckpointStatus


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkspaceCheckpoint(BaseModel):
    checkpoint_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    stage_id: str | None = None
    iteration: int = Field(default=0, ge=0)
    checkpoint_kind: CheckpointKind
    capture_mode: CaptureMode
    scope_paths: list[str] = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    status: CheckpointStatus = CheckpointStatus.AVAILABLE
    created_by: Literal["engine"] = "engine"
    created_at: datetime = Field(default_factory=_utc_now)
    restored_at: datetime | None = None
