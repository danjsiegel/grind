from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from grind.models.enums import TaskSourceKind, TaskStatus


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Task(BaseModel):
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    source_kind: TaskSourceKind
    raw_input: str = Field(min_length=1)
    normalized_scope: dict[str, Any] | None = None
    phase_label: str | None = None
    acceptance_checks: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
