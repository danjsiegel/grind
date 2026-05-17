from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from grind.models.enums import OperatorActionType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OperatorActionRecord(BaseModel):
    action_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    action_type: OperatorActionType
    note: str | None = None
    checkpoint_id: str | None = None
    payload: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=_utc_now)