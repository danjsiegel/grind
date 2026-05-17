from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from grind.models.enums import ModelRole


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ModelCallRecord(BaseModel):
    model_call_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    model_role: ModelRole
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    runtime_agent: str | None = None
    runtime_variant: str | None = None
    command: list[str] = Field(default_factory=list)
    status: str = Field(min_length=1)
    started_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    metadata: dict[str, Any] | None = None
    error_reason: str | None = None