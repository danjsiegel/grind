from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from grind.models.enums import OperatorStatus, RunState
from grind.models.enums import HoldType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Run(BaseModel):
    run_id: str = Field(min_length=1)
    repo_path: str = Field(min_length=1)
    policy_pack_path: str = Field(min_length=1)
    policy_schema_ver: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    state: RunState = RunState.CREATED
    requested_objective: str = Field(min_length=1)
    normalized_scope: dict[str, Any] | None = None
    operator_status: OperatorStatus = OperatorStatus.NONE
    current_worker_id: str | None = None
    current_hold_type: HoldType | None = None
    current_hold_reason: str | None = None
    current_hold_context: dict[str, Any] | None = None
    validation_commands_override: list[str] | None = None
    iteration_count: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=3, ge=1)
    budget_limit_usd: Decimal | None = None
    total_cost_usd: Decimal = Field(default=Decimal("0"))
