from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ValidationCommandSpec(BaseModel):
    command: str = Field(min_length=1)
    argv: list[str] = Field(default_factory=list)
    risk: Literal["safe", "elevated", "risky"] = "safe"
    timeout_seconds: int = Field(default=120, ge=1)


class ScopeRules(BaseModel):
    exclude: list[str] = Field(default_factory=list)


class PolicyPack(BaseModel):
    path: Path
    schema_ver: str = Field(min_length=1)
    validation_commands: list[ValidationCommandSpec]
    safe_paths: list[str] = Field(default_factory=list)
    scope_rules: ScopeRules = Field(default_factory=ScopeRules)
    forbidden_commands: list[str] = Field(default_factory=list)

    @property
    def directory(self) -> Path:
        return self.path.parent