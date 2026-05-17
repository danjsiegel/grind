from __future__ import annotations

from pathlib import Path

import yaml

from grind.policy.models import PolicyPack, ScopeRules, ValidationCommandSpec
from grind.validation.safety import normalize_shell_free_command


class PolicySchemaError(ValueError):
    pass


class PolicyLoader:
    def __init__(self, policy_pack_path: Path | str | None = None):
        self._policy_pack_path = Path(policy_pack_path) if policy_pack_path is not None else None

    def load(self, policy_pack_path: Path | str | None = None) -> PolicyPack:
        if isinstance(self, PolicyLoader):
            resolved_path = policy_pack_path or self._policy_pack_path
        else:
            resolved_path = self if policy_pack_path is None else policy_pack_path

        if resolved_path is None:
            raise TypeError("policy_pack_path is required")

        return PolicyLoader._load_from_path(Path(resolved_path))

    @staticmethod
    def _load_from_path(policy_pack_path: Path) -> PolicyPack:
        path = policy_pack_path / "project.yaml" if policy_pack_path.is_dir() else policy_pack_path
        if not path.exists():
            raise FileNotFoundError(f"policy pack not found: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        schema_ver = raw.get("schema_ver")
        if schema_ver != "1":
            raise PolicySchemaError(f"unsupported policy schema version: {schema_ver!r}")
        if "validation_commands" not in raw or "safe_paths" not in raw:
            raise PolicySchemaError("policy pack requires schema_ver, validation_commands, and safe_paths")

        validation_commands: list[ValidationCommandSpec] = []
        for item in raw["validation_commands"]:
            command = item.get("command")
            if not command:
                raise PolicySchemaError("policy validation command entries must include command")
            validation_commands.append(
                ValidationCommandSpec(
                    command=command,
                    argv=normalize_shell_free_command(command),
                    risk=item.get("risk", "safe"),
                    timeout_seconds=item.get("timeout_seconds", 120),
                )
            )

        return PolicyPack(
            path=path,
            schema_ver=schema_ver,
            validation_commands=validation_commands,
            safe_paths=list(raw["safe_paths"]),
            scope_rules=ScopeRules.model_validate(raw.get("scope_rules") or {}),
            forbidden_commands=list(raw.get("forbidden_commands") or []),
        )