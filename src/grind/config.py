from __future__ import annotations

from decimal import Decimal
import os
from pathlib import Path
from textwrap import dedent
from typing import Literal

import yaml
from pydantic import BaseModel, Field


DEFAULT_GITHUB_MODEL = "claude-sonnet-4.6"
DEFAULT_KILO_MODEL = "kilo/anthropic/claude-sonnet-4.6"


class StateConfig(BaseModel):
  kind: Literal["duckdb"] = "duckdb"
  path: str = ".grind/state/grind.duckdb"
  db_uri: str | None = None
  require_quack: bool = False


class ArtifactsConfig(BaseModel):
    root: str = ".grind/artifacts"


class RetrievalConfig(BaseModel):
    enabled: bool = True
    path: str = ".grind/state/lancedb"
    embedding_provider: Literal["openai"] = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_api_key_env: str = "OPENAI_API_KEY"
    embedding_dimensions: int = Field(default=3072, ge=64)
    allow_local_fallback: bool = True
    chunk_size: int = Field(default=1200, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    max_search_results: int = Field(default=5, ge=1)
    index_workspace_docs: bool = True
    index_workspace_specs: bool = True
    workspace_docs_globs: list[str] = Field(default_factory=lambda: ["README.md", "docs/**/*.md"])
    workspace_spec_globs: list[str] = Field(default_factory=lambda: [".local/specs/**/*.md"])


class RetentionConfig(BaseModel):
    mode: Literal["manual", "auto"] = "manual"
    export_root: str = ".grind/archive"
    keep_artifacts_days: int | None = None
    keep_last_terminal_runs: int | None = Field(default=None, ge=0)


class ValidationConfig(BaseModel):
  commands: list[str] = Field(default_factory=lambda: ["uv run pytest tests -q"])
  stop_on_failure: bool = True
  timeout_seconds: int = Field(default=120, ge=1)


class ExecutionConfig(BaseModel):
    max_iterations: int = Field(default=3, ge=1)
    budget_limit_usd: Decimal | None = None


class AdjudicationConfig(BaseModel):
    require_model_review_on_semantic_hard_fail: bool = False
    consensus_enabled: bool = False
    consensus_member_labels: list[str] = Field(
        default_factory=lambda: ["security_auditor", "senior_architect", "testing_specialist"]
    )


class ModelProfileConfig(BaseModel):
    provider: Literal["github_cli", "kilo_cli"]
    model: str = Field(min_length=1)
    agent: str | None = None
    variant: str | None = None


def default_model_profiles() -> dict[str, ModelProfileConfig]:
    return {
    "planner": ModelProfileConfig(provider="github_cli", model=DEFAULT_GITHUB_MODEL),
        "implementer": ModelProfileConfig(
            provider="kilo_cli",
      model=DEFAULT_KILO_MODEL,
            agent="code",
            variant="thinking",
        ),
        "checker": ModelProfileConfig(
            provider="kilo_cli",
      model=DEFAULT_KILO_MODEL,
            agent="ask",
            variant="instant",
        ),
    "adjudicator": ModelProfileConfig(provider="github_cli", model=DEFAULT_GITHUB_MODEL),
    }


class EngineConfig(BaseModel):
    state: StateConfig = Field(default_factory=StateConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    adjudication: AdjudicationConfig = Field(default_factory=AdjudicationConfig)
    models: dict[str, ModelProfileConfig] = Field(default_factory=default_model_profiles)

    def state_db_uri(self) -> str | None:
        return os.getenv("GRIND_DB_URI") or self.state.db_uri

    def state_path(self, cwd: Path) -> Path:
        return resolve_path(self.state.path, cwd=cwd)

    def artifacts_root(self, cwd: Path) -> Path:
        return resolve_path(self.artifacts.root, cwd=cwd)

    def retrieval_path(self, cwd: Path) -> Path:
        return resolve_path(self.retrieval.path, cwd=cwd)

    def retention_export_root(self, cwd: Path) -> Path:
        return resolve_path(self.retention.export_root, cwd=cwd)


def default_engine_config_path(cwd: Path | None = None) -> Path:
    base_dir = cwd or Path.cwd()
    return base_dir / ".grind" / "engine.yaml"


def resolve_path(value: str, *, cwd: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return cwd / path


def load_engine_config(path: Path) -> EngineConfig:
    if not path.exists():
        return EngineConfig()
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    return EngineConfig.model_validate(raw_config)


def render_default_engine_config() -> str:
    return dedent(
        """\
        # Grind engine configuration
        #
        # Storage
        # - state.path is the canonical DuckDB ledger location.
        # - artifacts.root holds prompts, responses, validation output, checkpoints,
        #   and other append-only run artifacts.
        #
        # Retention
        # - mode=manual means Grind does not auto-delete ledger data; use
        #   `grind prune` for explicit cleanup of old terminal runs.
        # - mode=auto prunes terminal runs after state-changing commands when
        #   keep_last_terminal_runs is set.
        # - export_root is reserved for future archive/export flows.
        # - keep_artifacts_days is still informational until policy-driven
        #   retention lands.
        #
        # Backend examples
        # - github_cli model values should match what your installed GitHub CLI
        #   environment exposes.
        # - kilo_cli model and agent values should come from `kilo models` and
        #   `kilo agent list` in your own environment.
        state:
          kind: duckdb
          path: .grind/state/grind.duckdb
          db_uri:
          require_quack: false

        artifacts:
          root: .grind/artifacts

        retrieval:
          enabled: true
          path: .grind/state/lancedb
          embedding_provider: openai
          embedding_model: text-embedding-3-large
          embedding_api_base: https://api.openai.com/v1
          embedding_api_key_env: OPENAI_API_KEY
          embedding_dimensions: 3072
          allow_local_fallback: true
          chunk_size: 1200
          chunk_overlap: 150
          max_search_results: 5
          index_workspace_docs: true
          index_workspace_specs: true
          workspace_docs_globs:
            - README.md
            - docs/**/*.md
          workspace_spec_globs:
            - .local/specs/**/*.md

        retention:
          mode: manual
          export_root: .grind/archive
          keep_artifacts_days:
          keep_last_terminal_runs:

        validation:
          commands:
            - uv run pytest tests -q
          stop_on_failure: true

        execution:
          max_iterations: 3
          budget_limit_usd:

        adjudication:
          require_model_review_on_semantic_hard_fail: false
          consensus_enabled: false
          consensus_member_labels:
            - security_auditor
            - senior_architect
            - testing_specialist

        models:
          planner:
            provider: github_cli
            model: {github_model}

          implementer:
            provider: kilo_cli
            model: {kilo_model}
            agent: code
            variant: thinking

          checker:
            provider: kilo_cli
            model: {kilo_model}
            agent: ask
            variant: instant

          adjudicator:
            provider: github_cli
            model: {github_model}
        """
    ).format(
        github_model=DEFAULT_GITHUB_MODEL,
        kilo_model=DEFAULT_KILO_MODEL,
    )


def init_engine_workspace(cwd: Path, *, force: bool = False) -> Path:
    config_path = default_engine_config_path(cwd)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = EngineConfig()
    config.state_path(cwd).parent.mkdir(parents=True, exist_ok=True)
    config.artifacts_root(cwd).mkdir(parents=True, exist_ok=True)
    config.retrieval_path(cwd).mkdir(parents=True, exist_ok=True)
    config.retention_export_root(cwd).mkdir(parents=True, exist_ok=True)
    policy_dir = cwd / ".grind" / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists() and not force:
        raise FileExistsError(f"configuration already exists: {config_path}")

    config_path.write_text(render_default_engine_config(), encoding="utf-8")
    policy_path = policy_dir / "project.yaml"
    if not policy_path.exists():
        template_path = Path(__file__).resolve().parents[2] / "templates" / "policy" / "project.yaml"
        policy_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path
