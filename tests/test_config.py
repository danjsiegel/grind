from __future__ import annotations

from pathlib import Path

from grind.config import (
    DEFAULT_GITHUB_MODEL,
    DEFAULT_KILO_MODEL,
    default_engine_config_path,
    init_engine_workspace,
    load_engine_config,
)


def test_default_engine_config_path_uses_repo_root(tmp_path: Path) -> None:
    path = default_engine_config_path(tmp_path)

    assert path == tmp_path / ".grind" / "engine.yaml"


def test_load_engine_config_resolves_default_storage_locations(tmp_path: Path) -> None:
    config_path = init_engine_workspace(tmp_path)
    config = load_engine_config(config_path)

    assert config.state.kind == "duckdb"
    assert config.state_path(tmp_path) == tmp_path / ".grind" / "state" / "grind.duckdb"
    assert config.state_db_uri() is None
    assert config.artifacts_root(tmp_path) == tmp_path / ".grind" / "artifacts"
    assert config.retrieval_path(tmp_path) == tmp_path / ".grind" / "state" / "lancedb"
    assert config.retention_export_root(tmp_path) == tmp_path / ".grind" / "archive"


def test_load_engine_config_reads_model_profiles(tmp_path: Path) -> None:
    config_path = init_engine_workspace(tmp_path)
    config = load_engine_config(config_path)

    assert config.retrieval.enabled is True
    assert config.retrieval.embedding_provider == "openai"
    assert config.retrieval.embedding_model == "text-embedding-3-large"
    assert config.retrieval.embedding_dimensions == 3072
    assert config.retrieval.workspace_docs_globs == ["README.md", "docs/**/*.md"]
    assert config.retrieval.workspace_spec_globs == [".local/specs/**/*.md"]
    assert config.models["planner"].provider == "github_cli"
    assert config.models["planner"].model == DEFAULT_GITHUB_MODEL
    assert config.models["implementer"].model == DEFAULT_KILO_MODEL
    assert config.models["implementer"].variant == "thinking"
    assert config.models["checker"].model == DEFAULT_KILO_MODEL
    assert config.models["checker"].agent == "ask"


def test_load_engine_config_defaults_when_file_missing(tmp_path: Path) -> None:
    config = load_engine_config(tmp_path / ".grind" / "engine.yaml")

    assert config.models["planner"].provider == "github_cli"
    assert config.validation.timeout_seconds == 120