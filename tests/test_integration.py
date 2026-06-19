from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import duckdb
import yaml

from grind.cli import main
from grind.engine.orchestrator import MinimalOrchestrator
from grind.providers import ModelInvocationResult
from grind.state import bootstrap_state_store, open_state_store
from grind.state.quack import QuackConnectionError


def _patch_quack_route(monkeypatch, tmp_path: Path) -> None:
    remote_path = tmp_path / "remote.duckdb"
    monkeypatch.setenv("GRIND_DB_URI", "quack:localhost")
    monkeypatch.setenv("GRIND_DB_TOKEN", "test-token")
    def fake_ensure_local_quack_server(path, uri):
        bootstrap_state_store(remote_path, db_uri=str(remote_path))
        return "test-token"

    monkeypatch.setattr("grind.state.store.ensure_local_quack_server", fake_ensure_local_quack_server)
    monkeypatch.setattr(
        "grind.state.store.quack_connect",
        lambda uri, token: duckdb.connect(str(remote_path)),
    )


def _patch_prompt_runner(monkeypatch) -> None:
    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd, timeout_seconds=300: ModelInvocationResult(
            command=["fake-planner"],
            stdout='{"plan":"ship it"}',
            stderr="",
            returncode=0,
        ),
    )


def _patch_probe_commands(monkeypatch, *, kilo_model: str) -> None:
    def fake_run(command, **kwargs):
        if command[:4] == ["gh", "auth", "status", "--json"]:
            return subprocess.CompletedProcess(command, 0, '{"hosts":[{"hostname":"github.com"}]}', "")
        if command[:2] == ["gh", "copilot"] and "--help" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "--prompt --model --output-format --allow-all-tools",
                "",
            )
        if command[:2] == ["gh", "copilot"] and "--prompt" in command:
            return subprocess.CompletedProcess(command, 0, "OK\n", "")
        if command[:3] == ["kilo", "auth", "list"]:
            return subprocess.CompletedProcess(command, 0, "default\n", "")
        if command[:2] == ["kilo", "models"]:
            return subprocess.CompletedProcess(command, 0, f"{kilo_model}\n", "")
        if command[:2] == ["kilo", "run"] and "--help" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "",
                "--model --agent --format --dir --variant --auto",
            )
        if command[:2] == ["kilo", "run"] and "--auto" in command:
            return subprocess.CompletedProcess(command, 0, '{"event":"step_finish"}\nOK\n', "")
        if command[:3] == ["kilo", "debug", "config"]:
            return subprocess.CompletedProcess(command, 0, "config ok\n", "")
        if command[:3] == ["kilo", "agent", "list"]:
            return subprocess.CompletedProcess(command, 0, "code\nask\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("grind.verification.probes.subprocess.run", fake_run)


def _assert_quack_run_smoke(tmp_path: Path, capsys) -> None:
    exit_code = main(["run", "quack smoke", "--cwd", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    with open_state_store(tmp_path / ".grind" / "state" / "grind.duckdb") as store:
        run = store.runs.get(payload["run_id"])

    assert run is not None
    assert run.run_id == payload["run_id"]


def test_github_cli_quack_smoke(tmp_path: Path, monkeypatch, capsys) -> None:
    _patch_quack_route(monkeypatch, tmp_path)
    _patch_prompt_runner(monkeypatch)
    _patch_probe_commands(monkeypatch, kilo_model="kilo/openai/test-model")

    assert main(["init", "--cwd", str(tmp_path)]) == 0
    capsys.readouterr()

    exit_code = main([
        "verify-backend",
        "--strict",
        "--backend",
        "github_cli",
        "--role",
        "planner",
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["overall_status"] == "passed"

    _assert_quack_run_smoke(tmp_path, capsys)


def test_kilo_cli_quack_smoke(tmp_path: Path, monkeypatch, capsys) -> None:
    _patch_quack_route(monkeypatch, tmp_path)
    _patch_prompt_runner(monkeypatch)
    kilo_model = "kilo/openai/test-model"
    _patch_probe_commands(monkeypatch, kilo_model=kilo_model)

    assert main(["init", "--cwd", str(tmp_path)]) == 0
    capsys.readouterr()

    exit_code = main([
        "verify-backend",
        "--strict",
        "--backend",
        "kilo_cli",
        "--model",
        kilo_model,
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["overall_status"] == "passed"

    _assert_quack_run_smoke(tmp_path, capsys)


def test_orchestrator_can_require_quack_via_config(tmp_path: Path) -> None:
    assert main(["init", "--cwd", str(tmp_path)]) == 0
    config_path = tmp_path / ".grind" / "engine.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("require_quack: false", "require_quack: true"),
        encoding="utf-8",
    )

    try:
        MinimalOrchestrator(cwd=tmp_path)
    except QuackConnectionError as error:
        assert "Quack is required by configuration" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected QuackConnectionError")


def test_verify_self_host_passes_for_quack_workspace(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "grind.cli._verify_self_host",
        lambda *, cwd, config_path, strict, smoke: {
            "overall_status": "passed",
            "quack": {"status": "passed", "reason": "mocked quack probe", "db_uri": "quack:localhost"},
            "backends": {"status": "passed", "roles": {}},
            "validation": {"status": "passed", "commands": []},
        },
    )

    exit_code = main([
        "verify-self-host",
        "--cwd",
        str(tmp_path),
        "--json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["overall_status"] == "passed"
    assert payload["quack"]["status"] == "passed"
    assert payload["backends"]["status"] == "passed"
    assert payload["validation"]["status"] == "passed"


def test_orchestrator_pins_local_quack_token_for_store_opens(tmp_path: Path, monkeypatch) -> None:
    assert main(["init", "--cwd", str(tmp_path)]) == 0
    config_path = tmp_path / ".grind" / "engine.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data["state"]["db_uri"] = "quack:localhost"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    captured_bootstrap_tokens: list[str | None] = []
    captured_open_tokens: list[str | None] = []

    monkeypatch.delenv("GRIND_DB_TOKEN", raising=False)
    monkeypatch.setattr("grind.engine.orchestrator.ensure_local_quack_server", lambda path, uri: "pinned-token")
    monkeypatch.setattr(
        "grind.engine.orchestrator.bootstrap_state_store",
        lambda database_path, **kwargs: captured_bootstrap_tokens.append(kwargs.get("quack_token")),
    )

    class _DummyStore:
        def __init__(self) -> None:
            self.connection = object()
            self.workers = SimpleNamespace(register=lambda worker: None)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr(
        "grind.engine.orchestrator.open_state_store",
        lambda database_path, **kwargs: captured_open_tokens.append(kwargs.get("quack_token")) or _DummyStore(),
    )

    orchestrator = MinimalOrchestrator(cwd=tmp_path)
    orchestrator._heartbeat_worker_once = lambda: None

    assert captured_bootstrap_tokens == ["pinned-token"]
    assert captured_open_tokens == ["pinned-token"]