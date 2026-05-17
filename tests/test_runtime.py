from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from grind.config import ModelProfileConfig
from grind.providers.runtime import invoke_text_prompt


class _CompletedProcess:
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_invoke_text_prompt_extracts_github_cli_usage(monkeypatch, tmp_path: Path) -> None:
    profile = ModelProfileConfig(provider="github_cli", model="gpt-5.4")

    monkeypatch.setattr(
        "grind.providers.runtime.subprocess.run",
        lambda *args, **kwargs: _CompletedProcess(
            stdout='{"type":"turn_completed","usage":{"input_tokens":321,"output_tokens":123,"cost_usd":"0.42"},"model":"gpt-5.4"}',
        ),
    )

    result = invoke_text_prompt(profile, prompt="test", cwd=tmp_path)

    assert result.input_tokens == 321
    assert result.output_tokens == 123
    assert result.estimated_cost_usd == Decimal("0.42")
    assert result.provider_metadata == {
        "estimated_cost_usd": Decimal("0.42"),
        "input_tokens": 321,
        "output_tokens": 123,
        "reported_model": "gpt-5.4",
    }


def test_invoke_text_prompt_extracts_kilo_cli_usage(monkeypatch, tmp_path: Path) -> None:
    profile = ModelProfileConfig(provider="kilo_cli", model="qwen-3.6-plus", agent="code")

    monkeypatch.setattr(
        "grind.providers.runtime.subprocess.run",
        lambda *args, **kwargs: _CompletedProcess(
            stdout='{"event":"run_finished","summary":{"prompt_tokens":210,"completion_tokens":34,"total_cost_usd":"0.11"},"resolved_model":"qwen-3.6-plus"}',
        ),
    )

    result = invoke_text_prompt(profile, prompt="test", cwd=tmp_path)

    assert result.input_tokens == 210
    assert result.output_tokens == 34
    assert result.estimated_cost_usd == Decimal("0.11")
    assert result.provider_metadata == {
        "estimated_cost_usd": Decimal("0.11"),
        "input_tokens": 210,
        "output_tokens": 34,
        "reported_model": "qwen-3.6-plus",
    }