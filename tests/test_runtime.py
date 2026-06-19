from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

from grind.config import ModelProfileConfig
from grind.providers.runtime import extract_json_output, extract_text_output, invoke_text_prompt


class _CompletedProcess:
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_invoke_text_prompt_extracts_github_cli_usage(monkeypatch, tmp_path: Path) -> None:
    profile = ModelProfileConfig(provider="github_cli", model="gpt-5.4")

    monkeypatch.setattr(
        "grind.providers.runtime._run_prompt_command",
        lambda **kwargs: (
            '{"type":"turn_completed","usage":{"input_tokens":321,"output_tokens":123,"cost_usd":"0.42"},"model":"gpt-5.4"}',
            "",
            0,
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
        "grind.providers.runtime._run_prompt_command",
        lambda **kwargs: (
            '{"event":"run_finished","summary":{"prompt_tokens":210,"completion_tokens":34,"total_cost_usd":"0.11"},"resolved_model":"qwen-3.6-plus"}',
            "",
            0,
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


def test_extract_text_output_prefers_embedded_json_payload() -> None:
    stdout = """
Planner transcript noise.

```json
{"plan":"inspect live code first\\nrun focused tests second"}
```
"""

    assert extract_text_output(stdout) == "inspect live code first\nrun focused tests second"


def test_extract_json_output_reads_embedded_json_payload() -> None:
    stdout = """
Noise before the final answer.

```json
{"plan":"inspect live code first"}
```
"""

    assert extract_json_output(stdout) == {"plan": "inspect live code first"}


def test_extract_text_output_prefers_embedded_json_payload_with_nested_fences() -> None:
    plan = """## Phase 2 Implementation Plan

### Step 1
Run focused validation.

```bash
uv run pytest tests/test_runtime.py -q
```

### Step 2
Stop on the first failure.
"""
    stdout = f"""
Planner transcript noise before the final answer.

```json
{json.dumps({'plan': plan})}
```

Trailing transcript that should not be treated as the plan.
"""

    assert extract_text_output(stdout).strip() == plan.strip()


def test_extract_text_output_prefers_final_event_stream_plan_over_tool_transcript() -> None:
    event_stream = "\n".join(
        [
            json.dumps({
                "type": "tool_use",
                "part": {
                    "type": "tool",
                    "tool": "read",
                    "state": {"output": "1: def noisy():\\n2: pass"},
                },
            }),
            json.dumps({
                "type": "text",
                "part": {
                    "type": "text",
                    "text": (
                        "Now I have a complete picture.\\n\\n"
                        '{"plan":"## Real Plan\\n\\n1. Run focused tests.\\n2. Stop on first failure."}'
                    ),
                },
            }),
        ]
    )

    assert extract_text_output(event_stream) == "## Real Plan\n\n1. Run focused tests.\n2. Stop on first failure."


def test_extract_text_output_scans_mixed_output_for_late_plan_event() -> None:
    mixed_output = "\n".join(
        [
            '"):',
            '167:         if marker in cleaned:',
            json.dumps({
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "intermediate noisy text without a plan",
                },
            }),
            json.dumps({
                "type": "text",
                "part": {
                    "type": "text",
                    "text": (
                        "Now I have a complete picture.\\n\\n"
                        '{"plan":"## Final Plan\\n\\n1. Inspect live code.\\n2. Run focused validation."}'
                    ),
                },
            }),
        ]
    )

    assert extract_text_output(mixed_output) == "## Final Plan\n\n1. Inspect live code.\n2. Run focused validation."


def test_invoke_text_prompt_wraps_timeout_as_model_error(monkeypatch, tmp_path: Path) -> None:
    profile = ModelProfileConfig(provider="kilo_cli", model="qwen-3.6-plus", agent="code")

    monkeypatch.setattr(
        "grind.providers.runtime._run_prompt_command",
        lambda **kwargs: (_ for _ in ()).throw(
            __import__("grind.providers.runtime", fromlist=["_PromptTimeoutError"])._PromptTimeoutError(
                stdout="partial stdout",
                stderr="partial stderr",
            )
        ),
    )

    try:
        invoke_text_prompt(profile, prompt="test", cwd=tmp_path, timeout_seconds=7)
    except Exception as error:
        assert str(error) == "model invocation timed out after 7 seconds"
        assert getattr(error, "result").stdout == "partial stdout"
        assert getattr(error, "result").stderr == "partial stderr"
        assert getattr(error, "result").returncode == 124
    else:  # pragma: no cover - defensive
        raise AssertionError("expected timeout to raise ModelInvocationError")