from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from time import perf_counter
import subprocess
from typing import Any

from grind.config import ModelProfileConfig


@dataclass(frozen=True)
class ModelInvocationResult:
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    estimated_cost_usd: Decimal | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    provider_metadata: dict[str, Any] | None = None


class ModelInvocationError(RuntimeError):
    def __init__(self, message: str, *, result: ModelInvocationResult | None = None):
        super().__init__(message)
        self.result = result


def invoke_text_prompt(
    profile: ModelProfileConfig,
    *,
    prompt: str,
    cwd: Path,
    timeout_seconds: int = 300,
) -> ModelInvocationResult:
    command = _build_command(profile, prompt=prompt)
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    latency_ms = int((perf_counter() - started) * 1000)
    provider_metadata = _extract_provider_metadata(profile.provider, completed.stdout)
    result = ModelInvocationResult(
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        estimated_cost_usd=_coerce_decimal(provider_metadata.get("estimated_cost_usd")),
        input_tokens=_coerce_int(provider_metadata.get("input_tokens")),
        output_tokens=_coerce_int(provider_metadata.get("output_tokens")),
        latency_ms=latency_ms,
        provider_metadata=provider_metadata,
    )
    if completed.returncode != 0:
        raise ModelInvocationError(
            completed.stderr or completed.stdout or "model invocation failed",
            result=result,
        )
    return result


def extract_text_output(stdout: str) -> str:
    stripped = stdout.strip()
    if not stripped:
        return ""

    parsed = _parse_direct_json_maybe(stripped)
    if parsed is None:
        embedded = _extract_embedded_json_candidate(stripped)
        if embedded is not None:
            parsed = _parse_direct_json_maybe(embedded)
    if parsed is None:
        parsed = _parse_json_maybe(stripped)
    if parsed is None:
        return stripped

    extracted = _extract_text_values(parsed)
    if extracted:
        return "\n".join(part for part in extracted if part.strip()).strip()
    return stripped


def extract_json_output(stdout: str) -> Any:
    payload = extract_text_output(stdout)
    if not payload:
        raise ModelInvocationError("model returned empty response")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        embedded = _extract_embedded_json_candidate(stdout)
        if embedded is not None:
            try:
                return json.loads(embedded)
            except json.JSONDecodeError:
                pass
        raise ModelInvocationError(f"model returned non-JSON response: {error}") from error


def _build_command(profile: ModelProfileConfig, *, prompt: str) -> list[str]:
    if profile.provider == "github_cli":
        return [
            "gh",
            "copilot",
            "--",
            "--prompt",
            prompt,
            "--model",
            profile.model,
            "--output-format",
            "json",
            "--allow-all-tools",
        ]

    command = [
        "kilo",
        "run",
        "--auto",
        "--format",
        "json",
        "--model",
        profile.model,
    ]
    if profile.agent:
        command.extend(["--agent", profile.agent])
    if profile.variant:
        command.extend(["--variant", profile.variant])
    command.append(prompt)
    return command


def _parse_json_maybe(payload: str) -> Any | None:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    values: list[Any] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(json.loads(stripped))
        except json.JSONDecodeError:
            return None
    return values or None


def _parse_direct_json_maybe(payload: str) -> Any | None:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _extract_embedded_json_candidate(payload: str) -> str | None:
    decoder = json.JSONDecoder()
    markers: list[int] = []
    for marker in ("```json", "```"):
        start = 0
        while True:
            index = payload.find(marker, start)
            if index == -1:
                break
            markers.append(index)
            start = index + len(marker)

    for marker_index in sorted(set(markers), reverse=True):
        candidate = payload[marker_index + 3 :].lstrip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()

        starts = [position for position in (candidate.find("{"), candidate.find("[")) if position != -1]
        if not starts:
            continue
        candidate = candidate[min(starts) :]

        try:
            _, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        return candidate[:end]

    return None


def _extract_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_extract_text_values(item))
        return output
    if isinstance(value, dict):
        output: list[str] = []
        for key in ("output_text", "text", "content", "message", "plan"):
            if key in value:
                output.extend(_extract_text_values(value[key]))
        for key in ("response", "data", "delta", "payload"):
            if key in value:
                output.extend(_extract_text_values(value[key]))
        return output
    return []


def _extract_provider_metadata(provider: str, stdout: str) -> dict[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        return {}

    parsed = _parse_json_maybe(stripped)
    if parsed is None:
        return {}

    if provider == "github_cli":
        return _extract_github_cli_metadata(parsed)
    if provider == "kilo_cli":
        return _extract_kilo_cli_metadata(parsed)
    return _extract_generic_metadata(parsed)


def _extract_github_cli_metadata(parsed: Any) -> dict[str, Any]:
    usage = _find_usage_block(
        parsed,
        candidate_keys={"usage", "token_usage", "billing", "metrics"},
        type_hints={"usage", "token_usage", "billing", "message_stop", "turn_completed"},
    )
    metadata = _extract_generic_metadata(parsed if usage is None else usage)
    model_name = _find_string_value(parsed, candidate_keys={"model", "model_name", "resolved_model"})
    if model_name is not None:
        metadata["reported_model"] = model_name
    return metadata


def _extract_kilo_cli_metadata(parsed: Any) -> dict[str, Any]:
    usage = _find_usage_block(
        parsed,
        candidate_keys={"usage", "token_usage", "cost", "metrics", "summary"},
        type_hints={"run_finished", "step_finish", "response_completed", "summary"},
    )
    metadata = _extract_generic_metadata(parsed if usage is None else usage)
    model_name = _find_string_value(parsed, candidate_keys={"model", "model_name", "resolved_model"})
    if model_name is not None:
        metadata["reported_model"] = model_name
    return metadata


def _extract_generic_metadata(parsed: Any) -> dict[str, Any]:
    return {
        "estimated_cost_usd": _find_decimal_value(
            parsed,
            candidate_keys={"cost_usd", "estimated_cost_usd", "total_cost_usd", "usd_cost", "usd"},
        ),
        "input_tokens": _find_int_value(
            parsed,
            candidate_keys={"input_tokens", "prompt_tokens", "tokens_in", "prompt_token_count"},
        ),
        "output_tokens": _find_int_value(
            parsed,
            candidate_keys={"output_tokens", "completion_tokens", "tokens_out", "completion_token_count"},
        ),
    }


def _find_usage_block(value: Any, *, candidate_keys: set[str], type_hints: set[str]) -> Any | None:
    if isinstance(value, dict):
        record_type = value.get("type") or value.get("event")
        if record_type in type_hints:
            for key in candidate_keys:
                if key in value:
                    return value[key]
        for key, nested in value.items():
            if key in candidate_keys:
                return nested
            match = _find_usage_block(nested, candidate_keys=candidate_keys, type_hints=type_hints)
            if match is not None:
                return match
        return None
    if isinstance(value, list):
        for item in value:
            match = _find_usage_block(item, candidate_keys=candidate_keys, type_hints=type_hints)
            if match is not None:
                return match
    return None


def _find_decimal_value(value: Any, *, candidate_keys: set[str]) -> Decimal | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in candidate_keys:
                decimal_value = _coerce_decimal(nested)
                if decimal_value is not None:
                    return decimal_value
            decimal_value = _find_decimal_value(nested, candidate_keys=candidate_keys)
            if decimal_value is not None:
                return decimal_value
        return None
    if isinstance(value, list):
        for item in value:
            decimal_value = _find_decimal_value(item, candidate_keys=candidate_keys)
            if decimal_value is not None:
                return decimal_value
    return None


def _find_int_value(value: Any, *, candidate_keys: set[str]) -> int | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in candidate_keys:
                int_value = _coerce_int(nested)
                if int_value is not None:
                    return int_value
            int_value = _find_int_value(nested, candidate_keys=candidate_keys)
            if int_value is not None:
                return int_value
        return None
    if isinstance(value, list):
        for item in value:
            int_value = _find_int_value(item, candidate_keys=candidate_keys)
            if int_value is not None:
                return int_value
    return None


def _find_string_value(value: Any, *, candidate_keys: set[str]) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in candidate_keys and isinstance(nested, str) and nested.strip():
                return nested
            string_value = _find_string_value(nested, candidate_keys=candidate_keys)
            if string_value is not None:
                return string_value
        return None
    if isinstance(value, list):
        for item in value:
            string_value = _find_string_value(item, candidate_keys=candidate_keys)
            if string_value is not None:
                return string_value
    return None


def _coerce_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None