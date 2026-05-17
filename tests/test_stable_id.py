from __future__ import annotations

import hashlib

import pytest

from grind.models.stable_id import _normalize, stable_id


# ── _normalize ────────────────────────────────────────────────────────────────

def test_normalize_strips_leading_trailing_whitespace() -> None:
    assert _normalize("  hello  ") == "hello"


def test_normalize_collapses_internal_whitespace() -> None:
    assert _normalize("hello   world") == "hello world"


def test_normalize_lowercases() -> None:
    assert _normalize("MyClass.Method") == "myclass.method"


def test_normalize_combined() -> None:
    assert _normalize("  Hello   WORLD  ") == "hello world"


def test_normalize_tabs_and_newlines() -> None:
    assert _normalize("hello\t\nworld") == "hello world"


# ── Primary strategy: file_path + primary_symbol ──────────────────────────────

def test_primary_strategy_is_deterministic() -> None:
    sid1 = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        primary_symbol="MyClass.method",
    )
    sid2 = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        primary_symbol="MyClass.method",
    )
    assert sid1 == sid2


def test_primary_strategy_produces_16_chars() -> None:
    sid = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        primary_symbol="MyClass.method",
    )
    assert len(sid) == 16


def test_primary_strategy_matches_spec_formula() -> None:
    """sha256(run_id | normalize(file_path) | category | normalize(symbol))[:16]"""
    raw = "run-abc|src/foo.py|correctness|myclass.method"
    expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
    sid = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        primary_symbol="MyClass.method",
    )
    assert sid == expected


def test_primary_strategy_normalizes_symbol_casing() -> None:
    sid_lower = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        primary_symbol="myclass.method",
    )
    sid_upper = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        primary_symbol="MyClass.Method",
    )
    assert sid_lower == sid_upper


# ── Fallback strategy: file_path + line_range ─────────────────────────────────

def test_fallback_strategy_is_deterministic() -> None:
    sid1 = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        line_range="42:58",
    )
    sid2 = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        line_range="42:58",
    )
    assert sid1 == sid2


def test_fallback_strategy_produces_16_chars() -> None:
    sid = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        line_range="42:58",
    )
    assert len(sid) == 16


def test_fallback_strategy_matches_spec_formula() -> None:
    """sha256(run_id | normalize(file_path) | category | normalize(line_range))[:16]"""
    raw = "run-abc|src/foo.py|correctness|42:58"
    expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
    sid = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        line_range="42:58",
    )
    assert sid == expected


# ── Last-resort strategy: stage_id only ──────────────────────────────────────

def test_last_resort_strategy_is_deterministic() -> None:
    sid1 = stable_id(run_id="run-abc", category="system_error", stage_id="stage-001")
    sid2 = stable_id(run_id="run-abc", category="system_error", stage_id="stage-001")
    assert sid1 == sid2


def test_last_resort_strategy_produces_16_chars() -> None:
    sid = stable_id(run_id="run-abc", category="system_error", stage_id="stage-001")
    assert len(sid) == 16


def test_last_resort_strategy_matches_spec_formula() -> None:
    """sha256(run_id | category | normalize(stage_id))[:16]"""
    raw = "run-abc|system_error|stage-001"
    expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
    sid = stable_id(run_id="run-abc", category="system_error", stage_id="stage-001")
    assert sid == expected


# ── Strategy priority ─────────────────────────────────────────────────────────

def test_primary_takes_precedence_over_line_range() -> None:
    """When both symbol and line_range are provided, the primary strategy wins."""
    sid_with_both = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        primary_symbol="my_func",
        line_range="10:20",
    )
    sid_symbol_only = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        primary_symbol="my_func",
    )
    assert sid_with_both == sid_symbol_only


def test_fallback_used_when_no_symbol() -> None:
    """When no symbol but line_range provided, fallback strategy is used."""
    sid_fallback = stable_id(
        run_id="run-abc",
        category="correctness",
        file_path="src/foo.py",
        line_range="10:20",
    )
    raw = "run-abc|src/foo.py|correctness|10:20"
    expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
    assert sid_fallback == expected


# ── Distinct inputs → distinct ids ───────────────────────────────────────────

def test_different_symbols_produce_different_ids() -> None:
    sid1 = stable_id(
        run_id="run-abc", category="correctness",
        file_path="src/foo.py", primary_symbol="func_a",
    )
    sid2 = stable_id(
        run_id="run-abc", category="correctness",
        file_path="src/foo.py", primary_symbol="func_b",
    )
    assert sid1 != sid2


def test_different_files_produce_different_ids() -> None:
    sid1 = stable_id(
        run_id="run-abc", category="correctness",
        file_path="src/foo.py", primary_symbol="func_a",
    )
    sid2 = stable_id(
        run_id="run-abc", category="correctness",
        file_path="src/bar.py", primary_symbol="func_a",
    )
    assert sid1 != sid2


def test_different_runs_produce_different_ids() -> None:
    sid1 = stable_id(
        run_id="run-001", category="correctness",
        file_path="src/foo.py", primary_symbol="func_a",
    )
    sid2 = stable_id(
        run_id="run-002", category="correctness",
        file_path="src/foo.py", primary_symbol="func_a",
    )
    assert sid1 != sid2


def test_different_categories_produce_different_ids() -> None:
    sid1 = stable_id(
        run_id="run-abc", category="correctness",
        file_path="src/foo.py", primary_symbol="func_a",
    )
    sid2 = stable_id(
        run_id="run-abc", category="security",
        file_path="src/foo.py", primary_symbol="func_a",
    )
    assert sid1 != sid2


# ── Error: no locator ─────────────────────────────────────────────────────────

def test_raises_without_any_locator() -> None:
    with pytest.raises(ValueError, match="stable_id requires"):
        stable_id(run_id="run-abc", category="correctness")


def test_raises_with_file_path_but_no_symbol_or_line_range() -> None:
    with pytest.raises(ValueError, match="stable_id requires"):
        stable_id(run_id="run-abc", category="correctness", file_path="src/foo.py")
