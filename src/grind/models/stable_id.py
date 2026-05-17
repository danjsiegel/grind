from __future__ import annotations

import hashlib
import re


def _normalize(text: str) -> str:
    """Strip leading/trailing whitespace, collapse internal whitespace, lowercase (§6.4.2)."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def stable_id(
    *,
    run_id: str,
    category: str,
    file_path: str | None = None,
    primary_symbol: str | None = None,
    line_range: str | None = None,
    stage_id: str | None = None,
) -> str:
    """Generate a 16-character hex stable_id for a finding per §6.4.2.

    Strategy priority (highest to lowest):
    1. Primary:    file_path + primary_symbol  (preferred when symbol is available)
    2. Fallback:   file_path + line_range
    3. Last resort: stage_id only

    The stable_id must not include title or description text — model phrasing is
    too unstable across iterations for string-hash deduplication.
    """
    if file_path is not None and primary_symbol is not None:
        raw = f"{run_id}|{_normalize(file_path)}|{category}|{_normalize(primary_symbol)}"
    elif file_path is not None and line_range is not None:
        raw = f"{run_id}|{_normalize(file_path)}|{category}|{_normalize(line_range)}"
    elif stage_id is not None:
        raw = f"{run_id}|{category}|{_normalize(stage_id)}"
    else:
        raise ValueError(
            "stable_id requires (file_path + primary_symbol), "
            "(file_path + line_range), or stage_id"
        )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
