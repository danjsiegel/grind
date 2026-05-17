from __future__ import annotations

from enum import StrEnum
from pathlib import Path
import re

from pydantic import BaseModel, Field


class EvidenceVerificationStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIABLE = "unverifiable"
    FAILED = "failed"


class EvidenceCheckResult(BaseModel):
    check: str = Field(min_length=1)
    status: EvidenceVerificationStatus
    detail: str = Field(min_length=1)


class CandidateFindingForVerification(BaseModel):
    stable_id: str = Field(min_length=16, max_length=16)
    title: str = Field(min_length=1)
    file_path: str | None = None
    primary_symbol: str | None = None
    line_range: str | None = None


class FindingVerificationRecord(BaseModel):
    stable_id: str = Field(min_length=16, max_length=16)
    title: str = Field(min_length=1)
    status: EvidenceVerificationStatus
    checks: list[EvidenceCheckResult] = Field(default_factory=list)


class EvidenceVerificationReport(BaseModel):
    report_id: str = Field(min_length=1)
    findings: list[FindingVerificationRecord] = Field(default_factory=list)

    @property
    def verified_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status == EvidenceVerificationStatus.VERIFIED)

    @property
    def failed_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status == EvidenceVerificationStatus.FAILED)

    @property
    def unverifiable_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status == EvidenceVerificationStatus.UNVERIFIABLE)


def verify_candidate_findings(
    *,
    report_id: str,
    candidates: list[CandidateFindingForVerification],
    cwd: Path,
) -> EvidenceVerificationReport:
    return EvidenceVerificationReport(
        report_id=report_id,
        findings=[_verify_candidate(candidate, cwd=cwd) for candidate in candidates],
    )


def _verify_candidate(candidate: CandidateFindingForVerification, *, cwd: Path) -> FindingVerificationRecord:
    checks: list[EvidenceCheckResult] = []
    file_content: str | None = None
    line_count: int | None = None
    resolved_path: Path | None = None

    if candidate.file_path:
        resolved_path = cwd / candidate.file_path
        if resolved_path.exists() and resolved_path.is_file():
            file_content = resolved_path.read_text(encoding="utf-8")
            line_count = len(file_content.splitlines())
            checks.append(
                EvidenceCheckResult(
                    check="file_path",
                    status=EvidenceVerificationStatus.VERIFIED,
                    detail=f"Found file at {candidate.file_path}.",
                )
            )
        else:
            checks.append(
                EvidenceCheckResult(
                    check="file_path",
                    status=EvidenceVerificationStatus.FAILED,
                    detail=f"File does not exist: {candidate.file_path}.",
                )
            )
    else:
        checks.append(
            EvidenceCheckResult(
                check="file_path",
                status=EvidenceVerificationStatus.UNVERIFIABLE,
                detail="Finding did not cite a file path.",
            )
        )

    if candidate.line_range:
        parsed_range = _parse_line_range(candidate.line_range)
        if parsed_range is None:
            checks.append(
                EvidenceCheckResult(
                    check="line_range",
                    status=EvidenceVerificationStatus.FAILED,
                    detail=f"Unrecognized line range format: {candidate.line_range}.",
                )
            )
        elif line_count is None:
            checks.append(
                EvidenceCheckResult(
                    check="line_range",
                    status=EvidenceVerificationStatus.FAILED,
                    detail="Cannot validate line range because the cited file is unavailable.",
                )
            )
        else:
            start_line, end_line = parsed_range
            if start_line >= 1 and end_line >= start_line and end_line <= line_count:
                checks.append(
                    EvidenceCheckResult(
                        check="line_range",
                        status=EvidenceVerificationStatus.VERIFIED,
                        detail=f"Line range {start_line}-{end_line} exists in the cited file.",
                    )
                )
            else:
                checks.append(
                    EvidenceCheckResult(
                        check="line_range",
                        status=EvidenceVerificationStatus.FAILED,
                        detail=f"Line range {candidate.line_range} is outside the cited file bounds ({line_count} lines).",
                    )
                )
    else:
        checks.append(
            EvidenceCheckResult(
                check="line_range",
                status=EvidenceVerificationStatus.UNVERIFIABLE,
                detail="Finding did not cite a line range.",
            )
        )

    if candidate.primary_symbol:
        if file_content is None or resolved_path is None:
            checks.append(
                EvidenceCheckResult(
                    check="primary_symbol",
                    status=EvidenceVerificationStatus.FAILED,
                    detail="Cannot validate symbol presence because the cited file is unavailable.",
                )
            )
        elif _symbol_exists(file_content, candidate.primary_symbol):
            checks.append(
                EvidenceCheckResult(
                    check="primary_symbol",
                    status=EvidenceVerificationStatus.VERIFIED,
                    detail=f"Found symbol reference for {candidate.primary_symbol} in {candidate.file_path}.",
                )
            )
        else:
            checks.append(
                EvidenceCheckResult(
                    check="primary_symbol",
                    status=EvidenceVerificationStatus.FAILED,
                    detail=f"Could not find the cited symbol {candidate.primary_symbol} in {candidate.file_path}.",
                )
            )
    else:
        checks.append(
            EvidenceCheckResult(
                check="primary_symbol",
                status=EvidenceVerificationStatus.UNVERIFIABLE,
                detail="Finding did not cite a primary symbol.",
            )
        )

    overall_status = _overall_status(checks)
    return FindingVerificationRecord(
        stable_id=candidate.stable_id,
        title=candidate.title,
        status=overall_status,
        checks=checks,
    )


def _parse_line_range(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*(?:[-:]\s*(\d+))?\s*", value)
    if match is None:
        return None
    start_line = int(match.group(1))
    end_group = match.group(2)
    end_line = int(end_group) if end_group is not None else start_line
    return start_line, end_line


def _symbol_exists(file_content: str, symbol: str) -> bool:
    escaped_symbol = re.escape(symbol)
    patterns = [
        rf"\bdef\s+{escaped_symbol}\b",
        rf"\bclass\s+{escaped_symbol}\b",
        rf"\b{escaped_symbol}\b",
    ]
    return any(re.search(pattern, file_content) for pattern in patterns)


def _overall_status(checks: list[EvidenceCheckResult]) -> EvidenceVerificationStatus:
    if any(check.status == EvidenceVerificationStatus.FAILED for check in checks):
        return EvidenceVerificationStatus.FAILED
    if any(check.status == EvidenceVerificationStatus.VERIFIED for check in checks):
        return EvidenceVerificationStatus.VERIFIED
    return EvidenceVerificationStatus.UNVERIFIABLE