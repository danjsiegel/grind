from __future__ import annotations

from pathlib import Path

from grind.engine.evidence_verifier import (
    CandidateFindingForVerification,
    EvidenceVerificationStatus,
    verify_candidate_findings,
)


def test_verify_candidate_finding_marks_existing_file_line_and_symbol_verified(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source_file = source_dir / "sample.py"
    source_file.write_text("def handler():\n    return 1\n", encoding="utf-8")

    report = verify_candidate_findings(
        report_id="artifact_report",
        candidates=[
            CandidateFindingForVerification(
                stable_id="1234567890abcdef",
                title="Handler is wrong",
                file_path="src/sample.py",
                primary_symbol="handler",
                line_range="1-2",
            )
        ],
        cwd=tmp_path,
    )

    assert report.verified_count == 1
    assert report.failed_count == 0
    assert report.findings[0].status == EvidenceVerificationStatus.VERIFIED


def test_verify_candidate_finding_marks_missing_file_as_failed(tmp_path: Path) -> None:
    report = verify_candidate_findings(
        report_id="artifact_report",
        candidates=[
            CandidateFindingForVerification(
                stable_id="1234567890abcdef",
                title="Missing file citation",
                file_path="src/missing.py",
                primary_symbol="ghost",
                line_range="10-12",
            )
        ],
        cwd=tmp_path,
    )

    assert report.failed_count == 1
    assert report.findings[0].status == EvidenceVerificationStatus.FAILED
    assert any(check.check == "file_path" and check.status == EvidenceVerificationStatus.FAILED for check in report.findings[0].checks)


def test_verify_candidate_finding_without_citations_is_unverifiable(tmp_path: Path) -> None:
    report = verify_candidate_findings(
        report_id="artifact_report",
        candidates=[
            CandidateFindingForVerification(
                stable_id="1234567890abcdef",
                title="Uncited concern",
            )
        ],
        cwd=tmp_path,
    )

    assert report.unverifiable_count == 1
    assert report.findings[0].status == EvidenceVerificationStatus.UNVERIFIABLE