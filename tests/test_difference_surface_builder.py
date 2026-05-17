from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from grind.artifacts import LocalArtifactStore
from grind.engine.checkpoints import capture_workspace_snapshot
from grind.engine.difference_surface_builder import build_difference_surface
from grind.models import Run, Task, TaskSourceKind, TaskStatus
from grind.validation import ValidationExecutionResult


def test_build_difference_surface_uses_authoritative_workspace_delta(tmp_path: Path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / ".grind" / "artifacts")
    tracked_file = tmp_path / "tracked.py"
    removed_file = tmp_path / "removed.txt"
    tracked_file.write_text("print('baseline')\n", encoding="utf-8")
    removed_file.write_text("remove me\n", encoding="utf-8")

    baseline_artifact = capture_workspace_snapshot(
        tmp_path,
        run_id="run_test",
        artifact_id="artifact_baseline",
        artifact_store=artifact_store,
    )

    tracked_file.write_text("print('changed')\n", encoding="utf-8")
    (tmp_path / "added.py").write_text("print('new')\n", encoding="utf-8")
    removed_file.unlink()

    run = Run(
        run_id="run_test",
        repo_path=str(tmp_path),
        policy_pack_path=str(tmp_path),
        policy_schema_ver="0.1",
        requested_objective="Change tracked files",
        max_iterations=4,
        budget_limit_usd=Decimal("10.00"),
    )
    task = Task(
        task_id="task_test",
        run_id="run_test",
        sequence=0,
        source_kind=TaskSourceKind.INLINE,
        raw_input="Change tracked files",
        status=TaskStatus.IN_PROGRESS,
    )

    result = build_difference_surface(
        cwd=tmp_path,
        run=run,
        task=task,
        iteration=1,
        observed_delta={
            "source_stage": "doing",
            "reported_touched_files": ["tracked.py"],
            "validation_hints": [],
            "claims_made": [],
        },
        validation_results=[
            ValidationExecutionResult(
                command="uv run pytest tests -q",
                returncode=0,
                stdout="passed",
                stderr="",
            )
        ],
        open_findings=[],
        baseline_snapshot_path=artifact_store.resolve_path(baseline_artifact),
        stop_on_failure=True,
    )

    assert result.modified_files == ["tracked.py"]
    assert result.added_files == ["added.py"]
    assert result.removed_files == ["removed.txt"]
    assert result.existing_reported_files == ["tracked.py"]
    assert result.unreported_changed_files == ["added.py", "removed.txt"]
    assert result.surface.observed_delta["authoritative_touched_files"] == [
        "added.py",
        "removed.txt",
        "tracked.py",
    ]
    assert result.surface.policy_delta["max_iterations"] == 4
    assert result.surface.validation_delta["stop_on_failure"] is True