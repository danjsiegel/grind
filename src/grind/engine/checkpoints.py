from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile

from grind.artifacts import ArtifactStore
from grind.models.artifact import ArtifactRecord


EXCLUDED_PATHS = {".git", ".grind", ".venv", "__pycache__"}


def capture_workspace_snapshot(
    cwd: Path,
    *,
    run_id: str,
    artifact_id: str,
    artifact_store: ArtifactStore,
) -> ArtifactRecord:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(cwd.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(cwd)
            if _is_excluded(relative_path):
                continue
            archive.add(path, arcname=str(relative_path))

    return artifact_store.write_bytes(
        run_id=run_id,
        artifact_id=artifact_id,
        artifact_type="checkpoint_snapshot",
        content=buffer.getvalue(),
        suffix=".tar.gz",
        metadata={"capture_mode": "safe_path_snapshot"},
    )


def restore_workspace_snapshot(cwd: Path, *, artifact_store: ArtifactStore, snapshot_artifact: ArtifactRecord) -> None:
    snapshot_bytes = artifact_store.read_bytes(snapshot_artifact)
    with tarfile.open(fileobj=BytesIO(snapshot_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe checkpoint member: {member.name}")
            archive.extract(member, path=cwd)


def _is_excluded(relative_path: Path) -> bool:
    return any(part in EXCLUDED_PATHS for part in relative_path.parts)