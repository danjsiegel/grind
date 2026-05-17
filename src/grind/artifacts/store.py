from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol
from typing import Any

from grind.models.artifact import ArtifactRecord


class ArtifactChecksumError(RuntimeError):
    pass


class ArtifactStore(Protocol):
    root: Path

    def write_text(self, *, run_id: str, artifact_id: str, artifact_type: str, content: str, suffix: str = ".txt", metadata: dict[str, Any] | None = None) -> ArtifactRecord: ...
    def write_json(self, *, run_id: str, artifact_id: str, artifact_type: str, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> ArtifactRecord: ...
    def write_bytes(self, *, run_id: str, artifact_id: str, artifact_type: str, content: bytes, suffix: str, metadata: dict[str, Any] | None = None) -> ArtifactRecord: ...
    def read_bytes(self, artifact: ArtifactRecord) -> bytes: ...
    def read_text(self, artifact: ArtifactRecord, *, encoding: str = "utf-8") -> str: ...
    def resolve_path(self, artifact: ArtifactRecord | str | Path) -> Path: ...


class LocalArtifactStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_text(
        self,
        *,
        run_id: str,
        artifact_id: str,
        artifact_type: str,
        content: str,
        suffix: str = ".txt",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        path = self._artifact_path(run_id=run_id, artifact_id=artifact_id, suffix=suffix)
        path.write_text(content, encoding="utf-8")
        return self._record_for_path(
            run_id=run_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            path=path,
            metadata=metadata,
        )

    def write_json(
        self,
        *,
        run_id: str,
        artifact_id: str,
        artifact_type: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        return self.write_text(
            run_id=run_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content=json.dumps(payload, indent=2, sort_keys=True) + "\n",
            suffix=".json",
            metadata=metadata,
        )

    def write_bytes(
        self,
        *,
        run_id: str,
        artifact_id: str,
        artifact_type: str,
        content: bytes,
        suffix: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        path = self._artifact_path(run_id=run_id, artifact_id=artifact_id, suffix=suffix)
        path.write_bytes(content)
        return self._record_for_path(
            run_id=run_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            path=path,
            metadata=metadata,
        )

    def read_bytes(self, artifact: ArtifactRecord) -> bytes:
        path = self.resolve_path(artifact)
        payload = path.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        if artifact.checksum and checksum != artifact.checksum:
            raise ArtifactChecksumError(
                f"artifact checksum mismatch for {artifact.artifact_id}: expected {artifact.checksum}, got {checksum}"
            )
        return payload

    def read_text(self, artifact: ArtifactRecord, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(artifact).decode(encoding)

    def resolve_path(self, artifact: ArtifactRecord | str | Path) -> Path:
        if isinstance(artifact, ArtifactRecord):
            candidate = Path(artifact.path)
        else:
            candidate = Path(artifact)
        if candidate.is_absolute():
            return candidate
        return self.root / candidate

    def _artifact_path(self, *, run_id: str, artifact_id: str, suffix: str) -> Path:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / f"{artifact_id}{suffix}"

    def _record_for_path(
        self,
        *,
        run_id: str,
        artifact_id: str,
        artifact_type: str,
        path: Path,
        metadata: dict[str, Any] | None,
    ) -> ArtifactRecord:
        payload = path.read_bytes()
        relative_path = path.relative_to(self.root)
        return ArtifactRecord(
            artifact_id=artifact_id,
            run_id=run_id,
            artifact_type=artifact_type,
            path=str(relative_path),
            checksum=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            metadata=metadata,
        )