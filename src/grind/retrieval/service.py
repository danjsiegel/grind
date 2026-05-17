from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

import lancedb

from grind.config import EngineConfig
from grind.models import RetrievalQueueRecord
from grind.retrieval.embeddings import ProviderEmbeddingAdapter
from grind.state import open_state_store


class LanceDBRetrievalService:
    INDEXABLE_ARTIFACT_TYPES: dict[str, str] = {
        "planning_prompt": "prompt_chunks",
        "do_prompt": "prompt_chunks",
        "act_prompt": "prompt_chunks",
        "planning_response": "run_summaries",
        "do_response": "run_summaries",
        "do_output": "run_summaries",
        "plan_review": "run_summaries",
        "difference_surface": "run_summaries",
        "semantic_audit_report": "run_summaries",
        "adjudication_report": "run_summaries",
        "act_output": "run_summaries",
        "validation_output": "run_summaries",
    }
    WORKSPACE_DOC_COLLECTION = "docs_chunks"
    WORKSPACE_SPEC_COLLECTION = "spec_chunks"

    def __init__(self, *, cwd: Path, config: EngineConfig):
        self.cwd = cwd
        self.config = config
        self.database_path = config.state_path(cwd)
        self.db_uri = config.state_db_uri()
        self.artifacts_root = config.artifacts_root(cwd)
        self.db_path = config.retrieval_path(cwd)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_adapter = ProviderEmbeddingAdapter(config.retrieval)

    def enqueue_run_artifacts(self, *, run_id: str) -> dict[str, object]:
        if not self.config.retrieval.enabled:
            return {"enabled": False, "run_id": run_id, "queued": 0, "collections": {}}

        queued = 0
        collections: dict[str, int] = {}
        with open_state_store(self.database_path, db_uri=self.db_uri) as store:
            artifacts = store.artifacts.list_by_run(run_id)
            for artifact in artifacts:
                collection = self.INDEXABLE_ARTIFACT_TYPES.get(artifact.artifact_type)
                if collection is None:
                    continue
                existing = store.retrieval_queue.get_existing(
                    run_id=run_id,
                    artifact_id=artifact.artifact_id,
                    collection=collection,
                )
                if existing is not None and existing.queue_status in {"pending", "running", "completed"}:
                    continue
                store.retrieval_queue.create(
                    RetrievalQueueRecord(
                        queue_id=f"queue_{secrets.token_hex(8)}",
                        run_id=run_id,
                        artifact_id=artifact.artifact_id,
                        collection=collection,
                    )
                )
                queued += 1
                collections[collection] = collections.get(collection, 0) + 1
        return {"enabled": True, "run_id": run_id, "queued": queued, "collections": collections}

    def process_run_queue(self, *, run_id: str) -> dict[str, object]:
        if not self.config.retrieval.enabled:
            return {"enabled": False, "run_id": run_id, "processed": 0, "documents_indexed": 0, "failed": 0}

        processed = 0
        failed = 0
        documents_indexed = 0
        with open_state_store(self.database_path, db_uri=self.db_uri) as store:
            queue_records = store.retrieval_queue.list_pending(run_id=run_id)
            for record in queue_records:
                artifact = store.artifacts.get(record.artifact_id)
                if artifact is None:
                    store.retrieval_queue.mark_failed(record.queue_id, last_error="artifact missing")
                    failed += 1
                    continue
                try:
                    store.retrieval_queue.mark_running(record.queue_id)
                    documents = self._documents_for_artifact(
                        run_id=run_id,
                        artifact_id=artifact.artifact_id,
                        artifact_type=artifact.artifact_type,
                        path=self._artifact_path(artifact.path),
                        collection=record.collection,
                    )
                    if documents:
                        self._replace_collection_documents(
                            record.collection,
                            documents,
                            delete_filter=f"artifact_id = {self._sql_literal(artifact.artifact_id)}",
                        )
                        documents_indexed += len(documents)
                    store.retrieval_queue.mark_completed(record.queue_id)
                    processed += 1
                except Exception as error:  # pragma: no cover - defensive for LanceDB runtime issues
                    store.retrieval_queue.mark_failed(record.queue_id, last_error=str(error))
                    failed += 1

            try:
                documents_indexed += self._index_findings_narratives(store=store, run_id=run_id)
            except Exception:  # pragma: no cover - defensive for LanceDB runtime issues
                failed += 1

        try:
            documents_indexed += self._index_workspace_sources()
        except Exception:  # pragma: no cover - defensive for LanceDB runtime issues
            failed += 1

        return {
            "enabled": True,
            "run_id": run_id,
            "processed": processed,
            "documents_indexed": documents_indexed,
            "failed": failed,
            "collections": self.collection_stats(run_id=run_id),
        }

    def index_run(self, *, run_id: str) -> dict[str, object]:
        enqueue_summary = self.enqueue_run_artifacts(run_id=run_id)
        process_summary = self.process_run_queue(run_id=run_id)
        return {
            "enabled": process_summary["enabled"],
            "run_id": run_id,
            "queued": enqueue_summary["queued"],
            "processed": process_summary["processed"],
            "documents_indexed": process_summary["documents_indexed"],
            "failed": process_summary["failed"],
            "queue_collections": enqueue_summary["collections"],
            "indexed_collections": process_summary["collections"],
        }

    def search(
        self,
        *,
        query: str,
        run_id: str | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object]:
        if not self.config.retrieval.enabled:
            return {"enabled": False, "query": query, "results": []}

        db = lancedb.connect(str(self.db_path))
        vector_batch = self.embedding_adapter.embed_texts([query])
        vector = vector_batch.vectors[0]
        requested_limit = limit or self.config.retrieval.max_search_results
        collection_names = [collection] if collection else [name for name in self._table_names(db) if not name.startswith("_")]

        results: list[dict[str, object]] = []
        for collection_name in collection_names:
            try:
                table = db.open_table(collection_name)
            except Exception:
                continue
            rows = table.search(vector).limit(max(requested_limit * 3, requested_limit)).to_list()
            for row in rows:
                if run_id is not None and not self._row_matches_run(row, run_id):
                    continue
                results.append(
                    {
                        "collection": collection_name,
                        "chunk_id": row.get("chunk_id"),
                        "run_id": row.get("run_id"),
                        "artifact_id": row.get("artifact_id"),
                        "artifact_type": row.get("artifact_type"),
                        "chunk_text": row.get("chunk_text"),
                        "metadata": self._decode_metadata(row.get("metadata_json")),
                        "score": row.get("_distance"),
                    }
                )

        results.sort(key=lambda item: item["score"] if item["score"] is not None else float("inf"))
        return {
            "enabled": True,
            "query": query,
            "run_id": run_id,
            "collection": collection,
            "results": results[:requested_limit],
        }

    def collection_stats(self, *, run_id: str | None = None) -> dict[str, int]:
        if not self.config.retrieval.enabled:
            return {}

        db = lancedb.connect(str(self.db_path))
        stats: dict[str, int] = {}
        for collection_name in [name for name in self._table_names(db) if not name.startswith("_")]:
            table = db.open_table(collection_name)
            rows = table.to_arrow().to_pylist()
            if run_id is not None:
                rows = [row for row in rows if self._row_matches_run(row, run_id)]
            stats[collection_name] = len(rows)
        return stats

    def _documents_for_artifact(
        self,
        *,
        run_id: str,
        artifact_id: str,
        artifact_type: str,
        path: Path,
        collection: str,
    ) -> list[dict[str, object]]:
        if not path.exists() or path.suffixes[-2:] == [".tar", ".gz"]:
            return []

        raw_text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".json":
            try:
                raw_text = json.dumps(json.loads(raw_text), indent=2, sort_keys=True)
            except json.JSONDecodeError:
                pass

        decorated = f"artifact_type: {artifact_type}\nartifact_id: {artifact_id}\n\n{raw_text.strip()}"
        chunks = self._chunk_text(decorated)
        embedding_batch = self.embedding_adapter.embed_texts(chunks)
        documents: list[dict[str, object]] = []
        for index, (chunk, vector) in enumerate(zip(chunks, embedding_batch.vectors)):
            documents.append(
                {
                    "chunk_id": f"{artifact_id}:{index}",
                    "vector": vector,
                    "chunk_text": chunk,
                    "run_id": run_id,
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "metadata_json": json.dumps(
                        {
                            "artifact_path": str(path),
                            "collection": collection,
                            "chunk_index": index,
                            "embedding_backend": embedding_batch.backend,
                            "embedding_provider": self.config.retrieval.embedding_provider,
                            "embedding_model": self.config.retrieval.embedding_model,
                        },
                        sort_keys=True,
                    ),
                }
            )
        return documents

    def _artifact_path(self, recorded_path: str) -> Path:
        candidate = Path(recorded_path)
        if candidate.is_absolute():
            return candidate
        return self.artifacts_root / candidate

    def _index_findings_narratives(self, *, store: object, run_id: str) -> int:
        findings = store.findings.list_by_run(run_id)
        self._replace_collection_documents(
            "findings_narratives",
            [],
            delete_filter=(
                f"run_id = {self._sql_literal(run_id)} AND artifact_type = {self._sql_literal('finding')}"
            ),
        )
        if not findings:
            return 0
        texts = []
        for finding in findings:
            text = (
                f"title: {finding.title}\n"
                f"severity: {finding.severity.value}\n"
                f"category: {finding.category.value}\n"
                f"rationale: {finding.rationale}\n"
                f"fix: {finding.exact_fix_action}"
            )
            texts.append(text)

        embedding_batch = self.embedding_adapter.embed_texts(texts)
        documents = []
        for finding, text, vector in zip(findings, texts, embedding_batch.vectors):
            documents.append(
                {
                    "chunk_id": f"finding:{finding.finding_id}",
                    "vector": vector,
                    "chunk_text": text,
                    "run_id": run_id,
                    "artifact_id": "",
                    "artifact_type": "finding",
                    "metadata_json": json.dumps(
                        {
                            "finding_id": finding.finding_id,
                            "stable_id": finding.stable_id,
                            "embedding_backend": embedding_batch.backend,
                            "embedding_provider": self.config.retrieval.embedding_provider,
                            "embedding_model": self.config.retrieval.embedding_model,
                        },
                        sort_keys=True,
                    ),
                }
            )
        self._replace_collection_documents("findings_narratives", documents)
        return len(documents)

    def _index_workspace_sources(self) -> int:
        documents_indexed = 0
        if self.config.retrieval.index_workspace_docs:
            documents_indexed += self._index_workspace_collection(
                collection=self.WORKSPACE_DOC_COLLECTION,
                globs=self.config.retrieval.workspace_docs_globs,
                artifact_type="workspace_doc",
            )
        if self.config.retrieval.index_workspace_specs:
            documents_indexed += self._index_workspace_collection(
                collection=self.WORKSPACE_SPEC_COLLECTION,
                globs=self.config.retrieval.workspace_spec_globs,
                artifact_type="workspace_spec",
            )
        return documents_indexed

    def _index_workspace_collection(self, *, collection: str, globs: list[str], artifact_type: str) -> int:
        documents_indexed = 0
        for path in self._workspace_files(globs):
            raw_text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not raw_text:
                continue

            relative_path = self._relative_path(path)
            source_key = f"source:{collection}:{self._stable_key(relative_path)}"
            decorated = f"source_path: {relative_path}\nartifact_type: {artifact_type}\n\n{raw_text}"
            chunks = self._chunk_text(decorated)
            if not chunks:
                continue

            embedding_batch = self.embedding_adapter.embed_texts(chunks)
            documents = []
            for index, (chunk, vector) in enumerate(zip(chunks, embedding_batch.vectors)):
                documents.append(
                    {
                        "chunk_id": f"{source_key}:{index}",
                        "vector": vector,
                        "chunk_text": chunk,
                        "run_id": "",
                        "artifact_id": "",
                        "artifact_type": artifact_type,
                        "metadata_json": json.dumps(
                            {
                                "source_path": relative_path,
                                "collection": collection,
                                "chunk_index": index,
                                "embedding_backend": embedding_batch.backend,
                                "embedding_provider": self.config.retrieval.embedding_provider,
                                "embedding_model": self.config.retrieval.embedding_model,
                            },
                            sort_keys=True,
                        ),
                    }
                )

            self._replace_collection_documents(
                collection,
                documents,
                delete_filter=f"chunk_id LIKE {self._sql_literal(f'{source_key}:%')}",
            )
            documents_indexed += len(documents)
        return documents_indexed

    def _replace_collection_documents(
        self,
        collection: str,
        documents: list[dict[str, object]],
        *,
        delete_filter: str | None = None,
    ) -> None:
        db = lancedb.connect(str(self.db_path))
        if collection in self._table_names(db):
            table = db.open_table(collection)
            if delete_filter is not None:
                table.delete(delete_filter)
            if documents:
                table.add(documents)
            return
        if documents:
            db.create_table(collection, data=documents)

    def _table_names(self, db: object) -> list[str]:
        if hasattr(db, "list_tables"):
            listing = db.list_tables()
            if hasattr(listing, "tables"):
                return list(listing.tables)
            if isinstance(listing, list):
                return listing
        if hasattr(db, "table_names"):
            return list(db.table_names())
        return []

    def _chunk_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.config.retrieval.chunk_size:
            return [text]

        step = max(self.config.retrieval.chunk_size - self.config.retrieval.chunk_overlap, 1)
        return [
            text[index : index + self.config.retrieval.chunk_size].strip()
            for index in range(0, len(text), step)
            if text[index : index + self.config.retrieval.chunk_size].strip()
        ]

    def _workspace_files(self, globs: list[str]) -> list[Path]:
        files: dict[str, Path] = {}
        for pattern in globs:
            for candidate in self.cwd.glob(pattern):
                if candidate.is_file():
                    files[str(candidate)] = candidate
        return [files[key] for key in sorted(files)]

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.cwd))
        except ValueError:
            return str(path)

    def _stable_key(self, value: str) -> str:
        import hashlib

        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _row_matches_run(self, row: dict[str, object], run_id: str) -> bool:
        row_run_id = row.get("run_id")
        return row_run_id in {None, "", run_id}

    def _sql_literal(self, value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def _decode_metadata(self, payload: Any) -> dict[str, object] | None:
        if payload in (None, ""):
            return None
        if isinstance(payload, dict):
            return payload
        try:
            return json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return {"raw": payload}