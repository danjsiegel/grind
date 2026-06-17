from __future__ import annotations

import json
import math
import re
import secrets
from pathlib import Path
from typing import Any

import lancedb

from grind.config import EngineConfig
from grind.models import RetrievalQueueRecord
from grind.retrieval.embeddings import ProviderEmbeddingAdapter
from grind.state import open_state_store


class LanceDBRetrievalService:
    COLLECTION_META_TABLE = "_collection_meta"
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
        strategy = self._search_strategy_for_backend(vector_batch.backend)
        requested_limit = limit or self.config.retrieval.max_search_results
        collection_names = [collection] if collection else [name for name in self._table_names(db) if not name.startswith("_")]

        results: list[dict[str, object]] = []
        collection_readiness: dict[str, dict[str, object]] = {}
        for collection_name in collection_names:
            try:
                table = db.open_table(collection_name)
            except Exception:
                collection_readiness[collection_name] = {
                    "state": "missing",
                    "vector_compatible": False,
                    "reason": "collection missing",
                    "search_strategy": "lexical",
                }
                continue
            all_rows = table.to_arrow().to_pylist()
            filtered_rows = [row for row in all_rows if run_id is None or self._row_matches_run(row, run_id)]
            readiness = self._collection_readiness(
                db=db,
                collection=collection_name,
                row_count=len(filtered_rows),
                query_backend=vector_batch.backend,
                query_model=vector_batch.model,
                query_dimensions=len(vector),
            )
            collection_strategy = strategy if readiness["vector_compatible"] else "lexical"
            collection_readiness[collection_name] = {
                **readiness,
                "search_strategy": collection_strategy,
            }
            results.extend(
                self._search_collection(
                    collection_name=collection_name,
                    table=table,
                    rows=filtered_rows,
                    query=query,
                    vector=vector,
                    strategy=collection_strategy,
                    limit=requested_limit,
                )
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return {
            "enabled": True,
            "query": query,
            "run_id": run_id,
            "collection": collection,
            "search_strategy": strategy,
            "collection_readiness": collection_readiness,
            "results": results[:requested_limit],
        }

    def collection_readiness(self, *, run_id: str | None = None) -> dict[str, dict[str, object]]:
        if not self.config.retrieval.enabled:
            return {}

        db = lancedb.connect(str(self.db_path))
        readiness: dict[str, dict[str, object]] = {}
        for collection_name in [name for name in self._table_names(db) if not name.startswith("_")]:
            table = db.open_table(collection_name)
            rows = table.to_arrow().to_pylist()
            if run_id is not None:
                rows = [row for row in rows if self._row_matches_run(row, run_id)]
            readiness[collection_name] = self._collection_readiness(
                db=db,
                collection=collection_name,
                row_count=len(rows),
                query_backend=None,
                query_model=None,
                query_dimensions=self.config.retrieval.embedding_dimensions,
            )
        return readiness

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

    def delete_run(self, *, run_id: str) -> dict[str, int]:
        if not self.config.retrieval.enabled:
            return {"enabled": False, "run_id": run_id, "documents_deleted": 0, "collections": {}}

        db = lancedb.connect(str(self.db_path))
        deleted = 0
        collections: dict[str, int] = {}
        delete_filter = f"run_id = {self._sql_literal(run_id)}"

        for collection_name in [name for name in self._table_names(db) if not name.startswith("_")]:
            table = db.open_table(collection_name)
            rows = [row for row in table.to_arrow().to_pylist() if self._row_matches_run(row, run_id)]
            run_rows = [row for row in rows if row.get("run_id") == run_id]
            if not run_rows:
                continue
            table.delete(delete_filter)
            collections[collection_name] = len(run_rows)
            deleted += len(run_rows)

        return {
            "enabled": True,
            "run_id": run_id,
            "documents_deleted": deleted,
            "collections": collections,
        }

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
                self._write_collection_meta(db=db, collection=collection, documents=documents, row_count=len(table.to_arrow().to_pylist()))
            return
        if documents:
            db.create_table(collection, data=documents)
            self._write_collection_meta(db=db, collection=collection, documents=documents, row_count=len(documents))

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

    def _write_collection_meta(
        self,
        *,
        db: object,
        collection: str,
        documents: list[dict[str, object]],
        row_count: int,
    ) -> None:
        first_metadata = self._decode_metadata(documents[0].get("metadata_json")) if documents else None
        backend = None
        if isinstance(first_metadata, dict):
            backend = first_metadata.get("embedding_backend")

        record = {
            "collection": collection,
            "embedding_backend": backend or "unknown",
            "embedding_provider": self.config.retrieval.embedding_provider,
            "embedding_model": self.config.retrieval.embedding_model,
            "embedding_dimensions": self.config.retrieval.embedding_dimensions,
            "readiness_state": "ready" if row_count > 0 else "empty",
            "document_count": row_count,
        }
        import datetime as _datetime

        record["updated_at"] = _datetime.datetime.now(_datetime.timezone.utc).isoformat()

        if self.COLLECTION_META_TABLE in self._table_names(db):
            table = db.open_table(self.COLLECTION_META_TABLE)
            table.delete(f"collection = {self._sql_literal(collection)}")
            table.add([record])
            return
        db.create_table(self.COLLECTION_META_TABLE, data=[record])

    def _read_collection_meta(self, *, db: object, collection: str) -> dict[str, object] | None:
        if self.COLLECTION_META_TABLE not in self._table_names(db):
            return None
        table = db.open_table(self.COLLECTION_META_TABLE)
        rows = table.to_arrow().to_pylist()
        for row in rows:
            if row.get("collection") == collection:
                return row
        return None

    def _collection_readiness(
        self,
        *,
        db: object,
        collection: str,
        row_count: int,
        query_backend: str | None,
        query_model: str | None,
        query_dimensions: int,
    ) -> dict[str, object]:
        if row_count <= 0:
            return {
                "state": "empty",
                "vector_compatible": False,
                "reason": "collection has no rows",
            }

        meta = self._read_collection_meta(db=db, collection=collection)
        if meta is None:
            return {
                "state": "stale",
                "vector_compatible": False,
                "reason": "collection metadata missing",
            }

        if query_backend is None:
            return {
                "state": str(meta.get("readiness_state") or "ready"),
                "vector_compatible": False,
                "reason": "query backend not provided",
            }

        compatible = (
            meta.get("embedding_backend") == query_backend
            and meta.get("embedding_provider") == self.config.retrieval.embedding_provider
            and meta.get("embedding_model") == (query_model or self.config.retrieval.embedding_model)
            and int(meta.get("embedding_dimensions") or 0) == query_dimensions
        )
        if compatible:
            return {
                "state": str(meta.get("readiness_state") or "ready"),
                "vector_compatible": True,
                "reason": "embedding metadata compatible",
            }
        return {
            "state": "incompatible",
            "vector_compatible": False,
            "reason": "collection embeddings do not match current query embedding profile",
        }

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

    def _search_strategy_for_backend(self, backend: str) -> str:
        if backend == "hash-fallback":
            return "hybrid_hash_lexical"
        if backend == "empty":
            return "lexical"
        return "vector"

    def _search_collection(
        self,
        *,
        collection_name: str,
        table: Any,
        rows: list[dict[str, object]],
        query: str,
        vector: list[float],
        strategy: str,
        limit: int,
    ) -> list[dict[str, object]]:
        candidates: dict[str, dict[str, object]] = {}

        if strategy in {"hybrid_hash_lexical", "vector"}:
            for row in self._vector_candidates(table=table, vector=vector, limit=limit):
                chunk_id = str(row.get("chunk_id"))
                vector_score = self._distance_to_similarity(row.get("_distance"))
                candidates[chunk_id] = self._candidate_payload(
                    collection_name=collection_name,
                    row=row,
                    lexical_score=0.0,
                    vector_score=vector_score,
                )

        if strategy in {"hybrid_hash_lexical", "lexical"}:
            for row in rows:
                lexical_score = self._lexical_score(query, row.get("chunk_text"))
                if lexical_score <= 0:
                    continue
                chunk_id = str(row.get("chunk_id"))
                existing = candidates.get(chunk_id)
                if existing is None:
                    candidates[chunk_id] = self._candidate_payload(
                        collection_name=collection_name,
                        row=row,
                        lexical_score=lexical_score,
                        vector_score=0.0,
                    )
                else:
                    existing["lexical_score"] = max(float(existing["lexical_score"]), lexical_score)

        for candidate in candidates.values():
            candidate["score"] = self._final_search_score(
                lexical_score=float(candidate["lexical_score"]),
                vector_score=float(candidate["vector_score"]),
                strategy=strategy,
            )

        return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)[: max(limit * 3, limit)]

    def _vector_candidates(self, *, table: Any, vector: list[float], limit: int) -> list[dict[str, object]]:
        try:
            return table.search(vector).limit(max(limit * 3, limit)).to_list()
        except Exception:
            return []

    def _candidate_payload(
        self,
        *,
        collection_name: str,
        row: dict[str, object],
        lexical_score: float,
        vector_score: float,
    ) -> dict[str, object]:
        return {
            "collection": collection_name,
            "chunk_id": row.get("chunk_id"),
            "run_id": row.get("run_id"),
            "artifact_id": row.get("artifact_id"),
            "artifact_type": row.get("artifact_type"),
            "chunk_text": row.get("chunk_text"),
            "metadata": self._decode_metadata(row.get("metadata_json")),
            "lexical_score": lexical_score,
            "vector_score": vector_score,
            "score": 0.0,
        }

    def _lexical_score(self, query: str, chunk_text: object) -> float:
        if not isinstance(chunk_text, str):
            return 0.0

        query_tokens = self._tokenize(query)
        document_tokens = self._tokenize(chunk_text)
        if not query_tokens or not document_tokens:
            return 0.0

        query_counts: dict[str, int] = {}
        for token in query_tokens:
            query_counts[token] = query_counts.get(token, 0) + 1

        document_counts: dict[str, int] = {}
        for token in document_tokens:
            document_counts[token] = document_counts.get(token, 0) + 1

        overlap = sum(min(document_counts.get(token, 0), count) for token, count in query_counts.items())
        if overlap == 0:
            return 0.0

        coverage = overlap / max(len(query_tokens), 1)
        density = overlap / max(len(document_tokens), 1)
        phrase_bonus = 0.15 if query.strip().lower() in chunk_text.lower() else 0.0
        return coverage * 0.8 + density * 0.2 + phrase_bonus

    def _final_search_score(self, *, lexical_score: float, vector_score: float, strategy: str) -> float:
        if strategy == "hybrid_hash_lexical":
            return lexical_score * 0.75 + vector_score * 0.25
        if strategy == "lexical":
            return lexical_score
        return vector_score

    def _distance_to_similarity(self, distance: object) -> float:
        if not isinstance(distance, (int, float)):
            return 0.0
        if math.isnan(distance):
            return 0.0
        return 1.0 / (1.0 + max(float(distance), 0.0))

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_./:-]+", text.lower())