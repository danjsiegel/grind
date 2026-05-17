from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request

from grind.config import RetrievalConfig


@dataclass(frozen=True)
class EmbeddingBatchResult:
    vectors: list[list[float]]
    backend: str
    model: str


class ProviderEmbeddingAdapter:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self._remote_disabled = False

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        normalized = [text for text in texts if text.strip()]
        if not normalized:
            return EmbeddingBatchResult(vectors=[], backend="empty", model=self.config.embedding_model)

        if not self._remote_disabled:
            api_key = os.getenv(self.config.embedding_api_key_env)
            if api_key:
                try:
                    return EmbeddingBatchResult(
                        vectors=self._embed_openai(texts=normalized, api_key=api_key),
                        backend=self.config.embedding_provider,
                        model=self.config.embedding_model,
                    )
                except Exception:
                    self._remote_disabled = True
            else:
                self._remote_disabled = True

        if not self.config.allow_local_fallback:
            raise ValueError(
                f"retrieval embeddings require {self.config.embedding_provider} credentials in "
                f"{self.config.embedding_api_key_env}"
            )

        return EmbeddingBatchResult(
            vectors=[self._hash_embed(text) for text in normalized],
            backend="hash-fallback",
            model=self.config.embedding_model,
        )

    def _embed_openai(self, *, texts: list[str], api_key: str) -> list[list[float]]:
        payload = {
            "input": texts,
            "model": self.config.embedding_model,
            "dimensions": self.config.embedding_dimensions,
        }
        request = urllib.request.Request(
            url=f"{self.config.embedding_api_base.rstrip('/')}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("embedding provider returned an invalid response")

        vectors = [item.get("embedding") for item in data if isinstance(item, dict)]
        if len(vectors) != len(texts) or not all(isinstance(vector, list) for vector in vectors):
            raise ValueError("embedding provider returned an incomplete response")
        return vectors

    def _hash_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.config.embedding_dimensions
        tokens = re.findall(r"[A-Za-z0-9_./:-]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, 16, 2):
                bucket = digest[offset] % self.config.embedding_dimensions
                direction = -1.0 if digest[offset + 1] % 2 else 1.0
                vector[bucket] += direction

        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude == 0:
            return vector
        return [component / magnitude for component in vector]