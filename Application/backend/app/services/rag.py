from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    id: str
    document_id: str
    filename: str
    page: int
    text: str


class DenseEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.last_scores: list[float] = []
        self.last_scores_by_candidate: dict[int, float] = {}

    def rerank(self, query: str, texts: list[str]) -> list[int]:
        scores = self.model.predict([(query, text) for text in texts])
        self.last_scores = np.asarray(scores, dtype="float32").tolist()
        return np.argsort(np.asarray(scores))[-len(texts) :][::-1].tolist()


class RetrievalBackend:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self._dense: np.ndarray | None = None
        self.encoder: DenseEncoder | None = None
        self.reranker: CrossEncoderReranker | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])
        self._dense = None

    def delete_document(self, document_id: str) -> int:
        before = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document_id]
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks]) if self.chunks else None
        self._dense = None
        return before - len(self.chunks)

    def _ensure_dense(self) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("Dense retrieval is not configured")
        if self._dense is None:
            self._dense = self.encoder.encode([chunk.text for chunk in self.chunks])
        return self._dense

    def _bm25(self, query: str, limit: int) -> list[int]:
        if self.bm25 is None:
            return []
        query_tokens = set(tokenize(query))
        scores = self.bm25.get_scores(list(query_tokens))
        overlaps = [len(query_tokens.intersection(tokenize(chunk.text))) for chunk in self.chunks]
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda index: (float(scores[index]), overlaps[index]),
            reverse=True,
        )
        return ranked[:limit]

    def _dense_search(self, query: str, limit: int) -> list[int]:
        matrix = self._ensure_dense()
        query_vector = self.encoder.encode([query])[0]
        scores = matrix @ query_vector
        return np.argsort(scores)[-limit:][::-1].tolist()

    def search(self, query: str, method: str = "hybrid", limit: int = 5) -> tuple[list[dict], float]:
        started = time.perf_counter()
        if not self.chunks:
            return [], 0.0
        candidate_limit = max(limit, 20)
        bm25_ids = self._bm25(query, candidate_limit)
        query_tokens = set(tokenize(query))
        bm25_scores = (
            self.bm25.get_scores(list(query_tokens)) if self.bm25 is not None else np.zeros(len(self.chunks))
        )
        if method == "bm25":
            ranked = bm25_ids
            score_map = {index: float(bm25_scores[index]) for index in bm25_ids}
        else:
            dense_ids = self._dense_search(query, candidate_limit)
            dense_matrix = self._ensure_dense()
            dense_scores = dense_matrix @ self.encoder.encode([query])[0]
            if method == "dense":
                ranked = dense_ids
                score_map = {index: float(dense_scores[index]) for index in dense_ids}
            else:
                scores: dict[int, float] = {}
                for ranking in (bm25_ids, dense_ids):
                    for rank, index in enumerate(ranking, 1):
                        scores[index] = scores.get(index, 0.0) + 1.0 / (60 + rank)
                ranked = sorted(scores, key=scores.get, reverse=True)
                score_map = scores
                if method == "rerank":
                    ranked = self._rerank(query, ranked)
                    if hasattr(self.reranker, "last_scores_by_candidate"):
                        score_map = self.reranker.last_scores_by_candidate
        results = [
            {
                "chunk_id": self.chunks[index].id,
                "document_id": self.chunks[index].document_id,
                "filename": self.chunks[index].filename,
                "page": self.chunks[index].page,
                "text": self.chunks[index].text,
                "rank": rank,
                "score": score_map.get(index, 0.0),
            }
            for rank, index in enumerate(ranked[:limit], 1)
        ]
        return results, (time.perf_counter() - started) * 1000

    def _rerank(self, query: str, candidate_ids: list[int]) -> list[int]:
        if self.reranker is None:
            raise RuntimeError("Cross-encoder reranker is not configured")
        texts = [self.chunks[index].text for index in candidate_ids]
        order = self.reranker.rerank(query, texts)
        if hasattr(self.reranker, "last_scores"):
            self.reranker.last_scores_by_candidate = {
                candidate_ids[position]: float(self.reranker.last_scores[position])
                for position in range(len(candidate_ids))
            }
        return [candidate_ids[index] for index in order]
