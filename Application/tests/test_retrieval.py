import unittest

import numpy as np

from app.services.rag import Chunk, RetrievalBackend


class FakeEncoder:
    def encode(self, texts):
        return np.array([
            np.array([0.0, 1.0], dtype="float32") if "beta" in text else np.array([1.0, 0.0], dtype="float32")
            for text in texts
        ])


class FakeReranker:
    def rerank(self, query, texts):
        del query
        return list(reversed(range(len(texts))))


def make_backend():
    backend = RetrievalBackend()
    backend.encoder = FakeEncoder()
    backend.reranker = FakeReranker()
    backend.add_chunks([
        Chunk("a", "doc-a", "a.pdf", 1, "alpha lexical evidence"),
        Chunk("b", "doc-b", "b.pdf", 2, "beta semantic evidence"),
    ])
    return backend


class RetrievalTests(unittest.TestCase):
    def test_bm25_retrieves_known_lexical_evidence(self):
        results, _ = make_backend().search("alpha", "bm25", 1)
        self.assertEqual(results[0]["chunk_id"], "a")

    def test_dense_retrieval_uses_encoder(self):
        results, _ = make_backend().search("question alpha", "dense", 1)
        self.assertEqual(results[0]["chunk_id"], "a")

    def test_hybrid_returns_union_of_lexical_and_dense_candidates(self):
        results, _ = make_backend().search("alpha", "hybrid", 2)
        self.assertEqual({item["chunk_id"] for item in results}, {"a", "b"})

    def test_rerank_reorders_hybrid_candidates(self):
        results, _ = make_backend().search("alpha", "rerank", 2)
        self.assertEqual(results[0]["chunk_id"], "b")

    def test_rerank_without_adapter_is_explicitly_unavailable(self):
        backend = make_backend()
        backend.reranker = None
        with self.assertRaisesRegex(RuntimeError, "reranker is not configured"):
            backend.search("alpha", "rerank", 1)


if __name__ == "__main__":
    unittest.main()
