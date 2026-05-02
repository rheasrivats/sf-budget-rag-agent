from types import SimpleNamespace

from app.services.retrieval import EvidenceChunk, _apply_cohere_rerank


def test_cohere_rerank_uses_response_order_and_scores(monkeypatch):
    import app.services.retrieval as retrieval

    class _FakeClient:
        def rerank(self, *, model, query, documents, top_n):
            assert model == "rerank-test"
            assert query == "budget deficit"
            assert documents == ["less relevant", "most relevant", "other"]
            assert top_n == 2
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=1, relevance_score=0.91),
                    SimpleNamespace(index=0, relevance_score=0.17),
                ]
            )

    monkeypatch.setattr(
        retrieval,
        "settings",
        SimpleNamespace(cohere_api_key="test", rerank_model="rerank-test"),
    )
    monkeypatch.setattr(retrieval, "_cohere_client", lambda: _FakeClient())

    chunks = [
        EvidenceChunk(chunk_id=1, source_id=10, score=0.95, content="less relevant", metadata={}),
        EvidenceChunk(chunk_id=2, source_id=11, score=0.75, content="most relevant", metadata={}),
        EvidenceChunk(chunk_id=3, source_id=12, score=0.70, content="other", metadata={}),
    ]

    ranked = _apply_cohere_rerank("budget deficit", chunks, final_k=2)

    assert [chunk.chunk_id for chunk in ranked] == [2, 1]
    assert ranked[0].score == 0.91
    assert ranked[0].metadata["retrieval_scores"] == {
        "semantic": 0.75,
        "cohere_relevance": 0.91,
        "final": 0.91,
        "reranker": "rerank-test",
    }
