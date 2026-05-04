from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import DocumentChunk, SourceDocument
from app.services import ingest
from app.services.retrieval import retrieve_chunks


def _make_local_db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine, future=True)


def test_collect_raw_documents_uses_html_seed_pages_for_discovery_only(monkeypatch):
    monkeypatch.setattr(
        ingest,
        "_extract_html",
        lambda _url: ("Budget", "Budget process text " * 30, []),
    )

    docs = ingest.collect_raw_documents(["https://sf.gov/topics/budget"])

    assert docs == []


def test_run_ingestion_deactivates_legacy_official_html_sources(monkeypatch):
    engine, Local = _make_local_db()
    monkeypatch.setattr(ingest, "collect_raw_documents", lambda _urls: [])

    with Local() as db:
        source = SourceDocument(
            url="https://sf.gov/topics/budget",
            title="Budget",
            content_type="html",
            source_kind="official",
            active=True,
        )
        db.add(source)
        db.flush()
        db.add(
            DocumentChunk(
                source_id=source.id,
                chunk_index=0,
                content="HTML page text",
                embedding_json=[1.0],
                metadata_json={"url": source.url, "source_kind": "official"},
            )
        )
        db.commit()

        result = ingest.run_ingestion(db)
        db.commit()

        assert result["updated_sources"] == 1
        assert source.active is False
        assert db.query(DocumentChunk).filter(DocumentChunk.source_id == source.id).count() == 0

    engine.dispose()


def test_retrieve_chunks_ignores_inactive_sources(monkeypatch):
    engine, Local = _make_local_db()

    class _Embeddings:
        def embed_query(self, _query):
            return [1.0]

    monkeypatch.setattr("app.services.retrieval.create_embeddings", lambda model: _Embeddings())
    monkeypatch.setattr(
        "app.services.retrieval._apply_cohere_rerank",
        lambda _query, chunks, final_k: chunks[:final_k],
    )

    with Local() as db:
        inactive = SourceDocument(
            url="https://sf.gov/topics/budget",
            title="Budget",
            content_type="html",
            source_kind="official",
            active=False,
        )
        active = SourceDocument(
            url="https://sf.gov/documents/budget.pdf",
            title="Budget PDF",
            content_type="pdf",
            source_kind="official",
            active=True,
        )
        db.add_all([inactive, active])
        db.flush()
        db.add_all(
            [
                DocumentChunk(
                    source_id=inactive.id,
                    chunk_index=0,
                    content="inactive html",
                    embedding_json=[1.0],
                    metadata_json={"url": inactive.url, "source_kind": "official"},
                ),
                DocumentChunk(
                    source_id=active.id,
                    chunk_index=0,
                    content="active pdf",
                    embedding_json=[1.0],
                    metadata_json={"url": active.url, "source_kind": "official"},
                ),
            ]
        )
        db.commit()

        chunks = retrieve_chunks(db, "budget", k=5, source_kinds=["official"])

        assert [chunk.source_id for chunk in chunks] == [active.id]

    engine.dispose()


def test_retrieve_chunks_falls_back_to_source_kind_from_source_row(monkeypatch):
    engine, Local = _make_local_db()

    class _Embeddings:
        def embed_query(self, _query):
            return [1.0]

    import app.services.retrieval as retrieval

    monkeypatch.setattr(retrieval, "create_embeddings", lambda model: _Embeddings())
    monkeypatch.setattr(retrieval, "_apply_cohere_rerank", lambda _query, chunks, final_k: chunks[:final_k])

    with Local() as db:
        source = SourceDocument(
            url="https://sf.gov/documents/budget.pdf",
            title="Budget PDF",
            content_type="pdf",
            source_kind="official",
            active=True,
        )
        db.add(source)
        db.flush()
        db.add(
            DocumentChunk(
                source_id=source.id,
                chunk_index=0,
                content="budget",
                embedding_json=[1.0],
                metadata_json={"url": source.url},
            )
        )
        db.commit()

        chunks = retrieve_chunks(db, "budget", k=1, source_kinds=["official"])

        assert chunks[0].metadata["source_kind"] == "official"
        assert chunks[0].metadata["title"] == "Budget PDF"

    engine.dispose()


def test_retrieve_chunks_sends_top_8_candidates_to_reranker_and_returns_top_8(monkeypatch):
    engine, Local = _make_local_db()

    class _Embeddings:
        def embed_query(self, _query):
            return [1.0, 0.0]

    import app.services.retrieval as retrieval

    monkeypatch.setattr(retrieval, "create_embeddings", lambda model: _Embeddings())
    monkeypatch.setattr(
        retrieval,
        "settings",
        SimpleNamespace(retriever_k=8, embedding_model="test", rerank_candidate_pool=8),
    )

    seen: dict[str, int] = {}

    def fake_rerank(_query, chunks, final_k):
        seen["candidate_count"] = len(chunks)
        seen["final_k"] = final_k
        return chunks[:final_k]

    monkeypatch.setattr(retrieval, "_apply_cohere_rerank", fake_rerank)

    with Local() as db:
        source = SourceDocument(
            url="https://sf.gov/documents/budget.pdf",
            title="Budget PDF",
            content_type="pdf",
            source_kind="official",
            active=True,
        )
        db.add(source)
        db.flush()
        for idx in range(12):
            db.add(
                DocumentChunk(
                    source_id=source.id,
                    chunk_index=idx,
                    content=f"chunk {idx}",
                    embedding_json=[float(20 - idx), 1.0],
                    metadata_json={"url": source.url, "source_kind": "official"},
                )
            )
        db.commit()

        chunks = retrieve_chunks(db, "budget", source_kinds=["official"])

        assert seen == {"candidate_count": 8, "final_k": 8}
        assert len(chunks) == 8

    engine.dispose()
