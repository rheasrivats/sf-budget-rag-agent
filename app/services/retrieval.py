from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DocumentChunk, SourceDocument
from rag_embeddings_store import create_embeddings

DEFAULT_RERANK_CANDIDATE_POOL = 8


@dataclass
class EvidenceChunk:
    chunk_id: int
    source_id: int
    score: float
    content: str
    metadata: dict


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _cohere_client():
    if not settings.cohere_api_key:
        raise RuntimeError("COHERE_API_KEY is required because retrieval reranking uses Cohere.")
    try:
        import cohere
    except ImportError as exc:
        raise RuntimeError("Install the 'cohere' package to use retrieval reranking.") from exc
    return cohere.ClientV2(api_key=settings.cohere_api_key)


def _apply_cohere_rerank(query: str, chunks: list[EvidenceChunk], final_k: int) -> list[EvidenceChunk]:
    if not chunks:
        return []

    top_n = min(final_k, len(chunks))
    documents = [chunk.content for chunk in chunks]
    response = _cohere_client().rerank(
        model=settings.rerank_model,
        query=query,
        documents=documents,
        top_n=top_n,
    )

    reranked: list[EvidenceChunk] = []
    for result in response.results:
        chunk = chunks[result.index]
        semantic = chunk.score
        relevance = float(result.relevance_score)
        chunk.score = relevance
        chunk.metadata["retrieval_scores"] = {
            "semantic": round(float(semantic), 6),
            "cohere_relevance": round(relevance, 6),
            "final": round(relevance, 6),
            "reranker": settings.rerank_model,
        }
        reranked.append(chunk)
    return reranked


def retrieve_chunks(
    db: Session,
    query: str,
    k: int | None = None,
    source_ids: Iterable[int] | None = None,
    source_kinds: Iterable[str] | None = None,
) -> list[EvidenceChunk]:
    if not query.strip():
        return []

    k = k or settings.retriever_k

    stmt = (
        select(DocumentChunk)
        .join(SourceDocument, DocumentChunk.source_id == SourceDocument.id)
        .where(SourceDocument.active.is_(True))
    )
    if source_kinds:
        stmt = stmt.where(SourceDocument.source_kind.in_(list(source_kinds)))

    if source_ids:
        stmt = stmt.where(DocumentChunk.source_id.in_(list(source_ids)))

    chunks = list(db.scalars(stmt))
    if not chunks:
        return []

    embeddings = create_embeddings(model=settings.embedding_model)
    q_vec = np.array(embeddings.embed_query(query), dtype=np.float32)

    scored: list[EvidenceChunk] = []
    for chunk in chunks:
        c_vec = np.array(chunk.embedding_json, dtype=np.float32)
        score = _cosine(q_vec, c_vec)
        scored.append(
            EvidenceChunk(
                chunk_id=chunk.id,
                source_id=chunk.source_id,
                score=score,
                content=chunk.content,
                metadata=dict(chunk.metadata_json),
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    # Default Q&A retrieval uses 8 semantic candidates, then returns all 8
    # after Cohere reranking. Larger explicit `k` calls keep enough candidates
    # to satisfy the caller rather than truncating below the requested result count.
    candidate_pool = max(k, settings.rerank_candidate_pool or DEFAULT_RERANK_CANDIDATE_POOL)
    return _apply_cohere_rerank(query, scored[:candidate_pool], k)


def render_evidence_context(chunks: list[EvidenceChunk]) -> str:
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source_url = chunk.metadata.get("url", "unknown")
        title = chunk.metadata.get("title", "")
        source_kind = chunk.metadata.get("source_kind", "official")
        plan_id = chunk.metadata.get("plan_id")
        plan_suffix = f" | Plan ID: {plan_id}" if plan_id is not None else ""
        lines.append(
            f"[{idx}] Source: {source_url}\n"
            f"Source Kind: {source_kind}{plan_suffix}\n"
            f"Title: {title}\n"
            f"Evidence: {chunk.content}"
        )
    return "\n\n".join(lines)
