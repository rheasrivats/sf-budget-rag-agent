from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings
from app.services.llm import qa_model
from app.services.retrieval import EvidenceChunk


class EvidenceAdequacy(BaseModel):
    sufficient: bool
    should_search_web: bool
    missing_facts: list[str] = Field(default_factory=list)
    rationale: str


def _context_for_adequacy(chunks: list[EvidenceChunk]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks[: settings.crag_adequacy_max_chunks], start=1):
        metadata = chunk.metadata or {}
        parts.append(
            "\n".join(
                [
                    f"[{idx}] Source: {metadata.get('url', 'unknown')}",
                    f"Title: {metadata.get('title', '')}",
                    f"Content: {chunk.content[: settings.crag_adequacy_chunk_chars]}",
                ]
            )
        )
    return "\n\n".join(parts)


def judge_evidence_adequacy(query: str, chunks: list[EvidenceChunk]) -> EvidenceAdequacy:
    if not chunks:
        return EvidenceAdequacy(
            sufficient=False,
            should_search_web=True,
            missing_facts=["No local evidence was retrieved."],
            rationale="No local evidence was retrieved.",
        )

    prompt = (
        "You are deciding whether retrieved local evidence is sufficient to answer a user's question. "
        "Return whether the evidence contains enough information to answer every requested part. "
        "If the question asks for multiple entities, rates, dates, totals, or changes, every requested slot must be present. "
        "Set should_search_web=true when any important requested fact is missing, ambiguous, stale, or only partially supported. "
        "Do not answer the user. Judge evidence sufficiency only.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved local evidence:\n{_context_for_adequacy(chunks)}"
    )
    model = qa_model(escalate=False).with_structured_output(EvidenceAdequacy, method="function_calling")
    return model.invoke(prompt)
