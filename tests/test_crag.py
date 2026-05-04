from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import PlanVersion
from app.services.retrieval import EvidenceChunk
from app.services.web_search import WebSearchResult, _coerce_results, web_results_to_chunks


def _make_local_db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine, future=True)


def _settings(*, api_key: str = "tvly-test", threshold: float = 0.35):
    return SimpleNamespace(
        crag_enabled=True,
        tavily_api_key=api_key,
        crag_rerank_threshold=threshold,
        crag_adequacy_check_enabled=True,
        crag_adequacy_max_chunks=5,
        crag_adequacy_chunk_chars=700,
        crag_tavily_max_results=5,
        crag_allowed_domains=("sf.gov", "media.api.sf.gov", "gov"),
        retriever_k=8,
        rerank_model="rerank-test",
        rerank_candidate_pool=8,
    )


def _chunk(chunk_id: int, score: float, *, source_kind: str = "official", url: str = "https://sf.gov/a"):
    return EvidenceChunk(
        chunk_id=chunk_id,
        source_id=chunk_id,
        score=score,
        content=f"content {chunk_id}",
        metadata={
            "url": url,
            "title": f"Doc {chunk_id}",
            "source_kind": source_kind,
            "retrieval_scores": {"cohere_relevance": score},
        },
    )


class _FakeModel:
    def invoke(self, _messages):
        return AIMessage(content="Answer from evidence.")


def _run_answer(
    monkeypatch,
    *,
    local_score: float,
    api_key: str = "tvly-test",
    tavily_error: Exception | None = None,
    force_crag: bool = False,
    query: str = "question",
    adequacy_should_search: bool = False,
):
    import app.services.chat as chat

    engine, Local = _make_local_db()
    monkeypatch.setattr(chat, "settings", _settings(api_key=api_key))
    monkeypatch.setattr(chat, "_ensure_plan_document_indexed", lambda _db, _plan: None)
    monkeypatch.setattr(chat, "qa_model", lambda escalate=False: _FakeModel())
    monkeypatch.setattr(chat, "should_escalate", lambda _query: False)
    monkeypatch.setattr(chat, "retrieve_chunks", lambda *_args, **_kwargs: [_chunk(1, local_score)])
    monkeypatch.setattr(
        chat,
        "judge_evidence_adequacy",
        lambda _query, _chunks: SimpleNamespace(
            model_dump=lambda mode="json": {
                "sufficient": not adequacy_should_search,
                "should_search_web": adequacy_should_search,
                "missing_facts": ["for-profit rate"] if adequacy_should_search else [],
                "rationale": "stub",
            }
        ),
    )

    tavily_calls = {"count": 0}

    def fake_tavily(*_args, **_kwargs):
        tavily_calls["count"] += 1
        if tavily_error:
            raise tavily_error
        return [WebSearchResult(title="Web", url="https://controller.gov/budget", content="web budget", score=0.8)]

    def fake_rerank(_query, chunks, final_k):
        assert len(chunks) == 2
        web = [chunk for chunk in chunks if chunk.metadata.get("source_kind") == "web"]
        local = [chunk for chunk in chunks if chunk.metadata.get("source_kind") == "official"]
        return (web + local)[:final_k]

    monkeypatch.setattr(chat, "tavily_search", fake_tavily)
    monkeypatch.setattr(chat, "rerank_chunks", fake_rerank)

    with Local() as db:
        plan = PlanVersion(memo_markdown="memo", model_used="test")
        db.add(plan)
        db.commit()
        result = chat.answer_question(
            db,
            plan=plan,
            thread_id="t",
            query=query,
            include_retrieved_context=True,
            force_crag=force_crag,
        )

    engine.dispose()
    return result, tavily_calls["count"]


def test_crag_triggers_below_threshold_and_reranks_web_with_local(monkeypatch):
    result, tavily_calls = _run_answer(monkeypatch, local_score=0.1)

    policy = result["debug"]["retrieval_policy"]
    assert tavily_calls == 1
    assert policy["crag_triggered"] is True
    assert policy["tavily_result_count"] == 1
    assert policy["final_source_mix"] == {"web": 1, "official": 1}
    assert result["retrieved_context"][0]["metadata"]["source_kind"] == "web"
    assert result["citations"][0]["source_kind"] == "web"


def test_crag_does_not_trigger_above_threshold(monkeypatch):
    result, tavily_calls = _run_answer(monkeypatch, local_score=0.9)

    policy = result["debug"]["retrieval_policy"]
    assert tavily_calls == 0
    assert policy["crag_triggered"] is False
    assert policy["adequacy_check_ran"] is True
    assert policy["final_source_mix"] == {"official": 1}


def test_crag_triggers_on_adequacy_failure_even_above_threshold(monkeypatch):
    result, tavily_calls = _run_answer(
        monkeypatch,
        local_score=0.9,
        query="What are the for-profit, non-profit, and public entity rates?",
        adequacy_should_search=True,
    )

    policy = result["debug"]["retrieval_policy"]
    assert tavily_calls == 1
    assert policy["crag_triggered"] is True
    assert policy["crag_trigger_reason"] == "inadequate_local_evidence"
    assert policy["adequacy_check_ran"] is True
    assert policy["adequacy_result"]["missing_facts"] == ["for-profit rate"]


def test_force_crag_triggers_even_above_threshold(monkeypatch):
    result, tavily_calls = _run_answer(monkeypatch, local_score=0.9, force_crag=True)

    policy = result["debug"]["retrieval_policy"]
    assert tavily_calls == 1
    assert policy["crag_triggered"] is True
    assert policy["force_crag"] is True


def test_crag_does_not_trigger_without_tavily_key(monkeypatch):
    result, tavily_calls = _run_answer(monkeypatch, local_score=0.1, api_key="")

    policy = result["debug"]["retrieval_policy"]
    assert tavily_calls == 0
    assert policy["crag_enabled"] is False
    assert policy["crag_triggered"] is False


def test_crag_falls_back_to_local_when_tavily_fails(monkeypatch):
    result, tavily_calls = _run_answer(monkeypatch, local_score=0.1, tavily_error=RuntimeError("search down"))

    policy = result["debug"]["retrieval_policy"]
    assert tavily_calls == 1
    assert policy["crag_triggered"] is True
    assert policy["crag_fallback_error"] == "search down"
    assert policy["final_source_mix"] == {"official": 1}


def test_tavily_results_normalize_to_temporary_web_chunks():
    chunks = web_results_to_chunks(
        [WebSearchResult(title="Budget", url="https://controller.gov/budget", content="Budget text", score=0.7)]
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == -1
    assert chunks[0].source_id == -1
    assert chunks[0].score == 0.7
    assert chunks[0].metadata["source_kind"] == "web"
    assert chunks[0].metadata["provider"] == "tavily"


def test_tavily_error_payload_is_not_treated_as_empty_results():
    try:
        _coerce_results({"error": RuntimeError("dns failed")})
    except RuntimeError as exc:
        assert "Tavily search failed" in str(exc)
    else:
        raise AssertionError("Expected Tavily error payload to raise RuntimeError")
