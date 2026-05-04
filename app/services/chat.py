from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ChatMessage, ChatThread, ComparisonRow, PlanVersion
from app.services.adequacy import judge_evidence_adequacy
from app.services.ingest import index_text_document
from app.services.llm import qa_model, should_escalate
from app.services.retrieval import render_evidence_context, rerank_chunks, retrieve_chunks
from app.services.web_search import tavily_search, web_results_to_chunks

MAX_HISTORY_MESSAGES = 12


def _chunks_for_eval(chunks) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "score": chunk.score,
            "content": chunk.content,
            "metadata": dict(chunk.metadata),
        }
        for chunk in chunks
    ]


def _chunks_for_trace(chunks) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "score": chunk.score,
            "source_kind": chunk.metadata.get("source_kind"),
            "title": chunk.metadata.get("title"),
            "url": chunk.metadata.get("url"),
            "retrieval_scores": chunk.metadata.get("retrieval_scores"),
            "content_excerpt": chunk.content[:1200],
        }
        for chunk in chunks
    ]


def _messages_for_trace(messages) -> list[dict[str, Any]]:
    return [
        {
            "type": message.__class__.__name__,
            "content": _to_json_safe(getattr(message, "content", "")),
        }
        for message in messages
    ]


def _query_mentions_generated_plan(query: str) -> bool:
    q = query.lower()
    signals = [
        "our plan",
        "proposed plan",
        "generated plan",
        "this plan",
        "plan version",
        "our proposal",
        "generated proposal",
        "compare to mayor",
        "comparison with mayor",
        "vs mayor",
    ]
    if any(signal in q for signal in signals):
        return True

    # "plan" alone can be ambiguous, so require additional context token.
    if "plan" in q and any(token in q for token in ["our", "generated", "proposed", "version"]):
        return True
    return False


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _to_json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    return str(value)


def _extract_answer_and_debug(ai_message: AIMessage) -> tuple[str, dict[str, Any]]:
    raw_content = _to_json_safe(ai_message.content)
    blocks: list[dict[str, Any]] = []
    reasoning_summaries: list[str] = []
    text_parts: list[str] = []

    if isinstance(raw_content, str):
        text = raw_content.strip()
        if text:
            text_parts.append(text)
        blocks.append({"type": "text", "text": raw_content})
    elif isinstance(raw_content, list):
        for item in raw_content:
            block = _to_json_safe(item)
            if isinstance(block, str):
                if block.strip():
                    text_parts.append(block.strip())
                blocks.append({"type": "text", "text": block})
                continue

            if not isinstance(block, dict):
                blocks.append({"type": "unknown", "value": str(block)})
                continue

            blocks.append(block)
            block_type = str(block.get("type", "")).lower()

            if block_type in {"text", "output_text"}:
                txt = block.get("text")
                if isinstance(txt, str) and txt.strip():
                    text_parts.append(txt.strip())
                continue

            if block_type == "reasoning":
                summary = block.get("summary")
                if isinstance(summary, list):
                    for s in summary:
                        s_safe = _to_json_safe(s)
                        if isinstance(s_safe, dict):
                            txt = s_safe.get("text")
                            if isinstance(txt, str) and txt.strip():
                                reasoning_summaries.append(txt.strip())
                continue

            txt = block.get("text")
            if isinstance(txt, str) and txt.strip():
                text_parts.append(txt.strip())
    else:
        rendered = str(raw_content)
        if rendered.strip():
            text_parts.append(rendered.strip())
        blocks.append({"type": "unknown", "value": rendered})

    answer = "\n\n".join(part for part in text_parts if part)
    if not answer:
        answer = "I couldn't find a clean final answer in the model output. Please try rephrasing the question."

    debug = {
        "response_blocks": blocks,
        "reasoning_summaries": reasoning_summaries,
        "tool_calls": _to_json_safe(getattr(ai_message, "tool_calls", None)) or [],
    }
    return answer, debug


def _load_or_create_thread(db: Session, plan_id: int, thread_id: str) -> ChatThread:
    thread = db.scalar(select(ChatThread).where(ChatThread.thread_id == thread_id))
    if thread is None:
        thread = ChatThread(thread_id=thread_id, plan_id=plan_id)
        db.add(thread)
        db.flush()

    if thread.plan_id != plan_id:
        raise ValueError("Thread is already pinned to another plan version")

    thread.updated_at = datetime.utcnow()
    return thread


def _recent_thread_messages(db: Session, thread_id: str) -> list[ChatMessage]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.thread_id == thread_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(MAX_HISTORY_MESSAGES)
    )
    rows = list(db.scalars(stmt))
    rows.reverse()
    return rows


def _plan_text_for_chat_index(plan: PlanVersion, rows: list[ComparisonRow]) -> str:
    lines = [
        f"Generated Plan Version #{plan.id}",
        f"Scope: {plan.scope_label}",
        f"Created At: {plan.created_at.isoformat()}",
        f"Mayor Total Budget (USD millions): {plan.mayor_total_budget}",
        f"Generated Total Budget (USD millions): {plan.generated_total_budget}",
        "",
        "Plan Memo:",
        plan.memo_markdown,
        "",
        "Comparison Summary:",
        str(plan.metadata_json.get("comparison_summary", {})),
        "",
        "Department Comparison Rows (USD millions):",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"Department={row.department}",
                    f"MayorDir={row.mayor_directional}",
                    f"ProposedDir={row.proposed_directional}",
                    f"DeltaDir={row.delta_directional}",
                    f"MayorFY26_27={row.mayor_fy_2026_27}",
                    f"ProposedFY26_27={row.proposed_fy_2026_27}",
                    f"Rationale={row.rationale}",
                ]
            )
        )
    return "\n".join(lines)


def _ensure_plan_document_indexed(db: Session, plan: PlanVersion) -> None:
    rows = list(
        db.scalars(select(ComparisonRow).where(ComparisonRow.plan_id == plan.id).order_by(ComparisonRow.department.asc()))
    )
    plan_text = _plan_text_for_chat_index(plan, rows)
    index_text_document(
        db,
        url=f"plan://{plan.id}",
        title=f"Generated Plan #{plan.id} ({plan.scope_label})",
        content_type="markdown",
        source_kind="plan",
        text=plan_text,
        metadata={
            "plan_id": plan.id,
            "scope_label": plan.scope_label,
            "created_at": plan.created_at.isoformat(),
            "model_used": plan.model_used,
            "spending_rule_passed": plan.spending_rule_passed,
        },
    )


def _citations_from_chunks(chunks) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        url = chunk.metadata.get("url")
        source_kind = chunk.metadata.get("source_kind", "official")
        plan_id = str(chunk.metadata.get("plan_id", ""))
        key = (url or "", source_kind, plan_id)
        if not url or key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "url": url,
                "label": chunk.metadata.get("title"),
                "source_kind": source_kind,
                "plan_id": chunk.metadata.get("plan_id"),
            }
        )
    return citations


def _top_relevance_score(chunks) -> float:
    if not chunks:
        return 0.0
    scores = chunks[0].metadata.get("retrieval_scores") or {}
    value = scores.get("cohere_relevance", chunks[0].score)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _source_mix(chunks) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        source_kind = str(chunk.metadata.get("source_kind") or "unknown")
        counts[source_kind] = counts.get(source_kind, 0) + 1
    return counts


def answer_question(
    db: Session,
    *,
    plan: PlanVersion,
    thread_id: str,
    query: str,
    escalate: bool = False,
    include_retrieved_context: bool = False,
    trace_collector=None,
    crag_enabled: bool | None = None,
    force_crag: bool = False,
    crag_rerank_threshold: float | None = None,
) -> dict[str, Any]:
    target_started = perf_counter()
    _load_or_create_thread(db, plan.id, thread_id)
    _ensure_plan_document_indexed(db, plan)

    include_plan_sources = _query_mentions_generated_plan(query)
    source_kinds = ["official", "plan"] if include_plan_sources else ["official"]
    retrieval_started = perf_counter()
    evidence_chunks = retrieve_chunks(db, query, source_kinds=source_kinds)
    retrieval_elapsed_ms = round((perf_counter() - retrieval_started) * 1000, 3)
    local_top_score = _top_relevance_score(evidence_chunks)
    crag_effective_enabled = (settings.crag_enabled if crag_enabled is None else crag_enabled) and bool(
        settings.tavily_api_key
    )
    crag_threshold = settings.crag_rerank_threshold if crag_rerank_threshold is None else crag_rerank_threshold
    crag_triggered = False
    crag_trigger_reason = None
    crag_fallback_error = None
    adequacy_check_ran = False
    adequacy_result = None
    adequacy_error = None
    tavily_results_payload: list[dict[str, Any]] = []
    tavily_result_count = 0

    score_triggered = local_top_score < crag_threshold
    if crag_effective_enabled and not force_crag and not score_triggered and settings.crag_adequacy_check_enabled:
        adequacy_check_ran = True
        try:
            adequacy = judge_evidence_adequacy(query, evidence_chunks)
            adequacy_result = adequacy.model_dump(mode="json")
        except Exception as exc:
            adequacy_error = str(exc)

    adequacy_triggered = bool(adequacy_result and adequacy_result.get("should_search_web"))
    if force_crag:
        crag_trigger_reason = "forced"
    elif score_triggered:
        crag_trigger_reason = "low_local_score"
    elif adequacy_triggered:
        crag_trigger_reason = "inadequate_local_evidence"

    if crag_effective_enabled and crag_trigger_reason:
        crag_triggered = True
        try:
            tavily_results = tavily_search(
                query,
                max_results=settings.crag_tavily_max_results,
                allowed_domains=settings.crag_allowed_domains,
            )
            tavily_results_payload = [
                {
                    "title": result.title,
                    "url": result.url,
                    "score": result.score,
                    "content_excerpt": result.content[:1200],
                }
                for result in tavily_results
            ]
            tavily_result_count = len(tavily_results)
            web_chunks = web_results_to_chunks(tavily_results)
            if web_chunks:
                evidence_chunks = rerank_chunks(query, evidence_chunks + web_chunks, final_k=settings.retriever_k)
            else:
                crag_fallback_error = "Tavily returned no usable results."
        except Exception as exc:
            crag_fallback_error = str(exc)

    evidence_context = render_evidence_context(evidence_chunks)
    answer_source_kinds = list(source_kinds)
    if any(chunk.metadata.get("source_kind") == "web" for chunk in evidence_chunks) and "web" not in answer_source_kinds:
        answer_source_kinds.append("web")
    score_preview = [
        {
            "chunk_id": chunk.chunk_id,
            "source_kind": chunk.metadata.get("source_kind"),
            "title": chunk.metadata.get("title"),
            "retrieval_scores": chunk.metadata.get("retrieval_scores"),
        }
        for chunk in evidence_chunks[:5]
    ]
    retrieval_policy = {
        "include_plan_sources": include_plan_sources,
        "allowed_source_kinds": answer_source_kinds,
        "local_source_kinds": source_kinds,
        "reranker": "cohere",
        "rerank_model": settings.rerank_model,
        "rerank_candidate_pool": settings.rerank_candidate_pool,
        "final_k": settings.retriever_k,
        "crag_enabled": crag_effective_enabled,
        "crag_triggered": crag_triggered,
        "crag_threshold": crag_threshold,
        "force_crag": force_crag,
        "crag_trigger_reason": crag_trigger_reason,
        "local_top_score": local_top_score,
        "adequacy_check_enabled": settings.crag_adequacy_check_enabled,
        "adequacy_check_ran": adequacy_check_ran,
        "adequacy_result": adequacy_result,
        "adequacy_error": adequacy_error,
        "tavily_result_count": tavily_result_count,
        "allowed_web_domains": list(settings.crag_allowed_domains),
        "crag_fallback_error": crag_fallback_error,
        "final_source_mix": _source_mix(evidence_chunks),
        "tavily_results": tavily_results_payload,
        "score_preview": score_preview,
    }

    history = _recent_thread_messages(db, thread_id)
    messages = [
        SystemMessage(
            content=(
                "You are an SF budget research assistant. Ground answers only in retrieved evidence "
                "from allowed source kinds. If the question is broader budget context, focus on official evidence. "
                "If uncertain, say so. "
                "Match the user's requested level of detail. "
                "Cite sources using markdown links at the end."
            )
        ),
        SystemMessage(
            content=(
                "Retrieval policy:\n"
                f"- include_plan_sources={include_plan_sources}\n"
                f"- allowed_source_kinds={answer_source_kinds}\n"
                f"- crag_triggered={crag_triggered}\n"
                "Do not use generated-plan claims unless plan sources were included."
            )
        ),
        SystemMessage(
            content=(
                "Current chat thread is pinned to this selected plan version for continuity:\n"
                f"Plan ID: {plan.id}\n"
                f"Plan scope: {plan.scope_label}\n"
                f"Plan summary metadata: {plan.metadata_json.get('comparison_summary', {})}"
            )
        ),
        SystemMessage(content=f"Retrieved evidence:\n{evidence_context}"),
    ]

    for row in history:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content))
        elif row.role == "assistant":
            messages.append(AIMessage(content=row.content))

    messages.append(HumanMessage(content=query))

    should_promote = escalate or should_escalate(query)
    model = qa_model(escalate=should_promote)
    model_started = perf_counter()
    response = model.invoke(messages)
    model_elapsed_ms = round((perf_counter() - model_started) * 1000, 3)
    answer, debug_payload = _extract_answer_and_debug(response)
    debug_payload["retrieval_policy"] = retrieval_policy
    debug_payload["timings_ms"] = {
        "retrieval": retrieval_elapsed_ms,
        "model": model_elapsed_ms,
        "total": round((perf_counter() - target_started) * 1000, 3),
    }

    db.add(ChatMessage(thread_id=thread_id, role="user", content=query, metadata_json={}))
    citations = _citations_from_chunks(evidence_chunks)
    db.add(
        ChatMessage(
            thread_id=thread_id,
            role="assistant",
            content=answer,
            metadata_json={
                "citations": citations,
                "model_escalated": should_promote,
                "include_plan_sources": include_plan_sources,
                "allowed_source_kinds": answer_source_kinds,
                "debug": debug_payload,
            },
        )
    )

    result = {
        "answer": answer,
        "citations": citations,
        "model_used": "gpt-5.4" if should_promote else "gpt-5.4-mini",
        "debug": debug_payload,
    }
    if include_retrieved_context:
        result["retrieved_context"] = _chunks_for_eval(evidence_chunks)

    if trace_collector is not None:
        trace_collector.record(
            "target_call",
            {
                "query": query,
                "thread_id": thread_id,
                "escalate": escalate,
                "selected_model": result["model_used"],
                "elapsed_ms": debug_payload["timings_ms"]["total"],
            },
        )
        trace_collector.record(
            "retrieval",
            {
                "policy": retrieval_policy,
                "elapsed_ms": retrieval_elapsed_ms,
                "chunks": _chunks_for_trace(evidence_chunks),
            },
        )
        trace_collector.record(
            "crag",
            {
                "enabled": crag_effective_enabled,
                "triggered": crag_triggered,
                "threshold": crag_threshold,
                "forced": force_crag,
                "trigger_reason": crag_trigger_reason,
                "local_top_score": local_top_score,
                "adequacy_check_ran": adequacy_check_ran,
                "adequacy_result": adequacy_result,
                "adequacy_error": adequacy_error,
                "tavily_result_count": tavily_result_count,
                "allowed_web_domains": list(settings.crag_allowed_domains),
                "fallback_error": crag_fallback_error,
                "final_source_mix": _source_mix(evidence_chunks),
                "tavily_results": tavily_results_payload,
            },
        )
        trace_collector.record("messages", _messages_for_trace(messages))
        trace_collector.record(
            "model_response",
            {
                "answer": answer,
                "citations": citations,
                "response_blocks": debug_payload.get("response_blocks", []),
                "reasoning_summaries": debug_payload.get("reasoning_summaries", []),
                "tool_calls": debug_payload.get("tool_calls", []),
                "elapsed_ms": model_elapsed_ms,
            },
        )

    return result
