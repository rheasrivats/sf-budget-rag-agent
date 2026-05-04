from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from app.config import settings
from app.services.retrieval import EvidenceChunk


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    content: str
    score: float | None = None


def _domain_allowed(url: str, allowed_domains: tuple[str, ...]) -> bool:
    if not allowed_domains:
        return True
    host = urlparse(url).netloc.lower()
    for domain in allowed_domains:
        normalized = domain.lower().lstrip(".")
        if normalized == "gov" and host.endswith(".gov"):
            return True
        if host == normalized or host.endswith(f".{normalized}"):
            return True
    return False


def _query_with_domain_hint(query: str, allowed_domains: tuple[str, ...]) -> str:
    if not allowed_domains:
        return query
    hints = []
    for domain in allowed_domains:
        normalized = domain.lower().lstrip(".")
        if normalized == "gov":
            hints.append("site:.gov")
        else:
            hints.append(f"site:{normalized}")
    return f"{query} ({' OR '.join(hints)})"


def _coerce_results(payload) -> list[WebSearchResult]:
    raw_results = payload
    if isinstance(payload, dict):
        if payload.get("error"):
            raise RuntimeError(f"Tavily search failed: {payload['error']}")
        raw_results = payload.get("results") or payload.get("documents") or []

    results: list[WebSearchResult] = []
    if not isinstance(raw_results, list):
        return results

    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or item.get("snippet") or item.get("raw_content") or "").strip()
        if not url or not content:
            continue
        score = item.get("score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        results.append(
            WebSearchResult(
                title=str(item.get("title") or url).strip(),
                url=url,
                content=content,
                score=score,
            )
        )
    return results


def tavily_search(
    query: str,
    *,
    max_results: int | None = None,
    allowed_domains: tuple[str, ...] | None = None,
) -> list[WebSearchResult]:
    if not settings.tavily_api_key:
        return []

    try:
        from langchain_tavily import TavilySearch
    except ImportError as exc:
        raise RuntimeError("Install the 'langchain-tavily' package to use CRAG web search.") from exc

    max_results = max_results or settings.crag_tavily_max_results
    allowed_domains = allowed_domains if allowed_domains is not None else settings.crag_allowed_domains
    allow_any_gov = any(domain.lower().lstrip(".") == "gov" for domain in allowed_domains)
    exact_domains = () if allow_any_gov else tuple(domain for domain in allowed_domains)
    tool = TavilySearch(
        max_results=max_results,
        topic="general",
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
        include_domains=list(exact_domains),
    )
    payload = tool.invoke({"query": _query_with_domain_hint(query, allowed_domains)})
    return [result for result in _coerce_results(payload) if _domain_allowed(result.url, allowed_domains)][:max_results]


def web_results_to_chunks(results: list[WebSearchResult]) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for idx, result in enumerate(results, start=1):
        score = result.score if result.score is not None else 0.0
        chunks.append(
            EvidenceChunk(
                chunk_id=-idx,
                source_id=-idx,
                score=score,
                content=result.content,
                metadata={
                    "url": result.url,
                    "title": result.title,
                    "content_type": "web",
                    "source_kind": "web",
                    "provider": "tavily",
                    "tavily_score": result.score,
                    "temporary": True,
                },
            )
        )
    return chunks
