from __future__ import annotations

import re
from typing import Any

from evals.types import DeterministicGrade, QACase

CLAIM_PATTERNS = [
    re.compile(r"\$?\s?\d[\d,]*(?:\.\d+)?\s?(?:million|billion|m|b)\b", re.IGNORECASE),
    re.compile(r"\d[\d,]*(?:\.\d+)?\s?(?:%|\s?percent)(?=$|\s|[,.);:])", re.IGNORECASE),
    re.compile(r"\bFY\s?\d{2,4}[-–]\d{2,4}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b",
        re.IGNORECASE,
    ),
]
CITATION_RE = re.compile(r"(https?://\S+|\[[^\]]+\]\([^)]+\)|\[\d+\])")
SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")

GRADER_DESCRIPTIONS = {
    "answer_nonempty": "Checks that the agent returned a non-empty answer.",
    "source_kind_policy": "Checks that retrieval used the source kinds expected by the case.",
    "no_plan_sources": "Checks that official-only cases did not use generated-plan citations or chunks.",
    "citation_urls": "Scores recall of required citation URLs in the agent's returned citations.",
    "retrieval_recall_at_k": "Scores whether gold relevant docs appear anywhere in the top-k retrieval results.",
    "retrieval_precision_at_k": "Scores what fraction of top-k retrieval results came from gold relevant docs.",
    "retrieval_mrr": "Scores how early the first gold relevant doc appears in the retrieval ranking.",
    "citation_validity": "Scores whether returned citations point to documents that were actually retrieved.",
    "citation_coverage": "Scores whether factual numeric/date claim sentences include citations.",
    "must_not_include": "Checks that forbidden literal text is absent from the answer.",
    "must_not_match": "Checks that forbidden regex patterns are absent from the answer.",
    "numeric_date_claims_supported": "Warns whether numeric/date claims in the answer appear in retrieved context.",
    "retrieval_nonempty": "Checks that the agent returned retrieved context unless the case permits empty retrieval.",
}
NON_BLOCKING_GRADERS = {"numeric_date_claims_supported"}


def _grade(**kwargs: Any) -> DeterministicGrade:
    """Create a deterministic grade with the trace-facing short description attached."""
    name = str(kwargs["name"])
    kwargs.setdefault("description", GRADER_DESCRIPTIONS.get(name, ""))
    kwargs.setdefault("blocking", name not in NON_BLOCKING_GRADERS)
    return DeterministicGrade(**kwargs)


def _answer_text(prediction: dict[str, Any]) -> str:
    return str(prediction.get("answer") or "")


def _observed_source_kinds(prediction: dict[str, Any]) -> list[str]:
    policy = (prediction.get("debug") or {}).get("retrieval_policy") or {}
    return list(policy.get("allowed_source_kinds") or [])


def _citation_urls(prediction: dict[str, Any]) -> list[str]:
    return [str(c.get("url")) for c in prediction.get("citations", []) if c.get("url")]


def _retrieved_context(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    return list(prediction.get("retrieved_context") or [])


def _retrieved_source_ids(prediction: dict[str, Any], k: int | None = None) -> list[int]:
    items = _retrieved_context(prediction)
    if k is not None:
        items = items[:k]
    source_ids: list[int] = []
    for item in items:
        source_id = item.get("source_id")
        if source_id is None:
            source_id = item.get("metadata", {}).get("source_id")
        try:
            source_ids.append(int(source_id))
        except (TypeError, ValueError):
            continue
    return source_ids


def _retrieved_urls(prediction: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    for item in _retrieved_context(prediction):
        url = item.get("metadata", {}).get("url")
        if url:
            urls.add(str(url))
    return urls


def _context_text(prediction: dict[str, Any]) -> str:
    return "\n\n".join(str(item.get("content") or "") for item in _retrieved_context(prediction))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _canonical_claim(claim: str) -> str:
    value = _normalize_text(claim)
    value = value.replace("–", "-")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\bfy\s+", "fy ", value)
    value = re.sub(r"\bfy\s?(\d)", r"fy \1", value)
    value = re.sub(r"\s*%\b", "%", value)
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\bm\b", "million", value)
    value = re.sub(r"\bb\b", "billion", value)
    return value


def extract_numeric_date_claims(text: str) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()
    for pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            claim = _canonical_claim(match.group(0))
            if claim and claim not in seen:
                seen.add(claim)
                claims.append(claim)
    return claims


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def _factual_claim_sentences(answer: str) -> list[str]:
    claims: list[str] = []
    for match in SENTENCE_RE.finditer(answer):
        sentence = match.group(0).strip()
        if sentence and extract_numeric_date_claims(sentence):
            claims.append(sentence)
    return claims


def grade_answer_nonempty(_case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade that the target returned usable answer text.

    Inputs:
    - `case` is intentionally unused because this is a universal sanity check.
    - `prediction["answer"]` is read and stripped.

    Output:
    - Passes with score 1.0 when answer text is non-empty.
    - Fails with score 0.0 when the answer is empty or missing.
    - `observed` is the answer character count after stripping.
    """
    answer = _answer_text(prediction).strip()
    passed = bool(answer)
    return _grade(
        name="answer_nonempty",
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation="Answer contains text." if passed else "Answer is empty.",
        observed=len(answer),
    )


def grade_source_kind_policy(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade that retrieval followed the case's allowed source-kind policy.

    Inputs:
    - `case.expected_source_kinds` lists the source kinds the case expects, usually `["official"]`.
    - `prediction["debug"]["retrieval_policy"]["allowed_source_kinds"]` is the policy used by the target.

    Output:
    - Passes with score 1.0 only when observed source kinds exactly match the expected list.
    - Fails with score 0.0 on policy drift, such as allowing generated plan sources in an official-only case.
    - `expected` and `observed` store the two source-kind lists for trace debugging.
    """
    observed = _observed_source_kinds(prediction)
    expected = case.expected_source_kinds
    return _grade(
        name="source_kind_policy",
        passed=observed == expected,
        score=1.0 if observed == expected else 0.0,
        explanation=f"Observed allowed source kinds: {observed}",
        expected=expected,
        observed=observed,
    )


def grade_no_plan_sources(_case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade that generated-plan evidence was not used.

    Inputs:
    - `prediction["citations"]` is scanned for `plan://` URLs.
    - `prediction["retrieved_context"]` is scanned for chunks whose metadata source kind is `plan`
      or whose URL starts with `plan://`.

    Output:
    - Passes with score 1.0 when no generated-plan citations or chunks are present.
    - Fails with score 0.0 when any generated-plan evidence appears.
    - `observed` lists offending plan citations and counts offending retrieved chunks.
    """
    urls = _citation_urls(prediction)
    retrieved = _retrieved_context(prediction)
    plan_citations = [url for url in urls if url.startswith("plan://")]
    plan_chunks = [
        item
        for item in retrieved
        if item.get("metadata", {}).get("source_kind") == "plan"
        or str(item.get("metadata", {}).get("url", "")).startswith("plan://")
    ]
    passed = not plan_citations and not plan_chunks
    return _grade(
        name="no_plan_sources",
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation="No plan sources observed." if passed else "Plan sources were observed.",
        observed={"plan_citations": plan_citations, "plan_chunk_count": len(plan_chunks)},
    )


def grade_citation_urls(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade recall of case-required citation URLs.

    Inputs:
    - `case.expected_citation_urls` contains canonical URLs the answer should cite.
    - `prediction["citations"]` contains URLs returned by the agent.

    Output:
    - Score is required citation URL recall: matched required URLs divided by required URLs.
    - Passes only when every required URL is present.
    - If the case has no required citation URLs, the score is 1.0 and the check passes.
    - `observed` records all returned citation URLs, missing required URLs, and the recall score.
    """
    observed = _citation_urls(prediction)
    missing = [url for url in case.expected_citation_urls if url not in observed]
    if not case.expected_citation_urls:
        score = 1.0
    else:
        score = (len(case.expected_citation_urls) - len(missing)) / len(case.expected_citation_urls)
    return _grade(
        name="citation_urls",
        passed=not missing,
        score=score,
        explanation="Required citation URLs are present." if not missing else f"Missing citations: {missing}",
        expected=case.expected_citation_urls,
        observed={"citation_urls": observed, "missing": missing, "citation_url_recall": score},
    )


def grade_retrieval_recall_at_k(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade whether gold documents appeared anywhere in top-k retrieval.

    Inputs:
    - `case.relevant_doc_ids` is the labeled set of gold source document IDs.
    - `case.retrieval_k` defines the prefix of retrieved chunks to inspect.
    - `case.min_recall_at_k` is the pass threshold.
    - `prediction["retrieved_context"]` supplies ranked retrieved chunks with source IDs.

    Output:
    - Score is unique-document recall: gold docs found in top-k divided by total gold docs.
    - Passes when score is at least `case.min_recall_at_k`.
    - If no gold docs are configured, this grader is informational and returns score `None`.
    - `observed` records retrieved doc IDs, gold hits, and recall.
    """
    if not case.relevant_doc_ids:
        return _grade(
            name="retrieval_recall_at_k",
            passed=True,
            score=None,
            explanation="No relevant_doc_ids configured.",
            expected=[],
            observed={"recall": None},
        )

    retrieved = set(_retrieved_source_ids(prediction, case.retrieval_k))
    relevant = set(case.relevant_doc_ids)
    hits = sorted(retrieved & relevant)
    recall = len(hits) / len(relevant)
    return _grade(
        name="retrieval_recall_at_k",
        passed=recall >= case.min_recall_at_k,
        score=recall,
        explanation=f"Recall@{case.retrieval_k} is {recall:.3f}.",
        expected={"relevant_doc_ids": case.relevant_doc_ids, "min_recall_at_k": case.min_recall_at_k},
        observed={"retrieved_doc_ids": _retrieved_source_ids(prediction, case.retrieval_k), "hits": hits, "recall": recall},
    )


def grade_retrieval_precision_at_k(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade how concentrated top-k retrieval is on gold documents.

    Inputs:
    - `case.relevant_doc_ids` is the labeled set of gold source document IDs.
    - `case.retrieval_k` defines the prefix of retrieved chunks to inspect.
    - `case.min_precision_at_k` is the pass threshold.
    - `prediction["retrieved_context"]` supplies ranked retrieved chunks with source IDs.

    Output:
    - Score is chunk-level precision: top-k chunks from gold docs divided by total retrieved top-k chunks.
    - Passes when score is at least `case.min_precision_at_k`.
    - If no gold docs are configured, this grader is informational and returns score `None`.
    - `observed` records retrieved doc IDs, per-chunk hits, and precision.
    """
    if not case.relevant_doc_ids:
        return _grade(
            name="retrieval_precision_at_k",
            passed=True,
            score=None,
            explanation="No relevant_doc_ids configured.",
            expected=[],
            observed={"precision": None},
        )

    retrieved_ids = _retrieved_source_ids(prediction, case.retrieval_k)
    if not retrieved_ids:
        precision = 0.0
        hits: list[int] = []
    else:
        relevant = set(case.relevant_doc_ids)
        hits = [source_id for source_id in retrieved_ids if source_id in relevant]
        precision = len(hits) / len(retrieved_ids)
    return _grade(
        name="retrieval_precision_at_k",
        passed=precision >= case.min_precision_at_k,
        score=precision,
        explanation=f"Precision@{case.retrieval_k} is {precision:.3f}.",
        expected={"relevant_doc_ids": case.relevant_doc_ids, "min_precision_at_k": case.min_precision_at_k},
        observed={"retrieved_doc_ids": retrieved_ids, "hits": hits, "precision": precision},
    )


def grade_retrieval_mrr(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade how highly the first gold document is ranked.

    Inputs:
    - `case.relevant_doc_ids` is the labeled set of gold source document IDs.
    - `case.retrieval_k` defines the prefix of retrieved chunks to inspect.
    - `case.min_mrr` is the pass threshold.
    - `prediction["retrieved_context"]` supplies ranked retrieved chunks with source IDs.

    Output:
    - Score is reciprocal rank: `1 / first_relevant_rank`, or 0.0 if no gold doc appears in top-k.
    - Passes when score is at least `case.min_mrr`.
    - If no gold docs are configured, this grader is informational and returns score `None`.
    - `observed` records retrieved doc IDs, first relevant rank, and MRR.
    """
    if not case.relevant_doc_ids:
        return _grade(
            name="retrieval_mrr",
            passed=True,
            score=None,
            explanation="No relevant_doc_ids configured.",
            expected=[],
            observed={"mrr": None},
        )

    relevant = set(case.relevant_doc_ids)
    retrieved_ids = _retrieved_source_ids(prediction, case.retrieval_k)
    reciprocal_rank = 0.0
    first_rank = None
    for idx, source_id in enumerate(retrieved_ids, start=1):
        if source_id in relevant:
            first_rank = idx
            reciprocal_rank = 1 / idx
            break
    return _grade(
        name="retrieval_mrr",
        passed=reciprocal_rank >= case.min_mrr,
        score=reciprocal_rank,
        explanation=f"MRR@{case.retrieval_k} is {reciprocal_rank:.3f}.",
        expected={"relevant_doc_ids": case.relevant_doc_ids, "min_mrr": case.min_mrr},
        observed={"retrieved_doc_ids": retrieved_ids, "first_relevant_rank": first_rank, "mrr": reciprocal_rank},
    )


def grade_must_not_include(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade that forbidden literal strings are absent from the answer.

    Inputs:
    - `case.must_not_include` lists case-insensitive literal substrings that should not appear.
    - `prediction["answer"]` is searched for those substrings.

    Output:
    - Passes with score 1.0 when none of the forbidden strings are present.
    - Fails with score 0.0 when any forbidden string appears.
    - `observed` lists the forbidden strings that were found.
    """
    answer = _answer_text(prediction)
    found = [item for item in case.must_not_include if _contains(answer, item)]
    return _grade(
        name="must_not_include",
        passed=not found,
        score=1.0 if not found else 0.0,
        explanation="Forbidden answer text is absent." if not found else f"Forbidden text present: {found}",
        expected=case.must_not_include,
        observed=found,
    )


def grade_must_not_match(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade that forbidden regex patterns are absent from the answer.

    Inputs:
    - `case.must_not_match` lists regular expressions that should not match the answer.
    - `prediction["answer"]` is searched with case-insensitive multiline regex matching.

    Output:
    - Passes with score 1.0 when no forbidden regex matches.
    - Fails with score 0.0 when any forbidden regex matches.
    - `observed` lists the regex patterns that matched.
    """
    answer = _answer_text(prediction)
    found = [pattern for pattern in case.must_not_match if _matches(answer, pattern)]
    return _grade(
        name="must_not_match",
        passed=not found,
        score=1.0 if not found else 0.0,
        explanation="Forbidden regex patterns are absent." if not found else f"Forbidden patterns matched: {found}",
        expected=case.must_not_match,
        observed=found,
    )


def grade_citation_validity(_case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade that citations are drawn from retrieved context.

    Inputs:
    - `prediction["citations"]` supplies the URLs the answer cited.
    - `prediction["retrieved_context"]` supplies the retrieved chunks and their source URLs.

    Output:
    - Score is valid cited URLs divided by total cited URLs; answers with no citations score 1.0 here.
    - Passes only when every cited URL appears in retrieved context.
    - `expected` stores retrieved URLs; `observed` stores cited URLs, invalid URLs, and validity score.
    """
    citation_urls = _citation_urls(prediction)
    retrieved_urls = _retrieved_urls(prediction)
    invalid = [url for url in citation_urls if url not in retrieved_urls]
    score = 1.0 if not citation_urls else (len(citation_urls) - len(invalid)) / len(citation_urls)
    return _grade(
        name="citation_validity",
        passed=not invalid,
        score=score,
        explanation="All citations point to retrieved context." if not invalid else f"Citations not retrieved: {invalid}",
        expected=sorted(retrieved_urls),
        observed={"citation_urls": citation_urls, "invalid": invalid, "validity": score},
    )


def grade_citation_coverage(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade whether factual numeric/date claim sentences include citations.

    Inputs:
    - `prediction["answer"]` is split into sentences.
    - Sentences containing money, percentages, fiscal years, or dates are treated as factual claim sentences.
    - `case.min_citation_coverage` is the pass threshold.

    Output:
    - Score is cited factual claim sentences divided by factual claim sentences.
    - Passes when score is at least `case.min_citation_coverage`.
    - If no numeric/date factual claim sentences are found, score is 1.0.
    - `observed` records factual claim sentences, cited claim sentences, and coverage.
    """
    claims = _factual_claim_sentences(_answer_text(prediction))
    if not claims:
        coverage = 1.0
        cited_claims: list[str] = []
    else:
        cited_claims = [claim for claim in claims if CITATION_RE.search(claim)]
        coverage = len(cited_claims) / len(claims)
    return _grade(
        name="citation_coverage",
        passed=coverage >= case.min_citation_coverage,
        score=coverage,
        explanation=f"Citation coverage is {coverage:.3f}.",
        expected={"min_citation_coverage": case.min_citation_coverage},
        observed={"factual_claims": claims, "cited_claims": cited_claims, "coverage": coverage},
    )


def grade_numeric_date_claims_supported(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Warn whether answer numeric/date claims are present in retrieved context.

    Inputs:
    - `case.check_numeric_date_claims` can disable the check for a case.
    - `case.min_numeric_date_support` is the warning threshold when enabled.
    - `prediction["answer"]` supplies answer claims.
    - `prediction["retrieved_context"]` supplies supportable claims from retrieved evidence.

    Output:
    - Extracts and canonicalizes money amounts, percentages, fiscal years, and dates.
    - Score is supported answer claims divided by total answer claims; answers with no such claims score 1.0.
    - Passes when score is at least `case.min_numeric_date_support`, but this grade is non-blocking.
    - `observed` records answer claims, context claims, supported claims, unsupported claims, and score.
    """
    if not case.check_numeric_date_claims:
        return _grade(
            name="numeric_date_claims_supported",
            passed=True,
            score=None,
            explanation="Numeric/date claim checking is disabled for this case.",
            expected="disabled",
            observed=[],
        )

    claims = extract_numeric_date_claims(_answer_text(prediction))
    context_claims = set(extract_numeric_date_claims(_context_text(prediction)))
    supported = [claim for claim in claims if claim in context_claims]
    unsupported = [claim for claim in claims if claim not in context_claims]
    score = 1.0 if not claims else len(supported) / len(claims)
    return _grade(
        name="numeric_date_claims_supported",
        passed=score >= case.min_numeric_date_support,
        score=score,
        explanation=(
            f"Numeric/date support score is {score:.3f}."
        ),
        expected={"min_numeric_date_support": case.min_numeric_date_support},
        observed={
            "claims": claims,
            "context_claims": sorted(context_claims),
            "supported": supported,
            "unsupported": unsupported,
            "score": score,
        },
    )


def grade_retrieval_nonempty(case: QACase, prediction: dict[str, Any]) -> DeterministicGrade:
    """Grade that retrieval returned context for the answer.

    Inputs:
    - `case.allow_empty_retrieval` controls whether empty retrieval is acceptable.
    - `prediction["retrieved_context"]` supplies the retrieved chunks.

    Output:
    - Passes when retrieved context exists, or when the case explicitly allows empty retrieval.
    - Score is 1.0 when context exists and 0.0 when it does not, even if the case allows empty retrieval.
    - `observed` is the number of retrieved context items.
    """
    retrieved = _retrieved_context(prediction)
    passed = case.allow_empty_retrieval or bool(retrieved)
    return _grade(
        name="retrieval_nonempty",
        passed=passed,
        score=1.0 if bool(retrieved) else 0.0,
        explanation="Retrieved context exists." if retrieved else "No retrieved context was returned.",
        observed=len(retrieved),
    )


DETERMINISTIC_GRADERS = [
    grade_answer_nonempty,
    grade_source_kind_policy,
    grade_no_plan_sources,
    grade_citation_urls,
    grade_retrieval_recall_at_k,
    grade_retrieval_precision_at_k,
    grade_retrieval_mrr,
    grade_citation_validity,
    grade_citation_coverage,
    grade_must_not_include,
    grade_must_not_match,
    grade_numeric_date_claims_supported,
    grade_retrieval_nonempty,
]


def run_deterministic_graders(case: QACase, prediction: dict[str, Any]) -> list[DeterministicGrade]:
    return [grader(case, prediction) for grader in DETERMINISTIC_GRADERS]
