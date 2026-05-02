from __future__ import annotations

from time import perf_counter
from typing import Any

from langchain_openai import ChatOpenAI

from evals.types import QACase, RubricGrade

RUBRIC_NAMES = ("correctness", "answer_relevance", "groundedness", "retrieval_relevance")


def _context_text(prediction: dict[str, Any], max_chars: int = 12000) -> str:
    parts = []
    for idx, item in enumerate(prediction.get("retrieved_context") or [], start=1):
        metadata = item.get("metadata") or {}
        parts.append(
            "\n".join(
                [
                    f"[{idx}] Source: {metadata.get('url', 'unknown')}",
                    f"Title: {metadata.get('title', '')}",
                    f"Source kind: {metadata.get('source_kind', '')}",
                    f"Content: {item.get('content', '')}",
                ]
            )
        )
    return "\n\n".join(parts)[:max_chars]


def _rubric_prompt(rubric_name: str, case: QACase, prediction: dict[str, Any]) -> str:
    context = _context_text(prediction)
    answer = prediction.get("answer", "")
    base = (
        "You are grading a RAG answer for a San Francisco budget research assistant. "
        "Return a strict score from 0.0 to 1.0, whether it passes, and a concise rationale. "
        "A score of 0.0 means the model output is the worst possible output for this rubric. "
        "A score of 1.0 means the model output is the best possible output for this rubric. "
        "Use 0.7 as the pass threshold. Penalize unsupported claims and fabricated specifics.\n\n"
        f"Rubric: {rubric_name}\n"
        f"Question: {case.query}\n"
        f"Reference answer: {case.reference_answer}\n"
        f"Actual answer: {answer}\n"
        f"Retrieved context:\n{context}\n"
    )
    if case.rubric_notes:
        base += f"\nCase notes: {case.rubric_notes}\n"

    rubric_instructions = {
        "correctness": (
            "Use the reference answer as the expected core facts, not as the only acceptable wording or structure. "
            "Do not penalize additional detail when it is relevant, factually correct, and supported by retrieved context. "
            "Penalize omissions, contradictions, unsupported specifics, or extra detail that changes the meaning of the answer "
            "or distracts from the user's question."
        ),
        "answer_relevance": "Grade whether the answer directly addresses the user's question without drifting.",
        "groundedness": (
            "Judge only whether the actual answer's factual claims are supported by the retrieved context. "
            "Do not compare groundedness to the reference answer."
        ),
        "retrieval_relevance": "Grade whether the retrieved context is relevant and useful for answering the question.",
    }
    return base + "\n" + rubric_instructions[rubric_name]


def grade_with_llm(
    case: QACase,
    prediction: dict[str, Any],
    *,
    model_name: str,
    threshold: float,
    trace_collector=None,
) -> dict[str, RubricGrade]:
    model = ChatOpenAI(model=model_name, temperature=0).with_structured_output(
        RubricGrade,
        method="function_calling",
    )
    grades: dict[str, RubricGrade] = {}
    for rubric_name in RUBRIC_NAMES:
        prompt = _rubric_prompt(rubric_name, case, prediction)
        started = perf_counter()
        grade = model.invoke(prompt)
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        grade.score = max(0.0, min(1.0, float(grade.score)))
        grade.passed = grade.score >= threshold
        grades[rubric_name] = grade
        if trace_collector is not None:
            trace_collector.append(
                "judge_calls",
                {
                    "rubric": rubric_name,
                    "model": model_name,
                    "prompt": prompt,
                    "grade": grade.model_dump(mode="json"),
                    "elapsed_ms": elapsed_ms,
                },
            )
    return grades
