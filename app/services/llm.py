from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from app.config import settings


def _build_model(model_name: str, effort: str | None = None, temperature: float = 0.1) -> ChatOpenAI:
    kwargs: dict[str, Any] = {"model": model_name, "temperature": temperature}
    if effort in {"low", "medium", "high"}:
        kwargs["reasoning"] = {"effort": effort, "summary": "auto"}
    return ChatOpenAI(**kwargs)


def planner_model() -> ChatOpenAI:
    return _build_model(
        model_name=settings.planner_model,
        effort=settings.planner_reasoning_effort,
        temperature=0.2,
    )


def comparison_model() -> ChatOpenAI:
    return _build_model(
        model_name=settings.planner_model,
        effort=settings.planner_reasoning_effort,
        temperature=0.1,
    )


def qa_model(escalate: bool = False) -> ChatOpenAI:
    if escalate:
        return _build_model(
            model_name=settings.planner_model,
            effort=settings.planner_reasoning_effort,
            temperature=0,
        )
    return _build_model(
        model_name=settings.chat_model_mini,
        effort=settings.qa_reasoning_effort,
        temperature=0,
    )


def should_escalate(query: str) -> bool:
    q = query.lower()
    # multi-part / scenario analysis queries get promoted.
    markers = ["what if", "scenario", "tradeoff", "compare", "across", "multi", "deficit", "sensitivity"]
    question_count = q.count("?")
    return question_count > 1 or any(marker in q for marker in markers) or len(q.split()) > 70
