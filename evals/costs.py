from __future__ import annotations

from typing import Any

from evals.judges import RUBRIC_NAMES, _rubric_prompt
from evals.types import CostEstimate, QACase

CHARS_PER_TOKEN = 4

MODEL_PRICES_PER_1M = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
}


def estimate_tokens(text: Any) -> int:
    return max(1, round(len(str(text or "")) / CHARS_PER_TOKEN))


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES_PER_1M.get(model, MODEL_PRICES_PER_1M["gpt-5.4-mini"])
    return round(
        (input_tokens * prices["input"] / 1_000_000) + (output_tokens * prices["output"] / 1_000_000),
        6,
    )


def _messages_tokens(trace_data: dict[str, Any]) -> int:
    messages = trace_data.get("messages") or []
    return sum(estimate_tokens(message.get("content", "")) for message in messages if isinstance(message, dict))


def _adequacy_tokens(trace_data: dict[str, Any]) -> int:
    crag = trace_data.get("crag") or {}
    result = crag.get("adequacy_result")
    if not result:
        return 0
    retrieval_chunks = ((trace_data.get("retrieval") or {}).get("chunks") or [])[:5]
    context_chars = sum(len(str(chunk.get("content_excerpt") or "")[:700]) for chunk in retrieval_chunks)
    prompt_chars = len(str((trace_data.get("target_call") or {}).get("query") or "")) + context_chars + 900
    return estimate_tokens("x" * prompt_chars)


def estimate_trial_cost(
    *,
    case: QACase,
    prediction: dict[str, Any],
    judge_model: str,
    trace_data: dict[str, Any],
) -> CostEstimate:
    answer = prediction.get("answer", "")
    model_used = str(prediction.get("model_used") or "gpt-5.4-mini")
    target_input_tokens = _messages_tokens(trace_data)
    target_output_tokens = estimate_tokens(answer)
    target_cost = estimate_cost_usd(model_used, target_input_tokens, target_output_tokens)

    adequacy_input_tokens = _adequacy_tokens(trace_data)
    adequacy_output_tokens = estimate_tokens((trace_data.get("crag") or {}).get("adequacy_result", "")) if adequacy_input_tokens else 0
    adequacy_model = "gpt-5.4-mini"
    adequacy_cost = estimate_cost_usd(adequacy_model, adequacy_input_tokens, adequacy_output_tokens) if adequacy_input_tokens else 0.0

    judge_input_tokens = 0
    judge_output_tokens = 0
    for rubric_name in RUBRIC_NAMES:
        judge_input_tokens += estimate_tokens(_rubric_prompt(rubric_name, case, prediction))
    for call in trace_data.get("judge_calls") or []:
        judge_output_tokens += estimate_tokens((call.get("grade") or {}).get("rationale", ""))
    if judge_output_tokens == 0:
        judge_output_tokens = 80 * len(RUBRIC_NAMES)
    judge_cost = estimate_cost_usd(judge_model, judge_input_tokens, judge_output_tokens)

    total_input = target_input_tokens + adequacy_input_tokens + judge_input_tokens
    total_output = target_output_tokens + adequacy_output_tokens + judge_output_tokens
    return CostEstimate(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
        estimated_cost_usd=round(target_cost + adequacy_cost + judge_cost, 6),
        by_component={
            "target": {
                "model": model_used,
                "input_tokens": target_input_tokens,
                "output_tokens": target_output_tokens,
                "estimated_cost_usd": target_cost,
            },
            "adequacy": {
                "model": adequacy_model,
                "input_tokens": adequacy_input_tokens,
                "output_tokens": adequacy_output_tokens,
                "estimated_cost_usd": adequacy_cost,
            },
            "judge": {
                "model": judge_model,
                "input_tokens": judge_input_tokens,
                "output_tokens": judge_output_tokens,
                "estimated_cost_usd": judge_cost,
            },
        },
        note=f"Approximate token estimate uses ~{CHARS_PER_TOKEN} characters per token and configured model price constants.",
    )


def sum_cost_estimates(estimates: list[CostEstimate]) -> CostEstimate:
    input_tokens = sum(item.input_tokens for item in estimates)
    output_tokens = sum(item.output_tokens for item in estimates)
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated_cost_usd=round(sum(item.estimated_cost_usd for item in estimates), 6),
        by_component={},
        note=f"Approximate token estimate uses ~{CHARS_PER_TOKEN} characters per token and configured model price constants.",
    )
