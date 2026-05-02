from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QACase(BaseModel):
    id: str
    query: str
    reference_answer: str
    plan_id: int | str | None = None
    thread_id: str | None = None
    escalate: bool = False
    expected_source_kinds: list[str] = Field(default_factory=lambda: ["official"])
    expected_citation_urls: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[int] = Field(default_factory=list)
    retrieval_k: int = 8
    min_recall_at_k: float = 1.0
    min_precision_at_k: float = 0.0
    min_mrr: float = 0.2
    min_citation_coverage: float = 0.0
    min_numeric_date_support: float = 0.8
    must_not_include: list[str] = Field(default_factory=list)
    must_not_match: list[str] = Field(default_factory=list)
    check_numeric_date_claims: bool = True
    allow_empty_retrieval: bool = False
    rubric_notes: str = ""


class QASuite(BaseModel):
    version: int
    suite: str
    default_plan_id: int | str = "latest"
    cases: list[QACase]


class DeterministicGrade(BaseModel):
    name: str
    passed: bool
    blocking: bool = True
    description: str = ""
    explanation: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    expected: Any = None
    observed: Any = None


class RubricGrade(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    rationale: str


class CaseTrialResult(BaseModel):
    case_id: str
    trial: int
    passed: bool
    answer: str
    model_used: str
    citations: list[dict[str, Any]]
    deterministic_grades: list[DeterministicGrade]
    rubric_grades: dict[str, RubricGrade]
    trace_path: str | None = None
    error: str | None = None


class CaseAggregateResult(BaseModel):
    case_id: str
    passed: bool
    trials: list[CaseTrialResult]
    pass_at_k: bool
    pass_caret_k: bool
    trial_pass_rate: float
    deterministic_passed: bool
    rubric_summary: dict[str, dict[str, float | bool]]
    failures: list[str] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    suite: str
    run_id: str
    limited: bool
    limit: int | None
    trials: int
    threshold: float
    critical_floor: float
    suite_passed: bool
    cases_passed: int
    cases_failed: int
    pass_at_k: float
    pass_caret_k: float
    trial_pass_rate: float
    rubric_means: dict[str, float]
    failures: list[dict[str, str]]
    cases: list[CaseAggregateResult]
