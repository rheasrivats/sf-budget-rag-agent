from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    run_at: datetime
    total_sources: int
    total_chunks: int
    updated_sources: int


class SourceItem(BaseModel):
    id: int
    url: str
    title: str
    content_type: str
    last_ingested_at: datetime
    chunk_count: int


class SourcesResponse(BaseModel):
    sources: list[SourceItem]
    last_refreshed_at: datetime | None
    stale: bool


class Citation(BaseModel):
    source_id: int | None = None
    url: str
    label: str | None = None


class PlanGenerateRequest(BaseModel):
    force_refresh_hint: bool = Field(default=False)


class PlanResponse(BaseModel):
    id: int
    created_at: datetime
    scope_label: str
    model_used: str
    mayor_total_budget: float | None
    generated_total_budget: float | None
    spending_rule_passed: bool
    memo_markdown: str
    citation_map: list[dict[str, Any]]
    metadata: dict[str, Any]


class ComparisonRowResponse(BaseModel):
    department: str
    mayor_directional: str
    proposed_directional: str
    delta_directional: str
    mayor_fy_2025_26: float | None
    mayor_fy_2026_27: float | None
    proposed_fy_2025_26: float | None
    proposed_fy_2026_27: float | None
    rationale: str
    citations: list[dict[str, Any]]


class ComparisonResponse(BaseModel):
    plan_id: int
    summary: dict[str, Any]
    rows: list[ComparisonRowResponse]


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default"
    escalate: bool = False


class ChatResponse(BaseModel):
    plan_id: int
    thread_id: str
    model_used: str
    answer: str
    citations: list[dict[str, Any]]
    debug: dict[str, Any] | None = None


class PlanListItem(BaseModel):
    id: int
    created_at: datetime
    scope_label: str
    spending_rule_passed: bool
    generated_total_budget: float | None


class PlanListResponse(BaseModel):
    plans: list[PlanListItem]
