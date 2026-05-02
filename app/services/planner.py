from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ComparisonRow, PlanVersion
from app.services.ingest import index_text_document
from app.services.llm import comparison_model, planner_model
from app.services.retrieval import EvidenceChunk, render_evidence_context, retrieve_chunks


class DepartmentBaseline(BaseModel):
    department: str
    mayor_fy_2025_26: float | None = None
    mayor_fy_2026_27: float | None = None
    mayor_directional: str = "unknown"
    citations: list[dict[str, Any]] = Field(default_factory=list)


class MayorBaseline(BaseModel):
    mayor_total_budget: float | None = None
    deficit_context: str
    departments: list[DepartmentBaseline] = Field(default_factory=list)


class DepartmentProposal(BaseModel):
    department: str
    proposed_fy_2025_26: float | None = None
    proposed_fy_2026_27: float | None = None
    proposed_directional: str = "maintain"
    rationale: str
    citations: list[dict[str, Any]] = Field(default_factory=list)


class GeneratedPlan(BaseModel):
    executive_summary: str
    strategic_priorities: str
    department_strategy: str
    deficit_handling: str
    risks_tradeoffs: str
    generated_total_budget: float | None = None
    citation_map: list[dict[str, Any]] = Field(default_factory=list)
    departments: list[DepartmentProposal] = Field(default_factory=list)


def _dedupe_chunks(chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
    seen: set[int] = set()
    out: list[EvidenceChunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        out.append(chunk)
    return out


def _collect_evidence(db: Session) -> list[EvidenceChunk]:
    general = retrieve_chunks(
        db,
        "San Francisco budget FY 2025-26 FY 2026-27 deficits departments allocations priorities",
        k=30,
    )
    mayor = retrieve_chunks(
        db,
        "Mayor Daniel Lurie proposed budget FY 2026-27 department allocations total budget deficit",
        k=30,
    )
    return _dedupe_chunks(general + mayor)


def _extract_baseline(context: str) -> MayorBaseline:
    model = comparison_model().with_structured_output(MayorBaseline, method="function_calling")
    prompt = (
        "Extract mayor-proposed baseline budget context for San Francisco from the evidence. "
        "Return defensible values only; if unknown, use null. "
        "Mayor total budget should be for the two-year proposal scope if available. "
        "Provide department-level baseline rows with citations.\n\n"
        f"Evidence:\n{context}"
    )
    return model.invoke(prompt)


def _draft_plan(context: str, baseline: MayorBaseline) -> GeneratedPlan:
    model = planner_model().with_structured_output(GeneratedPlan, method="function_calling")
    prompt = (
        "You are generating a comprehensive San Francisco budget plan in section-level format "
        "similar to the Mayor's proposal. Scope: FY 2025-26 and FY 2026-27, with emphasis on FY 2026-27. "
        "Hard constraints: generated_total_budget must not exceed mayor_total_budget if mayor_total_budget is known. "
        "Include explicit deficit-handling proposals and tradeoffs. "
        "Use only evidence-backed claims and provide citations."
        "\n\n"
        f"Mayor baseline (JSON): {baseline.model_dump_json()}\n\n"
        f"Evidence:\n{context}"
    )
    return model.invoke(prompt)


def _enforce_spending_cap(plan: GeneratedPlan, baseline: MayorBaseline) -> tuple[GeneratedPlan, bool]:
    if baseline.mayor_total_budget is None or plan.generated_total_budget is None:
        return plan, False

    if plan.generated_total_budget <= baseline.mayor_total_budget:
        return plan, True

    if not plan.departments:
        plan.generated_total_budget = baseline.mayor_total_budget
        plan.deficit_handling += (
            "\n\nSpending-cap adjustment: total spend was reduced to mayor baseline to satisfy cap."
        )
        return plan, True

    ratio = baseline.mayor_total_budget / max(plan.generated_total_budget, 1e-6)
    for dep in plan.departments:
        if dep.proposed_fy_2025_26 is not None:
            dep.proposed_fy_2025_26 = round(dep.proposed_fy_2025_26 * ratio, 2)
        if dep.proposed_fy_2026_27 is not None:
            dep.proposed_fy_2026_27 = round(dep.proposed_fy_2026_27 * ratio, 2)
    plan.generated_total_budget = round(baseline.mayor_total_budget, 2)
    plan.deficit_handling += (
        "\n\nSpending-cap adjustment applied automatically so generated total does not exceed mayor total."
    )
    return plan, True


def _memo_from_plan(plan: GeneratedPlan) -> str:
    return "\n\n".join(
        [
            "# Proposed Budget Plan (FY 2025-26 and FY 2026-27)",
            "## Executive Summary\n" + plan.executive_summary,
            "## Strategic Priorities\n" + plan.strategic_priorities,
            "## Department Strategy\n" + plan.department_strategy,
            "## Deficit Handling\n" + plan.deficit_handling,
            "## Risks and Tradeoffs\n" + plan.risks_tradeoffs,
        ]
    )


def _build_comparison_rows(plan: GeneratedPlan, baseline: MayorBaseline) -> list[ComparisonRow]:
    by_dep_baseline = {d.department.lower(): d for d in baseline.departments}
    rows: list[ComparisonRow] = []

    for dep_plan in plan.departments:
        key = dep_plan.department.lower()
        base = by_dep_baseline.get(key)
        mayor_25 = base.mayor_fy_2025_26 if base else None
        mayor_26 = base.mayor_fy_2026_27 if base else None
        mayor_dir = base.mayor_directional if base else "unknown"

        proposed_25 = dep_plan.proposed_fy_2025_26
        proposed_26 = dep_plan.proposed_fy_2026_27
        proposed_dir = dep_plan.proposed_directional

        delta = "unknown"
        if mayor_26 is not None and proposed_26 is not None:
            if proposed_26 > mayor_26:
                delta = "increase"
            elif proposed_26 < mayor_26:
                delta = "decrease"
            else:
                delta = "flat"

        citations = []
        if base:
            citations.extend(base.citations)
        citations.extend(dep_plan.citations)

        rows.append(
            ComparisonRow(
                department=dep_plan.department,
                mayor_directional=mayor_dir,
                proposed_directional=proposed_dir,
                delta_directional=delta,
                mayor_fy_2025_26=mayor_25,
                mayor_fy_2026_27=mayor_26,
                proposed_fy_2025_26=proposed_25,
                proposed_fy_2026_27=proposed_26,
                rationale=dep_plan.rationale,
                citations=citations,
            )
        )

    return rows


def _top_delta_summary(rows: list[ComparisonRow]) -> dict[str, Any]:
    increases = [r for r in rows if r.delta_directional == "increase"]
    decreases = [r for r in rows if r.delta_directional == "decrease"]
    return {
        "increase_count": len(increases),
        "decrease_count": len(decreases),
        "largest_increases": [r.department for r in increases[:5]],
        "largest_decreases": [r.department for r in decreases[:5]],
        "deficit_actions_hint": "See deficit handling section in plan memo.",
    }


def _plan_index_text(plan_row: PlanVersion, rows: list[ComparisonRow], baseline: MayorBaseline) -> str:
    lines: list[str] = [
        f"Generated Plan Version #{plan_row.id}",
        f"Scope: {plan_row.scope_label}",
        f"Created At: {plan_row.created_at.isoformat()}",
        f"Mayor Total Budget (USD millions): {plan_row.mayor_total_budget}",
        f"Generated Total Budget (USD millions): {plan_row.generated_total_budget}",
        f"Spending Rule Passed: {plan_row.spending_rule_passed}",
        "",
        "Plan Memo:",
        plan_row.memo_markdown,
        "",
        "Baseline Deficit Context:",
        baseline.deficit_context,
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


def _index_generated_plan_document(
    db: Session, plan_row: PlanVersion, rows: list[ComparisonRow], baseline: MayorBaseline
) -> None:
    plan_text = _plan_index_text(plan_row, rows, baseline)
    index_text_document(
        db,
        url=f"plan://{plan_row.id}",
        title=f"Generated Plan #{plan_row.id} ({plan_row.scope_label})",
        content_type="markdown",
        source_kind="plan",
        text=plan_text,
        metadata={
            "plan_id": plan_row.id,
            "scope_label": plan_row.scope_label,
            "created_at": plan_row.created_at.isoformat(),
            "model_used": plan_row.model_used,
            "spending_rule_passed": plan_row.spending_rule_passed,
        },
    )


def generate_plan(db: Session) -> PlanVersion:
    chunks = _collect_evidence(db)
    if not chunks:
        raise RuntimeError("No indexed evidence available. Run ingestion first.")

    context = render_evidence_context(chunks)
    baseline = _extract_baseline(context)
    plan = _draft_plan(context, baseline)
    plan, spending_ok = _enforce_spending_cap(plan, baseline)

    memo = _memo_from_plan(plan)

    rows = _build_comparison_rows(plan, baseline)

    plan_row = PlanVersion(
        created_at=datetime.utcnow(),
        status="ready",
        scope_label="FY 2025-26 and FY 2026-27 (FY 2026-27 emphasis)",
        model_used="gpt-5.4 (high reasoning)",
        mayor_total_budget=baseline.mayor_total_budget,
        generated_total_budget=plan.generated_total_budget,
        spending_rule_passed=spending_ok,
        memo_markdown=memo,
        citation_map=plan.citation_map,
        metadata_json={
            "department_count": len(plan.departments),
            "deficit_context": baseline.deficit_context,
            "evidence_chunk_count": len(chunks),
        },
    )
    db.add(plan_row)
    db.flush()

    for row in rows:
        row.plan_id = plan_row.id
        db.add(row)

    _index_generated_plan_document(db, plan_row, rows, baseline)

    # store summary in metadata for quick UI.
    plan_row.metadata_json = {
        **plan_row.metadata_json,
        "comparison_summary": _top_delta_summary(rows),
    }

    db.flush()
    return plan_row


def get_plan(db: Session, plan_id: int) -> PlanVersion | None:
    return db.scalar(select(PlanVersion).where(PlanVersion.id == plan_id))


def list_plans(db: Session, limit: int = 20) -> list[PlanVersion]:
    stmt = select(PlanVersion).order_by(PlanVersion.created_at.desc()).limit(limit)
    return list(db.scalars(stmt))
