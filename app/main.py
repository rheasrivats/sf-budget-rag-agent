from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Base, engine, get_db
from app.models import ComparisonRow, PlanVersion
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ComparisonResponse,
    ComparisonRowResponse,
    IngestResponse,
    PlanGenerateRequest,
    PlanListItem,
    PlanListResponse,
    PlanResponse,
    SourcesResponse,
)
from app.services.chat import answer_question
from app.services.exports import comparison_to_csv_bytes, memo_to_pdf_bytes
from app.services.ingest import run_ingestion
from app.services.planner import generate_plan, get_plan, list_plans
from app.services.sources import get_sources_status

app = FastAPI(title=settings.app_name)

Base.metadata.create_all(bind=engine)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/ingest/run", response_model=IngestResponse)
def ingest_run(db: Session = Depends(get_db)):
    result = run_ingestion(db)
    db.commit()
    return IngestResponse(**result)


@app.get("/sources", response_model=SourcesResponse)
def sources(db: Session = Depends(get_db)):
    return SourcesResponse(**get_sources_status(db))


@app.post("/plans/generate", response_model=PlanResponse)
def plans_generate(_payload: PlanGenerateRequest, db: Session = Depends(get_db)):
    try:
        plan = generate_plan(db)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()

    return PlanResponse(
        id=plan.id,
        created_at=plan.created_at,
        scope_label=plan.scope_label,
        model_used=plan.model_used,
        mayor_total_budget=plan.mayor_total_budget,
        generated_total_budget=plan.generated_total_budget,
        spending_rule_passed=plan.spending_rule_passed,
        memo_markdown=plan.memo_markdown,
        citation_map=plan.citation_map,
        metadata=plan.metadata_json,
    )


@app.get("/plans", response_model=PlanListResponse)
def plans(db: Session = Depends(get_db)):
    items = [
        PlanListItem(
            id=plan.id,
            created_at=plan.created_at,
            scope_label=plan.scope_label,
            spending_rule_passed=plan.spending_rule_passed,
            generated_total_budget=plan.generated_total_budget,
        )
        for plan in list_plans(db)
    ]
    return PlanListResponse(plans=items)


@app.get("/plans/{plan_id}", response_model=PlanResponse)
def plan_by_id(plan_id: int, db: Session = Depends(get_db)):
    plan = get_plan(db, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Plan not found")

    return PlanResponse(
        id=plan.id,
        created_at=plan.created_at,
        scope_label=plan.scope_label,
        model_used=plan.model_used,
        mayor_total_budget=plan.mayor_total_budget,
        generated_total_budget=plan.generated_total_budget,
        spending_rule_passed=plan.spending_rule_passed,
        memo_markdown=plan.memo_markdown,
        citation_map=plan.citation_map,
        metadata=plan.metadata_json,
    )


@app.get("/plans/{plan_id}/comparison", response_model=ComparisonResponse)
def plan_comparison(plan_id: int, db: Session = Depends(get_db)):
    plan = get_plan(db, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Plan not found")

    rows = list(
        db.scalars(select(ComparisonRow).where(ComparisonRow.plan_id == plan_id).order_by(ComparisonRow.department.asc()))
    )

    return ComparisonResponse(
        plan_id=plan_id,
        summary=plan.metadata_json.get("comparison_summary", {}),
        rows=[
            ComparisonRowResponse(
                department=row.department,
                mayor_directional=row.mayor_directional,
                proposed_directional=row.proposed_directional,
                delta_directional=row.delta_directional,
                mayor_fy_2025_26=row.mayor_fy_2025_26,
                mayor_fy_2026_27=row.mayor_fy_2026_27,
                proposed_fy_2025_26=row.proposed_fy_2025_26,
                proposed_fy_2026_27=row.proposed_fy_2026_27,
                rationale=row.rationale,
                citations=row.citations,
            )
            for row in rows
        ],
    )


@app.post("/plans/{plan_id}/chat", response_model=ChatResponse)
def plan_chat(plan_id: int, payload: ChatRequest, db: Session = Depends(get_db)):
    plan = get_plan(db, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Plan not found")

    try:
        result = answer_question(
            db,
            plan=plan,
            thread_id=payload.thread_id,
            query=payload.query,
            escalate=payload.escalate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()

    return ChatResponse(
        plan_id=plan_id,
        thread_id=payload.thread_id,
        model_used=result["model_used"],
        answer=result["answer"],
        citations=result["citations"],
        debug=result.get("debug"),
    )


@app.get("/plans/{plan_id}/export/pdf")
def export_pdf(plan_id: int, db: Session = Depends(get_db)):
    plan = get_plan(db, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Plan not found")

    content = memo_to_pdf_bytes(plan)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=plan_{plan_id}.pdf"},
    )


@app.get("/plans/{plan_id}/comparison/export/csv")
def export_comparison_csv(plan_id: int, db: Session = Depends(get_db)):
    rows = list(db.scalars(select(ComparisonRow).where(ComparisonRow.plan_id == plan_id)))
    if not rows:
        raise HTTPException(status_code=404, detail="No comparison rows found for plan")

    content = comparison_to_csv_bytes(rows)
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=plan_{plan_id}_comparison.csv"},
    )


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True})
