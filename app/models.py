from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    content_type: Mapped[str] = mapped_column(String(32), default="html")
    source_kind: Mapped[str] = mapped_column(String(32), default="official")
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chunks: Mapped[list[DocumentChunk]] = relationship("DocumentChunk", back_populates="source", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("source_documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding_json: Mapped[list[float]] = mapped_column(JSON)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    source: Mapped[SourceDocument] = relationship("SourceDocument", back_populates="chunks")

    __table_args__ = (UniqueConstraint("source_id", "chunk_index", name="uq_chunk_per_source"),)


class PlanVersion(Base):
    __tablename__ = "plan_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    status: Mapped[str] = mapped_column(String(32), default="ready")
    scope_label: Mapped[str] = mapped_column(String(128), default="FY 2025-26 and FY 2026-27")
    model_used: Mapped[str] = mapped_column(String(128), default="")
    mayor_total_budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    generated_total_budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    spending_rule_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    memo_markdown: Mapped[str] = mapped_column(Text, default="")
    citation_map: Mapped[list[dict]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    comparisons: Mapped[list[ComparisonRow]] = relationship("ComparisonRow", back_populates="plan", cascade="all, delete-orphan")


class ComparisonRow(Base):
    __tablename__ = "comparison_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("plan_versions.id", ondelete="CASCADE"), index=True)
    department: Mapped[str] = mapped_column(String(255), index=True)
    mayor_directional: Mapped[str] = mapped_column(String(64), default="unknown")
    proposed_directional: Mapped[str] = mapped_column(String(64), default="unknown")
    delta_directional: Mapped[str] = mapped_column(String(64), default="unknown")
    mayor_fy_2025_26: Mapped[float | None] = mapped_column(Float, nullable=True)
    mayor_fy_2026_27: Mapped[float | None] = mapped_column(Float, nullable=True)
    proposed_fy_2025_26: Mapped[float | None] = mapped_column(Float, nullable=True)
    proposed_fy_2026_27: Mapped[float | None] = mapped_column(Float, nullable=True)
    rationale: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list[dict]] = mapped_column(JSON, default=list)

    plan: Mapped[PlanVersion] = relationship("PlanVersion", back_populates="comparisons")


class ChatThread(Base):
    __tablename__ = "chat_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("plan_versions.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
