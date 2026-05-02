from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DocumentChunk, SourceDocument


def get_sources_status(db: Session) -> dict:
    sources = list(
        db.scalars(
            select(SourceDocument)
            .where(SourceDocument.active.is_(True))
            .order_by(SourceDocument.last_ingested_at.desc())
        )
    )
    if not sources:
        return {"sources": [], "last_refreshed_at": None, "stale": True}

    counts = {
        source_id: count
        for source_id, count in db.execute(
            select(DocumentChunk.source_id, func.count(DocumentChunk.id)).group_by(DocumentChunk.source_id)
        )
    }

    last_refreshed = max(source.last_ingested_at for source in sources)
    stale = last_refreshed < datetime.utcnow() - timedelta(hours=settings.stale_hours)

    items = [
        {
            "id": s.id,
            "url": s.url,
            "title": s.title,
            "content_type": s.content_type,
            "last_ingested_at": s.last_ingested_at,
            "chunk_count": int(counts.get(s.id, 0)),
        }
        for s in sources
    ]

    return {"sources": items, "last_refreshed_at": last_refreshed, "stale": stale}
