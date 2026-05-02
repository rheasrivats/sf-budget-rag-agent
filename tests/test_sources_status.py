from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base
from app.models import SourceDocument
from app.services.sources import get_sources_status


def test_sources_status_stale_flag():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, future=True)

    with Local() as db:  # type: Session
        old = SourceDocument(
            url="https://sf.gov/old",
            title="old",
            content_type="html",
            last_ingested_at=datetime.utcnow() - timedelta(hours=72),
        )
        db.add(old)
        db.commit()

        status = get_sources_status(db)
        assert status["stale"] is True
        assert len(status["sources"]) == 1


def test_sources_status_lists_only_active_sources():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, future=True)

    with Local() as db:  # type: Session
        active = SourceDocument(
            url="https://sf.gov/active.pdf",
            title="active",
            content_type="pdf",
            active=True,
        )
        inactive = SourceDocument(
            url="https://sf.gov/topics/budget",
            title="inactive",
            content_type="html",
            active=False,
        )
        db.add_all([active, inactive])
        db.commit()

        status = get_sources_status(db)
        assert [source["url"] for source in status["sources"]] == ["https://sf.gov/active.pdf"]
