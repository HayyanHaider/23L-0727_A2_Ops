"""A persistent, queryable audit record with no uploaded image bytes."""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    filename: Mapped[str] = mapped_column(String(255))
    label: Mapped[str] = mapped_column(String(32))
    ai_score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(80))
    model_sha256: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)


class Store:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True,
                                    connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
        Base.metadata.create_all(self.engine)

    def check(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def add(self, **values):
        with Session(self.engine) as session:
            row = Prediction(**values)
            session.add(row)
            session.commit()

    def recent(self, limit=100):
        with Session(self.engine) as session:
            return list(session.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)))
