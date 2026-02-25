"""
SQLAlchemy ORM models for the Email Triage Observation Storage.

All writes use natural-key upserts (idempotent).
PII-sensitive columns use hashes or encrypted values.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# pipeline_runs
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    message_id: Mapped[str] = mapped_column(String(512), nullable=False, index=True)

    # Version fingerprint
    dictionary_version: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(16), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(64), nullable=False)
    stoplist_version: Mapped[str] = mapped_column(String(64), nullable=False)
    ner_model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)

    # Output summary
    topics: Mapped[list | None] = mapped_column(JSON)
    sentiment_value: Mapped[str | None] = mapped_column(String(16))
    sentiment_confidence: Mapped[float | None] = mapped_column(Float)
    priority_value: Mapped[str | None] = mapped_column(String(16))
    priority_confidence: Mapped[float | None] = mapped_column(Float)
    customer_status_value: Mapped[str | None] = mapped_column(String(16))
    customer_status_source: Mapped[str | None] = mapped_column(String(64))

    # Performance
    processing_time_ms: Mapped[float | None] = mapped_column(Float)
    keywords_extracted: Mapped[int | None] = mapped_column(Integer)
    entities_extracted: Mapped[int | None] = mapped_column(Integer)
    warnings: Mapped[list | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# messages
# ─────────────────────────────────────────────────────────────────────────────

class Message(Base):
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    from_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256 of from_raw
    subject_redacted: Mapped[str | None] = mapped_column(Text)           # PII stripped
    text_hash: Mapped[str | None] = mapped_column(String(64))
    language: Mapped[str] = mapped_column(String(8), default="it")
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("ix_messages_from_hash", "from_hash"),)


# ─────────────────────────────────────────────────────────────────────────────
# label_registry
# ─────────────────────────────────────────────────────────────────────────────

class LabelRegistry(Base):
    __tablename__ = "label_registry"

    label_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active"
    )  # active | proposed | deprecated | merged
    merged_into: Mapped[str | None] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    entries: Mapped[list["LexiconEntry"]] = relationship(
        back_populates="label", cascade="all, delete-orphan"
    )


# ─────────────────────────────────────────────────────────────────────────────
# lexicon_entries
# ─────────────────────────────────────────────────────────────────────────────

class LexiconEntry(Base):
    __tablename__ = "lexicon_entries"

    entry_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    label_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("label_registry.label_id"), nullable=False
    )
    dict_type: Mapped[str] = mapped_column(String(8), nullable=False)   # "regex" | "ner"
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    surface_forms: Mapped[list | None] = mapped_column(JSON)
    regex_pattern: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="candidate"
    )  # candidate | active | quarantined | rejected | deprecated
    doc_freq: Mapped[int] = mapped_column(Integer, default=0)
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_score: Mapped[float | None] = mapped_column(Float)
    dict_version_added: Mapped[int | None] = mapped_column(Integer)
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    label: Mapped["LabelRegistry"] = relationship(back_populates="entries")

    __table_args__ = (
        UniqueConstraint("label_id", "dict_type", "lemma", name="uq_lexicon_label_type_lemma"),
        Index("ix_lexicon_label_status", "label_id", "status"),
        Index("ix_lexicon_lemma", "lemma"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# keyword_observations
# ─────────────────────────────────────────────────────────────────────────────

class KeywordObservation(Base):
    __tablename__ = "keyword_observations"

    obs_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    message_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("messages.message_id"), nullable=False
    )
    label_id: Mapped[str] = mapped_column(String(64), nullable=False)
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    count: Mapped[int] = mapped_column(Integer, default=1)
    embedding_score: Mapped[float | None] = mapped_column(Float)
    dict_version: Mapped[int] = mapped_column(Integer, nullable=False)
    promoted_to_active: Mapped[bool] = mapped_column(Boolean, default=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "message_id", "label_id", "lemma",
            name="uq_kw_observation_natural_key"
        ),
        Index("ix_kw_obs_label_lemma", "label_id", "lemma"),
        Index("ix_kw_obs_observed_at", "observed_at"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# entity_observations
# ─────────────────────────────────────────────────────────────────────────────

class EntityObservation(Base):
    __tablename__ = "entity_observations"

    obs_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    message_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("messages.message_id"), nullable=False
    )
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    value_hash: Mapped[str] = mapped_column(String(64), nullable=False)   # SHA-256
    value_encrypted: Mapped[str | None] = mapped_column(Text)             # Fernet
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    dict_version: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "message_id", "entity_type", "value_hash",
            name="uq_entity_observation_natural_key"
        ),
        Index("ix_entity_obs_type", "entity_type"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# promotion_events  (audit trail)
# ─────────────────────────────────────────────────────────────────────────────

class PromotionEvent(Base):
    __tablename__ = "promotion_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    label_id: Mapped[str] = mapped_column(String(64), nullable=False)
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    dict_type: Mapped[str] = mapped_column(String(8), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)  # promoted | quarantined | rejected
    from_status: Mapped[str] = mapped_column(String(16), nullable=False)
    to_status: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    dictionary_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_promotion_label_lemma", "label_id", "lemma"),
        Index("ix_promotion_created_at", "created_at"),
    )
