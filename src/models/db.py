"""
SQLAlchemy 2.0 async ORM models — allineati allo schema PostgreSQL del documento (sezione 5).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# pipeline_runs
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pipeline_version: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    dict_version_used: Mapped[int] = mapped_column(Integer, nullable=False)
    dict_version_new: Mapped[int | None] = mapped_column(Integer, nullable=True)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    # Metriche aggregate
    observations_created: Mapped[int] = mapped_column(Integer, default=0)
    entities_created: Mapped[int] = mapped_column(Integer, default=0)
    messages_processed: Mapped[int] = mapped_column(Integer, default=0)
    errors_count: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    keyword_observations: Mapped[list["KeywordObservation"]] = relationship(
        back_populates="run", lazy="noload"
    )


# ─────────────────────────────────────────────────────────────────────────────
# messages
# ─────────────────────────────────────────────────────────────────────────────

class Message(Base):
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # text_canonical NON viene persistito in DB per default (privacy); usare S3/vault se serve
    mittente: Mapped[str | None] = mapped_column(String(512), nullable=True)
    destinatario: Mapped[str | None] = mapped_column(String(512), nullable=True)
    lingua: Mapped[str | None] = mapped_column(String(8), nullable=True)
    oggetto_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # hash subject
    timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pii_flags: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    keyword_observations: Mapped[list["KeywordObservation"]] = relationship(
        back_populates="message", lazy="noload"
    )
    entity_observations: Mapped[list["EntityObservation"]] = relationship(
        back_populates="message", lazy="noload"
    )


# ─────────────────────────────────────────────────────────────────────────────
# keyword_observations
# ─────────────────────────────────────────────────────────────────────────────

class KeywordObservation(Base):
    __tablename__ = "keyword_observations"

    obs_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=True
    )
    label_id: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_id: Mapped[str] = mapped_column(String(64), nullable=False)
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    term: Mapped[str] = mapped_column(String(256), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    embedding_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    spans: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    evidence_quote_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dict_version: Mapped[int] = mapped_column(Integer, nullable=False)
    promoted_to_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Chiave naturale per idempotenza (sezione 3.4 doc)
        UniqueConstraint(
            "message_id", "label_id", "candidate_id", "dict_version",
            name="uq_kw_obs_natural_key",
        ),
        Index("ix_kw_obs_label_id", "label_id"),
        Index("ix_kw_obs_observed_at", "observed_at"),
        Index("ix_kw_obs_lemma", "lemma"),
        Index("ix_kw_obs_promoted", "promoted_to_active"),
    )

    message: Mapped["Message"] = relationship(back_populates="keyword_observations", lazy="noload")
    run: Mapped["PipelineRun | None"] = relationship(back_populates="keyword_observations", lazy="noload")


# ─────────────────────────────────────────────────────────────────────────────
# entity_observations
# ─────────────────────────────────────────────────────────────────────────────

class EntityObservation(Base):
    __tablename__ = "entity_observations"

    obs_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False
    )
    text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    start: Mapped[int] = mapped_column(Integer, nullable=False)
    end: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # regex|ner|lexicon|llm_ner
    extractor_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # PII: mai value in chiaro — solo hash (sezione 3.9 doc)
    value_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    value_enc: Mapped[str | None] = mapped_column(Text, nullable=True)  # AES-256 se necessario
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Chiave naturale per idempotenza (sezione 3.4 doc)
        UniqueConstraint(
            "message_id", "text_hash", "entity_type", "start", "end",
            "source", "extractor_version",
            name="uq_ent_obs_natural_key",
        ),
        Index("ix_ent_obs_message_id", "message_id"),
        Index("ix_ent_obs_entity_type", "entity_type"),
        Index("ix_ent_obs_observed_at", "observed_at"),
    )

    message: Mapped["Message"] = relationship(back_populates="entity_observations", lazy="noload")


# ─────────────────────────────────────────────────────────────────────────────
# label_registry
# ─────────────────────────────────────────────────────────────────────────────

class LabelRegistry(Base):
    __tablename__ = "label_registry"

    label_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active"
    )  # active | proposed | deprecated | merged
    merged_into: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lexicon_entries: Mapped[list["LexiconEntry"]] = relationship(
        back_populates="label", lazy="noload"
    )


# ─────────────────────────────────────────────────────────────────────────────
# lexicon_entries
# ─────────────────────────────────────────────────────────────────────────────

class LexiconEntry(Base):
    __tablename__ = "lexicon_entries"

    entry_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("label_registry.label_id"), nullable=False
    )
    dict_type: Mapped[str] = mapped_column(String(16), nullable=False)  # regex | ner
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    surface_forms: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    regex_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="candidate"
    )  # active | candidate | quarantined | rejected
    doc_freq: Mapped[int] = mapped_column(Integer, default=0)
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dict_version_added: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dict_version_deprecated: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quarantine_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)

    __table_args__ = (
        UniqueConstraint("label_id", "dict_type", "lemma", name="uq_lex_entry"),
        Index("ix_lex_label_status", "label_id", "status"),
        Index("ix_lex_lemma", "lemma"),
        Index("ix_lex_status", "status"),
    )

    label: Mapped["LabelRegistry"] = relationship(back_populates="lexicon_entries", lazy="noload")


# ─────────────────────────────────────────────────────────────────────────────
# promotion_events  (append-only log)
# ─────────────────────────────────────────────────────────────────────────────

class PromotionEvent(Base):
    __tablename__ = "promotion_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True  # promoter run_id — non referenzia pipeline_runs
    )
    label_id: Mapped[str] = mapped_column(String(128), nullable=False)
    lemma: Mapped[str] = mapped_column(String(256), nullable=False)
    dict_type: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # promoted_regex | promoted_ner | quarantined | rejected
    reason_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dict_version_prev: Mapped[int] = mapped_column(Integer, nullable=False)
    dict_version_new: Mapped[int] = mapped_column(Integer, nullable=False)
    doc_freq_at_promotion: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_score_at_promotion: Mapped[float | None] = mapped_column(Float, nullable=True)
    collision_labels: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_promo_run_id", "run_id"),
        Index("ix_promo_label_id", "label_id"),
        Index("ix_promo_dict_version_new", "dict_version_new"),
    )
