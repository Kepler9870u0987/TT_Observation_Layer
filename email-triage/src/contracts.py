"""
Pydantic v3.0 data contracts for the email triage pipeline.

All objects crossing module boundaries must be instances of these models.
Frozen dataclasses / Pydantic models guarantee structural immutability and
enable hashing / caching for determinism guarantees.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline version (determinism contract)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineVersion:
    """
    Immutable contract of all versioned components.
    Saved with every run so any output can be reproduced.
    """
    dictionary_version: int
    model_version: str          # e.g. "openai/gpt-4o-2025-11-20"
    model_type: Literal["chat", "reasoning"]
    parser_version: str         # e.g. "email-parser-1.3.0"
    stoplist_version: str       # e.g. "stopwords-it-2025.1"
    ner_model_version: str      # e.g. "it_core_news_lg-3.7.1"
    schema_version: str         # e.g. "json-schema-v3.0"

    def __str__(self) -> str:
        return (
            f"Pipeline-dict{self.dictionary_version}"
            f"-{self.model_version.split('/')[-1]}"
            f"-{self.schema_version}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dictionary_version": self.dictionary_version,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "parser_version": self.parser_version,
            "stoplist_version": self.stoplist_version,
            "ner_model_version": self.ner_model_version,
            "schema_version": self.schema_version,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RemovedSection:
    """Tracks what was stripped during canonicalization (for audit)."""
    section_type: str   # "quote" | "signature" | "disclaimer" | "reply_header"
    span_start: int
    span_end: int
    content: str


class EmailDocument(BaseModel):
    """Canonical email representation after parsing and preprocessing."""
    message_id: str
    from_raw: str
    to_raw: str = ""
    subject: str = ""
    body: str                   # original body (kept for audit)
    body_canonical: str         # cleaned body (quote/sig stripped)
    removed_sections: list[dict[str, Any]] = Field(default_factory=list)
    language: str = "it"
    parser_version: str
    canonicalization_version: str
    received_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def text_hash(self) -> str:
        raw = f"{self.message_id}|{self.body_canonical}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def full_text(self) -> str:
        return f"{self.subject}\n{self.body_canonical}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Candidates
# ─────────────────────────────────────────────────────────────────────────────

class Candidate(BaseModel):
    """A keyword candidate extracted deterministically before LLM."""
    candidate_id: str           # stable SHA1 of (source, term)
    source: Literal["subject", "body"]
    term: str
    lemma: str
    count: int
    embedding_score: float = 0.0
    composite_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Triage output (v3.0)
# ─────────────────────────────────────────────────────────────────────────────

class KeywordMatch(BaseModel):
    candidate_id: str
    lemma: str
    count: int = Field(ge=1)
    spans: list[list[int]] = Field(default_factory=list)

    @field_validator("spans")
    @classmethod
    def validate_spans(cls, v: list[list[int]]) -> list[list[int]]:
        for span in v:
            if len(span) != 2 or span[0] < 0 or span[1] <= span[0]:
                raise ValueError(f"Invalid span: {span}")
        return v


class Evidence(BaseModel):
    quote: str = Field(max_length=200)
    span: list[int] | None = None  # [start, end)

    @field_validator("span")
    @classmethod
    def validate_span(cls, v: list[int] | None) -> list[int] | None:
        if v is not None:
            if len(v) != 2 or v[0] < 0 or v[1] <= v[0]:
                raise ValueError(f"Invalid span: {v}")
        return v


# Topics enum — extend here as domain grows
TOPICS_ENUM: list[str] = [
    "FATTURAZIONE",
    "ASSISTENZA_TECNICA",
    "RECLAMO",
    "INFO_COMMERCIALI",
    "DOCUMENTI",
    "APPUNTAMENTO",
    "CONTRATTO",
    "GARANZIA",
    "SPEDIZIONE",
    "UNKNOWN_TOPIC",
]


class TopicAssignment(BaseModel):
    label_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_llm: float | None = None   # raw LLM confidence (before adjustment)
    keywords_in_text: list[KeywordMatch] = Field(default_factory=list, min_length=1)
    evidence: list[Evidence] = Field(default_factory=list, min_length=1)

    @field_validator("label_id")
    @classmethod
    def label_must_be_known(cls, v: str) -> str:
        if v not in TOPICS_ENUM:
            raise ValueError(f"Unknown label_id: {v!r}. Must be one of {TOPICS_ENUM}")
        return v


class CustomerStatus(BaseModel):
    value: Literal["new", "existing", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = ""    # "crm_exact_match" | "crm_domain_match" | "text_signal" | etc.


class Sentiment(BaseModel):
    value: Literal["positive", "neutral", "negative"]
    confidence: float = Field(ge=0.0, le=1.0)


class PrioritySignal(BaseModel):
    value: Literal["low", "medium", "high", "urgent"]
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    raw_score: float = 0.0


class TriageOutput(BaseModel):
    """Full triage result for one email."""
    message_id: str
    topics: list[TopicAssignment] = Field(min_length=1, max_length=5)
    sentiment: Sentiment
    priority: PrioritySignal
    customer_status: CustomerStatus
    dictionary_version: int
    pipeline_version: dict[str, Any]
    processing_time_ms: float | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def deduplicate_topics(self) -> "TriageOutput":
        seen: set[str] = set()
        unique: list[TopicAssignment] = []
        for t in self.topics:
            if t.label_id not in seen:
                seen.add(t.label_id)
                unique.append(t)
        self.topics = unique
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction (v3.0)
# ─────────────────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    entity_id: str              # stable id
    entity_type: str            # CODICE_FISCALE | IBAN | EMAIL | TELEFONO | PER | etc.
    value: str                  # raw value (will be PII-minimized before storage)
    value_normalized: str = ""
    span_start: int
    span_end: int
    source: Literal["regex", "spacy", "llm"]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class EntityExtractionOutput(BaseModel):
    message_id: str
    entities: list[Entity] = Field(default_factory=list)
    component_timings_ms: dict[str, float] = Field(default_factory=dict)
    ner_model_version: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Observation Storage
# ─────────────────────────────────────────────────────────────────────────────

class ObservationStats(BaseModel):
    keywords_written: int = 0
    entities_written: int = 0
    keywords_promoted: int = 0
    keywords_quarantined: int = 0
    errors: list[str] = Field(default_factory=list)


class ObservationBatch(BaseModel):
    """Acknowledgment returned after writing observations to storage."""
    run_id: str
    message_id: str
    stats: ObservationStats
    dictionary_version_before: int
    dictionary_version_after: int
    processed_at: datetime = Field(default_factory=datetime.utcnow)
