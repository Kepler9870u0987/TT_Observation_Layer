"""
Contratti Pydantic v2 per l'Observation Storage Layer.

Tre contratti versionati (sezione 3.3 del documento):
  - EntityExtractionOutput  : entità estratte dal triage
  - TriageOutput            : decisioni di triage + evidence + keyword
  - ObservationBatch        : payload POST /observations/batch
  - ObservationBatchAck     : risposta (ack) del layer di persistenza
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Primitive condivise
# ─────────────────────────────────────────────────────────────────────────────

class Span(BaseModel):
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def end_gt_start(self) -> "Span":
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must be > start ({self.start})")
        return self


class PipelineVersionPayload(BaseModel):
    """Payload serializzato di PipelineVersion ricevuto dal triage."""
    dictionaryversion: int
    modelversion: str
    model_type: Literal["chat", "reasoning"] = "chat"
    parserversion: str
    stoplistversion: str
    nermodelversion: str
    schemaversion: str
    toolcallingversion: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# EntityExtractionOutput
# ─────────────────────────────────────────────────────────────────────────────

PII_TYPES = frozenset({
    "CODICEFISCALE", "EMAIL", "TELEFONO", "IBAN",
    "PARTITAIVA", "CARTACREDITO",
})


class ExtractedEntity(BaseModel):
    text: str | None = None        # raw value — omitted / hashed for PII types
    label: str                     # tipo entità (CODICEFISCALE, PER, ORG, …)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    source: Literal["regex", "ner", "lexicon", "llm_ner"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    extractor_version: str = ""
    value_hash: str | None = None  # SHA-256 del valore raw (per PII)

    @model_validator(mode="after")
    def pii_must_have_hash(self) -> "ExtractedEntity":
        if self.label in PII_TYPES and self.value_hash is None:
            raise ValueError(
                f"entity type '{self.label}' is PII — value_hash is required, "
                "raw text must be omitted or None"
            )
        return self


class EntityExtractionMeta(BaseModel):
    id_conversazione: str
    id_messaggio: str
    status: str
    layer_version: str
    processing_time_ms: float
    component_timings_ms: dict[str, float] = Field(default_factory=dict)
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    entity_count: int
    errors: list[str] = Field(default_factory=list)


class EntityExtractionOutput(BaseModel):
    entities: list[ExtractedEntity]
    meta: EntityExtractionMeta


# ─────────────────────────────────────────────────────────────────────────────
# TriageOutput
# ─────────────────────────────────────────────────────────────────────────────

class KeywordInText(BaseModel):
    candidateid: str
    lemma: str
    term: str
    count: int = Field(..., ge=1)
    source: Literal["subject", "body", "header", "attachment"]
    embeddingscore: float | None = Field(default=None, ge=0.0, le=1.0)


class EvidenceItem(BaseModel):
    quote: str = Field(..., max_length=400)
    span: list[int] | None = None       # [start, end] su text_canonical
    span_llm: list[int] | None = None   # span originale LLM (diagnostica)
    span_status: str | None = None      # exact_match | fuzzy | not_found
    text_hash: str | None = None        # hash del text_canonical usato

    @field_validator("span", "span_llm", mode="before")
    @classmethod
    def validate_span(cls, v: Any) -> Any:
        if v is not None:
            if len(v) != 2:
                raise ValueError("span must be [start, end]")
            if v[0] >= v[1]:
                raise ValueError(f"span start ({v[0]}) must be < end ({v[1]})")
        return v


class TopicResult(BaseModel):
    labelid: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_llm: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_adjusted: float | None = Field(default=None, ge=0.0, le=1.0)
    keywordsintext: list[KeywordInText] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    keywords: list[KeywordInText] = Field(default_factory=list)  # alias/copia per compat


class SentimentResult(BaseModel):
    value: Literal["positive", "neutral", "negative"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class PriorityResult(BaseModel):
    value: Literal["low", "medium", "high", "urgent"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    rawscore: float | None = None


class CustomerStatusResult(BaseModel):
    value: Literal["new", "existing", "unknown"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str = ""


class TriageResult(BaseModel):
    topics: list[TopicResult]
    sentiment: SentimentResult
    priority: PriorityResult
    customerstatus: CustomerStatusResult


# ─────────────────────────────────────────────────────────────────────────────
# Observation record (già prodotto dal post-processing del triage)
# ─────────────────────────────────────────────────────────────────────────────

class ObservationRecord(BaseModel):
    obs_id: str
    message_id: str
    labelid: str
    candidateid: str
    lemma: str
    term: str
    count: int = Field(..., ge=1)
    embeddingscore: float | None = Field(default=None, ge=0.0, le=1.0)
    dict_version: int
    promoted_to_active: bool = False
    observed_at: datetime


class ProcessingDiagnostics(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    validation_retries: int = 0
    fallback_applied: bool = False


class ProcessingMetadata(BaseModel):
    postprocessing_duration_ms: float | None = None
    entities_extracted: int = 0
    observations_created: int = 0
    confidence_adjustments_applied: int = 0
    span_exact_match_count: int = 0
    span_fuzzy_match_count: int = 0
    span_not_found_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Email context
# ─────────────────────────────────────────────────────────────────────────────

class EmailContext(BaseModel):
    message_id: str
    id_conversazione: str | None = None
    testo_normalizzato: str | None = None
    mittente: str | None = None
    destinatario: str | None = None
    timestamp: datetime | None = None
    lingua: str | None = None
    oggetto: str | None = None
    allegati: list[Any] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# ObservationBatch — payload POST /observations/batch
# ─────────────────────────────────────────────────────────────────────────────

class PostProcessingBlock(BaseModel):
    message_id: str
    pipeline_version: PipelineVersionPayload
    triage: TriageResult
    entities: list[ExtractedEntity] = Field(default_factory=list)
    observations: list[ObservationRecord] = Field(default_factory=list)
    diagnostics: ProcessingDiagnostics = Field(default_factory=ProcessingDiagnostics)
    processing_metadata: ProcessingMetadata = Field(default_factory=ProcessingMetadata)


class ObservationBatch(BaseModel):
    """
    Payload principale del POST /observations/batch.
    Corrisponde alla struttura message_envelope.json prodotta dal triage.
    """
    email_context: EmailContext
    triage: TriageResult
    postprocessing: PostProcessingBlock
    ner_entities: EntityExtractionOutput | None = None


# ─────────────────────────────────────────────────────────────────────────────
# ObservationBatchAck — risposta del layer di persistenza
# ─────────────────────────────────────────────────────────────────────────────

class ObservationBatchAck(BaseModel):
    """
    Contratto di risposta (ack) dell'Observation Storage.
    Sezione 3.3 del documento: esito di persistenza, conteggi, dedup, errori/warning.
    """
    run_id: str
    message_id: str
    dict_version_used: int
    observations_created: int = 0
    observations_skipped_idempotent: int = 0
    entities_created: int = 0
    entities_skipped_idempotent: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    processing_time_ms: float | None = None
