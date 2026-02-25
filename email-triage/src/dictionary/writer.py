"""
IdempotentWriter — writes keyword and entity observations with ON CONFLICT upsert.

Pattern: freeze-in-run / update-end-of-run
  - During a pipeline run, the dictionary is read-only.
  - Observations are buffered and flushed only after the run completes.
  - Idempotency: re-submitting the same message_id is safe.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.config import get_settings
from src.contracts import (
    Entity,
    EntityExtractionOutput,
    ObservationBatch,
    ObservationStats,
    TriageOutput,
)
from src.dictionary.models import (
    EntityObservation,
    KeywordObservation,
    Message,
    PipelineRun,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
settings = get_settings()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class IdempotentWriter:
    """
    Writes pipeline run results to the Observation Storage.

    All writes use PostgreSQL INSERT … ON CONFLICT DO UPDATE (upsert)
    keyed on natural business keys, making any write idempotent.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Public API ────────────────────────────────────────────────────────────

    def write_run(
        self,
        triage: TriageOutput,
        entity_output: EntityExtractionOutput,
        from_raw: str,
        subject: str = "",
    ) -> ObservationBatch:
        """
        Persist a complete pipeline run result:
        1. Upsert message metadata
        2. Insert pipeline_run record
        3. Upsert keyword observations
        4. Upsert entity observations

        Returns ObservationBatch acknowledgment.
        """
        run_id = str(uuid.uuid4())
        stats = ObservationStats()

        try:
            # Step 1: upsert message
            self._upsert_message(triage.message_id, from_raw, subject)

            # Step 2: insert pipeline run
            self._insert_pipeline_run(run_id, triage, entity_output)

            # Step 3: upsert keyword observations
            kw_count = self._upsert_keyword_observations(triage)
            stats.keywords_written = kw_count

            # Step 4: upsert entity observations
            ent_count = self._upsert_entity_observations(triage.message_id, entity_output)
            stats.entities_written = ent_count

            self._session.commit()
            logger.info(
                "Run %s committed: %d keywords, %d entities",
                run_id, kw_count, ent_count,
            )

        except Exception as exc:
            self._session.rollback()
            logger.error("Write failed for run %s: %s", run_id, exc)
            stats.errors.append(str(exc))

        return ObservationBatch(
            run_id=run_id,
            message_id=triage.message_id,
            stats=stats,
            dictionary_version_before=triage.dictionary_version,
            dictionary_version_after=triage.dictionary_version,  # updated by promoter
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _upsert_message(
        self, message_id: str, from_raw: str, subject: str
    ) -> None:
        retention_until = datetime.utcnow() + timedelta(days=settings.pii_retention_days)
        stmt = (
            pg_insert(Message)
            .values(
                message_id=message_id,
                from_hash=_sha256(from_raw),
                subject_redacted=_redact_subject(subject),
                retention_until=retention_until,
            )
            .on_conflict_do_nothing(index_elements=["message_id"])
        )
        self._session.execute(stmt)

    def _insert_pipeline_run(
        self,
        run_id: str,
        triage: TriageOutput,
        entity_output: EntityExtractionOutput,
    ) -> None:
        run = PipelineRun(
            run_id=run_id,
            message_id=triage.message_id,
            dictionary_version=triage.dictionary_version,
            model_version=triage.pipeline_version.get("model_version", ""),
            model_type=triage.pipeline_version.get("model_type", "chat"),
            parser_version=triage.pipeline_version.get("parser_version", ""),
            stoplist_version=triage.pipeline_version.get("stoplist_version", ""),
            ner_model_version=triage.pipeline_version.get("ner_model_version", ""),
            schema_version=triage.pipeline_version.get("schema_version", ""),
            topics=[
                {"label_id": t.label_id, "confidence": t.confidence}
                for t in triage.topics
            ],
            sentiment_value=triage.sentiment.value,
            sentiment_confidence=triage.sentiment.confidence,
            priority_value=triage.priority.value,
            priority_confidence=triage.priority.confidence,
            customer_status_value=triage.customer_status.value,
            customer_status_source=triage.customer_status.source,
            processing_time_ms=triage.processing_time_ms,
            keywords_extracted=sum(
                len(t.keywords_in_text) for t in triage.topics
            ),
            entities_extracted=len(entity_output.entities),
            warnings=triage.warnings,
        )
        self._session.add(run)

    def _upsert_keyword_observations(self, triage: TriageOutput) -> int:
        count = 0
        for topic in triage.topics:
            for kw in topic.keywords_in_text:
                stmt = (
                    pg_insert(KeywordObservation)
                    .values(
                        obs_id=str(uuid.uuid4()),
                        message_id=triage.message_id,
                        label_id=topic.label_id,
                        lemma=kw.lemma,
                        count=kw.count,
                        embedding_score=None,
                        dict_version=triage.dictionary_version,
                        promoted_to_active=False,
                    )
                    .on_conflict_do_update(
                        constraint="uq_kw_observation_natural_key",
                        set_={
                            "count": kw.count,
                            "dict_version": triage.dictionary_version,
                        },
                    )
                )
                self._session.execute(stmt)
                count += 1
        return count

    def _upsert_entity_observations(
        self,
        message_id: str,
        entity_output: EntityExtractionOutput,
    ) -> int:
        from src.privacy.pii_handler import PIIHandler
        pii_handler = PIIHandler()
        count = 0

        for ent in entity_output.entities:
            value_hash = _sha256(ent.value)
            value_encrypted: str | None = None
            if pii_handler.is_encryption_available():
                value_encrypted = pii_handler.encrypt_value(ent.value)

            stmt = (
                pg_insert(EntityObservation)
                .values(
                    obs_id=str(uuid.uuid4()),
                    message_id=message_id,
                    entity_type=ent.entity_type,
                    value_hash=value_hash,
                    value_encrypted=value_encrypted,
                    source=ent.source,
                    confidence=ent.confidence,
                    dict_version=settings.default_dictionary_version,
                )
                .on_conflict_do_update(
                    constraint="uq_entity_observation_natural_key",
                    set_={"confidence": ent.confidence},
                )
            )
            self._session.execute(stmt)
            count += 1

        return count


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _redact_subject(subject: str) -> str:
    """Remove obvious PII patterns from subject line before storage."""
    import re

    patterns = [
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",       # email
        r"\b[A-Z]{6}\d{2}[A-EHLMPRST]\d{2}[A-Z]\d{3}[A-Z]\b",           # C.F.
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{0,16}\b",              # IBAN
        r"(?:\+39\s?)?\d[\d\s\-]{8,15}",                                  # phone
    ]
    result = subject
    for pat in patterns:
        result = re.sub(pat, "[REDACTED]", result, flags=re.IGNORECASE)
    return result
