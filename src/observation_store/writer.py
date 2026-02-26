"""
Observation Storage Writer.

persist_batch() è il punto di ingresso principale: prende un ObservationBatch
prodotto dal triage, lo persiste su PostgreSQL in una singola transazione
idempotente (ON CONFLICT DO NOTHING su chiavi naturali — sezione 3.4 del doc).
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.contracts import (
    ExtractedEntity,
    ObservationBatch,
    ObservationBatchAck,
    ObservationRecord,
    PII_TYPES,
)
from src.models.pipeline_version import PipelineVersion


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _build_pipeline_version(payload: dict[str, Any]) -> PipelineVersion:
    return PipelineVersion.from_dict(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Upsert helpers (raw SQL for ON CONFLICT DO NOTHING performance)
# ─────────────────────────────────────────────────────────────────────────────

_INSERT_MESSAGE = text("""
    INSERT INTO messages (message_id, text_hash, mittente, destinatario, lingua,
                          oggetto_hash, timestamp, pii_flags, created_at)
    VALUES (:message_id, :text_hash, :mittente, :destinatario, :lingua,
            :oggetto_hash, :timestamp, :pii_flags, :created_at)
    ON CONFLICT (message_id) DO NOTHING
""")

_INSERT_KW_OBS = text("""
    INSERT INTO keyword_observations
        (obs_id, message_id, run_id, label_id, candidate_id, lemma, term, count,
         embedding_score, spans, evidence_quote_hash, dict_version,
         promoted_to_active, observed_at, expires_at)
    VALUES
        (:obs_id, :message_id, :run_id, :label_id, :candidate_id, :lemma, :term,
         :count, :embedding_score, :spans, :evidence_quote_hash, :dict_version,
         :promoted_to_active, :observed_at, :expires_at)
    ON CONFLICT ON CONSTRAINT uq_kw_obs_natural_key DO NOTHING
""")

_INSERT_ENT_OBS = text("""
    INSERT INTO entity_observations
        (obs_id, message_id, text_hash, entity_type, start, "end", source,
         extractor_version, confidence, value_hash, value_enc, observed_at, expires_at)
    VALUES
        (:obs_id, :message_id, :text_hash, :entity_type, :start, :end, :source,
         :extractor_version, :confidence, :value_hash, :value_enc, :observed_at, :expires_at)
    ON CONFLICT ON CONSTRAINT uq_ent_obs_natural_key DO NOTHING
""")

_INSERT_RUN = text("""
    INSERT INTO pipeline_runs
        (run_id, started_at, pipeline_version, dict_version_used, schema_version, status)
    VALUES
        (:run_id, :started_at, :pipeline_version, :dict_version_used, :schema_version, 'running')
    ON CONFLICT (run_id) DO NOTHING
""")

_UPDATE_RUN = text("""
    UPDATE pipeline_runs SET
        finished_at = :finished_at,
        status = :status,
        observations_created = observations_created + :obs_created,
        entities_created = entities_created + :ent_created,
        messages_processed = messages_processed + 1,
        errors_count = errors_count + :errors
    WHERE run_id = :run_id
""")


# ─────────────────────────────────────────────────────────────────────────────
# _persist_observations
# ─────────────────────────────────────────────────────────────────────────────

async def _persist_observations(
    db: AsyncSession,
    observations: list[ObservationRecord],
    run_id: str,
    import_json,  # json stdlib — passed to avoid re-import
) -> tuple[int, int]:
    """Insert keyword observations. Returns (created, skipped)."""
    created = 0
    skipped = 0
    now = _now_utc()

    for rec in observations:
        result = await db.execute(
            _INSERT_KW_OBS,
            {
                "obs_id": rec.obs_id,
                "message_id": rec.message_id,
                "run_id": run_id,
                "label_id": rec.labelid,
                "candidate_id": rec.candidateid,
                "lemma": rec.lemma,
                "term": rec.term,
                "count": rec.count,
                "embedding_score": rec.embeddingscore,
                "spans": None,
                "evidence_quote_hash": None,
                "dict_version": rec.dict_version,
                "promoted_to_active": rec.promoted_to_active,
                "observed_at": rec.observed_at or now,
                "expires_at": None,
            },
        )
        rows = result.rowcount
        if rows and rows > 0:
            created += 1
        else:
            skipped += 1

    return created, skipped


# ─────────────────────────────────────────────────────────────────────────────
# _persist_entities
# ─────────────────────────────────────────────────────────────────────────────

async def _persist_entities(
    db: AsyncSession,
    entities: list[ExtractedEntity],
    message_id: str,
    text_hash: str | None,
) -> tuple[int, int]:
    """Insert entity observations. Returns (created, skipped)."""
    created = 0
    skipped = 0
    now = _now_utc()

    for ent in entities:
        # PII guard — se type è PII, raw text non deve essere nel payload
        is_pii = ent.label in PII_TYPES
        value_hash = ent.value_hash
        if is_pii and value_hash is None and ent.text is not None:
            value_hash = _sha256(ent.text)

        result = await db.execute(
            _INSERT_ENT_OBS,
            {
                "obs_id": str(uuid4()),
                "message_id": message_id,
                "text_hash": text_hash,
                "entity_type": ent.label,
                "start": ent.start,
                "end": ent.end,
                "source": ent.source,
                "extractor_version": ent.extractor_version or "",
                "confidence": ent.confidence,
                "value_hash": value_hash,
                "value_enc": None,
                "observed_at": now,
                "expires_at": None,
            },
        )
        rows = result.rowcount
        if rows and rows > 0:
            created += 1
        else:
            skipped += 1

    return created, skipped


# ─────────────────────────────────────────────────────────────────────────────
# persist_batch  — public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def persist_batch(
    db: AsyncSession,
    batch: ObservationBatch,
    run_id: str,
) -> ObservationBatchAck:
    """
    Persiste un ObservationBatch in una singola transazione PostgreSQL.

    Idempotente: ON CONFLICT DO NOTHING su chiavi naturali.
    Ritorna ObservationBatchAck con conteggi e warnings.
    """
    import json

    t_start = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []

    pp = batch.postprocessing
    pv = PipelineVersion.from_dict(pp.pipeline_version.model_dump())
    message_id = pp.message_id or batch.email_context.message_id
    text_canonical = batch.email_context.testo_normalizzato or ""
    text_hash = _sha256(text_canonical) if text_canonical else None

    try:
        async with db.begin():
            # 1. Registra pipeline run
            await db.execute(
                _INSERT_RUN,
                {
                    "run_id": run_id,
                    "started_at": _now_utc(),
                    "pipeline_version": json.dumps(pv.to_dict()),
                    "dict_version_used": pv.dictionary_version,
                    "schema_version": pv.schema_version,
                },
            )

            # 2. Upsert message
            subject = batch.email_context.oggetto
            await db.execute(
                _INSERT_MESSAGE,
                {
                    "message_id": message_id,
                    "text_hash": text_hash,
                    "mittente": batch.email_context.mittente,
                    "destinatario": batch.email_context.destinatario,
                    "lingua": batch.email_context.lingua,
                    "oggetto_hash": _sha256(subject) if subject else None,
                    "timestamp": batch.email_context.timestamp,
                    "pii_flags": json.dumps({}),
                    "created_at": _now_utc(),
                },
            )

            # 3. Keyword observations
            kw_created, kw_skipped = await _persist_observations(
                db, pp.observations, run_id, json
            )

            # 4. Entity observations — usa ner_entities se disponibile,
            #    altrimenti pp.entities
            raw_entities: list[ExtractedEntity] = []
            if batch.ner_entities and batch.ner_entities.entities:
                raw_entities = batch.ner_entities.entities
            elif pp.entities:
                raw_entities = pp.entities

            ent_created, ent_skipped = await _persist_entities(
                db, raw_entities, message_id, text_hash
            )

            # 5. Diagnostics warnings
            for w in pp.diagnostics.warnings:
                warnings.append(w)

            # 6. Aggiorna run
            await db.execute(
                _UPDATE_RUN,
                {
                    "run_id": run_id,
                    "finished_at": _now_utc(),
                    "status": "completed",
                    "obs_created": kw_created,
                    "ent_created": ent_created,
                    "errors": len(errors),
                },
            )

    except Exception as exc:
        errors.append(str(exc))
        return ObservationBatchAck(
            run_id=run_id,
            message_id=message_id,
            dict_version_used=pv.dictionary_version,
            errors=errors,
            processing_time_ms=round((time.perf_counter() - t_start) * 1000, 2),
        )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return ObservationBatchAck(
        run_id=run_id,
        message_id=message_id,
        dict_version_used=pv.dictionary_version,
        observations_created=kw_created,
        observations_skipped_idempotent=kw_skipped,
        entities_created=ent_created,
        entities_skipped_idempotent=ent_skipped,
        warnings=warnings,
        errors=errors,
        processing_time_ms=elapsed_ms,
    )
