"""
FastAPI routes for the Observation Layer.

POST /observations/batch   — persist a triage output batch
GET  /dictionaries/health  — dictionary health snapshot
POST /promoter/run         — trigger a promoter run (usually called by the batch job)
"""
from __future__ import annotations

import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.metrics import (
    BATCH_ERRORS_TOTAL,
    BATCH_PERSIST_DURATION,
    COLLISION_RATE,
    DICT_SIZE_BY_LABEL,
    ENTITIES_TOTAL,
    OBSERVATIONS_TOTAL,
    PROMOTER_DURATION,
    PROMOTION_EVENTS_TOTAL,
    QUARANTINED_TOTAL,
)
from src.db.session import get_db
from src.dictionary.collision_detector import (
    get_collision_rate_from_db,
)
from src.dictionary.promoter import KeywordPromoter
from src.dictionary.versioning import create_new_version
from src.models.contracts import ObservationBatch, ObservationBatchAck
from src.observation_store.reader import (
    get_dictionary_health,
    get_observations_since,
)
from src.observation_store.writer import persist_batch

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# POST /observations/batch
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/observations/batch",
    response_model=ObservationBatchAck,
    status_code=status.HTTP_201_CREATED,
    tags=["observations"],
    summary="Persist a triage observation batch",
)
async def post_observations_batch(
    batch: ObservationBatch,
    db: AsyncSession = Depends(get_db),
) -> ObservationBatchAck:
    """
    Persists keyword and entity observations from a triage message_envelope.

    Idempotent: sending the same message twice returns observations_created=0,
    skipped_idempotent=N on the second call.
    """
    run_id = str(uuid4())
    t_start = time.perf_counter()

    try:
        with BATCH_PERSIST_DURATION.time():
            ack = await persist_batch(db, batch, run_id)
    except Exception as exc:
        BATCH_ERRORS_TOTAL.inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # Update Prometheus counters
    if ack.observations_created > 0:
        # Increment per-label counters from topics in batch
        for topic in batch.postprocessing.triage.topics:
            OBSERVATIONS_TOTAL.labels(label_id=topic.labelid).inc(
                len(topic.keywordsintext)
            )
    for ent in batch.postprocessing.entities:
        ENTITIES_TOTAL.labels(entity_type=ent.label).inc()

    if ack.errors:
        BATCH_ERRORS_TOTAL.inc(len(ack.errors))

    return ack


# ─────────────────────────────────────────────────────────────────────────────
# GET /dictionaries/health
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/dictionaries/health",
    tags=["dictionaries"],
    summary="Dictionary health snapshot",
)
async def get_dict_health(db: AsyncSession = Depends(get_db)) -> dict:
    """Returns dictionary size, collision rate, quarantine count, per-label breakdown."""
    health = await get_dictionary_health(db)

    # Update Prometheus gauges
    COLLISION_RATE.set(health["collision_rate"])
    QUARANTINED_TOTAL.set(health["total_quarantined_entries"])
    for label, stats in health["by_label"].items():
        DICT_SIZE_BY_LABEL.labels(label_id=label).set(stats.get("active", 0))

    return health


# ─────────────────────────────────────────────────────────────────────────────
# POST /promoter/run
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/promoter/run",
    tags=["promoter"],
    summary="Trigger a promoter run to update dictionaries",
)
async def run_promoter(
    dict_version: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Reads unpromoted observations for dict_version, runs the KeywordPromoter,
    and writes a new dictionary version atomically.

    Called by the nightly batch job (scripts/batch_promote.py).
    """
    run_id = str(uuid4())

    try:
        with PROMOTER_DURATION.time():
            observations = await get_observations_since(db, dict_version=dict_version)
            if not observations:
                return {
                    "run_id": run_id,
                    "message": "No unpromoted observations found",
                    "dict_version_new": dict_version,
                }

            promoter = KeywordPromoter()
            updates = promoter.promote_keywords(observations)

            collision_rate = await get_collision_rate_from_db(db)
            COLLISION_RATE.set(collision_rate)

            new_version = await create_new_version(db, updates, dict_version, run_id)

    except RuntimeError as exc:
        # Advisory lock busy
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # Prometheus
    PROMOTION_EVENTS_TOTAL.labels(action="promoted_regex").inc(len(updates["regex_active"]))
    PROMOTION_EVENTS_TOTAL.labels(action="promoted_ner").inc(len(updates["ner_active"]))
    PROMOTION_EVENTS_TOTAL.labels(action="quarantined").inc(len(updates["quarantined"]))
    PROMOTION_EVENTS_TOTAL.labels(action="rejected").inc(len(updates["rejected"]))

    return {
        "run_id": run_id,
        "dict_version_prev": dict_version,
        "dict_version_new": new_version,
        "promoted_regex": len(updates["regex_active"]),
        "promoted_ner": len(updates["ner_active"]),
        "quarantined": len(updates["quarantined"]),
        "rejected": len(updates["rejected"]),
        "collision_rate": round(collision_rate, 4),
    }
