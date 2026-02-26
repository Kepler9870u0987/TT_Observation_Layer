"""Celery tasks for nightly batch operations."""
from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from src.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(name="src.worker.tasks.run_nightly_promoter", bind=True, max_retries=3)
def run_nightly_promoter(self):
    """Trigger the promoter via the API or directly via the service layer."""
    from src.db.session import AsyncSessionLocal
    from src.dictionary.collision_detector import get_collision_rate_from_db
    from src.dictionary.promoter import KeywordPromoter
    from src.dictionary.versioning import create_new_version
    from src.observation_store.reader import get_observations_since
    from sqlalchemy import text

    async def _run():
        async with AsyncSessionLocal() as db:
            # Get current dict version
            result = await db.execute(
                text("SELECT COALESCE(MAX(dict_version_added), 0) FROM lexicon_entries")
            )
            current_version = int(result.scalar() or 0)

            observations = await get_observations_since(db, dict_version=current_version)
            if not observations:
                logger.info("Promoter: no unpromoted observations for version %d", current_version)
                return {"status": "no_observations", "dict_version": current_version}

            promoter = KeywordPromoter()
            updates = promoter.promote_keywords(observations)
            run_id = str(uuid4())
            new_version = await create_new_version(db, updates, current_version, run_id)

            collision_rate = await get_collision_rate_from_db(db)

            logger.info(
                "Promoter completed: v%d → v%d | regex=%d ner=%d quarantined=%d rejected=%d collision=%.3f",
                current_version, new_version,
                len(updates["regex_active"]), len(updates["ner_active"]),
                len(updates["quarantined"]), len(updates["rejected"]),
                collision_rate,
            )
            return {
                "dict_version_prev": current_version,
                "dict_version_new": new_version,
                "promoted_regex": len(updates["regex_active"]),
                "promoted_ner": len(updates["ner_active"]),
                "quarantined": len(updates["quarantined"]),
                "rejected": len(updates["rejected"]),
                "collision_rate": round(collision_rate, 4),
            }

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("Promoter failed: %s", exc)
        raise self.retry(exc=exc, countdown=300)


@celery_app.task(name="src.worker.tasks.apply_retention_policy", bind=True, max_retries=3)
def apply_retention_policy(self):
    """Delete expired observations (GDPR retention)."""
    from src.db.session import AsyncSessionLocal
    from datetime import datetime, timezone
    from sqlalchemy import text

    async def _run():
        now = datetime.now(tz=timezone.utc)
        async with AsyncSessionLocal() as db:
            async with db.begin():
                kw_result = await db.execute(
                    text("DELETE FROM keyword_observations WHERE expires_at IS NOT NULL AND expires_at < :now"),
                    {"now": now},
                )
                ent_result = await db.execute(
                    text("DELETE FROM entity_observations WHERE expires_at IS NOT NULL AND expires_at < :now"),
                    {"now": now},
                )
                logger.info(
                    "Retention: deleted %d keyword_obs, %d entity_obs",
                    kw_result.rowcount, ent_result.rowcount,
                )
                return {
                    "keyword_observations_deleted": kw_result.rowcount,
                    "entity_observations_deleted": ent_result.rowcount,
                }

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("Retention policy failed: %s", exc)
        raise self.retry(exc=exc, countdown=60)
