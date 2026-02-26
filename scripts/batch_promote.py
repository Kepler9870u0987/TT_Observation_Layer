#!/usr/bin/env python
"""
scripts/batch_promote.py

Batch job per la promozione fine-run dei dizionari.
Eseguibile manualmente o schedulato da Celery Beat.

Usage:
    python -m scripts.batch_promote [--dict-version N]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("batch_promote")


async def main(dict_version: int | None = None) -> int:
    """
    Legge osservazioni non promosse, esegue il Promoter, crea nuova versione.
    Ritorna il nuovo dict_version.
    """
    from sqlalchemy import text

    from src.db.session import AsyncSessionLocal
    from src.dictionary.collision_detector import (
        compute_in_memory_collision_rate,
        find_colliding_lemmas,
    )
    from src.dictionary.promoter import KeywordPromoter
    from src.dictionary.versioning import create_new_version
    from src.observation_store.reader import get_observations_since

    async with AsyncSessionLocal() as db:
        # Determina versione corrente se non passata
        if dict_version is None:
            result = await db.execute(
                text("SELECT COALESCE(MAX(dict_version_added), 0) FROM lexicon_entries")
            )
            dict_version = int(result.scalar() or 0)

        logger.info("Starting promoter batch for dict_version=%d", dict_version)

        observations = await get_observations_since(db, dict_version=dict_version)
        logger.info("Found %d unpromoted observations", len(observations))

        if not observations:
            logger.info("Nothing to promote. Exiting.")
            return dict_version

        promoter = KeywordPromoter()
        updates = promoter.promote_keywords(observations)
        collision_index = promoter.compute_collision_index(
            promoter.compute_stats(observations)
        )
        collision_rate = compute_in_memory_collision_rate(collision_index)
        colliding = find_colliding_lemmas(collision_index)

        logger.info(
            "Promoter results: regex_active=%d  ner_active=%d  quarantined=%d  rejected=%d",
            len(updates["regex_active"]),
            len(updates["ner_active"]),
            len(updates["quarantined"]),
            len(updates["rejected"]),
        )
        logger.info("Collision rate: %.4f", collision_rate)
        if colliding:
            logger.warning("Colliding lemmas (%d): %s", len(colliding), list(colliding.keys())[:10])

        run_id = str(uuid4())
        new_version = await create_new_version(db, updates, dict_version, run_id)
        logger.info("Dictionary updated: v%d → v%d", dict_version, new_version)

        # Alert thresholds (sezione 6 doc)
        if collision_rate > 0.15:
            logger.warning("ALERT: collision_rate=%.4f > 0.15", collision_rate)

        total_obs = len(observations)
        promoted_total = len(updates["regex_active"]) + len(updates["ner_active"])
        promotion_rate = promoted_total / total_obs if total_obs > 0 else 0.0
        if promotion_rate < 0.02:
            logger.warning("ALERT: promotion_rate=%.4f < 0.02 (soglie troppo restrittive?)", promotion_rate)
        elif promotion_rate > 0.50:
            logger.warning("ALERT: promotion_rate=%.4f > 0.50 (soglie troppo permissive?)", promotion_rate)

        return new_version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run nightly dictionary promoter")
    parser.add_argument("--dict-version", type=int, default=None, help="Dictionary version to process")
    args = parser.parse_args()

    new_v = asyncio.run(main(args.dict_version))
    logger.info("Done. New dict_version=%d", new_v)
    sys.exit(0)
