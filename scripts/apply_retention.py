#!/usr/bin/env python
"""
scripts/apply_retention.py

Cancella le osservazioni scadute (GDPR retention policy).
Da eseguire quotidianamente (schedulato da Celery Beat o cron).

Usage:
    python -m scripts.apply_retention
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("apply_retention")


async def main() -> dict:
    from sqlalchemy import text

    from src.db.session import AsyncSessionLocal

    now = datetime.now(tz=timezone.utc)
    logger.info("Applying retention policy at %s", now.isoformat())

    async with AsyncSessionLocal() as db:
        async with db.begin():
            kw = await db.execute(
                text(
                    "DELETE FROM keyword_observations "
                    "WHERE expires_at IS NOT NULL AND expires_at < :now"
                ),
                {"now": now},
            )
            ent = await db.execute(
                text(
                    "DELETE FROM entity_observations "
                    "WHERE expires_at IS NOT NULL AND expires_at < :now"
                ),
                {"now": now},
            )

    result = {
        "keyword_observations_deleted": kw.rowcount,
        "entity_observations_deleted": ent.rowcount,
        "executed_at": now.isoformat(),
    }
    logger.info("Retention completed: %s", result)
    return result


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
