"""
Collision Detector — sezione 3.2 e 5.2 del documento.

Calcola collision_index e collision_rate sul DB e in-memory
(per il Promoter che opera su observations in-memory).
"""
from __future__ import annotations

from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_collision_index_from_db(db: AsyncSession) -> dict[str, set[str]]:
    """
    Ritorna {lemma: {label_id}} dagli lexicon_entries attivi.
    Lemmi con len > 1 sono in collisione cross-label.
    """
    q = text("""
        SELECT lemma, label_id
        FROM lexicon_entries
        WHERE status = 'active'
        GROUP BY lemma, label_id
    """)
    result = await db.execute(q)
    index: dict[str, set[str]] = defaultdict(set)
    for row in result.fetchall():
        index[row.lemma].add(row.label_id)
    return dict(index)


async def get_collision_rate_from_db(db: AsyncSession) -> float:
    """
    Frazione di lemmi attivi presenti in > 1 label.
    Alert threshold: > 0.15 (sezione 6 doc).
    """
    q = text("""
        SELECT
            COUNT(DISTINCT lemma) FILTER (WHERE cnt > 1)::float /
            NULLIF(COUNT(DISTINCT lemma), 0) AS rate
        FROM (
            SELECT lemma, COUNT(DISTINCT label_id) AS cnt
            FROM lexicon_entries
            WHERE status = 'active'
            GROUP BY lemma
        ) sub
    """)
    result = await db.execute(q)
    row = result.fetchone()
    return float(row.rate or 0.0) if row else 0.0


def compute_in_memory_collision_rate(
    collision_index: dict[str, set[str]]
) -> float:
    """Versione in-memory (per report batch senza DB round-trip)."""
    if not collision_index:
        return 0.0
    colliding = sum(1 for labels in collision_index.values() if len(labels) > 1)
    return colliding / len(collision_index)


def find_colliding_lemmas(
    collision_index: dict[str, set[str]]
) -> dict[str, list[str]]:
    """Ritorna {lemma: [label_id_1, label_id_2, …]} per soli lemmi in collisione."""
    return {
        lemma: sorted(labels)
        for lemma, labels in collision_index.items()
        if len(labels) > 1
    }
