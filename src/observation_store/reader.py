"""
Observation Storage Reader.

Query usate dal Promoter e dalle route di monitoring.
Tutte le funzioni accettano un AsyncSession e ritornano dati Python puri.
"""
from __future__ import annotations

from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# ─────────────────────────────────────────────────────────────────────────────
# Doc-frequency aggregato per (label_id, lemma)
# ─────────────────────────────────────────────────────────────────────────────

async def get_doc_freq_by_label_lemma(
    db: AsyncSession,
    dict_version: int | None = None,
    min_observed_at: str | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Ritorna {label_id: {lemma: {"doc_freq": int, "total_count": int, "avg_embedding_score": float}}}.

    - doc_freq  = numero di message_id distinti in cui la keyword è apparsa
    - total_count = somma dei count
    """
    filters = ["1=1"]
    params: dict = {}

    if dict_version is not None:
        filters.append("dict_version = :dict_version")
        params["dict_version"] = dict_version
    if min_observed_at is not None:
        filters.append("observed_at >= :min_observed_at")
        params["min_observed_at"] = min_observed_at

    where = " AND ".join(filters)
    q = text(f"""
        SELECT
            label_id,
            lemma,
            COUNT(DISTINCT message_id) AS doc_freq,
            SUM(count)                 AS total_count,
            AVG(embedding_score)       AS avg_embedding_score
        FROM keyword_observations
        WHERE {where}
        GROUP BY label_id, lemma
    """)

    result = await db.execute(q, params)
    rows = result.fetchall()

    stats: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        stats[row.label_id][row.lemma] = {
            "doc_freq": int(row.doc_freq),
            "total_count": int(row.total_count),
            "avg_embedding_score": float(row.avg_embedding_score or 0.0),
        }
    return dict(stats)


# ─────────────────────────────────────────────────────────────────────────────
# Collision index: lemma → set[label_id]
# ─────────────────────────────────────────────────────────────────────────────

async def get_collision_index(
    db: AsyncSession,
    dict_version: int | None = None,
) -> dict[str, set[str]]:
    """
    Ritorna {lemma: {label_id1, label_id2, …}}.
    Lemmi con len > 1 sono in collisione cross-label.
    """
    params: dict = {}
    version_filter = ""
    if dict_version is not None:
        version_filter = "WHERE dict_version = :dict_version"
        params["dict_version"] = dict_version

    q = text(f"""
        SELECT lemma, label_id
        FROM keyword_observations
        {version_filter}
        GROUP BY lemma, label_id
    """)

    result = await db.execute(q, params)
    index: dict[str, set[str]] = defaultdict(set)
    for row in result.fetchall():
        index[row.lemma].add(row.label_id)
    return dict(index)


# ─────────────────────────────────────────────────────────────────────────────
# Collision rate (aggregato)
# ─────────────────────────────────────────────────────────────────────────────

async def get_collision_rate(db: AsyncSession) -> float:
    """Frazione di lemmi attivi presenti in più di una label."""
    q = text("""
        SELECT
            COUNT(DISTINCT lemma) FILTER (WHERE cnt > 1)::float /
            NULLIF(COUNT(DISTINCT lemma), 0) AS collision_rate
        FROM (
            SELECT lemma, COUNT(DISTINCT label_id) AS cnt
            FROM keyword_observations
            GROUP BY lemma
        ) sub
    """)
    result = await db.execute(q)
    row = result.fetchone()
    return float(row.collision_rate or 0.0) if row else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Quarantined entries above threshold
# ─────────────────────────────────────────────────────────────────────────────

async def get_quarantined_above_threshold(
    db: AsyncSession,
    min_doc_freq: int = 10,
) -> list[dict]:
    """
    Entry in quarantina con doc_freq >= min_doc_freq.
    Usato per la review queue umana.
    """
    q = text("""
        SELECT entry_id, label_id, dict_type, lemma, status,
               doc_freq, total_count, embedding_score, quarantine_reason
        FROM lexicon_entries
        WHERE status = 'quarantined' AND doc_freq >= :min_doc_freq
        ORDER BY doc_freq DESC
    """)
    result = await db.execute(q, {"min_doc_freq": min_doc_freq})
    return [dict(row._mapping) for row in result.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Observations since a given run_id (per promoter batch)
# ─────────────────────────────────────────────────────────────────────────────

async def get_observations_since(
    db: AsyncSession,
    since_run_id: str | None = None,
    dict_version: int | None = None,
) -> list[dict]:
    """
    Legge le keyword observations non ancora promosse.
    Usato dal Promoter per decidere le promozioni del batch corrente.
    """
    filters = ["promoted_to_active = false"]
    params: dict = {}

    if dict_version is not None:
        filters.append("dict_version = :dict_version")
        params["dict_version"] = dict_version

    where = " AND ".join(filters)
    q = text(f"""
        SELECT obs_id, message_id, label_id, candidate_id, lemma, term,
               count, embedding_score, dict_version, observed_at
        FROM keyword_observations
        WHERE {where}
        ORDER BY observed_at ASC
    """)
    result = await db.execute(q, params)
    return [dict(row._mapping) for row in result.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Dictionary health snapshot
# ─────────────────────────────────────────────────────────────────────────────

async def get_dictionary_health(db: AsyncSession) -> dict:
    """Snapshot metriche di salute dei dizionari per GET /dictionaries/health."""
    size_q = text("""
        SELECT label_id, dict_type, status, COUNT(*) AS cnt
        FROM lexicon_entries
        GROUP BY label_id, dict_type, status
    """)
    coll_rate = await get_collision_rate(db)
    size_rows = (await db.execute(size_q)).fetchall()

    by_label: dict[str, dict] = defaultdict(lambda: {"active": 0, "candidate": 0, "quarantined": 0, "rejected": 0})
    total_active = 0
    total_quarantined = 0
    for row in size_rows:
        by_label[row.label_id][row.status] = by_label[row.label_id].get(row.status, 0) + int(row.cnt)
        if row.status == "active":
            total_active += int(row.cnt)
        elif row.status == "quarantined":
            total_quarantined += int(row.cnt)

    return {
        "collision_rate": round(coll_rate, 4),
        "total_active_entries": total_active,
        "total_quarantined_entries": total_quarantined,
        "by_label": dict(by_label),
    }
