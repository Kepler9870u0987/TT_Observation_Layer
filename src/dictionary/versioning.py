"""
Dictionary Versioning — sezione 5 del documento.

create_new_version() usa un PostgreSQL advisory lock per garantire
single-writer: due batch concorrenti non possono creare versioni duplicate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.dictionary.promoter import KeywordPromoter

# Advisory lock key stabile per il versioning dei dizionari
_ADVISORY_LOCK_KEY = 202602261  # intero arbitrario fisso


async def _get_current_max_version(db: AsyncSession) -> int:
    result = await db.execute(
        text("SELECT COALESCE(MAX(dict_version_added), 0) FROM lexicon_entries")
    )
    row = result.fetchone()
    return int(row[0]) if row else 0


async def create_new_version(
    db: AsyncSession,
    promoter_updates: dict[str, list[dict]],
    current_version: int,
    run_id: str | None = None,
) -> int:
    """
    Persiste le promozioni e crea una nuova dictionary version.

    Usa pg_try_advisory_xact_lock per garantire single-writer.
    Ritorna il nuovo dict_version_new = current_version + 1.
    """
    async with db.begin():
        # Advisory lock — rilasciato automaticamente al commit/rollback
        lock_result = await db.execute(
            text("SELECT pg_try_advisory_xact_lock(:key)"),
            {"key": _ADVISORY_LOCK_KEY},
        )
        locked = lock_result.scalar()
        if not locked:
            raise RuntimeError(
                "Could not acquire advisory lock for dictionary versioning. "
                "Another process is currently creating a new version."
            )

        # Ri-verifica la versione corrente sotto lock
        actual_current = await _get_current_max_version(db)
        if actual_current > current_version:
            # Un altro processo ha già fatto upgrade — usa la versione più recente
            current_version = actual_current

        new_version = current_version + 1
        now = datetime.now(tz=timezone.utc)

        all_updates: list[tuple[str, dict]] = []
        for action in ("regex_active", "ner_active"):
            for entry in promoter_updates.get(action, []):
                all_updates.append((action, entry))

        # Upsert lexicon_entries
        for action, entry in all_updates:
            dict_type = entry["dict_type"]
            label_id = entry["label_id"]
            lemma = entry["lemma"]
            entry_id = str(uuid4())

            await db.execute(
                text("""
                    INSERT INTO lexicon_entries
                        (entry_id, label_id, dict_type, lemma, surface_forms,
                         regex_pattern, status, doc_freq, total_count,
                         embedding_score, first_seen_at, last_seen_at,
                         dict_version_added)
                    VALUES
                        (:entry_id, :label_id, :dict_type, :lemma, :surface_forms::jsonb,
                         :regex_pattern, 'active', :doc_freq, :total_count,
                         :embedding_score, :now, :now, :version)
                    ON CONFLICT ON CONSTRAINT uq_lex_entry DO UPDATE SET
                        surface_forms  = EXCLUDED.surface_forms,
                        regex_pattern  = EXCLUDED.regex_pattern,
                        status         = 'active',
                        doc_freq       = EXCLUDED.doc_freq,
                        total_count    = EXCLUDED.total_count,
                        embedding_score = EXCLUDED.embedding_score,
                        last_seen_at   = EXCLUDED.last_seen_at
                """),
                {
                    "entry_id": entry_id,
                    "label_id": label_id,
                    "dict_type": dict_type,
                    "lemma": lemma,
                    "surface_forms": __import__("json").dumps(entry.get("surface_forms", [])),
                    "regex_pattern": entry.get("regex_pattern"),
                    "doc_freq": entry.get("doc_freq", 0),
                    "total_count": entry.get("total_count", 0),
                    "embedding_score": entry.get("avg_embedding_score"),
                    "now": now,
                    "version": new_version,
                },
            )

        # Quarantined
        for entry in promoter_updates.get("quarantined", []):
            entry_id = str(uuid4())
            await db.execute(
                text("""
                    INSERT INTO lexicon_entries
                        (entry_id, label_id, dict_type, lemma, surface_forms,
                         status, doc_freq, total_count, embedding_score,
                         first_seen_at, last_seen_at, dict_version_added, quarantine_reason)
                    VALUES
                        (:entry_id, :label_id, 'ner', :lemma, :surface_forms::jsonb,
                         'quarantined', :doc_freq, :total_count, :embedding_score,
                         :now, :now, :version, :reason)
                    ON CONFLICT ON CONSTRAINT uq_lex_entry DO UPDATE SET
                        status = 'quarantined',
                        doc_freq = EXCLUDED.doc_freq,
                        total_count = EXCLUDED.total_count,
                        last_seen_at = EXCLUDED.last_seen_at,
                        quarantine_reason = EXCLUDED.quarantine_reason
                """),
                {
                    "entry_id": entry_id,
                    "label_id": entry["label_id"],
                    "lemma": entry["lemma"],
                    "surface_forms": __import__("json").dumps(entry.get("surface_forms", [])),
                    "doc_freq": entry.get("doc_freq", 0),
                    "total_count": entry.get("total_count", 0),
                    "embedding_score": entry.get("avg_embedding_score"),
                    "now": now,
                    "version": new_version,
                    "reason": entry.get("reason", ""),
                },
            )

        # Promotion events log (append-only)
        for action, entry in all_updates + [
            ("quarantined", e) for e in promoter_updates.get("quarantined", [])
        ] + [
            ("rejected", e) for e in promoter_updates.get("rejected", [])
        ]:
            action_label = {
                "regex_active": "promoted_regex",
                "ner_active": "promoted_ner",
                "quarantined": "quarantined",
                "rejected": "rejected",
            }.get(action, action)

            await db.execute(
                text("""
                    INSERT INTO promotion_events
                        (event_id, run_id, label_id, lemma, dict_type, action,
                         reason_code, dict_version_prev, dict_version_new,
                         doc_freq_at_promotion, embedding_score_at_promotion,
                         collision_labels)
                    VALUES
                        (:event_id, :run_id, :label_id, :lemma, :dict_type, :action,
                         :reason_code, :prev, :new,
                         :doc_freq, :emb_score, :collision_labels::jsonb)
                """),
                {
                    "event_id": str(uuid4()),
                    "run_id": run_id,
                    "label_id": entry["label_id"],
                    "lemma": entry["lemma"],
                    "dict_type": entry.get("dict_type", "ner"),
                    "action": action_label,
                    "reason_code": entry.get("reason"),
                    "prev": current_version,
                    "new": new_version,
                    "doc_freq": entry.get("doc_freq"),
                    "emb_score": entry.get("avg_embedding_score"),
                    "collision_labels": __import__("json").dumps(
                        entry.get("collision_labels", [])
                    ),
                },
            )

        # Marca le observations come promoted
        promoted_lemmas = [e["lemma"] for _, e in all_updates if _ in ("regex_active", "ner_active")]
        if promoted_lemmas:
            await db.execute(
                text("""
                    UPDATE keyword_observations
                    SET promoted_to_active = true
                    WHERE lemma = ANY(:lemmas) AND dict_version = :version
                """),
                {"lemmas": promoted_lemmas, "version": current_version},
            )

    return new_version
