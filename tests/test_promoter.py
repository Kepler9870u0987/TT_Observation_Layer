"""
tests/test_promoter.py

Testa: promozione corretta, quarantena per collisione,
reject per count troppo basso, e lock single-writer.
"""
from __future__ import annotations

import pytest

from src.dictionary.promoter import KeywordPromoter


def _make_observations(
    label_id: str,
    lemma: str,
    n_messages: int,
    total_count: int,
    embedding_score: float = 0.5,
) -> list[dict]:
    return [
        {
            "label_id": label_id,
            "lemma": lemma,
            "term": lemma,
            "message_id": f"msg_{i}",
            "count": total_count // n_messages,
            "embedding_score": embedding_score,
        }
        for i in range(n_messages)
    ]


class TestPromoterDecisions:

    def _make_promoter(self) -> KeywordPromoter:
        return KeywordPromoter(config={
            "regex_min_doc_freq": 3,
            "regex_min_embedding_score": 0.35,
            "regex_min_total_count": 5,
            "regex_max_collision_labels": 1,
            "ner_min_doc_freq": 2,
            "ner_min_embedding_score": 0.25,
            "ner_min_total_count": 3,
            "ner_max_collision_labels": 2,
            "max_collision_labels": 2,
            "min_total_count": 3,
        })

    # ── Promozione corretta ──────────────────────────────────────────────

    def test_promotes_to_regex_when_high_evidence(self):
        """doc_freq >= 3, avg_emb >= 0.35, total >= 5, collision <=1 → regex_active."""
        promoter = self._make_promoter()
        obs = _make_observations("CONTRATTO", "contratto", n_messages=4, total_count=8, embedding_score=0.5)
        updates = promoter.promote_keywords(obs)

        assert any(e["lemma"] == "contratto" for e in updates["regex_active"]), (
            "Expected 'contratto' in regex_active"
        )
        assert any(e["lemma"] == "contratto" for e in updates["ner_active"]), (
            "Expected 'contratto' also in ner_active when promoted to regex"
        )

    def test_promotes_to_ner_only_when_lower_evidence(self):
        """doc_freq=2, avg_emb=0.3 → NER only (non regex)."""
        promoter = self._make_promoter()
        obs = _make_observations("RECLAMO", "problema", n_messages=2, total_count=4, embedding_score=0.3)
        updates = promoter.promote_keywords(obs)

        lemmas_ner = [e["lemma"] for e in updates["ner_active"]]
        lemmas_regex = [e["lemma"] for e in updates["regex_active"]]
        assert "problema" in lemmas_ner
        assert "problema" not in lemmas_regex

    # ── Quarantena per collisione ────────────────────────────────────────

    def test_quarantines_on_high_collision(self):
        """Stessa keyword su 3 label → quarantena (max_collision_labels=2)."""
        promoter = self._make_promoter()
        obs = (
            _make_observations("FATTURAZIONE", "fattura", 4, 8, 0.5)
            + _make_observations("RECLAMO", "fattura", 4, 8, 0.5)
            + _make_observations("CONTRATTO", "fattura", 4, 8, 0.5)
        )
        updates = promoter.promote_keywords(obs)

        quarantined_lemmas = [e["lemma"] for e in updates["quarantined"]]
        assert "fattura" in quarantined_lemmas, (
            "'fattura' with 3-label collision should be quarantined"
        )

    def test_quarantine_reason_high_collision(self):
        """La quarantena per collisione deve avere reason='high_collision'."""
        promoter = self._make_promoter()
        obs = (
            _make_observations("FATTURAZIONE", "fattura", 4, 8, 0.5)
            + _make_observations("RECLAMO", "fattura", 4, 8, 0.5)
            + _make_observations("CONTRATTO", "fattura", 4, 8, 0.5)
        )
        updates = promoter.promote_keywords(obs)

        quarantined = {e["lemma"]: e for e in updates["quarantined"]}
        assert quarantined["fattura"]["reason"] == "high_collision"

    # ── Reject per count troppo basso ───────────────────────────────────

    def test_rejects_low_total_count(self):
        """total_count < min_total_count (3) → rejected."""
        promoter = self._make_promoter()
        obs = _make_observations("GARANZIA", "garanzia", n_messages=1, total_count=2, embedding_score=0.6)
        updates = promoter.promote_keywords(obs)

        rejected_lemmas = [e["lemma"] for e in updates["rejected"]]
        assert "garanzia" in rejected_lemmas

    def test_reject_reason_low_count(self):
        """reason deve essere 'low_count'."""
        promoter = self._make_promoter()
        obs = _make_observations("GARANZIA", "garanzia", 1, 2, 0.6)
        updates = promoter.promote_keywords(obs)

        rejected = {e["lemma"]: e for e in updates["rejected"]}
        assert rejected["garanzia"]["reason"] == "low_count"

    # ── Regex pattern generation ─────────────────────────────────────────

    def test_regex_pattern_has_word_boundary(self):
        """Pattern generato deve usare \b word boundary."""
        pattern = KeywordPromoter._generate_regex_pattern(["fattura", "Fattura"])
        assert r"\b" in pattern

    def test_regex_pattern_case_insensitive_flag(self):
        """Pattern deve includere il flag (?i) per case-insensitività."""
        pattern = KeywordPromoter._generate_regex_pattern(["contratto"])
        assert "(?i)" in pattern

    def test_regex_pattern_escapes_special_chars(self):
        """Caratteri speciali devono essere escaped."""
        import re
        pattern = KeywordPromoter._generate_regex_pattern(["art.15", "par.3"])
        # Must compile without error
        compiled = re.compile(pattern)
        assert compiled.search("art.15 della norma")

    # ── Collision index accuracy ─────────────────────────────────────────

    def test_collision_index_single_label(self):
        """Lemma in una sola label → no collisione."""
        promoter = self._make_promoter()
        stats = {"CONTRATTO": {"contratto": {}}, "FATTURAZIONE": {"fattura": {}}}
        index = promoter.compute_collision_index(stats)
        assert len(index["contratto"]) == 1
        assert len(index["fattura"]) == 1

    def test_collision_index_multi_label(self):
        """Stesso lemma in 2 label → collisione."""
        promoter = self._make_promoter()
        stats = {
            "CONTRATTO": {"documento": {}},
            "DOCUMENTI": {"documento": {}},
        }
        index = promoter.compute_collision_index(stats)
        assert len(index["documento"]) == 2

    # ── Lock single-writer (concorrenza) ────────────────────────────────

    @pytest.mark.asyncio
    async def test_advisory_lock_prevents_concurrent_versioning(self):
        """
        create_new_version deve sollevare RuntimeError se l'advisory lock
        è già acquisito da un altro processo.
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.dictionary.versioning import create_new_version

        # Simula lock non disponibile (pg_try_advisory_xact_lock ritorna false)
        mock_lock_result = MagicMock()
        mock_lock_result.scalar.return_value = False

        mock_max_result = MagicMock()
        mock_max_result.scalar.return_value = 42

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=[mock_lock_result])
        mock_db.begin = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        ))

        with pytest.raises(RuntimeError, match="advisory lock"):
            await create_new_version(mock_db, {}, current_version=42)
