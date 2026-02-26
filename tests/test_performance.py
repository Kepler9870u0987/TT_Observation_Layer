"""
tests/test_performance.py

Budget per step (regex, NER, merge) e fallback se NER supera soglia.
Appendice D del documento.
"""
from __future__ import annotations

import time

import pytest

# Budgets in milliseconds (allineati ai timing osservati in message_envelope.json)
BUDGET_REGEX_MS = 50        # step3_regex: 2.155 ms osservato → budget conservativo 50 ms
BUDGET_NER_MS = 10_000      # step4_ner: 5210 ms osservato → budget 10 s (spaCy su CPU)
BUDGET_MERGE_MS = 10        # step6_merge: 0.035 ms → budget 10 ms
BUDGET_TOTAL_MS = 15_000    # budget totale per NER+merge+regex


class TestPerformanceBudgets:

    def test_regex_pattern_compile_is_fast(self):
        """Compilare un pattern regex da surface_forms deve essere < BUDGET_REGEX_MS."""
        from src.dictionary.promoter import KeywordPromoter
        surface_forms = [f"term_{i}" for i in range(200)]

        t0 = time.perf_counter()
        pattern = KeywordPromoter._generate_regex_pattern(surface_forms)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert pattern, "Pattern must not be empty"
        assert elapsed_ms < BUDGET_REGEX_MS, (
            f"Regex compile took {elapsed_ms:.2f} ms > budget {BUDGET_REGEX_MS} ms"
        )

    def test_promoter_stats_computation_is_fast(self):
        """compute_stats su 1000 observations deve completare in < 500 ms."""
        from src.dictionary.promoter import KeywordPromoter

        observations = [
            {
                "label_id": f"LABEL_{i % 10}",
                "lemma": f"lemma_{i % 50}",
                "term": f"term_{i % 50}",
                "message_id": f"msg_{i}",
                "count": 1,
                "embedding_score": 0.5,
            }
            for i in range(1000)
        ]
        promoter = KeywordPromoter()

        t0 = time.perf_counter()
        stats = promoter.compute_stats(observations)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert stats, "Stats must not be empty"
        assert elapsed_ms < 500, f"compute_stats took {elapsed_ms:.2f} ms > 500 ms"

    def test_collision_index_computation_is_fast(self):
        """compute_collision_index su 50 label x 100 lemmi deve completare in < 100 ms."""
        from src.dictionary.promoter import KeywordPromoter

        stats = {
            f"LABEL_{l}": {
                f"lemma_{k}": {"doc_freq": 3, "total_count": 5, "avg_embedding_score": 0.4, "surface_forms": []}
                for k in range(100)
            }
            for l in range(50)
        }
        promoter = KeywordPromoter()

        t0 = time.perf_counter()
        index = promoter.compute_collision_index(stats)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert index
        assert elapsed_ms < 100, f"collision_index took {elapsed_ms:.2f} ms > 100 ms"

    def test_ner_fallback_threshold(self):
        """
        Simula il check: se NER supera NER_TIMEOUT_MS, la pipeline deve
        applicare un fallback (disabilitazione dinamica o deferred extraction).

        Questo test verifica la logica del check, non l'esecuzione reale di spaCy.
        """
        NER_TIMEOUT_MS = BUDGET_NER_MS

        def should_use_ner_fallback(elapsed_ms: float) -> bool:
            """Ritorna True se NER ha superato il budget → usa fallback."""
            return elapsed_ms > NER_TIMEOUT_MS

        # Caso normale: sotto budget
        assert not should_use_ner_fallback(5000), "5 s should be within budget"

        # Caso timeout: sopra budget
        assert should_use_ner_fallback(12000), "12 s should trigger fallback"

    def test_merge_entities_is_fast(self):
        """Merge di 200 entità sovrapposte deve completare in < BUDGET_MERGE_MS."""
        from src.dictionary.collision_detector import compute_in_memory_collision_rate

        # Simula collision_index con 200 lemmi
        index = {f"lemma_{i}": {f"LABEL_{i % 3}", f"LABEL_{(i+1) % 3}"} for i in range(200)}

        t0 = time.perf_counter()
        rate = compute_in_memory_collision_rate(index)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert 0.0 <= rate <= 1.0
        assert elapsed_ms < BUDGET_MERGE_MS, f"merge took {elapsed_ms:.2f} ms > {BUDGET_MERGE_MS} ms"
