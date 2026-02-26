"""
tests/test_spans.py

Verifica che ogni span sia in [0, len(text_canonical)] e che
la substring corrisponda alla quote — Appendice D del documento.
"""
from __future__ import annotations

import copy

import pytest

from src.models.contracts import EvidenceItem, ObservationBatch


def _make_batch() -> dict:
    from tests.conftest import SAMPLE_BATCH_DICT
    return copy.deepcopy(SAMPLE_BATCH_DICT)


TEXT_CANONICAL = (
    "Gentile team, Volevo confermare che i dati sono corretti: "
    "Codice Fiscale: RSSMRA80A01H501U, come discusso. "
    "Ho verificato tutti i dettagli e sono d'accordo con i termini proposti: "
    "durata 24 mesi, rata mensile di \u20ac 450,00. "
    "Potete inviarmi il contratto definitivo in formato PDF editabile? "
    "Potete fornirmi un preventivo aggiornato con le nuove tariffe 2024? "
    "In allegato trovate il documento firmato. "
    "Contatto: mario.rossi@example.it, tel. +39 02 98765432. "
    "Cordiali saluti, Mario Rossi"
)


class TestSpans:

    def test_span_end_gt_start(self):
        """end deve essere strettamente maggiore di start."""
        item = EvidenceItem(quote="test", span=[0, 4])
        assert item.span[1] > item.span[0]

    def test_span_equal_start_end_raises(self):
        """span con start == end deve fallire la validazione."""
        with pytest.raises(Exception):
            EvidenceItem(quote="test", span=[5, 5])

    def test_span_inverted_raises(self):
        """span con start > end deve fallire la validazione."""
        with pytest.raises(Exception):
            EvidenceItem(quote="test", span=[10, 3])

    def test_spans_in_text_bounds(self):
        """Ogni span deve ricadere dentro la stringa canonica."""
        batch = ObservationBatch.model_validate(_make_batch())
        text = TEXT_CANONICAL
        for topic in batch.postprocessing.triage.topics:
            for ev in topic.evidence:
                if ev.span:
                    start, end = ev.span
                    assert 0 <= start < len(text), f"span start {start} out of bounds"
                    assert end <= len(text), f"span end {end} out of bounds"

    def test_span_substring_matches_quote_prefix(self):
        """
        La substring text[start:end] deve corrispondere alla quote
        (o essere marcata fuzzy con ragione).
        """
        batch = ObservationBatch.model_validate(_make_batch())
        text = TEXT_CANONICAL
        for topic in batch.postprocessing.triage.topics:
            for ev in topic.evidence:
                if ev.span and ev.span_status == "exact_match":
                    start, end = ev.span
                    substring = text[start:end]
                    quote_prefix = ev.quote[:len(substring)]
                    assert substring == quote_prefix or ev.quote.startswith(substring), (
                        f"Span [{start},{end}] = '{substring}' does not match quote prefix "
                        f"'{ev.quote[:40]}...'"
                    )

    def test_span_requires_two_elements(self):
        """Un span deve avere esattamente 2 elementi."""
        with pytest.raises(Exception):
            EvidenceItem(quote="test", span=[1, 2, 3])

    def test_entity_spans_in_bounds(self):
        """Le entità estratte devono avere start < end e end <= len(text)."""
        batch = ObservationBatch.model_validate(_make_batch())
        text = TEXT_CANONICAL
        for ent in batch.postprocessing.entities:
            assert ent.start < ent.end, f"Entity {ent.label}: start >= end"
            assert ent.end <= len(text), (
                f"Entity {ent.label} end={ent.end} > len(text)={len(text)}"
            )
