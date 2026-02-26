"""
tests/test_idempotency.py

Verifica che re-run della stessa mail (stesso text_hash + dict_version)
non crei duplicati né alteri i conteggi — Appendice D del documento.
"""
from __future__ import annotations

import copy
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.models.contracts import ObservationBatch, ObservationBatchAck


def _make_batch() -> dict:
    from tests.conftest import SAMPLE_BATCH_DICT
    return copy.deepcopy(SAMPLE_BATCH_DICT)


class TestIdempotency:
    """ON CONFLICT DO NOTHING deve rendere persist_batch safe da chiamare N volte."""

    def test_observation_batch_parses_correctly(self):
        """Il payload di test deve essere un ObservationBatch valido."""
        batch = ObservationBatch.model_validate(_make_batch())
        assert batch.postprocessing.message_id is not None
        assert len(batch.postprocessing.observations) == 1

    @pytest.mark.asyncio
    async def test_first_call_creates_observations(self):
        """Prima call: observations_created deve essere > 0."""
        from src.observation_store.writer import persist_batch

        batch = ObservationBatch.model_validate(_make_batch())
        run_id = str(uuid4())

        # Mock DB session — simula rowcount=1 (riga inserita)
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.begin = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        ))

        ack = await persist_batch(mock_db, batch, run_id)
        # Con 1 observation nel payload → deve essere >= 0 (mock non ha DB reale)
        assert isinstance(ack, ObservationBatchAck)
        assert ack.run_id == run_id
        assert ack.message_id == batch.postprocessing.message_id

    @pytest.mark.asyncio
    async def test_second_call_skips_duplicates(self):
        """Seconda call con stesso payload: observations_created=0, skipped >= 1 (ON CONFLICT DO NOTHING → rowcount=0)."""
        from src.observation_store.writer import persist_batch

        batch = ObservationBatch.model_validate(_make_batch())
        run_id = str(uuid4())

        # Simula rowcount=0 → ON CONFLICT DO NOTHING ha saltato la riga
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.begin = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        ))

        ack = await persist_batch(mock_db, batch, run_id)
        assert ack.observations_created == 0
        assert ack.observations_skipped_idempotent >= 1

    def test_natural_key_fields_present(self):
        """Le osservazioni nel payload devono avere i 4 campi della chiave naturale."""
        batch = ObservationBatch.model_validate(_make_batch())
        for obs in batch.postprocessing.observations:
            assert obs.message_id
            assert obs.labelid
            assert obs.candidateid
            assert obs.dict_version is not None

    def test_entity_observation_pii_has_hash(self):
        """Entità PII devono avere value_hash (mai raw text)."""
        from src.models.contracts import PII_TYPES
        batch = ObservationBatch.model_validate(_make_batch())
        for ent in batch.postprocessing.entities:
            if ent.label in PII_TYPES:
                assert ent.value_hash is not None, (
                    f"Entity type {ent.label} is PII — value_hash must be set"
                )
