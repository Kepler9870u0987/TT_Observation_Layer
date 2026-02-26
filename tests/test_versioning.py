"""
tests/test_versioning.py

Verifica che ogni observation abbia pipeline_version completa
e che i constraint NOT NULL siano rispettati — Appendice D del documento.
"""
from __future__ import annotations

import copy

import pytest

from src.models.contracts import ObservationBatch
from src.models.pipeline_version import PipelineVersion


def _make_batch() -> dict:
    from tests.conftest import SAMPLE_BATCH_DICT
    return copy.deepcopy(SAMPLE_BATCH_DICT)


class TestVersioning:

    def test_pipeline_version_all_fields_present(self):
        """PipelineVersion deve avere tutti i campi obbligatori compilati."""
        batch = ObservationBatch.model_validate(_make_batch())
        pv_dict = batch.postprocessing.pipeline_version.model_dump()

        required_fields = [
            "dictionaryversion", "modelversion", "model_type", "parserversion",
            "stoplistversion", "nermodelversion", "schemaversion", "toolcallingversion",
        ]
        for field in required_fields:
            assert pv_dict.get(field) is not None, f"Field '{field}' is None/missing"

    def test_pipeline_version_frozen_dataclass(self):
        """PipelineVersion deve essere immutabile (frozen=True)."""
        pv = PipelineVersion(
            dictionary_version=42,
            model_version="gpt-4o-2025-11-20",
            model_type="chat",
            parser_version="email-parser-1.3.0",
            stoplist_version="stopwords-it-2025.2",
            ner_model_version="it_core_news_lg-3.8.2",
            schema_version="json-schema-v3.3",
            tool_calling_version="openai-tool-calling-2026",
        )
        with pytest.raises((AttributeError, TypeError)):
            pv.dictionary_version = 99  # type: ignore[misc]

    def test_pipeline_version_round_trip(self):
        """to_dict() + from_dict() deve produrre la stessa istanza."""
        pv = PipelineVersion(
            dictionary_version=42,
            model_version="gpt-4o-2025-11-20",
            model_type="reasoning",
            parser_version="email-parser-1.3.0",
            stoplist_version="stopwords-it-2025.2",
            ner_model_version="it_core_news_lg-3.8.2",
            schema_version="json-schema-v3.3",
            tool_calling_version="openai-tool-calling-2026",
        )
        restored = PipelineVersion.from_dict(pv.to_dict())
        assert pv == restored

    def test_observations_have_dict_version(self):
        """Ogni observation record deve avere dict_version non None."""
        batch = ObservationBatch.model_validate(_make_batch())
        for obs in batch.postprocessing.observations:
            assert obs.dict_version is not None, "observation missing dict_version"

    def test_observations_have_message_id(self):
        """Ogni observation deve avere message_id."""
        batch = ObservationBatch.model_validate(_make_batch())
        for obs in batch.postprocessing.observations:
            assert obs.message_id, "observation missing message_id"

    def test_observations_have_observed_at(self):
        """Ogni observation deve avere observed_at."""
        batch = ObservationBatch.model_validate(_make_batch())
        for obs in batch.postprocessing.observations:
            assert obs.observed_at is not None, "observation missing observed_at"

    def test_model_type_valid_values(self):
        """model_type deve essere 'chat' o 'reasoning'."""
        pv_dict = _make_batch()["postprocessing"]["pipeline_version"]
        assert pv_dict["model_type"] in ("chat", "reasoning")

    def test_missing_dictionary_version_raises(self):
        """Un payload senza dictionaryversion deve fallire la validazione Pydantic."""
        batch_dict = _make_batch()
        del batch_dict["postprocessing"]["pipeline_version"]["dictionaryversion"]
        with pytest.raises(Exception):
            ObservationBatch.model_validate(batch_dict)
