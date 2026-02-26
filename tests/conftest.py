"""
shared pytest fixtures for the Observation Layer test suite.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone


# ── Minimal ObservationBatch fixture aligned to message_envelope.json ──────

SAMPLE_MESSAGE_ID = "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>"

SAMPLE_BATCH_DICT = {
    "email_context": {
        "message_id": SAMPLE_MESSAGE_ID,
        "id_conversazione": SAMPLE_MESSAGE_ID,
        "testo_normalizzato": (
            "Gentile team, Volevo confermare che i dati sono corretti: "
            "Codice Fiscale: RSSMRA80A01H501U, come discusso. "
            "Ho verificato tutti i dettagli e sono d'accordo con i termini proposti: "
            "durata 24 mesi, rata mensile di \u20ac 450,00. "
            "Potete inviarmi il contratto definitivo in formato PDF editabile? "
            "Potete fornirmi un preventivo aggiornato con le nuove tariffe 2024? "
            "In allegato trovate il documento firmato. "
            "Contatto: mario.rossi@example.it, tel. +39 02 98765432. "
            "Cordiali saluti, Mario Rossi"
        ),
        "mittente": "mario.rossi@example.it",
        "destinatario": "supporto@banca.it",
        "timestamp": "2026-02-24T12:02:09Z",
        "lingua": "it",
        "oggetto": None,
        "allegati": [],
    },
    "triage": {
        "topics": [
            {
                "labelid": "CONTRATTO",
                "confidence": 0.756,
                "confidence_llm": 0.95,
                "confidence_adjusted": 0.756,
                "keywordsintext": [
                    {
                        "candidateid": "L4CD0keGl10i4l43",
                        "lemma": "contrattare",
                        "term": "contratto",
                        "count": 1,
                        "source": "subject",
                        "embeddingscore": 0.449,
                    }
                ],
                "evidence": [
                    {
                        "quote": "Volevo confermare che i dati sono corretti",
                        "span": [13, 57],
                        "span_status": "exact_match",
                        "text_hash": "abc123",
                    }
                ],
                "keywords": [],
            }
        ],
        "sentiment": {"value": "neutral", "confidence": 0.7},
        "priority": {"value": "low", "confidence": 0.7, "signals": [], "rawscore": 0.0},
        "customerstatus": {"value": "existing", "confidence": 1.0, "source": "crm_exact_match"},
    },
    "postprocessing": {
        "message_id": SAMPLE_MESSAGE_ID,
        "pipeline_version": {
            "dictionaryversion": 42,
            "modelversion": "gemma3:4b-it-q8_0",
            "model_type": "chat",
            "parserversion": "email-parser-1.3.0",
            "stoplistversion": "stopwords-it-2025.2",
            "nermodelversion": "it_core_news_lg-3.8.2",
            "schemaversion": "json-schema-v3.3",
            "toolcallingversion": "openai-tool-calling-2026",
        },
        "triage": {
            "topics": [
                {
                    "labelid": "CONTRATTO",
                    "confidence": 0.756,
                    "confidence_llm": 0.95,
                    "confidence_adjusted": 0.756,
                    "keywordsintext": [
                        {
                            "candidateid": "L4CD0keGl10i4l43",
                            "lemma": "contrattare",
                            "term": "contratto",
                            "count": 1,
                            "source": "subject",
                            "embeddingscore": 0.449,
                        }
                    ],
                    "evidence": [],
                    "keywords": [],
                }
            ],
            "sentiment": {"value": "neutral", "confidence": 0.7},
            "priority": {"value": "low", "confidence": 0.7, "signals": [], "rawscore": 0.0},
            "customerstatus": {"value": "existing", "confidence": 1.0, "source": "crm_exact_match"},
        },
        "entities": [
            {
                "label": "CODICEFISCALE",
                "start": 73,
                "end": 89,
                "source": "regex",
                "confidence": 0.95,
                "extractor_version": "regex-v1.0",
                "value_hash": "e3b0c44298fc1c14",  # placeholder hash
            }
        ],
        "observations": [
            {
                "obs_id": "09e65b2c-41bd-42e9-8d60-df1ac821e993",
                "message_id": SAMPLE_MESSAGE_ID,
                "labelid": "CONTRATTO",
                "candidateid": "L4CD0keGl10i4l43",
                "lemma": "contrattare",
                "term": "contratto",
                "count": 1,
                "embeddingscore": 0.449,
                "dict_version": 42,
                "promoted_to_active": False,
                "observed_at": "2026-02-25T14:29:34.351696+00:00",
            }
        ],
        "diagnostics": {
            "warnings": [],
            "errors": [],
            "validation_retries": 0,
            "fallback_applied": False,
        },
        "processing_metadata": {
            "postprocessing_duration_ms": 11,
            "entities_extracted": 1,
            "observations_created": 1,
            "confidence_adjustments_applied": 1,
            "span_exact_match_count": 1,
            "span_fuzzy_match_count": 0,
            "span_not_found_count": 0,
        },
    },
    "ner_entities": None,
}
