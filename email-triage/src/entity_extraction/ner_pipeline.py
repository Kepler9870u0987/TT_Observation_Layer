"""
Hybrid NER pipeline (three layers, priority: RegEx > LLM-NER > spaCy).

Layer 1: RegEx  — high precision for structured entities (CF, IBAN, email, tel)
Layer 2: spaCy  — general NER (PER, ORG, LOC, DATE)
Layer 3: LLM    — dynamic, out-of-vocabulary entities (optional, tool calling)

Results are merged with source-priority deduplication.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from functools import lru_cache

from src.config import get_settings
from src.contracts import EmailDocument, Entity, EntityExtractionOutput

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: RegEx patterns (high precision)
# ─────────────────────────────────────────────────────────────────────────────

_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("CODICE_FISCALE", re.compile(
        r"\b[A-Z]{6}\d{2}[A-EHLMPRST]\d{2}[A-Z]\d{3}[A-Z]\b", re.IGNORECASE
    )),
    ("IBAN", re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{0,16}\b", re.IGNORECASE
    )),
    ("PARTITA_IVA", re.compile(r"\b\d{11}\b")),
    ("EMAIL", re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )),
    ("TELEFONO", re.compile(
        r"(?:\+39\s?)?(?:0\d{1,4}[\s\-]?\d{5,8}|\d{3}[\s\-]?\d{3}[\s\-]?\d{4})"
    )),
    ("IMPORTO", re.compile(
        r"€\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?|\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s?€"
    )),
    ("URL", re.compile(
        r"https?://[^\s<>\"']+", re.IGNORECASE
    )),
]


def _stable_entity_id(entity_type: str, value: str, span_start: int) -> str:
    raw = f"{entity_type}|{value}|{span_start}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def _extract_regex(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for entity_type, pattern in _REGEX_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            entities.append(
                Entity(
                    entity_id=_stable_entity_id(entity_type, value, match.start()),
                    entity_type=entity_type,
                    value=value,
                    value_normalized=value.strip().upper(),
                    span_start=match.start(),
                    span_end=match.end(),
                    source="regex",
                    confidence=1.0,
                )
            )
    return entities


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: spaCy NER
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_spacy_nlp():
    try:
        import spacy
        nlp = spacy.load(settings.spacy_model)
        logger.info("spaCy model loaded: %s", settings.spacy_model)
        return nlp
    except (ImportError, OSError) as exc:
        logger.warning("spaCy model not available (%s). Skipping spaCy NER.", exc)
        return None


def _extract_spacy(text: str) -> list[Entity]:
    nlp = _get_spacy_nlp()
    if nlp is None:
        return []
    try:
        doc = nlp(text[:10_000])  # cap for performance
    except Exception as exc:
        logger.warning("spaCy NER failed: %s", exc)
        return []

    entities: list[Entity] = []
    for ent in doc.ents:
        entity_type = ent.label_
        value = ent.text
        entities.append(
            Entity(
                entity_id=_stable_entity_id(entity_type, value, ent.start_char),
                entity_type=entity_type,
                value=value,
                value_normalized=value.strip(),
                span_start=ent.start_char,
                span_end=ent.end_char,
                source="spacy",
                confidence=0.85,
            )
        )
    return entities


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: LLM-NER (optional, out-of-vocabulary)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_llm_ner(
    text: str,
    existing_spans: set[tuple[int, int]],
) -> list[Entity]:
    """
    Use LLM (tool calling) to find entities missed by RegEx and spaCy.
    Only called for texts > 200 chars (avoid overhead for short emails).
    """
    if len(text) < 200:
        return []

    try:
        import instructor
        from openai import OpenAI
        from pydantic import BaseModel

        class _LLMEntity(BaseModel):
            entity_type: str
            value: str
            context: str = ""  # surrounding phrase for confidence

        class _LLMEntities(BaseModel):
            entities: list[_LLMEntity]

        key = settings.openrouter_api_key or settings.openai_api_key
        client = instructor.from_openai(
            OpenAI(api_key=key, base_url=settings.llm_base_url),
            mode=instructor.Mode.JSON,
        )
        result: _LLMEntities = client.chat.completions.create(
            model=settings.llm_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract named entities from the Italian email text. "
                        "Focus on: person names (PER), organizations (ORG), "
                        "locations (LOC), product names, contract references, "
                        "and any domain-specific identifiers. "
                        "Return only entities not already captured as "
                        "CODICE_FISCALE, IBAN, EMAIL, TELEFONO, IMPORTO."
                    ),
                },
                {"role": "user", "content": text[:3000]},
            ],
            response_model=_LLMEntities,
            temperature=0.0,
            timeout=settings.llm_timeout_seconds,
        )

        entities: list[Entity] = []
        for llm_ent in result.entities:
            # Find actual span in text
            idx = text.find(llm_ent.value)
            if idx == -1:
                continue
            span = (idx, idx + len(llm_ent.value))
            if span in existing_spans:
                continue
            entities.append(
                Entity(
                    entity_id=_stable_entity_id(llm_ent.entity_type, llm_ent.value, idx),
                    entity_type=llm_ent.entity_type,
                    value=llm_ent.value,
                    value_normalized=llm_ent.value.strip(),
                    span_start=idx,
                    span_end=idx + len(llm_ent.value),
                    source="llm",
                    confidence=0.70,
                )
            )
        return entities

    except Exception as exc:
        logger.warning("LLM-NER failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Merge (priority: regex > llm > spacy)
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_PRIORITY = {"regex": 0, "llm": 1, "spacy": 2}


def _merge_entities(
    regex_ents: list[Entity],
    spacy_ents: list[Entity],
    llm_ents: list[Entity],
) -> list[Entity]:
    """
    Merge layers with overlap deduplication.
    For overlapping spans, keep the entity with highest source priority.
    """
    all_ents: list[Entity] = regex_ents + llm_ents + spacy_ents

    # Sort by span start, then by source priority
    all_ents.sort(key=lambda e: (e.span_start, _SOURCE_PRIORITY[e.source]))

    merged: list[Entity] = []
    for ent in all_ents:
        # Check for overlap with already-accepted entities
        overlaps = any(
            not (ent.span_end <= acc.span_start or ent.span_start >= acc.span_end)
            for acc in merged
        )
        if not overlaps:
            merged.append(ent)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities(
    doc: EmailDocument,
    use_llm_ner: bool = True,
) -> EntityExtractionOutput:
    """
    Run the full three-layer NER pipeline.

    Args:
        doc: canonical EmailDocument
        use_llm_ner: if False, skip the LLM layer (faster, for batch processing)

    Returns:
        EntityExtractionOutput with merged entities and timing info.
    """
    text = doc.full_text
    timings: dict[str, float] = {}

    # Layer 1: RegEx
    t0 = time.perf_counter()
    regex_ents = _extract_regex(text)
    timings["regex_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Layer 2: spaCy
    t0 = time.perf_counter()
    spacy_ents = _extract_spacy(text)
    timings["spacy_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Layer 3: LLM (optional)
    llm_ents: list[Entity] = []
    if use_llm_ner:
        existing_spans = {(e.span_start, e.span_end) for e in regex_ents + spacy_ents}
        t0 = time.perf_counter()
        llm_ents = _extract_llm_ner(text, existing_spans)
        timings["llm_ner_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    merged = _merge_entities(regex_ents, spacy_ents, llm_ents)

    return EntityExtractionOutput(
        message_id=doc.message_id,
        entities=merged,
        component_timings_ms=timings,
        ner_model_version=settings.ner_model_version,
    )
