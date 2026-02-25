"""
Semantic enrichment of candidates using KeyBERT + sentence-transformers.
Model is loaded lazily and cached for the process lifetime.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from src.contracts import Candidate, EmailDocument

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_VERSION = "paraphrase-multilingual-mpnet-base-v2"


@lru_cache(maxsize=1)
def _get_kw_model():
    """Load KeyBERT model once per process."""
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer

        model = KeyBERT(model=SentenceTransformer(EMBEDDING_MODEL_VERSION))
        logger.info("KeyBERT model loaded: %s", EMBEDDING_MODEL_VERSION)
        return model
    except ImportError as exc:
        logger.warning("KeyBERT/sentence-transformers not available: %s", exc)
        return None


def enrich_candidates_with_embeddings(
    doc: EmailDocument,
    candidates: list[Candidate],
    top_n: int = 50,
) -> list[Candidate]:
    """
    Add an `embedding_score` to each candidate using KeyBERT cosine similarity.

    If KeyBERT is not installed, candidates are returned unchanged with
    embedding_score=0.0 (graceful degradation).
    """
    kw_model = _get_kw_model()
    if kw_model is None:
        return candidates

    full_text = f"{doc.subject}\n{doc.body_canonical}".strip()
    if not full_text:
        return candidates

    try:
        keybert_keywords = kw_model.extract_keywords(
            full_text,
            keyphrase_ngram_range=(1, 3),
            stop_words="italian",
            top_n=top_n,
            use_mmr=True,       # maximal marginal relevance for diversity
            diversity=0.5,
        )
    except Exception as exc:
        logger.warning("KeyBERT extraction failed: %s", exc)
        return candidates

    # Build lookup: term → score
    kb_scores: dict[str, float] = {kw: float(score) for kw, score in keybert_keywords}

    for cand in candidates:
        score = kb_scores.get(cand.term, kb_scores.get(cand.lemma, 0.0))
        cand.embedding_score = score

    return candidates
