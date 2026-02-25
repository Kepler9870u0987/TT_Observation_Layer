"""
Deterministic keyword candidate extraction.

Rules:
- Same email → same candidates (no randomness, stable sort).
- candidate_id is a SHA-1 of (source, term) — stable across runs.
- LLM may only SELECT from these candidates; it cannot invent new ones.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Literal

import numpy as np

from src.contracts import Candidate, EmailDocument
from src.candidates.stopwords import STOPWORDS_IT, BLACKLIST_PATTERNS

STOPLIST_VERSION = "stopwords-it-2025.1"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zàèéìòùA-ZÀÈÉÌÒÙ]+(?:'[a-zàèéìòùA-ZÀÈÉÌÒÙ]+)?|\d{2,}")


def _tokenize(text: str) -> list[str]:
    return [m.lower() for m in _TOKEN_RE.findall(text)]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _stable_id(source: str, term: str) -> str:
    raw = f"{source}|{term}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_candidates(doc: EmailDocument) -> list[Candidate]:
    """
    Extract all unigram/bigram/trigram candidates from subject + body.
    Returns a deterministically sorted list.
    """
    sources: list[tuple[Literal["subject", "body"], str]] = []
    if doc.subject:
        sources.append(("subject", doc.subject))
    if doc.body_canonical:
        sources.append(("body", doc.body_canonical))

    # Count raw occurrences
    counts: Counter[tuple[str, str]] = Counter()
    for source, text in sources:
        tokens = _tokenize(text)
        all_terms = tokens + _ngrams(tokens, 2) + _ngrams(tokens, 3)
        for term in all_terms:
            if len(term) >= 3:
                counts[(source, term)] += 1

    candidates: list[Candidate] = []
    for (source, term), count in counts.items():
        cid = _stable_id(source, term)
        candidates.append(
            Candidate(
                candidate_id=cid,
                source=source,  # type: ignore[arg-type]
                term=term,
                lemma=term,     # lemma updated by enrichment step if spaCy available
                count=count,
                composite_score=0.0,
            )
        )

    # Stable deterministic sort: source asc, count desc, term asc
    candidates.sort(key=lambda c: (c.source, -c.count, c.term))
    return candidates


def filter_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """
    Apply hard filters to remove stopwords, blacklisted terms, and noise.
    Deterministic: same input → same output.
    """
    result: list[Candidate] = []
    for cand in candidates:
        lemma = cand.lemma.lower()
        term = cand.term.lower()

        # Stopword check
        if lemma in STOPWORDS_IT or term in STOPWORDS_IT:
            continue

        # Blacklist patterns
        if any(re.search(pat, term, re.IGNORECASE) for pat in BLACKLIST_PATTERNS):
            continue

        # Minimum length (already checked in build, defensive)
        if len(term) < 3:
            continue

        result.append(cand)

    return result


def score_candidate_composite(
    cand: Candidate,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Composite relevance score combining frequency, semantic score, and source.
    Used to rank candidates before sending top-N to LLM.
    """
    if weights is None:
        weights = {"count": 0.3, "embedding": 0.5, "subject_bonus": 0.2}

    count_norm = float(np.log1p(cand.count)) / 5.0
    embedding_score = cand.embedding_score
    subject_bonus = 1.0 if cand.source == "subject" else 0.0

    score = (
        weights["count"] * count_norm
        + weights["embedding"] * embedding_score
        + weights["subject_bonus"] * subject_bonus
    )
    return float(score)


def rank_and_trim_candidates(
    candidates: list[Candidate],
    max_candidates: int = 100,
) -> list[Candidate]:
    """Score, rank, and return top-N candidates for LLM prompt."""
    for cand in candidates:
        cand.composite_score = score_candidate_composite(cand)
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return candidates[:max_candidates]
