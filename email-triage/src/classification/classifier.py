"""
LLM-based email classifier using Instructor (tool calling / structured outputs).

Two model routes:
- "chat"      → topics, sentiment, customer_status (gpt-4o-*)
- "reasoning" → priority / triage (o1-*, Qwen3-* — reduces under-triage 20-30%)

The LLM may only SELECT candidates from the provided candidate list.
All output is validated (multi-stage) before returning.
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Literal

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.contracts import (
    TOPICS_ENUM,
    Candidate,
    CustomerStatus,
    EmailDocument,
    Evidence,
    KeywordMatch,
    PrioritySignal,
    Sentiment,
    TopicAssignment,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# LLM response schemas (Pydantic → instructor → tool calling)
# ─────────────────────────────────────────────────────────────────────────────

class _TopicLLM(BaseModel):
    label_id: str = Field(description="Topic label from closed taxonomy")
    confidence: float = Field(ge=0.0, le=1.0)
    keywords_in_text: list[KeywordMatch] = Field(
        min_length=1, max_length=15,
        description="Keywords ONLY from the provided candidate list",
    )
    evidence: list[Evidence] = Field(
        min_length=1, max_length=2,
        description="Exact quotes from the email supporting this topic",
    )

    @field_validator("label_id")
    @classmethod
    def valid_label(cls, v: str) -> str:
        if v not in TOPICS_ENUM:
            raise ValueError(f"label_id {v!r} not in allowed topics")
        return v


class _ClassificationOutput(BaseModel):
    """Schema returned by the chat model (topics + sentiment)."""
    topics: list[_TopicLLM] = Field(min_length=1, max_length=5)
    sentiment: Sentiment


class _PriorityOutput(BaseModel):
    """Schema returned by the reasoning model (priority only)."""
    priority: PrioritySignal


# ─────────────────────────────────────────────────────────────────────────────
# LLM client factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_client() -> instructor.Instructor:
    key = settings.openrouter_api_key or settings.openai_api_key
    return instructor.from_openai(
        OpenAI(api_key=key, base_url=settings.llm_base_url),
        mode=instructor.Mode.JSON,          # JSON mode for broad provider support
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_classification_messages(
    doc: EmailDocument,
    candidates: list[Candidate],
    dictionary_version: int,
) -> list[dict]:
    top_candidates = candidates[:settings.max_candidates_to_llm]
    payload = {
        "dictionary_version": dictionary_version,
        "subject": doc.subject,
        "from": doc.from_raw,
        "body": doc.body_canonical[: settings.max_body_chars],
        "allowed_topics": TOPICS_ENUM,
        "candidate_keywords": [
            {
                "candidate_id": c.candidate_id,
                "term": c.term,
                "lemma": c.lemma,
                "count": c.count,
                "source": c.source,
                "score": round(c.composite_score, 3),
            }
            for c in top_candidates
        ],
    }
    system = (
        "You are an expert email triage classifier for Italian customer service.\n"
        "You MUST return JSON matching the schema exactly.\n"
        "Select keywords ONLY from 'candidate_keywords' using their candidate_id.\n"
        "DO NOT invent keywords, labels, or fields not in the schema.\n\n"
        "TOPICS: select 1-5 labels from allowed_topics. Use UNKNOWN_TOPIC if unclear.\n"
        "SENTIMENT:\n"
        "  positive = satisfaction, gratitude, positive tone\n"
        "  neutral  = informational, no emotional charge\n"
        "  negative = complaints, frustration, dissatisfaction\n"
        "KEYWORDS: each must have count >= 1 and exist in candidate_keywords."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _build_priority_messages(doc: EmailDocument) -> list[dict]:
    system = (
        "You are an expert email triage system. Determine the priority level.\n"
        "urgent = immediate action required (SLA breach, blocking, legal threat)\n"
        "high   = same-day response needed\n"
        "medium = 1-2 days acceptable\n"
        "low    = routine, no time pressure\n"
        "Provide 'signals': list of phrases/keywords justifying the priority.\n"
        "Think step by step before deciding."
    )
    user_content = (
        f"Subject: {doc.subject}\nFrom: {doc.from_raw}\n\n{doc.body_canonical[:4000]}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Core classification
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _call_chat_model(
    messages: list[dict],
    response_model: type,
) -> object:
    client = _make_client()
    return client.chat.completions.create(
        model=settings.llm_chat_model,
        messages=messages,
        response_model=response_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
    )


@retry(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _call_reasoning_model(messages: list[dict]) -> _PriorityOutput:
    client = _make_client()
    return client.chat.completions.create(
        model=settings.llm_reasoning_model,
        messages=messages,
        response_model=_PriorityOutput,
        # Reasoning models: no temperature parameter
        timeout=settings.llm_timeout_seconds,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_candidate_ids(
    output: _ClassificationOutput,
    valid_ids: set[str],
) -> tuple[_ClassificationOutput, list[str]]:
    """
    Remove any keyword whose candidate_id is not in valid_ids.
    Returns (cleaned_output, list_of_warnings).
    """
    warnings: list[str] = []
    for topic in output.topics:
        valid_kws = []
        for kw in topic.keywords_in_text:
            if kw.candidate_id in valid_ids:
                valid_kws.append(kw)
            else:
                warnings.append(
                    f"Removed invented candidate_id={kw.candidate_id!r} "
                    f"for topic {topic.label_id}"
                )
        topic.keywords_in_text = valid_kws or topic.keywords_in_text[:1]
    return output, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Confidence adjustment
# ─────────────────────────────────────────────────────────────────────────────

def _adjust_topic_confidence(
    topic: _TopicLLM,
    candidate_map: dict[str, Candidate],
    collision_counts: dict[str, int],
) -> float:
    """
    Composite confidence: 0.3*llm + 0.4*keyword_quality + 0.2*evidence + 0.1*collision_penalty
    """
    import numpy as np

    llm_conf = topic.confidence
    kw_scores = [
        candidate_map[kw.candidate_id].composite_score
        for kw in topic.keywords_in_text
        if kw.candidate_id in candidate_map
    ]
    avg_kw = float(np.mean(kw_scores)) if kw_scores else 0.0
    evidence_score = min(len(topic.evidence) / 2.0, 1.0)

    penalty_scores = []
    for kw in topic.keywords_in_text:
        cand = candidate_map.get(kw.candidate_id)
        if cand:
            n_labels = collision_counts.get(cand.lemma, 1)
            penalty_scores.append(1.0 / max(n_labels, 1))
    avg_penalty = float(np.mean(penalty_scores)) if penalty_scores else 1.0

    adjusted = (
        0.3 * llm_conf
        + 0.4 * avg_kw
        + 0.2 * evidence_score
        + 0.1 * avg_penalty
    )
    return float(min(max(adjusted, 0.0), 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_email(
    doc: EmailDocument,
    candidates: list[Candidate],
    dictionary_version: int,
    collision_counts: dict[str, int] | None = None,
) -> tuple[list[TopicAssignment], Sentiment, PrioritySignal]:
    """
    Classify an email using the dual-model routing strategy:
    - Chat model  → topics + sentiment
    - Reasoning model → priority

    Returns:
        (topics, sentiment, priority)
    """
    if collision_counts is None:
        collision_counts = {}

    candidate_map: dict[str, Candidate] = {c.candidate_id: c for c in candidates}
    valid_ids: set[str] = set(candidate_map.keys())

    # ── Chat model: topics + sentiment ────────────────────────────────────────
    classification_msgs = _build_classification_messages(
        doc, candidates, dictionary_version
    )
    raw_cls: _ClassificationOutput = _call_chat_model(
        classification_msgs, _ClassificationOutput
    )
    raw_cls, warnings = _validate_candidate_ids(raw_cls, valid_ids)
    if warnings:
        logger.warning("Candidate validation warnings for %s: %s", doc.message_id, warnings)

    # Build TopicAssignment with adjusted confidence
    topic_assignments: list[TopicAssignment] = []
    seen_labels: set[str] = set()
    for t in raw_cls.topics:
        if t.label_id in seen_labels:
            continue
        seen_labels.add(t.label_id)
        adjusted_conf = _adjust_topic_confidence(t, candidate_map, collision_counts)
        topic_assignments.append(
            TopicAssignment(
                label_id=t.label_id,
                confidence=adjusted_conf,
                confidence_llm=t.confidence,
                keywords_in_text=t.keywords_in_text,
                evidence=t.evidence,
            )
        )

    # ── Reasoning model: priority ─────────────────────────────────────────────
    priority_msgs = _build_priority_messages(doc)
    try:
        priority_out: _PriorityOutput = _call_reasoning_model(priority_msgs)
        priority = priority_out.priority
    except Exception as exc:
        logger.error("Reasoning model failed for %s: %s. Falling back.", doc.message_id, exc)
        # Fallback to rule-based priority (imported lazily to avoid circular)
        from src.enrichment.priority import PriorityScorer
        scorer = PriorityScorer()
        priority = scorer.score(doc, raw_cls.sentiment.value, "unknown")

    return topic_assignments, raw_cls.sentiment, priority
