"""
Rule-based priority scorer (parametric, weights learnable from data).

Used as:
1. The primary scorer when a reasoning LLM is not configured.
2. The fallback when the reasoning model call fails.
3. A validation cross-check against the LLM priority output.
"""
from __future__ import annotations

import re
from datetime import datetime

from src.contracts import EmailDocument, PrioritySignal

# ─────────────────────────────────────────────────────────────────────────────
# Keyword lists
# ─────────────────────────────────────────────────────────────────────────────

URGENT_TERMS: list[str] = [
    "urgente", "urgentissimo", "bloccante", "diffida", "disdetta",
    "reclamo", "rimborso", "guasto", "fermo", "critico",
    "sla", "multa", "penale", "segnalazione", "esposto",
    "avvocato", "legale", "tribunale",
]

HIGH_TERMS: list[str] = [
    "problema", "errore", "non funziona", "malfunzionamento",
    "assistenza", "supporto", "anomalia", "interruzione",
    "ritardo", "mancato", "disservizio",
]

# ─────────────────────────────────────────────────────────────────────────────
# Deadline detector
# ─────────────────────────────────────────────────────────────────────────────

_DEADLINE_PATTERNS: list[re.Pattern] = [
    re.compile(r"entro\s+il\s+\d{1,2}[/\-\.]\d{1,2}", re.IGNORECASE),
    re.compile(r"scadenza:?\s*\d{4}-\d{2}-\d{2}", re.IGNORECASE),
    re.compile(r"entro\s+\d{1,2}\s+giorni", re.IGNORECASE),
    re.compile(r"entro\s+(oggi|domani|stasera|stanotte)", re.IGNORECASE),
    re.compile(r"scade\s+(oggi|domani)", re.IGNORECASE),
]


def _deadline_urgency(text: str) -> int:
    """Return 0-2 urgency boost based on deadline mentions."""
    matches = sum(bool(p.search(text)) for p in _DEADLINE_PATTERNS)
    return min(matches, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Priority scorer
# ─────────────────────────────────────────────────────────────────────────────

class PriorityScorer:
    """
    Deterministic, parametric priority scorer.

    Default weights can be overridden or learned from historical data
    via `calibrate_from_data()`.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "urgent_terms": 3.0,
        "high_terms": 1.5,
        "sentiment_negative": 2.0,
        "customer_new": 1.0,
        "deadline_signal": 2.0,
        "vip_customer": 2.5,
    }

    # Score thresholds → priority bucket
    _THRESHOLDS = [
        (7.0, "urgent", 0.95),
        (4.0, "high", 0.85),
        (2.0, "medium", 0.75),
        (0.0, "low", 0.70),
    ]

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def score(
        self,
        doc: EmailDocument,
        sentiment_value: str,
        customer_value: str,
        vip_status: bool = False,
    ) -> PrioritySignal:
        """Compute priority for a single email document."""
        text = f"{doc.subject}\n{doc.body_canonical}".lower()
        raw_score = 0.0
        signals: list[str] = []

        # Urgent terms
        urgent_count = sum(1 for t in URGENT_TERMS if t in text)
        if urgent_count:
            raw_score += self.weights["urgent_terms"] * urgent_count
            signals.append(f"urgent_keywords:{urgent_count}")

        # High-priority terms
        high_count = sum(1 for t in HIGH_TERMS if t in text)
        if high_count:
            raw_score += self.weights["high_terms"] * high_count
            signals.append(f"high_keywords:{high_count}")

        # Sentiment
        if sentiment_value == "negative":
            raw_score += self.weights["sentiment_negative"]
            signals.append("negative_sentiment")

        # Customer status
        if customer_value == "new":
            raw_score += self.weights["customer_new"]
            signals.append("new_customer")

        # Deadline
        deadline_boost = _deadline_urgency(text)
        if deadline_boost:
            raw_score += self.weights["deadline_signal"] * deadline_boost
            signals.append(f"deadline_signal:{deadline_boost}")

        # VIP
        if vip_status:
            raw_score += self.weights["vip_customer"]
            signals.append("vip_customer")

        # Bucketing
        priority_val = "low"
        confidence = 0.70
        for threshold, label, conf in self._THRESHOLDS:
            if raw_score >= threshold:
                priority_val = label
                confidence = conf
                break

        return PrioritySignal(
            value=priority_val,  # type: ignore[arg-type]
            confidence=confidence,
            signals=signals,
            raw_score=raw_score,
        )

    def calibrate_from_data(self, training_data) -> "PriorityScorer":
        """
        Learn optimal weights from labelled historical data.

        training_data: DataFrame with columns:
            [urgent_terms_count, high_terms_count, is_negative_sentiment,
             is_new_customer, has_deadline, is_vip, priority_true]
        """
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder

            feature_cols = [
                "urgent_terms_count", "high_terms_count",
                "is_negative_sentiment", "is_new_customer",
                "has_deadline", "is_vip",
            ]
            X = training_data[feature_cols].values
            le = LabelEncoder()
            y = le.fit_transform(training_data["priority_true"])

            model = LogisticRegression(multi_class="ovr", max_iter=500)
            model.fit(X, y)

            # Map learned coefficients back to weight names
            coef_means = np.abs(model.coef_).mean(axis=0)
            for idx, col in enumerate(feature_cols):
                weight_key = {
                    "urgent_terms_count": "urgent_terms",
                    "high_terms_count": "high_terms",
                    "is_negative_sentiment": "sentiment_negative",
                    "is_new_customer": "customer_new",
                    "has_deadline": "deadline_signal",
                    "is_vip": "vip_customer",
                }.get(col)
                if weight_key:
                    self.weights[weight_key] = float(coef_means[idx] * 3.0)
        except ImportError:
            pass  # sklearn not available → keep defaults

        return self
