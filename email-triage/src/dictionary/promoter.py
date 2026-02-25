"""
KeywordPromoter — reviews observations and promotes/quarantines/rejects
keywords in the lexicon using deterministic rules.

Promotion lifecycle: candidate → active → quarantined → rejected | deprecated
Human-in-the-loop: quarantined keywords require manual review before promotion.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from src.dictionary.models import (
    KeywordObservation,
    LexiconEntry,
    PromotionEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class PromoterConfig:
    """All thresholds in one place."""
    regex_min_doc_freq: int = 3
    regex_min_embedding_score: float = 0.35
    ner_min_doc_freq: int = 2
    ner_min_embedding_score: float = 0.25
    min_total_count: int = 5
    max_collision_labels: int = 2       # lemma in >2 labels → quarantine
    alert_collision_rate_threshold: float = 0.15
    alert_churn_rate_threshold: float = 0.30
    alert_promotion_rate_min: float = 0.02
    alert_promotion_rate_max: float = 0.50


@dataclass
class PromoterResult:
    promoted: int = 0
    quarantined: int = 0
    rejected: int = 0
    already_active: int = 0
    alerts: list[str] = field(default_factory=list)


class KeywordPromoter:
    """
    Runs end-of-batch to evaluate `candidate` lexicon entries and decide:
    - active      → enough doc_freq + low collision
    - quarantined → too ambiguous (high collision) or borderline
    - rejected    → below minimum thresholds
    """

    def __init__(
        self,
        config: PromoterConfig | None = None,
    ) -> None:
        self.config = config or PromoterConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_promotion_cycle(
        self,
        session: Session,
        dictionary_version: int,
    ) -> PromoterResult:
        """
        Evaluate all candidate lexicon entries and promote / quarantine.
        Writes PromotionEvent records for every state change (audit trail).
        """
        result = PromoterResult()

        # Pre-compute collision index: lemma → set of label_ids
        collision_index = self._build_collision_index(session)

        # Pull all candidate entries
        candidates = session.execute(
            select(LexiconEntry).where(LexiconEntry.status == "candidate")
        ).scalars().all()

        for entry in candidates:
            # Aggregate stats from observations
            stats = self._get_observation_stats(session, entry.label_id, entry.lemma)
            doc_freq = stats["doc_freq"]
            total_count = stats["total_count"]
            avg_embedding = stats["avg_embedding_score"] or 0.0

            collision_labels = len(collision_index.get(entry.lemma, set()))

            # ── Decision tree ─────────────────────────────────────────────────
            if collision_labels > self.config.max_collision_labels:
                # Too ambiguous → quarantine (requires human review)
                new_status = "quarantined"
                reason = (
                    f"collision_labels={collision_labels} "
                    f"> max={self.config.max_collision_labels}"
                )
                result.quarantined += 1

            elif total_count < self.config.min_total_count:
                # Not enough evidence yet → keep as candidate
                continue

            elif entry.dict_type == "regex":
                if (
                    doc_freq >= self.config.regex_min_doc_freq
                    and avg_embedding >= self.config.regex_min_embedding_score
                ):
                    new_status = "active"
                    reason = f"doc_freq={doc_freq}, embedding={avg_embedding:.3f}"
                    result.promoted += 1
                else:
                    new_status = "rejected"
                    reason = (
                        f"doc_freq={doc_freq} < {self.config.regex_min_doc_freq} "
                        f"or embedding={avg_embedding:.3f} < {self.config.regex_min_embedding_score}"
                    )
                    result.rejected += 1

            elif entry.dict_type == "ner":
                if (
                    doc_freq >= self.config.ner_min_doc_freq
                    and avg_embedding >= self.config.ner_min_embedding_score
                ):
                    new_status = "active"
                    reason = f"doc_freq={doc_freq}, embedding={avg_embedding:.3f}"
                    result.promoted += 1
                else:
                    new_status = "rejected"
                    reason = (
                        f"doc_freq={doc_freq} < {self.config.ner_min_doc_freq} "
                        f"or embedding={avg_embedding:.3f} < {self.config.ner_min_embedding_score}"
                    )
                    result.rejected += 1
            else:
                continue

            # Apply state change
            old_status = entry.status
            entry.status = new_status
            entry.last_seen_at = datetime.utcnow()

            # Audit log
            session.add(
                PromotionEvent(
                    event_id=str(uuid.uuid4()),
                    label_id=entry.label_id,
                    lemma=entry.lemma,
                    dict_type=entry.dict_type,
                    action=new_status,
                    from_status=old_status,
                    to_status=new_status,
                    reason=reason,
                    dictionary_version=dictionary_version,
                )
            )

        session.commit()

        # ── Health alerts ─────────────────────────────────────────────────────
        total = result.promoted + result.quarantined + result.rejected
        if total > 0:
            promotion_rate = result.promoted / total
            if not (
                self.config.alert_promotion_rate_min
                <= promotion_rate
                <= self.config.alert_promotion_rate_max
            ):
                result.alerts.append(
                    f"ALERT: promotion_rate={promotion_rate:.2f} outside "
                    f"[{self.config.alert_promotion_rate_min}, "
                    f"{self.config.alert_promotion_rate_max}]"
                )

        # Collision rate across all active entries
        collision_rate = self._compute_collision_rate(session)
        if collision_rate > self.config.alert_collision_rate_threshold:
            result.alerts.append(
                f"ALERT: collision_rate={collision_rate:.2f} "
                f"> {self.config.alert_collision_rate_threshold}"
            )

        for alert in result.alerts:
            logger.warning(alert)

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_collision_index(self, session: Session) -> dict[str, set[str]]:
        """lemma → {label_ids where it appears as active/candidate}."""
        rows = session.execute(
            select(LexiconEntry.lemma, LexiconEntry.label_id)
            .where(LexiconEntry.status.in_(["active", "candidate"]))
        ).all()
        index: dict[str, set[str]] = {}
        for lemma, label_id in rows:
            index.setdefault(lemma, set()).add(label_id)
        return index

    def _get_observation_stats(
        self, session: Session, label_id: str, lemma: str
    ) -> dict[str, Any]:
        row = session.execute(
            select(
                func.count(KeywordObservation.obs_id).label("doc_freq"),
                func.sum(KeywordObservation.count).label("total_count"),
                func.avg(KeywordObservation.embedding_score).label("avg_embedding_score"),
            ).where(
                KeywordObservation.label_id == label_id,
                KeywordObservation.lemma == lemma,
            )
        ).one()
        return {
            "doc_freq": row.doc_freq or 0,
            "total_count": row.total_count or 0,
            "avg_embedding_score": row.avg_embedding_score,
        }

    def _compute_collision_rate(self, session: Session) -> float:
        total_q = select(func.count(LexiconEntry.entry_id)).where(
            LexiconEntry.status == "active"
        )
        collision_q = (
            select(func.count())
            .select_from(
                select(LexiconEntry.lemma)
                .where(LexiconEntry.status == "active")
                .group_by(LexiconEntry.lemma)
                .having(func.count(LexiconEntry.label_id.distinct()) > 1)
                .subquery()
            )
        )
        total = session.execute(total_q).scalar() or 0
        collisions = session.execute(collision_q).scalar() or 0
        return collisions / total if total else 0.0
