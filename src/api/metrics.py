"""Prometheus metrics for the Observation Layer — sezione F del piano."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Counters ─────────────────────────────────────────────────────────────────

OBSERVATIONS_TOTAL = Counter(
    "observation_layer_observations_total",
    "Total keyword observations persisted",
    ["label_id"],
)

ENTITIES_TOTAL = Counter(
    "observation_layer_entities_total",
    "Total entity observations persisted",
    ["entity_type"],
)

BATCH_ERRORS_TOTAL = Counter(
    "observation_layer_batch_errors_total",
    "Total errors during batch persistence",
)

PROMOTION_EVENTS_TOTAL = Counter(
    "observation_layer_promotion_events_total",
    "Total promotion events by action",
    ["action"],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

COLLISION_RATE = Gauge(
    "observation_layer_collision_rate",
    "Fraction of lemmas appearing in more than one label (alert > 0.15)",
)

PROMOTION_RATE = Gauge(
    "observation_layer_promotion_rate",
    "Fraction of observations promoted to active (alert < 0.02 or > 0.50)",
)

QUARANTINED_TOTAL = Gauge(
    "observation_layer_quarantined_total",
    "Total entries currently in quarantine status",
)

DICT_SIZE_BY_LABEL = Gauge(
    "observation_layer_dict_size_by_label",
    "Number of active lexicon entries per label",
    ["label_id"],
)

UNKNOWN_TOPIC_RATE = Gauge(
    "observation_layer_unknown_topic_rate",
    "Fraction of messages tagged UNKNOWN_TOPIC (alert > 0.20)",
)

CHURN_RATE = Gauge(
    "observation_layer_churn_rate",
    "Fraction of lexicon entries added or removed between dict versions (alert > 0.30)",
)

CURRENT_DICT_VERSION = Gauge(
    "observation_layer_current_dict_version",
    "Current active dictionary version",
)

# ── Histograms ────────────────────────────────────────────────────────────────

BATCH_PERSIST_DURATION = Histogram(
    "observation_layer_batch_persist_duration_seconds",
    "Duration of a persist_batch call",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PROMOTER_DURATION = Histogram(
    "observation_layer_promoter_duration_seconds",
    "Duration of a full promoter run",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)
