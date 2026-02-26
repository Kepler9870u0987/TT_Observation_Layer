"""Initial schema — all tables for the Observation Layer.

Revision ID: 0001
Revises:
Create Date: 2026-02-26
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── pipeline_runs ─────────────────────────────────────────────────────────
    op.create_table(
        "pipeline_runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pipeline_version", JSONB, nullable=False),
        sa.Column("dict_version_used", sa.Integer, nullable=False),
        sa.Column("dict_version_new", sa.Integer, nullable=True),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("observations_created", sa.Integer, nullable=False, server_default="0"),
        sa.Column("entities_created", sa.Integer, nullable=False, server_default="0"),
        sa.Column("messages_processed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("errors_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("metrics", JSONB, nullable=True),
    )

    # ── messages ──────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("message_id", sa.String(512), primary_key=True),
        sa.Column("text_hash", sa.String(64), nullable=True),
        sa.Column("mittente", sa.String(512), nullable=True),
        sa.Column("destinatario", sa.String(512), nullable=True),
        sa.Column("lingua", sa.String(8), nullable=True),
        sa.Column("oggetto_hash", sa.String(64), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pii_flags", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_messages_text_hash", "messages", ["text_hash"])

    # ── label_registry ────────────────────────────────────────────────────────
    op.create_table(
        "label_registry",
        sa.Column("label_id", sa.String(128), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("merged_into", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── lexicon_entries ───────────────────────────────────────────────────────
    op.create_table(
        "lexicon_entries",
        sa.Column("entry_id", sa.String(64), primary_key=True),
        sa.Column("label_id", sa.String(128), sa.ForeignKey("label_registry.label_id"), nullable=False),
        sa.Column("dict_type", sa.String(16), nullable=False),
        sa.Column("lemma", sa.String(256), nullable=False),
        sa.Column("surface_forms", JSONB, nullable=False),
        sa.Column("regex_pattern", sa.Text, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="candidate"),
        sa.Column("doc_freq", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("embedding_score", sa.Float, nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dict_version_added", sa.Integer, nullable=True),
        sa.Column("dict_version_deprecated", sa.Integer, nullable=True),
        sa.Column("quarantine_reason", sa.String(128), nullable=True),
    )
    op.create_unique_constraint("uq_lex_entry", "lexicon_entries", ["label_id", "dict_type", "lemma"])
    op.create_index("ix_lex_label_status", "lexicon_entries", ["label_id", "status"])
    op.create_index("ix_lex_lemma", "lexicon_entries", ["lemma"])
    op.create_index("ix_lex_status", "lexicon_entries", ["status"])

    # ── keyword_observations ──────────────────────────────────────────────────
    op.create_table(
        "keyword_observations",
        sa.Column("obs_id", sa.String(64), primary_key=True),
        sa.Column("message_id", sa.String(512), sa.ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("pipeline_runs.run_id"), nullable=True),
        sa.Column("label_id", sa.String(128), nullable=False),
        sa.Column("candidate_id", sa.String(64), nullable=False),
        sa.Column("lemma", sa.String(256), nullable=False),
        sa.Column("term", sa.String(256), nullable=False),
        sa.Column("count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("embedding_score", sa.Float, nullable=True),
        sa.Column("spans", JSONB, nullable=True),
        sa.Column("evidence_quote_hash", sa.String(64), nullable=True),
        sa.Column("dict_version", sa.Integer, nullable=False),
        sa.Column("promoted_to_active", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_unique_constraint(
        "uq_kw_obs_natural_key",
        "keyword_observations",
        ["message_id", "label_id", "candidate_id", "dict_version"],
    )
    op.create_index("ix_kw_obs_label_id", "keyword_observations", ["label_id"])
    op.create_index("ix_kw_obs_observed_at", "keyword_observations", ["observed_at"])
    op.create_index("ix_kw_obs_lemma", "keyword_observations", ["lemma"])
    op.create_index("ix_kw_obs_promoted", "keyword_observations", ["promoted_to_active"])

    # ── entity_observations ───────────────────────────────────────────────────
    op.create_table(
        "entity_observations",
        sa.Column("obs_id", sa.String(64), primary_key=True),
        sa.Column("message_id", sa.String(512), sa.ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=True),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("start", sa.Integer, nullable=False),
        sa.Column("end", sa.Integer, nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("extractor_version", sa.String(64), nullable=False, server_default=""),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("value_hash", sa.String(64), nullable=True),
        sa.Column("value_enc", sa.Text, nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_unique_constraint(
        "uq_ent_obs_natural_key",
        "entity_observations",
        ["message_id", "text_hash", "entity_type", "start", "end", "source", "extractor_version"],
    )
    op.create_index("ix_ent_obs_message_id", "entity_observations", ["message_id"])
    op.create_index("ix_ent_obs_entity_type", "entity_observations", ["entity_type"])
    op.create_index("ix_ent_obs_observed_at", "entity_observations", ["observed_at"])

    # ── promotion_events ──────────────────────────────────────────────────────
    op.create_table(
        "promotion_events",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("pipeline_runs.run_id"), nullable=True),
        sa.Column("label_id", sa.String(128), nullable=False),
        sa.Column("lemma", sa.String(256), nullable=False),
        sa.Column("dict_type", sa.String(16), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("reason_code", sa.String(128), nullable=True),
        sa.Column("dict_version_prev", sa.Integer, nullable=False),
        sa.Column("dict_version_new", sa.Integer, nullable=False),
        sa.Column("doc_freq_at_promotion", sa.Integer, nullable=True),
        sa.Column("embedding_score_at_promotion", sa.Float, nullable=True),
        sa.Column("collision_labels", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_promo_run_id", "promotion_events", ["run_id"])
    op.create_index("ix_promo_label_id", "promotion_events", ["label_id"])
    op.create_index("ix_promo_dict_version_new", "promotion_events", ["dict_version_new"])

    # ── Seed label_registry with default topics ────────────────────────────────
    op.execute(
        """
        INSERT INTO label_registry (label_id, name, status) VALUES
        ('FATTURAZIONE',       'Fatturazione',       'active'),
        ('ASSISTENZA_TECNICA', 'Assistenza Tecnica', 'active'),
        ('RECLAMO',            'Reclamo',            'active'),
        ('INFO_COMMERCIALI',   'Info Commerciali',   'active'),
        ('DOCUMENTI',          'Documenti',          'active'),
        ('APPUNTAMENTO',       'Appuntamento',       'active'),
        ('CONTRATTO',          'Contratto',          'active'),
        ('GARANZIA',           'Garanzia',           'active'),
        ('SPEDIZIONE',         'Spedizione',         'active'),
        ('UNKNOWN_TOPIC',      'Unknown Topic',      'active')
        ON CONFLICT (label_id) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_table("promotion_events")
    op.drop_table("entity_observations")
    op.drop_table("keyword_observations")
    op.drop_table("lexicon_entries")
    op.drop_table("label_registry")
    op.drop_table("messages")
    op.drop_table("pipeline_runs")
