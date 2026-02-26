"""Drop FK from promotion_events.run_id to pipeline_runs.

promotion_events.run_id rappresenta un promoter-run, non un pipeline-run:
rimuoviamo la FK e rendiamo la colonna nullable.

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-26
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rimuovi il vincolo FK (il nome è stato generato da SQLAlchemy)
    op.drop_constraint(
        "promotion_events_run_id_fkey",
        "promotion_events",
        type_="foreignkey",
    )
    # Rendi run_id nullable (i promoter run non sono pipeline run)
    op.alter_column(
        "promotion_events",
        "run_id",
        existing_type=sa.Text(),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "promotion_events",
        "run_id",
        existing_type=sa.Text(),
        nullable=False,
    )
    op.create_foreign_key(
        "promotion_events_run_id_fkey",
        "promotion_events",
        "pipeline_runs",
        ["run_id"],
        ["run_id"],
    )
