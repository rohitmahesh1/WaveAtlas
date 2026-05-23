"""Add event and fit metadata columns.

Revision ID: 0003_event_fit_metadata
Revises: 0002_image_upload_kind
Create Date: 2026-05-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_event_fit_metadata"
down_revision = "0002_image_upload_kind"
branch_labels = None
depends_on = None


TABLES = ("waves", "peaks")
COLUMNS = ("event_polarity", "event_kind", "fit_target")


def upgrade() -> None:
    for table in TABLES:
        for column in COLUMNS:
            op.add_column(table, sa.Column(column, sa.String(), nullable=True))
            op.create_index(f"ix_{table}_{column}", table, [column])
        op.create_index(f"ix_{table}_job_event_kind", table, ["job_id", "event_kind"])

    _backfill_from_metrics()


def downgrade() -> None:
    for table in TABLES:
        op.drop_index(f"ix_{table}_job_event_kind", table_name=table)
        for column in reversed(COLUMNS):
            op.drop_index(f"ix_{table}_{column}", table_name=table)
            op.drop_column(table, column)


def _backfill_from_metrics() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        for table in TABLES:
            for column in COLUMNS:
                op.execute(
                    f"UPDATE {table} "
                    f"SET {column} = metrics ->> '{column}' "
                    f"WHERE {column} IS NULL AND metrics ->> '{column}' IS NOT NULL"
                )
    elif dialect == "sqlite":
        for table in TABLES:
            for column in COLUMNS:
                op.execute(
                    f"UPDATE {table} "
                    f"SET {column} = json_extract(metrics, '$.{column}') "
                    f"WHERE {column} IS NULL AND json_valid(metrics)"
                )
