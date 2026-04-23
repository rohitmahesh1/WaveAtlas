"""Add image upload artifact kind.

Revision ID: 0002_add_image_upload_artifact_kind
Revises: 0001_initial_schema
Create Date: 2026-04-22
"""

from __future__ import annotations

from alembic import op


revision = "0002_add_image_upload_artifact_kind"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TYPE artifactkind ADD VALUE IF NOT EXISTS 'upload_image'")


def downgrade() -> None:
    # PostgreSQL enum values cannot be removed safely without recreating the type.
    pass
