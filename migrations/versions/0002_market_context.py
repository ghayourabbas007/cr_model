"""market_context table for funding rate / open interest

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-17
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "market_context",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), autoincrement=True, nullable=False),
        sa.Column("symbol_id", sa.Integer(), nullable=False),
        sa.Column("metric", sa.String(32), nullable=False),
        sa.Column("ts", sa.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_market_context"),
        sa.ForeignKeyConstraint(
            ["symbol_id"], ["symbols.id"], name="fk_market_context_symbol_id_symbols", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("symbol_id", "metric", "ts", name="uq_market_context_key"),
    )


def downgrade() -> None:
    op.drop_table("market_context")
