"""predictions.kind (live | oos) and unique key including it

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-17
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # New unique key is created before the old one is dropped: on MySQL the FK on model_run_id
    # needs an index whose first column is model_run_id at all times.
    with op.batch_alter_table("predictions") as batch:
        batch.add_column(sa.Column("kind", sa.String(8), server_default="live", nullable=False))
        batch.create_unique_constraint(
            "uq_predictions_run_time_kind", ["model_run_id", "as_of_time", "horizon_days", "kind"]
        )
        batch.drop_constraint("uq_predictions_key", type_="unique")


def downgrade() -> None:
    op.execute("DELETE FROM predictions WHERE kind <> 'live'")
    with op.batch_alter_table("predictions") as batch:
        batch.create_unique_constraint("uq_predictions_key", ["model_run_id", "as_of_time", "horizon_days"])
        batch.drop_constraint("uq_predictions_run_time_kind", type_="unique")
        batch.drop_column("kind")
