"""initial schema: symbols, klines, features, model_runs, predictions, backtest_results

Revision ID: 0001
Revises:
Create Date: 2026-10-17
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BigId = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
DateTimeMs = sa.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql")
Price = sa.Numeric(28, 12, asdecimal=False)
Qty = sa.Numeric(38, 12, asdecimal=False)


def _created_at() -> sa.Column:
    return sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False)


def upgrade() -> None:
    op.create_table(
        "symbols",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("base_asset", sa.String(10), nullable=False),
        sa.Column("quote_asset", sa.String(10), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_target", sa.Boolean(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_symbols"),
        sa.UniqueConstraint("symbol", name="uq_symbols_symbol"),
    )

    op.create_table(
        "klines",
        sa.Column("id", BigId, autoincrement=True, nullable=False),
        sa.Column("symbol_id", sa.Integer(), nullable=False),
        sa.Column("interval", sa.String(4), nullable=False),
        sa.Column("open_time", sa.DateTime(), nullable=False),
        sa.Column("close_time", DateTimeMs, nullable=False),
        sa.Column("open", Price, nullable=False),
        sa.Column("high", Price, nullable=False),
        sa.Column("low", Price, nullable=False),
        sa.Column("close", Price, nullable=False),
        sa.Column("volume", Qty, nullable=False),
        sa.Column("quote_volume", Qty, nullable=False),
        sa.Column("trades", sa.BigInteger(), nullable=False),
        sa.Column("taker_buy_base_volume", Qty, nullable=True),
        sa.Column("taker_buy_quote_volume", Qty, nullable=True),
        sa.Column("source", sa.String(8), nullable=False),
        sa.Column("ingested_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_klines"),
        sa.ForeignKeyConstraint(
            ["symbol_id"], ["symbols.id"], name="fk_klines_symbol_id_symbols", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("symbol_id", "interval", "open_time", name="uq_klines_symbol_interval_open"),
    )
    op.create_index("ix_klines_interval_open_time", "klines", ["interval", "open_time"])

    op.create_table(
        "features",
        sa.Column("id", BigId, autoincrement=True, nullable=False),
        sa.Column("symbol_id", sa.Integer(), nullable=False),
        sa.Column("interval", sa.String(4), nullable=False),
        sa.Column("as_of_time", sa.DateTime(), nullable=False),
        sa.Column("feature_set_version", sa.String(32), nullable=False),
        sa.Column("feature_values", sa.JSON(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_features"),
        sa.ForeignKeyConstraint(
            ["symbol_id"], ["symbols.id"], name="fk_features_symbol_id_symbols", ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "symbol_id", "interval", "as_of_time", "feature_set_version", name="uq_features_key"
        ),
    )

    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_type", sa.String(32), nullable=False),
        sa.Column("task", sa.String(16), nullable=False),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("symbol_id", sa.Integer(), nullable=False),
        sa.Column("interval", sa.String(4), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("feature_set_version", sa.String(32), nullable=False),
        sa.Column("train_start", sa.DateTime(), nullable=True),
        sa.Column("train_end", sa.DateTime(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("artifact_path", sa.String(512), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.String(1024), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_model_runs"),
        sa.ForeignKeyConstraint(
            ["symbol_id"], ["symbols.id"], name="fk_model_runs_symbol_id_symbols", ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "model_name", "symbol_id", "horizon_days", "version", name="uq_model_runs_version"
        ),
    )
    op.create_index("ix_model_runs_active", "model_runs", ["symbol_id", "horizon_days", "is_active"])

    op.create_table(
        "predictions",
        sa.Column("id", BigId, autoincrement=True, nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=False),
        sa.Column("symbol_id", sa.Integer(), nullable=False),
        sa.Column("as_of_time", sa.DateTime(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("reference_price", Price, nullable=False),
        sa.Column("prob_up", sa.Float(), nullable=True),
        sa.Column("return_q10", sa.Float(), nullable=True),
        sa.Column("return_q50", sa.Float(), nullable=True),
        sa.Column("return_q90", sa.Float(), nullable=True),
        sa.Column("price_q10", Price, nullable=True),
        sa.Column("price_q50", Price, nullable=True),
        sa.Column("price_q90", Price, nullable=True),
        sa.Column("realized_return", sa.Float(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_predictions"),
        sa.ForeignKeyConstraint(
            ["model_run_id"], ["model_runs.id"],
            name="fk_predictions_model_run_id_model_runs", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["symbol_id"], ["symbols.id"], name="fk_predictions_symbol_id_symbols", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("model_run_id", "as_of_time", "horizon_days", name="uq_predictions_key"),
    )
    op.create_index("ix_predictions_symbol_time", "predictions", ["symbol_id", "as_of_time"])

    op.create_table(
        "backtest_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=True),
        sa.Column("symbol_id", sa.Integer(), nullable=False),
        sa.Column("strategy", sa.String(64), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("period_start", sa.DateTime(), nullable=False),
        sa.Column("period_end", sa.DateTime(), nullable=False),
        sa.Column("fee_bps", sa.Float(), nullable=False),
        sa.Column("slippage_bps", sa.Float(), nullable=False),
        sa.Column("total_return", sa.Float(), nullable=False),
        sa.Column("sharpe", sa.Float(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("max_drawdown", sa.Float(), nullable=False),
        sa.Column("n_trades", sa.Integer(), nullable=False),
        sa.Column("benchmark_total_return", sa.Float(), nullable=False),
        sa.Column("benchmark_sharpe", sa.Float(), nullable=True),
        sa.Column("benchmark_max_drawdown", sa.Float(), nullable=True),
        sa.Column("beats_benchmark", sa.Boolean(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("equity_curve", sa.JSON(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_backtest_results"),
        sa.ForeignKeyConstraint(
            ["model_run_id"], ["model_runs.id"],
            name="fk_backtest_results_model_run_id_model_runs", ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["symbol_id"], ["symbols.id"],
            name="fk_backtest_results_symbol_id_symbols", ondelete="CASCADE",
        ),
    )
    op.create_index("ix_backtest_symbol_created", "backtest_results", ["symbol_id", "created_at"])


def downgrade() -> None:
    op.drop_table("backtest_results")
    op.drop_table("predictions")
    op.drop_table("model_runs")
    op.drop_table("features")
    op.drop_table("klines")
    op.drop_table("symbols")
