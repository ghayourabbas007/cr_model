"""ORM models.

Conventions:
- All datetimes are naive UTC (MySQL DATETIME has no timezone).
- Kline `open_time` is the candle's open time; a candle is only "available" at its close_time.
- Prices use NUMERIC for exact storage but are returned to Python as float for pandas.
- Types are kept portable (MySQL in production, SQLite in tests).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forecaster.db.base import Base

# BIGINT autoincrement on MySQL, INTEGER on SQLite (which only autoincrements INTEGER PKs).
BigId = BigInteger().with_variant(Integer(), "sqlite")
Price = Numeric(28, 12, asdecimal=False)
Qty = Numeric(38, 12, asdecimal=False)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class Symbol(TimestampMixin, Base):
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    base_asset: Mapped[str] = mapped_column(String(10), nullable=False)
    quote_asset: Mapped[str] = mapped_column(String(10), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # True for symbols we forecast; False for context-only symbols (e.g. BTC as a market factor).
    is_target: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    klines: Mapped[list[Kline]] = relationship(back_populates="symbol_ref")

    def __repr__(self) -> str:
        return f"<Symbol {self.symbol}>"


class Kline(Base):
    __tablename__ = "klines"
    __table_args__ = (
        UniqueConstraint("symbol_id", "interval", "open_time", name="uq_klines_symbol_interval_open"),
        Index("ix_klines_interval_open_time", "interval", "open_time"),
    )

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False)
    interval: Mapped[str] = mapped_column(String(4), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    close_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open: Mapped[float] = mapped_column(Price, nullable=False)
    high: Mapped[float] = mapped_column(Price, nullable=False)
    low: Mapped[float] = mapped_column(Price, nullable=False)
    close: Mapped[float] = mapped_column(Price, nullable=False)
    volume: Mapped[float] = mapped_column(Qty, nullable=False)
    quote_volume: Mapped[float] = mapped_column(Qty, nullable=False)
    trades: Mapped[int] = mapped_column(BigInteger, nullable=False)
    taker_buy_base_volume: Mapped[float | None] = mapped_column(Qty)
    taker_buy_quote_volume: Mapped[float | None] = mapped_column(Qty)
    # "api" or "dump" (data.binance.vision), useful when reconciling gaps.
    source: Mapped[str] = mapped_column(String(8), default="api", nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    symbol_ref: Mapped[Symbol] = relationship(back_populates="klines")


class FeatureRow(TimestampMixin, Base):
    """One feature vector per (symbol, interval, as_of_time, feature_set_version).

    `as_of_time` is the close time of the last candle used, i.e. the moment the features
    became knowable. Stored as JSON so the feature set can evolve without schema changes;
    `feature_set_version` ties rows to the code that produced them.
    """

    __tablename__ = "features"
    __table_args__ = (
        UniqueConstraint(
            "symbol_id", "interval", "as_of_time", "feature_set_version", name="uq_features_key"
        ),
    )

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False)
    interval: Mapped[str] = mapped_column(String(4), nullable=False)
    as_of_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(32), nullable=False)
    values: Mapped[dict] = mapped_column(JSON, nullable=False)


class ModelRun(TimestampMixin, Base):
    """A trained model version. `version` is unique per (model_name, symbol, horizon)."""

    __tablename__ = "model_runs"
    __table_args__ = (
        UniqueConstraint("model_name", "symbol_id", "horizon_days", "version", name="uq_model_runs_version"),
        Index("ix_model_runs_active", "symbol_id", "horizon_days", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "lgbm_direction"
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)  # baseline|lightgbm|xgboost|lstm
    task: Mapped[str] = mapped_column(String(16), nullable=False)  # classification|quantile|regression
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False)
    interval: Mapped[str] = mapped_column(String(4), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(32), nullable=False)
    train_start: Mapped[datetime | None] = mapped_column(DateTime)
    train_end: Mapped[datetime | None] = mapped_column(DateTime)
    params: Mapped[dict | None] = mapped_column(JSON)
    # Walk-forward validation metrics (accuracy, auc, brier, mae, coverage, ...).
    metrics: Mapped[dict | None] = mapped_column(JSON)
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str | None] = mapped_column(String(1024))


class Prediction(TimestampMixin, Base):
    """A forecast emitted at `as_of_time` for `horizon_days` ahead. All values are estimates."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("model_run_id", "as_of_time", "horizon_days", name="uq_predictions_key"),
        Index("ix_predictions_symbol_time", "symbol_id", "as_of_time"),
    )

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False)
    as_of_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    reference_price: Mapped[float] = mapped_column(Price, nullable=False)
    prob_up: Mapped[float | None] = mapped_column(Float)
    return_q10: Mapped[float | None] = mapped_column(Float)
    return_q50: Mapped[float | None] = mapped_column(Float)
    return_q90: Mapped[float | None] = mapped_column(Float)
    price_q10: Mapped[float | None] = mapped_column(Price)
    price_q50: Mapped[float | None] = mapped_column(Price)
    price_q90: Mapped[float | None] = mapped_column(Price)
    # Filled in once the horizon has elapsed, for live calibration tracking.
    realized_return: Mapped[float | None] = mapped_column(Float)


class BacktestResult(TimestampMixin, Base):
    __tablename__ = "backtest_results"
    __table_args__ = (Index("ix_backtest_symbol_created", "symbol_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"))
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    fee_bps: Mapped[float] = mapped_column(Float, nullable=False)
    slippage_bps: Mapped[float] = mapped_column(Float, nullable=False)
    total_return: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe: Mapped[float | None] = mapped_column(Float)
    win_rate: Mapped[float | None] = mapped_column(Float)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    n_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    benchmark_total_return: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_sharpe: Mapped[float | None] = mapped_column(Float)
    benchmark_max_drawdown: Mapped[float | None] = mapped_column(Float)
    # Explicit verdict so poor results can't be hidden: did the strategy beat buy-and-hold
    # on risk-adjusted return after costs?
    beats_benchmark: Mapped[bool] = mapped_column(Boolean, nullable=False)
    params: Mapped[dict | None] = mapped_column(JSON)
    # [{"t": iso8601, "strategy": equity, "benchmark": equity}, ...]
    equity_curve: Mapped[list | None] = mapped_column(JSON)
