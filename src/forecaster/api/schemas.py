"""Response models. All forecast values are estimates; responses carry the disclaimer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PriceRange(BaseModel):
    low: float = Field(description="10th percentile price")
    median: float = Field(description="50th percentile price")
    high: float = Field(description="90th percentile price")
    interval: float = Field(0.80, description="Nominal probability that the price ends inside [low, high]")


class ReturnQuantiles(BaseModel):
    """Quantiles of the h-day forward log return."""

    q10: float
    q50: float
    q90: float


class ModelRef(BaseModel):
    name: str
    version: str
    run_id: int
    backtest_interval_coverage: float | None = Field(
        None, description="Share of out-of-sample outcomes that fell inside the 80% range")
    walk_forward_verdict: str


class HorizonForecast(BaseModel):
    horizon_days: int
    as_of_time: datetime = Field(description="Close time of the last candle used (UTC)")
    target_time: datetime
    reference_price: float
    prob_up: float
    prob_down: float
    direction: Literal["up", "down"]
    log_return_quantiles: ReturnQuantiles
    price_range: PriceRange
    model: ModelRef
    stale: bool = Field(description="True if the latest candle is more than 2 days old")


class ForecastResponse(BaseModel):
    symbol: str
    generated_at: datetime
    is_estimate: bool = True
    disclaimer: str
    forecasts: list[HorizonForecast]


class BacktestSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    strategy: str
    horizon_days: int
    model_run_id: int | None
    model_version: str | None
    period_start: datetime
    period_end: datetime
    fee_bps: float
    slippage_bps: float
    total_return: float
    sharpe: float | None
    max_drawdown: float
    win_rate: float | None
    n_trades: int
    exposure: float | None
    benchmark_total_return: float
    benchmark_sharpe: float | None
    benchmark_max_drawdown: float | None
    beats_benchmark: bool
    p_value: float | None = Field(description="Bootstrap p-value of the Sharpe difference vs buy-and-hold")
    verdict: str
    created_at: datetime
    equity_curve: list[dict] | None = None


class BacktestResponse(BaseModel):
    symbol: str
    disclaimer: str
    results: list[BacktestSummary]


class ModelRunInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    symbol: str
    model_name: str
    model_type: str
    version: str
    horizon_days: int
    is_active: bool
    status: str
    train_start: datetime | None
    train_end: datetime | None
    created_at: datetime
    accuracy: float | None = None
    roc_auc: float | None = None
    brier: float | None = None
    brier_skill_vs_climatology: float | None = None
    mae_q50: float | None = None
    coverage_80: float | None = None
    oos_rows: int | None = None
    verdict: str


class ModelsResponse(BaseModel):
    serving_model: str
    models: list[ModelRunInfo]


class DataStatus(BaseModel):
    symbol: str
    last_candle_close: datetime | None
    stale: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    version: str
    read_only: bool = True
    data: list[DataStatus]
    active_models: int
    disclaimer: str
