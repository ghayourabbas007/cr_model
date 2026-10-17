"""Read-only REST API. Run with `forecaster serve` (binds to 127.0.0.1 by default).

Only GET endpoints exist; nothing here can place orders or modify market positions.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import timedelta

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from forecaster import DISCLAIMER, __version__
from forecaster.api.schemas import (
    BacktestResponse,
    BacktestSummary,
    DataStatus,
    ForecastResponse,
    HealthResponse,
    HorizonForecast,
    ModelRef,
    ModelRunInfo,
    ModelsResponse,
    PriceRange,
    ReturnQuantiles,
)
from forecaster.backtest.runner import latest_backtests
from forecaster.config import Settings, get_settings
from forecaster.data.store import last_open_time
from forecaster.db.models import ModelRun, Symbol
from forecaster.db.repository import get_symbol
from forecaster.logging_setup import get_logger
from forecaster.modeling.predict import NoActiveModel, forecast
from forecaster.modeling.registry import active_run
from forecaster.modeling.train import model_verdict
from forecaster.timeutil import utcnow

log = get_logger(__name__)
STALE_AFTER = timedelta(days=2)


class ForecastCache:
    """Forecasts only change when a new daily candle closes or a new model is activated,
    so cache by (symbol, horizon, model run, latest candle)."""

    def __init__(self, max_items: int = 256):
        self._data: dict[tuple, dict] = {}
        self._lock = threading.Lock()
        self.max_items = max_items

    def get(self, key: tuple) -> dict | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: tuple, value: dict) -> None:
        with self._lock:
            if len(self._data) >= self.max_items:
                self._data.pop(next(iter(self._data)))
            self._data[key] = value


def create_app(settings: Settings | None = None, session_factory: sessionmaker | None = None) -> FastAPI:
    s = settings or get_settings()
    if session_factory is None:
        from forecaster.db.session import get_sessionmaker
        session_factory = get_sessionmaker()

    app = FastAPI(
        title="Crypto Forecaster API",
        version=__version__,
        description=f"Read-only probabilistic forecasts for Binance spot markets. {DISCLAIMER}",
    )
    app.state.settings = s
    app.state.session_factory = session_factory
    app.state.cache = ForecastCache()
    if s.api_cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=s.api_cors_origins, allow_methods=["GET"],
                           allow_headers=["*"])

    def get_session(request: Request) -> Iterator[Session]:
        session = request.app.state.session_factory()
        try:
            yield session
        finally:
            session.close()

    def target_symbol(session: Session, symbol: str) -> Symbol:
        sym = get_symbol(session, symbol)
        if sym is None or not sym.is_target:
            raise HTTPException(404, f"{symbol.upper()} is not a configured forecast target "
                                     f"(TARGET_SYMBOLS={','.join(s.target_symbols)})")
        return sym

    @app.get("/health", response_model=HealthResponse, tags=["status"])
    def health(session: Session = Depends(get_session)) -> HealthResponse:
        try:
            session.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            log.exception("health check: database unavailable")
            db_ok = False
        data, n_active = [], 0
        if db_ok:
            now = utcnow()
            for name in s.target_symbols:
                sym = get_symbol(session, name)
                last = last_open_time(session, sym.id, "1d") if sym else None
                close = last + timedelta(days=1) if last else None
                data.append(DataStatus(symbol=name, last_candle_close=close,
                                       stale=close is None or now - close > STALE_AFTER))
            n_active = len(session.scalars(select(ModelRun.id).where(ModelRun.is_active.is_(True))).all())
        ok = db_ok and data and not any(d.stale for d in data) and n_active > 0
        return HealthResponse(status="ok" if ok else "degraded", database=db_ok, version=__version__,
                              data=data, active_models=n_active, disclaimer=DISCLAIMER)

    @app.get("/models", response_model=ModelsResponse, tags=["models"])
    def models(symbol: str | None = None, all_versions: bool = Query(False, alias="all"),
               session: Session = Depends(get_session)) -> ModelsResponse:
        q = select(ModelRun, Symbol.symbol).join(Symbol, Symbol.id == ModelRun.symbol_id)
        if symbol:
            q = q.where(Symbol.symbol == symbol.upper())
        if not all_versions:
            q = q.where(ModelRun.is_active.is_(True))
        q = q.order_by(Symbol.symbol, ModelRun.horizon_days, ModelRun.model_name, ModelRun.id.desc())
        out = []
        for run, sym_name in session.execute(q).all():
            m = run.metrics or {}
            cmp = m.get("vs_climatology") or {}
            out.append(ModelRunInfo(
                id=run.id, symbol=sym_name, model_name=run.model_name, model_type=run.model_type,
                version=run.version, horizon_days=run.horizon_days, is_active=run.is_active, status=run.status,
                train_start=run.train_start, train_end=run.train_end, created_at=run.created_at,
                accuracy=m.get("accuracy"), roc_auc=m.get("roc_auc"), brier=m.get("brier"),
                brier_skill_vs_climatology=cmp.get("brier_skill"), mae_q50=m.get("mae_q50"),
                coverage_80=m.get("coverage_80"), oos_rows=m.get("n"),
                verdict=model_verdict(run.model_name, m),
            ))
        return ModelsResponse(serving_model=s.serving_model, models=out)

    @app.get("/forecast/{symbol}", response_model=ForecastResponse, tags=["forecast"])
    def get_forecast(
        symbol: str,
        model: str | None = Query(None, description="Model name; default SERVING_MODEL"),
        horizons: str | None = Query(None, description="Comma-separated days, default FORECAST_HORIZONS"),
        session: Session = Depends(get_session),
    ) -> ForecastResponse:
        sym = target_symbol(session, symbol)
        name = model or s.serving_model
        try:
            hs = [int(h) for h in horizons.split(",")] if horizons else s.forecast_horizons
        except ValueError:
            raise HTTPException(422, "horizons must be comma-separated integers") from None
        last = last_open_time(session, sym.id, "1d")
        if last is None:
            raise HTTPException(503, f"No market data for {sym.symbol}; run `forecaster data backfill`")

        items = []
        for h in hs:
            run = active_run(session, sym.id, h, name)
            if run is None:
                raise HTTPException(503, f"No active '{name}' model for {sym.symbol} h={h}d; "
                                         f"run `forecaster train`")
            key = (sym.symbol, h, name, run.id, last)
            f = app.state.cache.get(key)
            if f is None:
                try:
                    f = forecast(session, s, sym.symbol, h, name, store=True)
                except NoActiveModel as exc:
                    raise HTTPException(503, str(exc)) from exc
                app.state.cache.put(key, f)
            items.append(HorizonForecast(
                horizon_days=h, as_of_time=f["as_of_time"], target_time=f["target_time"],
                reference_price=f["reference_price"], prob_up=f["prob_up"], prob_down=1 - f["prob_up"],
                direction="up" if f["prob_up"] >= 0.5 else "down",
                log_return_quantiles=ReturnQuantiles(q10=f["return_q10"], q50=f["return_q50"], q90=f["return_q90"]),
                price_range=PriceRange(low=f["price_q10"], median=f["price_q50"], high=f["price_q90"]),
                model=ModelRef(name=name, version=run.version, run_id=run.id,
                               backtest_interval_coverage=f["interval_coverage_backtest"],
                               walk_forward_verdict=model_verdict(name, run.metrics)),
                stale=f["stale"],
            ))
        return ForecastResponse(symbol=sym.symbol, generated_at=utcnow(), disclaimer=DISCLAIMER, forecasts=items)

    @app.get("/backtest/{symbol}", response_model=BacktestResponse, tags=["backtest"])
    def get_backtest(symbol: str, horizon: int | None = None, include_curve: bool = False,
                     session: Session = Depends(get_session)) -> BacktestResponse:
        sym = target_symbol(session, symbol)
        rows = latest_backtests(session, sym.id, horizon)
        if not rows:
            raise HTTPException(404, f"No backtests for {sym.symbol}; run `forecaster backtest`")
        results = []
        for r in sorted(rows, key=lambda r: (r.horizon_days, r.strategy)):
            p = r.params or {}
            results.append(BacktestSummary(
                id=r.id, strategy=r.strategy, horizon_days=r.horizon_days, model_run_id=r.model_run_id,
                model_version=p.get("model_version"), period_start=r.period_start, period_end=r.period_end,
                fee_bps=r.fee_bps, slippage_bps=r.slippage_bps, total_return=r.total_return, sharpe=r.sharpe,
                max_drawdown=r.max_drawdown, win_rate=r.win_rate, n_trades=r.n_trades,
                exposure=(p.get("strategy_stats") or {}).get("exposure"),
                benchmark_total_return=r.benchmark_total_return, benchmark_sharpe=r.benchmark_sharpe,
                benchmark_max_drawdown=r.benchmark_max_drawdown, beats_benchmark=r.beats_benchmark,
                p_value=(p.get("bootstrap") or {}).get("p_value"), verdict=p.get("verdict", ""),
                created_at=r.created_at, equity_curve=r.equity_curve if include_curve else None,
            ))
        return BacktestResponse(symbol=sym.symbol, disclaimer=DISCLAIMER, results=results)

    return app
