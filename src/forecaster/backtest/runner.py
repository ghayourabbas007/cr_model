"""Backtest a trained model's walk-forward out-of-sample predictions and store the result."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecaster.backtest.engine import StrategyParams, positions_from_signals, run_backtest
from forecaster.backtest.metrics import bootstrap_sharpe_diff, summarize, verdict
from forecaster.config import Settings
from forecaster.data.store import load_klines
from forecaster.db.models import BacktestResult, ModelRun, Prediction
from forecaster.db.repository import get_symbol
from forecaster.logging_setup import get_logger
from forecaster.modeling.registry import active_run

log = get_logger(__name__)


@dataclass
class BacktestReport:
    symbol: str
    horizon: int
    model_name: str
    model_run_id: int
    result_id: int
    strategy: dict
    benchmark: dict
    bootstrap: dict
    beats_benchmark: bool
    verdict: str


def strategy_params(s: Settings) -> StrategyParams:
    return StrategyParams(enter_long=s.bt_enter_long, exit_long=s.bt_exit_long, allow_short=s.bt_allow_short,
                          enter_short=s.bt_enter_short, exit_short=s.bt_exit_short,
                          require_median_agrees=s.bt_require_median_agrees)


def load_oos_predictions(session: Session, run_id: int) -> pd.DataFrame:
    rows = session.execute(
        select(Prediction.as_of_time, Prediction.prob_up, Prediction.return_q50)
        .where(Prediction.model_run_id == run_id, Prediction.kind == "oos")
        .order_by(Prediction.as_of_time)
    ).all()
    df = pd.DataFrame(rows, columns=["as_of_time", "prob_up", "q50"])
    return df.set_index(pd.DatetimeIndex(df.pop("as_of_time")))


def daily_closes_by_close_time(session: Session, symbol: str) -> pd.Series:
    k = load_klines(session, symbol, "1d")
    return pd.Series(k["close"].to_numpy(), index=k.index + pd.Timedelta(days=1), name="close")


def backtest_model(session: Session, s: Settings, symbol: str, horizon: int, model_name: str,
                   params: StrategyParams | None = None, run: ModelRun | None = None) -> BacktestReport:
    params = params or strategy_params(s)
    sym = get_symbol(session, symbol)
    if sym is None:
        raise LookupError(f"Unknown symbol {symbol}")
    run = run or active_run(session, sym.id, horizon, model_name)
    if run is None:
        raise LookupError(f"No active {model_name} model for {symbol} h={horizon}; run `forecaster train`")

    oos = load_oos_predictions(session, run.id)
    if oos.empty:
        raise LookupError(f"Run {run.id} has no out-of-sample predictions to backtest")
    decisions = positions_from_signals(oos["prob_up"], oos["q50"], params)
    out = run_backtest(daily_closes_by_close_time(session, symbol), decisions, s.fee_bps + s.slippage_bps)

    strat = summarize(out.returns, out.position, out.trades, out.costs)
    bench = summarize(out.benchmark_returns)
    boot = bootstrap_sharpe_diff(out.returns, out.benchmark_returns, n=s.bt_bootstrap_samples,
                                 block=s.bt_bootstrap_block, seed=s.random_seed)
    beats, text = verdict(strat, bench, boot)

    curve = [{"t": t.isoformat(), "strategy": float(a), "benchmark": float(b), "position": float(p)}
             for t, a, b, p in zip(out.equity.index, out.equity, out.benchmark_equity, out.position)]
    res = BacktestResult(
        model_run_id=run.id, symbol_id=sym.id, strategy=f"{model_name}_{params.name()}", horizon_days=horizon,
        period_start=out.returns.index[0].to_pydatetime(), period_end=out.returns.index[-1].to_pydatetime(),
        fee_bps=s.fee_bps, slippage_bps=s.slippage_bps, total_return=strat["total_return"],
        sharpe=strat["sharpe"], win_rate=strat["win_rate"], max_drawdown=strat["max_drawdown"],
        n_trades=strat["n_trades"], benchmark_total_return=bench["total_return"],
        benchmark_sharpe=bench["sharpe"], benchmark_max_drawdown=bench["max_drawdown"],
        beats_benchmark=beats,
        params={"strategy_params": asdict(params), "model_version": run.version, "strategy_stats": strat,
                "benchmark_stats": bench, "bootstrap": boot, "verdict": text},
        equity_curve=curve,
    )
    session.add(res)
    session.commit()
    log.info("backtest complete", extra={"symbol": symbol, "horizon": horizon, "model": model_name,
                                         "total_return": round(strat["total_return"], 4),
                                         "bh_total_return": round(bench["total_return"], 4),
                                         "sharpe": strat["sharpe"], "bh_sharpe": bench["sharpe"], "verdict": text})
    return BacktestReport(symbol, horizon, model_name, run.id, res.id, strat, bench, boot, beats, text)


def latest_backtests(session: Session, symbol_id: int, horizon: int | None = None) -> list[BacktestResult]:
    """Most recent backtest per (strategy, horizon) for a symbol."""
    q = select(BacktestResult).where(BacktestResult.symbol_id == symbol_id)
    if horizon is not None:
        q = q.where(BacktestResult.horizon_days == horizon)
    latest: dict[tuple[str, int], BacktestResult] = {}
    for r in session.scalars(q.order_by(BacktestResult.created_at.desc(), BacktestResult.id.desc())):
        latest.setdefault((r.strategy, r.horizon_days), r)
    return list(latest.values())
