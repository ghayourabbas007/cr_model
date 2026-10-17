from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecaster.backtest.engine import StrategyParams, positions_from_signals, run_backtest
from forecaster.backtest.metrics import bootstrap_sharpe_diff, max_drawdown, sharpe, summarize, verdict
from forecaster.backtest.runner import backtest_model, latest_backtests
from forecaster.config import Settings
from forecaster.data.store import upsert_klines
from forecaster.db.models import BacktestResult
from forecaster.db.repository import sync_symbols
from forecaster.modeling.train import train_symbol
from tests.fakes import synthetic_klines

T0 = datetime(2022, 1, 1)


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range(T0, periods=n, freq="1D")


def _close(values) -> pd.Series:
    return pd.Series(values, index=_days(len(values)), dtype=float)


# --- signals ---------------------------------------------------------------------------------

def test_hysteresis_long_flat() -> None:
    prob = pd.Series([0.50, 0.56, 0.52, 0.49, 0.54, 0.60, np.nan, 0.40], index=_days(8))
    pos = positions_from_signals(prob)
    assert pos.tolist() == [0, 1, 1, 0, 0, 1, 1, 0]  # NaN keeps the position


def test_long_short_and_median_filter() -> None:
    prob = pd.Series([0.60, 0.40, 0.48, 0.60], index=_days(4))
    q50 = pd.Series([-0.01, -0.02, 0.0, 0.02], index=_days(4))
    p = StrategyParams(allow_short=True, require_median_agrees=True)
    assert positions_from_signals(prob, q50, p).tolist() == [0, -1, -1, 1]


# --- engine ------------------------------------------------------------------------------------

def test_signal_applies_to_next_day_return_only() -> None:
    """A rule that buys after up-days on an alternating series must lose: it is always one day
    late. A look-ahead bug (applying today's signal to today's return) would make it win."""
    close = _close([100, 110, 99, 108.9, 98.01, 107.811, 97.03, 106.73])
    up_today = (close.pct_change() > 0).astype(float)
    decisions = positions_from_signals(up_today, p=StrategyParams(enter_long=0.5, exit_long=0.5))
    out = run_backtest(close, decisions, cost_bps=0)
    assert out.equity.iloc[-1] < 1.0
    assert (out.position.shift(-1).dropna().index == out.position.index[:-1]).all()
    # Every day held is a day after an up-move, i.e. a down day on this series.
    held_days = out.position[out.position > 0].index
    assert (close.pct_change().loc[held_days] < 0).all()


def test_always_long_equals_buy_and_hold() -> None:
    close = _close(np.linspace(100, 150, 30))
    decisions = pd.Series(1.0, index=close.index[:-1])
    out = run_backtest(close, decisions, cost_bps=15)
    np.testing.assert_allclose(out.returns.to_numpy(), out.benchmark_returns.to_numpy())
    assert out.equity.iloc[-1] < close.iloc[-1] / close.iloc[0]  # costs were paid


def test_costs_on_flat_prices() -> None:
    close = _close([100.0] * 6)
    decisions = pd.Series([1, 0, 1, 0, 1], index=close.index[:-1], dtype=float)
    out = run_backtest(close, decisions, cost_bps=10)
    # held: [0,1,0,1,0,1] -> 5 position changes plus a final close-out = 6 charges of 10 bps
    assert out.costs.sum() == pytest.approx(6 * 0.001)
    assert out.equity.iloc[-1] == pytest.approx((1 - 0.001) ** 4 * (1 - 0.002))  # last day: flip + close-out
    assert len(out.trades) == 3 and all(t["return"] < 0 for t in out.trades)


def test_flip_long_to_short_costs_double() -> None:
    close = _close([100.0] * 4)
    out = run_backtest(close, pd.Series([1, -1, -1], index=close.index[:-1], dtype=float), cost_bps=10)
    assert out.costs.tolist() == pytest.approx([0, 0.001, 0.002, 0.001])


def test_price_gap_carries_position() -> None:
    close = _close([100, 100, 120, 120]).drop(_days(4)[2])  # day 2 missing; day 3 closes at 120
    out = run_backtest(close, pd.Series(1.0, index=close.index[:-1]), cost_bps=0)
    assert out.equity.iloc[-1] == pytest.approx(1.2)


# --- metrics -----------------------------------------------------------------------------------

def test_basic_metrics() -> None:
    assert max_drawdown(pd.Series([1.0, 1.2, 0.9, 1.3])) == pytest.approx(-0.25)
    assert sharpe(np.full(10, 0.01)) is None  # zero volatility
    s = summarize(pd.Series([0.0, 0.1, -0.05]), pd.Series([0, 1, 1]),
                  [{"return": 0.1}, {"return": -0.02}], pd.Series([0, 0.001, 0]))
    assert s["total_return"] == pytest.approx(1.1 * 0.95 - 1)
    assert s["win_rate"] == 0.5 and s["n_trades"] == 2 and s["exposure"] == pytest.approx(2 / 3)


def test_bootstrap_identical_series_has_no_edge() -> None:
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.03, 500))
    boot = bootstrap_sharpe_diff(r, r, n=200)
    assert boot["sharpe_diff"] == 0 and boot["p_value"] == 1.0


def test_bootstrap_detects_large_edge() -> None:
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0.0, 0.03, 1000))
    strat = bench * 0.5 + 0.004
    assert bootstrap_sharpe_diff(strat, bench, n=500)["p_value"] < 0.01


def test_verdicts_are_explicit() -> None:
    lose = verdict({"sharpe": 0.4}, {"sharpe": 0.9}, {"p_value": 0.8})
    lucky = verdict({"sharpe": 1.0}, {"sharpe": 0.9}, {"p_value": 0.4})
    real = verdict({"sharpe": 2.0}, {"sharpe": 0.5}, {"p_value": 0.01})
    assert lose == (False, "does NOT beat buy-and-hold (risk-adjusted, after costs)")
    assert lucky[0] and "NOT statistically significant" in lucky[1]
    assert real[0] and real[1].startswith("beats buy-and-hold")


# --- end-to-end --------------------------------------------------------------------------------

def test_backtest_trained_model_end_to_end(session: Session, tmp_path) -> None:
    syms = sync_symbols(session, ["SOLUSDT", "BTCUSDT", "ETHUSDT"], ["SOLUSDT"])
    for i, sym in enumerate(syms):
        upsert_klines(session, sym.id, "1d", synthetic_klines(T0, 800, seed=10 + i).reset_index(), "api")
    session.commit()
    s = Settings(model_dir=tmp_path, intervals=["1d"], include_funding_features=False,
                 wf_min_train_days=365, wf_test_days=90, bt_bootstrap_samples=100)
    now = T0 + timedelta(days=801)
    train_symbol(session, s, "SOLUSDT", 7, ["momentum"], now=now)

    rep = backtest_model(session, s, "SOLUSDT", 7, "momentum")
    assert isinstance(rep.beats_benchmark, bool) and rep.verdict
    row = session.get(BacktestResult, rep.result_id)
    assert row.fee_bps == s.fee_bps and row.slippage_bps == s.slippage_bps
    assert row.period_start < row.period_end
    assert len(row.equity_curve) == rep.strategy["days"]
    assert row.params["verdict"] == rep.verdict

    backtest_model(session, s, "SOLUSDT", 7, "momentum")
    latest = latest_backtests(session, syms[0].id, 7)
    assert len(latest) == 1 and latest[0].id > rep.result_id
    assert len(session.scalars(select(BacktestResult)).all()) == 2
