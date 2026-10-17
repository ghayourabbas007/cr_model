from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecaster.features import indicators as ind


def s(values) -> pd.Series:
    return pd.Series(values, dtype="float64")


def test_sma_and_ema_warmup_and_values() -> None:
    x = s([1, 2, 3, 4, 5])
    assert ind.sma(x, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert ind.sma(x, 3).isna().sum() == 2
    e = ind.ema(x, 3)  # alpha = 0.5, seeded at the first value
    assert np.isnan(e[1]) and e[2] == pytest.approx(2.25)


def test_rsi_extremes() -> None:
    assert ind.rsi(s(range(1, 40)), 14).dropna().eq(100).all()
    assert ind.rsi(s(range(40, 1, -1)), 14).dropna().eq(0).all()
    assert ind.rsi(s([5.0] * 30), 14).dropna().eq(50).all()


def test_rsi_bounded() -> None:
    r = ind.rsi(pd.Series(np.random.default_rng(1).normal(0, 1, 300).cumsum() + 100), 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_true_range_uses_previous_close_on_gap_up() -> None:
    tr = ind.true_range(s([10, 12]), s([9, 11]), s([10, 11.5]))
    assert tr.tolist() == [1.0, 2.0]  # second bar: high 12 - prev close 10


def test_bollinger_pctb() -> None:
    pctb, width = ind.bollinger(s([1, 2, 3, 4, 5]), 5, 2.0)
    sd = np.sqrt(2)
    assert pctb.iloc[-1] == pytest.approx((5 - (3 - 2 * sd)) / (4 * sd))
    assert width.iloc[-1] == pytest.approx(4 * sd / 3)


def test_drawdown() -> None:
    assert ind.drawdown(s([10, 12, 9]), 3).iloc[-1] == pytest.approx(-0.25)


def test_macd_hist_is_line_minus_signal() -> None:
    x = pd.Series(np.linspace(100, 150, 80))
    line, sig, hist = ind.macd(x)
    pd.testing.assert_series_equal(hist, line - sig)
    assert line.dropna().gt(0).all()  # fast EMA above slow EMA in an uptrend


def test_rolling_beta_of_scaled_series() -> None:
    x = pd.Series(np.random.default_rng(2).normal(0, 1, 100))
    assert ind.rolling_beta(2 * x, x, 30).dropna().round(10).eq(2).all()


@pytest.mark.parametrize("fn", [
    lambda c: ind.sma(c, 10), lambda c: ind.ema(c, 10), lambda c: ind.rsi(c, 14),
    lambda c: ind.macd(c)[2], lambda c: ind.bollinger(c, 20)[0], lambda c: ind.drawdown(c, 30),
    lambda c: ind.atr(c * 1.01, c * 0.99, c, 14), lambda c: ind.log_return(c, 7),
])
def test_indicators_are_causal(fn) -> None:
    """Values up to t must not change when data after t is appended or altered."""
    c = pd.Series(np.exp(np.random.default_rng(3).normal(0, 0.02, 200).cumsum()) * 100)
    full = fn(c)
    altered = c.copy()
    altered.iloc[120:] *= 3.0
    for t in (60, 119):
        np.testing.assert_allclose(fn(c.iloc[: t + 1]).to_numpy(), full.iloc[: t + 1].to_numpy(), equal_nan=True)
    np.testing.assert_allclose(fn(altered).iloc[:120].to_numpy(), full.iloc[:120].to_numpy(), equal_nan=True)
