"""Causal technical indicators. Every function uses only current and past values.

Rolling-window indicators return NaN until the window is full and whenever the window
contains a missing candle. EWM-based ones (EMA, RSI, ATR, MACD) carry their state across
gaps; the feature builder blanks rows whose own candle is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(close: pd.Series, k: int = 1) -> pd.Series:
    return np.log(close / close.shift(k))


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (RMA), as used by RSI and ATR."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder((-delta).clip(lower=0), n)
    out = 100 - 100 / (1 + gain / loss)
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    return out.mask((loss == 0) & (gain == 0), 50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """Returns (%B, bandwidth)."""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    pctb = (close - lower) / (upper - lower)
    return pctb, (upper - lower) / mid


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.where(high.notna() & low.notna())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def drawdown(close: pd.Series, n: int) -> pd.Series:
    """Distance below the highest close of the last n periods (<= 0)."""
    return close / close.rolling(n, min_periods=n).max() - 1


def rolling_beta(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
    return y.rolling(n, min_periods=n).cov(x) / x.rolling(n, min_periods=n).var()
