"""Pure functions for parsing, validating and checking continuity of kline data."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from forecaster.timeutil import interval_delta

RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]
COLUMNS = [
    "open_time", "close_time", "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_base_volume", "taker_buy_quote_volume",
]
_FLOAT_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
               "taker_buy_base_volume", "taker_buy_quote_volume"]

# Epoch values above this are microseconds (Binance spot dumps switched to µs in 2025).
_US_THRESHOLD = 10**14


def epoch_to_datetime(values: pd.Series) -> pd.Series:
    """Epoch ms or µs (mixed allowed) -> naive UTC datetimes."""
    v = pd.to_numeric(values).astype("int64")
    ms = np.where(v > _US_THRESHOLD, v // 1000, v)
    return pd.Series(pd.to_datetime(ms, unit="ms"), index=values.index)


def parse_klines(rows: list[list]) -> pd.DataFrame:
    """Raw Binance kline arrays (API or dump CSV rows) -> typed DataFrame with COLUMNS."""
    if len(rows) == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}).astype(
            {"open_time": "datetime64[ns]", "close_time": "datetime64[ns]", "trades": "int64"}
        )
    raw = pd.DataFrame([r[:12] for r in rows], columns=RAW_COLUMNS[: len(rows[0][:12])])
    df = pd.DataFrame({
        "open_time": epoch_to_datetime(raw["open_time"]),
        "close_time": epoch_to_datetime(raw["close_time"]),
    })
    for col in _FLOAT_COLS:
        df[col] = pd.to_numeric(raw[col], errors="coerce").astype("float64")
    df["trades"] = pd.to_numeric(raw["trades"], errors="coerce").astype("Int64")
    return df[COLUMNS]


def validate_klines(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate open_times (keep last), NaN rows and OHLC-inconsistent rows.

    Returns (clean_df sorted by open_time, number_of_rows_dropped).
    """
    n = len(df)
    out = df.drop_duplicates("open_time", keep="last")
    out = out.dropna(subset=["open_time", "close_time", "open", "high", "low", "close", "volume", "trades"])
    body_hi = out[["open", "close"]].max(axis=1)
    body_lo = out[["open", "close"]].min(axis=1)
    ok = (out["low"] > 0) & (out["high"] >= body_hi) & (out["low"] <= body_lo) & (out["volume"] >= 0)
    out = out[ok].sort_values("open_time").reset_index(drop=True)
    return out, n - len(out)


def only_closed(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Keep candles that have fully closed by `now` (the forming candle is excluded)."""
    return df[df["close_time"] < pd.Timestamp(now)].reset_index(drop=True)


def find_gaps(open_times: pd.Series | list, interval: str) -> list[tuple[datetime, datetime]]:
    """Missing candle ranges between the first and last stored candle.

    Returns [(first_missing_open_time, last_missing_open_time), ...].
    """
    t = pd.Series(pd.to_datetime(pd.Series(open_times))).drop_duplicates().sort_values().reset_index(drop=True)
    if len(t) < 2:
        return []
    delta = pd.Timedelta(interval_delta(interval))
    diffs = t.diff()
    gaps = []
    for i in np.flatnonzero((diffs > delta).to_numpy()):
        gaps.append(((t[i - 1] + delta).to_pydatetime(), (t[i] - delta).to_pydatetime()))
    return gaps
