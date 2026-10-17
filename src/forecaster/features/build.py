"""Feature and target construction on a daily calendar.

Timing convention (the core leakage rule):
    A daily row is indexed by the candle's `open_time` t. Its features use candles up to and
    including t, which closes at t + 1 day. That moment is `as_of_time`: the earliest time the
    row could exist in real life. Any input that is joined in (other symbols, 4h candles,
    funding, open interest) must have been published at or before `as_of_time`.

Targets look forward from `as_of_time` and live in a separate frame, so they can never be
used as features by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from forecaster.features import indicators as ind

FEATURE_SET_VERSION = "v1"
META_COLUMNS = ["as_of_time", "ref_close"]

DAY = pd.Timedelta(days=1)
H4 = pd.Timedelta(hours=4)

RETURN_LOOKBACKS = (1, 3, 7, 14, 30, 60)
VOL_WINDOWS = (7, 14, 30)
SMA_WINDOWS = (7, 20, 50, 200)
EMA_WINDOWS = (12, 26)
DRAWDOWN_WINDOWS = (30, 90)
CROSS_RETURN_LOOKBACKS = (1, 7, 30)
CROSS_WINDOW = 30
INTRADAY_RV_BARS = 42  # 7 days of 4h bars


@dataclass
class FeatureInputs:
    """Raw inputs for one target symbol. Kline frames are indexed by open_time."""

    daily: pd.DataFrame
    cross: dict[str, pd.DataFrame] = field(default_factory=dict)  # prefix -> 1d klines
    intraday: pd.DataFrame | None = None                           # 4h klines
    funding: pd.Series | None = None                               # index = funding time
    open_interest: pd.Series | None = None                         # index = daily bucket start


def to_calendar(df: pd.DataFrame, freq: pd.Timedelta) -> pd.DataFrame:
    """Reindex onto a gap-free calendar. Missing candles become NaN rows, never filled."""
    if df.empty:
        return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, name="open_time")
    return df[~df.index.duplicated(keep="last")].reindex(idx)


def align_asof(values: pd.Series, available_at: pd.Index, as_of: pd.Index, tolerance: pd.Timedelta) -> np.ndarray:
    """For each as_of time, the latest value whose availability time is <= as_of.

    Values older than `tolerance` are treated as missing rather than carried forward.
    """
    # merge_asof needs identical datetime resolutions on both sides.
    src = pd.DataFrame({"t": pd.DatetimeIndex(available_at).astype("datetime64[ns]"),
                        "v": values.to_numpy()}).dropna()
    src = src.sort_values("t")
    dst = pd.DataFrame({"t": pd.DatetimeIndex(as_of).astype("datetime64[ns]")})
    if src.empty:
        return np.full(len(dst), np.nan)
    merged = pd.merge_asof(dst, src, on="t", direction="backward", tolerance=tolerance,
                           allow_exact_matches=True)
    return merged["v"].to_numpy()


def build_features(inp: FeatureInputs) -> pd.DataFrame:
    """Daily feature frame indexed by open_time with META_COLUMNS + feature columns."""
    d = to_calendar(inp.daily, DAY)
    missing = d["close"].isna()
    c, h, lo, v = d["close"], d["high"], d["low"], d["volume"]
    r1 = ind.log_return(c, 1)
    as_of = d.index + DAY

    f: dict[str, pd.Series] = {}
    for k in RETURN_LOOKBACKS:
        f[f"ret_{k}"] = ind.log_return(c, k)
    for w in VOL_WINDOWS:
        f[f"vol_{w}"] = r1.rolling(w, min_periods=w).std()
    f["atr_14"] = ind.atr(h, lo, c, 14) / c
    f["range_1"] = (h - lo) / c
    for w in SMA_WINDOWS:
        f[f"sma_ratio_{w}"] = c / ind.sma(c, w) - 1
    for w in EMA_WINDOWS:
        f[f"ema_ratio_{w}"] = c / ind.ema(c, w) - 1
    f["rsi_14"] = ind.rsi(c, 14)
    line, sig, hist = ind.macd(c)
    f["macd"], f["macd_signal"], f["macd_hist"] = line / c, sig / c, hist / c
    f["bb_pctb_20"], f["bb_width_20"] = ind.bollinger(c, 20)
    f["volume_ratio_20"] = np.log(v / ind.sma(v, 20))
    f["volume_trend_5_20"] = ind.sma(v, 5) / ind.sma(v, 20) - 1
    f["trades_ratio_20"] = np.log(d["trades"] / ind.sma(d["trades"], 20))
    f["taker_buy_ratio_7"] = ind.sma(d["taker_buy_base_volume"], 7) / ind.sma(v, 7)
    for w in DRAWDOWN_WINDOWS:
        f[f"drawdown_{w}"] = ind.drawdown(c, w)

    for prefix, xdf in inp.cross.items():
        if xdf.empty:
            continue
        x = to_calendar(xdf, DAY)["close"].reindex(d.index)  # same open_time => same as_of
        xr1 = ind.log_return(x, 1)
        for k in CROSS_RETURN_LOOKBACKS:
            f[f"{prefix}_ret_{k}"] = ind.log_return(x, k)
        f[f"{prefix}_corr_{CROSS_WINDOW}"] = r1.rolling(CROSS_WINDOW, min_periods=CROSS_WINDOW).corr(xr1)
        f[f"{prefix}_beta_{CROSS_WINDOW}"] = ind.rolling_beta(r1, xr1, CROSS_WINDOW)
        f[f"rel_ret_7_vs_{prefix}"] = f["ret_7"] - f[f"{prefix}_ret_7"]

    if inp.intraday is not None and not inp.intraday.empty:
        i = to_calendar(inp.intraday, H4)
        rv = ind.log_return(i["close"], 1).rolling(INTRADAY_RV_BARS, min_periods=INTRADAY_RV_BARS).std()
        f["rv_4h_7d"] = pd.Series(align_asof(rv, i.index + H4, as_of, H4), index=d.index)

    if inp.funding is not None and not inp.funding.empty:
        fr = inp.funding.sort_index()
        fr = fr[~fr.index.duplicated(keep="last")]
        fr7 = fr.rolling("7D").mean()  # window (t-7d, t]: causal
        f["funding_last"] = pd.Series(align_asof(fr, fr.index, as_of, DAY), index=d.index)
        f["funding_7d"] = pd.Series(align_asof(fr7, fr7.index, as_of, DAY), index=d.index)

    if inp.open_interest is not None and not inp.open_interest.empty:
        oi = inp.open_interest.sort_index()
        oi = oi.reindex(pd.date_range(oi.index.min(), oi.index.max(), freq=DAY))
        chg = np.log(oi / oi.shift(7))
        # Conservative: a daily OI bucket is treated as known only once the day has ended.
        f["oi_change_7"] = pd.Series(align_asof(chg, chg.index + DAY, as_of, DAY), index=d.index)

    out = pd.DataFrame(f, index=d.index)
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[missing.to_numpy(), :] = np.nan
    out.insert(0, "ref_close", c)
    out.insert(0, "as_of_time", as_of)
    out.index.name = "open_time"
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in META_COLUMNS and not col.startswith("fwd_")
            and not col.startswith("target_end_time")]


def build_targets(daily: pd.DataFrame, horizons: list[int] | tuple[int, ...]) -> pd.DataFrame:
    """Forward targets per row: log return from close(t) to close(t + h days), direction,
    and `target_end_time_h` (when the target becomes known). NaN where the future is unknown."""
    d = to_calendar(daily, DAY)
    c = d["close"]
    out = pd.DataFrame(index=d.index)
    for hz in horizons:
        lr = np.log(c.shift(-hz) / c)
        out[f"fwd_logret_{hz}"] = lr
        out[f"fwd_up_{hz}"] = (lr > 0).astype(float).where(lr.notna())
        out[f"target_end_time_{hz}"] = d.index + DAY + pd.Timedelta(days=hz)
    return out


def make_dataset(features: pd.DataFrame, targets: pd.DataFrame, horizon: int,
                 now: datetime | None = None) -> pd.DataFrame:
    """Join features with one horizon's targets for training.

    Keeps only rows whose target is realized (target_end_time <= now) and non-NaN.
    """
    cols = [f"fwd_logret_{horizon}", f"fwd_up_{horizon}", f"target_end_time_{horizon}"]
    df = features.join(targets[cols], how="inner")
    if now is not None:
        df = df[df[f"target_end_time_{horizon}"] <= pd.Timestamp(now)]
    return df.dropna(subset=[f"fwd_logret_{horizon}"])


# --- leakage guard -------------------------------------------------------------------------

class LeakageError(AssertionError):
    pass


def truncate_inputs(inp: FeatureInputs, cutoff: datetime) -> FeatureInputs:
    """Inputs as they would have looked at `cutoff`: only data published at or before it."""
    cut = pd.Timestamp(cutoff)

    def klines(df: pd.DataFrame | None, span: pd.Timedelta):
        return None if df is None else df[df.index + span <= cut]

    return FeatureInputs(
        daily=klines(inp.daily, DAY),
        cross={k: klines(v, DAY) for k, v in inp.cross.items()},
        intraday=klines(inp.intraday, H4),
        funding=None if inp.funding is None else inp.funding[inp.funding.index <= cut],
        open_interest=None if inp.open_interest is None else inp.open_interest[inp.open_interest.index + DAY <= cut],
    )


def check_no_lookahead(inp: FeatureInputs, cutoffs: list[datetime], builder=build_features,
                       rtol: float = 1e-9, atol: float = 1e-12) -> None:
    """Rebuild features from data truncated at each cutoff and require that every row known by
    then is identical to the full-history build. Raises LeakageError listing offending columns."""
    full = builder(inp)
    cols = feature_columns(full)
    for cutoff in cutoffs:
        part = builder(truncate_inputs(inp, cutoff))
        known = full[full["as_of_time"] <= pd.Timestamp(cutoff)]
        if not part.index.isin(known.index).all():
            raise LeakageError(f"Truncated build at {cutoff} has rows not knowable by then")
        # Trailing calendar rows for missing candles exist only in the full build; they must be empty.
        extra = known.index.difference(part.index)
        if len(extra) and known.loc[extra, cols].notna().to_numpy().any():
            raise LeakageError(f"Rows {list(extra)} have values at {cutoff} but no source candle")
        known = known.loc[part.index]
        bad = []
        for col in cols:
            a = known[col].to_numpy(dtype=float)
            b = part[col].to_numpy(dtype=float) if col in part else np.full(len(a), np.nan)
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                bad.append(col)
        if bad:
            raise LeakageError(f"Features use data after {cutoff}: {bad}")


def default_cutoffs(inp: FeatureInputs, n: int = 5) -> list[datetime]:
    """Evenly spaced cutoffs across the history, aligned to daily closes."""
    idx = inp.daily.index
    picks = np.linspace(len(idx) * 0.3, len(idx) - 2, n).astype(int)
    return [(idx[p] + DAY).to_pydatetime() for p in picks] + [(idx[-1] + DAY - timedelta(hours=12)).to_pydatetime()]
