from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from forecaster.db.repository import sync_symbols
from forecaster.features.build import (
    META_COLUMNS,
    FeatureInputs,
    LeakageError,
    build_features,
    build_targets,
    check_no_lookahead,
    default_cutoffs,
    feature_columns,
    make_dataset,
)
from forecaster.features.store import load_features, save_features
from tests.fakes import synthetic_klines

T0 = datetime(2021, 1, 1)
N = 400
DAY = timedelta(days=1)


def _funding(start: datetime, days: int, seed: int = 9) -> pd.Series:
    idx = pd.date_range(start, periods=days * 3, freq="8h")
    return pd.Series(np.random.default_rng(seed).normal(1e-4, 5e-5, len(idx)), index=idx, name="funding_rate")


def _oi(start: datetime, days: int) -> pd.Series:
    idx = pd.date_range(start, periods=days, freq="1D")
    return pd.Series(np.linspace(1e6, 2e6, days), index=idx, name="open_interest")


@pytest.fixture
def inputs() -> FeatureInputs:
    return FeatureInputs(
        daily=synthetic_klines(T0, N, seed=1),
        cross={"btc": synthetic_klines(T0 - 50 * DAY, N + 80, seed=2),
               "eth": synthetic_klines(T0, N + 5, seed=3)},
        intraday=synthetic_klines(T0, N * 6, "4h", seed=4, vol=0.01),
        funding=_funding(T0 + 30 * DAY, N),
        open_interest=_oi(T0 + 300 * DAY, 120),
    )


def test_feature_frame_shape_and_timing(inputs: FeatureInputs) -> None:
    f = build_features(inputs)
    assert len(f) == N
    assert (f["as_of_time"] == f.index + pd.Timedelta(days=1)).all()
    cols = feature_columns(f)
    for expected in ["ret_7", "vol_30", "atr_14", "sma_ratio_200", "rsi_14", "macd_hist", "bb_pctb_20",
                     "volume_ratio_20", "drawdown_90", "btc_ret_7", "btc_corr_30", "eth_beta_30",
                     "rel_ret_7_vs_btc", "rv_4h_7d", "funding_7d", "oi_change_7"]:
        assert expected in cols, expected
    assert not set(META_COLUMNS) & set(cols)
    assert np.isfinite(f[cols].to_numpy(dtype=float)[~np.isnan(f[cols].to_numpy(dtype=float))]).all()


def test_no_lookahead_full_feature_set(inputs: FeatureInputs) -> None:
    cutoffs = default_cutoffs(inputs, 6) + [
        T0 + 250 * DAY + timedelta(hours=3),   # mid-candle
        T0 + 251 * DAY,                         # exactly at a daily close
    ]
    check_no_lookahead(inputs, cutoffs)


@pytest.mark.parametrize("leak", [
    lambda df: df["ref_close"].shift(-1),                         # tomorrow's price
    lambda df: df["ret_1"].rolling(5, center=True).mean(),        # centred window
    lambda df: df["ret_1"] - df["ret_1"].mean(),                  # full-sample normalisation
])
def test_leakage_check_catches_leaky_features(inputs: FeatureInputs, leak) -> None:
    def leaky_builder(inp):
        df = build_features(inp)
        df["leaky"] = leak(df)
        return df

    with pytest.raises(LeakageError, match="leaky"):
        check_no_lookahead(inputs, default_cutoffs(inputs, 3), builder=leaky_builder)


def test_missing_candle_is_not_interpolated(inputs: FeatureInputs) -> None:
    gap_day = inputs.daily.index[250]
    inputs.daily = inputs.daily.drop(gap_day)
    f = build_features(inputs)
    assert gap_day in f.index and np.isnan(f.loc[gap_day, "ref_close"])
    assert f.loc[gap_day, feature_columns(f)].isna().all()
    nxt = gap_day + pd.Timedelta(days=1)
    assert np.isnan(f.loc[nxt, "ret_1"])                       # needs the missing close
    assert np.isnan(f.loc[gap_day + pd.Timedelta(days=19), "sma_ratio_20"])
    assert not np.isnan(f.loc[gap_day + pd.Timedelta(days=20), "sma_ratio_20"])
    # The truncation check must also hold right around the gap.
    check_no_lookahead(inputs, [gap_day + pd.Timedelta(days=1), gap_day + pd.Timedelta(days=2)])


def test_funding_alignment_respects_publish_time() -> None:
    daily = synthetic_klines(T0, 60)
    day = T0 + 40 * DAY
    as_of = day + DAY
    funding = pd.Series([0.001, 0.002, 0.009], index=[as_of - timedelta(hours=8), as_of,
                                                     as_of + timedelta(milliseconds=1)])
    f = build_features(FeatureInputs(daily=daily, funding=funding))
    assert f.loc[day, "funding_last"] == pytest.approx(0.002)        # published exactly at as_of: ok
    assert f.loc[day + DAY, "funding_last"] == pytest.approx(0.009)  # the later print shows up next day
    assert np.isnan(f.loc[T0 + 10 * DAY, "funding_last"])            # no stale carry-forward


def test_targets_values_and_end_time() -> None:
    daily = synthetic_klines(T0, 50)
    t = build_targets(daily, [7, 14])
    c = daily["close"]
    assert t["fwd_logret_7"].iloc[10] == pytest.approx(np.log(c.iloc[17] / c.iloc[10]))
    assert t["fwd_logret_7"].iloc[-7:].isna().all() and t["fwd_logret_7"].iloc[-8:-7].notna().all()
    assert t["fwd_up_14"].iloc[-14:].isna().all()
    assert (t["fwd_up_7"].dropna() == (t["fwd_logret_7"].dropna() > 0)).all()
    assert t["target_end_time_7"].iloc[0] == T0 + 8 * DAY   # close of day t+7


def test_dataset_excludes_unrealized_targets(inputs: FeatureInputs) -> None:
    f = build_features(inputs)
    t = build_targets(inputs.daily, [7])
    row = f.index[300]
    now = t.loc[row, "target_end_time_7"] - timedelta(seconds=1)
    ds = make_dataset(f, t, 7, now=now)
    assert ds.index.max() < row
    assert (ds["target_end_time_7"] <= pd.Timestamp(now)).all()
    assert not [c for c in feature_columns(ds) if c.startswith(("fwd_", "target_"))]


def test_feature_storage_roundtrip(session: Session, inputs: FeatureInputs) -> None:
    sym = sync_symbols(session, ["SOLUSDT"], ["SOLUSDT"])[0]
    session.commit()
    f = build_features(inputs)
    n = save_features(session, sym.id, "1d", f, "test")
    session.commit()
    assert n == N
    assert save_features(session, sym.id, "1d", f, "test") == N  # upsert, no duplicates
    session.commit()
    back = load_features(session, "SOLUSDT", "1d", "test")
    assert len(back) == N
    cols = feature_columns(f)
    np.testing.assert_allclose(back[cols].to_numpy(), f[cols].to_numpy(dtype=float), equal_nan=True)
    pd.testing.assert_index_equal(back.index, f.index, check_names=False)
