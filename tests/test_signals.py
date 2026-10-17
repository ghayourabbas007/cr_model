from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from forecaster.signals import (
    FLAT,
    LONG,
    SHORT,
    SignalRules,
    make_signal,
    signal_history,
    summarize_history,
)

T = datetime(2026, 10, 17)


def fc(prob_up: float, q10: float, q50: float, q90: float, ref: float = 100.0, h: int = 7, **kw) -> dict:
    return {"symbol": "SOLUSDT", "horizon_days": h, "as_of_time": T, "reference_price": ref, "prob_up": prob_up,
            "price_q10": q10, "price_q50": q50, "price_q90": q90, **kw}


def test_long_plan_levels_and_rr() -> None:
    sig = make_signal(fc(0.62, 90, 104, 115), atr=4.0)
    assert sig.side == LONG and sig.strength == "moderate"
    assert sig.entry == 100 and (sig.entry_low, sig.entry_high) == (99.0, 101.0)
    assert sig.stop_loss == 90           # q10 is further than 1 ATR
    assert (sig.tp1, sig.tp2) == (104, 115)
    assert sig.rr_tp1 == pytest.approx(0.4) and sig.rr_tp2 == pytest.approx(1.5)
    assert sig.valid_until == T + timedelta(days=7)
    assert any("Stop" in x for x in sig.invalidation) and any("Time" in x for x in sig.invalidation)


def test_stop_is_never_tighter_than_one_atr() -> None:
    sig = make_signal(fc(0.6, 99, 103, 110), atr=5.0)
    assert sig.side == LONG and sig.stop_loss == 95


def test_short_plan_mirrors_levels() -> None:
    sig = make_signal(fc(0.38, 85, 96, 108), atr=3.0)
    assert sig.side == SHORT
    assert sig.stop_loss == 108 and (sig.tp1, sig.tp2) == (96, 85)
    assert sig.rr_tp2 == pytest.approx(15 / 8)


@pytest.mark.parametrize(("forecast", "rules", "reason"), [
    (fc(0.50, 90, 101, 110), SignalRules(), "no-trade band"),
    (fc(0.60, 90, 99, 110), SignalRules(), "median forecast is not above"),
    (fc(0.40, 90, 98, 110), SignalRules(allow_short=False), "shorts are disabled"),
    (fc(0.60, 80, 102, 104), SignalRules(), "risk/reward"),
])
def test_no_trade_cases(forecast: dict, rules: SignalRules, reason: str) -> None:
    sig = make_signal(forecast, atr=2.0, rules=rules)
    assert sig.side == FLAT and reason in sig.reason and sig.entry is None


def test_stale_data_is_an_invalidation() -> None:
    assert "stale" in make_signal(fc(0.6, 90, 104, 115, stale=True), atr=4.0).invalidation[0]


# --- historical replay -------------------------------------------------------------------------

def _candles(rows: list[tuple[float, float, float]], start: datetime) -> pd.DataFrame:
    """rows of (high, low, close); flat ATR warm-up of 20 days is prepended."""
    warm = [(101.0, 99.0, 100.0)] * 20
    data = warm + rows
    idx = pd.date_range(start - timedelta(days=20), periods=len(data), freq="1D")
    return pd.DataFrame({"open": [c for _, _, c in data], "high": [h for h, _, _ in data],
                         "low": [lo for _, lo, _ in data], "close": [c for _, _, c in data]}, index=idx)


def _oos(as_of: datetime, prob_up: float, q10=90.0, q50=104.0, q90=115.0) -> pd.DataFrame:
    return pd.DataFrame({"prob_up": [prob_up], "reference_price": [100.0], "price_q10": [q10],
                         "price_q50": [q50], "price_q90": [q90]}, index=pd.DatetimeIndex([as_of]))


def test_history_target_hit_uses_candles_after_signal() -> None:
    start = datetime(2024, 1, 1)
    # Candle opening at `start` closed at 100 -> signal as_of = start + 1 day.
    candles = _candles([(101, 99, 100), (103, 99, 102), (105, 101, 104), (104, 100, 103)], start)
    as_of = pd.Timestamp(start + timedelta(days=1))
    h = signal_history(_oos(as_of, 0.6), candles, horizon=3, rules=SignalRules(cost_bps=0))
    assert len(h) == 1 and h.iloc[0]["outcome"] == "target"
    assert h.iloc[0]["r"] == pytest.approx(4 / 10)  # +4 to TP1, risk 10 (stop at q10 = 90)


def test_history_stop_first_when_both_touched_and_costs() -> None:
    start = datetime(2024, 1, 1)
    candles = _candles([(101, 99, 100), (106, 89, 100), (100, 99, 100), (100, 99, 100)], start)
    h = signal_history(_oos(pd.Timestamp(start + timedelta(days=1)), 0.6), candles, 3,
                       SignalRules(cost_bps=10))
    assert h.iloc[0]["outcome"] == "stop"
    assert h.iloc[0]["r"] == pytest.approx((-10 - 0.2) / 10)  # -1R minus 2 x 10 bps of 100


def test_history_expiry_and_unfinished_and_flat() -> None:
    start = datetime(2024, 1, 1)
    candles = _candles([(101, 99, 100), (102, 98, 101), (102, 98, 101.5), (102, 98, 102)], start)
    a1 = pd.Timestamp(start + timedelta(days=1))
    expired = signal_history(_oos(a1, 0.6), candles, 3, SignalRules(cost_bps=0))
    assert expired.iloc[0]["outcome"] == "expired" and expired.iloc[0]["r"] == pytest.approx(0.2)
    assert signal_history(_oos(a1, 0.6), candles, 5).empty     # path not finished yet
    assert signal_history(_oos(a1, 0.5), candles, 3).empty     # no-trade band


def test_summary() -> None:
    h = pd.DataFrame({"side": [LONG, SHORT, LONG], "outcome": ["target", "stop", "expired"], "r": [0.5, -1.0, 0.1]})
    s = summarize_history(h)
    assert s["signals"] == 3 and s["win_rate"] == pytest.approx(2 / 3)
    assert s["avg_r"] == pytest.approx(-0.4 / 3) and s["stopped"] == pytest.approx(1 / 3)
    assert summarize_history(pd.DataFrame()) == {"signals": 0}
