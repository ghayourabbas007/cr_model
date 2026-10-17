"""Rule-based trade plans derived from a probabilistic forecast. Research output, not advice.

Direction comes from P(up) with a neutral band, and the median forecast must agree:
    LONG  if P(up) >= enter_long  and median price > entry
    SHORT if P(up) <= enter_short and median price < entry   (only if allow_short)
    otherwise NO TRADE.
Levels for a LONG (SHORT mirrors them):
    entry     = latest daily close (the forecast's reference price), entry zone = entry ± k·ATR
    stop loss = lower of the 10th-percentile price and entry − m·ATR (never tighter than m ATRs)
    TP1 / TP2 = median / 90th-percentile price
A plan whose risk/reward to TP2 is below `min_rr` is downgraded to NO TRADE.

`signal_history` replays the same rules over walk-forward out-of-sample predictions with daily
high/low paths, after costs, to show how such plans would have fared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

LONG, SHORT, FLAT = "LONG", "SHORT", "NO TRADE"


@dataclass(frozen=True)
class SignalRules:
    enter_long: float = 0.55
    enter_short: float = 0.45
    allow_short: bool = True
    sl_atr_mult: float = 1.0
    entry_zone_atr: float = 0.25
    min_rr: float = 1.0
    cost_bps: float = 15.0  # per side: fee + slippage


@dataclass
class TradeSignal:
    symbol: str
    horizon_days: int
    as_of_time: datetime
    valid_until: datetime
    side: str
    reason: str
    prob_up: float
    reference_price: float
    atr: float
    entry: float | None = None
    entry_low: float | None = None
    entry_high: float | None = None
    stop_loss: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    rr_tp1: float | None = None
    rr_tp2: float | None = None
    strength: str = "none"
    invalidation: list[str] = field(default_factory=list)

    def pct(self, level: float | None) -> float | None:
        return None if level is None else level / self.reference_price - 1


def _strength(prob_up: float) -> str:
    edge = abs(prob_up - 0.5)
    return "weak" if edge < 0.08 else ("moderate" if edge < 0.15 else "strong")


def levels(side: str, entry: float, atr: float, q10: float, q50: float, q90: float,
           rules: SignalRules) -> tuple[float, float, float]:
    """(stop_loss, tp1, tp2) for a side."""
    if side == LONG:
        return min(q10, entry - rules.sl_atr_mult * atr), q50, q90
    return max(q90, entry + rules.sl_atr_mult * atr), q50, q10


def _rr(side: str, entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    reward = (tp - entry) if side == LONG else (entry - tp)
    return reward / risk if risk > 0 else float("nan")


def decide_side(prob_up: float, ref: float, q50: float, rules: SignalRules) -> tuple[str, str]:
    if prob_up >= rules.enter_long:
        if q50 > ref:
            return LONG, f"P(up) {prob_up:.0%} ≥ {rules.enter_long:.0%} and median above entry"
        return FLAT, f"P(up) {prob_up:.0%} favours up but the median forecast is not above entry"
    if prob_up <= rules.enter_short:
        if not rules.allow_short:
            return FLAT, f"P(up) {prob_up:.0%} favours down, but shorts are disabled"
        if q50 < ref:
            return SHORT, f"P(up) {prob_up:.0%} ≤ {rules.enter_short:.0%} and median below entry"
        return FLAT, f"P(up) {prob_up:.0%} favours down but the median forecast is not below entry"
    return FLAT, (f"P(up) {prob_up:.0%} is inside the no-trade band "
                  f"{rules.enter_short:.0%}–{rules.enter_long:.0%}: no directional edge")


def make_signal(forecast: dict, atr: float, rules: SignalRules | None = None) -> TradeSignal:
    """Trade plan from one horizon's forecast (as returned by modeling.predict.forecast)."""
    rules = rules or SignalRules()
    ref, p = forecast["reference_price"], forecast["prob_up"]
    q10, q50, q90 = forecast["price_q10"], forecast["price_q50"], forecast["price_q90"]
    as_of = forecast["as_of_time"]
    sig = TradeSignal(symbol=forecast["symbol"], horizon_days=forecast["horizon_days"], as_of_time=as_of,
                      valid_until=as_of + timedelta(days=forecast["horizon_days"]), side=FLAT, reason="",
                      prob_up=p, reference_price=ref, atr=atr, strength=_strength(p))
    side, reason = decide_side(p, ref, q50, rules)
    sig.reason = reason
    if side == FLAT:
        sig.invalidation = [f"Re-check after the next daily close; a signal needs P(up) ≥ {rules.enter_long:.0%}"
                            + (f" or ≤ {rules.enter_short:.0%}." if rules.allow_short else ".")]
        return sig

    sl, tp1, tp2 = levels(side, ref, atr, q10, q50, q90, rules)
    rr1, rr2 = _rr(side, ref, sl, tp1), _rr(side, ref, sl, tp2)
    if not rr2 >= rules.min_rr:
        sig.reason = f"{side} bias, but risk/reward to TP2 is {rr2:.2f} < {rules.min_rr:.2f}"
        return sig

    zone = rules.entry_zone_atr * atr
    sig.side, sig.entry, sig.entry_low, sig.entry_high = side, ref, ref - zone, ref + zone
    sig.stop_loss, sig.tp1, sig.tp2, sig.rr_tp1, sig.rr_tp2 = sl, tp1, tp2, rr1, rr2
    below, above = ("below", "above") if side == LONG else ("above", "below")
    sig.invalidation = [
        f"Stop: price trades {below} {sl:,.4f}; the thesis is wrong, exit.",
        f"Time: the forecast horizon ends {sig.valid_until:%Y-%m-%d %H:%M} UTC; exit at market if no target is hit.",
        f"Flip: a later daily forecast moves P(up) {'below' if side == LONG else 'above'} 50%; exit.",
        f"Entry: skip if price is outside {sig.entry_low:,.4f}–{sig.entry_high:,.4f} or already "
        f"{above} TP1 ({tp1:,.4f}) before you can enter.",
    ]
    if forecast.get("stale"):
        sig.invalidation.insert(0, "Data is stale (latest candle more than 2 days old): do not act on it.")
    return sig


def atr_by_close_time(candles: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR keyed by the candle's close time (= the as_of time it becomes known)."""
    from forecaster.features.indicators import atr

    a = atr(candles["high"], candles["low"], candles["close"], n)
    return pd.Series(a.to_numpy(), index=candles.index + pd.Timedelta(days=1), name="atr")


def signal_history(oos: pd.DataFrame, candles: pd.DataFrame, horizon: int, rules: SignalRules | None = None,
                   tp: str = "tp1") -> pd.DataFrame:
    """Replay the signal rules over out-of-sample predictions.

    oos: indexed by as_of_time with prob_up, reference_price, price_q10/q50/q90.
    candles: daily OHLC indexed by open_time. A signal at as_of A is entered at the close at A and
    managed over the next `horizon` daily candles (open_time A … A + h − 1 days). If a candle's range
    touches both stop and target, the stop is assumed first (conservative). Unfinished signals are
    skipped. R multiples are after round-trip costs.
    """
    rules = rules or SignalRules()
    atr = atr_by_close_time(candles)
    rows = []
    hi, lo, cl = candles["high"], candles["low"], candles["close"]
    for as_of, r in oos.iterrows():
        a = atr.get(as_of, np.nan)
        if np.isnan(a):
            continue
        ref = r["reference_price"]
        side, _ = decide_side(r["prob_up"], ref, r["price_q50"], rules)
        if side == FLAT:
            continue
        sl, tp1, tp2 = levels(side, ref, a, r["price_q10"], r["price_q50"], r["price_q90"], rules)
        target = tp1 if tp == "tp1" else tp2
        if not _rr(side, ref, sl, tp2) >= rules.min_rr:
            continue
        path = candles.index[(candles.index >= as_of) & (candles.index < as_of + pd.Timedelta(days=horizon))]
        if len(path) < horizon:
            continue  # not finished yet (or a data gap)
        sign = 1 if side == LONG else -1
        exit_price, outcome = cl[path[-1]], "expired"
        for t in path:
            hit_sl = lo[t] <= sl if side == LONG else hi[t] >= sl
            hit_tp = hi[t] >= target if side == LONG else lo[t] <= target
            if hit_sl:
                exit_price, outcome = sl, "stop"
                break
            if hit_tp:
                exit_price, outcome = target, "target"
                break
        risk = abs(ref - sl)
        cost = 2 * rules.cost_bps / 1e4 * ref
        r_mult = (sign * (exit_price - ref) - cost) / risk
        rows.append({"as_of_time": as_of, "side": side, "entry": ref, "stop_loss": sl, "target": target,
                     "exit": exit_price, "outcome": outcome, "r": r_mult})
    return pd.DataFrame(rows, columns=["as_of_time", "side", "entry", "stop_loss", "target", "exit",
                                       "outcome", "r"])


def summarize_history(h: pd.DataFrame) -> dict:
    if h.empty:
        return {"signals": 0}
    return {
        "signals": len(h),
        "longs": int((h["side"] == LONG).sum()),
        "shorts": int((h["side"] == SHORT).sum()),
        "win_rate": float((h["r"] > 0).mean()),
        "target_hit": float((h["outcome"] == "target").mean()),
        "stopped": float((h["outcome"] == "stop").mean()),
        "expired": float((h["outcome"] == "expired").mean()),
        "avg_r": float(h["r"].mean()),
        "total_r": float(h["r"].sum()),
    }


def rules_from_settings(s) -> SignalRules:
    return SignalRules(enter_long=s.bt_enter_long, enter_short=s.bt_enter_short, allow_short=s.signal_allow_short,
                       sl_atr_mult=s.signal_sl_atr_mult, entry_zone_atr=s.signal_entry_zone_atr,
                       min_rr=s.signal_min_rr, cost_bps=s.fee_bps + s.slippage_bps)
