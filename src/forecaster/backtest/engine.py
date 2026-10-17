"""Vectorised daily backtest of a rule-based strategy driven by model signals.

Timing: a signal stamped `as_of_time` t (the close of day t) sets the position held from close t
to close t+1, so it earns the return realized at t+1. Never the return at t.

Costs: (fee_bps + slippage_bps) per unit of position change, charged on the day the change
takes effect. Entering long from flat costs 1x; flipping long->short costs 2x. Open positions
are closed (and charged) at the end of the backtest, for both strategy and benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    enter_long: float = 0.55
    exit_long: float = 0.50
    allow_short: bool = False
    enter_short: float = 0.45
    exit_short: float = 0.50
    require_median_agrees: bool = False  # also require q50 > 0 for longs (q50 < 0 for shorts)

    def name(self) -> str:
        return "long_short" if self.allow_short else "long_flat"


def positions_from_signals(prob_up: pd.Series, q50: pd.Series | None = None,
                           p: StrategyParams | None = None) -> pd.Series:
    """Hysteresis state machine: enter above `enter_long`, leave below `exit_long`, etc.
    A missing signal keeps the current position."""
    p = p or StrategyParams()
    probs = prob_up.to_numpy(dtype=float)
    meds = q50.to_numpy(dtype=float) if q50 is not None else np.zeros(len(probs))
    out = np.zeros(len(probs))
    pos = 0.0
    for i, (pr, md) in enumerate(zip(probs, meds, strict=True)):
        if not np.isnan(pr):
            long_ok = pr >= p.enter_long and (not p.require_median_agrees or md > 0)
            short_ok = p.allow_short and pr <= p.enter_short and (not p.require_median_agrees or md < 0)
            if (pos > 0 and pr < p.exit_long) or (pos < 0 and pr > p.exit_short):
                pos = 0.0
            if pos == 0:
                pos = 1.0 if long_ok else (-1.0 if short_ok else 0.0)
        out[i] = pos
    return pd.Series(out, index=prob_up.index, name="position")


@dataclass
class BacktestOutput:
    returns: pd.Series            # strategy daily net returns
    benchmark_returns: pd.Series  # buy-and-hold daily net returns
    position: pd.Series           # position held during each day (earning that day's return)
    costs: pd.Series
    trades: list[dict] = field(default_factory=list)

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    @property
    def benchmark_equity(self) -> pd.Series:
        return (1 + self.benchmark_returns).cumprod()


def run_backtest(close: pd.Series, decisions: pd.Series, cost_bps: float) -> BacktestOutput:
    """close: daily closes indexed by close time (as_of). decisions: target position indexed by
    the as_of time it was decided. Positions are carried through exchange gaps."""
    close = close.dropna().sort_index()
    start = decisions.index.min()
    # The last decision earns the return to the next available close (which may follow a gap).
    later = close.index[close.index > decisions.index.max()]
    end = later[0] if len(later) else decisions.index.max()
    idx = close.index[(close.index >= start) & (close.index <= end)]
    if len(idx) < 2:
        raise ValueError("Not enough price data in the backtest window")

    # Across an exchange gap the full move is realized on the next available close.
    ret = close.pct_change().reindex(idx).fillna(0.0)
    ret.iloc[0] = 0.0  # day 0 is the first decision time; nothing is held yet
    decided = decisions.reindex(idx).ffill().fillna(0.0)
    held = decided.shift(1).fillna(0.0)      # decision at t applies to the return at t+1

    rate = cost_bps / 1e4
    turnover = held.diff().abs()
    turnover.iloc[0] = abs(held.iloc[0])
    costs = turnover * rate
    costs.iloc[-1] += abs(held.iloc[-1]) * rate  # close out at the end
    strat = held * ret - costs

    bh_costs = pd.Series(0.0, index=idx)
    bh_costs.iloc[1] += rate                     # buy at the first close, held from day 1
    bh_costs.iloc[-1] += rate                    # sell at the end
    bh_held = pd.Series(1.0, index=idx)
    bh_held.iloc[0] = 0.0
    bench = bh_held * ret - bh_costs

    return BacktestOutput(strat, bench, held, costs, _trades(held, ret, rate))


def _trades(held: pd.Series, ret: pd.Series, rate: float) -> list[dict]:
    """Contiguous runs of a non-zero position, with net return after entry and exit costs."""
    trades = []
    h = held.to_numpy()
    r = ret.to_numpy()
    i, n = 0, len(h)
    while i < n:
        if h[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and h[j + 1] == h[i]:
            j += 1
        gross = float(np.prod(1 + h[i] * r[i : j + 1]))
        net = gross * (1 - rate) * (1 - rate) - 1
        trades.append({"entry": held.index[i], "exit": held.index[j], "side": "long" if h[i] > 0 else "short",
                       "days": j - i + 1, "return": net})
        i = j + 1
    return trades
