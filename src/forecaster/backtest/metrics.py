"""Performance statistics. Crypto trades every day, so annualisation uses 365 periods."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 365


def sharpe(r: pd.Series | np.ndarray) -> float | None:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return None if np.isnan(sd) or sd < 1e-12 else float(r.mean() / sd * np.sqrt(PERIODS))


def sortino(r: pd.Series | np.ndarray) -> float | None:
    r = np.asarray(r, dtype=float)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    return None if downside < 1e-12 else float(r.mean() / downside * np.sqrt(PERIODS))


def max_drawdown(equity: pd.Series) -> float:
    e = equity.to_numpy(dtype=float)
    return float(np.min(e / np.maximum.accumulate(e) - 1))


def summarize(returns: pd.Series, position: pd.Series | None = None, trades: list[dict] | None = None,
              costs: pd.Series | None = None) -> dict:
    eq = (1 + returns).cumprod()
    days = max(len(returns) - 1, 1)
    out = {
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": float(eq.iloc[-1] ** (PERIODS / days) - 1) if eq.iloc[-1] > 0 else -1.0,
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(PERIODS)),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(eq),
        "days": int(len(returns)),
    }
    if position is not None:
        out["exposure"] = float((position != 0).mean())
    if trades is not None:
        rets = [t["return"] for t in trades]
        out["n_trades"] = len(trades)
        out["win_rate"] = float(np.mean([x > 0 for x in rets])) if rets else None
        out["avg_trade_return"] = float(np.mean(rets)) if rets else None
    if costs is not None:
        out["total_costs"] = float(costs.sum())
    return out


def bootstrap_sharpe_diff(strategy: pd.Series, benchmark: pd.Series, n: int = 2000, block: int = 20,
                          seed: int = 42) -> dict:
    """Paired circular block bootstrap of Sharpe(strategy) - Sharpe(benchmark).

    Blocks preserve autocorrelation (positions persist for days). p_value is the share of
    resamples where the strategy does not beat the benchmark; small = evidence of real edge.
    """
    a = strategy.to_numpy(dtype=float)
    b = benchmark.to_numpy(dtype=float)
    m = len(a)
    if m < 2 * block:
        return {"sharpe_diff": None, "p_value": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block))
    offsets = np.arange(block)
    diffs = np.empty(n)
    for k in range(n):
        starts = rng.integers(0, m, n_blocks)
        idx = ((starts[:, None] + offsets) % m).ravel()[:m]
        sa, sb = sharpe(a[idx]), sharpe(b[idx])
        diffs[k] = (sa or 0.0) - (sb or 0.0)
    point = (sharpe(a) or 0.0) - (sharpe(b) or 0.0)
    return {
        "sharpe_diff": float(point),
        "p_value": float(np.mean(diffs <= 0)),
        "ci_low": float(np.quantile(diffs, 0.05)),
        "ci_high": float(np.quantile(diffs, 0.95)),
    }


def verdict(strategy: dict, benchmark: dict, boot: dict, alpha: float = 0.05) -> tuple[bool, str]:
    """Risk-adjusted comparison after costs. Returns (beats_benchmark, human-readable verdict)."""
    s, b = strategy["sharpe"], benchmark["sharpe"]
    beats = s is not None and (b is None or s > b)
    if not beats:
        return False, "does NOT beat buy-and-hold (risk-adjusted, after costs)"
    p = boot.get("p_value")
    if p is not None and p < alpha:
        return True, f"beats buy-and-hold (Sharpe, after costs; bootstrap p={p:.3f})"
    return True, (f"higher Sharpe than buy-and-hold but NOT statistically significant "
                  f"(bootstrap p={p:.2f})" if p is not None else "higher Sharpe; too little data to test")
