"""Data access for the dashboard. No Streamlit imports here, so it is unit-testable."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecaster.backtest.runner import latest_backtests
from forecaster.config import Settings
from forecaster.data.store import load_klines
from forecaster.db.models import ModelRun, Prediction
from forecaster.db.repository import get_symbol
from forecaster.features import indicators as ind
from forecaster.modeling.predict import NoActiveModel, forecast
from forecaster.modeling.registry import active_run
from forecaster.modeling.train import model_verdict


def candles_with_indicators(session: Session, symbol: str, days: int = 365) -> pd.DataFrame:
    """Daily candles plus chart indicators. Indicators are computed on the full history and then
    trimmed, so long windows (SMA 200) are correct at the left edge of the chart."""
    k = load_klines(session, symbol, "1d")
    if k.empty:
        return k
    c = k["close"]
    out = k[["open", "high", "low", "close", "volume"]].copy()
    for w in (20, 50, 200):
        out[f"sma_{w}"] = ind.sma(c, w)
    mid = ind.sma(c, 20)
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    out["bb_upper"], out["bb_lower"] = mid + 2 * sd, mid - 2 * sd
    out["rsi_14"] = ind.rsi(c, 14)
    return out.iloc[-days:]


def current_forecasts(session: Session, s: Settings, symbol: str, model: str,
                      store: bool = True) -> tuple[list[dict], str | None]:
    """Forecasts for every configured horizon, or ([], reason) if none can be produced."""
    out = []
    for h in s.forecast_horizons:
        try:
            out.append(forecast(session, s, symbol, h, model, store=store))
        except NoActiveModel as exc:
            return [], str(exc)
    return out, None


def model_table(session: Session, symbol: str) -> pd.DataFrame:
    sym = get_symbol(session, symbol)
    if sym is None:
        return pd.DataFrame()
    runs = session.scalars(
        select(ModelRun).where(ModelRun.symbol_id == sym.id, ModelRun.is_active.is_(True))
        .order_by(ModelRun.horizon_days, ModelRun.model_name)
    ).all()
    rows = []
    for r in runs:
        m = r.metrics or {}
        cmp = m.get("vs_climatology") or {}
        rows.append({
            "horizon": r.horizon_days, "model": r.model_name, "version": r.version,
            "accuracy": m.get("accuracy"), "roc_auc": m.get("roc_auc"), "brier": m.get("brier"),
            "brier_skill": cmp.get("brier_skill"), "mae_q50": m.get("mae_q50"),
            "coverage_80": m.get("coverage_80"), "always_up_acc": m.get("always_up_accuracy"),
            "oos_rows": m.get("n"), "verdict": model_verdict(r.model_name, m), "run_id": r.id,
        })
    return pd.DataFrame(rows)


def oos_predictions(session: Session, symbol: str, horizon: int, model: str) -> pd.DataFrame:
    """Walk-forward out-of-sample predictions of the active run (for calibration/coverage charts)."""
    sym = get_symbol(session, symbol)
    run = active_run(session, sym.id, horizon, model) if sym else None
    if run is None:
        return pd.DataFrame()
    rows = session.execute(
        select(Prediction.as_of_time, Prediction.reference_price, Prediction.prob_up, Prediction.return_q10,
               Prediction.return_q50, Prediction.return_q90, Prediction.price_q10, Prediction.price_q50,
               Prediction.price_q90,
               Prediction.realized_return)
        .where(Prediction.model_run_id == run.id, Prediction.kind == "oos")
        .order_by(Prediction.as_of_time)
    ).all()
    df = pd.DataFrame(rows, columns=["as_of_time", "reference_price", "prob_up", "q10", "q50", "q90",
                                     "price_q10", "price_q50", "price_q90", "realized_return"])
    df["up"] = (df["realized_return"] > 0).astype(float)
    df["inside"] = ((df["realized_return"] >= df["q10"]) & (df["realized_return"] <= df["q90"])).astype(float)
    return df.set_index("as_of_time")


def reliability(oos: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Mean predicted P(up) vs observed up-frequency per probability bin."""
    if oos.empty:
        return pd.DataFrame(columns=["bin", "mean_pred", "frac_up", "count"])
    b = (oos["prob_up"] * bins).astype(int).clip(upper=bins - 1)
    g = oos.groupby(b).agg(mean_pred=("prob_up", "mean"), frac_up=("up", "mean"), count=("up", "size"))
    return g.reset_index(names="bin")


def rolling_coverage(oos: pd.DataFrame, window: int = 90) -> pd.Series:
    """Share of realized returns inside the predicted 80% range over a trailing window."""
    return oos["inside"].rolling(window, min_periods=window // 2).mean().rename("coverage_80")


def backtests(session: Session, symbol: str) -> list[dict]:
    sym = get_symbol(session, symbol)
    if sym is None:
        return []
    out = []
    for r in sorted(latest_backtests(session, sym.id), key=lambda r: (r.horizon_days, r.strategy)):
        curve = pd.DataFrame(r.equity_curve or [])
        if not curve.empty:
            curve["t"] = pd.to_datetime(curve["t"])
            curve = curve.set_index("t")
        p = r.params or {}
        out.append({"row": r, "curve": curve, "verdict": p.get("verdict", ""),
                    "p_value": (p.get("bootstrap") or {}).get("p_value"),
                    "strategy_stats": p.get("strategy_stats") or {}, "benchmark_stats": p.get("benchmark_stats") or {}})
    return out
