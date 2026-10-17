"""Live forecasts from the active model, and prediction persistence."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecaster import DISCLAIMER
from forecaster.config import Settings
from forecaster.data.store import load_klines, upsert_rows
from forecaster.db.models import ModelRun, Prediction
from forecaster.db.repository import get_symbol
from forecaster.features.build import build_features
from forecaster.features.pipeline import load_inputs
from forecaster.modeling.registry import active_run, load_artifact
from forecaster.timeutil import utcnow

_KEY = ["model_run_id", "as_of_time", "horizon_days", "kind"]
_UPDATE = ["reference_price", "prob_up", "return_q10", "return_q50", "return_q90",
           "price_q10", "price_q50", "price_q90", "realized_return", "symbol_id"]
STALE_AFTER = timedelta(days=2)


class NoActiveModel(LookupError):
    pass


def _f(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def save_predictions(session: Session, run_id: int, symbol_id: int, horizon: int, frame: pd.DataFrame,
                     kind: str) -> int:
    """frame: prob_up, q10, q50, q90, ref_close, as_of_time and optionally y_ret (realized)."""
    records = []
    for r in frame.itertuples(index=False):
        ref = float(r.ref_close)
        records.append({
            "model_run_id": run_id, "symbol_id": symbol_id, "horizon_days": horizon, "kind": kind,
            "as_of_time": pd.Timestamp(r.as_of_time).to_pydatetime(), "reference_price": ref,
            "prob_up": _f(r.prob_up), "return_q10": _f(r.q10), "return_q50": _f(r.q50), "return_q90": _f(r.q90),
            "price_q10": ref * math.exp(r.q10), "price_q50": ref * math.exp(r.q50),
            "price_q90": ref * math.exp(r.q90), "realized_return": _f(getattr(r, "y_ret", None)),
        })
    upsert_rows(session, Prediction.__table__, records, _KEY, _UPDATE, touch=None)
    return len(records)


def forecast(session: Session, s: Settings, symbol: str, horizon: int, model_name: str | None = None,
             now: datetime | None = None, store: bool = True) -> dict:
    """Forecast from the latest closed daily candle using the active model."""
    name = model_name or s.serving_model
    now = now or utcnow()
    sym = get_symbol(session, symbol)
    if sym is None:
        raise LookupError(f"Unknown symbol {symbol}")
    run = active_run(session, sym.id, horizon, name)
    if run is None:
        raise NoActiveModel(f"No active {name} model for {symbol} h={horizon}; run `forecaster train`")

    model = load_artifact(run.artifact_path)
    feats = build_features(load_inputs(session, s, symbol))
    avail = feats[(feats["as_of_time"] <= pd.Timestamp(now)) & feats["ref_close"].notna()]
    if avail.empty:
        raise LookupError(f"No closed candles for {symbol} before {now}")
    if hasattr(model, "set_history"):
        model.set_history(feats)
    row = avail.iloc[[-1]]
    p = model.predict(row)
    as_of = row["as_of_time"].iloc[0].to_pydatetime()
    ref = float(row["ref_close"].iloc[0])
    q = {k: float(p[k].iloc[0]) for k in ("q10", "q50", "q90")}

    if store:
        frame = p.assign(ref_close=ref, as_of_time=as_of)
        save_predictions(session, run.id, sym.id, horizon, frame, "live")
        session.commit()

    return {
        "symbol": symbol, "horizon_days": horizon, "model": name, "model_version": run.version,
        "model_run_id": run.id, "as_of_time": as_of, "target_time": as_of + timedelta(days=horizon),
        "reference_price": ref, "prob_up": float(p["prob_up"].iloc[0]),
        "return_q10": q["q10"], "return_q50": q["q50"], "return_q90": q["q90"],
        "price_q10": ref * math.exp(q["q10"]), "price_q50": ref * math.exp(q["q50"]),
        "price_q90": ref * math.exp(q["q90"]),
        "interval_coverage_nominal": 0.80,
        "interval_coverage_backtest": (run.metrics or {}).get("coverage_80"),
        "stale": (now - as_of) > STALE_AFTER,
        "is_estimate": True,
        "disclaimer": DISCLAIMER,
    }


def update_realized(session: Session, symbol: str) -> int:
    """Fill realized_return for past live predictions whose horizon has elapsed."""
    sym = get_symbol(session, symbol)
    if sym is None:
        return 0
    pending = session.scalars(
        select(Prediction).where(Prediction.symbol_id == sym.id, Prediction.kind == "live",
                                 Prediction.realized_return.is_(None))
    ).all()
    if not pending:
        return 0
    closes = load_klines(session, symbol, "1d")["close"]
    n = 0
    for pr in pending:
        # The close at as_of + h is the close of the daily candle opening at as_of + h - 1 day.
        key = pd.Timestamp(pr.as_of_time + timedelta(days=pr.horizon_days - 1))
        if key in closes.index and pr.reference_price:
            pr.realized_return = float(np.log(closes[key] / pr.reference_price))
            n += 1
    session.commit()
    return n


def list_runs(session: Session, symbol_id: int | None = None) -> list[ModelRun]:
    q = select(ModelRun).order_by(ModelRun.created_at.desc(), ModelRun.id.desc())
    if symbol_id is not None:
        q = q.where(ModelRun.symbol_id == symbol_id)
    return list(session.scalars(q))
