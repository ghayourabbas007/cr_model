"""Persist and reload versioned feature frames (features table, one JSON vector per row)."""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecaster.data.store import upsert_rows
from forecaster.db.models import FeatureRow, Symbol
from forecaster.features.build import feature_columns

_KEY = ["symbol_id", "interval", "as_of_time", "feature_set_version"]


def _clean(v) -> float | None:
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) or math.isinf(f) else f


def save_features(session: Session, symbol_id: int, interval: str, frame: pd.DataFrame, version: str) -> int:
    """Upsert one row per as_of_time. NaN is stored as JSON null. `ref_close` is stored too."""
    cols = ["ref_close", *feature_columns(frame)]
    records = [
        {
            "symbol_id": symbol_id,
            "interval": interval,
            "as_of_time": row.as_of_time.to_pydatetime(),
            "feature_set_version": version,
            "feature_values": {c: _clean(getattr(row, c)) for c in cols},
        }
        for row in frame.reset_index(drop=True).itertuples(index=False)
        if not math.isnan(row.ref_close)  # rows for missing candles carry no information
    ]
    upsert_rows(session, FeatureRow.__table__, records, _KEY, ["feature_values"], touch=None)
    return len(records)


def load_features(session: Session, symbol: str, interval: str, version: str) -> pd.DataFrame:
    """Reload a stored feature frame indexed by open_time (as_of_time - 1 interval for 1d)."""
    q = (
        select(FeatureRow.as_of_time, FeatureRow.feature_values)
        .join(Symbol, Symbol.id == FeatureRow.symbol_id)
        .where(Symbol.symbol == symbol, FeatureRow.interval == interval,
               FeatureRow.feature_set_version == version)
        .order_by(FeatureRow.as_of_time)
    )
    rows = session.execute(q).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r.feature_values for r in rows], dtype="float64")
    df.insert(0, "as_of_time", pd.to_datetime([r.as_of_time for r in rows]))
    df.index = pd.DatetimeIndex(df["as_of_time"] - pd.Timedelta(days=1), name="open_time")
    return df
