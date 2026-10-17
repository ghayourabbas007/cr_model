"""Walk-forward (rolling-origin) validation with purging. Never random splits.

For a test block starting at as_of time T, a training row is allowed only if its target was
fully realized by T (target_end_time <= T). Because h-day targets overlap, this purges the
last h - 1 rows before every test block, whose targets a naive time split would leak.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecaster.logging_setup import get_logger
from forecaster.modeling.base import ForecastModel

log = get_logger(__name__)


@dataclass
class Fold:
    number: int
    train_idx: np.ndarray  # positional
    test_idx: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(ds: pd.DataFrame, horizon: int, min_train: int = 365, test_size: int = 90,
                        step: int | None = None, max_train: int | None = None) -> list[Fold]:
    """Expanding window by default; rolling if max_train (rows) is given."""
    as_of = ds["as_of_time"].to_numpy()
    end = ds[f"target_end_time_{horizon}"].to_numpy()
    if not (np.diff(as_of.astype("int64")) > 0).all():
        raise ValueError("Dataset must be strictly ordered by as_of_time")
    n = len(ds)
    pos = np.arange(n)
    folds, start, k = [], min_train, 0
    while start < n:
        stop = min(start + test_size, n)
        t0 = as_of[start]
        mask = (pos < start) & (end <= t0)
        if max_train is not None:
            mask &= pos >= start - max_train
        train = pos[mask]
        if len(train) >= min(min_train, 100):
            folds.append(Fold(k, train, pos[start:stop], pd.Timestamp(t0), pd.Timestamp(as_of[stop - 1])))
            k += 1
        start += step or test_size
    return folds


ModelFactory = Callable[[], ForecastModel]


def run_walk_forward(ds: pd.DataFrame, horizon: int, factory: ModelFactory, folds: list[Fold],
                     history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit a fresh model per fold on its training rows only; return out-of-fold predictions."""
    y_up, y_ret = ds[f"fwd_up_{horizon}"], ds[f"fwd_logret_{horizon}"]
    parts = []
    for f in folds:
        model = factory()
        if history is not None and hasattr(model, "set_history"):
            model.set_history(history)
        tr, te = ds.iloc[f.train_idx], ds.iloc[f.test_idx]
        model.fit(tr, y_up.iloc[f.train_idx], y_ret.iloc[f.train_idx])
        pred = model.predict(te)
        pred["y_up"] = y_up.iloc[f.test_idx].to_numpy()
        pred["y_ret"] = y_ret.iloc[f.test_idx].to_numpy()
        pred["as_of_time"] = te["as_of_time"].to_numpy()
        pred["ref_close"] = te["ref_close"].to_numpy()
        pred["fold"] = f.number
        parts.append(pred)
        log.debug("fold done", extra={"model": getattr(model, "name", "?"), "fold": f.number,
                                      "train": len(f.train_idx), "test": len(f.test_idx)})
    if not parts:
        raise ValueError("No walk-forward folds: not enough history for min_train")
    return pd.concat(parts)
