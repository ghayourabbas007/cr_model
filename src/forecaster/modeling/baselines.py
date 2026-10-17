"""Baselines every real model must beat.

- climatology: unconditional training-window up-rate and empirical return quantiles.
  Equivalent to "no information beyond history"; its direction call is "always the majority
  class", which for a rising market is the buy-and-hold view.
- momentum: conditions the same statistics on the sign of the trailing h-day return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.modeling.base import QUANTILES, to_frame


class ClimatologyModel:
    name = "climatology"

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.features_: list[str] = []

    def fit(self, X, y_up, y_ret):
        self.p_ = float(np.mean(y_up))
        self.q_ = np.quantile(y_ret, QUANTILES)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return to_frame(X.index, np.full(n, self.p_), np.tile(self.q_, (n, 1)))


class MomentumModel:
    name = "momentum"

    def __init__(self, horizon: int):
        self.horizon = horizon
        # Closest trailing-return feature to the forecast horizon.
        self.signal = "ret_7" if horizon <= 10 else "ret_14"
        self.features_ = [self.signal]

    def fit(self, X, y_up, y_ret):
        sig = np.sign(X[self.signal].fillna(0.0).to_numpy())
        y_up, y_ret = np.asarray(y_up), np.asarray(y_ret)
        self.fallback_ = (float(y_up.mean()), np.quantile(y_ret, QUANTILES))
        self.table_ = {}
        for s in (-1.0, 0.0, 1.0):
            m = sig == s
            if m.sum() >= 30:
                self.table_[s] = (float(y_up[m].mean()), np.quantile(y_ret[m], QUANTILES))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        sig = np.sign(X[self.signal].fillna(0.0).to_numpy())
        p = np.empty(len(X))
        q = np.empty((len(X), 3))
        for i, s in enumerate(sig):
            p[i], q[i] = self.table_.get(s, self.fallback_)
        return to_frame(X.index, p, q)
