"""Common model interface.

Every model outputs, per row: prob_up (probability the h-day forward log return is > 0) and
q10 / q50 / q90 (quantiles of the h-day forward log return). These are estimates only.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
PRED_COLUMNS = ["prob_up", "q10", "q50", "q90"]


class ForecastModel(Protocol):
    name: str
    features_: list[str]

    def fit(self, X: pd.DataFrame, y_up: pd.Series, y_ret: pd.Series) -> ForecastModel: ...

    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...


def sort_quantiles(q: np.ndarray) -> np.ndarray:
    """Enforce q10 <= q50 <= q90 row-wise (independently fitted quantile models can cross)."""
    return np.sort(q, axis=1)


def to_frame(index: pd.Index, prob_up: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    q = sort_quantiles(np.asarray(q, dtype=float))
    return pd.DataFrame({"prob_up": np.clip(prob_up, 0.0, 1.0), "q10": q[:, 0], "q50": q[:, 1],
                         "q90": q[:, 2]}, index=index)
