"""Gradient-boosted trees: a direction classifier plus one quantile regressor per quantile.

Backends: LightGBM (default) or XGBoost. Hyperparameters are deliberately conservative:
daily crypto data is small (a few thousand rows) and very noisy, so shallow trees, strong
regularisation and slow learning rates generalise better than tuned-to-death models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from forecaster.features.build import feature_columns
from forecaster.modeling.base import QUANTILES, to_frame

LGBM_DEFAULTS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4,
                     min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                     reg_lambda=1.0, verbose=-1)
XGB_DEFAULTS = dict(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=10,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", verbosity=0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class GBMModel:
    def __init__(self, horizon: int, backend: str = "lightgbm", params: dict | None = None,
                 calibrate: bool = True, calibration_frac: float = 0.2, seed: int = 42):
        if backend not in ("lightgbm", "xgboost"):
            raise ValueError(f"Unknown backend {backend!r}")
        self.horizon = horizon
        self.backend = backend
        self.name = "lgbm" if backend == "lightgbm" else "xgb"
        self.params = {**(LGBM_DEFAULTS if backend == "lightgbm" else XGB_DEFAULTS), **(params or {})}
        self.calibrate = calibrate
        self.calibration_frac = calibration_frac
        self.seed = seed

    def _classifier(self):
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(objective="binary", random_state=self.seed, **self.params)
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             random_state=self.seed, **self.params)

    def _quantile_regressor(self, alpha: float):
        if self.backend == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(objective="quantile", alpha=alpha, random_state=self.seed, **self.params)
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                            random_state=self.seed, **self.params)

    def fit(self, X: pd.DataFrame, y_up: pd.Series, y_ret: pd.Series) -> GBMModel:
        self.features_ = feature_columns(X)
        Xf = X[self.features_].astype(float)
        y_up = np.asarray(y_up, dtype=int)
        y_ret = np.asarray(y_ret, dtype=float)
        n = len(Xf)

        # Time-ordered calibration: fit the classifier on the earlier part, the Platt calibrator
        # on the most recent part, with a horizon-length purge so their targets do not overlap.
        n_cal = int(n * self.calibration_frac)
        self.calibrator_ = None
        if self.calibrate and n_cal >= 100 and n - n_cal - self.horizon >= 200:
            cut = n - n_cal
            fit_end = cut - self.horizon
            self.clf_ = self._classifier().fit(Xf.iloc[:fit_end], y_up[:fit_end])
            raw = self.clf_.predict_proba(Xf.iloc[cut:])[:, 1]
            if len(np.unique(y_up[cut:])) == 2:
                self.calibrator_ = LogisticRegression(C=1.0).fit(_logit(raw).reshape(-1, 1), y_up[cut:])
        else:
            self.clf_ = self._classifier().fit(Xf, y_up)

        self.qr_ = [self._quantile_regressor(a).fit(Xf, y_ret) for a in QUANTILES]
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X.reindex(columns=self.features_).astype(float)
        p = self.clf_.predict_proba(Xf)[:, 1]
        if self.calibrator_ is not None:
            p = self.calibrator_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        q = np.column_stack([m.predict(Xf) for m in self.qr_])
        return to_frame(X.index, p, q)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.clf_.feature_importances_, index=self.features_).sort_values(ascending=False)
