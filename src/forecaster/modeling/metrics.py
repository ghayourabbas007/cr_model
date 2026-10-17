"""Evaluation metrics for probabilistic direction and quantile forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from forecaster.modeling.base import QUANTILES


def classification_metrics(y_up, prob_up) -> dict:
    y = np.asarray(y_up, dtype=int)
    p = np.clip(np.asarray(prob_up, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "base_rate_up": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def pinball(y, q, tau: float) -> float:
    err = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * err, (tau - 1) * err)))


def quantile_metrics(y_ret, q10, q50, q90) -> dict:
    y = np.asarray(y_ret, dtype=float)
    lo, mid, hi = (np.asarray(a, dtype=float) for a in (q10, q50, q90))
    return {
        "mae_q50": float(np.mean(np.abs(y - mid))),
        "mae_zero": float(np.mean(np.abs(y))),  # naive "no change" forecast
        "coverage_80": float(np.mean((y >= lo) & (y <= hi))),  # nominal 0.80
        "below_q10": float(np.mean(y < lo)),                   # nominal 0.10
        "above_q90": float(np.mean(y > hi)),                   # nominal 0.10
        "mean_width_80": float(np.mean(hi - lo)),
        **{f"pinball_q{int(t * 100)}": pinball(y, q, t) for t, q in zip(QUANTILES, (lo, mid, hi))},
    }


def calibration_table(y_up, prob_up, bins: int = 10) -> list[dict]:
    """Reliability diagram data: mean predicted vs observed frequency per probability bin."""
    df = pd.DataFrame({"y": np.asarray(y_up, dtype=float), "p": np.asarray(prob_up, dtype=float)})
    df["bin"] = np.minimum((df["p"] * bins).astype(int), bins - 1)
    g = df.groupby("bin").agg(mean_pred=("p", "mean"), frac_up=("y", "mean"), count=("y", "size"))
    return [{"bin": int(b), **{k: float(v) for k, v in row.items()}} for b, row in g.iterrows()]


def evaluate(oof: pd.DataFrame) -> dict:
    """Metrics for an out-of-fold prediction frame (prob_up, q10, q50, q90, y_up, y_ret)."""
    return {
        **classification_metrics(oof["y_up"], oof["prob_up"]),
        **quantile_metrics(oof["y_ret"], oof["q10"], oof["q50"], oof["q90"]),
        "calibration": calibration_table(oof["y_up"], oof["prob_up"]),
        "n_folds": int(oof["fold"].nunique()) if "fold" in oof else None,
    }


def compare_to_baseline(m: dict, base: dict) -> dict:
    """Skill scores vs a baseline (positive = better than baseline). Honest verdicts included."""
    bss = 1 - m["brier"] / base["brier"] if base["brier"] else None
    pss = 1 - m["pinball_q50"] / base["pinball_q50"] if base["pinball_q50"] else None
    auc = m.get("roc_auc")
    return {
        "brier_skill": bss,
        "pinball_q50_skill": pss,
        "accuracy_delta": m["accuracy"] - base["accuracy"],
        "coverage_error_80": m["coverage_80"] - 0.80,
        "beats_baseline_brier": bool(bss is not None and bss > 0),
        "beats_baseline_auc": bool(auc is not None and auc > 0.5 and auc > (base.get("roc_auc") or 0.5)),
    }
