from __future__ import annotations

import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecaster.config import Settings
from forecaster.data.store import upsert_klines
from forecaster.db.models import ModelRun, Prediction
from forecaster.db.repository import sync_symbols
from forecaster.modeling.baselines import ClimatologyModel, MomentumModel
from forecaster.modeling.metrics import (
    calibration_table,
    classification_metrics,
    compare_to_baseline,
    evaluate,
    pinball,
    quantile_metrics,
)
from forecaster.modeling.predict import forecast
from forecaster.modeling.registry import activate, load_artifact, record_run, save_artifact
from forecaster.modeling.train import train_symbol
from forecaster.modeling.walkforward import run_walk_forward, walk_forward_splits
from tests.fakes import synthetic_klines

T0 = datetime(2021, 1, 1)
H = 7


def signal_dataset(n: int = 900, horizon: int = H, strength: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Features f0..f4 plus ret_7/ret_14; the forward return depends on f0 only (if strength > 0)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(T0, periods=n, freq="1D", name="open_time")
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)], index=idx)
    X["ret_7"] = rng.normal(0, 0.1, n)
    X["ret_14"] = rng.normal(0, 0.1, n)
    y = 0.03 * strength * X["f0"] + rng.normal(0, 0.03, n)
    ds = X.copy()
    ds.insert(0, "as_of_time", idx + pd.Timedelta(days=1))
    ds.insert(1, "ref_close", 100.0)
    ds[f"fwd_logret_{horizon}"] = y
    ds[f"fwd_up_{horizon}"] = (y > 0).astype(float)
    ds[f"target_end_time_{horizon}"] = idx + pd.Timedelta(days=1 + horizon)
    return ds


# --- walk-forward ---------------------------------------------------------------------------

def test_splits_are_purged_ordered_and_non_overlapping() -> None:
    ds = signal_dataset(800)
    folds = walk_forward_splits(ds, H, min_train=365, test_size=90)
    assert len(folds) == 5
    end = ds[f"target_end_time_{H}"].to_numpy()
    seen = set()
    for f in folds:
        test_start = ds["as_of_time"].iloc[f.test_idx[0]]
        assert f.train_idx.max() < f.test_idx.min()
        assert end[f.train_idx].max() <= test_start.to_datetime64()          # purged
        # Row t is allowed iff its target is known by the test start: t + h + 1 <= start + 1, so the
        # h - 1 rows immediately before the test block are purged.
        assert f.test_idx.min() - f.train_idx.max() - 1 == H - 1
        assert not seen & set(f.test_idx)
        seen |= set(f.test_idx)
    assert folds[0].test_idx[0] == 365 and max(seen) == len(ds) - 1


def test_rolling_window_limits_train_size() -> None:
    folds = walk_forward_splits(signal_dataset(900), H, min_train=365, test_size=90, max_train=300)
    assert all(len(f.train_idx) <= 300 for f in folds)


def test_splits_require_time_order() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(signal_dataset(500).iloc[::-1], H)


class SpyModel:
    name = "spy"

    def __init__(self, log):
        self.log = log
        self.features_ = []

    def fit(self, X, y_up, y_ret):
        self.train_index = X.index
        return self

    def predict(self, X):
        self.log.append((self.train_index, X.index))
        return ClimatologyModel(H).fit(X, np.ones(len(X)) * 0.5, np.zeros(len(X))).predict(X)


def test_walk_forward_never_trains_on_test_or_overlapping_rows() -> None:
    ds = signal_dataset(700)
    folds = walk_forward_splits(ds, H, 365, 60)
    log: list = []
    oof = run_walk_forward(ds, H, lambda: SpyModel(log), folds)
    end = ds[f"target_end_time_{H}"]
    for train_idx, test_idx in log:
        assert not set(train_idx) & set(test_idx)
        assert end.loc[train_idx].max() <= ds.loc[test_idx, "as_of_time"].min()
    assert oof.index.is_unique and len(oof) == len(ds) - 365


# --- metrics --------------------------------------------------------------------------------

def test_classification_metrics_perfect_and_random() -> None:
    y = np.array([0, 1, 1, 0, 1])
    perfect = classification_metrics(y, y.astype(float))
    assert perfect["accuracy"] == 1 and perfect["roc_auc"] == 1 and perfect["brier"] < 1e-6
    coin = classification_metrics(y, np.full(5, 0.5))
    assert coin["brier"] == pytest.approx(0.25) and coin["roc_auc"] == 0.5


def test_quantile_metrics_and_pinball() -> None:
    y = np.array([-0.2, 0.0, 0.05, 0.3])
    m = quantile_metrics(y, np.full(4, -0.1), np.zeros(4), np.full(4, 0.1))
    assert m["coverage_80"] == 0.5 and m["below_q10"] == 0.25 and m["above_q90"] == 0.25
    assert pinball([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert pinball([-1.0], [0.0], 0.9) == pytest.approx(0.1)


def test_calibration_table_counts() -> None:
    p = np.linspace(0, 1, 101)
    table = calibration_table((p > 0.5).astype(int), p)
    assert sum(r["count"] for r in table) == 101 and len(table) == 10


def test_compare_to_baseline_flags_honestly() -> None:
    base = {"brier": 0.25, "pinball_q50": 0.05, "accuracy": 0.55, "roc_auc": 0.5, "coverage_80": 0.8}
    worse = {**base, "brier": 0.26, "roc_auc": 0.49}
    better = {**base, "brier": 0.22, "roc_auc": 0.62, "accuracy": 0.6}
    assert compare_to_baseline(worse, base)["beats_baseline_brier"] is False
    assert compare_to_baseline(worse, base)["beats_baseline_auc"] is False
    assert compare_to_baseline(better, base)["brier_skill"] == pytest.approx(0.12)


# --- baselines ------------------------------------------------------------------------------

def test_climatology_is_constant_and_ordered() -> None:
    ds = signal_dataset(400)
    pred = ClimatologyModel(H).fit(ds, ds[f"fwd_up_{H}"], ds[f"fwd_logret_{H}"]).predict(ds.iloc[:5])
    assert pred["prob_up"].nunique() == 1
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()


def test_momentum_conditions_on_sign() -> None:
    ds = signal_dataset(600)
    ds[f"fwd_up_{H}"] = (ds["ret_7"] > 0).astype(float)  # perfect momentum world
    m = MomentumModel(H).fit(ds, ds[f"fwd_up_{H}"], ds[f"fwd_logret_{H}"])
    pred = m.predict(ds)
    assert pred.loc[ds["ret_7"] > 0, "prob_up"].eq(1.0).all()
    assert pred.loc[ds["ret_7"] < 0, "prob_up"].eq(0.0).all()


# --- gradient boosting ----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_gbm_finds_real_signal(backend: str) -> None:
    pytest.importorskip(backend)
    from forecaster.modeling.gbm import GBMModel

    ds = signal_dataset(900)
    folds = walk_forward_splits(ds, H, 365, 90)
    base = evaluate(run_walk_forward(ds, H, lambda: ClimatologyModel(H), folds))
    oof = run_walk_forward(ds, H, lambda: GBMModel(H, backend=backend), folds)
    m = evaluate(oof)
    assert m["roc_auc"] > 0.65
    assert compare_to_baseline(m, base)["beats_baseline_brier"]
    assert 0.6 < m["coverage_80"] < 0.95
    assert (oof["q10"] <= oof["q50"]).all() and (oof["q50"] <= oof["q90"]).all()
    assert oof["prob_up"].between(0, 1).all()


def test_gbm_shows_no_skill_on_pure_noise() -> None:
    """Guard against fake skill: on noise the model must not look meaningfully better than baseline."""
    pytest.importorskip("lightgbm")
    from forecaster.modeling.gbm import GBMModel

    ds = signal_dataset(900, strength=0.0, seed=5)
    folds = walk_forward_splits(ds, H, 365, 90)
    base = evaluate(run_walk_forward(ds, H, lambda: ClimatologyModel(H), folds))
    m = evaluate(run_walk_forward(ds, H, lambda: GBMModel(H), folds))
    assert compare_to_baseline(m, base)["brier_skill"] < 0.05


def test_gbm_artifact_roundtrip(tmp_path) -> None:
    pytest.importorskip("lightgbm")
    from forecaster.modeling.gbm import GBMModel

    ds = signal_dataset(500)
    model = GBMModel(H).fit(ds, ds[f"fwd_up_{H}"], ds[f"fwd_logret_{H}"])
    path = save_artifact(model, tmp_path, "SOLUSDT", H, "v1", {"x": 1})
    pd.testing.assert_frame_equal(load_artifact(path).predict(ds.iloc[-20:]), model.predict(ds.iloc[-20:]))


# --- LSTM -----------------------------------------------------------------------------------

def test_lstm_smoke_and_pickle_roundtrip() -> None:
    pytest.importorskip("torch")
    from forecaster.modeling.lstm import LSTMModel

    ds = signal_dataset(400)
    model = LSTMModel(H, seq_len=10, hidden=8, epochs=3).set_history(ds)
    model.fit(ds.iloc[:300], ds[f"fwd_up_{H}"].iloc[:300], ds[f"fwd_logret_{H}"].iloc[:300])
    pred = model.predict(ds.iloc[300:])
    assert pred["prob_up"].between(0, 1).all()
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()

    clone = pickle.loads(pickle.dumps(model))
    assert clone.history is None
    clone.set_history(ds)
    np.testing.assert_allclose(clone.predict(ds.iloc[300:]).to_numpy(), pred.to_numpy(), rtol=1e-5)


def test_lstm_uses_only_past_rows_in_windows() -> None:
    pytest.importorskip("torch")
    from forecaster.modeling.lstm import LSTMModel

    ds = signal_dataset(200)
    model = LSTMModel(H, seq_len=5, hidden=4, epochs=1).set_history(ds)
    model.fit(ds.iloc[:150], ds[f"fwd_up_{H}"].iloc[:150], ds[f"fwd_logret_{H}"].iloc[:150])
    before = model._windows(ds.index[[100]])
    altered = ds.copy()
    altered.iloc[101:, 2:7] = 99.0  # change every future feature value
    model.set_history(altered)
    np.testing.assert_array_equal(model._windows(ds.index[[100]]), before)


# --- registry + end-to-end ------------------------------------------------------------------

def test_activation_keeps_one_active_version(session: Session) -> None:
    sym = sync_symbols(session, ["SOLUSDT"], ["SOLUSDT"])[0]
    m = ClimatologyModel(H)
    runs = []
    for v in ("v1", "v2"):
        runs.append(record_run(session, model=m, symbol_id=sym.id, horizon=H, version=v, feature_set_version="t",
                               train_start=None, train_end=None, metrics={}, artifact_path=None))
        activate(session, runs[-1])
    session.commit()
    active = session.scalars(select(ModelRun).where(ModelRun.is_active.is_(True))).all()
    assert [r.version for r in active] == ["v2"]


def test_train_and_forecast_end_to_end(session: Session, tmp_path) -> None:
    syms = sync_symbols(session, ["SOLUSDT", "BTCUSDT", "ETHUSDT"], ["SOLUSDT"])
    for i, sym in enumerate(syms):
        upsert_klines(session, sym.id, "1d", synthetic_klines(T0, 900, seed=i).reset_index(), "api")
    session.commit()
    s = Settings(model_dir=tmp_path, intervals=["1d"], include_funding_features=False,
                 wf_min_train_days=365, wf_test_days=90)

    results = train_symbol(session, s, "SOLUSDT", H, ["momentum"], now=T0 + timedelta(days=901))
    assert [r.model_name for r in results] == ["climatology", "momentum"]
    assert results[1].metrics["vs_climatology"] is not None
    assert results[1].verdict in {"beats baseline", "mixed", "does NOT beat baseline"}

    oos = session.scalars(select(Prediction).where(Prediction.kind == "oos",
                                                   Prediction.model_run_id == results[1].run_id)).all()
    assert len(oos) == results[1].metrics["n"]
    assert all(p.realized_return is not None and p.price_q10 <= p.price_q90 for p in oos)

    f = forecast(session, s, "SOLUSDT", H, model_name="momentum", now=T0 + timedelta(days=901))
    assert f["as_of_time"] == T0 + timedelta(days=900)
    assert f["price_q10"] <= f["price_q50"] <= f["price_q90"]
    assert 0 <= f["prob_up"] <= 1 and f["is_estimate"] and "not financial advice" in f["disclaimer"]
    live = session.scalars(select(Prediction).where(Prediction.kind == "live")).all()
    assert len(live) == 1 and live[0].realized_return is None
