"""Retrain: walk-forward evaluation of every model vs the climatology baseline, then a final
fit on all realized data, artifact save, model_runs record, OOS predictions and activation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from forecaster.config import Settings
from forecaster.db.repository import get_symbol
from forecaster.features.build import FEATURE_SET_VERSION, build_features, build_targets, make_dataset
from forecaster.features.pipeline import load_inputs
from forecaster.logging_setup import get_logger
from forecaster.modeling.metrics import compare_to_baseline, evaluate
from forecaster.modeling.predict import save_predictions
from forecaster.modeling.registry import (
    activate,
    make_model,
    new_version,
    record_run,
    relative_artifact_path,
    save_artifact,
)
from forecaster.modeling.walkforward import run_walk_forward, walk_forward_splits
from forecaster.timeutil import utcnow

log = get_logger(__name__)
BASELINE = "climatology"


@dataclass
class TrainResult:
    symbol: str
    horizon: int
    model_name: str
    run_id: int
    version: str
    metrics: dict

    @property
    def verdict(self) -> str:
        return model_verdict(self.model_name, self.metrics)


def model_verdict(model_name: str, metrics: dict | None) -> str:
    """Walk-forward verdict vs climatology: Brier skill and AUC must both improve."""
    if model_name == BASELINE:
        return "baseline"
    cmp = (metrics or {}).get("vs_climatology") or {}
    if cmp.get("beats_baseline_brier") and cmp.get("beats_baseline_auc"):
        return "beats baseline"
    if cmp.get("beats_baseline_brier") or cmp.get("beats_baseline_auc"):
        return "mixed"
    return "does NOT beat baseline"


def train_symbol(session: Session, s: Settings, symbol: str, horizon: int, model_names: list[str],
                 now: datetime | None = None) -> list[TrainResult]:
    now = now or utcnow()
    sym = get_symbol(session, symbol)
    if sym is None:
        raise LookupError(f"Unknown symbol {symbol}; run `forecaster db init`")

    inp = load_inputs(session, s, symbol)
    feats = build_features(inp)
    ds = make_dataset(feats, build_targets(inp.daily, [horizon]), horizon, now)
    folds = walk_forward_splits(ds, horizon, s.wf_min_train_days, s.wf_test_days,
                                max_train=s.wf_max_train_days)
    if not folds:
        raise ValueError(f"{symbol}: {len(ds)} usable rows is not enough for walk-forward "
                         f"(WF_MIN_TRAIN_DAYS={s.wf_min_train_days})")
    log.info("walk-forward setup", extra={"symbol": symbol, "horizon": horizon, "rows": len(ds),
                                          "folds": len(folds), "oos_rows": sum(len(f.test_idx) for f in folds)})

    names = list(dict.fromkeys([BASELINE, *model_names]))  # baseline always evaluated
    oofs, metrics = {}, {}
    for name in names:
        oofs[name] = run_walk_forward(ds, horizon, lambda n=name: make_model(n, horizon, s), folds, history=feats)
        metrics[name] = evaluate(oofs[name])
    base = metrics[BASELINE]

    version = new_version(now)
    y_up, y_ret = ds[f"fwd_up_{horizon}"], ds[f"fwd_logret_{horizon}"]
    results = []
    for name in names:
        m = metrics[name]
        m["vs_climatology"] = None if name == BASELINE else compare_to_baseline(m, base)
        m["always_up_accuracy"] = m["base_rate_up"]
        m["oos_start"] = str(folds[0].test_start)
        m["oos_end"] = str(folds[-1].test_end)

        model = make_model(name, horizon, s)
        if hasattr(model, "set_history"):
            model.set_history(feats)
        model.fit(ds, y_up, y_ret)
        meta = {"symbol": symbol, "horizon": horizon, "model": name, "version": version,
                "feature_set_version": FEATURE_SET_VERSION, "features": model.features_,
                "trained_at": now, "rows": len(ds)}
        path = save_artifact(model, s.model_dir, symbol, horizon, version, meta)
        run = record_run(session, model=model, symbol_id=sym.id, horizon=horizon, version=version,
                         feature_set_version=FEATURE_SET_VERSION,
                         train_start=ds["as_of_time"].min().to_pydatetime(),
                         train_end=ds["as_of_time"].max().to_pydatetime(),
                         metrics=m, artifact_path=relative_artifact_path(path, s.model_dir))
        save_predictions(session, run.id, sym.id, horizon, oofs[name], "oos")
        activate(session, run)
        session.commit()
        res = TrainResult(symbol, horizon, name, run.id, version, m)
        log.info("model trained", extra={"symbol": symbol, "horizon": horizon, "model": name, "run_id": run.id,
                                         "brier": round(m["brier"], 4), "auc": m["roc_auc"],
                                         "coverage_80": round(m["coverage_80"], 3), "verdict": res.verdict})
        results.append(res)
    return results
