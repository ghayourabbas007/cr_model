"""Model factories, artifact persistence and model_runs bookkeeping (versioning / activation)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from forecaster.config import Settings
from forecaster.db.models import ModelRun
from forecaster.modeling.baselines import ClimatologyModel, MomentumModel
from forecaster.timeutil import utcnow

MODEL_TYPES = {"climatology": "baseline", "momentum": "baseline", "lgbm": "lightgbm",
               "xgb": "xgboost", "lstm": "lstm"}
TASK = "dir+quantile"


def make_model(name: str, horizon: int, s: Settings):
    if name == "climatology":
        return ClimatologyModel(horizon)
    if name == "momentum":
        return MomentumModel(horizon)
    if name in ("lgbm", "xgb"):
        from forecaster.modeling.gbm import GBMModel
        return GBMModel(horizon, backend="lightgbm" if name == "lgbm" else "xgboost", seed=s.random_seed)
    if name == "lstm":
        from forecaster.modeling.lstm import LSTMModel
        return LSTMModel(horizon, seq_len=s.lstm_seq_len, epochs=s.lstm_epochs, seed=s.random_seed)
    raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_TYPES)}")


def model_params(model) -> dict:
    keep = (int, float, str, bool, type(None))
    out = {k: v for k, v in vars(model).items() if not k.endswith("_") and isinstance(v, keep)}
    if isinstance(getattr(model, "params", None), dict):
        out["params"] = model.params
    return out


def new_version(now: datetime | None = None) -> str:
    return (now or utcnow()).strftime("%Y%m%dT%H%M%SZ")


def save_artifact(model, model_dir: Path, symbol: str, horizon: int, version: str, meta: dict) -> Path:
    """Artifacts are pickles: only ever load files this system wrote itself."""
    d = Path(model_dir) / symbol / model.name / f"h{horizon}" / version
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "model.joblib")
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return d / "model.joblib"


def relative_artifact_path(path: Path, model_dir: Path) -> str:
    """Stored form of an artifact path: relative to MODEL_DIR, POSIX separators, so the database
    stays valid when the model directory moves (another machine, a deployed snapshot)."""
    try:
        return Path(path).resolve().relative_to(Path(model_dir).resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_artifact_path(stored: str, model_dir: Path) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else Path(model_dir) / p


def load_artifact(path: str | Path, model_dir: Path | None = None):
    return joblib.load(resolve_artifact_path(str(path), model_dir) if model_dir is not None else path)


def record_run(session: Session, *, model, symbol_id: int, horizon: int, version: str,
               feature_set_version: str, train_start, train_end, metrics: dict, artifact_path: str | None,
               notes: str | None = None) -> ModelRun:
    run = ModelRun(
        model_name=model.name, model_type=MODEL_TYPES[model.name], task=TASK, version=version,
        symbol_id=symbol_id, interval="1d", horizon_days=horizon, feature_set_version=feature_set_version,
        train_start=train_start, train_end=train_end, params=model_params(model), metrics=metrics,
        artifact_path=artifact_path, status="trained", is_active=False, notes=notes,
    )
    session.add(run)
    session.flush()
    return run


def activate(session: Session, run: ModelRun) -> None:
    """Make `run` the active version for its (symbol, horizon, model_name)."""
    session.execute(
        update(ModelRun)
        .where(ModelRun.symbol_id == run.symbol_id, ModelRun.horizon_days == run.horizon_days,
               ModelRun.model_name == run.model_name, ModelRun.id != run.id)
        .values(is_active=False)
    )
    run.is_active = True
    session.flush()


def active_run(session: Session, symbol_id: int, horizon: int, model_name: str) -> ModelRun | None:
    return session.scalar(
        select(ModelRun).where(ModelRun.symbol_id == symbol_id, ModelRun.horizon_days == horizon,
                               ModelRun.model_name == model_name, ModelRun.is_active.is_(True))
    )
