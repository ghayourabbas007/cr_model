"""Streamlit Community Cloud entry point: the dashboard over a committed, static snapshot.

The cloud cannot reach MySQL, and Binance blocks US cloud regions, so the app reads
deploy/snapshot/ (SQLite + model artifacts) built locally with `forecaster snapshot`.
Settings can still be overridden via Streamlit secrets (exposed as environment variables).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAP = HERE / "snapshot"
sys.path.insert(0, str(ROOT / "src"))

for key, value in {
    "DATABASE_URL": f"sqlite:///{(SNAP / 'forecast.db').as_posix()}?check_same_thread=false",
    "MODEL_DIR": str(SNAP / "models"),
    "TARGET_SYMBOLS": "SOLUSDT",
    "SERVING_MODEL": "lgbm",
    # torch is not installed in the cloud, so the LSTM is listed in metrics/backtests but not served.
    "TRAIN_MODELS": "climatology,momentum,lgbm,xgb",
    "LOG_LEVEL": "WARNING",
}.items():
    os.environ.setdefault(key, value)
os.environ["FORECASTER_SNAPSHOT"] = "1"

runpy.run_path(str(ROOT / "src" / "forecaster" / "dashboard" / "app.py"), run_name="__main__")
