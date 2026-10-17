from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from forecaster.backtest.runner import backtest_model
from forecaster.config import Settings
from forecaster.dashboard import charts
from forecaster.dashboard import data as dd
from forecaster.data.store import upsert_klines
from forecaster.db.repository import sync_symbols
from forecaster.modeling.train import train_symbol
from tests.fakes import synthetic_klines

T0 = datetime(2022, 1, 1)
APP = Path(__file__).resolve().parents[1] / "src" / "forecaster" / "dashboard" / "app.py"


def _seed(session: Session, s: Settings) -> None:
    syms = sync_symbols(session, ["SOLUSDT", "BTCUSDT", "ETHUSDT"], ["SOLUSDT"])
    for i, sym in enumerate(syms):
        upsert_klines(session, sym.id, "1d", synthetic_klines(T0, 800, seed=30 + i).reset_index(), "api")
    session.commit()
    train_symbol(session, s, "SOLUSDT", 7, ["momentum"], now=T0 + timedelta(days=801))
    backtest_model(session, s, "SOLUSDT", 7, "momentum")


@pytest.fixture
def seeded(session: Session, tmp_path) -> tuple[Session, Settings]:
    s = Settings(model_dir=tmp_path, intervals=["1d"], include_funding_features=False, forecast_horizons=[7],
                 serving_model="momentum", train_models=["climatology", "momentum"], bt_bootstrap_samples=50)
    _seed(session, s)
    return session, s


# --- data layer --------------------------------------------------------------------------------

def test_candles_with_indicators_trims_after_computing(seeded) -> None:
    session, _ = seeded
    df = dd.candles_with_indicators(session, "SOLUSDT", days=90)
    assert len(df) == 90
    assert df["sma_200"].notna().all()  # computed on full history, so valid at the left edge
    assert (df["bb_upper"] >= df["bb_lower"]).all() and df["rsi_14"].between(0, 100).all()


def test_forecasts_models_oos_and_backtests(seeded) -> None:
    session, s = seeded
    fc, reason = dd.current_forecasts(session, s, "SOLUSDT", "momentum")
    assert reason is None and len(fc) == 1 and fc[0]["price_q10"] <= fc[0]["price_q90"]
    none, why = dd.current_forecasts(session, s, "SOLUSDT", "lgbm")
    assert none == [] and "forecaster train" in why

    table = dd.model_table(session, "SOLUSDT")
    assert set(table["model"]) == {"climatology", "momentum"} and table["verdict"].notna().all()

    oos = dd.oos_predictions(session, "SOLUSDT", 7, "momentum")
    assert len(oos) > 300 and set(oos["inside"].unique()) <= {0.0, 1.0}
    rel = dd.reliability(oos)
    assert rel["count"].sum() == len(oos) and rel["mean_pred"].between(0, 1).all()
    cov = dd.rolling_coverage(oos).dropna()
    assert cov.between(0, 1).all()

    bts = dd.backtests(session, "SOLUSDT")
    assert len(bts) == 1 and {"strategy", "benchmark"} <= set(bts[0]["curve"].columns)
    assert isinstance(bts[0]["curve"].index, pd.DatetimeIndex) and bts[0]["verdict"]


# --- charts ------------------------------------------------------------------------------------

def _no_dual_axis(fig) -> bool:
    layout = fig.layout.to_plotly_json()
    return not any(v.get("overlaying") for k, v in layout.items() if k.startswith("yaxis"))


@pytest.mark.parametrize("dark", [False, True])
def test_charts_build_with_one_axis_per_panel(seeded, dark: bool) -> None:
    session, s = seeded
    t = charts.theme(dark)
    candles = dd.candles_with_indicators(session, "SOLUSDT", 200)
    fc, _ = dd.current_forecasts(session, s, "SOLUSDT", "momentum")
    oos = dd.oos_predictions(session, "SOLUSDT", 7, "momentum")
    figs = [
        charts.candlestick(candles, "SOLUSDT", dark),
        charts.forecast_fan(candles["close"], fc, dark),
        charts.reliability_chart(dd.reliability(oos), dark),
        charts.coverage_chart(dd.rolling_coverage(oos).dropna(), dark),
        charts.equity_chart(dd.backtests(session, "SOLUSDT")[0]["curve"], "Equity", dark),
    ]
    for fig in figs:
        assert _no_dual_axis(fig)
        assert fig.layout.hovermode in ("x unified", "closest")
    assert figs[0].data[2].increasing.line.color == t.up
    assert any(tr.name == "80% range (estimate)" for tr in figs[1].data)
    assert "not financial advice" in figs[1].layout.title.text
    named = [tr.name for tr in figs[4].data if tr.showlegend is not False]
    assert named == ["Strategy (after costs)", "Buy & hold"]


def test_dark_and_light_palettes_differ_only_in_steps() -> None:
    assert charts.LIGHT.s1 != charts.DARK.s1 and charts.LIGHT.ink != charts.DARK.ink


# --- app smoke test ----------------------------------------------------------------------------

def test_streamlit_app_renders(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    db = tmp_path / "dash.db"
    url = f"sqlite:///{db.as_posix()}?check_same_thread=false"
    env = {"DATABASE_URL": url, "MODEL_DIR": str(tmp_path / "models"), "TARGET_SYMBOLS": "SOLUSDT",
           "SERVING_MODEL": "momentum", "TRAIN_MODELS": "climatology,momentum", "INTERVALS": "1d",
           "FORECAST_HORIZONS": "7", "INCLUDE_FUNDING_FEATURES": "false", "BT_BOOTSTRAP_SAMPLES": "50"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine

    root = Path(__file__).resolve().parents[1]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    with Session(create_engine(url)) as session:
        _seed(session, Settings())

    at = testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("not financial advice" in w.value for w in at.warning)
    assert any(m.label == "P(up) in 7 days" for m in at.metric)
    assert [t.label for t in at.tabs] == ["Forecast", "Market", "Model performance", "Backtest"]
