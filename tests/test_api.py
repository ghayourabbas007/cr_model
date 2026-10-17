from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from forecaster.api.app import create_app
from forecaster.backtest.runner import backtest_model
from forecaster.config import Settings
from forecaster.data.store import upsert_klines
from forecaster.db.models import Prediction
from forecaster.db.repository import sync_symbols
from forecaster.modeling.train import train_symbol
from tests.fakes import synthetic_klines

T0 = datetime(2022, 1, 1)


@pytest.fixture
def client(migrated_engine, tmp_path):
    # TestClient serves requests from another thread; allow SQLite connections across threads.
    engine = create_engine(migrated_engine.url, connect_args={"check_same_thread": False})
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = Settings(model_dir=tmp_path, intervals=["1d"], include_funding_features=False,
                 serving_model="momentum", forecast_horizons=[7, 14], wf_min_train_days=365,
                 wf_test_days=90, bt_bootstrap_samples=50)
    with factory() as session:
        syms = sync_symbols(session, ["SOLUSDT", "BTCUSDT", "ETHUSDT"], ["SOLUSDT"])
        for i, sym in enumerate(syms):
            upsert_klines(session, sym.id, "1d", synthetic_klines(T0, 800, seed=20 + i).reset_index(), "api")
        session.commit()
        for h in (7, 14):
            train_symbol(session, s, "SOLUSDT", h, ["momentum"], now=T0 + timedelta(days=801))
        backtest_model(session, s, "SOLUSDT", 7, "momentum")
    app = create_app(s, factory)
    with TestClient(app) as c:
        c.factory = factory
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["database"] is True and body["read_only"] is True
    assert body["active_models"] == 4  # climatology + momentum, two horizons
    assert body["data"][0]["symbol"] == "SOLUSDT"
    assert body["data"][0]["stale"] is True and body["status"] == "degraded"  # synthetic data is old
    assert "not financial advice" in body["disclaimer"]


def test_forecast_shape_and_labels(client: TestClient) -> None:
    r = client.get("/forecast/solusdt")  # case-insensitive
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "SOLUSDT" and body["is_estimate"] is True
    assert "not financial advice" in body["disclaimer"]
    assert [f["horizon_days"] for f in body["forecasts"]] == [7, 14]
    for f in body["forecasts"]:
        pr, q = f["price_range"], f["log_return_quantiles"]
        assert pr["low"] <= pr["median"] <= pr["high"] and pr["interval"] == 0.8
        assert q["q10"] <= q["q50"] <= q["q90"]
        assert f["prob_up"] + f["prob_down"] == pytest.approx(1.0)
        assert f["direction"] == ("up" if f["prob_up"] >= 0.5 else "down")
        assert f["model"]["name"] == "momentum" and f["model"]["walk_forward_verdict"]
        assert f["as_of_time"].startswith((T0 + timedelta(days=800)).date().isoformat())


def test_forecast_is_cached_and_stored_once(client: TestClient) -> None:
    client.get("/forecast/SOLUSDT")
    client.get("/forecast/SOLUSDT")
    assert len(client.app.state.cache._data) == 2
    with client.factory() as session:
        n = session.scalar(select(func.count()).select_from(Prediction).where(Prediction.kind == "live"))
    assert n == 2


def test_forecast_errors(client: TestClient) -> None:
    assert client.get("/forecast/BTCUSDT").status_code == 404     # context symbol, not a target
    assert client.get("/forecast/DOGEUSDT").status_code == 404
    r = client.get("/forecast/SOLUSDT", params={"model": "lgbm"})
    assert r.status_code == 503 and "forecaster train" in r.json()["detail"]
    assert client.get("/forecast/SOLUSDT", params={"horizons": "x"}).status_code == 422
    one = client.get("/forecast/SOLUSDT", params={"horizons": "7"}).json()
    assert [f["horizon_days"] for f in one["forecasts"]] == [7]


def test_backtest(client: TestClient) -> None:
    body = client.get("/backtest/SOLUSDT").json()
    assert len(body["results"]) == 1
    r = body["results"][0]
    assert r["strategy"] == "momentum_long_flat" and r["horizon_days"] == 7
    assert isinstance(r["beats_benchmark"], bool) and r["verdict"]
    assert r["equity_curve"] is None and "not financial advice" in body["disclaimer"]
    curve = client.get("/backtest/SOLUSDT", params={"include_curve": True}).json()["results"][0]["equity_curve"]
    assert len(curve) > 100 and {"t", "strategy", "benchmark"} <= set(curve[0])
    assert client.get("/backtest/SOLUSDT", params={"horizon": 14}).status_code == 404


def test_models(client: TestClient) -> None:
    body = client.get("/models").json()
    assert body["serving_model"] == "momentum"
    assert {(m["model_name"], m["horizon_days"]) for m in body["models"]} == {
        ("climatology", 7), ("climatology", 14), ("momentum", 7), ("momentum", 14)}
    mom = next(m for m in body["models"] if m["model_name"] == "momentum")
    assert mom["is_active"] and mom["brier"] is not None and mom["verdict"]
    assert client.get("/models", params={"symbol": "ETHUSDT"}).json()["models"] == []


def test_api_is_read_only(client: TestClient) -> None:
    assert client.post("/forecast/SOLUSDT").status_code == 405
    methods = {m for path in client.app.openapi()["paths"].values() for m in path}
    assert methods == {"get"}
