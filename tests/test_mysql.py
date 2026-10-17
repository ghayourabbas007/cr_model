"""MySQL-specific checks.

The compile tests always run. The integration test runs only when FORECASTER_TEST_MYSQL_URL points
at a disposable, empty database, e.g.
    mysql+pymysql://forecaster:pw@127.0.0.1:3306/crypto_forecast_test?charset=utf8mb4
It migrates up, exercises upserts / millisecond times / JSON, and migrates back down to base.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Session

from forecaster.data.store import load_klines, upsert_klines, upsert_statement
from forecaster.db.models import FeatureRow, Kline, Prediction
from forecaster.db.repository import sync_symbols
from tests.fakes import as_df, make_rows

MYSQL_URL = os.environ.get("FORECASTER_TEST_MYSQL_URL")
ROOT = Path(__file__).resolve().parents[1]


def _compile(stmt) -> str:
    return str(stmt.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": False}))


def test_kline_upsert_compiles_to_on_duplicate_key_update() -> None:
    rec = {"symbol_id": 1, "interval": "1d", "open_time": datetime(2024, 1, 1), "close": 1.0}
    sql = _compile(upsert_statement("mysql", Kline.__table__, [rec], ["symbol_id", "interval", "open_time"],
                                    ["close"]))
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "`interval`" in sql                       # reserved word is quoted
    assert "close = VALUES(close)" in sql or "close = new.close" in sql
    assert "ingested_at = now()" in sql


def test_feature_upsert_compiles_without_touch_column() -> None:
    rec = {"symbol_id": 1, "interval": "1d", "as_of_time": datetime(2024, 1, 2), "feature_set_version": "v1",
           "feature_values": {"a": 1.0}}
    sql = _compile(upsert_statement("mysql", FeatureRow.__table__, [rec], [], ["feature_values"], touch=None))
    assert "ON DUPLICATE KEY UPDATE" in sql and "ingested_at" not in sql


def test_unknown_dialect_is_rejected() -> None:
    with pytest.raises(NotImplementedError):
        upsert_statement("postgresql", Kline.__table__, [{}], [], [])


@pytest.mark.skipif(not MYSQL_URL, reason="set FORECASTER_TEST_MYSQL_URL to run MySQL integration tests")
def test_mysql_round_trip() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", MYSQL_URL.replace("%", "%%"))
    engine = create_engine(MYSQL_URL)
    if inspect(engine).get_table_names():
        pytest.skip("FORECASTER_TEST_MYSQL_URL database is not empty; refusing to touch it")
    command.upgrade(cfg, "head")
    try:
        with Session(engine) as session:
            sym = sync_symbols(session, ["SOLUSDT"], ["SOLUSDT"])[0]
            session.commit()
            rows = make_rows(datetime(2024, 1, 1), 5)
            for _ in range(2):
                upsert_klines(session, sym.id, "1d", as_df(rows), "api")
                session.commit()
            assert session.scalar(select(func.count()).select_from(Kline)) == 5
            k = session.scalars(select(Kline).order_by(Kline.open_time)).first()
            # DATETIME(3): 23:59:59.999 must not round up to the next day.
            assert k.close_time == datetime(2024, 1, 1) + timedelta(days=1, milliseconds=-1)
            assert load_klines(session, "SOLUSDT", "1d")["close"].iloc[0] == pytest.approx(101.0)

            session.add(FeatureRow(symbol_id=sym.id, interval="1d", as_of_time=datetime(2024, 1, 2),
                                   feature_set_version="t", feature_values={"x": 1.5, "y": None}))
            session.commit()
            assert session.scalars(select(FeatureRow)).one().feature_values == {"x": 1.5, "y": None}
            cols = {c["name"]: c for c in inspect(engine).get_columns("predictions")}
            assert "kind" in cols and "DOUBLE" in str(cols["prob_up"]["type"]).upper()
            assert Prediction.__table__.c.kind.server_default is not None
    finally:
        engine.dispose()
        command.downgrade(cfg, "base")
