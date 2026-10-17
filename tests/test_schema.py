from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from forecaster.db import Base, Kline, Symbol
from forecaster.db.repository import split_symbol, sync_symbols

EXPECTED_TABLES = {"symbols", "klines", "features", "model_runs", "predictions", "backtest_results"}


def test_migration_creates_all_tables(migrated_engine) -> None:
    assert EXPECTED_TABLES <= set(inspect(migrated_engine).get_table_names())


def test_migration_matches_orm_columns(migrated_engine) -> None:
    """Hand-written migration and ORM models must not drift apart."""
    insp = inspect(migrated_engine)
    for table in Base.metadata.sorted_tables:
        db_cols = {c["name"] for c in insp.get_columns(table.name)}
        assert db_cols == set(table.columns.keys()), table.name


def _kline(symbol_id: int, t: datetime) -> Kline:
    return Kline(
        symbol_id=symbol_id, interval="1d", open_time=t, close_time=t + timedelta(days=1, milliseconds=-1),
        open=1.0, high=2.0, low=0.5, close=1.5, volume=10, quote_volume=15, trades=3,
    )


def test_kline_unique_key_prevents_duplicates(session: Session) -> None:
    sym = sync_symbols(session, ["SOLUSDT"], ["SOLUSDT"])[0]
    t = datetime(2024, 1, 1)
    session.add(_kline(sym.id, t))
    session.commit()
    session.add(_kline(sym.id, t))
    with pytest.raises(IntegrityError):
        session.commit()


def test_sync_symbols_is_idempotent(session: Session) -> None:
    sync_symbols(session, ["SOLUSDT", "BTCUSDT"], ["SOLUSDT"])
    session.commit()
    sync_symbols(session, ["SOLUSDT", "BTCUSDT", "ETHUSDT"], ["SOLUSDT", "ETHUSDT"])
    session.commit()
    rows = {s.symbol: s for s in session.query(Symbol)}
    assert set(rows) == {"SOLUSDT", "BTCUSDT", "ETHUSDT"}
    assert rows["ETHUSDT"].is_target and not rows["BTCUSDT"].is_target
    assert (rows["SOLUSDT"].base_asset, rows["SOLUSDT"].quote_asset) == ("SOL", "USDT")


@pytest.mark.parametrize(
    ("sym", "expected"), [("SOLUSDT", ("SOL", "USDT")), ("ETHBTC", ("ETH", "BTC")), ("BTCFDUSD", ("BTC", "FDUSD"))]
)
def test_split_symbol(sym: str, expected: tuple[str, str]) -> None:
    assert split_symbol(sym) == expected


def test_split_symbol_unknown_quote() -> None:
    with pytest.raises(ValueError):
        split_symbol("FOOBAR")
