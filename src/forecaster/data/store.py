"""Idempotent persistence for klines and market context (dialect-aware upserts)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import pandas as pd
from sqlalchemy import Table, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from forecaster.data.klines import COLUMNS
from forecaster.db.models import Kline, MarketContext, Symbol

BATCH_SIZE = 1000
_KLINE_KEY = ["symbol_id", "interval", "open_time"]
_KLINE_UPDATE = [c for c in COLUMNS if c != "open_time"] + ["source"]


def upsert_statement(dialect: str, table: Table, chunk: list[dict], key: list[str], update: list[str],
                     touch: str | None = "ingested_at"):
    """INSERT ... ON DUPLICATE KEY UPDATE (MySQL) / ON CONFLICT DO UPDATE (SQLite)."""
    touched = {touch: func.now()} if touch else {}
    if dialect == "mysql":
        stmt = mysql_insert(table).values(chunk)
        return stmt.on_duplicate_key_update({c: stmt.inserted[c] for c in update} | touched)
    if dialect == "sqlite":
        stmt = sqlite_insert(table).values(chunk)
        return stmt.on_conflict_do_update(index_elements=key,
                                          set_={c: stmt.excluded[c] for c in update} | touched)
    raise NotImplementedError(f"Upsert not implemented for dialect {dialect!r}")


def upsert_rows(session: Session, table: Table, records: list[dict], key: list[str], update: list[str],
                touch: str | None = "ingested_at") -> None:
    dialect = session.get_bind().dialect.name
    for i in range(0, len(records), BATCH_SIZE):
        session.execute(upsert_statement(dialect, table, records[i : i + BATCH_SIZE], key, update, touch))


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with plain Python scalars (no numpy/pandas NA types)."""
    obj = df.astype(object).where(df.notna(), None)
    out = obj.to_dict("records")
    for rec in out:
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp):
                rec[k] = v.to_pydatetime()
    return out


def upsert_klines(session: Session, symbol_id: int, interval: str, df: pd.DataFrame, source: str) -> int:
    """Insert or update klines keyed by (symbol_id, interval, open_time). Returns rows written."""
    if df.empty:
        return 0
    records = _records(df[COLUMNS])
    for rec in records:
        rec.update(symbol_id=symbol_id, interval=interval, source=source)
    upsert_rows(session, Kline.__table__, records, _KLINE_KEY, _KLINE_UPDATE)
    return len(records)


def upsert_context(
    session: Session, symbol_id: int, metric: str, points: Iterable[tuple[datetime, float]]
) -> int:
    records = [{"symbol_id": symbol_id, "metric": metric, "ts": ts, "value": float(v)} for ts, v in points]
    if records:
        upsert_rows(session, MarketContext.__table__, records, ["symbol_id", "metric", "ts"], ["value"])
    return len(records)


def last_open_time(session: Session, symbol_id: int, interval: str) -> datetime | None:
    return session.scalar(
        select(func.max(Kline.open_time)).where(Kline.symbol_id == symbol_id, Kline.interval == interval)
    )


def last_context_ts(session: Session, symbol_id: int, metric: str) -> datetime | None:
    return session.scalar(
        select(func.max(MarketContext.ts)).where(
            MarketContext.symbol_id == symbol_id, MarketContext.metric == metric
        )
    )


def stored_open_times(
    session: Session, symbol_id: int, interval: str, since: datetime | None = None
) -> pd.Series:
    q = select(Kline.open_time).where(Kline.symbol_id == symbol_id, Kline.interval == interval)
    if since is not None:
        q = q.where(Kline.open_time >= since)
    return pd.Series(session.scalars(q.order_by(Kline.open_time)).all(), dtype="datetime64[ns]")


def load_klines(
    session: Session, symbol: str, interval: str, start: datetime | None = None, end: datetime | None = None
) -> pd.DataFrame:
    """Klines for one symbol/interval as a DataFrame indexed by open_time."""
    q = (
        select(*(getattr(Kline, c) for c in COLUMNS))
        .join(Symbol, Symbol.id == Kline.symbol_id)
        .where(Symbol.symbol == symbol, Kline.interval == interval)
    )
    if start is not None:
        q = q.where(Kline.open_time >= start)
    if end is not None:
        q = q.where(Kline.open_time <= end)
    rows = session.execute(q.order_by(Kline.open_time)).all()
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.set_index("open_time")


def load_context(session: Session, symbol: str, metric: str) -> pd.Series:
    q = (
        select(MarketContext.ts, MarketContext.value)
        .join(Symbol, Symbol.id == MarketContext.symbol_id)
        .where(Symbol.symbol == symbol, MarketContext.metric == metric)
        .order_by(MarketContext.ts)
    )
    rows = session.execute(q).all()
    return pd.Series([r.value for r in rows], index=pd.DatetimeIndex([r.ts for r in rows]), name=metric)
