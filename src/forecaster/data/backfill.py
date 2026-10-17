"""Historical backfill, gap repair and incremental updates for klines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal, Protocol

import pandas as pd
from sqlalchemy.orm import Session

from forecaster.data.klines import find_gaps, only_closed, parse_klines, validate_klines
from forecaster.data.store import last_open_time, stored_open_times, upsert_klines
from forecaster.db.models import Symbol
from forecaster.logging_setup import get_logger
from forecaster.timeutil import day_start, interval_delta, iter_months

log = get_logger(__name__)

Source = Literal["auto", "dump", "api"]


class KlineAPI(Protocol):
    def get_klines(self, symbol: str, interval: str, start: datetime, end: datetime | None = None) -> list[list]: ...


class KlineDumps(Protocol):
    def fetch_month(self, symbol: str, interval: str, year: int, month: int) -> pd.DataFrame | None: ...


@dataclass
class SyncReport:
    symbol: str
    interval: str
    rows_dump: int = 0
    rows_api: int = 0
    rows_gap_fill: int = 0
    invalid_dropped: int = 0
    unfilled_gaps: list[tuple[datetime, datetime]] = field(default_factory=list)

    def as_log(self) -> dict:
        return {
            "symbol": self.symbol, "interval": self.interval, "rows_dump": self.rows_dump,
            "rows_api": self.rows_api, "rows_gap_fill": self.rows_gap_fill,
            "invalid_dropped": self.invalid_dropped, "unfilled_gaps": len(self.unfilled_gaps),
        }


def _store(session: Session, sym: Symbol, interval: str, df: pd.DataFrame, now: datetime,
           source: str, report: SyncReport) -> int:
    df, bad = validate_klines(only_closed(df, now))
    report.invalid_dropped += bad
    n = upsert_klines(session, sym.id, interval, df, source)
    session.commit()
    return n


def backfill(
    session: Session,
    sym: Symbol,
    interval: str,
    start: date,
    now: datetime,
    api: KlineAPI,
    dumps: KlineDumps | None = None,
    source: Source = "auto",
) -> SyncReport:
    """Load history from `start` to `now`.

    auto: monthly dumps for complete months, REST API for the current month, then API gap repair.
    dump: dumps only.  api: REST API only.  Safe to re-run: every write is an upsert.
    """
    report = SyncReport(sym.symbol, interval)
    api_from = day_start(start)

    if source in ("auto", "dump"):
        if dumps is None:
            raise ValueError("A dump source is required for source='auto' or 'dump'")
        this_month = date(now.year, now.month, 1)
        for y, m in iter_months(start, this_month):
            df = dumps.fetch_month(sym.symbol, interval, y, m)
            if df is None:
                log.debug("no dump for month", extra={"symbol": sym.symbol, "interval": interval,
                                                      "month": f"{y}-{m:02d}"})
                continue
            df = df[df["open_time"] >= pd.Timestamp(api_from)]
            report.rows_dump += _store(session, sym, interval, df, now, "dump", report)
        api_from = max(api_from, day_start(this_month))

    if source in ("auto", "api"):
        rows = api.get_klines(sym.symbol, interval, api_from)
        report.rows_api += _store(session, sym, interval, parse_klines(rows), now, "api", report)

    if source != "dump":
        report.rows_gap_fill, report.unfilled_gaps = repair_gaps(session, sym, interval, api, now)

    log.info("backfill complete", extra=report.as_log())
    return report


def update(
    session: Session,
    sym: Symbol,
    interval: str,
    now: datetime,
    api: KlineAPI,
    default_start: date,
    gap_lookback: timedelta = timedelta(days=90),
) -> SyncReport:
    """Fetch closed candles after the last stored one, then repair recent gaps."""
    report = SyncReport(sym.symbol, interval)
    last = last_open_time(session, sym.id, interval)
    start = last + interval_delta(interval) if last is not None else day_start(default_start)
    if start + interval_delta(interval) <= now:
        rows = api.get_klines(sym.symbol, interval, start)
        report.rows_api = _store(session, sym, interval, parse_klines(rows), now, "api", report)
    report.rows_gap_fill, report.unfilled_gaps = repair_gaps(
        session, sym, interval, api, now, since=now - gap_lookback
    )
    log.info("update complete", extra=report.as_log())
    return report


def repair_gaps(
    session: Session,
    sym: Symbol,
    interval: str,
    api: KlineAPI,
    now: datetime,
    since: datetime | None = None,
) -> tuple[int, list[tuple[datetime, datetime]]]:
    """Refetch missing candles via the API. Returns (rows_written, gaps_still_missing).

    Some gaps are genuine (exchange outages / trading halts) and can never be filled; they are
    reported, not invented. Downstream feature code must handle them explicitly.
    """
    delta = interval_delta(interval)
    gaps = find_gaps(stored_open_times(session, sym.id, interval, since), interval)
    written = 0
    for gap_start, gap_end in gaps:
        rows = api.get_klines(sym.symbol, interval, gap_start, gap_end + delta - timedelta(milliseconds=1))
        df, _ = validate_klines(only_closed(parse_klines(rows), now))
        written += upsert_klines(session, sym.id, interval, df, "api")
    session.commit()

    remaining = find_gaps(stored_open_times(session, sym.id, interval, since), interval) if gaps else []
    if remaining:
        log.warning("gaps remain after repair (likely exchange outages)", extra={
            "symbol": sym.symbol, "interval": interval, "gaps": [(s.isoformat(), e.isoformat()) for s, e in remaining],
        })
    return written, remaining
