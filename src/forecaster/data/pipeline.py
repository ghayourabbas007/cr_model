"""Settings-driven entry points used by the CLI and the scheduler."""

from __future__ import annotations

from forecaster.config import Settings
from forecaster.data import backfill as bf
from forecaster.data.binance_client import BinancePublicClient
from forecaster.data.context import update_context
from forecaster.data.dumps import BinanceDumpSource
from forecaster.data.http import RetryingHTTP
from forecaster.data.klines import find_gaps
from forecaster.data.store import stored_open_times
from forecaster.db.repository import get_symbol, sync_symbols
from forecaster.db.session import session_scope
from forecaster.logging_setup import get_logger
from forecaster.timeutil import utcnow

log = get_logger(__name__)


def build_client(s: Settings) -> BinancePublicClient:
    http = RetryingHTTP(weight_soft_limit=s.binance_weight_soft_limit)
    return BinancePublicClient(s.binance_base_url, s.binance_futures_base_url, http=http)


def run_backfill(s: Settings, symbols: list[str] | None = None, intervals: list[str] | None = None,
                 source: bf.Source = "auto") -> list[bf.SyncReport]:
    client = build_client(s)
    dumps = BinanceDumpSource(s.binance_data_dump_url)
    reports = []
    with session_scope() as session:
        sync_symbols(session, s.all_symbols, s.target_symbols)
        session.commit()
        for name in symbols or s.all_symbols:
            sym = get_symbol(session, name)
            if sym is None:
                raise ValueError(f"{name} is not configured in SYMBOLS / TARGET_SYMBOLS")
            for interval in intervals or s.intervals:
                reports.append(bf.backfill(session, sym, interval, s.backfill_start, utcnow(),
                                           client, dumps, source))
    return reports


def run_update(s: Settings) -> list[bf.SyncReport]:
    client = build_client(s)
    reports = []
    with session_scope() as session:
        syms = sync_symbols(session, s.all_symbols, s.target_symbols)
        session.commit()
        for sym in syms:
            for interval in s.intervals:
                try:
                    reports.append(bf.update(session, sym, interval, utcnow(), client, s.backfill_start))
                except Exception:
                    # One failing symbol must not block the others.
                    session.rollback()
                    log.exception("update failed", extra={"symbol": sym.symbol, "interval": interval})
    return reports


def run_context(s: Settings) -> list[dict]:
    client = build_client(s)
    out = []
    with session_scope() as session:
        for sym in sync_symbols(session, s.all_symbols, s.target_symbols):
            try:
                out.append(update_context(session, sym, client, utcnow(), s.backfill_start))
            except Exception:
                session.rollback()
                log.exception("context update failed", extra={"symbol": sym.symbol})
    return out


def report_gaps(s: Settings) -> dict[tuple[str, str], list]:
    out = {}
    with session_scope() as session:
        for name in s.all_symbols:
            sym = get_symbol(session, name)
            if sym is None:
                continue
            for interval in s.intervals:
                out[(name, interval)] = find_gaps(stored_open_times(session, sym.id, interval), interval)
    return out


def scheduled_job(s: Settings) -> None:
    run_update(s)
    if s.include_market_context:
        run_context(s)
