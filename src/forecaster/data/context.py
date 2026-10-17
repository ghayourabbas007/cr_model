"""Optional market-context ingestion from USD-M futures: funding rate and open interest.

BTC dominance is not available from Binance; the market-wide factor is instead BTC's own
returns, derived from BTCUSDT klines in the feature phase.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Protocol

from sqlalchemy.orm import Session

from forecaster.data.binance_client import INVALID_SYMBOL, BinanceAPIError
from forecaster.data.store import last_context_ts, upsert_context
from forecaster.db.models import Symbol
from forecaster.logging_setup import get_logger
from forecaster.timeutil import day_start, from_ms

log = get_logger(__name__)

FUNDING_RATE = "funding_rate"
OPEN_INTEREST = "open_interest"
OPEN_INTEREST_VALUE = "open_interest_value"


class ContextAPI(Protocol):
    def get_funding_rates(self, symbol: str, start: datetime, end: datetime | None = None) -> list[dict]: ...
    def get_open_interest_hist(self, symbol: str, period: str = "1d", limit: int = 30) -> list[dict]: ...


def update_funding(session: Session, sym: Symbol, api: ContextAPI, default_start: date) -> int:
    last = last_context_ts(session, sym.id, FUNDING_RATE)
    start = last + timedelta(milliseconds=1) if last else day_start(default_start)
    rows = api.get_funding_rates(sym.symbol, start)
    n = upsert_context(session, sym.id, FUNDING_RATE,
                       ((from_ms(int(r["fundingTime"])), float(r["fundingRate"])) for r in rows))
    session.commit()
    return n


def update_open_interest(session: Session, sym: Symbol, api: ContextAPI, now: datetime) -> int:
    rows = api.get_open_interest_hist(sym.symbol, period="1d", limit=30)
    # The latest daily bucket may still be forming; keep only completed days.
    rows = [r for r in rows if from_ms(int(r["timestamp"])) + timedelta(days=1) <= now]
    n = upsert_context(session, sym.id, OPEN_INTEREST,
                       ((from_ms(int(r["timestamp"])), float(r["sumOpenInterest"])) for r in rows))
    n += upsert_context(session, sym.id, OPEN_INTEREST_VALUE,
                        ((from_ms(int(r["timestamp"])), float(r["sumOpenInterestValue"])) for r in rows))
    session.commit()
    return n


def update_context(session: Session, sym: Symbol, api: ContextAPI, now: datetime, default_start: date) -> dict:
    """Update all context metrics for one symbol. Symbols without a perpetual are skipped."""
    result = {"symbol": sym.symbol, FUNDING_RATE: 0, OPEN_INTEREST: 0}
    try:
        result[FUNDING_RATE] = update_funding(session, sym, api, default_start)
        result[OPEN_INTEREST] = update_open_interest(session, sym, api, now)
    except BinanceAPIError as exc:
        session.rollback()
        if exc.code != INVALID_SYMBOL:
            raise
        log.info("no USD-M perpetual for symbol; skipping context", extra={"symbol": sym.symbol})
    log.info("context update complete", extra=result)
    return result
