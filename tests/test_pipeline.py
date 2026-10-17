from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forecaster.data.backfill import backfill, repair_gaps, update
from forecaster.data.binance_client import BinanceAPIError
from forecaster.data.context import FUNDING_RATE, OPEN_INTEREST, update_context
from forecaster.data.store import load_klines, stored_open_times, upsert_klines
from forecaster.db.models import Kline, MarketContext
from forecaster.db.repository import sync_symbols
from forecaster.timeutil import to_ms
from tests.fakes import FakeDumps, FakeKlineAPI, as_df, make_rows

T0 = datetime(2024, 1, 1)


def _sym(session: Session):
    sym = sync_symbols(session, ["SOLUSDT"], ["SOLUSDT"])[0]
    session.commit()
    return sym


def _count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Kline))


def test_upsert_is_idempotent_and_updates(session: Session) -> None:
    sym = _sym(session)
    rows = make_rows(T0, 5)
    upsert_klines(session, sym.id, "1d", as_df(rows), "api")
    session.commit()
    upsert_klines(session, sym.id, "1d", as_df(rows), "api")
    session.commit()
    assert _count(session) == 5

    rows[0][4] = "999"  # revised close
    upsert_klines(session, sym.id, "1d", as_df(rows[:1]), "dump")
    session.commit()
    df = load_klines(session, "SOLUSDT", "1d")
    assert len(df) == 5 and df["close"].iloc[0] == 999.0


def test_backfill_auto_uses_dumps_then_api_and_skips_forming_candle(session: Session) -> None:
    sym = _sym(session)
    rows = make_rows(T0, 75)                      # Jan 1 .. Mar 15
    now = datetime(2024, 3, 14, 12)               # Mar 14 candle is still forming
    dumps, api = FakeDumps(rows), FakeKlineAPI(rows)

    report = backfill(session, sym, "1d", date(2024, 1, 1), now, api, dumps, "auto")

    assert dumps.calls == [(2024, 1), (2024, 2)]
    assert api.calls[0][2] == datetime(2024, 3, 1)  # API only for the current month
    assert report.rows_dump == 31 + 29
    assert report.rows_api == 13                   # Mar 1 .. Mar 13
    times = stored_open_times(session, sym.id, "1d")
    assert times.max() == datetime(2024, 3, 13)
    assert report.unfilled_gaps == []


def test_backfill_rerun_does_not_duplicate(session: Session) -> None:
    sym = _sym(session)
    rows = make_rows(T0, 40)
    now = T0 + timedelta(days=40)
    for _ in range(2):
        backfill(session, sym, "1d", date(2024, 1, 1), now, FakeKlineAPI(rows), FakeDumps(rows))
    assert _count(session) == 40


def test_gap_in_dump_is_repaired_from_api(session: Session) -> None:
    sym = _sym(session)
    rows = make_rows(T0, 40)
    hole = [r for i, r in enumerate(rows) if i not in (10, 11, 12)]
    report = backfill(session, sym, "1d", date(2024, 1, 1), T0 + timedelta(days=40),
                      FakeKlineAPI(rows), FakeDumps(hole))
    assert report.rows_gap_fill == 3
    assert report.unfilled_gaps == []
    assert _count(session) == 40


def test_unfillable_gap_is_reported_not_invented(session: Session) -> None:
    sym = _sym(session)
    rows = make_rows(T0, 20)
    outage = {T0 + timedelta(days=5)}
    upsert_klines(session, sym.id, "1d", as_df([r for i, r in enumerate(rows) if i != 5]), "api")
    session.commit()
    written, remaining = repair_gaps(session, sym, "1d", FakeKlineAPI(rows, missing=outage),
                                     T0 + timedelta(days=20))
    assert written == 0
    assert remaining == [(T0 + timedelta(days=5), T0 + timedelta(days=5))]
    assert _count(session) == 19


def test_incremental_update_starts_after_last_candle(session: Session) -> None:
    sym = _sym(session)
    rows = make_rows(T0, 30)
    upsert_klines(session, sym.id, "1d", as_df(rows[:20]), "api")
    session.commit()

    api = FakeKlineAPI(rows)
    now = T0 + timedelta(days=25, hours=3)
    report = update(session, sym, "1d", now, api, date(2024, 1, 1))

    assert api.calls[0][2] == T0 + timedelta(days=20)
    assert report.rows_api == 5                    # days 20..24 closed; day 25 forming
    assert stored_open_times(session, sym.id, "1d").max() == T0 + timedelta(days=24)


def test_update_noop_when_up_to_date(session: Session) -> None:
    sym = _sym(session)
    rows = make_rows(T0, 10)
    upsert_klines(session, sym.id, "1d", as_df(rows), "api")
    session.commit()
    api = FakeKlineAPI(rows)
    update(session, sym, "1d", T0 + timedelta(days=10, hours=1), api, date(2024, 1, 1))
    assert not any(c[3] is None for c in api.calls)  # no open-ended fetch was needed


class FakeContextAPI:
    def __init__(self, fail_code: int | None = None):
        self.fail_code = fail_code

    def get_funding_rates(self, symbol, start, end=None):
        if self.fail_code:
            raise BinanceAPIError(400, self.fail_code, "Invalid symbol.")
        return [{"fundingTime": to_ms(T0 + timedelta(hours=8 * i)), "fundingRate": "0.0001"} for i in range(6)]

    def get_open_interest_hist(self, symbol, period="1d", limit=30):
        return [{"timestamp": to_ms(T0 + timedelta(days=i)), "sumOpenInterest": "1000",
                 "sumOpenInterestValue": "100000"} for i in range(3)]


def test_context_ingestion_keeps_completed_days_only(session: Session) -> None:
    sym = _sym(session)
    now = T0 + timedelta(days=2, hours=6)          # day-2 OI bucket not complete
    res = update_context(session, sym, FakeContextAPI(), now, date(2024, 1, 1))
    assert res[FUNDING_RATE] == 6
    oi = session.scalars(select(MarketContext).where(MarketContext.metric == OPEN_INTEREST)).all()
    assert len(oi) == 2


def test_context_skips_symbol_without_perpetual(session: Session) -> None:
    sym = _sym(session)
    res = update_context(session, sym, FakeContextAPI(fail_code=-1121), T0, date(2024, 1, 1))
    assert res[FUNDING_RATE] == 0
