"""Contract tests against real Binance PUBLIC endpoints (read-only, no keys).

Opt-in because they need the network: FORECASTER_LIVE_TESTS=1 pytest -m live
They verify that the parsers match Binance's actual response formats, which mocks cannot.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from forecaster.data.binance_client import BinancePublicClient
from forecaster.data.dumps import BinanceDumpSource
from forecaster.data.klines import find_gaps, only_closed, parse_klines, validate_klines
from forecaster.timeutil import utcnow

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("FORECASTER_LIVE_TESTS") != "1", reason="set FORECASTER_LIVE_TESTS=1"),
]


def test_spot_klines_contract() -> None:
    now = utcnow()
    rows = BinancePublicClient().get_klines("SOLUSDT", "1d", now - timedelta(days=10))
    df, bad = validate_klines(only_closed(parse_klines(rows), now))
    assert bad == 0 and 8 <= len(df) <= 10
    assert (df["close_time"] - df["open_time"]).eq(timedelta(days=1, milliseconds=-1)).all()
    assert find_gaps(df["open_time"], "1d") == []
    assert df["close"].gt(0).all() and df["trades"].gt(0).all()


def test_monthly_dump_contract_and_checksum() -> None:
    df = BinanceDumpSource().fetch_month("SOLUSDT", "1d", 2024, 1)
    assert df is not None and len(df) == 31
    assert df["open_time"].iloc[0] == datetime(2024, 1, 1)
    clean, bad = validate_klines(df)
    assert bad == 0 and len(clean) == 31


def test_recent_dump_uses_microseconds_and_still_parses() -> None:
    now = utcnow()
    y, m = (now.year, now.month - 2) if now.month > 2 else (now.year - 1, now.month + 10)
    df = BinanceDumpSource().fetch_month("SOLUSDT", "1d", y, m)
    assert df is not None and df["open_time"].iloc[0] == datetime(y, m, 1)


def test_futures_context_contract() -> None:
    client = BinancePublicClient()
    funding = client.get_funding_rates("SOLUSDT", utcnow() - timedelta(days=3))
    assert len(funding) >= 6 and {"fundingTime", "fundingRate"} <= set(funding[0])
    oi = client.get_open_interest_hist("SOLUSDT", "1d", 5)
    assert len(oi) == 5 and {"timestamp", "sumOpenInterest", "sumOpenInterestValue"} <= set(oi[0])
