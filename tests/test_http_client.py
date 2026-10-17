from __future__ import annotations

from datetime import datetime

import pytest
import requests

from forecaster.data.binance_client import KLINES_LIMIT, BinanceAPIError, BinancePublicClient
from forecaster.data.dumps import BinanceDumpSource, DumpChecksumError
from forecaster.data.http import HTTPStatusError, RetryingHTTP
from tests.fakes import FakeResponse, FakeSession, make_rows, make_zip


def _http(responses, **kw):
    sleeps: list[float] = []
    http = RetryingHTTP(FakeSession(responses), sleep=sleeps.append, **kw)
    return http, sleeps


def test_retry_after_is_honoured_on_429() -> None:
    http, sleeps = _http([FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, [1])])
    assert http.get("u").json() == [1]
    assert sleeps == [7.0]


def test_backoff_on_connection_errors_then_success() -> None:
    http, sleeps = _http([requests.ConnectionError(), requests.Timeout(), FakeResponse(200, [])],
                         backoff_base=1.0)
    assert http.get("u").status_code == 200
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0  # exponential with jitter


def test_gives_up_after_max_retries() -> None:
    http, sleeps = _http([FakeResponse(503)] * 3, max_retries=2)
    with pytest.raises(HTTPStatusError):
        http.get("u")
    assert len(sleeps) == 2


def test_non_retryable_status_raises_immediately() -> None:
    http, sleeps = _http([FakeResponse(400, {"code": -1121, "msg": "Invalid symbol."})])
    with pytest.raises(HTTPStatusError):
        http.get("u")
    assert sleeps == []


def test_weight_throttle_pauses_until_next_minute() -> None:
    sleeps: list[float] = []
    http = RetryingHTTP(
        FakeSession([FakeResponse(200, [], headers={"X-MBX-USED-WEIGHT-1M": "3100"})]),
        weight_soft_limit=3000, sleep=sleeps.append, clock=lambda: 1_000_040.0,  # 20s into a minute
    )
    http.get("u")
    assert sleeps == [41.0]


def test_client_paginates_klines() -> None:
    start = datetime(2020, 1, 1)
    all_rows = make_rows(start, KLINES_LIMIT + 5)
    session = FakeSession([FakeResponse(200, all_rows[:KLINES_LIMIT]),
                           FakeResponse(200, all_rows[KLINES_LIMIT:])])
    client = BinancePublicClient(http=RetryingHTTP(session, sleep=lambda s: None))
    rows = client.get_klines("SOLUSDT", "1d", start)
    assert len(rows) == KLINES_LIMIT + 5
    assert session.requests[1][1]["startTime"] == all_rows[KLINES_LIMIT - 1][0] + 1


def test_client_maps_binance_error() -> None:
    session = FakeSession([FakeResponse(400, {"code": -1121, "msg": "Invalid symbol."})])
    client = BinancePublicClient(http=RetryingHTTP(session, sleep=lambda s: None))
    with pytest.raises(BinanceAPIError) as exc:
        client.get_funding_rates("FOOUSDT", datetime(2024, 1, 1))
    assert exc.value.code == -1121


def test_client_never_sends_credentials() -> None:
    session = FakeSession([FakeResponse(200, [])])
    BinancePublicClient(http=RetryingHTTP(session)).get_klines("SOLUSDT", "1d", datetime(2024, 1, 1))
    assert "X-MBX-APIKEY" not in session.headers
    assert "signature" not in session.requests[0][1]


@pytest.mark.parametrize("header", [False, True])
def test_dump_fetch_with_checksum(header: bool) -> None:
    import hashlib

    content = make_zip(make_rows(datetime(2024, 1, 1), 31), header=header)
    digest = hashlib.sha256(content).hexdigest()
    session = FakeSession([FakeResponse(200, content=content),
                           FakeResponse(200, text=f"{digest}  SOLUSDT-1d-2024-01.zip")])
    df = BinanceDumpSource(http=RetryingHTTP(session)).fetch_month("SOLUSDT", "1d", 2024, 1)
    assert len(df) == 31
    assert session.requests[0][0].endswith("/data/spot/monthly/klines/SOLUSDT/1d/SOLUSDT-1d-2024-01.zip")


def test_dump_checksum_mismatch_raises() -> None:
    content = make_zip(make_rows(datetime(2024, 1, 1), 2))
    session = FakeSession([FakeResponse(200, content=content), FakeResponse(200, text="deadbeef  x.zip")])
    with pytest.raises(DumpChecksumError):
        BinanceDumpSource(http=RetryingHTTP(session)).fetch_month("SOLUSDT", "1d", 2024, 1)


def test_dump_missing_month_returns_none() -> None:
    session = FakeSession([FakeResponse(404)])
    assert BinanceDumpSource(http=RetryingHTTP(session)).fetch_month("SOLUSDT", "1d", 2019, 1) is None
