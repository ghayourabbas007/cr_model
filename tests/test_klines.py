from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from forecaster.data.klines import epoch_to_datetime, find_gaps, only_closed, parse_klines, validate_klines
from tests.fakes import make_rows

T0 = datetime(2024, 1, 1)


def test_parse_klines_types_and_times() -> None:
    df = parse_klines(make_rows(T0, 3))
    assert list(df["open_time"]) == [T0, T0 + timedelta(days=1), T0 + timedelta(days=2)]
    assert df["close_time"][0] == T0 + timedelta(days=1) - timedelta(milliseconds=1)
    assert df["open"].dtype == "float64"
    assert df["open"][0] == 100.0 and df["trades"][0] == 42


def test_microsecond_timestamps_are_detected() -> None:
    ms = parse_klines(make_rows(T0, 2))
    us = parse_klines(make_rows(T0, 2, micros=True))
    pd.testing.assert_frame_equal(ms, us)


def test_mixed_epoch_units() -> None:
    s = pd.Series([1704067200000, 1704153600000000])
    assert list(epoch_to_datetime(s)) == [T0, T0 + timedelta(days=1)]


def test_parse_empty() -> None:
    df = parse_klines([])
    assert df.empty and "open_time" in df.columns


def test_validate_drops_duplicates_and_bad_ohlc() -> None:
    rows = make_rows(T0, 4)
    rows.append(list(rows[1]))          # duplicate open_time
    rows[2][2] = "50"                   # high below open -> invalid
    rows[3][3] = "0"                    # non-positive low -> invalid
    df, dropped = validate_klines(parse_klines(rows))
    assert dropped == 3
    assert list(df["open_time"]) == [T0, T0 + timedelta(days=1)]


def test_only_closed_excludes_forming_candle() -> None:
    df = parse_klines(make_rows(T0, 3))
    now = T0 + timedelta(days=2, hours=5)   # third candle still open
    assert len(only_closed(df, now)) == 2
    assert len(only_closed(df, T0 + timedelta(days=3))) == 3


def test_find_gaps() -> None:
    times = [T0, T0 + timedelta(days=1), T0 + timedelta(days=4), T0 + timedelta(days=5), T0 + timedelta(days=7)]
    assert find_gaps(times, "1d") == [
        (T0 + timedelta(days=2), T0 + timedelta(days=3)),
        (T0 + timedelta(days=6), T0 + timedelta(days=6)),
    ]


def test_find_gaps_none_and_4h() -> None:
    assert find_gaps([T0 + timedelta(hours=4 * i) for i in range(10)], "4h") == []
    assert find_gaps([T0], "1d") == []
    assert find_gaps([], "1d") == []
