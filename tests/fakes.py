"""Test doubles: synthetic klines and in-memory Binance API / dump sources."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta

import pandas as pd

from forecaster.data.klines import parse_klines
from forecaster.timeutil import interval_delta, to_ms


def make_rows(start: datetime, n: int, interval: str = "1d", *, micros: bool = False,
              price: float = 100.0) -> list[list]:
    """n consecutive raw Binance kline rows starting at `start`."""
    delta = interval_delta(interval)
    scale = 1000 if micros else 1
    rows = []
    for i in range(n):
        t = start + i * delta
        o = price + i
        rows.append([
            to_ms(t) * scale, str(o), str(o + 2), str(o - 1), str(o + 1), "10.5",
            (to_ms(t + delta) - 1) * scale, "1050.0", 42, "5.0", "500.0", "0",
        ])
    return rows


def make_zip(rows: list[list], header: bool = False) -> bytes:
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                     "taker_buy_volume,taker_buy_quote_volume,ignore")
    lines += [",".join(str(v) for v in r) for r in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SOLUSDT-1d-2024-01.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


class FakeKlineAPI:
    """Serves klines from an in-memory universe, honouring start/end like Binance."""

    def __init__(self, rows: list[list], missing: set[datetime] | None = None):
        self.rows = rows
        self.missing = missing or set()
        self.calls: list[tuple[str, str, datetime, datetime | None]] = []

    def get_klines(self, symbol, interval, start, end=None):
        self.calls.append((symbol, interval, start, end))
        lo, hi = to_ms(start), to_ms(end) if end else None
        return [
            r for r in self.rows
            if r[0] >= lo and (hi is None or r[0] <= hi) and _ms_to_dt(r[0]) not in self.missing
        ]


class FakeDumps:
    def __init__(self, rows: list[list]):
        self.df = parse_klines(rows)
        self.calls: list[tuple[int, int]] = []

    def fetch_month(self, symbol, interval, year, month):
        self.calls.append((year, month))
        t = self.df["open_time"]
        sel = self.df[(t.dt.year == year) & (t.dt.month == month)]
        return sel.reset_index(drop=True) if len(sel) else None


class FakeResponse:
    def __init__(self, status: int = 200, json_data=None, headers=None, content: bytes = b"", text: str = ""):
        self.status_code = status
        self._json = json_data
        self.headers = headers or {}
        self.content = content
        self.text = text
        self.url = "https://fake"

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.headers: dict = {}
        self.requests: list[tuple[str, dict | None]] = []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, params))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ms_to_dt(ms: int) -> datetime:
    return datetime(1970, 1, 1) + timedelta(milliseconds=ms)


def as_df(rows: list[list]) -> pd.DataFrame:
    return parse_klines(rows)
