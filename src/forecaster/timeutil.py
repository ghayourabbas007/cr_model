"""Naive-UTC time helpers. The whole system stores and compares naive UTC datetimes."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

INTERVAL_DELTAS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)


def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, UTC).replace(tzinfo=None)


def day_start(d: date) -> datetime:
    return datetime.combine(d, time())


def interval_delta(interval: str) -> timedelta:
    try:
        return INTERVAL_DELTAS[interval]
    except KeyError:
        raise ValueError(f"Unsupported interval {interval!r}") from None


def iter_months(start: date, end_exclusive: date):
    """Yield (year, month) from start's month up to, not including, end_exclusive's month."""
    y, m = start.year, start.month
    while (y, m) < (end_exclusive.year, end_exclusive.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
