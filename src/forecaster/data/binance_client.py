"""Read-only client for public Binance market-data endpoints. No API keys, no order endpoints."""

from __future__ import annotations

from datetime import datetime

from forecaster.data.http import HTTPStatusError, RetryingHTTP
from forecaster.timeutil import to_ms

KLINES_LIMIT = 1000
FUNDING_LIMIT = 1000

# Binance error code for an unknown symbol (e.g. a spot pair with no perpetual future).
INVALID_SYMBOL = -1121


class BinanceAPIError(Exception):
    def __init__(self, status: int, code: int | None, msg: str):
        self.status, self.code, self.msg = status, code, msg
        super().__init__(f"Binance error {code} (HTTP {status}): {msg}")


class BinancePublicClient:
    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        futures_base_url: str = "https://fapi.binance.com",
        http: RetryingHTTP | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.futures_base_url = futures_base_url.rstrip("/")
        self.http = http or RetryingHTTP(weight_soft_limit=3000)

    def _get_json(self, url: str, params: dict):
        try:
            return self.http.get(url, params).json()
        except HTTPStatusError as exc:
            try:
                body = exc.response.json()
            except ValueError:
                body = {}
            raise BinanceAPIError(exc.status_code, body.get("code"), body.get("msg", str(exc))) from exc

    def get_klines(
        self, symbol: str, interval: str, start: datetime, end: datetime | None = None
    ) -> list[list]:
        """All raw kline rows with open_time in [start, end], paginating as needed.

        Includes the currently-forming candle if `end` is None; callers filter to closed candles.
        """
        url = f"{self.base_url}/api/v3/klines"
        cursor = to_ms(start)
        end_ms = to_ms(end) if end is not None else None
        rows: list[list] = []
        while True:
            params = {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": KLINES_LIMIT}
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = self._get_json(url, params)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < KLINES_LIMIT:
                break
            cursor = int(batch[-1][0]) + 1
        return rows

    def get_funding_rates(self, symbol: str, start: datetime, end: datetime | None = None) -> list[dict]:
        """USD-M perpetual funding history: [{fundingTime, fundingRate, markPrice, ...}]."""
        url = f"{self.futures_base_url}/fapi/v1/fundingRate"
        cursor = to_ms(start)
        end_ms = to_ms(end) if end is not None else None
        rows: list[dict] = []
        while True:
            params = {"symbol": symbol, "startTime": cursor, "limit": FUNDING_LIMIT}
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = self._get_json(url, params)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < FUNDING_LIMIT:
                break
            cursor = int(batch[-1]["fundingTime"]) + 1
        return rows

    def get_open_interest_hist(self, symbol: str, period: str = "1d", limit: int = 30) -> list[dict]:
        """Open-interest history. Binance only serves roughly the last 30 days, so the
        scheduled job accumulates it over time."""
        url = f"{self.futures_base_url}/futures/data/openInterestHist"
        return self._get_json(url, {"symbol": symbol, "period": period, "limit": limit})
