"""Bulk historical klines from data.binance.vision (public monthly zip dumps)."""

from __future__ import annotations

import hashlib
import io
import zipfile

import pandas as pd

from forecaster.data.http import RetryingHTTP
from forecaster.data.klines import parse_klines
from forecaster.logging_setup import get_logger

log = get_logger(__name__)


class DumpChecksumError(Exception):
    pass


def read_kline_zip(content: bytes) -> pd.DataFrame:
    """Parse a Binance kline zip. Handles CSVs with or without a header row."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            raw = pd.read_csv(fh, header=None)
    raw = raw[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()]
    return parse_klines(raw.iloc[:, :12].values.tolist())


class BinanceDumpSource:
    def __init__(
        self,
        base_url: str = "https://data.binance.vision",
        http: RetryingHTTP | None = None,
        verify_checksum: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.http = http or RetryingHTTP()
        self.verify_checksum = verify_checksum

    def monthly_url(self, symbol: str, interval: str, year: int, month: int) -> str:
        return (f"{self.base_url}/data/spot/monthly/klines/{symbol}/{interval}/"
                f"{symbol}-{interval}-{year}-{month:02d}.zip")

    def fetch_month(self, symbol: str, interval: str, year: int, month: int) -> pd.DataFrame | None:
        """Klines for one month, or None if no dump exists (before listing, or not yet published)."""
        url = self.monthly_url(symbol, interval, year, month)
        resp = self.http.get(url, ok_statuses=(200, 404))
        if resp.status_code == 404:
            return None
        content = resp.content
        if self.verify_checksum:
            self._verify(url, content)
        return read_kline_zip(content)

    def _verify(self, url: str, content: bytes) -> None:
        resp = self.http.get(url + ".CHECKSUM", ok_statuses=(200, 404))
        if resp.status_code == 404:
            log.warning("no checksum published for dump", extra={"url": url})
            return
        expected = resp.text.split()[0].strip().lower()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise DumpChecksumError(f"SHA-256 mismatch for {url}: expected {expected}, got {actual}")
