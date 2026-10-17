"""HTTP GET with retries, exponential backoff, Retry-After handling and request-weight throttling."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Container

import requests

from forecaster.logging_setup import get_logger

log = get_logger(__name__)

# 429 = rate limited, 418 = IP auto-banned after ignoring 429s. Both send Retry-After.
RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}
WEIGHT_HEADER = "X-MBX-USED-WEIGHT-1M"


class HTTPStatusError(Exception):
    def __init__(self, response: requests.Response):
        self.response = response
        self.status_code = response.status_code
        super().__init__(f"HTTP {response.status_code} for {response.url}: {response.text[:300]}")


class RetryingHTTP:
    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        timeout: float = 20.0,
        max_retries: int = 6,
        backoff_base: float = 1.0,
        backoff_cap: float = 60.0,
        weight_soft_limit: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "forecaster/0.1 (read-only market data)")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.weight_soft_limit = weight_soft_limit
        self.sleep = sleep
        self.clock = clock

    def get(
        self, url: str, params: dict | None = None, ok_statuses: Container[int] = (200,)
    ) -> requests.Response:
        """Return the response if its status is in ok_statuses; retry transient failures."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                self._wait(attempt, None, reason=type(exc).__name__, url=url)
                continue

            self._throttle(resp)
            if resp.status_code in ok_statuses:
                return resp
            if resp.status_code not in RETRYABLE_STATUS:
                raise HTTPStatusError(resp)

            last_error = HTTPStatusError(resp)
            self._wait(attempt, _retry_after(resp), reason=f"HTTP {resp.status_code}", url=url)

        assert last_error is not None
        raise last_error

    def _wait(self, attempt: int, retry_after: float | None, *, reason: str, url: str) -> None:
        if attempt >= self.max_retries:
            return
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(self.backoff_cap, self.backoff_base * 2**attempt) * (0.5 + random.random() / 2)
        log.warning("request failed, retrying", extra={"reason": reason, "url": url, "attempt": attempt + 1,
                                                        "delay_s": round(delay, 2)})
        self.sleep(delay)

    def _throttle(self, resp: requests.Response) -> None:
        if self.weight_soft_limit is None:
            return
        used = resp.headers.get(WEIGHT_HEADER)
        if used is None or int(used) < self.weight_soft_limit:
            return
        # Weight resets at the start of each wall-clock minute.
        delay = 60 - (self.clock() % 60) + 1
        log.info("approaching request-weight limit, pausing", extra={"used_weight": int(used),
                                                                      "delay_s": round(delay, 1)})
        self.sleep(delay)


def _retry_after(resp: requests.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
