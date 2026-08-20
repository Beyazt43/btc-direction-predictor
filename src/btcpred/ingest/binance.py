"""Client for Binance public market data (`/api/v3/klines`)."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import TracebackType
from typing import Any, Self

import httpx

logger = logging.getLogger(__name__)

# Binance returns each kline as a positional array; name the offsets we read.
_OPEN_TIME = 0
_OPEN = 1
_HIGH = 2
_LOW = 3
_CLOSE = 4
_VOLUME = 5
_CLOSE_TIME = 6
_QUOTE_VOLUME = 7
_NUM_TRADES = 8

# klines caps out at 1000 rows per request.
MAX_LIMIT = 1000

_INTERVAL_UNITS = {
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def interval_to_timedelta(interval: str) -> timedelta:
    """Convert a Binance interval string ('1h', '5m') into a timedelta."""
    if not interval or interval[-1] not in _INTERVAL_UNITS or not interval[:-1].isdigit():
        raise ValueError(f"unsupported interval: {interval!r}")
    return timedelta(**{_INTERVAL_UNITS[interval[-1]]: int(interval[:-1])})


def _to_datetime(epoch_millis: int) -> datetime:
    return datetime.fromtimestamp(epoch_millis / 1000, tz=UTC)


@dataclass(frozen=True, slots=True)
class Kline:
    """One OHLCV candle.

    Prices are Decimal rather than float: these land in NUMERIC(18,8) columns and
    binary floating point cannot represent decimal fractions exactly.
    """

    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal | None
    num_trades: int | None

    @classmethod
    def from_api(cls, row: list[Any]) -> "Kline":
        return cls(
            open_time=_to_datetime(row[_OPEN_TIME]),
            close_time=_to_datetime(row[_CLOSE_TIME]),
            open=Decimal(row[_OPEN]),
            high=Decimal(row[_HIGH]),
            low=Decimal(row[_LOW]),
            close=Decimal(row[_CLOSE]),
            volume=Decimal(row[_VOLUME]),
            quote_volume=Decimal(row[_QUOTE_VOLUME]) if row[_QUOTE_VOLUME] is not None else None,
            num_trades=int(row[_NUM_TRADES]) if row[_NUM_TRADES] is not None else None,
        )

    def is_closed(self, now: datetime) -> bool:
        """Whether the candle is final.

        Binance happily returns the in-progress candle, whose OHLC values are
        still moving. Ingesting one would write a value that later changes, so
        every downstream consumer must treat unclosed candles as non-existent.
        """
        return self.close_time < now


class BinanceClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._client = client or httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._owns_client = client is None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = MAX_LIMIT,
    ) -> list[Kline]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, MAX_LIMIT),
        }
        if start_time is not None:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.timestamp() * 1000)

        payload = await self._get("/api/v3/klines", params)
        return [Kline.from_api(row) for row in payload]

    async def _get(self, path: str, params: dict[str, Any]) -> list[Any]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.get(path, params=params)
            except httpx.RequestError as exc:  # network-level failure
                last_error = exc
                await self._backoff(attempt)
                continue

            # 429 = rate limited, 418 = banned for ignoring 429s. Both are
            # recoverable if we actually wait for the window Binance names.
            if response.status_code in (429, 418):
                retry_after = float(response.headers.get("Retry-After", 2**attempt))
                logger.warning("binance rate limited, sleeping %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                last_error = httpx.HTTPStatusError(
                    "rate limited", request=response.request, response=response
                )
                continue

            if response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    "server error", request=response.request, response=response
                )
                await self._backoff(attempt)
                continue

            response.raise_for_status()
            return response.json()

        assert last_error is not None
        raise last_error

    @staticmethod
    async def _backoff(attempt: int) -> None:
        await asyncio.sleep(2**attempt)
