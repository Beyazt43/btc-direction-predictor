from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from btcpred.ingest.binance import BinanceClient, Kline, interval_to_timedelta

# A real /api/v3/klines row, values abridged only in length.
SAMPLE_ROW = [
    1787220000000,
    "71950.00000000",
    "72100.00000000",
    "71800.00000000",
    "71995.99000000",
    "2350.96460000",
    1787223599999,
    "169283915.11427815",
    335431,
    "1180.11427815",
    "84928.46694368",
    "0",
]


def test_interval_to_timedelta():
    assert interval_to_timedelta("1h") == timedelta(hours=1)
    assert interval_to_timedelta("15m") == timedelta(minutes=15)
    assert interval_to_timedelta("1d") == timedelta(days=1)


@pytest.mark.parametrize("bad", ["1y", "h", "abc", ""])
def test_interval_to_timedelta_rejects_unknown(bad):
    with pytest.raises(ValueError):
        interval_to_timedelta(bad)


def test_from_api_parses_fields():
    k = Kline.from_api(SAMPLE_ROW)
    assert k.open_time == datetime(2026, 8, 20, 10, tzinfo=UTC)
    assert k.close_time == datetime(2026, 8, 20, 10, 59, 59, 999000, tzinfo=UTC)
    assert k.high == Decimal("72100.00000000")
    assert k.num_trades == 335431


def test_prices_are_exact_decimals():
    """Guards the float trap: 71995.99 has no exact binary representation."""
    k = Kline.from_api(SAMPLE_ROW)
    assert isinstance(k.close, Decimal)
    assert k.close == Decimal("71995.99000000")
    # The same value via float would not round-trip cleanly.
    assert Decimal(str(float("71995.99"))) == Decimal("71995.99")
    assert k.close != Decimal(0.1) * Decimal(3)


def test_is_closed_boundary():
    """The in-progress candle must never be treated as final."""
    k = Kline.from_api(SAMPLE_ROW)
    assert k.is_closed(k.close_time + timedelta(microseconds=1))
    assert not k.is_closed(k.close_time)
    assert not k.is_closed(k.close_time - timedelta(minutes=30))


@pytest.mark.asyncio
async def test_fetch_klines_retries_after_rate_limit():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=[SAMPLE_ROW])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://x") as http:
        client = BinanceClient("https://x", client=http)
        klines = await client.fetch_klines("BTCUSDT", "1h")

    assert calls["n"] == 2
    assert len(klines) == 1


@pytest.mark.asyncio
async def test_fetch_klines_raises_after_exhausting_retries():
    transport = httpx.MockTransport(lambda r: httpx.Response(503))
    async with httpx.AsyncClient(transport=transport, base_url="https://x") as http:
        client = BinanceClient("https://x", client=http, max_retries=1)
        with pytest.raises(httpx.HTTPStatusError):
            await client.fetch_klines("BTCUSDT", "1h")
