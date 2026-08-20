from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from btcpred.ingest import service
from btcpred.ingest.binance import Kline

HOUR = timedelta(hours=1)
START = datetime(2026, 8, 1, tzinfo=UTC)


def make_kline(open_time: datetime) -> Kline:
    return Kline(
        open_time=open_time,
        close_time=open_time + HOUR - timedelta(milliseconds=1),
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("0.5"),
        close=Decimal("1.5"),
        volume=Decimal("10"),
        quote_volume=Decimal("15"),
        num_trades=3,
    )


class FakeClient:
    """Returns canned pages and records the start_time of each request."""

    def __init__(self, pages: list[list[Kline]]):
        self.pages = pages
        self.requested_starts: list[datetime] = []

    async def fetch_klines(self, symbol, interval, *, start_time=None, limit=1000):
        self.requested_starts.append(start_time)
        return self.pages.pop(0) if self.pages else []


@pytest.fixture
def captured_upserts(monkeypatch):
    """Replace persistence so the paging logic can be tested without a database."""
    captured: list[list[Kline]] = []

    @asynccontextmanager
    async def fake_scope():
        yield None

    async def fake_upsert(session, klines, *, symbol, source="binance"):
        captured.append(list(klines))
        return len(klines)

    monkeypatch.setattr(service, "session_scope", fake_scope)
    monkeypatch.setattr(service, "upsert_price_bars", fake_upsert)
    return captured


@pytest.mark.asyncio
async def test_in_progress_candle_is_never_persisted(captured_upserts):
    now = START + 3 * HOUR + timedelta(minutes=20)
    # Three closed candles plus the one currently forming.
    page = [make_kline(START + i * HOUR) for i in range(4)]
    client = FakeClient([page])

    result = await service.sync_symbol(
        client, symbol="BTCUSDT", interval="1h", start=START, now=now
    )

    persisted = [k.open_time for k in captured_upserts[0]]
    assert persisted == [START, START + HOUR, START + 2 * HOUR]
    assert result.written == 3
    assert result.latest_open_time == START + 2 * HOUR


@pytest.mark.asyncio
async def test_short_page_ends_paging(captured_upserts):
    now = START + 10 * HOUR
    client = FakeClient([[make_kline(START + i * HOUR) for i in range(5)]])

    result = await service.sync_symbol(
        client, symbol="BTCUSDT", interval="1h", start=START, now=now
    )

    assert result.requests == 1


@pytest.mark.asyncio
async def test_full_page_continues_from_next_candle(captured_upserts, monkeypatch):
    monkeypatch.setattr(service, "MAX_LIMIT", 2)
    now = START + 5 * HOUR
    client = FakeClient(
        [
            [make_kline(START), make_kline(START + HOUR)],
            [make_kline(START + 2 * HOUR), make_kline(START + 3 * HOUR)],
            [make_kline(START + 4 * HOUR)],
        ]
    )

    result = await service.sync_symbol(
        client, symbol="BTCUSDT", interval="1h", start=START, now=now
    )

    # Each request resumes one interval past the last candle of the previous page.
    assert client.requested_starts == [START, START + 2 * HOUR, START + 4 * HOUR]
    assert result.requests == 3


@pytest.mark.asyncio
async def test_stops_when_cursor_would_not_advance(captured_upserts):
    """A stuck API must not spin the loop forever."""
    now = START + 100 * HOUR
    stale = [make_kline(START - 5 * HOUR)]
    client = FakeClient([stale, stale, stale])

    result = await service.sync_symbol(
        client, symbol="BTCUSDT", interval="1h", start=START, now=now
    )

    assert result.requests == 1


@pytest.mark.asyncio
async def test_resumes_from_newest_stored_candle(captured_upserts, monkeypatch):
    newest = START + 5 * HOUR

    async def fake_latest(session, symbol):
        return newest

    monkeypatch.setattr(service, "latest_bar_open_time", fake_latest)
    client = FakeClient([[make_kline(newest + HOUR)]])

    await service.sync_symbol(client, symbol="BTCUSDT", interval="1h", now=newest + 3 * HOUR)

    assert client.requested_starts == [newest + HOUR]
