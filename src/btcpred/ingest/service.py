"""Ingestion orchestration: pull closed candles from Binance into price_bars."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from btcpred.db.session import session_scope
from btcpred.ingest.binance import MAX_LIMIT, BinanceClient, interval_to_timedelta
from btcpred.ingest.repository import latest_bar_open_time, upsert_price_bars

logger = logging.getLogger(__name__)

# Used only when the table is empty and no explicit start was given.
DEFAULT_LOOKBACK_CANDLES = 500


@dataclass(frozen=True, slots=True)
class SyncResult:
    fetched: int
    written: int
    requests: int
    latest_open_time: datetime | None

    @property
    def up_to_date(self) -> bool:
        return self.fetched == 0


async def sync_symbol(
    client: BinanceClient,
    *,
    symbol: str,
    interval: str,
    start: datetime | None = None,
    now: datetime | None = None,
    max_requests: int = 1000,
) -> SyncResult:
    """Pull every closed candle from `start` (or the last stored one) up to now.

    Polling and backfill are the same operation at different distances, so they
    share one code path: resuming from the newest stored candle means a
    scheduler that was down for a week catches itself up on the next tick
    instead of leaving a permanent hole.
    """
    now = now or datetime.now(UTC)
    step = interval_to_timedelta(interval)

    cursor = start
    if cursor is None:
        async with session_scope() as session:
            newest = await latest_bar_open_time(session, symbol)
        cursor = newest + step if newest else now - step * DEFAULT_LOOKBACK_CANDLES

    fetched = written = requests = 0
    latest = None

    while cursor < now and requests < max_requests:
        klines = await client.fetch_klines(symbol, interval, start_time=cursor, limit=MAX_LIMIT)
        requests += 1
        if not klines:
            break

        # Never persist the in-progress candle: its OHLC values are still moving.
        closed = [k for k in klines if k.is_closed(now)]
        fetched += len(closed)

        if closed:
            async with session_scope() as session:
                written += await upsert_price_bars(session, closed, symbol=symbol)
            latest = closed[-1].open_time

        advanced = klines[-1].open_time + step
        if advanced <= cursor:
            # Defensive: the API did not move forward, so stop rather than spin.
            logger.warning("cursor did not advance past %s; stopping", cursor)
            break
        cursor = advanced

        # A short page means we have reached the present.
        if len(klines) < MAX_LIMIT:
            break

    logger.info(
        "sync %s: fetched=%d written=%d requests=%d latest=%s",
        symbol,
        fetched,
        written,
        requests,
        latest,
    )
    return SyncResult(fetched=fetched, written=written, requests=requests, latest_open_time=latest)


async def run_sync(
    *,
    symbol: str,
    interval: str,
    base_url: str,
    start: datetime | None = None,
) -> SyncResult:
    """Convenience wrapper that owns the HTTP client for a single run."""
    async with BinanceClient(base_url) as client:
        return await sync_symbol(client, symbol=symbol, interval=interval, start=start)
