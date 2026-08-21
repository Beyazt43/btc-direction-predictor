"""Scheduled jobs.

Each job is defensive about its own failures: a long-running poller must survive
a bad Binance response or a brief database outage, because the next tick is the
recovery mechanism. Jobs therefore log and swallow, rather than propagate.
"""

import logging

from btcpred.config import get_settings
from btcpred.ingest.service import SyncResult, run_sync

logger = logging.getLogger(__name__)

INGEST_JOB_ID = "ingest_price_bars"


async def ingest_job() -> SyncResult | None:
    """Pull any newly closed candles into price_bars.

    Safe to run more often than candles close: sync_symbol resumes from the
    newest stored candle, so a tick with nothing new to fetch costs one request
    and writes nothing.
    """
    settings = get_settings()
    try:
        result = await run_sync(
            symbol=settings.binance_symbol,
            interval=settings.binance_interval,
            base_url=settings.binance_base_url,
        )
    except Exception:
        # Swallowed deliberately: losing one poll is recoverable, losing the
        # scheduler is not.
        logger.exception("ingestion tick failed; will retry on next tick")
        return None

    if result.written:
        logger.info(
            "ingested %d new candle(s), latest=%s",
            result.written,
            result.latest_open_time,
        )
    else:
        logger.debug("no new candles")
    return result
