"""APScheduler process: owns the recurring ingestion loop.

Runs as its own container, separate from the API, so that an ingestion or
retraining failure cannot take down request serving, and either can be
restarted or scaled independently.
"""

import asyncio
import logging
import signal
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from btcpred.config import Settings, get_settings
from btcpred.db.session import get_engine
from btcpred.scheduler.jobs import INGEST_JOB_ID, ingest_job

logger = logging.getLogger(__name__)


def build_scheduler(settings: Settings | None = None) -> AsyncIOScheduler:
    settings = settings or get_settings()
    scheduler = AsyncIOScheduler(timezone=UTC)

    scheduler.add_job(
        ingest_job,
        IntervalTrigger(minutes=settings.ingest_interval_minutes),
        id=INGEST_JOB_ID,
        name="ingest price bars",
        # A slow tick (a first-run backfill, say) must not overlap the next one:
        # concurrent runs would race on the same rows and duplicate requests.
        max_instances=1,
        # If ticks were missed, run once rather than replaying each one. Nothing
        # is lost, because sync_symbol resumes from the newest stored candle.
        coalesce=True,
        misfire_grace_time=settings.ingest_interval_minutes * 60,
        # Ingest immediately on boot instead of idling for a full interval.
        next_run_time=datetime.now(UTC),
    )
    return scheduler


def _install_signal_handlers(stop: asyncio.Event) -> None:
    """Stop cleanly on SIGTERM/SIGINT so `docker compose down` is not a kill."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows event loops do not implement add_signal_handler.
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))


async def run_forever(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    scheduler = build_scheduler(settings)
    stop = asyncio.Event()
    _install_signal_handlers(stop)

    scheduler.start()
    logger.info(
        "scheduler started: symbol=%s interval=%s poll=%dmin",
        settings.binance_symbol,
        settings.binance_interval,
        settings.ingest_interval_minutes,
    )

    try:
        await stop.wait()
    finally:
        logger.info("shutting down scheduler")
        # wait=True lets an in-flight ingest finish its transaction.
        scheduler.shutdown(wait=True)
        await get_engine().dispose()
        logger.info("scheduler stopped")
