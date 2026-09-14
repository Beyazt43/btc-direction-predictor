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
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from btcpred.config import Settings, get_settings
from btcpred.db.session import get_engine
from btcpred.scheduler.jobs import (
    DRIFT_JOB_ID,
    RETRAIN_JOB_ID,
    TICK_JOB_ID,
    drift_job,
    retrain_job,
    tick_job,
)

logger = logging.getLogger(__name__)


def build_scheduler(settings: Settings | None = None) -> AsyncIOScheduler:
    settings = settings or get_settings()
    scheduler = AsyncIOScheduler(timezone=UTC)

    scheduler.add_job(
        tick_job,
        IntervalTrigger(minutes=settings.ingest_interval_minutes),
        id=TICK_JOB_ID,
        name="lifecycle tick (ingest, resolve, predict)",
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

    scheduler.add_job(
        retrain_job,
        CronTrigger.from_crontab(settings.retrain_cron, timezone=UTC),
        id=RETRAIN_JOB_ID,
        name="daily retrain",
        max_instances=1,
        coalesce=True,
        # A retrain missed because the box was down at 02:00 should run when
        # it comes back, not wait until tomorrow: a stale model costs more than
        # a late retrain. Twelve hours covers any plausible outage without
        # letting a run land on top of the next scheduled one.
        misfire_grace_time=12 * 60 * 60,
    )

    scheduler.add_job(
        drift_job,
        CronTrigger.from_crontab(settings.drift_cron, timezone=UTC),
        id=DRIFT_JOB_ID,
        name="daily drift check",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=12 * 60 * 60,
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
        "scheduler started: symbol=%s interval=%s poll=%dmin retrain='%s' drift='%s'",
        settings.binance_symbol,
        settings.binance_interval,
        settings.ingest_interval_minutes,
        settings.retrain_cron,
        settings.drift_cron,
    )

    try:
        await stop.wait()
    finally:
        logger.info("shutting down scheduler")
        # wait=True lets an in-flight ingest finish its transaction.
        scheduler.shutdown(wait=True)
        await get_engine().dispose()
        logger.info("scheduler stopped")
