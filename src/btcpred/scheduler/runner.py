"""APScheduler process: owns the recurring ingestion loop.

Runs as its own container, separate from the API, so that an ingestion or
retraining failure cannot take down request serving, and either can be
restarted or scaled independently.
"""

import asyncio
import logging
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path

import sqlalchemy as sa
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from btcpred.backup.service import list_backups
from btcpred.config import Settings, get_settings
from btcpred.db.session import get_engine, session_scope
from btcpred.db.tables import drift_checks, model_versions
from btcpred.scheduler.jobs import (
    BACKUP_JOB_ID,
    DRIFT_JOB_ID,
    RETRAIN_JOB_ID,
    TICK_JOB_ID,
    backup_job,
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

    # The grace period below only covers a job whose time passed while the
    # process was alive (a blocked event loop). It does NOT cover a cold start:
    # with an in-memory job store, a process started at 10:00 simply schedules
    # the next 02:00 and has no idea one was missed. On a machine that is off
    # overnight the daily jobs would therefore never run. `schedule_catch_up`
    # handles that case on boot by checking how stale each job's output is.
    scheduler.add_job(
        retrain_job,
        CronTrigger.from_crontab(settings.retrain_cron, timezone=UTC),
        id=RETRAIN_JOB_ID,
        name="daily retrain",
        max_instances=1,
        coalesce=True,
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

    scheduler.add_job(
        backup_job,
        CronTrigger.from_crontab(settings.backup_cron, timezone=UTC),
        id=BACKUP_JOB_ID,
        name="daily backup",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=12 * 60 * 60,
    )
    return scheduler


# A daily job whose last output is older than this is pulled forward on boot.
# Twenty hours rather than 24 so that a catch-up run at 09:30 today still
# counts as due at 09:00 tomorrow, keeping the cadence daily on a machine that
# is never on at the scheduled hour.
STALE_AFTER = timedelta(hours=20)
# Give the first tick time to catch up on bars before retraining on them.
CATCH_UP_DELAY = timedelta(minutes=2)


async def schedule_catch_up(
    scheduler: AsyncIOScheduler,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Pull forward any daily job whose last output is stale.

    The same self-healing principle as ingestion, applied to the daily jobs:
    the schedule is a clock, and what makes the system tolerate downtime is
    checking state on boot rather than trusting that the clock was running.
    Returns the ids of the jobs that were pulled forward.
    """
    now = now or datetime.now(UTC)
    settings = settings or get_settings()
    pulled: list[str] = []

    async with session_scope() as session:
        last_train = await session.scalar(sa.select(sa.func.max(model_versions.c.trained_at)))
        last_check = await session.scalar(sa.select(sa.func.max(drift_checks.c.checked_at)))

    def stale(ts: datetime | None) -> bool:
        return ts is None or now - ts > STALE_AFTER

    if stale(last_train):
        scheduler.modify_job(RETRAIN_JOB_ID, next_run_time=now + CATCH_UP_DELAY)
        pulled.append(RETRAIN_JOB_ID)
        logger.info("retrain is stale (last %s); running in %s", last_train, CATCH_UP_DELAY)
    if stale(last_check):
        # After the retrain, so the check reads the new version's quantiles.
        scheduler.modify_job(DRIFT_JOB_ID, next_run_time=now + CATCH_UP_DELAY * 3)
        pulled.append(DRIFT_JOB_ID)
        logger.info("drift check is stale (last %s); running in %s", last_check, CATCH_UP_DELAY * 3)

    # The backup's freshness lives on the filesystem rather than in a table, but
    # the reasoning is the same: a machine that is never on at 03:30 UTC would
    # otherwise never back up at all.
    if stale(_last_backup_at(settings)):
        scheduler.modify_job(BACKUP_JOB_ID, next_run_time=now + CATCH_UP_DELAY * 4)
        pulled.append(BACKUP_JOB_ID)
        logger.info("backup is stale; running in %s", CATCH_UP_DELAY * 4)
    return pulled


def _last_backup_at(settings: Settings) -> datetime | None:
    try:
        backups = list_backups(Path(settings.backup_dir))
    except OSError:
        return None
    if not backups:
        return None
    try:
        return datetime.fromisoformat(backups[0].manifest.created_at)
    except ValueError:
        return None


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
        "scheduler started: symbol=%s interval=%s poll=%dmin retrain='%s' drift='%s' backup='%s'",
        settings.binance_symbol,
        settings.binance_interval,
        settings.ingest_interval_minutes,
        settings.retrain_cron,
        settings.drift_cron,
        settings.backup_cron,
    )
    try:
        await schedule_catch_up(scheduler, settings=settings)
    except Exception:
        # The cron schedule still stands; only the boot-time catch-up is lost.
        logger.exception("could not check for stale daily jobs")

    try:
        await stop.wait()
    finally:
        logger.info("shutting down scheduler")
        # wait=True lets an in-flight ingest finish its transaction.
        scheduler.shutdown(wait=True)
        await get_engine().dispose()
        logger.info("scheduler stopped")
