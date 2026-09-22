"""Scheduled jobs.

Each job is defensive about its own failures: a long-running poller must survive
a bad Binance response or a brief database outage, because the next tick is the
recovery mechanism. Jobs therefore log and swallow, rather than propagate.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from btcpred.backup.service import BackupInfo, create_backup, prune, verify_backup
from btcpred.config import get_settings
from btcpred.db.session import get_engine, session_scope
from btcpred.ingest.binance import interval_to_timedelta
from btcpred.ingest.service import SyncResult, run_sync
from btcpred.monitoring.drift import DriftCheck, check_model, record_check
from btcpred.predict.repository import resolve_predictions
from btcpred.predict.service import (
    MODEL_NAMES,
    PredictionResult,
    clear_model_cache,
    generate_predictions,
)

logger = logging.getLogger(__name__)

TICK_JOB_ID = "lifecycle_tick"
RETRAIN_JOB_ID = "daily_retrain"
DRIFT_JOB_ID = "daily_drift_check"
BACKUP_JOB_ID = "daily_backup"
# Kept for callers that reference the ingest job by its original id.
INGEST_JOB_ID = TICK_JOB_ID


@dataclass(frozen=True, slots=True)
class TickResult:
    ingest: SyncResult | None
    resolved: int
    predictions: list[PredictionResult]


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


async def resolve_job() -> int:
    """Fill in outcomes for predictions whose target hour has now closed."""
    settings = get_settings()
    try:
        async with session_scope() as session:
            return await resolve_predictions(
                session,
                symbol=settings.binance_symbol,
                interval=interval_to_timedelta(settings.binance_interval),
            )
    except Exception:
        logger.exception("resolution failed; will retry on next tick")
        return 0


async def predict_job() -> list[PredictionResult]:
    """Predict the next hour from the newest closed bar, for every live model."""
    settings = get_settings()
    try:
        async with session_scope() as session:
            return await generate_predictions(
                session,
                symbol=settings.binance_symbol,
                interval=interval_to_timedelta(settings.binance_interval),
                model_dir=Path(settings.model_dir),
            )
    except Exception:
        logger.exception("prediction failed; will retry on next tick")
        return []


async def tick_job() -> TickResult:
    """One pass of the §5 lifecycle: ingest, then resolve, then predict.

    The order is the leakage guard. Resolution runs before prediction so the
    hour that just closed is scored before the next one is called, and
    prediction runs last so it only ever sees bars that ingestion has
    confirmed closed.
    """
    ingest = await ingest_job()
    resolved = await resolve_job()
    predictions = await predict_job()
    _beat()
    return TickResult(ingest=ingest, resolved=resolved, predictions=predictions)


def _beat() -> None:
    """Record that a tick completed, for the container healthcheck."""
    try:
        Path(get_settings().heartbeat_path).touch()
    except OSError:
        logger.exception("could not write heartbeat")


async def retrain_job() -> int:
    """Run the daily retrain in a subprocess, returning its exit code.

    A subprocess rather than an in-process call for one reason: fitting is
    CPU-bound, and inside the event loop it would stall the two-minute tick for
    the better part of a minute. Reusing the CLI also means the scheduled path
    and the manual path are the same code, exit code included (2 means a
    version was trained but refused by the activation gate).
    """
    logger.info("starting daily retrain")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "btcpred.models",
        "train",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await proc.communicate()
    text = output.decode(errors="replace")

    if proc.returncode == 0:
        logger.info("retrain finished; new versions active\n%s", text)
    elif proc.returncode == 2:
        logger.warning("retrain finished but a version was refused by the gate\n%s", text)
    else:
        logger.error("retrain failed (exit %s)\n%s", proc.returncode, text)

    # Whatever happened, the next tick must consult the registry afresh rather
    # than serve a version that may just have been retired.
    clear_model_cache()
    return int(proc.returncode or 0)


async def drift_job() -> list[DriftCheck]:
    """Score each model's live 30-day window against its own live history.

    Records a row per model whatever the outcome, so the dashboard shows a
    continuous series rather than only the days something went wrong. An alert
    is logged at WARNING with the top-moving features attached; nothing is
    retrained in response, by decision (context.md §9 item 3).
    """
    settings = get_settings()
    interval = interval_to_timedelta(settings.binance_interval)
    checks: list[DriftCheck] = []
    for name in MODEL_NAMES:
        try:
            async with session_scope() as session:
                check = await check_model(
                    session, name, symbol=settings.binance_symbol, interval=interval
                )
                await record_check(session, check)
                checks.append(check)
        except Exception:
            logger.exception("drift check failed for %s; will retry tomorrow", name)
    return checks


async def backup_job() -> BackupInfo | None:
    """Write a verified backup of the irreplaceable tables, then prune old ones.

    The prediction record cannot be rebuilt, and after a host migration it lives
    on exactly one machine. This runs in-process rather than as a subprocess: it
    is a few seconds of I/O over a small dataset, not a CPU-bound fit.

    The new backup is verified before anything is pruned, so a corrupt write can
    never take the last good copy with it.
    """
    settings = get_settings()
    root = Path(settings.backup_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        async with get_engine().begin() as conn:
            info = await create_backup(conn, root)
        verify_backup(info.path)
        prune(root, settings.backup_keep)
    except Exception:
        logger.exception("backup failed; previous backups are untouched")
        return None

    logger.info("backup %s: %d rows, %.0f KB", info.name, info.total_rows, info.size_bytes / 1024)
    return info
