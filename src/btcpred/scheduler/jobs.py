"""Scheduled jobs.

Each job is defensive about its own failures: a long-running poller must survive
a bad Binance response or a brief database outage, because the next tick is the
recovery mechanism. Jobs therefore log and swallow, rather than propagate.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from btcpred.config import get_settings
from btcpred.db.session import session_scope
from btcpred.ingest.binance import interval_to_timedelta
from btcpred.ingest.service import SyncResult, run_sync
from btcpred.predict.repository import resolve_predictions
from btcpred.predict.service import PredictionResult, generate_predictions

logger = logging.getLogger(__name__)

TICK_JOB_ID = "lifecycle_tick"
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
    return TickResult(ingest=ingest, resolved=resolved, predictions=predictions)
