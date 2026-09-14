"""Persistence for the prediction lifecycle (context.md §5)."""

import logging
from datetime import datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.db.tables import predictions, price_bars

logger = logging.getLogger(__name__)


def is_live(interval: timedelta) -> sa.ColumnElement[bool]:
    """The one definition of an honest live prediction.

    The outcome for `target_open_time` becomes knowable when that bar closes,
    at `target_open_time + interval`. A prediction written after that was made
    with the answer visible. Every query that separates live from
    back-generated predictions must use this expression rather than restating
    the comparison, so the two can never drift apart.
    """
    return predictions.c.predicted_at < predictions.c.target_open_time + interval


async def prediction_exists(
    session: AsyncSession,
    *,
    model_name: str,
    model_version: str,
    target_open_time: datetime,
) -> bool:
    stmt = sa.select(sa.literal(True)).where(
        predictions.c.model_name == model_name,
        predictions.c.model_version == model_version,
        predictions.c.target_open_time == target_open_time,
    )
    return (await session.scalar(stmt)) is not None


async def insert_prediction(
    session: AsyncSession,
    *,
    model_name: str,
    model_version: str,
    target_open_time: datetime,
    predicted_direction: int,
    predicted_proba: float,
) -> bool:
    """Log a prediction, returning False if one already exists for this target.

    `predicted_at` is deliberately left to the database default so the app can
    never stamp a prediction with a time other than when it was actually
    written -- that column is what §5's liveness rule reads.
    """
    stmt = (
        insert(predictions)
        .values(
            model_name=model_name,
            model_version=model_version,
            target_open_time=target_open_time,
            predicted_direction=predicted_direction,
            predicted_proba=predicted_proba,
        )
        .on_conflict_do_nothing(constraint="uq_predictions_model_version_target")
        .returning(predictions.c.id)
    )
    return (await session.scalar(stmt)) is not None


async def resolve_predictions(session: AsyncSession, *, symbol: str, interval: timedelta) -> int:
    """Fill in outcomes for every prediction whose target bar has now closed.

    One statement resolves everything pending, which makes it both cheap to
    run every tick and self-healing: a resolution missed during downtime is
    picked up on the next run with no special casing.

    A prediction for `target_open_time = T` is about whether close(T) exceeds
    close(T - interval), matching the label definition in §5.
    """
    bar = price_bars.alias("bar")
    prev = price_bars.alias("prev")

    stmt = (
        sa.update(predictions)
        .values(
            # Postgres will not cast boolean directly to smallint; CASE is explicit.
            actual_direction=sa.case((bar.c.close > prev.c.close, 1), else_=0),
            actual_log_return=sa.func.ln(bar.c.close / prev.c.close),
            resolved_at=sa.func.now(),
        )
        .where(
            predictions.c.actual_direction.is_(None),
            bar.c.symbol == symbol,
            bar.c.open_time == predictions.c.target_open_time,
            prev.c.symbol == symbol,
            prev.c.open_time == bar.c.open_time - interval,
        )
    )
    result = await session.execute(stmt)
    resolved = int(result.rowcount or 0)
    if resolved:
        logger.info("resolved %d prediction(s)", resolved)
    return resolved


async def pending_count(session: AsyncSession) -> int:
    stmt = (
        sa.select(sa.func.count())
        .select_from(predictions)
        .where(predictions.c.actual_direction.is_(None))
    )
    return int(await session.scalar(stmt) or 0)
