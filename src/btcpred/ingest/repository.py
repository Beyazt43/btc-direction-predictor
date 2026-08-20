"""Persistence for ingested price bars."""

import logging
from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.db.tables import price_bars
from btcpred.ingest.binance import Kline

logger = logging.getLogger(__name__)

# Columns carrying candle data; a change in any of them is a real correction.
_DATA_COLUMNS = (
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
)


def _as_row(kline: Kline, symbol: str, source: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "source": source,
        "open_time": kline.open_time,
        "close_time": kline.close_time,
        "open": kline.open,
        "high": kline.high,
        "low": kline.low,
        "close": kline.close,
        "volume": kline.volume,
        "quote_volume": kline.quote_volume,
        "num_trades": kline.num_trades,
    }


async def upsert_price_bars(
    session: AsyncSession,
    klines: Sequence[Kline],
    *,
    symbol: str,
    source: str = "binance",
) -> int:
    """Insert candles idempotently, returning the number of rows actually written.

    Re-ingesting an unchanged candle is a true no-op: the UPDATE is guarded by
    IS DISTINCT FROM, so repeated polling neither rewrites the row nor bumps
    ingested_at, which keeps that column meaningful as an audit trail of when a
    value was genuinely first seen or corrected.
    """
    if not klines:
        return 0

    rows = [_as_row(k, symbol, source) for k in klines]
    stmt = insert(price_bars).values(rows)

    changed = sa.tuple_(*(price_bars.c[name] for name in _DATA_COLUMNS)).is_distinct_from(
        sa.tuple_(*(stmt.excluded[name] for name in _DATA_COLUMNS))
    )

    stmt = stmt.on_conflict_do_update(
        constraint="uq_price_bars_symbol_open_time",
        set_={name: stmt.excluded[name] for name in _DATA_COLUMNS}
        | {"ingested_at": sa.func.now(), "source": stmt.excluded.source},
        where=changed,
    ).returning(price_bars.c.open_time)

    result = await session.execute(stmt)
    written = len(result.fetchall())
    logger.debug("upserted %d/%d candles for %s", written, len(rows), symbol)
    return written


async def latest_bar_open_time(session: AsyncSession, symbol: str) -> datetime | None:
    """Open time of the most recent stored candle, or None when empty."""
    stmt = sa.select(sa.func.max(price_bars.c.open_time)).where(price_bars.c.symbol == symbol)
    return await session.scalar(stmt)


async def count_bars(session: AsyncSession, symbol: str) -> int:
    stmt = sa.select(sa.func.count()).select_from(price_bars).where(price_bars.c.symbol == symbol)
    return await session.scalar(stmt) or 0
