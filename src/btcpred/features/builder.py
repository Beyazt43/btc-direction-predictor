"""Feature construction — the single source of truth shared by training and serving.

Two rules hold this module together:

1. Every feature is scale-free (returns, ratios, z-scores). A raw price level
   would teach the model 2026 price levels and generalise to nothing.
2. Nothing here may look forward. There is deliberately no `shift(-1)` in this
   file; the only forward-looking operation in the codebase lives in
   `btcpred.features.labels`, where it belongs.

`compute_features` is pure and vectorised, and both entry points go through it:
serving takes the last row of a short window, training takes every row of a long
one. Sharing one implementation is what makes train/serve skew structurally
impossible rather than merely intended.
"""

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.db.tables import price_bars

RETURN_LAGS = 6
WINDOWS = (6, 24, 168)

# Longest rolling window, plus one bar because returns consume a close.
WARMUP_BARS = max(WINDOWS) + 1

FEATURE_NAMES: tuple[str, ...] = (
    *(f"ret_lag_{i}" for i in range(1, RETURN_LAGS + 1)),
    *(f"ret_mean_{w}" for w in WINDOWS),
    *(f"rv_{w}" for w in WINDOWS),
    "rv_ratio_6_168",
    *(f"vol_z_{w}" for w in WINDOWS),
    "trades_z_24",
    "range_pct",
    "body_ratio",
    "close_pos",
)

_BAR_COLUMNS = (
    price_bars.c.open_time,
    price_bars.c.open,
    price_bars.c.high,
    price_bars.c.low,
    price_bars.c.close,
    price_bars.c.volume,
    price_bars.c.num_trades,
)


class InsufficientHistoryError(ValueError):
    """Raised when there are too few bars to fill every rolling window.

    Deliberately an error rather than a row of NaNs: a silent NaN row is
    something a model will happily train on.
    """


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window)
    std = rolling.std()
    # A zero-variance window carries no information; call it neutral rather than
    # dividing by zero.
    return ((series - rolling.mean()) / std.replace(0.0, np.nan)).fillna(0.0)


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Vectorised feature computation over bars ordered oldest to newest.

    Pure: no I/O, no clock, no hidden state. Given the same bars it returns the
    same features, which is what makes the train/serve identity testable.
    """
    if not bars["open_time"].is_monotonic_increasing:
        raise ValueError("bars must be ordered by open_time ascending")

    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    open_ = bars["open"].astype(float)
    volume = bars["volume"].astype(float)
    trades = bars["num_trades"].astype(float)

    # Log returns: additive across time and symmetric between up and down moves.
    ret = np.log(close / close.shift(1))

    out = pd.DataFrame(index=bars.index)
    out["open_time"] = bars["open_time"]

    # ret_lag_1 is the most recently completed return, i.e. this bar's own.
    for i in range(1, RETURN_LAGS + 1):
        out[f"ret_lag_{i}"] = ret.shift(i - 1)

    for w in WINDOWS:
        out[f"ret_mean_{w}"] = ret.rolling(w).mean()
        out[f"rv_{w}"] = ret.rolling(w).std()
        out[f"vol_z_{w}"] = _zscore(volume, w)

    # Short vol against long vol: a scale-free read on the volatility regime.
    out["rv_ratio_6_168"] = out["rv_6"] / out["rv_168"].replace(0.0, np.nan)
    out["trades_z_24"] = _zscore(trades, 24)

    # Intra-bar shape, all normalised by the bar itself so nothing carries a
    # price level. A perfectly flat bar has no shape, so it reads as neutral.
    bar_range = (high - low).replace(0.0, np.nan)
    out["range_pct"] = ((high - low) / close).fillna(0.0)
    out["body_ratio"] = ((close - open_).abs() / bar_range).fillna(0.0)
    out["close_pos"] = ((close - low) / bar_range).fillna(0.5)

    return out


async def load_bars(
    session: AsyncSession,
    *,
    as_of_time: pd.Timestamp | None = None,
    symbol: str = "BTCUSDT",
    start: pd.Timestamp | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load bars, never revealing anything after `as_of_time`.

    This is the one place the cutoff is applied. Everything downstream inherits
    the guarantee instead of re-implementing it. Note that price_bars only ever
    contains closed candles: the ingestion layer refuses in-progress ones, so no
    additional close_time filter is needed here.
    """
    stmt = sa.select(*_BAR_COLUMNS).where(price_bars.c.symbol == symbol)
    if as_of_time is not None:
        stmt = stmt.where(price_bars.c.open_time <= as_of_time)
    if start is not None:
        stmt = stmt.where(price_bars.c.open_time >= start)

    if limit is not None:
        # Take the newest `limit` bars, then restore chronological order.
        stmt = stmt.order_by(price_bars.c.open_time.desc()).limit(limit)
        rows = (await session.execute(stmt)).fetchall()
        rows = list(reversed(rows))
    else:
        stmt = stmt.order_by(price_bars.c.open_time.asc())
        rows = (await session.execute(stmt)).fetchall()

    return pd.DataFrame(
        rows,
        columns=["open_time", "open", "high", "low", "close", "volume", "num_trades"],
    )


async def build_feature_window(
    session: AsyncSession,
    as_of_time: pd.Timestamp,
    *,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """The warmup window of features ending at `as_of_time`, for live inference.

    Sequence models (ARIMA) need the run-up to filter their state; row-wise
    models only need the final row. Both are served from this one frame, which
    is computed through the same function training uses, so what the live
    system sees is identical to what training would have produced.
    """
    bars = await load_bars(session, as_of_time=as_of_time, symbol=symbol, limit=WARMUP_BARS)
    if len(bars) < WARMUP_BARS:
        raise InsufficientHistoryError(
            f"need {WARMUP_BARS} bars as of {as_of_time}, found {len(bars)}"
        )
    if bars["open_time"].iloc[-1] != as_of_time:
        raise InsufficientHistoryError(
            f"no bar at {as_of_time}; newest available is {bars['open_time'].iloc[-1]}"
        )

    features = compute_features(bars)
    if features.iloc[-1][list(FEATURE_NAMES)].isna().any():
        raise InsufficientHistoryError(f"incomplete feature row at {as_of_time}")
    return features


async def build_feature_row(
    session: AsyncSession,
    as_of_time: pd.Timestamp,
    *,
    symbol: str = "BTCUSDT",
) -> pd.Series:
    """Features as of `as_of_time`, as a single row."""
    window = await build_feature_window(session, as_of_time, symbol=symbol)
    return window.iloc[-1]


async def build_training_frame(
    session: AsyncSession,
    *,
    symbol: str = "BTCUSDT",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Features for every bar in a range, for training and backtesting.

    Rows inside the warmup window are dropped rather than imputed: they are not
    rows the live system could ever have produced.
    """
    bars = await load_bars(session, as_of_time=end, symbol=symbol, start=start)
    if len(bars) < WARMUP_BARS:
        raise InsufficientHistoryError(f"need {WARMUP_BARS} bars, found {len(bars)}")

    features = compute_features(bars)
    return features.dropna(subset=list(FEATURE_NAMES)).reset_index(drop=True)
