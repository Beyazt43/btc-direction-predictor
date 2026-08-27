"""Joins features to labels to produce a training dataset.

Kept separate from both neighbours on purpose. `builder` computes features and
never looks forward; `labels` looks forward and computes nothing else. This
module is the only place the two meet, and it does so by joining on `open_time`
rather than by position, so a warmup row dropped at the front and an unlabelled
row dropped at the back cannot silently shift the alignment between them.
"""

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.features.builder import (
    FEATURE_NAMES,
    WARMUP_BARS,
    InsufficientHistoryError,
    compute_features,
    load_bars,
)
from btcpred.features.labels import LABEL_COLUMN, RETURN_COLUMN, add_labels


def build_dataset(bars: pd.DataFrame) -> pd.DataFrame:
    """Features joined to their labels, with incomplete rows removed.

    Two kinds of row cannot survive and both are dropped rather than imputed:
    the first `WARMUP_BARS` have no complete rolling windows, and the final bar
    has no label because the hour it would be labelled against has not closed.
    """
    if len(bars) < WARMUP_BARS:
        raise InsufficientHistoryError(f"need {WARMUP_BARS} bars, found {len(bars)}")

    features = compute_features(bars)
    labelled = add_labels(bars)

    merged = features.merge(
        labelled[["open_time", LABEL_COLUMN, RETURN_COLUMN]],
        on="open_time",
        how="inner",
        validate="one_to_one",
    )
    return merged.dropna(subset=list(FEATURE_NAMES)).reset_index(drop=True)


async def build_training_dataset(
    session: AsyncSession,
    *,
    symbol: str = "BTCUSDT",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load bars and return a model-ready frame of features plus labels."""
    bars = await load_bars(session, as_of_time=end, symbol=symbol, start=start)
    return build_dataset(bars)
