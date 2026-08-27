import numpy as np
import pandas as pd
import pytest

from btcpred.features.builder import FEATURE_NAMES, WARMUP_BARS, InsufficientHistoryError
from btcpred.features.dataset import build_dataset
from btcpred.features.labels import LABEL_COLUMN, RETURN_COLUMN

from .test_features import make_bars


def test_dataset_carries_features_and_labels_with_no_gaps():
    data = build_dataset(make_bars(400))

    assert set(FEATURE_NAMES).issubset(data.columns)
    assert LABEL_COLUMN in data.columns
    assert RETURN_COLUMN in data.columns
    assert data[list(FEATURE_NAMES)].isna().sum().sum() == 0
    assert data[LABEL_COLUMN].isna().sum() == 0


def test_warmup_and_unlabelled_rows_are_both_dropped():
    bars = make_bars(400)
    data = build_dataset(bars)

    # Warmup at the front, one unlabelled bar at the back.
    assert len(data) == 400 - (WARMUP_BARS - 1) - 1
    assert data["open_time"].min() == bars["open_time"].iloc[WARMUP_BARS - 1]
    assert data["open_time"].max() < bars["open_time"].max()


def test_labels_align_to_the_right_features():
    """The join is on open_time, not position.

    Rows are dropped from both ends by different amounts, so positional
    alignment would silently pair each feature row with the wrong label.
    """
    bars = make_bars(400)
    data = build_dataset(bars)

    close = bars.set_index("open_time")["close"]
    for t in data["open_time"].iloc[[0, 50, len(data) - 1]]:
        row = data.loc[data["open_time"] == t].iloc[0]
        nxt = close.index[close.index.get_loc(t) + 1]
        assert row[LABEL_COLUMN] == int(close[nxt] > close[t])
        np.testing.assert_allclose(row[RETURN_COLUMN], np.log(close[nxt] / close[t]))


def test_short_history_is_rejected():
    with pytest.raises(InsufficientHistoryError):
        build_dataset(make_bars(WARMUP_BARS - 5))


def test_duplicate_bars_are_caught_not_silently_joined():
    """A one_to_one join guards against duplicated open_times producing a frame
    larger than the input."""
    bars = make_bars(300)
    doubled = pd.concat([bars, bars.iloc[[100]]]).sort_values("open_time").reset_index(drop=True)

    with pytest.raises(Exception, match="(?i)merge|unique|one_to_one|monotonic"):
        build_dataset(doubled)
