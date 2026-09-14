import numpy as np
import pandas as pd
import pytest

from btcpred.models.splits import split_holdout, split_holdout_by_time, walk_forward_splits


def test_test_blocks_are_contiguous_and_chronological():
    folds = walk_forward_splits(1000, n_splits=4, test_size=100, min_train_size=10)

    assert len(folds) == 4
    # Deliberately not strict: folds[1:] is one shorter, which is the point.
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test[-1] + 1 == b.test[0], "test blocks must tile without gaps"
    assert folds[-1].test[-1] == 999, "folds must reach the most recent sample"


def test_training_never_overlaps_or_follows_testing():
    """The property that random splits would violate."""
    folds = walk_forward_splits(1000, n_splits=4, test_size=100, min_train_size=10)

    for fold in folds:
        assert fold.train.max() < fold.test.min()
        assert not set(fold.train.tolist()) & set(fold.test.tolist())


def test_embargo_leaves_a_real_gap():
    """§8: the last training label reaches into close(t+1), so a bar must be dropped.

    With embargo=1 the final training row's label depends on the bar immediately
    after it, which is the embargoed bar -- never a test bar.
    """
    for embargo in (1, 5):
        folds = walk_forward_splits(
            1000, n_splits=3, test_size=100, embargo=embargo, min_train_size=10
        )
        for fold in folds:
            assert fold.gap() == embargo


def test_zero_embargo_touches_the_seam():
    """Demonstrates what the embargo prevents, so the guard is not vacuous."""
    folds = walk_forward_splits(1000, n_splits=3, test_size=100, embargo=0, min_train_size=10)
    assert all(fold.gap() == 0 for fold in folds)


def test_expanding_window_grows():
    folds = walk_forward_splits(1000, n_splits=4, test_size=100, min_train_size=10)

    sizes = [f.train_size for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]
    assert all(f.train[0] == 0 for f in folds), "expanding folds start at the beginning"


def test_sliding_window_is_fixed_length_and_moves():
    """§8's ablation: does old data help or hurt?"""
    folds = walk_forward_splits(
        1000, n_splits=4, test_size=100, window="sliding", sliding_size=200, min_train_size=10
    )

    assert {f.train_size for f in folds} == {200}
    starts = [f.train[0] for f in folds]
    assert starts == sorted(starts)
    assert starts[0] < starts[-1], "the window must actually slide"


def test_sliding_requires_a_size():
    with pytest.raises(ValueError, match="sliding_size"):
        walk_forward_splits(1000, n_splits=2, test_size=100, window="sliding")


def test_refuses_when_history_is_too_short():
    with pytest.raises(ValueError, match="not enough history"):
        walk_forward_splits(100, n_splits=5, test_size=30, min_train_size=50)


def test_holdout_is_the_most_recent_block_and_is_embargoed():
    dev, holdout = split_holdout(1000, holdout_size=200, embargo=1)

    assert holdout[0] == 800
    assert holdout[-1] == 999
    assert len(holdout) == 200
    assert dev[-1] == 798, "one bar embargoed between dev and holdout"
    assert not set(dev.tolist()) & set(holdout.tolist())


def test_holdout_rejects_impossible_sizes():
    with pytest.raises(ValueError, match="smaller than"):
        split_holdout(100, holdout_size=100)


def test_folds_over_dev_never_reach_the_holdout():
    """The discipline that keeps the holdout untouched until the end."""
    dev, holdout = split_holdout(2000, holdout_size=400)
    folds = walk_forward_splits(len(dev), n_splits=4, test_size=100, min_train_size=10)

    highest_seen = max(int(np.concatenate([f.train, f.test]).max()) for f in folds)
    assert highest_seen < holdout[0]


def _hourly(n: int, start: str = "2026-06-01") -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq="h", tz="UTC"))


def test_frozen_holdout_is_pinned_to_the_dates():
    times = _hourly(24 * 120)  # June through September
    start = pd.Timestamp("2026-07-16", tz="UTC")
    end = pd.Timestamp("2026-09-14", tz="UTC")

    dev, holdout = split_holdout_by_time(times, start, end)

    assert times[holdout[0]] == start
    assert times[holdout[-1]] == end - pd.Timedelta(hours=1)
    assert len(holdout) == 60 * 24


def test_frozen_holdout_does_not_move_when_data_accrues():
    """The property a trailing window lacks: more data must not shift the window."""
    start = pd.Timestamp("2026-07-16", tz="UTC")
    end = pd.Timestamp("2026-09-14", tz="UTC")

    _, before = split_holdout_by_time(_hourly(24 * 110), start, end)
    _, after = split_holdout_by_time(_hourly(24 * 200), start, end)

    assert len(before) == len(after) == 60 * 24
    assert before[0] == after[0]


def test_frozen_holdout_embargoes_the_bar_before_it():
    """The last development row's label reaches into the first holdout bar."""
    times = _hourly(24 * 120)
    start = pd.Timestamp("2026-07-16", tz="UTC")
    end = pd.Timestamp("2026-09-14", tz="UTC")

    dev, holdout = split_holdout_by_time(times, start, end, embargo=1)

    assert times[dev[-1]] == start - pd.Timedelta(hours=2)
    assert holdout[0] - dev[-1] == 2, "exactly one embargoed bar between them"


def test_frozen_holdout_needs_data_on_both_sides():
    start = pd.Timestamp("2026-07-16", tz="UTC")
    end = pd.Timestamp("2026-09-14", tz="UTC")

    with pytest.raises(ValueError, match="no bars"):
        split_holdout_by_time(_hourly(24 * 10, "2026-05-01"), start, end)
    with pytest.raises(ValueError, match="no development"):
        split_holdout_by_time(_hourly(24 * 10, "2026-07-16"), start, end)
