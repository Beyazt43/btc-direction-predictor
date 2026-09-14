"""Walk-forward splitting with purging, per context.md §8.

No random splits, no shuffled k-fold, ever. Test data must always lie strictly
in the future of training data, and the boundary between them needs an embargo:
the label for bar `t` depends on `close(t+1)`, so a training set ending exactly
where the test set begins has already leaked its last label across the seam.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

WindowMode = Literal["expanding", "sliding"]


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split, as positional indices into a chronological frame."""

    index: int
    train: np.ndarray
    test: np.ndarray

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)

    def gap(self) -> int:
        """Bars sitting between the end of training and the start of testing."""
        return int(self.test[0] - self.train[-1] - 1)


def walk_forward_splits(
    n_samples: int,
    *,
    n_splits: int = 5,
    test_size: int,
    embargo: int = 1,
    min_train_size: int = 1,
    window: WindowMode = "expanding",
    sliding_size: int | None = None,
) -> list[Fold]:
    """Build contiguous, chronologically ordered folds ending at the last sample.

    `embargo` bars are dropped from the *end of training*, not inserted before
    the test set, so no observation is silently skipped: the embargoed bars are
    simply unusable for training against this particular test block.

    `window="sliding"` implements §8's fixed-window ablation — does old data help
    or hurt? — and is the same splitter with a truncated training set.
    """
    if test_size < 1:
        raise ValueError("test_size must be at least 1")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if embargo < 0:
        raise ValueError("embargo must not be negative")
    if window == "sliding" and not sliding_size:
        raise ValueError("sliding_size is required when window='sliding'")

    span = n_splits * test_size
    first_test_start = n_samples - span
    if first_test_start - embargo < min_train_size:
        raise ValueError(
            f"not enough history: {n_samples} samples cannot provide {n_splits} folds "
            f"of {test_size} with embargo {embargo} and min_train_size {min_train_size}"
        )

    folds: list[Fold] = []
    for i in range(n_splits):
        test_start = first_test_start + i * test_size
        train_end = test_start - embargo  # exclusive

        train_start = 0 if window == "expanding" else max(0, train_end - int(sliding_size))

        folds.append(
            Fold(
                index=i,
                train=np.arange(train_start, train_end),
                test=np.arange(test_start, test_start + test_size),
            )
        )
    return folds


def split_holdout(
    n_samples: int, holdout_size: int, embargo: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Carve off the final chronological holdout (§8).

    Kept deliberately separate from `walk_forward_splits` so that touching the
    holdout is an explicit act. §8 allows it exactly once, at the very end;
    folding it into the fold generator would make that discipline easy to lose.
    """
    if holdout_size >= n_samples:
        raise ValueError("holdout_size must be smaller than the sample count")

    holdout_start = n_samples - holdout_size
    dev_end = holdout_start - embargo
    if dev_end < 1:
        raise ValueError("holdout leaves no development data")
    return np.arange(0, dev_end), np.arange(holdout_start, n_samples)


def describe(folds: Sequence[Fold]) -> str:
    lines = []
    for f in folds:
        lines.append(
            f"fold {f.index}: train[{f.train[0]}:{f.train[-1] + 1}] "
            f"({f.train_size}) gap={f.gap()} test[{f.test[0]}:{f.test[-1] + 1}] ({f.test_size})"
        )
    return "\n".join(lines)


def split_holdout_by_time(
    open_times: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    embargo: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve off a holdout pinned to a date range rather than a trailing count.

    A trailing window drifts forward as data accrues and, once production
    trains on everything, becomes data the search has already tuned on. A
    fixed [start, end) window cannot move, which is what makes "touched once"
    a property that survives a living system.

    Development data ends `embargo` bars before `start`: the last development
    row's label depends on the bar after it, which would otherwise be the first
    holdout bar.
    """
    times = pd.to_datetime(open_times, utc=True).reset_index(drop=True)
    dev = np.flatnonzero(times < start)
    holdout = np.flatnonzero((times >= start) & (times < end))

    if len(holdout) == 0:
        raise ValueError(f"no bars fall inside the holdout window [{start}, {end})")
    if embargo:
        dev = dev[:-embargo] if len(dev) > embargo else dev[:0]
    if len(dev) == 0:
        raise ValueError("holdout window leaves no development data before it")
    return dev, holdout
