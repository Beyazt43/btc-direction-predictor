"""Walk-forward evaluation and model training, per context.md §8.

The three-way discipline matters more than any single metric: walk-forward folds
select hyperparameters, and the final chronological holdout is touched exactly
once, at the end. Nothing in this module reads the holdout — `evaluate_holdout`
is separate and must be called deliberately.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from btcpred.features.builder import FEATURE_NAMES
from btcpred.features.labels import LABEL_COLUMN
from btcpred.models.base import RETURN_SERIES_COLUMN, DirectionModel
from btcpred.models.gbt import GbtDirectionModel, sample_params
from btcpred.models.metrics import Evaluation, evaluate
from btcpred.models.splits import Fold, walk_forward_splits

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], DirectionModel]


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    model_name: str
    per_fold: list[Evaluation]
    pooled: Evaluation
    hyperparameters: dict[str, Any]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([e.accuracy for e in self.per_fold]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "pooled": self.pooled.to_dict(),
            "mean_fold_accuracy": self.mean_accuracy,
            "per_fold_accuracy": [e.accuracy for e in self.per_fold],
        }


def _previous_direction(frame: pd.DataFrame) -> np.ndarray:
    """Persistence baseline: repeat the previous hour's direction.

    Row i's own return is `ret_lag_1`, so its sign is the direction of the hour
    that just closed — exactly what a persistence forecaster would carry forward.
    """
    return (frame[RETURN_SERIES_COLUMN].to_numpy(dtype=float) > 0).astype(int)


def _fold_predictions(model: DirectionModel, frame: pd.DataFrame, fold: Fold) -> np.ndarray:
    """Predict a fold's test rows.

    The model is handed every row up to the end of the test block, not the test
    rows alone. ARIMA needs that sequential context to filter its state; row-wise
    models are unaffected. It is not leakage: one-step-ahead prediction at row i
    uses only observations strictly before i+1, all of which were genuinely
    available at that moment.
    """
    prefix = frame.iloc[: int(fold.test[-1]) + 1]
    return np.asarray(model.predict_proba(prefix))[fold.test]


def walk_forward_evaluate(
    factory: ModelFactory,
    frame: pd.DataFrame,
    folds: Sequence[Fold],
) -> WalkForwardReport:
    """Fit and score across folds, pooling predictions for the headline number.

    Pooling gives a single reference metric at the full test n, which matters:
    per-fold accuracies on a few hundred bars each carry noise bands wide enough
    to be uninterpretable on their own (§8's sample-size table).
    """
    per_fold: list[Evaluation] = []
    all_true: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    all_prev: list[np.ndarray] = []
    model: DirectionModel | None = None

    for fold in folds:
        model = factory()
        model.fit(frame.iloc[fold.train])

        proba = _fold_predictions(model, frame, fold)
        test = frame.iloc[fold.test]
        y_true = test[LABEL_COLUMN].to_numpy(dtype=int)
        prev = _previous_direction(test)

        per_fold.append(
            evaluate(y_true, (proba >= 0.5).astype(int), proba=proba, previous_direction=prev)
        )
        all_true.append(y_true)
        all_proba.append(proba)
        all_prev.append(prev)

    y_true = np.concatenate(all_true)
    proba = np.concatenate(all_proba)
    pooled = evaluate(
        y_true,
        (proba >= 0.5).astype(int),
        proba=proba,
        previous_direction=np.concatenate(all_prev),
    )

    if model is None:
        raise ValueError("no folds were supplied")

    return WalkForwardReport(
        model_name=model.name,
        per_fold=per_fold,
        pooled=pooled,
        hyperparameters=model.hyperparameters,
    )


def random_search(
    frame: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    n_configs: int = 30,
    seed: int = 0,
    objective: str = "log_loss",
) -> tuple[dict[str, Any], list[WalkForwardReport]]:
    """Select GBT hyperparameters on walk-forward folds only.

    Ranked on pooled log loss rather than accuracy: with accuracy hovering near
    the base rate, log loss discriminates between configurations that are
    confidently wrong and ones that are honestly uncertain, which is the
    difference that matters for §7's probability comparison.
    """
    rng = np.random.default_rng(seed)
    reports: list[WalkForwardReport] = []

    for i in range(n_configs):
        params = sample_params(rng)
        report = walk_forward_evaluate(lambda p=params: GbtDirectionModel(p), frame, folds)
        reports.append(report)
        logger.info(
            "config %d/%d: log_loss=%.5f acc=%.4f",
            i + 1,
            n_configs,
            report.pooled.log_loss if report.pooled.log_loss is not None else float("nan"),
            report.pooled.accuracy,
        )

    def score(report: WalkForwardReport) -> float:
        if objective == "accuracy":
            return -report.pooled.accuracy
        return report.pooled.log_loss if report.pooled.log_loss is not None else float("inf")

    best = min(reports, key=score)
    return best.hyperparameters, reports


def build_folds(n_dev: int, *, n_splits: int = 5, embargo: int = 1, **kwargs: Any) -> list[Fold]:
    """Folds over the development set, sized so each test block is meaningful.

    The series is divided into `n_splits + 2` blocks rather than `n_splits + 1`:
    the extra block means the first fold trains on two blocks instead of one,
    which both leaves room for the embargo and stops the earliest fold from
    fitting a regime-sensitive model on an implausibly short history.
    """
    test_size = kwargs.pop("test_size", None) or max(1, n_dev // (n_splits + 2))
    return walk_forward_splits(
        n_dev,
        n_splits=n_splits,
        test_size=test_size,
        embargo=embargo,
        min_train_size=kwargs.pop("min_train_size", test_size),
        **kwargs,
    )


def evaluate_holdout(model: DirectionModel, frame: pd.DataFrame, holdout: np.ndarray) -> Evaluation:
    """Score the final holdout. §8 allows this exactly once, at the very end.

    Kept out of every selection path on purpose: the moment a holdout result
    feeds back into a choice, it stops being a holdout.
    """
    prefix = frame.iloc[: int(holdout[-1]) + 1]
    proba = np.asarray(model.predict_proba(prefix))[holdout]
    test = frame.iloc[holdout]
    return evaluate(
        test[LABEL_COLUMN].to_numpy(dtype=int),
        (proba >= 0.5).astype(int),
        proba=proba,
        previous_direction=_previous_direction(test),
    )


def feature_columns() -> list[str]:
    return list(FEATURE_NAMES)
