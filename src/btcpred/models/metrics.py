"""Evaluation metrics, per context.md §8.

Accuracy alone is close to meaningless on this task, so every evaluation
carries the baselines it must beat and an MCC that exposes a model which is
merely predicting the majority class. A near-zero MCC at 52% accuracy is the
honest tell that nothing is being learned.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar


@dataclass(frozen=True, slots=True)
class Evaluation:
    n: int
    accuracy: float
    mcc: float
    log_loss: float | None
    auc: float | None
    majority_baseline: float
    persistence_baseline: float
    random_baseline: float
    base_rate: float
    standard_error: float
    predicted_up_rate: float

    @property
    def edge_over_majority(self) -> float:
        return self.accuracy - self.majority_baseline

    @property
    def beats_majority(self) -> bool:
        """Whether the edge clears two standard errors.

        Deliberately strict: at n=168 the noise band is ±7.6pp, so an unqualified
        "we beat the baseline" is usually a statement about sample size.
        """
        return self.edge_over_majority > 2 * self.standard_error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_int_array(values: Any) -> np.ndarray:
    return np.asarray(values).astype(int).ravel()


def evaluate(
    y_true: Any,
    y_pred: Any,
    *,
    proba: Any | None = None,
    previous_direction: Any | None = None,
) -> Evaluation:
    """Score a set of directional calls against the baselines that matter.

    `previous_direction` supplies the persistence baseline (predict the same
    direction as the previous hour); omit it and that baseline reports NaN
    rather than silently substituting something else.
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot evaluate an empty prediction set")

    n = int(y_true.size)
    accuracy = float((y_true == y_pred).mean())
    base_rate = float(y_true.mean())

    # "Always up" is a real competitor, not a formality (§6).
    majority = float(max(base_rate, 1.0 - base_rate))

    if previous_direction is not None:
        persistence = float((y_true == _as_int_array(previous_direction)).mean())
    else:
        persistence = float("nan")

    ll = auc = None
    if proba is not None:
        proba = np.asarray(proba, dtype=float).ravel()
        # A single-class fold makes both metrics undefined; report None rather
        # than letting sklearn raise mid-run.
        if len(np.unique(y_true)) > 1:
            ll = float(log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15)))
            auc = float(roc_auc_score(y_true, proba))

    return Evaluation(
        n=n,
        accuracy=accuracy,
        mcc=float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        log_loss=ll,
        auc=auc,
        majority_baseline=majority,
        persistence_baseline=persistence,
        random_baseline=0.5,
        base_rate=base_rate,
        # SE at p=0.5, the §8 table: what any accuracy claim must clear.
        standard_error=float(np.sqrt(0.25 / n)),
        predicted_up_rate=float(y_pred.mean()),
    )


def mcnemar_test(y_true: Any, pred_a: Any, pred_b: Any) -> dict[str, float]:
    """Compare two classifiers on the same test set (§8).

    The correct test for paired binary outcomes: it looks only at the cases
    where the two models disagree, which is what stops 52.3% vs 51.6% being
    narrated as a real difference when it is noise.
    """
    y_true = _as_int_array(y_true)
    correct_a = _as_int_array(pred_a) == y_true
    correct_b = _as_int_array(pred_b) == y_true

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))

    both = int(np.sum(correct_a & correct_b))
    neither = int(np.sum(~correct_a & ~correct_b))
    table = [[both, only_a], [only_b, neither]]
    # exact=True below ~25 discordant pairs, where the chi-square approximation
    # is unreliable.
    result = mcnemar(table, exact=(only_a + only_b) < 25, correction=True)

    return {
        "a_only_correct": float(only_a),
        "b_only_correct": float(only_b),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }
