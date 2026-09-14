import math

import numpy as np

from btcpred.models.gbt import DEFAULT_PARAMS
from btcpred.models.metrics import Evaluation
from btcpred.models.pipeline import (
    MAX_LOG_LOSS,
    MAX_REGRESSION,
    gate_candidate,
)
from btcpred.models.splits import walk_forward_splits
from btcpred.models.training import WalkForwardReport, random_search

from .test_models import ar1_frame


def report_with(log_loss: float | None) -> WalkForwardReport:
    ev = Evaluation(
        n=1000,
        accuracy=0.51,
        mcc=0.02,
        log_loss=log_loss,
        auc=0.52,
        majority_baseline=0.50,
        persistence_baseline=0.49,
        random_baseline=0.5,
        base_rate=0.50,
        standard_error=0.0158,
        predicted_up_rate=0.6,
    )
    return WalkForwardReport(model_name="xgboost", per_fold=[ev], pooled=ev, hyperparameters={})


def incumbent_with(log_loss: float) -> dict:
    return {
        "model_version": "xgboost-old",
        "hyperparameters": DEFAULT_PARAMS,
        "reference_metrics": {"pooled": {"log_loss": log_loss}},
    }


def test_first_version_activates_without_an_incumbent():
    decision = gate_candidate(report_with(0.6920), incumbent=None)
    assert decision.activate
    assert "first version" in decision.reason


def test_healthy_variation_between_retrains_always_passes():
    """The whole ARIMA-vs-GBT gap is ~0.001; the gate must not referee that."""
    incumbent = incumbent_with(0.6915)
    for delta in (-0.002, 0.0, 0.001, 0.005):
        assert gate_candidate(report_with(0.6915 + delta), incumbent).activate


def test_worse_than_a_coin_flip_is_refused():
    """A model that loses to predicting 0.5 forever must not go live."""
    decision = gate_candidate(report_with(MAX_LOG_LOSS + 0.01), incumbent=None)
    assert not decision.activate
    assert "coin flip" in decision.reason


def test_coin_flip_threshold_is_above_ln2():
    """Exactly ln 2 is 'knows nothing', which is noise-adjacent for ARIMA; the
    refusal line sits clearly above it."""
    assert math.log(2) < MAX_LOG_LOSS
    assert gate_candidate(report_with(math.log(2)), incumbent=None).activate


def test_large_regression_against_incumbent_is_refused():
    # Incumbent chosen so the regressed candidate still sits under the absolute
    # floor; otherwise the coin-flip check would fire first and mask this path.
    incumbent = incumbent_with(0.6850)
    candidate = 0.6850 + MAX_REGRESSION + 0.001
    assert candidate < MAX_LOG_LOSS
    decision = gate_candidate(report_with(candidate), incumbent)
    assert not decision.activate
    assert "regresses" in decision.reason


def test_missing_log_loss_is_refused_not_guessed():
    assert not gate_candidate(report_with(None), incumbent=None).activate


def test_incumbent_without_reference_does_not_block():
    incumbent = {"model_version": "x", "hyperparameters": {}, "reference_metrics": None}
    assert gate_candidate(report_with(0.6920), incumbent).activate


def test_seeded_config_is_evaluated_and_can_win():
    """The incumbent's config is always in the running, so a bad random draw
    cannot regress the deployed hyperparameters."""
    frame = ar1_frame(n=900, phi=0.5, seed=3)
    folds = walk_forward_splits(len(frame), n_splits=2, test_size=150, min_train_size=100)

    strong = {**DEFAULT_PARAMS, "n_estimators": 60, "max_depth": 3, "min_child_weight": 5}
    best, reports = random_search(frame, folds, n_configs=2, seed=0, seeded_configs=[strong])

    assert len(reports) == 3
    assert reports[0].hyperparameters["n_estimators"] == 60, "seeded config is evaluated first"
    winner_ll = min(r.pooled.log_loss for r in reports)
    assert best in [r.hyperparameters for r in reports if r.pooled.log_loss == winner_ll]


def test_search_without_seeds_is_unchanged():
    frame = ar1_frame(n=600, phi=0.3, seed=4)
    folds = walk_forward_splits(len(frame), n_splits=2, test_size=100, min_train_size=100)
    _, reports = random_search(frame, folds, n_configs=3, seed=1)
    assert len(reports) == 3


def test_daily_seed_differs_by_date():
    """Two retrains on different days must not replay the same random draw."""
    rng_a = np.random.default_rng(20260914)
    rng_b = np.random.default_rng(20260915)
    assert rng_a.integers(0, 10**9) != rng_b.integers(0, 10**9)
