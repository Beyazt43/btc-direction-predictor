import numpy as np
import pytest

from btcpred.models.metrics import evaluate, mcnemar_test


def test_perfect_and_inverted_predictions():
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0])

    assert evaluate(y, y).accuracy == 1.0
    assert evaluate(y, y).mcc == pytest.approx(1.0)
    assert evaluate(y, 1 - y).accuracy == 0.0
    assert evaluate(y, 1 - y).mcc == pytest.approx(-1.0)


def test_always_up_scores_the_base_rate_but_zero_mcc():
    """§8's point: MCC is what exposes a model that only predicts the majority.

    A constant predictor can look respectable on accuracy alone; MCC calls it.
    """
    y = np.array([1] * 52 + [0] * 48)
    result = evaluate(y, np.ones_like(y))

    assert result.accuracy == pytest.approx(0.52)
    assert result.majority_baseline == pytest.approx(0.52)
    assert result.mcc == 0.0
    assert result.edge_over_majority == pytest.approx(0.0)
    assert not result.beats_majority


def test_majority_baseline_uses_the_larger_class():
    mostly_down = np.array([0] * 70 + [1] * 30)
    assert evaluate(mostly_down, np.zeros_like(mostly_down)).majority_baseline == pytest.approx(0.7)


def test_standard_error_matches_the_sample_size_table():
    """§8's table: one week of hourly bars carries a +/-7.6pp band."""
    y = np.random.default_rng(0).integers(0, 2, 168)
    se = evaluate(y, y).standard_error

    assert se == pytest.approx(np.sqrt(0.25 / 168))
    assert 2 * se == pytest.approx(0.0772, abs=0.001)


def test_beats_majority_requires_clearing_the_noise_band():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 168)

    # A one-point edge on a week of data is not a result.
    pred = y.copy()
    flip = rng.choice(len(y), size=int(len(y) * 0.48), replace=False)
    pred[flip] = 1 - pred[flip]
    assert not evaluate(y, pred).beats_majority


def test_persistence_baseline_is_reported_only_when_supplied():
    y = np.array([1, 0, 1, 0])

    assert np.isnan(evaluate(y, y).persistence_baseline)
    assert evaluate(y, y, previous_direction=y).persistence_baseline == pytest.approx(1.0)


def test_probability_metrics_need_both_classes():
    y = np.ones(10, dtype=int)
    result = evaluate(y, y, proba=np.full(10, 0.9))

    assert result.log_loss is None
    assert result.auc is None


def test_confident_and_wrong_costs_more_log_loss_than_honestly_unsure():
    """Why log loss ranks the hyperparameter search rather than accuracy."""
    y = np.array([1, 0, 1, 0])
    unsure = evaluate(y, np.ones_like(y), proba=np.full(4, 0.51))
    confident = evaluate(y, np.ones_like(y), proba=np.array([0.99, 0.99, 0.99, 0.99]))

    assert confident.log_loss > unsure.log_loss


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        evaluate([], [])


def test_mcnemar_sees_no_difference_between_identical_models():
    y = np.random.default_rng(2).integers(0, 2, 200)
    result = mcnemar_test(y, y, y)

    assert result["a_only_correct"] == 0
    assert result["b_only_correct"] == 0
    assert result["p_value"] == pytest.approx(1.0)


def test_mcnemar_detects_a_genuinely_better_model():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    good = y.copy()
    bad = y.copy()
    flip = rng.choice(len(y), size=120, replace=False)
    bad[flip] = 1 - bad[flip]

    result = mcnemar_test(y, good, bad)

    assert result["a_only_correct"] == 120
    assert result["p_value"] < 0.001


def test_mcnemar_does_not_call_a_narrow_gap_significant():
    """The 52.3% vs 51.6% case §8 warns about."""
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 1000)
    a = y.copy()
    b = y.copy()
    a[rng.choice(1000, 20, replace=False)] ^= 1
    b[rng.choice(1000, 23, replace=False)] ^= 1

    assert mcnemar_test(y, a, b)["p_value"] > 0.05
