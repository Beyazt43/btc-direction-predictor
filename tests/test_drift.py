import math

import numpy as np
import pytest

from btcpred.monitoring.drift import (
    MIN_OBSERVED,
    MIN_REFERENCE,
    Z_ALERT,
    Sample,
    classify,
    population_stability_index,
    two_proportion_z,
)


def sample(n: int, accuracy: float, up_rate: float = 0.5) -> Sample:
    return Sample(n=n, correct=round(n * accuracy), ups=round(n * up_rate))


# --- the statistic -----------------------------------------------------------


def test_z_is_zero_when_nothing_changed():
    # Sizes chosen so 0.52 is an exact count on both sides.
    assert two_proportion_z(sample(500, 0.52), sample(2500, 0.52)) == pytest.approx(0.0, abs=1e-9)


def test_z_is_negative_when_observed_is_worse():
    assert two_proportion_z(sample(720, 0.48), sample(3000, 0.52)) < 0


def test_z_matches_the_section_8_table():
    """§8: at n=720 and p≈0.5 the SE is ~1.9pp, so a 3-point drop is roughly
    1.5σ against a large reference -- not detectable at 2σ. That table is the
    reason the alert threshold is what it is."""
    z = two_proportion_z(sample(720, 0.49), sample(20000, 0.52))
    assert -2.0 < z < -1.0


def test_z_reaches_alert_for_a_drop_the_window_can_actually_see():
    z = two_proportion_z(sample(720, 0.47), sample(20000, 0.52))
    assert z < Z_ALERT


def test_z_undefined_without_data():
    assert two_proportion_z(sample(0, 0.5), sample(100, 0.5)) is None
    assert two_proportion_z(sample(100, 0.5), sample(0, 0.5)) is None


# --- classification ----------------------------------------------------------


def test_short_window_is_insufficient_not_ok():
    status, _ = classify(sample(MIN_OBSERVED - 1, 0.9), sample(5000, 0.5), z=None)
    assert status == "insufficient"


def test_warming_up_until_the_reference_is_a_full_window():
    """§8: the reference must carry at least as much data as the window."""
    status, reason = classify(sample(720, 0.52), sample(MIN_REFERENCE - 1, 0.52), z=0.0)
    assert status == "warming_up"
    assert str(MIN_REFERENCE) in reason


def test_sanity_floor_fires_even_while_warming_up():
    """A model losing to 'always up' on its own window is an alert from week one."""
    obs = Sample(n=300, correct=120, ups=180)  # acc 0.40, majority 0.60
    status, reason = classify(obs, sample(0, 0.5), z=None)
    assert status == "alert"
    assert "majority" in reason


def test_ok_when_inside_the_band():
    status, _ = classify(sample(720, 0.515), sample(3000, 0.52), z=-0.3)
    assert status == "ok"


def test_alert_when_two_sigma_below_reference():
    status, reason = classify(sample(720, 0.47), sample(3000, 0.52), z=-2.4)
    assert status == "alert"
    assert "z=-2.40" in reason


def test_improvement_is_never_an_alert():
    """One-sided by design: only a drop is drift."""
    status, _ = classify(sample(720, 0.58), sample(3000, 0.52), z=+3.5)
    assert status == "ok"


# --- PSI ----------------------------------------------------------------------


def test_psi_is_near_zero_for_the_training_distribution():
    rng = np.random.default_rng(0)
    train = rng.normal(size=20000)
    edges = list(np.quantile(train, [i / 10 for i in range(1, 10)]))

    assert population_stability_index(rng.normal(size=5000), edges) < 0.02


def test_psi_grows_with_a_shift():
    rng = np.random.default_rng(1)
    train = rng.normal(size=20000)
    edges = list(np.quantile(train, [i / 10 for i in range(1, 10)]))

    small = population_stability_index(rng.normal(0.3, 1.0, 5000), edges)
    large = population_stability_index(rng.normal(1.5, 1.0, 5000), edges)

    assert small < large
    assert large > 0.2, "a 1.5σ mean shift is conventionally 'significant'"


def test_psi_ignores_non_finite_values_and_handles_empty():
    edges = [float(i) for i in range(1, 10)]
    assert math.isnan(population_stability_index(np.array([]), edges))
    with_nan = population_stability_index(np.array([np.nan, 5.0, np.inf, 5.0]), edges)
    assert math.isfinite(with_nan)
