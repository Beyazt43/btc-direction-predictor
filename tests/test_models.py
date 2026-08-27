import numpy as np
import pandas as pd
import pytest

from btcpred.features.builder import FEATURE_NAMES
from btcpred.features.labels import LABEL_COLUMN
from btcpred.models.arima import ArimaDirectionModel
from btcpred.models.gbt import GbtDirectionModel, sample_params
from btcpred.models.splits import walk_forward_splits
from btcpred.models.training import walk_forward_evaluate


def ar1_frame(n: int = 1500, phi: float = 0.0, seed: int = 5) -> pd.DataFrame:
    """A frame whose next-hour direction is predictable exactly when phi != 0."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, 0.01)

    frame = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES})
    frame["ret_lag_1"] = r
    # Label of row i is the direction of r[i+1]; the last row is unlabelled.
    frame[LABEL_COLUMN] = np.append((r[1:] > 0).astype(int), 0)
    return frame.iloc[:-1].reset_index(drop=True)


def test_arima_recovers_a_known_signal():
    """Sanity floor: the model must work when structure genuinely exists."""
    frame = ar1_frame(phi=0.6)
    model = ArimaDirectionModel().fit(frame)

    acc = (model.predict(frame) == frame[LABEL_COLUMN].to_numpy()).mean()
    assert acc > 0.65


def test_arima_uses_the_past_not_the_future():
    """The alignment guard.

    P(up) at row i must track r[i] (known) and only imperfectly track r[i+1]
    (the thing being predicted). A near-perfect correlation with r[i+1] would
    mean the forecast had already seen its own answer.
    """
    frame = ar1_frame(phi=0.6)
    proba = ArimaDirectionModel().fit(frame).predict_proba(frame)
    r = frame["ret_lag_1"].to_numpy()

    corr_past = np.corrcoef(proba[:-1], r[:-1])[0, 1]
    corr_future = np.corrcoef(proba[:-1], r[1:])[0, 1]

    assert corr_past > 0.95
    assert corr_future < 0.8, "correlation with the future implies leakage"


def test_arima_direction_follows_the_sign_of_phi():
    positive = ar1_frame(phi=0.6)
    negative = ar1_frame(phi=-0.6)

    r_pos = positive["ret_lag_1"].to_numpy()
    r_neg = negative["ret_lag_1"].to_numpy()
    p_pos = ArimaDirectionModel().fit(positive).predict_proba(positive)
    p_neg = ArimaDirectionModel().fit(negative).predict_proba(negative)

    assert np.corrcoef(p_pos[:-1], r_pos[:-1])[0, 1] > 0.9
    assert np.corrcoef(p_neg[:-1], r_neg[:-1])[0, 1] < -0.9


def test_arima_on_noise_stays_near_a_coin_flip():
    """The expected regime for real BTC data: no structure, no confidence."""
    proba = ArimaDirectionModel().fit(ar1_frame(phi=0.0)).predict_proba(ar1_frame(phi=0.0))

    assert abs(float(np.mean(proba)) - 0.5) < 0.05
    assert float(np.std(proba)) < 0.1, "a noise series must not produce confident calls"


def test_arima_returns_one_probability_per_row():
    frame = ar1_frame()
    assert len(ArimaDirectionModel().fit(frame).predict_proba(frame)) == len(frame)


def test_unfitted_models_refuse_to_predict():
    frame = ar1_frame()
    with pytest.raises(RuntimeError, match="not fitted"):
        ArimaDirectionModel().predict_proba(frame)
    with pytest.raises(RuntimeError, match="not fitted"):
        GbtDirectionModel().predict_proba(frame)


def test_gbt_learns_a_planted_feature():
    """Confirms the GBT is wired to the feature columns, not just to noise."""
    rng = np.random.default_rng(11)
    n = 2000
    frame = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES})
    frame[LABEL_COLUMN] = (frame["close_pos"] > 0).astype(int)

    params = {"n_estimators": 80, "max_depth": 3, "min_child_weight": 5}
    model = GbtDirectionModel(params).fit(frame)
    acc = (model.predict(frame) == frame[LABEL_COLUMN].to_numpy()).mean()

    assert acc > 0.9
    assert model.feature_importance()["close_pos"] == max(model.feature_importance().values())


def test_gbt_probabilities_are_valid():
    frame = ar1_frame()
    proba = GbtDirectionModel({"n_estimators": 40}).fit(frame).predict_proba(frame)

    assert proba.shape == (len(frame),)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_sampled_params_stay_in_the_shallow_space():
    """§7: depth is capped because the risk here is fitting noise."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        params = sample_params(rng)
        assert 2 <= params["max_depth"] <= 4
        assert params["min_child_weight"] >= 10
        assert 0.6 <= params["subsample"] <= 1.0
        assert "scale_pos_weight" not in params


def test_walk_forward_evaluate_reports_every_fold():
    frame = ar1_frame(n=1200, phi=0.4)
    folds = walk_forward_splits(len(frame), n_splits=3, test_size=150, min_train_size=100)

    report = walk_forward_evaluate(ArimaDirectionModel, frame, folds)

    assert report.model_name == "arima"
    assert len(report.per_fold) == 3
    assert report.pooled.n == 450
    assert report.pooled.accuracy > 0.55, "signal should survive walk-forward"


def test_arima_artifact_stores_only_coefficients():
    """A statsmodels result pickles its whole training series (~12MB for two
    years of hourly bars). Retrained daily that would fill the artifact volume
    with gigabytes describing a three-parameter model."""
    import pickle

    frame = ar1_frame(n=4000, phi=0.4)
    model = ArimaDirectionModel().fit(frame)
    before = model.predict_proba(frame)

    blob = pickle.dumps(model)
    after = pickle.loads(blob).predict_proba(frame)

    assert len(blob) < 10_000, f"artifact unexpectedly large: {len(blob)} bytes"
    np.testing.assert_allclose(before, after)


def test_gbt_hyperparameters_round_trip():
    """The registry stores this dict and reproducibility depends on rebuilding
    an identical model from it."""
    original = GbtDirectionModel({"max_depth": 4, "n_estimators": 55})
    rebuilt = GbtDirectionModel(original.hyperparameters)

    assert rebuilt.hyperparameters == original.hyperparameters
    assert rebuilt.random_state == original.random_state
