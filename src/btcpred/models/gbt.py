"""Gradient-boosted trees — the challenger.

Classifies the binary label directly (context.md §7), rather than forecasting a
return and thresholding it.

The search space is deliberately narrow and shallow. At R² ≈ 0.004 the failure
mode is memorising noise, not underfitting, so depth is capped at 4,
min_child_weight starts high, and regularisation is on by default. §8 warns
about overfitting the validation scheme itself, which is why the budget is ~30
configurations rather than several hundred.
"""

from typing import Any, Self

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from btcpred.features.builder import FEATURE_NAMES
from btcpred.features.labels import LABEL_COLUMN
from btcpred.models.base import DirectionModel

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
}


def sample_params(rng: np.random.Generator) -> dict[str, Any]:
    """Draw one configuration from the search space.

    No class weighting: at a 50.36% base rate there is no imbalance to correct,
    and scale_pos_weight would only distort the probabilities that §7 wants to
    compare on log loss.
    """
    return {
        "n_estimators": int(rng.integers(100, 601)),
        "max_depth": int(rng.integers(2, 5)),
        "learning_rate": float(10 ** rng.uniform(-2.0, -1.0)),
        "min_child_weight": int(rng.integers(10, 201)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "reg_lambda": float(10 ** rng.uniform(0.0, 1.7)),
        "reg_alpha": float(rng.choice([0.0, 0.1, 1.0, 5.0])),
        "gamma": float(rng.choice([0.0, 0.1, 0.5])),
    }


class GbtDirectionModel(DirectionModel):
    name = "xgboost"

    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 0) -> None:
        merged = {**DEFAULT_PARAMS, **(params or {})}
        # `hyperparameters` round-trips back through this constructor -- the
        # registry stores that dict and reproducibility depends on rebuilding an
        # identical model from it -- so random_state must be accepted here rather
        # than colliding with the keyword below.
        self.random_state = int(merged.pop("random_state", random_state))
        self.params = merged
        self._model: XGBClassifier | None = None

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {**self.params, "random_state": self.random_state}

    def fit(self, frame: pd.DataFrame) -> Self:
        self._model = XGBClassifier(
            **self.params,
            random_state=self.random_state,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
        )
        self._model.fit(
            frame[list(FEATURE_NAMES)].to_numpy(dtype=float),
            frame[LABEL_COLUMN].to_numpy(dtype=int),
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        x = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
        return self._model.predict_proba(x)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        """Importances by name — settles whether the volume family earns its place."""
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return dict(
            zip(FEATURE_NAMES, self._model.feature_importances_.astype(float).tolist(), strict=True)
        )
