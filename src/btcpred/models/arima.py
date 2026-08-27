"""ARIMA baseline — univariate, no exogenous regressors, no seasonal terms.

Order is fixed at (1,0,0) by decision (context.md §7): selection on an early
window found nothing to choose between candidates, BIC preferred white noise,
and an order that churns between retrains would make versions incomparable.
Only coefficients are refit.

`d = 0` because the series is already log-returns; differencing again would
over-difference and inject a spurious MA(1) term.
"""

import logging
import warnings
from typing import Any, Self

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from btcpred.models.base import RETURN_SERIES_COLUMN, DirectionModel

logger = logging.getLogger(__name__)

DEFAULT_ORDER = (1, 0, 0)


class ArimaDirectionModel(DirectionModel):
    name = "arima"

    def __init__(self, order: tuple[int, int, int] = DEFAULT_ORDER, trend: str = "c") -> None:
        self.order = tuple(order)
        self.trend = trend
        self._result = None

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"order": list(self.order), "trend": self.trend}

    def __getstate__(self) -> dict[str, Any]:
        """Persist coefficients only, not the fitted state object.

        A statsmodels result carries its entire training series, which pickles to
        roughly 12MB for two years of hourly bars. Retrained daily that would
        accumulate gigabytes on the artifact volume for a model defined by three
        numbers. Coefficients reproduce predictions exactly, so only they are
        stored.
        """
        state = self.__dict__.copy()
        result = state.pop("_result", None)
        state["_params"] = None if result is None else np.asarray(result.params)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        params = state.pop("_params", None)
        self.__dict__.update(state)
        self._result = None
        if params is not None:
            # `filter` applies known coefficients without re-estimating; the
            # placeholder series is replaced by `apply` at prediction time.
            placeholder = np.zeros(max(10, sum(self.order) + 2))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._result = ARIMA(placeholder, order=self.order, trend=self.trend).filter(params)

    def fit(self, frame: pd.DataFrame) -> Self:
        series = frame[RETURN_SERIES_COLUMN].to_numpy(dtype=float)
        with warnings.catch_warnings():
            # On a near-white-noise series the optimiser often reports
            # non-convergence while still returning usable coefficients. That is
            # the expected regime here, not an error worth failing the retrain.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._result = ARIMA(series, order=self.order, trend=self.trend).fit()
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """P(up) for each row, from the one-step-ahead forecast distribution.

        Alignment is the subtle part. Row `i` is labelled with the direction of
        `r[i+1]`, so the value needed at row `i` is E[r[i+1] | r[<=i]] — the
        forecast made *standing at* row i, not the fit of row i itself. Hence
        predictions are taken over [1, n]: index n is genuinely one step beyond
        the supplied data.

        The caller must pass a contiguous block that includes leading context;
        the model applies its fitted coefficients to that series rather than
        assuming it continues the training data.
        """
        if self._result is None:
            raise RuntimeError("model is not fitted")

        series = frame[RETURN_SERIES_COLUMN].to_numpy(dtype=float)
        if series.size == 0:
            return np.empty(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self._result.apply(series, refit=False)
            prediction = applied.get_prediction(start=1, end=series.size)

        mu = np.asarray(prediction.predicted_mean, dtype=float)
        sigma = np.asarray(prediction.se_mean, dtype=float)

        # §7: the forecast distribution is what makes P(log-return > 0) fall out,
        # and therefore what makes this comparable to the GBT on log loss.
        sigma = np.where(sigma > 0, sigma, np.nan)
        proba = norm.cdf(mu / sigma)
        return np.where(np.isfinite(proba), proba, 0.5)
