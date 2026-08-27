"""Common interface for the two model families.

context.md §7 makes the asymmetry deliberate: ARIMA forecasts a log-return and
thresholds it, the GBT classifies the label directly. They are not forced into
one paradigm. What they do share is the obligation to emit a calibrated P(up),
which is what allows a log-loss and AUC comparison rather than a bare accuracy
race.

Both consume the same frame from `btcpred.features`, so neither can see
anything the other could not have.
"""

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd

# The feature frame column holding this bar's own log return, r_t. ARIMA treats
# the sequence of these as its series; the GBT treats it as one feature.
RETURN_SERIES_COLUMN = "ret_lag_1"


class DirectionModel(ABC):
    """Predicts P(next hour closes up)."""

    name: str

    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> Self:
        """Train on a chronological frame of features and labels."""

    @abstractmethod
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """P(up) for every row, using only that row's own information."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Directional call. Threshold at 0.5, matching 'forecast > 0' for ARIMA."""
        return (self.predict_proba(frame) >= 0.5).astype(int)

    @property
    @abstractmethod
    def hyperparameters(self) -> dict[str, Any]:
        """Recorded in the registry so a version can be reproduced."""
