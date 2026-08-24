"""Target construction.

This module is quarantined on purpose. It contains the codebase's only
forward-looking operation, so an auditor asking "where could lookahead bias
enter?" has exactly one file to read. `btcpred.features.builder` contains no
`shift(-1)` at all, and must not gain one.

Label, for a bar with open_time = t:

    direction(t) = 1 if close(t+1) > close(t) else 0

Strictly binary, no neutral class (see context.md §6). The realised log return is
kept alongside it so accuracy can be sliced by move magnitude after the fact,
which is what replaces a dead zone.
"""

import numpy as np
import pandas as pd

LABEL_COLUMN = "direction"
RETURN_COLUMN = "forward_log_return"


def add_labels(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach next-hour direction and realised log return to each bar.

    The final row is dropped: its label depends on a bar that has not closed yet,
    so it is precisely the row a leaky pipeline would keep.
    """
    if not bars["open_time"].is_monotonic_increasing:
        raise ValueError("bars must be ordered by open_time ascending")

    close = bars["close"].astype(float)
    # The one legitimate shift(-1) in the project: this is the target, looking
    # forward by definition, not a feature.
    next_close = close.shift(-1)

    out = bars.copy()
    out[RETURN_COLUMN] = np.log(next_close / close)
    out[LABEL_COLUMN] = (next_close > close).astype("Int8")

    return out.loc[next_close.notna()].reset_index(drop=True)
