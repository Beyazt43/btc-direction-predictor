import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btcpred.features.builder import (
    FEATURE_NAMES,
    WARMUP_BARS,
    compute_features,
)
from btcpred.features.labels import LABEL_COLUMN, RETURN_COLUMN, add_labels


def make_bars(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A plausible OHLCV frame: random walk with consistent highs and lows."""
    rng = np.random.default_rng(seed)
    close = 60000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0, 0.003, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.lognormal(7, 0.5, n),
            "num_trades": rng.integers(1000, 50000, n).astype(float),
        }
    )


def test_no_feature_looks_forward():
    """The core leakage guard, tested structurally.

    If any feature at row k depended on data after row k, truncating the input
    just past k would change it. Nothing may move.
    """
    bars = make_bars()
    full = compute_features(bars)

    for k in (WARMUP_BARS, WARMUP_BARS + 50, len(bars) - 1):
        truncated = compute_features(bars.iloc[: k + 1])
        pd.testing.assert_series_equal(
            full.iloc[k][list(FEATURE_NAMES)],
            truncated.iloc[-1][list(FEATURE_NAMES)],
            check_names=False,
        )


def test_serving_window_matches_full_history():
    """What live inference computes from a short window must equal what training
    computed from all of history for the same timestamp."""
    bars = make_bars()
    training = compute_features(bars)

    as_of = len(bars) - 1
    window = bars.iloc[as_of + 1 - WARMUP_BARS : as_of + 1]
    serving = compute_features(window)

    assert len(window) == WARMUP_BARS
    np.testing.assert_allclose(
        serving.iloc[-1][list(FEATURE_NAMES)].astype(float).to_numpy(),
        training.iloc[as_of][list(FEATURE_NAMES)].astype(float).to_numpy(),
        rtol=1e-12,
    )


def test_features_are_scale_free():
    """Doubling every price must not move a single feature.

    A feature that shifts is carrying a price level, and would teach the model
    2026 prices rather than market behaviour.
    """
    bars = make_bars()
    scaled = bars.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] *= 2.0

    base = compute_features(bars).dropna(subset=list(FEATURE_NAMES))
    doubled = compute_features(scaled).dropna(subset=list(FEATURE_NAMES))

    np.testing.assert_allclose(
        base[list(FEATURE_NAMES)].to_numpy(),
        doubled[list(FEATURE_NAMES)].to_numpy(),
        rtol=1e-9,
    )


def test_warmup_rows_are_incomplete_and_later_rows_are_not():
    features = compute_features(make_bars())
    assert features.iloc[WARMUP_BARS - 2][list(FEATURE_NAMES)].isna().any()
    assert features.iloc[WARMUP_BARS - 1][list(FEATURE_NAMES)].notna().all()


def test_flat_bar_does_not_produce_nan():
    """A zero-range bar has no shape; it must read neutral, not NaN."""
    bars = make_bars()
    bars.loc[300, ["open", "high", "low", "close"]] = 60000.0

    row = compute_features(bars).iloc[300]

    assert row["range_pct"] == 0.0
    assert row["body_ratio"] == 0.0
    assert row["close_pos"] == 0.5


def test_rejects_unordered_bars():
    bars = make_bars().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="ascending"):
        compute_features(bars)


def test_labels_use_the_next_bar():
    bars = make_bars(n=10)
    labelled = add_labels(bars)

    # The last bar is dropped: its label needs a bar that has not closed.
    assert len(labelled) == 9
    for i in range(9):
        expected = 1 if bars["close"][i + 1] > bars["close"][i] else 0
        assert labelled[LABEL_COLUMN][i] == expected

    np.testing.assert_allclose(
        labelled[RETURN_COLUMN].to_numpy(dtype=float),
        np.log(bars["close"].shift(-1) / bars["close"]).dropna().to_numpy(),
    )


def _negative_shift_calls(path: Path) -> list[int]:
    """Line numbers of any `.shift(<negative>)` call, ignoring prose."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "shift"
            and node.args
        ):
            arg = node.args[0]
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                found.append(node.lineno)
    return found


def test_builder_contains_no_backward_shift():
    """Enforces the quarantine: forward-looking code lives only in labels.py.

    Checked against the parsed syntax tree rather than the raw text, so prose
    discussing shift(-1) does not trip it and a real call cannot hide in a
    string.
    """
    assert _negative_shift_calls(Path("src/btcpred/features/builder.py")) == []


def test_labels_is_where_the_forward_shift_lives():
    """The counterpart: the quarantine is only meaningful if it is populated."""
    assert _negative_shift_calls(Path("src/btcpred/features/labels.py")) != []
