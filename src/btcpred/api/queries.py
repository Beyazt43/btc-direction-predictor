"""Read-side queries for the API.

Everything here reads the live log through `live_calls`, the same subquery
drift monitoring uses, and scores it with the same `evaluate` and
`mcnemar_test` the training pipeline uses. The dashboard therefore cannot
disagree with the training report or the drift job about what a number means.
"""

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.db.tables import drift_checks, model_versions, predictions, price_bars
from btcpred.models.metrics import evaluate, mcnemar_test
from btcpred.predict.repository import live_calls
from btcpred.predict.service import MODEL_NAMES

# §8's windows. The 7-day one is shown because people will ask for it, and
# labelled indicative because at n=168 its noise band is ±7.6pp.
WINDOWS: dict[str, int | None] = {"7d": 7, "30d": 30, "all": None}

# §6: accuracy sliced by move magnitude, in basis points of log return.
# Boundaries are fixed for legibility; n per bucket is reported so a thin
# bucket is visibly thin.
MAGNITUDE_BUCKETS: list[tuple[str, float, float]] = [
    ("< 10bp", 0.0, 10.0),
    ("10-50bp", 10.0, 50.0),
    ("50-100bp", 50.0, 100.0),
    ("> 100bp", 100.0, math.inf),
]


async def _calls_frame(
    session: AsyncSession,
    model_name: str,
    interval: timedelta,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    calls = live_calls(model_name, interval)
    stmt = sa.select(calls)
    if start is not None:
        stmt = stmt.where(calls.c.target_open_time >= start)
    if end is not None:
        stmt = stmt.where(calls.c.target_open_time < end)
    rows = (await session.execute(stmt.order_by(calls.c.target_open_time))).mappings().all()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "target_open_time",
                "model_version",
                "predicted_at",
                "predicted_direction",
                "predicted_proba",
                "actual_direction",
                "actual_log_return",
            ]
        )
    return frame


def _previous_direction(frame: pd.DataFrame, interval: timedelta) -> np.ndarray | None:
    """Persistence baseline input: the direction of the hour before each target.

    Only defined where the previous hour is also in the frame; a gap (an outage)
    leaves NaN, and rows with NaN are dropped from that baseline alone.
    """
    if len(frame) < 2:
        return None
    t = pd.to_datetime(frame["target_open_time"], utc=True)
    # .copy(): under pandas copy-on-write the array from to_numpy() is read-only.
    prev = frame["actual_direction"].shift(1).to_numpy(dtype=float).copy()
    contiguous = (t - t.shift(1)) == interval
    prev[~contiguous.to_numpy()] = np.nan
    return prev


def _score(frame: pd.DataFrame, interval: timedelta) -> dict[str, Any] | None:
    if frame.empty:
        return None
    y = frame["actual_direction"].to_numpy(dtype=int)
    pred = frame["predicted_direction"].to_numpy(dtype=int)
    proba = frame["predicted_proba"].to_numpy(dtype=float)

    ev = evaluate(y, pred, proba=proba)
    out = ev.to_dict()

    prev = _previous_direction(frame, interval)
    if prev is not None:
        mask = ~np.isnan(prev)
        if mask.sum() > 0:
            out["persistence_baseline"] = float((y[mask] == prev[mask].astype(int)).mean())
            out["persistence_n"] = int(mask.sum())
    out["edge_over_majority"] = ev.edge_over_majority
    out["beats_majority"] = ev.beats_majority
    out["ci95_halfwidth"] = 1.96 * ev.standard_error
    return out


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    window: str
    days: int | None
    start: datetime | None
    metrics: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def live_metrics(
    session: AsyncSession, model_name: str, interval: timedelta, now: datetime | None = None
) -> list[WindowMetrics]:
    now = now or datetime.now(UTC)
    out: list[WindowMetrics] = []
    for label, days in WINDOWS.items():
        start = now - timedelta(days=days) if days else None
        frame = await _calls_frame(session, model_name, interval, start=start, end=now)
        out.append(WindowMetrics(label, days, start, _score(frame, interval)))
    return out


async def daily_series(
    session: AsyncSession, model_name: str, interval: timedelta
) -> list[dict[str, Any]]:
    """Per-day hit rate plus a 30-day rolling accuracy and majority baseline.

    The rolling lines are what §8 says to read; the daily points are shown so
    the noise they are computed over is visible rather than hidden.
    """
    frame = await _calls_frame(session, model_name, interval)
    if frame.empty:
        return []
    frame["day"] = pd.to_datetime(frame["target_open_time"], utc=True).dt.floor("D")
    frame["hit"] = (frame["predicted_direction"] == frame["actual_direction"]).astype(int)
    frame["up"] = frame["actual_direction"].astype(int)

    daily = frame.groupby("day").agg(n=("hit", "size"), hits=("hit", "sum"), ups=("up", "sum"))
    daily["accuracy"] = daily["hits"] / daily["n"]

    roll = daily[["n", "hits", "ups"]].rolling(30, min_periods=1).sum()
    daily["rolling_n"] = roll["n"]
    daily["rolling_accuracy"] = roll["hits"] / roll["n"]
    up_rate = roll["ups"] / roll["n"]
    daily["rolling_majority"] = np.maximum(up_rate, 1 - up_rate)
    daily["rolling_se"] = np.sqrt(0.25 / roll["n"])

    return [
        {
            "day": idx.date().isoformat(),
            "n": int(r.n),
            "accuracy": float(r.accuracy),
            "rolling_n": int(r.rolling_n),
            "rolling_accuracy": float(r.rolling_accuracy),
            "rolling_majority": float(r.rolling_majority),
            "rolling_se": float(r.rolling_se),
        }
        for idx, r in daily.iterrows()
    ]


async def paired_comparison(
    session: AsyncSession, interval: timedelta, days: int | None = None
) -> dict[str, Any]:
    """ARIMA vs GBT on the same target hours, per §8.

    Scored only where both have a live resolved call, so the comparison is
    paired and McNemar is the right test. A model that missed an hour the other
    called is not penalised for it here; that shows up in coverage instead.
    """
    now = datetime.now(UTC)
    start = now - timedelta(days=days) if days else None
    frames = {
        name: await _calls_frame(session, name, interval, start=start, end=now)
        for name in MODEL_NAMES
    }
    a, b = (frames[n] for n in MODEL_NAMES)
    if a.empty or b.empty:
        return {"n": 0, "models": {}, "mcnemar": None}

    joined = a.merge(b, on="target_open_time", suffixes=("_a", "_b"), how="inner")
    if joined.empty:
        return {"n": 0, "models": {}, "mcnemar": None}

    y = joined["actual_direction_a"].to_numpy(dtype=int)
    result: dict[str, Any] = {"n": int(len(joined)), "models": {}}
    for name, suffix in zip(MODEL_NAMES, ("_a", "_b"), strict=True):
        pred = joined[f"predicted_direction{suffix}"].to_numpy(dtype=int)
        proba = joined[f"predicted_proba{suffix}"].to_numpy(dtype=float)
        result["models"][name] = evaluate(y, pred, proba=proba).to_dict()

    result["mcnemar"] = mcnemar_test(
        y,
        joined["predicted_direction_a"].to_numpy(dtype=int),
        joined["predicted_direction_b"].to_numpy(dtype=int),
    )
    result["mcnemar"]["a"] = MODEL_NAMES[0]
    result["mcnemar"]["b"] = MODEL_NAMES[1]
    return result


async def accuracy_by_magnitude(
    session: AsyncSession, interval: timedelta, days: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    """§6's replacement for a dead zone: accuracy as a function of move size."""
    now = datetime.now(UTC)
    start = now - timedelta(days=days) if days else None
    out: dict[str, list[dict[str, Any]]] = {}
    for name in MODEL_NAMES:
        frame = await _calls_frame(session, name, interval, start=start, end=now)
        buckets = []
        if not frame.empty:
            bp = frame["actual_log_return"].abs().to_numpy(dtype=float) * 10_000
            hit = (frame["predicted_direction"] == frame["actual_direction"]).to_numpy()
            for label, lo, hi in MAGNITUDE_BUCKETS:
                mask = (bp >= lo) & (bp < hi)
                n = int(mask.sum())
                buckets.append(
                    {
                        "bucket": label,
                        "n": n,
                        "accuracy": float(hit[mask].mean()) if n else None,
                        "se": float(math.sqrt(0.25 / n)) if n else None,
                    }
                )
        else:
            buckets = [
                {"bucket": label, "n": 0, "accuracy": None, "se": None}
                for label, _, _ in MAGNITUDE_BUCKETS
            ]
        out[name] = buckets
    return out


async def drift_history(session: AsyncSession, limit: int = 60) -> list[dict[str, Any]]:
    stmt = sa.select(drift_checks).order_by(drift_checks.c.checked_at.desc()).limit(limit)
    rows = [dict(r) for r in (await session.execute(stmt)).mappings()]
    return list(reversed(rows))


async def versions(session: AsyncSession) -> list[dict[str, Any]]:
    stmt = sa.select(
        model_versions.c.model_name,
        model_versions.c.model_version,
        model_versions.c.trained_at,
        model_versions.c.activated_at,
        model_versions.c.retired_at,
        model_versions.c.train_start,
        model_versions.c.train_end,
        model_versions.c.reference_metrics["pooled"].label("reference"),
        model_versions.c.reference_metrics["gate"].label("gate"),
    ).order_by(model_versions.c.trained_at.desc())
    return [dict(r) for r in (await session.execute(stmt)).mappings()]


async def health(session: AsyncSession, symbol: str) -> dict[str, Any]:
    """Liveness as seen from the database, which needs no shared filesystem.

    The scheduler's heartbeat file lives in its own container; what the API can
    honestly report is when data last moved.
    """
    now = datetime.now(UTC)
    last_bar = await session.scalar(
        sa.select(sa.func.max(price_bars.c.open_time)).where(price_bars.c.symbol == symbol)
    )
    last_prediction = await session.scalar(sa.select(sa.func.max(predictions.c.predicted_at)))
    n_predictions = await session.scalar(sa.select(sa.func.count()).select_from(predictions))
    active = (
        await session.execute(
            sa.select(model_versions.c.model_name, model_versions.c.model_version).where(
                model_versions.c.activated_at.is_not(None), model_versions.c.retired_at.is_(None)
            )
        )
    ).all()

    def age(ts: datetime | None) -> float | None:
        return (now - ts).total_seconds() if ts else None

    return {
        "now": now,
        "last_bar_open_time": last_bar,
        "last_bar_age_seconds": age(last_bar),
        "last_prediction_at": last_prediction,
        "last_prediction_age_seconds": age(last_prediction),
        "predictions_logged": int(n_predictions or 0),
        "active_versions": {m: v for m, v in active},
        # A prediction older than an hour plus a couple of ticks means the
        # scheduler is not keeping up, whatever its container says.
        "scheduler_ok": age(last_prediction) is not None and age(last_prediction) < 3900,
    }
