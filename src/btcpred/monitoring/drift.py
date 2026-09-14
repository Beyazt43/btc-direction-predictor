"""Drift detection, per context.md §8 and §9 item 4.

Drift is observed-minus-reference, tested against the sample-size table in §8.
Both sides come from the live prediction log: observed is the most recent 30
days, reference is everything live before that. Nothing here reads a
walk-forward number -- the stored XGBoost figure is the maximum over a search
and reads high, so comparing live accuracy to it would report drift from day
one. Comparing the live log to its own history sidesteps the bias entirely and
is the ordinary definition of concept drift: has recent performance departed
from established performance.

An alert raises attention, never a retrain. The retrain already runs daily, a
cadence far faster than a 30-day window can detect anything, so a
drift-triggered retrain would land long after the fix had shipped. What an
alert asks is whether the world moved or the model did, and the PSI diagnostic
is there to help answer that.
"""

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.db.tables import drift_checks
from btcpred.features.builder import FEATURE_NAMES
from btcpred.features.dataset import build_training_dataset
from btcpred.models.registry import active_version
from btcpred.predict.repository import live_calls

logger = logging.getLogger(__name__)

WINDOW_DAYS = 30
# §8: the 30-day window is the primary signal because shorter ones carry noise
# bands wider than any degradation worth detecting.
MIN_OBSERVED = 7 * 24  # below a week of calls the window is not worth scoring
MIN_REFERENCE = WINDOW_DAYS * 24  # the reference needs at least as much as the window
Z_ALERT = -2.0  # one-sided: only a drop is drift
STATUSES = ("ok", "alert", "warming_up", "insufficient")


@dataclass(frozen=True, slots=True)
class Sample:
    n: int
    correct: int
    ups: int

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.n if self.n else None

    @property
    def majority_baseline(self) -> float | None:
        if not self.n:
            return None
        p = self.ups / self.n
        return max(p, 1.0 - p)


@dataclass(frozen=True, slots=True)
class DriftCheck:
    model_name: str
    checked_at: datetime
    window_start: datetime
    window_end: datetime
    observed: Sample
    reference: Sample
    z_score: float | None
    status: str
    reason: str
    feature_psi: dict[str, float] | None

    def to_row(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "model_name": self.model_name,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "observed_n": self.observed.n,
            "observed_accuracy": self.observed.accuracy,
            "majority_baseline": self.observed.majority_baseline,
            "reference_n": self.reference.n,
            "reference_accuracy": self.reference.accuracy,
            "z_score": self.z_score,
            "status": self.status,
            "reason": self.reason,
            "feature_psi": self.feature_psi,
        }


async def sample_accuracy(
    session: AsyncSession,
    model_name: str,
    *,
    interval: timedelta,
    start: datetime | None,
    end: datetime,
) -> Sample:
    calls = live_calls(model_name, interval)
    conditions = [calls.c.target_open_time < end]
    if start is not None:
        conditions.append(calls.c.target_open_time >= start)

    stmt = sa.select(
        sa.func.count(),
        sa.func.coalesce(
            sa.func.sum(
                sa.case((calls.c.predicted_direction == calls.c.actual_direction, 1), else_=0)
            ),
            0,
        ),
        sa.func.coalesce(sa.func.sum(calls.c.actual_direction), 0),
    ).where(*conditions)
    n, correct, ups = (await session.execute(stmt)).one()
    return Sample(n=int(n), correct=int(correct), ups=int(ups))


def two_proportion_z(observed: Sample, reference: Sample) -> float | None:
    """Pooled two-proportion z statistic, negative when observed is worse.

    This is the test §8's CI table is built from: at p≈0.5 the standard error
    of a 30-day window is ~1.9pp, so a real 3-point drop needs the 30 days to
    show at all, and a week cannot show it.
    """
    if not observed.n or not reference.n:
        return None
    p_obs = observed.correct / observed.n
    p_ref = reference.correct / reference.n
    pooled = (observed.correct + reference.correct) / (observed.n + reference.n)
    se = math.sqrt(pooled * (1 - pooled) * (1 / observed.n + 1 / reference.n))
    if se == 0:
        return None
    return (p_obs - p_ref) / se


def population_stability_index(
    values: np.ndarray, edges: list[float], epsilon: float = 1e-4
) -> float:
    """PSI of `values` against ten equal-mass reference bins.

    The reference share of every bin is 0.1 by construction (the edges are the
    training deciles), so only the observed histogram needs computing.
    Conventional reading: < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant.

    That reading only applies to fast-moving features. A 30-day window holds
    720 values of a 168-hour rolling statistic, which is about four effectively
    independent samples; they cannot represent a two-year distribution whatever
    the regime, so `rv_168` and `ret_mean_168` read as 'significant' every day
    (observed on real data: 1.2-1.8 while `close_pos` sits at 0.01). This is
    why PSI is a diagnostic ranking here and never an alert source.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    counts = np.histogram(values, bins=[-np.inf, *edges, np.inf])[0]
    observed = counts / values.size
    reference = np.full_like(observed, 1.0 / len(observed))
    observed = np.clip(observed, epsilon, None)
    return float(np.sum((observed - reference) * np.log(observed / reference)))


async def feature_psi(
    session: AsyncSession,
    version: dict[str, Any],
    *,
    symbol: str,
    start: datetime,
    end: datetime,
) -> dict[str, float] | None:
    """PSI per feature over [start, end) against the version's training deciles."""
    quantiles = ((version.get("reference_metrics") or {}).get("feature_quantiles")) or None
    if not quantiles:
        return None

    frame = await build_training_dataset(
        session, symbol=symbol, start=pd.Timestamp(start), end=pd.Timestamp(end)
    )
    if frame.empty:
        return None

    return {
        name: round(population_stability_index(frame[name].to_numpy(), quantiles[name]), 4)
        for name in FEATURE_NAMES
        if name in quantiles
    }


def classify(observed: Sample, reference: Sample, z: float | None) -> tuple[str, str]:
    if observed.n < MIN_OBSERVED:
        return "insufficient", f"only {observed.n} live resolved calls in the window"

    # The sanity floor applies from the first week: a model that cannot beat
    # "always up" on its own window is worth knowing about regardless of history.
    acc = observed.accuracy
    majority = observed.majority_baseline
    floor = majority - 2 * math.sqrt(0.25 / observed.n) if majority is not None else None
    if acc is not None and floor is not None and acc < floor:
        return "alert", f"accuracy {acc:.4f} is 2σ below the majority baseline {majority:.4f}"

    if reference.n < MIN_REFERENCE:
        return "warming_up", (
            f"reference has {reference.n} calls; needs {MIN_REFERENCE} before drift can be tested"
        )
    if z is None:
        return "insufficient", "z-score undefined"
    if z < Z_ALERT:
        return "alert", (
            f"accuracy {acc:.4f} vs reference {reference.accuracy:.4f}: z={z:.2f} (< {Z_ALERT})"
        )
    return "ok", f"accuracy {acc:.4f} vs reference {reference.accuracy:.4f}: z={z:.2f}"


async def check_model(
    session: AsyncSession,
    model_name: str,
    *,
    symbol: str,
    interval: timedelta,
    now: datetime | None = None,
) -> DriftCheck:
    now = now or datetime.now(UTC)
    window_end = now
    window_start = now - timedelta(days=WINDOW_DAYS)

    observed = await sample_accuracy(
        session, model_name, interval=interval, start=window_start, end=window_end
    )
    reference = await sample_accuracy(
        session, model_name, interval=interval, start=None, end=window_start
    )
    z = two_proportion_z(observed, reference)
    status, reason = classify(observed, reference, z)

    psi = None
    version = await active_version(session, model_name)
    if version is not None and observed.n:
        try:
            psi = await feature_psi(
                session, version, symbol=symbol, start=window_start, end=window_end
            )
        except Exception:
            logger.exception("feature PSI failed for %s; continuing without it", model_name)

    check = DriftCheck(
        model_name=model_name,
        checked_at=now,
        window_start=window_start,
        window_end=window_end,
        observed=observed,
        reference=reference,
        z_score=z,
        status=status,
        reason=reason,
        feature_psi=psi,
    )

    if status == "alert":
        movers = sorted((psi or {}).items(), key=lambda kv: -kv[1])[:5]
        logger.warning(
            "DRIFT ALERT %s: %s | top feature movers (PSI): %s",
            model_name,
            reason,
            ", ".join(f"{k}={v:.3f}" for k, v in movers) or "n/a",
        )
    else:
        logger.info("drift %s: %s -- %s", model_name, status, reason)
    return check


async def record_check(session: AsyncSession, check: DriftCheck) -> None:
    await session.execute(sa.insert(drift_checks).values(**check.to_row()))


async def latest_checks(session: AsyncSession, limit: int = 10) -> list[dict[str, Any]]:
    stmt = sa.select(drift_checks).order_by(drift_checks.c.checked_at.desc()).limit(limit)
    return [dict(r) for r in (await session.execute(stmt)).mappings()]
