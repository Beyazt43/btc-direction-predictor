"""Generate and resolve predictions (context.md §5).

Order matters, and it is the leakage guard: a prediction for hour t+1 is only
generated once bar t is confirmed closed and ingested. Both models are then
served from the same feature window, computed by the same function training
used.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.features.builder import FEATURE_NAMES, InsufficientHistoryError, build_feature_window
from btcpred.ingest.repository import latest_bar_open_time
from btcpred.models.base import DirectionModel
from btcpred.models.registry import active_version, load_artifact
from btcpred.predict.repository import insert_prediction, prediction_exists

logger = logging.getLogger(__name__)

MODEL_NAMES = ("arima", "xgboost")

# Artifacts are tiny and versions change once a day; caching by version id
# means a tick that predicts costs no disk I/O for the model itself.
_model_cache: dict[str, DirectionModel] = {}


class FeatureMismatchError(RuntimeError):
    """The live feature set differs from the one the model was trained on.

    This is train/serve skew made concrete, and it is exactly what the
    registry's feature_names column exists to catch. Refusing to predict is
    the only safe response.
    """


@dataclass(frozen=True, slots=True)
class PredictionResult:
    model_name: str
    model_version: str
    target_open_time: datetime
    direction: int
    proba: float
    inserted: bool


def _load_model(version: dict, model_dir: Path) -> DirectionModel:
    version_id = version["model_version"]
    if version_id not in _model_cache:
        _model_cache[version_id] = load_artifact(version["artifact_path"], model_dir)
    return _model_cache[version_id]


def _check_features(version: dict) -> None:
    trained_on = version.get("feature_names")
    if trained_on is not None and list(trained_on) != list(FEATURE_NAMES):
        raise FeatureMismatchError(
            f"{version['model_version']} was trained on {len(trained_on)} features, "
            f"live code has {len(FEATURE_NAMES)}"
        )


async def generate_predictions(
    session: AsyncSession,
    *,
    symbol: str,
    interval: timedelta,
    model_dir: Path,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> list[PredictionResult]:
    """Predict the next hour from the newest closed bar, for every live model.

    Idempotent: a target hour that already has a prediction for a given version
    is skipped, so running this every tick is safe and a tick that arrives
    mid-hour simply does nothing.
    """
    newest = await latest_bar_open_time(session, symbol)
    if newest is None:
        logger.warning("no bars ingested yet; nothing to predict from")
        return []

    as_of = pd.Timestamp(newest)
    target = newest + interval
    results: list[PredictionResult] = []
    window: pd.DataFrame | None = None

    for model_name in model_names:
        version = await active_version(session, model_name)
        if version is None:
            logger.warning("no active %s version; skipping", model_name)
            continue

        if await prediction_exists(
            session,
            model_name=model_name,
            model_version=version["model_version"],
            target_open_time=target,
        ):
            continue

        _check_features(version)

        if window is None:
            try:
                window = await build_feature_window(session, as_of, symbol=symbol)
            except InsufficientHistoryError as exc:
                logger.warning("cannot build features as of %s: %s", as_of, exc)
                return results

        model = _load_model(version, model_dir)
        proba = float(model.predict_proba(window)[-1])
        direction = int(proba >= 0.5)

        inserted = await insert_prediction(
            session,
            model_name=model_name,
            model_version=version["model_version"],
            target_open_time=target,
            predicted_direction=direction,
            predicted_proba=proba,
        )
        results.append(
            PredictionResult(
                model_name=model_name,
                model_version=version["model_version"],
                target_open_time=target,
                direction=direction,
                proba=proba,
                inserted=inserted,
            )
        )
        if inserted:
            logger.info(
                "%s predicted %s for %s (p_up=%.4f, from bar %s)",
                model_name,
                "UP" if direction else "DOWN",
                target,
                proba,
                as_of,
            )

    return results


def clear_model_cache() -> None:
    _model_cache.clear()
