"""Training pipeline: evaluate, fit, persist, register.

Follows §8's three-way discipline. Hyperparameters are chosen on walk-forward
folds over the development set; the final chronological holdout is never read
here. `evaluate_holdout` exists for that and is called deliberately, once, when
the project is ready to report a final number.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.config import Settings, get_settings
from btcpred.features.builder import FEATURE_NAMES
from btcpred.features.dataset import build_training_dataset
from btcpred.models.arima import ArimaDirectionModel
from btcpred.models.base import DirectionModel
from btcpred.models.gbt import GbtDirectionModel
from btcpred.models.registry import make_version_id, register_version, save_artifact
from btcpred.models.splits import split_holdout
from btcpred.models.training import (
    WalkForwardReport,
    build_folds,
    random_search,
    walk_forward_evaluate,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainedModel:
    model: DirectionModel
    version_id: str
    artifact_path: Path
    report: WalkForwardReport
    train_start: datetime
    train_end: datetime


async def train_model(
    session: AsyncSession,
    model_name: str,
    *,
    settings: Settings | None = None,
    dataset: pd.DataFrame | None = None,
    n_splits: int = 5,
    n_configs: int = 30,
    seed: int = 42,
    activate: bool = True,
) -> TrainedModel:
    """Train one model end to end and register the resulting version.

    The returned version is registered with its walk-forward metrics attached,
    which is what §8 means by a *reference* number: drift is later measured as
    observed-minus-reference, so the reference has to travel with the version
    rather than living in a notebook.
    """
    settings = settings or get_settings()
    if dataset is None:
        dataset = await build_training_dataset(session, symbol=settings.binance_symbol)

    holdout_bars = settings.holdout_days * 24
    dev_idx, _ = split_holdout(len(dataset), holdout_bars)
    dev = dataset.iloc[dev_idx].reset_index(drop=True)
    logger.info(
        "dataset=%d dev=%d holdout=%d (holdout not read)", len(dataset), len(dev), holdout_bars
    )

    folds = build_folds(len(dev), n_splits=n_splits)

    if model_name == "arima":
        report = walk_forward_evaluate(ArimaDirectionModel, dev, folds)
        model: DirectionModel = ArimaDirectionModel()
    elif model_name == "xgboost":
        best_params, _ = random_search(dev, folds, n_configs=n_configs, seed=seed)
        report = walk_forward_evaluate(lambda: GbtDirectionModel(best_params), dev, folds)
        model = GbtDirectionModel(best_params)
    else:
        raise ValueError(f"unknown model: {model_name!r}")

    # Final fit uses the whole development set: folds were for selection, and
    # the deployed model should see every observation it is allowed to.
    model.fit(dev)

    train_start = pd.Timestamp(dev["open_time"].min()).to_pydatetime()
    train_end = pd.Timestamp(dev["open_time"].max()).to_pydatetime()
    trained_at = datetime.now(UTC)

    version_id = make_version_id(
        model.name,
        hyperparameters=model.hyperparameters,
        feature_names=list(FEATURE_NAMES),
        train_start=train_start,
        train_end=train_end,
        trained_at=trained_at,
    )
    artifact_path = save_artifact(model, Path(settings.model_dir), version_id)

    reference_metrics = report.to_dict()
    # Selection bias is a property of the number, so it is recorded with it: a
    # searched model's walk-forward score is the maximum over configurations and
    # reads high. Only the holdout settles it.
    reference_metrics["selection_biased"] = model_name == "xgboost"
    reference_metrics["n_configs_searched"] = n_configs if model_name == "xgboost" else 0

    await register_version(
        session,
        model_name=model.name,
        version_id=version_id,
        train_start=train_start,
        train_end=train_end,
        artifact_path=artifact_path,
        hyperparameters=model.hyperparameters,
        feature_names=list(FEATURE_NAMES),
        reference_metrics=reference_metrics,
        activate=activate,
    )

    return TrainedModel(
        model=model,
        version_id=version_id,
        artifact_path=artifact_path,
        report=report,
        train_start=train_start,
        train_end=train_end,
    )


async def train_all(
    session: AsyncSession,
    *,
    settings: Settings | None = None,
    n_configs: int = 30,
    activate: bool = True,
) -> dict[str, TrainedModel]:
    """Train both families against one shared dataset.

    Sharing the dataset is not just an optimisation: it guarantees the baseline
    and the challenger were scored on identical rows, which is what makes the
    §8 head-to-head comparison meaningful.
    """
    settings = settings or get_settings()
    dataset = await build_training_dataset(session, symbol=settings.binance_symbol)

    results: dict[str, TrainedModel] = {}
    for name in ("arima", "xgboost"):
        results[name] = await train_model(
            session,
            name,
            settings=settings,
            dataset=dataset,
            n_configs=n_configs,
            activate=activate,
        )
    return results


def format_report(trained: TrainedModel) -> str:
    ev = trained.report.pooled
    lines = [
        f"{trained.report.model_name}  version={trained.version_id}",
        f"  accuracy    {ev.accuracy:.4f}   (majority {ev.majority_baseline:.4f}, "
        f"persistence {ev.persistence_baseline:.4f})",
        f"  edge        {ev.edge_over_majority:+.4f}  SE {ev.standard_error:.4f}  "
        f"clears 2SE: {ev.beats_majority}",
        f"  mcc         {ev.mcc:+.4f}",
        f"  log loss    {ev.log_loss:.5f}   (0.69315 = knowing nothing)",
        f"  auc         {ev.auc:.4f}",
        f"  artifact    {trained.artifact_path}",
    ]
    return "\n".join(lines)
