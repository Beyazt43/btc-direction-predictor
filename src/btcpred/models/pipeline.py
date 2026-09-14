"""Training pipeline: evaluate, fit, persist, register, gate.

Follows §8's three-way discipline. Hyperparameters are chosen on walk-forward
folds; the final chronological holdout is never read by the production path.
`evaluate_holdout` exists for the one-time offline evaluation and is called
deliberately, once, when the project is ready to report a final number.
"""

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.config import Settings, get_settings
from btcpred.features.builder import FEATURE_NAMES
from btcpred.features.dataset import build_training_dataset
from btcpred.models.arima import ArimaDirectionModel
from btcpred.models.base import DirectionModel
from btcpred.models.gbt import GbtDirectionModel
from btcpred.models.registry import (
    active_version,
    make_version_id,
    register_version,
    save_artifact,
)
from btcpred.models.splits import split_holdout
from btcpred.models.training import (
    WalkForwardReport,
    build_folds,
    random_search,
    walk_forward_evaluate,
)

logger = logging.getLogger(__name__)

MODEL_NAMES = ("arima", "xgboost")

# Activation gate. Both thresholds are set to catch a broken retrain, never the
# day-to-day noise between healthy ones: the entire ARIMA-vs-GBT gap is about
# 0.001 of log loss, and these margins are an order of magnitude wider.
COIN_FLIP_LOG_LOSS = math.log(2)  # 0.69315: what predicting 0.5 forever scores
MAX_LOG_LOSS = 0.70  # clearly worse than knowing nothing
MAX_REGRESSION = 0.01  # clearly worse than the incumbent's reference


@dataclass(frozen=True, slots=True)
class GateDecision:
    activate: bool
    reason: str


@dataclass(frozen=True, slots=True)
class TrainedModel:
    model: DirectionModel
    version_id: str
    artifact_path: Path
    report: WalkForwardReport
    train_start: datetime
    train_end: datetime
    gate: GateDecision
    incumbent_version: str | None


def gate_candidate(report: WalkForwardReport, incumbent: dict[str, Any] | None) -> GateDecision:
    """Decide whether a freshly trained version may go live.

    Two catastrophe checks and nothing subtler. A candidate is refused if it is
    clearly worse than a coin flip, or clearly worse than the incumbent's own
    reference score. Normal variation between healthy retrains is far inside
    both margins, so a healthy candidate always activates -- the gate exists
    for the retrain that ran on a corrupted day or a broken build, not to
    referee two working models.
    """
    log_loss = report.pooled.log_loss
    if log_loss is None:
        return GateDecision(False, "no log loss available (single-class folds)")
    if log_loss >= MAX_LOG_LOSS:
        return GateDecision(
            False, f"log loss {log_loss:.5f} is worse than a coin flip ({COIN_FLIP_LOG_LOSS:.5f})"
        )
    if incumbent is None:
        return GateDecision(True, "no incumbent; first version activates")

    ref = ((incumbent.get("reference_metrics") or {}).get("pooled") or {}).get("log_loss")
    if ref is None:
        return GateDecision(True, "incumbent has no reference log loss to compare against")
    if log_loss > ref + MAX_REGRESSION:
        return GateDecision(
            False,
            f"log loss {log_loss:.5f} regresses incumbent {ref:.5f} by more than {MAX_REGRESSION}",
        )
    return GateDecision(True, f"log loss {log_loss:.5f} vs incumbent {ref:.5f}")


async def train_model(
    session: AsyncSession,
    model_name: str,
    *,
    settings: Settings | None = None,
    dataset: pd.DataFrame | None = None,
    include_holdout: bool = True,
    gated: bool = True,
    n_splits: int = 5,
    n_configs: int = 30,
    seed: int | None = None,
) -> TrainedModel:
    """Train one model end to end, register the version, and gate activation.

    `include_holdout=True` is the production setting: the deployed model trains
    on everything, because the live prediction log is its real out-of-sample
    test and a model held two months back on data buys no evaluation benefit.
    `include_holdout=False` is the offline path for the one-time §8 evaluation.

    The registered version carries its walk-forward metrics as the *reference*
    §8 speaks of: drift is later measured as observed-minus-reference, so the
    reference must travel with the version rather than live in a notebook.
    """
    settings = settings or get_settings()
    if dataset is None:
        dataset = await build_training_dataset(session, symbol=settings.binance_symbol)

    if include_holdout:
        train = dataset.reset_index(drop=True)
    else:
        dev_idx, _ = split_holdout(len(dataset), settings.holdout_days * 24)
        train = dataset.iloc[dev_idx].reset_index(drop=True)
    logger.info(
        "%s: dataset=%d train=%d (holdout %s)",
        model_name,
        len(dataset),
        len(train),
        "included" if include_holdout else "excluded",
    )

    folds = build_folds(len(train), n_splits=n_splits)
    incumbent = await active_version(session, model_name)

    # Seed differs per day so the search explores rather than replaying one
    # draw; the incumbent's own config is always in the running regardless.
    if seed is None:
        seed = int(datetime.now(UTC).strftime("%Y%m%d"))

    if model_name == "arima":
        report = walk_forward_evaluate(ArimaDirectionModel, train, folds)
        model: DirectionModel = ArimaDirectionModel()
    elif model_name == "xgboost":
        seeded = [incumbent["hyperparameters"]] if incumbent else []
        best_params, _ = random_search(
            train, folds, n_configs=n_configs, seed=seed, seeded_configs=seeded
        )
        report = walk_forward_evaluate(lambda: GbtDirectionModel(best_params), train, folds)
        model = GbtDirectionModel(best_params)
    else:
        raise ValueError(f"unknown model: {model_name!r}")

    # Final fit uses everything: folds were for selection, and the deployed
    # model should see every observation it is allowed to.
    model.fit(train)

    train_start = pd.Timestamp(train["open_time"].min()).to_pydatetime()
    train_end = pd.Timestamp(train["open_time"].max()).to_pydatetime()
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
    # searched model's walk-forward score is the maximum over configurations
    # and reads high. Only the holdout, or the live log, settles it.
    reference_metrics["selection_biased"] = model_name == "xgboost"
    reference_metrics["n_configs_searched"] = n_configs if model_name == "xgboost" else 0
    reference_metrics["holdout_included"] = include_holdout

    gate = gate_candidate(report, incumbent) if gated else GateDecision(True, "gate bypassed")
    reference_metrics["gate"] = {"activated": gate.activate, "reason": gate.reason}
    if not gate.activate:
        logger.warning("%s %s NOT activated: %s", model_name, version_id, gate.reason)

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
        activate=gate.activate,
    )

    return TrainedModel(
        model=model,
        version_id=version_id,
        artifact_path=artifact_path,
        report=report,
        train_start=train_start,
        train_end=train_end,
        gate=gate,
        incumbent_version=incumbent["model_version"] if incumbent else None,
    )


async def train_all(
    session: AsyncSession,
    *,
    settings: Settings | None = None,
    include_holdout: bool = True,
    gated: bool = True,
    n_configs: int = 30,
    seed: int | None = None,
) -> dict[str, TrainedModel]:
    """Train both families against one shared dataset.

    Sharing the dataset is not just an optimisation: it guarantees the baseline
    and the challenger were scored on identical rows, which is what makes the
    §8 head-to-head comparison meaningful.
    """
    settings = settings or get_settings()
    dataset = await build_training_dataset(session, symbol=settings.binance_symbol)

    results: dict[str, TrainedModel] = {}
    for name in MODEL_NAMES:
        results[name] = await train_model(
            session,
            name,
            settings=settings,
            dataset=dataset,
            include_holdout=include_holdout,
            gated=gated,
            n_configs=n_configs,
            seed=seed,
        )
    return results


def format_report(trained: TrainedModel) -> str:
    ev = trained.report.pooled
    status = "ACTIVE" if trained.gate.activate else "NOT ACTIVATED"
    lines = [
        f"{trained.report.model_name}  version={trained.version_id}  [{status}]",
        f"  gate        {trained.gate.reason}",
        f"  incumbent   {trained.incumbent_version or '(none)'}",
        f"  accuracy    {ev.accuracy:.4f}   (majority {ev.majority_baseline:.4f}, "
        f"persistence {ev.persistence_baseline:.4f})",
        f"  edge        {ev.edge_over_majority:+.4f}  SE {ev.standard_error:.4f}  "
        f"clears 2SE: {ev.beats_majority}",
        f"  mcc         {ev.mcc:+.4f}",
        f"  log loss    {ev.log_loss:.5f}   ({COIN_FLIP_LOG_LOSS:.5f} = knowing nothing)",
        f"  auc         {ev.auc:.4f}",
        f"  artifact    {trained.artifact_path}",
    ]
    return "\n".join(lines)
