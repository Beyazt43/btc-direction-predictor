"""Model artifact persistence and version registration.

A version is only useful if it can be tied back to what produced it. §5's rule
that a missed prediction may only be back-generated with the model that was live
at that hour depends on this table being complete and its validity windows being
accurate.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from btcpred.db.tables import model_versions
from btcpred.models.base import DirectionModel

logger = logging.getLogger(__name__)


def make_version_id(
    model_name: str,
    *,
    hyperparameters: dict[str, Any],
    feature_names: list[str],
    train_start: datetime,
    train_end: datetime,
    trained_at: datetime | None = None,
) -> str:
    """Timestamp for ordering, content hash for identity.

    The timestamp makes versions sortable and human-readable in logs; the hash
    means two versions with identical inputs are recognisably the same model,
    and any change to hyperparameters, features or training window produces a
    different id rather than silently reusing one.
    """
    trained_at = trained_at or datetime.now(UTC)
    payload = json.dumps(
        {
            "model": model_name,
            "hyperparameters": hyperparameters,
            "features": feature_names,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{model_name}-{trained_at:%Y%m%dT%H%M%SZ}-{digest}"


def save_artifact(model: DirectionModel, model_dir: Path, version_id: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{version_id}.joblib"
    joblib.dump(model, path)
    logger.info("saved artifact %s", path)
    return path


def load_artifact(path: Path | str, model_dir: Path | None = None) -> DirectionModel:
    """Load a persisted model.

    Registry paths are relative POSIX strings; when `model_dir` is given they
    are resolved against it, so a version trained elsewhere loads from the
    local volume rather than from wherever it happened to be saved.
    """
    p = Path(path)
    if model_dir is not None and not p.is_absolute():
        p = Path(model_dir) / p.name
    return joblib.load(p)


async def register_version(
    session: AsyncSession,
    *,
    model_name: str,
    version_id: str,
    train_start: datetime,
    train_end: datetime,
    artifact_path: Path | str,
    hyperparameters: dict[str, Any],
    feature_names: list[str],
    reference_metrics: dict[str, Any] | None = None,
    activate: bool = False,
) -> None:
    """Insert a version, optionally making it the live one.

    Activation retires the incumbent in the same transaction. The registry's
    partial unique index permits only one active version per model, so doing
    this in two steps would either fail or leave a window with none.
    """
    now = datetime.now(UTC)
    if activate:
        await session.execute(
            sa.update(model_versions)
            .where(
                model_versions.c.model_name == model_name,
                model_versions.c.activated_at.is_not(None),
                model_versions.c.retired_at.is_(None),
            )
            .values(retired_at=now)
        )

    await session.execute(
        sa.insert(model_versions).values(
            model_name=model_name,
            model_version=version_id,
            train_start=train_start,
            train_end=train_end,
            # POSIX form regardless of host: the path is written by whichever
            # process trained (a Windows dev box, say) and read by the Linux
            # container, and a backslash survives neither direction.
            artifact_path=Path(artifact_path).as_posix(),
            hyperparameters=hyperparameters,
            feature_names=feature_names,
            reference_metrics=reference_metrics,
            activated_at=now if activate else None,
        )
    )
    logger.info("registered %s %s (active=%s)", model_name, version_id, activate)


async def active_version(session: AsyncSession, model_name: str) -> dict[str, Any] | None:
    """The version currently live for a model, if any."""
    stmt = sa.select(model_versions).where(
        model_versions.c.model_name == model_name,
        model_versions.c.activated_at.is_not(None),
        model_versions.c.retired_at.is_(None),
    )
    row = (await session.execute(stmt)).mappings().first()
    return dict(row) if row else None


async def version_live_at(
    session: AsyncSession, model_name: str, moment: datetime
) -> dict[str, Any] | None:
    """Which version was live at `moment` — the §5 back-generation lookup.

    Answerable even for hours when nothing was predicted, which is precisely the
    downtime case that makes the question worth asking.
    """
    stmt = sa.select(model_versions).where(
        model_versions.c.model_name == model_name,
        model_versions.c.activated_at.is_not(None),
        model_versions.c.activated_at <= moment,
        sa.or_(model_versions.c.retired_at.is_(None), model_versions.c.retired_at > moment),
    )
    row = (await session.execute(stmt)).mappings().first()
    return dict(row) if row else None
