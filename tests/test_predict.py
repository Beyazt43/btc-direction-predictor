from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql

from btcpred.features.builder import FEATURE_NAMES
from btcpred.predict import service
from btcpred.predict.repository import is_live
from btcpred.predict.service import FeatureMismatchError, generate_predictions
from btcpred.scheduler import jobs

HOUR = timedelta(hours=1)
NEWEST = datetime(2026, 9, 14, 6, tzinfo=UTC)


class FakeModel:
    name = "fake"

    def __init__(self, proba: float):
        self._proba = proba

    def predict_proba(self, frame):
        return np.full(len(frame), self._proba)


@pytest.fixture
def stubbed(monkeypatch):
    """Replace every I/O boundary so the lifecycle logic runs without a DB."""
    state = {
        "versions": {},
        "existing": set(),
        "inserted": [],
        "models": {},
    }

    async def fake_latest(session, symbol):
        return NEWEST

    async def fake_active(session, model_name):
        return state["versions"].get(model_name)

    async def fake_exists(session, *, model_name, model_version, target_open_time):
        return (model_name, model_version, target_open_time) in state["existing"]

    async def fake_insert(session, **row):
        state["inserted"].append(row)
        return True

    async def fake_window(session, as_of, *, symbol):
        assert as_of == pd.Timestamp(NEWEST), "features must be built as of the newest bar"
        return pd.DataFrame({name: np.zeros(169) for name in FEATURE_NAMES})

    def fake_load(version, model_dir):
        return state["models"][version["model_version"]]

    monkeypatch.setattr(service, "latest_bar_open_time", fake_latest)
    monkeypatch.setattr(service, "active_version", fake_active)
    monkeypatch.setattr(service, "prediction_exists", fake_exists)
    monkeypatch.setattr(service, "insert_prediction", fake_insert)
    monkeypatch.setattr(service, "build_feature_window", fake_window)
    monkeypatch.setattr(service, "_load_model", fake_load)
    service.clear_model_cache()
    return state


def version(model_name: str, vid: str, features=None) -> dict:
    return {
        "model_name": model_name,
        "model_version": vid,
        "artifact_path": f"models/{vid}.joblib",
        "feature_names": list(FEATURE_NAMES) if features is None else features,
    }


async def run(model_names=("arima", "xgboost")):
    return await generate_predictions(
        None,
        symbol="BTCUSDT",
        interval=HOUR,
        model_dir=Path("models"),
        model_names=model_names,
    )


@pytest.mark.asyncio
async def test_predicts_the_hour_after_the_newest_closed_bar(stubbed):
    """§5: the prediction is for t+1, made from bar t and everything before."""
    stubbed["versions"]["arima"] = version("arima", "a1")
    stubbed["models"]["a1"] = FakeModel(0.6)

    results = await run(("arima",))

    assert len(results) == 1
    assert results[0].target_open_time == NEWEST + HOUR
    assert results[0].direction == 1
    assert results[0].proba == pytest.approx(0.6)
    assert stubbed["inserted"][0]["target_open_time"] == NEWEST + HOUR


@pytest.mark.asyncio
async def test_threshold_matches_the_models_own_predict(stubbed):
    stubbed["versions"]["arima"] = version("arima", "a1")
    stubbed["models"]["a1"] = FakeModel(0.49)

    (result,) = await run(("arima",))

    assert result.direction == 0


@pytest.mark.asyncio
async def test_already_predicted_target_is_skipped(stubbed):
    """Running every two minutes must not produce a new row every two minutes."""
    stubbed["versions"]["arima"] = version("arima", "a1")
    stubbed["models"]["a1"] = FakeModel(0.6)
    stubbed["existing"].add(("arima", "a1", NEWEST + HOUR))

    assert await run(("arima",)) == []
    assert stubbed["inserted"] == []


@pytest.mark.asyncio
async def test_model_without_active_version_is_skipped_not_fatal(stubbed):
    stubbed["versions"]["xgboost"] = version("xgboost", "x1")
    stubbed["models"]["x1"] = FakeModel(0.55)

    results = await run()

    assert [r.model_name for r in results] == ["xgboost"]


@pytest.mark.asyncio
async def test_feature_mismatch_refuses_to_predict(stubbed):
    """Train/serve skew made concrete: the registry knows what the model saw."""
    stubbed["versions"]["arima"] = version("arima", "a1", features=["only_one"])
    stubbed["models"]["a1"] = FakeModel(0.6)

    with pytest.raises(FeatureMismatchError):
        await run(("arima",))
    assert stubbed["inserted"] == []


@pytest.mark.asyncio
async def test_both_models_share_one_feature_window(stubbed, monkeypatch):
    """Serving both from the same frame is what keeps the comparison fair."""
    calls = {"n": 0}
    original = service.build_feature_window

    async def counting_window(*args, **kwargs):
        calls["n"] += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, "build_feature_window", counting_window)
    stubbed["versions"]["arima"] = version("arima", "a1")
    stubbed["versions"]["xgboost"] = version("xgboost", "x1")
    stubbed["models"]["a1"] = FakeModel(0.6)
    stubbed["models"]["x1"] = FakeModel(0.4)

    results = await run()

    assert len(results) == 2
    assert calls["n"] == 1


def test_is_live_compares_against_the_target_hour_close():
    """The single definition every liveness query must go through."""
    sql = str(is_live(HOUR).compile(dialect=postgresql.dialect()))

    assert "predictions.predicted_at < predictions.target_open_time + " in sql


@pytest.mark.asyncio
async def test_tick_runs_all_three_stages_even_when_one_fails(monkeypatch):
    """The order is ingest, resolve, predict, and a failure must not stop the rest."""
    order: list[str] = []

    async def ok_ingest():
        order.append("ingest")
        return None

    async def failing_resolve():
        order.append("resolve")
        raise RuntimeError("db hiccup")

    async def ok_predict():
        order.append("predict")
        return []

    monkeypatch.setattr(jobs, "ingest_job", ok_ingest)
    monkeypatch.setattr(jobs, "predict_job", ok_predict)

    @asynccontextmanager
    async def fake_scope():
        yield None

    async def fake_resolve(session, *, symbol, interval):
        return await failing_resolve()

    monkeypatch.setattr(jobs, "session_scope", fake_scope)
    monkeypatch.setattr(jobs, "resolve_predictions", fake_resolve)
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: type("S", (), {"binance_symbol": "BTCUSDT", "binance_interval": "1h"})(),
    )

    result = await jobs.tick_job()

    assert order == ["ingest", "resolve", "predict"]
    assert result.resolved == 0
