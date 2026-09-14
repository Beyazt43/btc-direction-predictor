from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from btcpred.api import queries
from btcpred.api.main import app, get_interval, get_session

HOUR = timedelta(hours=1)


# --- pure computation ---------------------------------------------------------


def calls(n: int, hit_rate: float, start: datetime | None = None, gap_at: int | None = None):
    """A frame of resolved live calls with a controllable hit rate and an optional gap."""
    rng = np.random.default_rng(0)
    start = start or datetime(2026, 9, 1, tzinfo=UTC)
    times = [start + i * HOUR for i in range(n)]
    if gap_at is not None:
        times = [t + (HOUR if i >= gap_at else timedelta()) for i, t in enumerate(times)]
    actual = rng.integers(0, 2, n)
    hit = rng.random(n) < hit_rate
    pred = np.where(hit, actual, 1 - actual)
    return pd.DataFrame(
        {
            "target_open_time": times,
            "model_version": "v",
            "predicted_at": times,
            "predicted_direction": pred,
            "predicted_proba": np.where(pred == 1, 0.55, 0.45),
            "actual_direction": actual,
            "actual_log_return": rng.normal(0, 0.005, n),
        }
    )


def test_score_reports_the_baselines_and_the_band():
    out = queries._score(calls(720, 0.55), HOUR)
    assert out["n"] == 720
    assert 0.5 < out["accuracy"] < 0.6
    assert "majority_baseline" in out and "persistence_baseline" in out
    assert out["ci95_halfwidth"] == pytest.approx(1.96 * np.sqrt(0.25 / 720))


def test_score_is_none_on_empty():
    assert queries._score(calls(0, 0.5), HOUR) is None


def test_persistence_baseline_skips_across_a_gap():
    """An outage leaves a hole; the hour after it has no 'previous hour' to persist."""
    frame = calls(50, 0.5, gap_at=25)
    prev = queries._previous_direction(frame, HOUR)
    assert np.isnan(prev[0]), "first row has no predecessor"
    assert np.isnan(prev[25]), "row after the gap must not borrow across it"
    assert not np.isnan(prev[24]) and not np.isnan(prev[26])


def test_magnitude_bucket_boundaries_are_in_basis_points():
    labels = [b[0] for b in queries.MAGNITUDE_BUCKETS]
    assert labels == ["< 10bp", "10-50bp", "50-100bp", "> 100bp"]
    lo = [b[1] for b in queries.MAGNITUDE_BUCKETS]
    assert lo == [0.0, 10.0, 50.0, 100.0]


# --- HTTP layer, with the database stubbed ---------------------------------------


class FakeSession:
    pass


@pytest.fixture
def client(monkeypatch):
    async def fake_session():
        yield FakeSession()

    app.dependency_overrides[get_session] = fake_session
    app.dependency_overrides[get_interval] = lambda: HOUR
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_route(client, monkeypatch):
    async def fake_health(session, symbol):
        return {"scheduler_ok": True, "predictions_logged": 3, "active_versions": {}}

    monkeypatch.setattr(queries, "health", fake_health)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["scheduler_ok"] is True


def test_unknown_model_is_a_404(client):
    assert client.get("/metrics/live/prophet/daily").status_code == 404


def test_windows_are_the_section_8_windows(client, monkeypatch):
    async def fake_live(session, model_name, interval, now=None):
        return [queries.WindowMetrics(w, d, None, None) for w, d in queries.WINDOWS.items()]

    monkeypatch.setattr(queries, "live_metrics", fake_live)
    body = client.get("/metrics/live").json()
    assert set(body) == {"arima", "xgboost"}
    assert [w["window"] for w in body["arima"]] == ["7d", "30d", "all"]


def test_days_query_is_validated(client):
    assert client.get("/metrics/comparison?days=0").status_code == 422
    assert client.get("/drift?limit=0").status_code == 422


def test_dashboard_renders_with_no_data(client, monkeypatch):
    """The page must not 500 on an empty database."""

    async def fake_health(session, symbol):
        return {
            "scheduler_ok": False,
            "last_bar_age_seconds": None,
            "last_prediction_age_seconds": None,
            "predictions_logged": 0,
            "active_versions": {},
        }

    async def fake_live(session, model_name, interval, now=None):
        return [queries.WindowMetrics(w, d, None, None) for w, d in queries.WINDOWS.items()]

    async def fake_cmp(session, interval, days=None):
        return {"n": 0, "models": {}, "mcnemar": None}

    async def fake_mag(session, interval, days=None):
        labels = [b[0] for b in queries.MAGNITUDE_BUCKETS]
        empty = [{"bucket": lb, "n": 0, "accuracy": None, "se": None} for lb in labels]
        return {m: empty for m in ("arima", "xgboost")}

    async def fake_drift(session, limit=60):
        return []

    async def fake_daily(session, model_name, interval):
        return []

    for name, fn in [
        ("health", fake_health),
        ("live_metrics", fake_live),
        ("paired_comparison", fake_cmp),
        ("accuracy_by_magnitude", fake_mag),
        ("drift_history", fake_drift),
        ("daily_series", fake_daily),
    ]:
        monkeypatch.setattr(queries, name, fn)

    r = client.get("/")
    assert r.status_code == 200
    assert "no resolved live calls" in r.text
    assert "no paired resolved calls" in r.text
    assert "no drift checks recorded" in r.text


def test_api_exposes_no_write_routes():
    """Read-only by design: every write path lives on the CLI."""
    methods = {m for r in app.routes for m in getattr(r, "methods", set())}
    assert methods <= {"GET", "HEAD"}
