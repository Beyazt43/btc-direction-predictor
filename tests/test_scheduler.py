import pytest

from btcpred.config import Settings
from btcpred.ingest.service import SyncResult
from btcpred.scheduler import jobs
from btcpred.scheduler.jobs import INGEST_JOB_ID, ingest_job
from btcpred.scheduler.runner import build_scheduler


def make_settings(**overrides) -> Settings:
    """Build settings without reading .env, which does not exist in CI."""
    values = {
        "postgres_user": "u",
        "postgres_password": "p",
        "postgres_db": "d",
        "ingest_interval_minutes": 10,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_ingest_job_is_registered_with_configured_interval():
    scheduler = build_scheduler(make_settings(ingest_interval_minutes=7))
    job = scheduler.get_job(INGEST_JOB_ID)

    assert job is not None
    assert job.trigger.interval.total_seconds() == 7 * 60


def test_overlapping_runs_are_prevented():
    """A slow tick must not run concurrently with the next one."""
    job = build_scheduler(make_settings()).get_job(INGEST_JOB_ID)

    assert job.max_instances == 1
    assert job.coalesce is True


def test_first_run_is_immediate():
    """Booting should ingest right away, not after a full interval."""
    job = build_scheduler(make_settings()).get_job(INGEST_JOB_ID)

    assert job.next_run_time is not None


@pytest.mark.asyncio
async def test_ingest_job_returns_result_on_success(monkeypatch):
    expected = SyncResult(fetched=3, written=3, requests=1, latest_open_time=None)

    async def fake_run_sync(**kwargs):
        return expected

    monkeypatch.setattr(jobs, "run_sync", fake_run_sync)
    monkeypatch.setattr(jobs, "get_settings", make_settings)

    assert await ingest_job() is expected


@pytest.mark.asyncio
async def test_ingest_job_swallows_failures(monkeypatch):
    """One failed poll must not kill the scheduler; the next tick recovers."""

    async def boom(**kwargs):
        raise RuntimeError("binance is down")

    monkeypatch.setattr(jobs, "run_sync", boom)
    monkeypatch.setattr(jobs, "get_settings", make_settings)

    assert await ingest_job() is None
