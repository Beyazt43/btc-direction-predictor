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


class _FakeScalarSession:
    def __init__(self, values):
        self._values = list(values)

    async def scalar(self, stmt):
        return self._values.pop(0)


def _scope_with(values):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def scope():
        yield _FakeScalarSession(values)

    return scope


@pytest.mark.asyncio
async def test_cold_start_does_not_replay_a_missed_cron_on_its_own():
    """The bug that motivated catch-up: a fresh in-memory store schedules the
    *next* 02:00 and knows nothing about the one that was missed."""
    from datetime import UTC, datetime

    from btcpred.scheduler.runner import RETRAIN_JOB_ID

    scheduler = build_scheduler(make_settings())
    scheduler.start(paused=True)
    try:
        nxt = scheduler.get_job(RETRAIN_JOB_ID).next_run_time
        assert nxt > datetime.now(UTC), "a cold start schedules the future run only"
    finally:
        scheduler.shutdown(wait=False)


@pytest.mark.asyncio
async def test_stale_daily_jobs_are_pulled_forward_on_boot(monkeypatch, tmp_path):
    from datetime import UTC, datetime, timedelta

    from btcpred.scheduler import runner
    from btcpred.scheduler.runner import (
        BACKUP_JOB_ID,
        DRIFT_JOB_ID,
        RETRAIN_JOB_ID,
        schedule_catch_up,
    )

    now = datetime(2026, 9, 15, 9, 0, tzinfo=UTC)
    # Last retrain 26h ago, last drift check never, no backups on disk.
    monkeypatch.setattr(runner, "session_scope", _scope_with([now - timedelta(hours=26), None]))

    settings = make_settings(backup_dir=tmp_path)
    scheduler = build_scheduler(settings)
    scheduler.start(paused=True)
    try:
        pulled = await schedule_catch_up(scheduler, now=now, settings=settings)
        assert pulled == [RETRAIN_JOB_ID, DRIFT_JOB_ID, BACKUP_JOB_ID]
        assert scheduler.get_job(RETRAIN_JOB_ID).next_run_time == now + runner.CATCH_UP_DELAY
        assert scheduler.get_job(DRIFT_JOB_ID).next_run_time == now + runner.CATCH_UP_DELAY * 3
    finally:
        scheduler.shutdown(wait=False)


@pytest.mark.asyncio
async def test_fresh_daily_jobs_are_left_on_their_cron(monkeypatch, tmp_path):
    from datetime import UTC, datetime, timedelta

    from btcpred.backup.service import MANIFEST, Manifest
    from btcpred.scheduler import runner
    from btcpred.scheduler.runner import DRIFT_JOB_ID, RETRAIN_JOB_ID, schedule_catch_up

    now = datetime(2026, 9, 15, 9, 0, tzinfo=UTC)
    monkeypatch.setattr(
        runner, "session_scope", _scope_with([now - timedelta(hours=3), now - timedelta(hours=2)])
    )
    recent = tmp_path / "btcpred-20260915T060000Z"
    recent.mkdir()
    (recent / MANIFEST).write_text(
        Manifest(
            created_at=(now - timedelta(hours=3)).isoformat(),
            alembic_revision="x",
            rows={},
            sha256={},
        ).to_json()
    )

    settings = make_settings(backup_dir=tmp_path)
    scheduler = build_scheduler(settings)
    scheduler.start(paused=True)
    try:
        before = {
            RETRAIN_JOB_ID: scheduler.get_job(RETRAIN_JOB_ID).next_run_time,
            DRIFT_JOB_ID: scheduler.get_job(DRIFT_JOB_ID).next_run_time,
        }
        assert await schedule_catch_up(scheduler, now=now, settings=settings) == []
        for job_id, t in before.items():
            assert scheduler.get_job(job_id).next_run_time == t
    finally:
        scheduler.shutdown(wait=False)


def test_stale_threshold_keeps_a_daily_cadence_on_a_daytime_machine():
    """A machine on from 09:00 each day: a 09:30 catch-up must count as due
    again at 09:00 next morning, which 24h would miss and 20h catches."""
    from datetime import timedelta

    from btcpred.scheduler.runner import STALE_AFTER

    assert timedelta(hours=23, minutes=30) > STALE_AFTER
    assert timedelta(hours=12) < STALE_AFTER


@pytest.mark.asyncio
async def test_stale_backup_is_pulled_forward_on_boot(monkeypatch, tmp_path):
    """A machine never on at 03:30 UTC would otherwise never back up at all."""
    from datetime import UTC, datetime, timedelta

    from btcpred.scheduler import runner
    from btcpred.scheduler.runner import BACKUP_JOB_ID, schedule_catch_up

    now = datetime(2026, 9, 23, 9, 0, tzinfo=UTC)
    settings = make_settings(backup_dir=tmp_path)
    # Retrain and drift are fresh; only the backup should move.
    monkeypatch.setattr(
        runner, "session_scope", _scope_with([now - timedelta(hours=2), now - timedelta(hours=1)])
    )

    scheduler = build_scheduler(settings)
    scheduler.start(paused=True)
    try:
        pulled = await schedule_catch_up(scheduler, now=now, settings=settings)
        assert pulled == [BACKUP_JOB_ID], "no backups on disk means the backup is due"
        assert scheduler.get_job(BACKUP_JOB_ID).next_run_time == now + runner.CATCH_UP_DELAY * 4
    finally:
        scheduler.shutdown(wait=False)


@pytest.mark.asyncio
async def test_recent_backup_is_left_alone(monkeypatch, tmp_path):
    from datetime import UTC, datetime, timedelta

    from btcpred.backup.service import MANIFEST, Manifest
    from btcpred.scheduler import runner
    from btcpred.scheduler.runner import BACKUP_JOB_ID, schedule_catch_up

    now = datetime(2026, 9, 23, 9, 0, tzinfo=UTC)
    recent = now - timedelta(hours=3)
    path = tmp_path / "btcpred-20260923T060000Z"
    path.mkdir()
    (path / MANIFEST).write_text(
        Manifest(created_at=recent.isoformat(), alembic_revision="x", rows={}, sha256={}).to_json()
    )

    settings = make_settings(backup_dir=tmp_path)
    monkeypatch.setattr(
        runner, "session_scope", _scope_with([now - timedelta(hours=2), now - timedelta(hours=1)])
    )

    scheduler = build_scheduler(settings)
    scheduler.start(paused=True)
    try:
        before = scheduler.get_job(BACKUP_JOB_ID).next_run_time
        assert await schedule_catch_up(scheduler, now=now, settings=settings) == []
        assert scheduler.get_job(BACKUP_JOB_ID).next_run_time == before
    finally:
        scheduler.shutdown(wait=False)
