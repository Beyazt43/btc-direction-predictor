# Runbook: move the running system to a different host

Use this when the live system has to move machines — a laptop swap, a move to a VPS, a rebuild.

**The thing that matters:** the repo is replaceable and `price_bars` is re-downloadable, but the
`predictions` table is not. Every row in it was committed before its outcome was knowable, and the
system never back-fills. A migration that loses it loses the only genuinely out-of-sample record the
project has. Everything below exists to protect that table.

## What actually has to move

| | how | irreplaceable? |
|---|---|---|
| source code | `git clone` | no |
| `.env` | copy of `.env.example` (no secrets beyond a local Postgres password) | no |
| `price_bars` | in the dump, but also re-downloadable from Binance | no |
| **`predictions`, `model_versions`, `drift_checks`** | **`pg_dump` only** | **yes** |
| model artifacts (`models` volume) | not in git, not in the dump — **retrain on the new host** | no |

Model artifacts are the trap. They live in a Docker named volume, so a fresh host has an empty one
while the restored registry still points at their paths. Predictions then fail quietly — caught and
logged by `predict_job`, with no rows written. Step 9 fixes this and must not be skipped.

## A note on backups

Since the nightly backup job exists, `BACKUP_DIR` on the old host already holds verified snapshots of
exactly the tables this runbook protects. Restoring one on the new host is an alternative to the
`pg_dump` route below:

```bash
docker compose up -d db
docker compose run --rm scheduler alembic upgrade head
docker compose run --rm scheduler python -m btcpred.backup restore <name> --yes-destroy-current-data
```

The dump route is still the default here because it captures the database as of the moment you cut
over, whereas the newest backup may be up to a day old — and during a migration, a day is the whole
point.

## Preconditions on the new host

- Docker Desktop (or Docker Engine) and Git installed.
- Docker set to start on login, or the stack will not come back after a reboot.
- Enough disk for the database (a few hundred MB) — the artifacts are under 400 KB.

## Phase 1 — prepare the new host (no downtime; the old host keeps predicting)

```bash
git clone https://github.com/Beyazt43/btc-direction-predictor.git
cd btc-direction-predictor
cp .env.example .env          # Windows: copy .env.example .env
docker compose build
docker compose pull db
```

Nothing here touches the old host. Do it at leisure.

## Phase 2 — cutover (this is the only gap in the record)

**On the OLD host**, take a dump and stop the stack:

```bash
mkdir -p migrate
docker compose exec -T db pg_dump -U btcpred -d btcpred --no-owner --no-acl > migrate/btcpred_dump.sql
docker compose exec -T db psql -U btcpred -d btcpred -c "SELECT (SELECT count(*) FROM price_bars) bars, (SELECT count(*) FROM predictions) preds, (SELECT count(*) FROM model_versions) versions, (SELECT count(*) FROM drift_checks) drift;"
docker compose stop
```

Write those four counts down. Use `stop`, never `down -v` — the old volumes are the rollback.

Copy `migrate/btcpred_dump.sql` to the new host (a few MB; USB, cloud drive, `scp`, anything).

**On the NEW host**, restore into an empty database *before* anything runs migrations:

```bash
docker compose up -d db                                    # only Postgres
docker compose cp btcpred_dump.sql db:/tmp/dump.sql
docker compose exec -T db psql -U btcpred -d btcpred -f /tmp/dump.sql
```

`docker compose cp` avoids shell redirection, which PowerShell does not support for input.

The dump carries `alembic_version` at head, so the `migrate` service will later find nothing to do.
That is expected, not a failure.

Confirm the counts match what you wrote down:

```bash
docker compose exec -T db psql -U btcpred -d btcpred -c "SELECT (SELECT count(*) FROM price_bars) bars, (SELECT count(*) FROM predictions) preds, (SELECT count(*) FROM model_versions) versions, (SELECT count(*) FROM drift_checks) drift;"
```

**Retrain, to populate the empty artifact volume** (see the trap above):

```bash
docker compose run --rm scheduler python -m btcpred.models train
```

This registers new versions and activates them through the usual gate. Pre-migration versions stay in
the registry as retired with artifact paths that no longer resolve — harmless, unless you later try to
roll back to one.

Start everything:

```bash
docker compose up -d
```

## Verification

```bash
docker compose ps                                          # db, api, scheduler all healthy
curl -s http://localhost:8000/health                       # scheduler_ok true, counts as restored
docker compose logs scheduler --since 5m | grep -E "ingested|resolved|predicted"
```

On the dashboard at <http://localhost:8000>, the rolling chart must show the **full history**, not a
fresh start. If it begins today, the restore did not take — stop, and go to rollback.

Within a few minutes of boot you should also see the catch-up lines (`is stale ... running in`)
followed by a retrain and a drift check, because a machine that is off overnight never sees the
02:00/03:00 UTC cron times.

## Rollback

The old host still has its volumes and is one command from running:

```bash
docker compose up -d        # on the OLD host
```

Do not run `docker compose down -v` on the old host until the new one has been predicting cleanly for
a few days. That flag deletes the volumes, and with them the only copy of anything the dump missed.

## Host settings worth checking

- Sleep disabled on AC; lid-close set to do nothing if the lid will be shut.
- Battery charge limit (60–80%) in BIOS if available — constant 100% is what kills aging laptop
  batteries, and a swollen battery is a genuine hazard.
- Windows Update active hours set so a reboot does not land mid-day.

Thermals are not a concern: the tick is ~2 seconds every 2 minutes and the daily retrain is ~60
seconds, so the machine idles almost all the time.

## Note on host uptime

The README records that a host running only during the day censors the live sample — roughly 11 of 24
hours, with 01:00–06:00 UTC never observed. Moving to a machine that stays on closes that hole and is
the main reason to bother. If the new host will also be intermittent, the bias note in the README
stays accurate and should not be quietly dropped.
