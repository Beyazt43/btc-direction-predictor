# BTC/USD next-hour direction — a living ARIMA-vs-XGBoost comparison

A small MLOps system that predicts whether the next hourly BTC/USDT candle closes up or down, and then keeps itself honest about it: it ingests live data, predicts every hour, scores every prediction against what actually happened, retrains daily behind an activation gate, and monitors its own live record for drift.

**This is not a trading bot, and it does not claim to beat the market.** Hourly BTC direction is close to unpredictable, and the system says so — the base rate over two years is 50.36% up, which is statistically indistinguishable from a coin flip, and the classical baseline's order selection concludes the return series is white noise. The point of the project is the machinery around that honest result: the leakage guards, the evaluation discipline, the retraining and monitoring loop, and a model comparison that is argued rather than just tabulated.

The design reasoning lives in [`context.md`](context.md). This README is the tour.

---

## What is running

Every two minutes, one scheduler tick does three things in a fixed order:

```
ingest   → pull any newly closed 1h candle from Binance (idempotent, self-healing)
resolve  → score every pending prediction whose target hour has now closed
predict  → from the newest closed bar, predict the next hour with each live model
```

The order is the leakage guard: a prediction for hour `t+1` is only ever made from a bar `t` that ingestion has confirmed closed, and the hour that just closed is scored before the next one is called. Predictions land within about a minute of the candle close.

Once a day, a retrain runs both models on all available data and activates the new versions if they pass a sanity gate; an hour later, a drift check scores the last 30 days of live calls against the live log's own earlier history. Nothing is ever back-filled: an hour the scheduler was down for stays empty in the record, and is visibly empty on the dashboard.

## Quickstart

```bash
cp .env.example .env
docker compose up -d
```

That starts Postgres, runs migrations, and brings up the scheduler and the API. The scheduler ingests 500 candles on first boot and starts predicting once a model exists:

```bash
docker compose run --rm scheduler python -m btcpred.ingest --since 2024-08-20   # backfill history (18 requests, ~20s)
docker compose run --rm scheduler python -m btcpred.models train                 # train, register and activate both models
```

Then open **http://localhost:8000** for the dashboard, or **http://localhost:8000/docs** for the API.

### CLI

| command | what it does |
|---|---|
| `python -m btcpred.ingest [--since DATE]` | catch up from the newest stored candle, or backfill from a date |
| `python -m btcpred.models train [--model arima\|xgboost] [--force-activate]` | the daily retrain: walk-forward evaluate, fit, register, gate |
| `python -m btcpred.models versions` | the registry |
| `python -m btcpred.models activate <model> <version>` | rollback / manual promotion |
| `python -m btcpred.models evaluate-holdout` | the one-time §8 holdout number — run once |
| `python -m btcpred.monitoring check [--dry-run]` | run the drift check now |
| `python -m btcpred.monitoring history` | recent recorded checks |

The API is read-only. Every state change goes through the CLI or the scheduler, so a bug or a bad actor on the web surface cannot retrain or roll back anything.

### API

| endpoint | returns |
|---|---|
| `GET /health` | liveness as seen from the database: last bar, last prediction, active versions |
| `GET /metrics/live` | per-model accuracy over 7d (indicative) / 30d / all, with baselines, MCC, log loss, and the ±2 SE band |
| `GET /metrics/live/{model}/daily` | daily hit rate plus 30-day rolling accuracy, rolling majority baseline, rolling SE |
| `GET /metrics/comparison?days=` | ARIMA vs XGBoost on the same hours: accuracy, MCC, log loss, AUC, McNemar |
| `GET /metrics/by-magnitude?days=` | accuracy sliced by \|log return\| in basis points, with n per bucket |
| `GET /drift` | recorded drift checks |
| `GET /versions` | model registry |

Every number is computed from **live, resolved** predictions only, deduplicated to one call per target hour, through the same query the drift job uses and the same scoring functions the training pipeline uses. The dashboard, the drift monitor and the training report cannot disagree about what a metric means.

---

## Architecture

Four containers, one image:

```
db         Postgres 16. Volumes: pgdata (bars, predictions, registry), models (artifacts).
migrate    One-shot: alembic upgrade head. api and scheduler wait for it to succeed.
scheduler  APScheduler in its own process: the 2-minute tick, the 02:00 retrain, the 03:00 drift check.
api        FastAPI, read-only. Serves the JSON endpoints and the dashboard.
```

The scheduler is deliberately separate from the API so an ingestion or retraining failure cannot take down request serving, and either can be restarted independently. It is APScheduler rather than Airflow or Prefect because there are three jobs; an orchestrator would be three extra containers to run three functions. It is APScheduler rather than a bare loop because the jobs need both interval and cron triggers, overlap protection, and clean shutdown, and hand-rolling those is where bugs live.

The durability does not come from the scheduler. Its job store is in-memory and a restart forgets everything. That is fine only because every job is idempotent and self-healing — ingestion resumes from the newest stored candle, predictions are unique per (model, version, hour), resolution catches up in one statement — so the schedule is a clock and the resilience is in the job design. Both long-lived containers carry a healthcheck (the scheduler's is a heartbeat file the tick touches, so a *hung* loop reads unhealthy, not just a dead one) and `restart: unless-stopped`.

### Tables

```
price_bars      OHLCV per hour. UNIQUE (symbol, open_time) makes ingestion idempotent; the upsert is
                guarded by IS DISTINCT FROM so a re-ingested unchanged candle is a true no-op and
                ingested_at stays meaningful as an audit trail.
predictions     One row per (model, version, target hour). predicted_at is set by the database, never
                the app. actual_direction and actual_log_return are filled when the target hour closes.
model_versions  The registry: training window, hyperparameters, feature list, reference metrics,
                training-time feature deciles, and an activated_at / retired_at validity window.
                A partial unique index allows one active version per model; predictions carry a
                foreign key to it, so a prediction can never name a version whose window is unknown.
drift_checks    One row per model per day: the observed 30-day window, the reference, z, status,
                and per-feature PSI.
```

`predicted_at < target_open_time + 1h` is the single definition of an honest live prediction — written before the outcome was knowable. It lives in one function, `is_live()`, and every query that separates live from reconstructed rows goes through it. Nothing in the system currently reconstructs anything; the rule exists so that if something ever does, it cannot hide.

### The leakage guard

Feature construction is one pure function, `compute_features()`, and both training and serving go through it: training takes every row of a long frame, serving takes the last row of a 169-bar window. Train/serve skew is structurally impossible rather than merely intended, and a test asserts that the short window and the full history agree to within 1e-12 on real data.

Nothing in `features/builder.py` looks forward. The one forward-looking operation in the codebase — the `shift(-1)` that builds the label — is quarantined in `features/labels.py`, and a test parses the builder's syntax tree to assert it contains no negative shift. Every feature is scale-free (returns, ratios, z-scores); a test doubles every price and asserts no feature moves, because a feature that did would be teaching the model 2026 price levels.

The registry stores each version's feature list. If the live code's feature set ever differs from what a model was trained on, the predictor refuses to run it.

---

## The model comparison

Two model families, deliberately not forced into one paradigm:

- **ARIMA(1,0,0)** on the log-return series — a classical statistical forecaster repurposed for a directional call. It forecasts the next return with a distribution, so `P(up) = Φ(μ/σ)` falls out of the forecast mean and standard error.
- **XGBoost** on 20 engineered features — a discriminative model trained natively on the binary label.

Both emit a calibrated `P(up)`, so they can be compared on log loss and AUC rather than a bare accuracy race.

### There is no linear structure, and the baseline says so

Order selection ran an AIC/BIC grid over `p, q ∈ 0..3` on the first 8,000 bars, holdout untouched. The entire 16-model spread was 12 AIC units, and the top eight sat within 2.5 of each other — conventionally indistinguishable. **BIC selected ARIMA(0,0,0): white noise.** Out of sample, the AIC winner (0,0,3) was the second-worst directional performer, at 49.3% against 49.95% for always-up. Likelihood-based selection does not transfer to a directional call.

The deployed baseline is ARIMA(1,0,0), the smallest non-degenerate choice: ARIMA(0,0,0) has a positive fitted mean, so it predicts "up" every hour and *is* the majority-class baseline the evaluation already tracks. Adopting it would collapse the two-model comparison into one. There are no seasonal terms (lag-24 return autocorrelation is below the noise floor) and no exogenous regressors (by decision, to keep the comparison legible), which is why the model is called ARIMA and not SARIMAX — the letters would claim capabilities it never exercises.

This is the headline result, not a footnote. The classical baseline finds nothing to model, and reports that honestly.

### What the GBT sees

Twenty scale-free features across five families, with lag depth set from the data rather than a rule of thumb:

| family | features | why |
|---|---|---|
| return lags | `ret_lag_1..6` | lags 1–2 clear the noise floor on correlation with the label; 3–6 do not, but are cheap |
| return aggregates | `ret_mean_{6,24,168}` | the 6h and 24h means clear the floor where individual longer lags do not |
| volatility | `rv_{6,24,168}`, `rv_ratio_6_168` | volatility clustering is strong (|r| autocorr 0.26 at lag 1) and has a weekly cycle: autocorr at lag 168 (0.17) exceeds lag 48 (0.10) |
| volume / activity | `vol_z_{6,24,168}`, `trades_z_24` | on probation: every volume feature sits below the noise floor on linear correlation, kept because the GBT's edge is interactions |
| candle shape | `range_pct`, `body_ratio`, `close_pos` | `close_pos` — where the close sits in the bar's range — is the strongest single feature (r = −0.062), a mean-reversion signal |

Feature importance after training matched the pre-analysis: `close_pos` leads at 0.107 against 0.05 uniform, and the volume family totals 0.167 across four features, below its uniform share. Volume stays on probation; the live log will settle it.

### Results

Walk-forward over the full history (5 expanding folds, 1-bar embargo, n = 12,830 pooled test rows):

| | accuracy | majority | MCC | log loss | AUC |
|---|---|---|---|---|---|
| ARIMA(1,0,0) | 0.5138 | 0.5003 | +0.030 | 0.69276 | 0.519 |
| XGBoost | 0.5242 | 0.5003 | +0.049 | 0.69114 | 0.538 |
| *knowing nothing* | 0.5003 | — | 0 | **0.69315** | 0.5 |

Read the log-loss column first. `ln 2 = 0.69315` is what predicting 0.5 forever scores. ARIMA beats it by 0.0004; XGBoost by 0.002. Both models are barely informative, and MCC near zero at 52% accuracy is the honest tell that very little is being learned.

**The XGBoost row is optimistically biased.** It is the best of 30 searched configurations, scored on the same folds that selected it. The registry stores it with `selection_biased: true` so it is never read as clean. ARIMA has no search and no bias, so the head-to-head flatters the challenger. Only two things settle it honestly:

- **The frozen holdout** — the 60 days before the first live prediction, `2026-07-16` to `2026-09-14`, evaluated once by a model trained only on data before it. The command exists (`evaluate-holdout`); as of this writing it has not been run.
- **The live log** — the genuinely out-of-sample test, accumulating since 2026-09-14, on the dashboard.

What to expect live: the pre-analysis suggested `close_pos` alone is worth roughly 1–3 points over the base rate, and the walk-forward numbers land in that range. If the live 30-day accuracy settles around 51–53% with MCC near zero, that is the result, and it is the expected one.

---

## Evaluation discipline

- **Expanding-window walk-forward, never random splits.** Test blocks tile forward; training never overlaps or follows testing.
- **A 1-bar embargo** between train and test. The label for bar `t` depends on `close(t+1)`, so a training set ending exactly where the test set begins has already leaked its last label across the seam. Small at a 1-hour horizon; free to fix.
- **Hyperparameters are chosen on the folds only.** The holdout is pinned to a fixed date range, not a trailing count, so daily retrains (which train on everything) cannot drag it forward and quietly tune on it.
- **Baselines travel with every number:** majority class ("always up" is a real competitor with BTC's drift), persistence (repeat the previous hour), and random. MCC is reported alongside accuracy because a constant predictor can look respectable on accuracy alone.
- **McNemar's test** for model-vs-model, on paired outcomes — it looks only at the hours where the two disagreed, which is what stops 52.3% vs 51.6% being narrated as a real difference.

### Why the drift window is 30 days

At true accuracy near 0.5, the standard error of a window of `n` hourly calls is `√(0.25/n)`:

| window | n | SE | 95% CI |
|---|---|---|---|
| 1 week | 168 | 3.9% | ±7.6pp |
| 30 days | 720 | 1.9% | ±3.7pp |
| 90 days | 2,160 | 1.1% | ±2.1pp |

A one-week rolling accuracy cannot detect a 3-point degradation: its noise band is twice the effect. A naive "alert if weekly accuracy drops 3 points" would be near-pure false alarms. The 30-day window is the primary signal; the dashboard shows the 7-day one but labels it *indicative*.

---

## Retraining and drift

**Retraining** runs daily at 02:00 UTC on all available data, as a subprocess so CPU-bound fitting cannot stall the tick. The incumbent's hyperparameters are always seeded into the random search, so yesterday's configuration is in the running every day and a bad random draw can never regress the deployed one. That reduces the activation gate to two catastrophe checks — refuse if worse than a coin flip (log loss ≥ 0.70) or worse than the incumbent's reference by more than 0.01, both an order of magnitude wider than day-to-day noise. Refused versions stay registered for inspection. Rollback is one CLI command, transactional against the one-active-version-per-model index.

**Drift** is checked daily at 03:00 UTC: the last 30 days of live calls against every live call before that, pooled two-proportion z, one-sided, alert at z < −2. The reference is the live log's own history — *not* the walk-forward number, which selection inflates; measured against that, the model would look like it was drifting from day one. A sanity floor (2σ below the window's own majority baseline) applies from the first week regardless.

**An alert never triggers a retrain.** Daily cadence already caps staleness at 24 hours, and a 30-day signal fires long after any fix has shipped. What an alert asks is *did the world move or did the model*, and per-feature PSI against the active version's training deciles is attached to help answer it. PSI is a diagnostic, never an alert source: on real data, `rv_168` reads a PSI of 1.2–1.8 every day — 720 values of a 168-hour rolling statistic are about four independent samples and cannot match a two-year distribution — while `close_pos` sits at 0.01. As an alert it would fire forever.

---

## Honest limitations

- **It runs on a laptop.** Docker Desktop stopped twice during development; `unless-stopped` recovered the containers within seconds each time, but one outage still cost two hours of live predictions permanently. The system does not back-fill by design, so the gap is visible. The largest current threat to the live record is the host, not drift. Anything always-on — a small VPS, a Pi — would do; the whole workload is under 400KB of artifacts and a ~50-second daily retrain.
- **The live record is young.** As of this writing it is hours old. The dashboard reports `insufficient` rather than a verdict until there are at least a week of resolved calls, and `warming_up` until the reference has 30 days. It will be a month before the drift check can say anything, and that is correct.
- **The XGBoost walk-forward number is biased upward** by selection, as described. The holdout and the live log are the arbiters.
- **Volume features are unproven.** Kept on the hypothesis that they matter through interactions; feature importance says they contribute below their share. The live log may retire them.
- **The 7-day window exists because people ask for it,** not because it can support a conclusion.

---

## Repository

```
src/btcpred/
  config.py            pydantic-settings; everything from .env
  db/                  async engine, SQLAlchemy Core tables mirroring the migrations (alembic check verifies)
  ingest/              Binance client (Decimal prices, closed candles only), idempotent upsert, self-healing sync
  features/            builder.py (no lookahead, scale-free), labels.py (the one shift(-1)), dataset.py (the join)
  models/              ARIMA and XGBoost behind one interface; walk-forward splits; metrics; registry; training pipeline
  predict/             generate and resolve predictions; is_live() and live_calls()
  monitoring/          drift check
  scheduler/           the tick, the retrain, the drift job; heartbeat
  api/                 read-only FastAPI and the dashboard
migrations/            four Alembic revisions, hand-written, async env
tests/                 118 tests, none of which need a database or the network
context.md             the design document: every decision and the reasoning behind it
```

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Python 3.12, Postgres 16, FastAPI, SQLAlchemy 2 (async, asyncpg), Alembic, APScheduler, statsmodels, XGBoost, pandas 3.
