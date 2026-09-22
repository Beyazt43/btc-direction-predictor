# BTC/USD Next-Hour Direction Predictor — Project Context

> Handoff document. Captures architecture decisions made during design discussion, before implementation.
> Status: **every §9 design step is resolved and `docker compose up` brings up the full stack.**
> Built: compose stack (db / migrate / scheduler / api, all healthchecked, `unless-stopped`), `config.py`,
> Alembic migrations for all four tables, Binance ingestion (idempotent, self-healing,
> closed-candles-only), the feature builder and label construction, both models with walk-forward
> evaluation, the `model_versions` registry with artifacts, the scheduler tick that runs
> ingest → resolve → predict every 2 minutes, the daily retrain with gated activation and rollback,
> the frozen §8 holdout window with its one-time `evaluate-holdout` command, daily drift checks
> recorded to `drift_checks`, and the read-only API with the live dashboard.
> **Predictions have been logging since 2026-09-14**, so downtime loses data that cannot be
> back-generated (§10). Always-on hosting is due, not deferred.
> **The one-time §8 holdout evaluation was run on 2026-09-15** (receipt: `docs/holdout_evaluation.txt`).
> Nothing remains on the design list; the live log is accumulating.
---

## 1. Project Goal

Build an MLOps-flavored portfolio project: a BTC/USD **next-hour price direction** predictor, framed as a *living system* rather than a one-off model.

**Explicitly NOT the goal:** a profitable trading bot. Price prediction at this horizon is genuinely hard; near-random accuracy is the expected outcome and is reported honestly.

**Actual goals:**
1. Demonstrate infrastructure competence — ingestion, retraining, prediction logging, drift monitoring, deployment.
2. Demonstrate a real **model-comparison narrative** — classical time-series baseline vs. gradient-boosted trees, with reasoning about *why* each family behaves as it does on non-stationary, sentiment-driven crypto data, not just two accuracy numbers.

**Deployment mode:** runs against **real-time data, continuously**. Start the live system as early as possible in the build sequence to begin banking prediction history before the dashboard exists.

---

## 2. Builder Background (calibration for code style / explanation depth)

- Backend-leaning CS student pivoting to AI/ML Engineering.
- Comfortable with: FastAPI, PostgreSQL (asyncpg / SQLModel), Docker + docker-compose, Alembic migrations.

---

## 3. Data Source & Ingestion

### Source: Binance public REST API — `/api/v3/klines`

Chosen over CoinGecko because:
- Native OHLCV candles at exact intervals (no aggregation across exchanges).
- No API key required for public market data.
- Standard source for crypto ML work — reads better in a portfolio.

Pull **1-hour klines directly.** Do not pull 1-minute and resample. (Sub-hour data for intra-hour volatility features is a v2 idea, not a launch requirement.)

### Scheduling

- **APScheduler in its own dedicated container**, separate from the FastAPI API container.
  - Rationale to narrate in writeup: separation of concerns — ingestion/retraining failures don't affect API uptime; independent restart/scaling.
- Deliberately **not** Airflow/Prefect — overengineering for this scope.
- **Ingestion cadence:** poll every 5–15 min for the latest *closed* hourly candle. Binance returns in-progress candles too, so filter `close_time < now()`.
- Websockets are unnecessary complexity for an hourly target.

### Ingestion is idempotent

`UNIQUE (symbol, open_time)` + `ON CONFLICT DO NOTHING` (or `DO UPDATE` to allow late candle corrections). Scheduler double-fires and backfills are safe.

### Poll cadence erodes the prediction horizon — **RESOLVED: `INGEST_INTERVAL_MINUTES=2`**

The cadence above has a consequence that only bites once predictions are being generated. Bar `t` closes at `:59:59.999`, but at a 10-minute poll it may not be *detected* for another 10 minutes. The prediction for `t+1` is therefore logged up to a sixth of the way into the very hour it predicts.

This is **not leakage** — no data from `t+1` is used. But "next-hour predictor" then means, in practice, the remaining ~50 minutes, which is the kind of detail a careful reader will catch in the writeup.

Tightening `INGEST_INTERVAL_MINUTES` to 1–2 closes the gap for almost nothing: `/api/v3/klines` costs weight 2 per request against a generous limit. Set to 2 minutes when the prediction job was built. Observed live: the 07:00 bar was ingested and the 08:00 prediction logged 43 seconds after the candle closed.

---

## 4. Schema

```sql
CREATE TABLE price_bars (
    id            BIGSERIAL PRIMARY KEY,
    symbol        TEXT NOT NULL DEFAULT 'BTCUSDT',
    open_time     TIMESTAMPTZ NOT NULL,
    close_time    TIMESTAMPTZ NOT NULL,
    open          NUMERIC(18,8) NOT NULL,
    high          NUMERIC(18,8) NOT NULL,
    low           NUMERIC(18,8) NOT NULL,
    close         NUMERIC(18,8) NOT NULL,
    volume        NUMERIC(24,8) NOT NULL,
    quote_volume  NUMERIC(24,8),
    num_trades    INTEGER,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    source        TEXT NOT NULL DEFAULT 'binance',
    UNIQUE (symbol, open_time)
);
```

```sql
CREATE TABLE predictions (
    id                  BIGSERIAL PRIMARY KEY,
    model_name          TEXT NOT NULL,              -- 'arima' | 'xgboost'
    model_version       TEXT NOT NULL,              -- hash or retrain timestamp
    target_open_time    TIMESTAMPTZ NOT NULL,       -- the hour being predicted
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    predicted_direction SMALLINT NOT NULL,          -- 1 = up, 0 = down
    predicted_proba     DOUBLE PRECISION,
    actual_direction    SMALLINT,                   -- filled once the hour closes
    actual_log_return   DOUBLE PRECISION,           -- realized return, for sliced analysis
    resolved_at         TIMESTAMPTZ,
    UNIQUE (model_name, model_version, target_open_time)
);
```

```sql
CREATE TABLE model_versions (
    id                BIGSERIAL PRIMARY KEY,
    model_name        TEXT NOT NULL,              -- 'arima' | 'xgboost'
    model_version     TEXT NOT NULL,
    trained_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    train_start       TIMESTAMPTZ NOT NULL,       -- reproducibility + leakage audit
    train_end         TIMESTAMPTZ NOT NULL,
    activated_at      TIMESTAMPTZ,                -- validity window (§5)
    retired_at        TIMESTAMPTZ,
    artifact_path     TEXT NOT NULL,
    hyperparameters   JSONB,
    feature_names     JSONB,                      -- train/serve skew detection
    reference_metrics JSONB,                      -- §8 reference walk-forward metrics
    UNIQUE (model_name, model_version),
    CHECK (train_end > train_start),
    CHECK (retired_at IS NULL OR activated_at IS NOT NULL)
);

-- At most one live version per model: a second active row makes "which version
-- was live at hour T" ambiguous exactly when it matters.
CREATE UNIQUE INDEX uq_model_versions_one_active_per_model ON model_versions (model_name)
    WHERE activated_at IS NOT NULL AND retired_at IS NULL;

-- Every logged prediction must name a registered version, or its validity
-- window is unknown and §5's back-generation rule cannot be applied.
ALTER TABLE predictions ADD CONSTRAINT fk_predictions_model_version
    FOREIGN KEY (model_name, model_version)
    REFERENCES model_versions (model_name, model_version);
```

```sql
CREATE TABLE drift_checks (
    id                 BIGSERIAL PRIMARY KEY,
    checked_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_name         TEXT NOT NULL,
    window_start       TIMESTAMPTZ NOT NULL,       -- observed: last 30 days of live calls
    window_end         TIMESTAMPTZ NOT NULL,
    observed_n         INTEGER NOT NULL,
    observed_accuracy  DOUBLE PRECISION,
    majority_baseline  DOUBLE PRECISION,
    reference_n        INTEGER NOT NULL,           -- reference: every live call before the window
    reference_accuracy DOUBLE PRECISION,
    z_score            DOUBLE PRECISION,
    status             TEXT NOT NULL,              -- ok | alert | warming_up | insufficient
    reason             TEXT,
    feature_psi        JSONB,                      -- diagnostic, never an alert source
    CHECK (status IN ('ok','alert','warming_up','insufficient'))
);
```

### Schema design rationale (for the writeup)

- **`UNIQUE (symbol, open_time)`** → idempotent ingestion.
- **`open_time` and `close_time` stored separately** → needed to be airtight about "what info was actually available at prediction time." A candle can still be updating until `close_time` passes.
- **`ingested_at` separate from `open_time`** → leakage tripwire / audit trail. If a backtest ever used a row whose `ingested_at` postdates when it should have been available, that's a bug.
- **`actual_log_return` on predictions** → enables post-hoc sliced accuracy (see §6, dead-zone decision).

Alembic migrations for both tables.

---

## 5. Target Construction

### Label definition

For a bar with `open_time = t`:

```
direction(t) = 1 if close(t+1h) > close(t) else 0
```

Strictly binary. No neutral class (see §6).

### Prediction lifecycle (order matters — this is the leakage guard)

1. Bar for hour `t` closes on Binance.
2. Ingestion job pulls it, confirmed closed (`close_time(t) < now()`).
3. **Only then** generate prediction for hour `t+1`, using `close(t)` and everything prior as the most recent known point.
4. Log prediction with `target_open_time = t+1`.
5. One hour later, bar `t+1` closes and is ingested → resolve: set `actual_direction`, `actual_log_return`, `resolved_at`.

Each newly closed bar triggers both (a) resolution of the pending prediction for that hour and (b) generation of the next prediction.

### Leakage prevention — **APPROVED, core architectural constraint**

Build a single strict function:

```python
def build_feature_row(as_of_time): ...
```

- It queries **only** `price_bars WHERE open_time <= as_of_time`.
- **Training-set construction and live inference call this same function identically.**
- Structural guarantee: training cannot see anything live inference couldn't.
- Writeup framing: *"single source of truth for feature construction, shared between training and serving, to eliminate train/serve skew and lookahead bias."*

**Known trap to guard against:** careless pandas `.rolling()` / `.shift()`. A `shift(-1)` instead of `shift(1)` invalidates an entire backtest. Also: never let any feature touch `high(t+1)` or `low(t+1)` — leaks future info even if `close` handling is correct.

### Downtime and back-generated predictions — **RESOLVED: no schema change needed**

The system will not always be running (see §10). Missed hours therefore need a defensible story, and the existing schema already provides one.

The outcome of hour `T` becomes knowable at `T + 1h`, once `close(T)` is final. So:

```
honest live prediction  ⟺  predicted_at < target_open_time + interval
```

Anything logged later was written when the answer was already visible. This is exactly why §4 stores `predicted_at` separately from `target_open_time` — **no extra column and no migration are required** to tell the two apart.

**Consequences for the build:**

- Encode the rule once, as a shared helper or SQL view. Re-deriving the comparison ad hoc in each dashboard query is how the definitions drift apart.
- Report live and backfilled accuracy as **separate lines**, never pooled. An honest gap in the live record is more credible than a suspiciously unbroken one.
- A missed prediction may only be back-generated using **the model version that was live at that hour**. Reconstructing it with a later retrain leaks future data into a past prediction — the model has by then seen data from after `target_open_time`. This constrains the retraining job (§9 item 3): model versions and their validity windows must be recoverable, or back-generation is off the table entirely.

---

## 6. Dead-Zone Decision — **RESOLVED: strict binary, instrument for it instead**

### Decision
No neutral/flat class. Every hour is up or down.

### Rationale
A fixed dead zone (e.g. ±0.05%) is **non-stationary in effect**. In calm regimes it swallows ~40% of hours; in volatile regimes ~5%. Class distribution would then shift with volatility regime — permanently confounding *model degradation* with *market got calmer*. Since the entire monitoring layer is built on "accuracy dropped → drift," this would make drift alerts uninterpretable. It directly undermines why classification was chosen in the first place.

### What replaces it
The dead zone's legitimate benefit — removing label noise from unpredictable micro-moves — is obtained **analytically instead of architecturally**:

- Store `actual_log_return` alongside `actual_direction`.
- Compute sliced accuracy post-hoc as a dashboard view, e.g. *accuracy on hours where |return| > 0.1%*.
- Yields a better dashboard panel than a third class would: **accuracy as a function of move magnitude.**

### Future work (README line, not build now)
If a dead zone is wanted properly, use a **volatility-normalized** threshold (±0.25σ of recent realized vol) rather than a fixed percentage — regime-stable.

### Base rate caveat
**The base rate is not 50%.** BTC has historical upward drift; hourly up-moves may run 50.5–52%. Compute this on the training set and report it — "always predict up" is a real competitor.

---

## 7. Model Layer

### Baseline: ARIMA(1,0,0) — univariate, **no exogenous regressors, no seasonal terms (DECIDED)**

The baseline sees only its own log-return history. Rationale: it keeps the comparison legible — classical linear time-series structure against learned nonlinear structure over richer inputs — so any GBT edge is attributable to model family and inputs together, rather than to a partial overlap in what each model was fed.

**The honest name is ARIMA — not SARIMAX, and not SARIMA.** The `X` is the exogenous regressor and there is none; the `S` is the seasonal component and order selection dropped that too (see below). Both letters would claim capabilities the model never exercises, which is exactly what an interviewer probes. `model_name` is therefore `'arima'`.

- **Target: log-return** `ln(close(t+1)/close(t))` — **DECIDED.**
  - Stationary(-ish), which matters for ARIMA-family assumptions.
  - Maps directly to direction (`> 0` → up) with no price-level reconstruction.
- Directional call = forecast-then-threshold at 0.

### Order selection — **DECIDED: fixed ARIMA(1,0,0), coefficients refit each retrain**

`d = 0`. The target is already log-*returns*, i.e. log-price differenced once. Differencing again over-differences and injects a spurious MA(1) coefficient near −0.5.

The order is chosen **once**, on an early window, and only coefficients are refit thereafter. An order that churns between retrains makes versions incomparable: an accuracy change could be the market or could be the structure moving, with no way to tell which.

No seasonal terms. Lag-24 return autocorrelation measured −0.0064, below the 0.0076 noise floor, and `s=24` would grow the statsmodels state space enough to turn each candidate fit from seconds into many minutes inside a daily retrain.

### The baseline finds no structure — **and that is the result**

AIC/BIC grid over `p,q ∈ 0..3` on the first 8,000 bars (holdout untouched):

| order | AIC | BIC | ΔAIC |
|---|---|---|---|
| (0,0,3) | −61527.74 | −61492.81 | +0.00 |
| (3,0,0) | −61527.74 | −61492.80 | +0.00 |
| (1,0,0) | −61526.93 | −61505.97 | +0.82 |
| **(0,0,0)** | −61521.78 | **−61507.81** | +5.96 |

The whole 16-model spread is 12 AIC units and the top eight fall within 2.5 of each other — conventionally indistinguishable. **BIC selects white noise.**

Out-of-sample directional accuracy over the following 2,000 bars (always-up = 49.95%, SE = 1.12pp): (0,0,3) 49.30%, (1,0,1) 48.80%, (2,0,2) 49.75%, (1,0,0) 50.10%, (0,0,0) 49.95%. The **AIC winner is the second-worst directional performer** — likelihood-based selection does not transfer to a directional call.

**Why (1,0,0) and not (0,0,0):** ARIMA(0,0,0) has a positive fitted mean, so it predicts "up" 100% of the time — it *is* the majority-class baseline §8 already tracks separately, and adopting it would collapse the two-model comparison into one. (1,0,0) is the smallest non-degenerate choice: within 0.82 AIC of the best and 1.84 BIC of white noise, which is a tie under conventional reading.

This is not a disappointing result to be buried. §1 commits to reporting near-random outcomes honestly, and "classical linear time-series modelling finds no exploitable structure in BTC hourly returns, and BIC says so explicitly" is a stronger writeup line than a tuned baseline that happens to land at 51%.

### Challenger: XGBoost / LightGBM
- **Classifies the binary label directly** (native discriminative training on the task).
- **Hyperparameter search — DECIDED:** focused random search, ~30 configs, shallow trees (`max_depth` 2–4, high `min_child_weight`, strong subsampling and regularisation). Sized to a target with R² ≈ 0.004, where the risk is fitting noise rather than underfitting. Cheap enough to rerun at every retrain instead of tuning once and pretending it still holds.
- **No class weighting.** At a 50.36% base rate the imbalance is nil; `scale_pos_weight` would solve a problem that does not exist.

### The asymmetry is intentional
ARIMA = forecast-then-threshold; GBT = classify directly. Do **not** artificially force both into the same paradigm. The asymmetry reflects *why* these two families are being compared: one is a classical statistical forecaster repurposed for a directional call, the other is trained natively on it. Narrate this.

### Both emit comparable probabilities
ARIMA gives a forecast *distribution*, so `P(log-return > 0)` falls out of the forecast mean and standard error. This means both models can be compared on **log loss and AUC**, not just thresholded accuracy — a substantially richer head-to-head.

---

## 8. Evaluation Methodology — **APPROVED**

Evaluation and drift monitoring are the same measurement taken at two points. Designed together.

### Splitting
- **Expanding-window walk-forward.** Train `[0, t]`, test `[t+gap, t+h]`, roll forward, aggregate fold metrics.
- **No random splits, no shuffled k-fold, no `train_test_split`.**
- **Embargo gap = 1 bar minimum.** The label for bar `t` depends on `close(t+1)`, so without a gap the final training sample's *label* reaches into the test period. Small at a 1-hour horizon but free to fix. Writeup phrase: *"purging and embargo between train and test folds."*
- **Sliding-window ablation:** a fixed 6-month training window that moves, run as a variant. The comparison — does old data help or hurt? — is a regime-awareness point. Costs one config flag.

### The holdout result — **RUN ONCE, 2026-09-15**

Window `[2026-07-16, 2026-09-14)`, n = 1,440; training strictly before it (16,511 rows), search confined to that data; seed 0, 30 configs; registry untouched. Receipt in `docs/holdout_evaluation.txt`.

| | walk-forward | holdout | majority | MCC | log loss | AUC | edge | clears 2SE (2.6pp) |
|---|---|---|---|---|---|---|---|---|
| ARIMA(1,0,0) | 0.5123 | 0.5215 | 0.5083 | +0.043 | 0.69281 | 0.529 | +1.3pp | no |
| XGBoost | 0.5204 | 0.5396 | 0.5083 | +0.079 | 0.69037 | 0.545 | +3.1pp | **yes** (z ≈ 2.4) |

ARIMA behaves like the baseline it is measured against, as the order selection predicted. XGBoost clears the bar on this window — the first number in the project to do so — but the walk-forward estimate of the same edge over 12,830 rows is +1.2pp, so the honest reading is *probably real, probably small, this window favourable*. Both holdout figures exceeded walk-forward; a decoy window checked beforehand showed walk-forward ≈ holdout, so this is window variance, not inflation. A holdout that collapsed relative to walk-forward would have been the leakage signal; it did not. The live log sizes the edge from here.

### Three-way discipline
1. Walk-forward folds → hyperparameter selection.
2. Final **chronological holdout** (most recent ~2 months) → touched **exactly once**, at the end.

Otherwise you overfit to the validation scheme itself.

### Metrics
| Metric | Role |
|---|---|
| **Accuracy** | Primary; feeds drift detection |
| **Majority-class baseline** | Report alongside — "always up" is a real competitor |
| **Persistence baseline** | Predict same direction as previous hour |
| **Random baseline** | Sanity floor |
| **Matthews correlation coefficient (MCC)** | Robust to class imbalance; near-zero MCC at 52% accuracy is the honest tell that nothing is being learned |
| **Log loss / AUC-ROC** | On probabilities; enables fair ARIMA-vs-GBT comparison |

If neither model beats "always up," **report that honestly** — it makes the project more credible, not less.

### Significance testing
**McNemar's test** for model-vs-model comparison — correct test for two classifiers on the same test set with paired binary outcomes. Prevents narrating 52.3% vs 51.6% as a real difference when it's noise.

### Sample-size constraint (drives drift thresholds)

At true accuracy ≈ 0.50, SE = `√(0.25/n)`:

| Window | n | SE | 95% CI |
|---|---|---|---|
| 1 week | 168 | 3.9% | ±7.6% |
| 2 weeks | 336 | 2.7% | ±5.3% |
| 30 days | 720 | 1.9% | ±3.7% |
| 90 days | 2160 | 1.1% | ±2.1% |

**Consequence:** a rolling 1-week live accuracy *cannot* detect a 3-point degradation — the noise band is twice the effect. A naive "alert if live accuracy < training accuracy − 3%" on a weekly window would be near-pure false alarms.

→ **30-day rolling window is the primary drift signal.** Shorter windows may appear on the dashboard but must be clearly labeled *indicative only*.

"Why the drift threshold is what it is" is a strong README paragraph — most portfolio projects skip this reasoning.

### Reference vs. observed — **REVISED when built (2026-09-14)**
- **Observed** = accuracy over the last 30 days of *live* calls (`is_live`), one per target hour.
- **Reference** = accuracy over every live call *before* that window — **the live log's own history, not the walk-forward number.** The stored XGBoost walk-forward figure is the maximum over a 30-config search and reads high; measured against it, the model would look like it was drifting from day one. Comparing the log to its own history sidesteps the bias entirely and is the ordinary definition of concept drift. Cost: for the first 30 days only the sanity floor applies, which is all the SE table above says a short window can support anyway.
- **Drift** = pooled two-proportion z, one-sided, alert at **z < −2**. At n=720 that is ~3.7pp, matching the table; a 3-point drop does not reach it, by design. An improvement is never an alert.
- **Sanity floor** from week one regardless of history: alert if the window's accuracy is 2σ below its own majority baseline.
- **Alert, never retrain.** Daily retraining already caps staleness at 24h; a 30-day signal fires long after any fix has shipped. What an alert asks is *did the world move or did the model*, and that is what feature PSI is attached for.

### Feature drift — **DECIDED: diagnostic on alerts, never an alert source**
PSI per feature against the active version's training deciles (stored with each version), computed every check and surfaced only when a performance alert fires. Real data shows why it cannot be an alert: `rv_168` and `ret_mean_168` read PSI 1.2–1.8 *every day*, because 720 values of a 168-hour rolling statistic are about four independent samples and cannot match a two-year distribution whatever the regime — while `close_pos` sits at 0.01. The conventional 0.1/0.2 thresholds apply only to fast features; for slow ones the value is the *ranking*, day over day.

---

## 9. Still Open — Next Design Steps

Work through these in order, same step-by-step mode:

1. **Feature engineering** — contents of `build_feature_row(as_of_time)`; lag structure; what the GBT sees that ARIMA doesn't.
2. ~~**Model layer implementation detail**~~ — **RESOLVED, see §7.** Fixed ARIMA(1,0,0) with coefficients refit each retrain, no seasonal terms (so the honest name is **ARIMA**, not SARIMA or SARIMAX); focused ~30-config random search over shallow trees for the GBT; both versioned in the `model_versions` registry (§4) with explicit `activated_at`/`retired_at` validity windows and artifacts on a Docker named volume.
3. ~~**Retraining job**~~ — **RESOLVED.** Daily at `RETRAIN_CRON` (02:00 UTC). **A missed run is caught up on boot by checking staleness, not by APScheduler's misfire grace** — the grace only covers a blocked event loop, and with an in-memory job store a cold start simply schedules the next 02:00 with no memory of the missed one. On boot, any daily job whose last output is older than 20h is pulled forward (retrain at +2min, drift at +6min), so a machine that is only on in the daytime still gets a daily retrain and a daily drift check. **Production trains on all data** — the 60-day holdout is a writeup device evaluated once by a model trained only on data before it, and the live prediction log is the real out-of-sample test for anything deployed. **The incumbent's hyperparameters are always seeded into the search**, so a bad random draw can never regress the deployed configuration; that reduces the activation gate to two catastrophe checks (worse than a coin flip at log loss ≥ 0.70, or worse than the incumbent's reference by > 0.01), both an order of magnitude wider than day-to-day noise. **No off-schedule trigger**: daily cadence caps staleness at 24h while §8's drift signal is a 30-day window, so drift should raise an alert (item 4), not a retrain. Rollback is `python -m btcpred.models activate <model> <version>`. Retrain runs as a subprocess so CPU-bound fitting cannot stall the 2-minute tick. The §5 constraint is met: `model_versions` records validity windows and every prediction carries a FK to its version.
4. ~~**Drift detection**~~ — **RESOLVED, see §8.** Reference is the live log's own history (not the selection-biased walk-forward number); 30-day window, one-sided 2σ; sanity floor against the majority baseline from week one; alert never retrains; PSI is a diagnostic attached to alerts and never an alert source. Daily at 03:00 UTC, one `drift_checks` row per model per day.
5. ~~**Service layer**~~ — **RESOLVED.** Read-only FastAPI: every write path stays on the CLI, and a test asserts no route accepts anything but GET. All numbers come from live resolved calls through the one shared `live_calls` subquery (earliest call per target hour) and the same `evaluate`/`mcnemar_test` the training pipeline uses, so the dashboard, the drift job and the training report cannot disagree. Endpoints: `/health`, `/metrics/live` (7d indicative / 30d / all), `/metrics/live/{model}/daily`, `/metrics/comparison` (paired, McNemar), `/metrics/by-magnitude` (fixed bp buckets, n per bucket), `/drift`, `/versions`. Dashboard is one Jinja2 page with Chart.js from a CDN: rolling accuracy with the ±2SE band around the majority baseline, the windows table, the paired comparison, magnitude bars, drift history. The §5 constraint is met by construction: only `is_live` rows are ever counted, and nothing back-generates.

**Two decisions deferred on purpose, to be settled when the prediction job is built** (both matter only once something is being predicted):

- ~~**Poll cadence** (§3)~~ — **done**, 2 minutes.
- **Always-on hosting** (§10) — **now due.** Prediction history started accruing on 2026-09-14. The resource profile is known: ARIMA artifact ~350 bytes, XGBoost ~320KB, a full retrain with the 30-config search ~50s. Nothing about the workload requires more than the smallest always-on box.

---

## 10. Build-Order Note

> **Observed 2026-09-21:** with the host running only during the day, the live log captures ~11 of 24 hours
> and 01:00–06:00 UTC has never been sampled. Measured over the full history, that block is 49.9% up
> against 50.5% for the sampled hours (inside the noise band) but ~15% less volatile (42.4 vs 49.8 bp),
> so the bias is second-order; the binding cost is power — at 11 calls/day the SE shrinks ~1.5× more
> slowly than under continuous operation, and an edge the size of the walk-forward estimate needs
> ~7,000 calls to clear 2σ.

> **Observed 2026-09-14:** Docker Desktop stopped twice in one session on the dev machine. `restart: unless-stopped` recovered the containers within seconds each time, but the second outage still cost the 10:00 and 11:00 live predictions permanently — the system does not back-fill, by design. The host, not drift, is currently the largest threat to the live log.

Because this runs on **real-time data**, get ingestion + prediction logging live **early** — before the dashboard, before the comparison writeup. Prediction history accumulates in wall-clock time and cannot be back-generated honestly. Every day the live loop isn't running is a day of drift-monitoring data permanently lost.
