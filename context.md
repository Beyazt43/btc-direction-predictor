# BTC/USD Next-Hour Direction Predictor — Project Context

> Handoff document. Captures architecture decisions made during design discussion, before implementation.
> Status: **the live loop is running; the next design step is the retraining job (§9 item 3).**
> Built: compose stack (db / migrate / scheduler), `config.py`, Alembic migrations for all three tables,
> Binance ingestion (idempotent, self-healing, closed-candles-only), the feature builder and label
> construction, both models with walk-forward evaluation, the `model_versions` registry with artifacts,
> and the scheduler tick that runs ingest → resolve → predict every 2 minutes.
> **Predictions have been logging since 2026-09-14**, so downtime now loses data that cannot be
> back-generated (§10). Always-on hosting is due, not deferred.
> Not yet built: the `api` service (compose points at a `btcpred.api.main:app` that does not exist),
> retraining job, drift detection, dashboard.
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
- Comfortable: FastAPI, PostgreSQL (asyncpg / SQLModel), Docker + docker-compose, Alembic migrations.
- Learning: ML fundamentals (Microsoft ML-For-Beginners, IBM ML Professional Certificate — clustering complete). Recently covered classical time-series: ARIMA / SARIMAX / SVR.
- Implication: infrastructure code can be idiomatic and assume competence. ML-specific choices benefit from explicit reasoning.
- Working preference: **step-by-step, decision-by-decision.** Do not dump full solutions.

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

### Reference vs. observed
- **Reference** = walk-forward accuracy, stored with each model version.
- **Observed** = rolling accuracy computed from the `predictions` table.
- **Drift** = divergence between them, tested against the CI table above.

---

## 9. Still Open — Next Design Steps

Work through these in order, same step-by-step mode:

1. **Feature engineering** — contents of `build_feature_row(as_of_time)`; lag structure; what the GBT sees that ARIMA doesn't.
2. ~~**Model layer implementation detail**~~ — **RESOLVED, see §7.** Fixed ARIMA(1,0,0) with coefficients refit each retrain, no seasonal terms (so the honest name is **ARIMA**, not SARIMA or SARIMAX); focused ~30-config random search over shallow trees for the GBT; both versioned in the `model_versions` registry (§4) with explicit `activated_at`/`retired_at` validity windows and artifacts on a Docker named volume.
3. **Retraining job** — daily cadence; expanding vs. sliding window in production; what triggers an off-schedule retrain; model versioning & rollback. **Carries a constraint from §5:** model versions and their validity windows must stay recoverable, or missed predictions can never be back-generated honestly.
4. **Drift detection** — concrete thresholds using §8's CI numbers; what triggers an *alert* vs. a *retrain*; whether to also monitor feature drift (PSI / KS) in addition to performance drift.
5. **Service layer** — FastAPI structure, endpoints, dashboard for live accuracy + baseline-vs-GBT comparison + accuracy-by-move-magnitude panel. **Carries a constraint from §5:** live and back-generated predictions must be reported as separate lines, using the shared criterion rather than an ad hoc filter per query.

**Two decisions deferred on purpose, to be settled when the prediction job is built** (both matter only once something is being predicted):

- ~~**Poll cadence** (§3)~~ — **done**, 2 minutes.
- **Always-on hosting** (§10) — **now due.** Prediction history started accruing on 2026-09-14. The resource profile is known: ARIMA artifact ~350 bytes, XGBoost ~320KB, a full retrain with the 30-config search ~50s. Nothing about the workload requires more than the smallest always-on box.

---

## 10. Build-Order Note

Because this runs on **real-time data**, get ingestion + prediction logging live **early** — before the dashboard, before the comparison writeup. Prediction history accumulates in wall-clock time and cannot be back-generated honestly. Every day the live loop isn't running is a day of drift-monitoring data permanently lost.
