# BTC/USD Next-Hour Direction Predictor — Project Context

> Handoff document. Captures architecture decisions made during design discussion, before implementation.
> Status: **design phase complete through evaluation methodology.** Feature engineering onward still open.

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
    model_name          TEXT NOT NULL,              -- 'sarimax' | 'xgboost'
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

### Baseline: SARIMAX
- **Target: log-return** `ln(close(t+1)/close(t))` — **DECIDED.**
  - Stationary(-ish), which matters for ARIMA/SARIMAX assumptions.
  - Maps directly to direction (`> 0` → up) with no price-level reconstruction.
- Directional call = forecast-then-threshold at 0.

### Challenger: XGBoost / LightGBM
- **Classifies the binary label directly** (native discriminative training on the task).

### The asymmetry is intentional
SARIMAX = forecast-then-threshold; GBT = classify directly. Do **not** artificially force both into the same paradigm. The asymmetry reflects *why* these two families are being compared: one is a classical statistical forecaster repurposed for a directional call, the other is trained natively on it. Narrate this.

### Both emit comparable probabilities
SARIMAX gives a forecast *distribution*, so `P(log-return > 0)` falls out of the forecast mean and standard error. This means both models can be compared on **log loss and AUC**, not just thresholded accuracy — a substantially richer head-to-head.

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
| **Log loss / AUC-ROC** | On probabilities; enables fair SARIMAX-vs-GBT comparison |

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

1. **Feature engineering** — contents of `build_feature_row(as_of_time)`; lag structure; what the GBT sees that SARIMAX doesn't.
2. **Model layer implementation detail** — SARIMAX order selection & refit cadence; GBT hyperparameter space; how both get versioned.
3. **Retraining job** — daily cadence; expanding vs. sliding window in production; what triggers an off-schedule retrain; model versioning & rollback.
4. **Drift detection** — concrete thresholds using §8's CI numbers; what triggers an *alert* vs. a *retrain*; whether to also monitor feature drift (PSI / KS) in addition to performance drift.
5. **Service layer** — FastAPI structure, endpoints, dashboard for live accuracy + baseline-vs-GBT comparison + accuracy-by-move-magnitude panel.

---

## 10. Build-Order Note

Because this runs on **real-time data**, get ingestion + prediction logging live **early** — before the dashboard, before the comparison writeup. Prediction history accumulates in wall-clock time and cannot be back-generated honestly. Every day the live loop isn't running is a day of drift-monitoring data permanently lost.
