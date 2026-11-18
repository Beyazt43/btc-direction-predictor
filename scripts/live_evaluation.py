"""Regenerate docs/live_evaluation.txt from the committed live log.

The README's live result was computed from the running database, which no
longer exists. This recomputes it from docs/live_log/ with the project's own
scoring code (`evaluate`, `mcnemar_test`) and the same definition of an honest
live call the dashboard used: per model and target hour, the earliest
prediction made before that hour's bar closed, counted once it has resolved.

The log is frozen, so this is deterministic and safe to re-run:

    docker compose run --rm --no-deps \
        -v "$PWD/docs:/app/docs" -v "$PWD/scripts:/app/scripts" \
        scheduler python scripts/live_evaluation.py > docs/live_evaluation.txt
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from btcpred.models.metrics import evaluate, mcnemar_test

LOG = Path("docs/live_log")
INTERVAL = pd.Timedelta(hours=1)
# The README's "Live result" section was computed on Nov 18 with this hour as
# the newest resolved one; reproducing it from the frozen log proves the table.
SNAPSHOT_LAST_HOUR = pd.Timestamp("2026-11-18 12:00", tz="UTC")
MODELS = ("arima", "xgboost")
LN2 = math.log(2)


def read(name: str) -> pd.DataFrame:
    # The backup writes NULL as \N (Postgres COPY convention).
    return pd.read_csv(LOG / f"{name}.csv.gz", na_values=[r"\N"], keep_default_na=False)


def live_calls(preds: pd.DataFrame) -> pd.DataFrame:
    p = preds.copy()
    for col in ("target_open_time", "predicted_at"):
        p[col] = pd.to_datetime(p[col], utc=True, format="ISO8601")
    p = p[(p["predicted_at"] < p["target_open_time"] + INTERVAL) & p["actual_direction"].notna()]
    # Whole earliest row per (model, hour), as the database's DISTINCT ON does;
    # groupby().first() would take the first non-null value per column.
    p = p.sort_values("predicted_at")
    p = p.drop_duplicates(["model_name", "target_open_time"], keep="first")
    wide = p.pivot(index="target_open_time", columns="model_name")
    wide = wide.dropna(subset=[("actual_direction", m) for m in MODELS])
    return wide.sort_index()


def gain_z(gains: np.ndarray) -> tuple[float, float]:
    mean = float(gains.mean())
    return mean, mean / (float(gains.std(ddof=1)) / math.sqrt(len(gains)))


def per_call_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def report(title: str, wide: pd.DataFrame) -> None:
    y = wide[("actual_direction", "arima")].to_numpy(dtype=int)
    n = len(y)
    print(f"\n== {title} ==")
    first, last = (f"{t:%Y-%m-%d %H:%M}" for t in (wide.index[0], wide.index[-1]))
    majority_class = "up" if y.mean() >= 0.5 else "down"
    print(f"paired live hours   {n}   ({first} .. {last} UTC)")
    print(f"hours closed up     {y.mean():.4f}   (majority class: {majority_class})")
    print()
    print(
        f"{'model':8} {'accuracy':>8} {'majority':>8} {'edge':>8} {'±2SE':>6} {'MCC':>7}"
        f" {'log loss':>9} {'AUC':>6} {'up-calls':>8}"
    )
    ll = {}
    for m in MODELS:
        pred = wide[("predicted_direction", m)].to_numpy(dtype=int)
        proba = wide[("predicted_proba", m)].to_numpy(dtype=float)
        ev = evaluate(y, pred, proba=proba)
        ll[m] = per_call_log_loss(y, proba)
        edge = ev.edge_over_majority * 100
        band = 2 * ev.standard_error * 100
        print(
            f"{m:8} {ev.accuracy:8.4f} {ev.majority_baseline:8.4f} {edge:+7.2f}pp"
            f" {band:5.1f}pp {ev.mcc:+7.4f} {ev.log_loss:9.5f} {ev.auc:6.4f}"
            f" {ev.predicted_up_rate:8.4f}"
        )
    print()
    for m in MODELS:
        mean, z = gain_z(LN2 - ll[m])
        print(
            f"log-loss gain over ln 2, {m:8} {mean:+.5f}   z = {z:+.2f}"
            "   (paired per call, one-sided)"
        )
    mean, z = gain_z(ll["arima"] - ll["xgboost"])
    print(f"xgboost log loss below arima      {mean:+.5f}   z = {z:+.2f}")
    mc = mcnemar_test(
        y,
        wide[("predicted_direction", "arima")].to_numpy(dtype=int),
        wide[("predicted_direction", "xgboost")].to_numpy(dtype=int),
    )
    print(
        f"McNemar: arima-only correct {mc['a_only_correct']:.0f}, xgboost-only correct"
        f" {mc['b_only_correct']:.0f}, p = {mc['p_value']:.3f}"
    )


def main() -> None:
    manifest = json.loads((LOG / "manifest.json").read_text())
    print("Live evaluation receipt — btc-direction-predictor")
    print(f"source: backup {manifest['source_backup']}, alembic {manifest['alembic_revision']}")
    for name, digest in sorted(manifest["sha256"].items()):
        actual = hashlib.sha256((LOG / name).read_bytes()).hexdigest()
        status = "ok" if actual == digest else "MISMATCH"
        rows = manifest["rows"][name.split(".")[0]]
        print(f"  {name:22} sha256 {status}  rows {rows}")

    preds = read("predictions")
    unresolved = int(preds["actual_direction"].isna().sum())
    print(f"prediction rows {len(preds)}; unresolved at stop {unresolved} (excluded)")

    wide = live_calls(preds)
    snapshot = wide[wide.index <= SNAPSHOT_LAST_HOUR]
    report("Snapshot reproduced by the README 'Live result' section", snapshot)
    report("Final: the whole frozen live log", wide)

    drift = read("drift_checks").sort_values("checked_at").groupby("model_name").tail(1)
    print("\n== Last drift check per model ==")
    for _, r in drift.iterrows():
        print(
            f"{r['model_name']:8} {r['checked_at'][:16]}  {r['status']:10}"
            f" observed {r['observed_accuracy']:.4f} (n={r['observed_n']})"
            f"  reference {r['reference_accuracy']:.4f} (n={r['reference_n']})"
            f"  z = {r['z_score']:+.2f}"
        )


if __name__ == "__main__":
    main()
