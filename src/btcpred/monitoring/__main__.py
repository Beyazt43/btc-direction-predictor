"""Run a drift check on demand, or show recent ones.

python -m btcpred.monitoring check     # score both models now and record the result
python -m btcpred.monitoring history   # the last few recorded checks
"""

import argparse
import asyncio
import logging

from btcpred.config import get_settings
from btcpred.db.session import get_engine, session_scope
from btcpred.ingest.binance import interval_to_timedelta
from btcpred.monitoring.drift import check_model, latest_checks, record_check
from btcpred.predict.service import MODEL_NAMES


def _fmt(value: float | None, width: int = 6) -> str:
    return f"{value:{width}.4f}" if value is not None else " " * (width - 3) + "n/a"


async def cmd_check(args: argparse.Namespace) -> int:
    settings = get_settings()
    interval = interval_to_timedelta(settings.binance_interval)
    worst = "ok"
    async with session_scope() as session:
        for name in MODEL_NAMES:
            check = await check_model(
                session, name, symbol=settings.binance_symbol, interval=interval
            )
            if not args.dry_run:
                await record_check(session, check)
            print(
                f"{name:8} {check.status:12} observed={check.observed.n:4d} "
                f"acc={_fmt(check.observed.accuracy)} maj={_fmt(check.observed.majority_baseline)} "
                f"| reference={check.reference.n:5d} acc={_fmt(check.reference.accuracy)} "
                f"| z={_fmt(check.z_score)}"
            )
            print(f"         {check.reason}")
            if check.status == "alert":
                worst = "alert"
    return 3 if worst == "alert" else 0


async def cmd_history(args: argparse.Namespace) -> int:
    async with session_scope() as session:
        rows = await latest_checks(session, limit=args.limit)
    if not rows:
        print("no drift checks recorded yet")
        return 0
    for r in rows:
        print(
            f"{r['checked_at']:%Y-%m-%d %H:%M}  {r['model_name']:8} {r['status']:12} "
            f"obs n={r['observed_n']:4d} acc={_fmt(r['observed_accuracy'])}  "
            f"ref n={r['reference_n']:5d} acc={_fmt(r['reference_accuracy'])}  "
            f"z={_fmt(r['z_score'])}"
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift monitoring.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run the drift check now")
    check.add_argument("--dry-run", action="store_true", help="compute but do not record")
    check.set_defaults(func=cmd_check)

    history = sub.add_parser("history", help="show recent recorded checks")
    history.add_argument("--limit", type=int, default=10)
    history.set_defaults(func=cmd_history)

    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    async def run() -> int:
        try:
            return await args.func(args)
        finally:
            await get_engine().dispose()

    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
