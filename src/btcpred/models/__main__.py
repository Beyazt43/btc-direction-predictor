"""Model management from the command line.

python -m btcpred.models train                       # daily retrain: both models, gated
python -m btcpred.models train --model arima         # one model
python -m btcpred.models train --force-activate      # bypass the gate
python -m btcpred.models train --exclude-holdout     # offline path for the §8 evaluation
python -m btcpred.models versions                    # registry listing
python -m btcpred.models activate xgboost <version>  # rollback / manual promotion
"""

import argparse
import asyncio
import logging

from btcpred.config import get_settings
from btcpred.db.session import get_engine, session_scope
from btcpred.models.pipeline import format_report, train_all, train_model
from btcpred.models.registry import activate_version, list_versions


async def cmd_train(args: argparse.Namespace) -> int:
    settings = get_settings()
    common = {
        "settings": settings,
        "include_holdout": not args.exclude_holdout,
        "gated": not args.force_activate,
        "n_configs": args.n_configs,
        "seed": args.seed,
    }
    async with session_scope() as session:
        if args.model == "all":
            results = await train_all(session, **common)
        else:
            results = {args.model: await train_model(session, args.model, **common)}

    for trained in results.values():
        print(format_report(trained))
        print()

    # Non-zero when anything was refused, so a scheduler can see it in the
    # exit code without parsing output.
    return 0 if all(t.gate.activate for t in results.values()) else 2


async def cmd_versions(args: argparse.Namespace) -> int:
    async with session_scope() as session:
        rows = await list_versions(session, args.model)
    if not rows:
        print("no versions registered")
        return 0
    for r in rows:
        state = (
            "active"
            if r["activated_at"] and not r["retired_at"]
            else ("retired" if r["retired_at"] else "never activated")
        )
        pooled = (r.get("reference_metrics") or {}).get("pooled") or {}
        ll = pooled.get("log_loss")
        acc = pooled.get("accuracy")
        print(
            f"{r['model_name']:8} {r['model_version']:40} {state:16} "
            f"trained={r['trained_at']:%Y-%m-%d %H:%M}  "
            f"acc={acc:.4f} ll={ll:.5f}"
            if ll is not None
            else f"{r['model_name']:8} {r['model_version']:40} {state:16} "
            f"trained={r['trained_at']:%Y-%m-%d %H:%M}"
        )
    return 0


async def cmd_activate(args: argparse.Namespace) -> int:
    async with session_scope() as session:
        await activate_version(session, args.model, args.version)
    print(f"activated {args.model} {args.version}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, list, and activate direction models.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train and register (the daily retrain)")
    train.add_argument("--model", choices=["all", "arima", "xgboost"], default="all")
    train.add_argument("--n-configs", type=int, default=30, help="GBT search budget (§7: ~30)")
    train.add_argument("--seed", type=int, default=None, help="search seed (default: today's date)")
    train.add_argument(
        "--exclude-holdout",
        action="store_true",
        help="train on the development set only; the offline path for the §8 evaluation",
    )
    train.add_argument(
        "--force-activate", action="store_true", help="activate regardless of the gate"
    )
    train.set_defaults(func=cmd_train)

    versions = sub.add_parser("versions", help="list registered versions")
    versions.add_argument("--model", choices=["arima", "xgboost"], default=None)
    versions.set_defaults(func=cmd_versions)

    activate = sub.add_parser("activate", help="make a registered version live (rollback)")
    activate.add_argument("model", choices=["arima", "xgboost"])
    activate.add_argument("version")
    activate.set_defaults(func=cmd_activate)

    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
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
