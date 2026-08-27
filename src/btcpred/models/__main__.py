"""Train models from the command line.

python -m btcpred.models                    # train both, activate them
python -m btcpred.models --model arima      # just the baseline
python -m btcpred.models --no-activate      # register without going live
"""

import argparse
import asyncio
import logging

from btcpred.config import get_settings
from btcpred.db.session import get_engine, session_scope
from btcpred.models.pipeline import format_report, train_all, train_model


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    async with session_scope() as session:
        if args.model == "all":
            results = await train_all(
                session,
                settings=settings,
                n_configs=args.n_configs,
                activate=not args.no_activate,
            )
        else:
            trained = await train_model(
                session,
                args.model,
                settings=settings,
                n_configs=args.n_configs,
                activate=not args.no_activate,
            )
            results = {args.model: trained}

    for trained in results.values():
        print(format_report(trained))
        print()

    await get_engine().dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register direction models.")
    parser.add_argument("--model", choices=["all", "arima", "xgboost"], default="all")
    parser.add_argument(
        "--n-configs",
        type=int,
        default=30,
        help="GBT random search budget (context.md §7 sets this at ~30)",
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="register the version without making it live",
    )
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
