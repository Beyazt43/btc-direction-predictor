"""Run ingestion once from the command line.

python -m btcpred.ingest                 # catch up from the newest stored candle
python -m btcpred.ingest --since 2024-01-01   # backfill history
"""

import argparse
import asyncio
import logging
from datetime import UTC, datetime

from btcpred.config import get_settings
from btcpred.ingest.service import run_sync


def _parse_since(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest Binance klines into price_bars.")
    parser.add_argument(
        "--since",
        type=_parse_since,
        default=None,
        help="ISO date/time to backfill from (default: resume from newest stored candle)",
    )
    parser.add_argument("--symbol", default=settings.binance_symbol)
    parser.add_argument("--interval", default=settings.binance_interval)
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    result = asyncio.run(
        run_sync(
            symbol=args.symbol,
            interval=args.interval,
            base_url=settings.binance_base_url,
            start=args.since,
        )
    )
    print(
        f"fetched={result.fetched} written={result.written} "
        f"requests={result.requests} latest={result.latest_open_time}"
    )


if __name__ == "__main__":
    main()
