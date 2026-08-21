"""Entrypoint for the scheduler container: `python -m btcpred.scheduler`."""

import asyncio
import logging

from btcpred.config import get_settings
from btcpred.scheduler.runner import run_forever


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    asyncio.run(run_forever(settings))


if __name__ == "__main__":
    main()
