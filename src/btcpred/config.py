from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "db"
    postgres_port: int = 5432

    # Binance
    binance_base_url: str = "https://api.binance.com"
    binance_symbol: str = "BTCUSDT"
    binance_interval: str = "1h"

    # Scheduler
    ingest_interval_minutes: int = 2
    retrain_cron: str = "0 2 * * *"

    # Models
    # Trained artifacts live on a Docker named volume, with the path recorded in
    # the model_versions registry.
    model_dir: Path = Path("models")

    # The frozen holdout window for the one-time §8 evaluation: the 60 days
    # ending at the first live prediction (2026-09-14). Pinned to dates rather
    # than a trailing count on purpose. Production retrains train on everything
    # and select hyperparameters on it, so a trailing window would drift forward
    # each day and quietly become tuned-on data. A fixed window cannot.
    holdout_start: date = date(2026, 7, 16)
    holdout_end: date = date(2026, 9, 14)

    # Touched at the end of every successful tick. The container healthcheck
    # fails if it goes stale, which is the only way a hung event loop -- as
    # opposed to a crashed process -- becomes visible to Docker.
    heartbeat_path: Path = Path("heartbeat")

    # App
    log_level: str = "INFO"
    environment: str = "development"

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
