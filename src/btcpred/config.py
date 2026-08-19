from functools import lru_cache

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
    ingest_interval_minutes: int = 10
    retrain_cron: str = "0 2 * * *"

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
