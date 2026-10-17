"""Application settings, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

VALID_INTERVALS = {"1h", "4h", "1d"}

CsvStr = Annotated[list[str], NoDecode]
CsvInt = Annotated[list[int], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    log_format: str = "console"

    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "forecaster"
    db_password: SecretStr = SecretStr("")
    db_name: str = "crypto_forecast"
    database_url: SecretStr | None = None

    symbols: CsvStr = Field(default_factory=lambda: ["SOLUSDT", "BTCUSDT", "ETHUSDT"])
    target_symbols: CsvStr = Field(default_factory=lambda: ["SOLUSDT"])
    intervals: CsvStr = Field(default_factory=lambda: ["1d", "4h"])
    backfill_start: date = date(2020, 8, 1)
    binance_base_url: str = "https://api.binance.com"
    binance_futures_base_url: str = "https://fapi.binance.com"
    binance_data_dump_url: str = "https://data.binance.vision"

    forecast_horizons: CsvInt = Field(default_factory=lambda: [7, 14])
    model_dir: Path = Path("artifacts/models")

    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    # Never required. Only ever read from the environment; never logged.
    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None

    @field_validator("symbols", "target_symbols", "intervals", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("forecast_horizons", mode="before")
    @classmethod
    def _split_csv_int(cls, v: object) -> object:
        if isinstance(v, str):
            return [int(s) for s in v.split(",") if s.strip()]
        return v

    @field_validator("symbols", "target_symbols")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        return [s.upper() for s in v]

    @field_validator("intervals")
    @classmethod
    def _check_intervals(cls, v: list[str]) -> list[str]:
        bad = set(v) - VALID_INTERVALS
        if bad:
            raise ValueError(f"Unsupported intervals {sorted(bad)}; allowed: {sorted(VALID_INTERVALS)}")
        return v

    @field_validator("forecast_horizons")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if any(h <= 0 for h in v):
            raise ValueError("Forecast horizons must be positive day counts")
        return v

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url is not None:
            return self.database_url.get_secret_value()
        pwd = quote_plus(self.db_password.get_secret_value())
        return (
            f"mysql+pymysql://{self.db_user}:{pwd}@{self.db_host}:{self.db_port}/"
            f"{self.db_name}?charset=utf8mb4"
        )

    @property
    def all_symbols(self) -> list[str]:
        """Configured symbols plus any target symbol not already listed, order preserved."""
        return list(dict.fromkeys([*self.symbols, *self.target_symbols]))


@lru_cache
def get_settings() -> Settings:
    return Settings()
