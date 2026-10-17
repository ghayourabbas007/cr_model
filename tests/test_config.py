from __future__ import annotations

import pytest
from pydantic import ValidationError

from forecaster.config import Settings


def test_csv_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYMBOLS", "solusdt, btcusdt ,ETHUSDT")
    monkeypatch.setenv("TARGET_SYMBOLS", "SOLUSDT,XRPUSDT")
    monkeypatch.setenv("FORECAST_HORIZONS", "7,14")
    s = Settings()
    assert s.symbols == ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
    assert s.forecast_horizons == [7, 14]
    assert s.all_symbols == ["SOLUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT"]


def test_invalid_interval_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVALS", "1d,7m")
    with pytest.raises(ValidationError):
        Settings()


def test_url_from_parts_escapes_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p@ss:word")
    monkeypatch.setenv("DB_NAME", "x")
    url = Settings().sqlalchemy_url
    assert url.startswith("mysql+pymysql://u:p%40ss%3Aword@")
    assert url.endswith("/x?charset=utf8mb4")


def test_database_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///foo.db")
    assert Settings().sqlalchemy_url == "sqlite:///foo.db"


def test_secrets_not_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_PASSWORD", "hunter2")
    monkeypatch.setenv("BINANCE_API_SECRET", "topsecret")
    text = repr(Settings())
    assert "hunter2" not in text
    assert "topsecret" not in text


def test_api_keys_optional() -> None:
    s = Settings()
    assert s.binance_api_key is None
    assert s.binance_api_secret is None
