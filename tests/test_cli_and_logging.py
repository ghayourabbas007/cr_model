from __future__ import annotations

import json
import logging

import pytest
from typer.testing import CliRunner

from forecaster.cli import app
from forecaster.logging_setup import ConsoleFormatter, JsonFormatter

runner = CliRunner()


def test_help_lists_every_command_group() -> None:
    out = runner.invoke(app, ["--help"]).output
    for cmd in ("db", "data", "features", "models", "train", "predict", "backtest", "serve", "dashboard"):
        assert cmd in out


def test_db_init_and_data_gaps_on_sqlite(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'cli.db').as_posix()}")
    monkeypatch.setenv("SYMBOLS", "SOLUSDT,BTCUSDT")
    monkeypatch.setenv("CROSS_ASSET_SYMBOLS", "BTCUSDT")
    res = runner.invoke(app, ["db", "init"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["data", "gaps"])
    assert res.exit_code == 0 and "SOLUSDT 1d: 0 gap(s)" in res.output
    res = runner.invoke(app, ["models", "list"])
    assert res.exit_code == 0


def _record(**extra) -> logging.LogRecord:
    rec = logging.LogRecord("forecaster.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_json_logs_are_structured_and_mask_secrets() -> None:
    line = JsonFormatter().format(_record(symbol="SOLUSDT", db_password="hunter2", api_key="abc"))
    payload = json.loads(line)
    assert payload["msg"] == "hello world" and payload["symbol"] == "SOLUSDT" and payload["level"] == "INFO"
    assert payload["db_password"] == "***" and payload["api_key"] == "***"
    assert "hunter2" not in line and "abc" not in line


def test_console_logs_include_extras() -> None:
    line = ConsoleFormatter().format(_record(rows=12))
    assert "hello world" in line and "rows=12" in line
