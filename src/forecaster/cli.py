"""Command-line entry point: `forecaster --help`.

Later phases add commands here (backfill, update, features, train, backtest, serve).
"""

from __future__ import annotations

from pathlib import Path

import typer
from alembic import command
from alembic.config import Config

from forecaster.config import get_settings
from forecaster.db.repository import sync_symbols
from forecaster.db.session import session_scope
from forecaster.logging_setup import configure_logging, get_logger

app = typer.Typer(help="Read-only Binance price forecasting. Never places orders.", no_args_is_help=True)
db_app = typer.Typer(help="Database management.", no_args_is_help=True)
app.add_typer(db_app, name="db")

log = get_logger("forecaster.cli")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@app.callback()
def _main() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_format)


def _alembic_config() -> Config:
    return Config(str(PROJECT_ROOT / "alembic.ini"))


@db_app.command("upgrade")
def db_upgrade(revision: str = "head") -> None:
    """Apply migrations up to REVISION (default: head)."""
    command.upgrade(_alembic_config(), revision)
    log.info("migrations applied", extra={"revision": revision})


@db_app.command("downgrade")
def db_downgrade(revision: str = typer.Argument(..., help="Target revision, e.g. -1 or base")) -> None:
    """Roll migrations back to REVISION."""
    command.downgrade(_alembic_config(), revision)
    log.info("migrations rolled back", extra={"revision": revision})


@db_app.command("sync-symbols")
def db_sync_symbols() -> None:
    """Insert/update the symbols table from the SYMBOLS / TARGET_SYMBOLS settings."""
    s = get_settings()
    with session_scope() as session:
        rows = sync_symbols(session, s.all_symbols, s.target_symbols)
        names = [r.symbol for r in rows]
    log.info("symbols synced", extra={"symbols": names, "targets": s.target_symbols})


@db_app.command("init")
def db_init() -> None:
    """Run migrations and sync symbols in one step."""
    db_upgrade("head")
    db_sync_symbols()


if __name__ == "__main__":
    app()
