"""Command-line entry point: `forecaster --help`.

Later phases add commands here (backfill, update, features, train, backtest, serve).
"""

from __future__ import annotations

from enum import Enum
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
data_app = typer.Typer(help="Market-data ingestion (public endpoints only).", no_args_is_help=True)
app.add_typer(data_app, name="data")
features_app = typer.Typer(help="Feature engineering and leakage checks.", no_args_is_help=True)
app.add_typer(features_app, name="features")

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


class BackfillSource(str, Enum):
    auto = "auto"
    dump = "dump"
    api = "api"


@data_app.command("backfill")
def data_backfill(
    symbol: list[str] = typer.Option(None, "--symbol", "-s", help="Repeatable. Default: all configured."),
    interval: list[str] = typer.Option(None, "--interval", "-i", help="Repeatable. Default: INTERVALS."),
    source: BackfillSource = typer.Option(BackfillSource.auto, help="auto = dumps + API; dump; api"),
) -> None:
    """Load full history from BACKFILL_START. Idempotent; safe to re-run."""
    from forecaster.data.pipeline import run_backfill

    syms = [x.upper() for x in symbol] if symbol else None
    reports = run_backfill(get_settings(), syms, interval or None, source.value)
    _print_reports(reports)


@data_app.command("update")
def data_update() -> None:
    """Fetch new closed candles for all configured symbols and repair recent gaps."""
    from forecaster.data.pipeline import run_update

    _print_reports(run_update(get_settings()))


@data_app.command("context")
def data_context() -> None:
    """Update funding rate and open interest (USD-M futures)."""
    from forecaster.data.pipeline import run_context

    for row in run_context(get_settings()):
        typer.echo(row)


@data_app.command("gaps")
def data_gaps() -> None:
    """List missing candle ranges in stored data."""
    from forecaster.data.pipeline import report_gaps

    for (sym, interval), gaps in report_gaps(get_settings()).items():
        typer.echo(f"{sym} {interval}: {len(gaps)} gap(s)")
        for start, end in gaps:
            typer.echo(f"    {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M}")


@data_app.command("schedule")
def data_schedule(run_now: bool = typer.Option(True, help="Run one update immediately on start.")) -> None:
    """Run the incremental updater on a cron schedule (blocks)."""
    from forecaster.data.scheduler import run_scheduler

    run_scheduler(get_settings(), run_now)


@features_app.command("build")
def features_build(
    symbol: list[str] = typer.Option(None, "--symbol", "-s", help="Repeatable. Default: TARGET_SYMBOLS."),
) -> None:
    """Build daily features from stored klines and save them to the features table."""
    from forecaster.features.pipeline import run_build_features

    syms = [x.upper() for x in symbol] if symbol else None
    for name, n in run_build_features(get_settings(), syms).items():
        typer.echo(f"{name}: {n} feature rows stored")


@features_app.command("check-leakage")
def features_check_leakage(
    symbol: str = typer.Option("SOLUSDT", "--symbol", "-s"),
    cutoffs: int = typer.Option(5, help="Number of truncation points to test."),
) -> None:
    """Rebuild features from data truncated at several cutoffs; fail if any value changes."""
    from forecaster.features.build import LeakageError
    from forecaster.features.pipeline import run_leakage_check

    try:
        tested = run_leakage_check(get_settings(), symbol.upper(), cutoffs)
    except LeakageError as exc:
        typer.secho(f"LEAKAGE DETECTED: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"OK: no look-ahead at {len(tested)} cutoffs "
                f"({tested[0]:%Y-%m-%d} .. {tested[-1]:%Y-%m-%d %H:%M})", fg=typer.colors.GREEN)


def _print_reports(reports) -> None:
    for r in reports:
        typer.echo(f"{r.symbol:<10} {r.interval:<3} dump={r.rows_dump:<6} api={r.rows_api:<6} "
                   f"gap_fill={r.rows_gap_fill:<4} invalid={r.invalid_dropped:<3} "
                   f"unfilled_gaps={len(r.unfilled_gaps)}")


if __name__ == "__main__":
    app()
