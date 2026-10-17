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
models_app = typer.Typer(help="Model versions.", no_args_is_help=True)
app.add_typer(models_app, name="models")

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


@app.command("train")
def train(
    symbol: list[str] = typer.Option(None, "--symbol", "-s", help="Repeatable. Default: TARGET_SYMBOLS."),
    horizon: list[int] = typer.Option(None, "--horizon", help="Repeatable. Default: FORECAST_HORIZONS."),
    models: str = typer.Option(None, help="Comma-separated, e.g. lgbm,lstm. Default: TRAIN_MODELS."),
) -> None:
    """Retrain: walk-forward evaluation vs baseline, final fit, save new versions, activate."""
    from forecaster.modeling.train import train_symbol

    s = get_settings()
    names = [m.strip() for m in models.split(",")] if models else s.train_models
    for sym in [x.upper() for x in symbol] if symbol else s.target_symbols:
        for h in horizon or s.forecast_horizons:
            with session_scope() as session:
                results = train_symbol(session, s, sym, h, names)
            typer.echo(f"\n{sym}  horizon={h}d  (out-of-sample, walk-forward)")
            typer.echo(f"  {'model':<12}{'acc':>7}{'auc':>7}{'brier':>8}{'BSS':>8}{'mae50':>8}"
                       f"{'cov80':>7}  verdict")
            for r in results:
                m = r.metrics
                cmp = m.get("vs_climatology") or {}
                auc = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "  n/a"
                bss = f"{cmp['brier_skill']:+.3f}" if cmp else "      -"
                typer.echo(f"  {r.model_name:<12}{m['accuracy']:>7.3f}{auc:>7}{m['brier']:>8.4f}{bss:>8}"
                           f"{m['mae_q50']:>8.4f}{m['coverage_80']:>7.2f}  {r.verdict}")
            typer.echo(f"  always-up accuracy: {results[0].metrics['always_up_accuracy']:.3f}   "
                       f"BSS = Brier skill vs climatology (>0 is better)   cov80 nominal = 0.80")


@app.command("predict")
def predict(
    symbol: list[str] = typer.Option(None, "--symbol", "-s"),
    model: str = typer.Option(None, help="Default: SERVING_MODEL."),
    store: bool = typer.Option(True, help="Save to the predictions table."),
) -> None:
    """Forecast 7/14-day direction probability and price range from the latest closed candle."""
    from forecaster import DISCLAIMER
    from forecaster.modeling.predict import forecast, update_realized

    s = get_settings()
    for sym in [x.upper() for x in symbol] if symbol else s.target_symbols:
        with session_scope() as session:
            update_realized(session, sym)
            for h in s.forecast_horizons:
                f = forecast(session, s, sym, h, model, store=store)
                stale = "  [STALE DATA]" if f["stale"] else ""
                typer.echo(f"{sym} +{h}d from {f['as_of_time']:%Y-%m-%d} (price {f['reference_price']:.4f}) "
                           f"[{f['model']} {f['model_version']}]{stale}\n"
                           f"  P(up) = {f['prob_up']:.1%}\n"
                           f"  80% range: {f['price_q10']:.4f} .. {f['price_q90']:.4f}  "
                           f"(median {f['price_q50']:.4f})")
    typer.secho(f"\n{DISCLAIMER}", fg=typer.colors.YELLOW)


@models_app.command("list")
def models_list(symbol: str = typer.Option(None, "--symbol", "-s")) -> None:
    """Show model versions with their walk-forward metrics."""
    from forecaster.db.repository import get_symbol
    from forecaster.modeling.predict import list_runs

    with session_scope() as session:
        sym_id = None
        if symbol:
            sym = get_symbol(session, symbol)
            sym_id = sym.id if sym else -1
        for r in list_runs(session, sym_id):
            m = r.metrics or {}
            auc = m.get("roc_auc")
            typer.echo(f"{r.id:>5} {'*' if r.is_active else ' '} {r.model_name:<12} h={r.horizon_days:<3}"
                       f"{r.version}  brier={m.get('brier', float('nan')):.4f}  "
                       f"auc={auc if auc is None else round(auc, 3)}  cov80={m.get('coverage_80', float('nan')):.2f}")


@models_app.command("activate")
def models_activate(run_id: int) -> None:
    """Make a specific model version active (e.g. roll back)."""
    from forecaster.db.models import ModelRun
    from forecaster.modeling.registry import activate

    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        if run is None:
            raise typer.BadParameter(f"No model run {run_id}")
        activate(session, run)
    typer.echo(f"Activated run {run_id}")


@app.command("backtest")
def backtest(
    symbol: list[str] = typer.Option(None, "--symbol", "-s"),
    horizon: list[int] = typer.Option(None, "--horizon"),
    models: str = typer.Option(None, help="Comma-separated. Default: TRAIN_MODELS."),
) -> None:
    """Backtest active models' out-of-sample signals vs buy-and-hold, after fees and slippage."""
    from forecaster.backtest.runner import backtest_model

    s = get_settings()
    names = [m.strip() for m in models.split(",")] if models else s.train_models
    for sym in [x.upper() for x in symbol] if symbol else s.target_symbols:
        for h in horizon or s.forecast_horizons:
            typer.echo(f"\n{sym}  signals from {h}d models  (costs: {s.fee_bps:g} bps fee + "
                       f"{s.slippage_bps:g} bps slippage per side)")
            typer.echo(f"  {'model':<12}{'total':>9}{'B&H':>9}{'sharpe':>8}{'B&H':>7}{'maxDD':>8}{'B&H':>8}"
                       f"{'trades':>7}{'win%':>6}{'expo':>6}{'p':>6}  verdict")
            any_beats = False
            for name in names:
                with session_scope() as session:
                    try:
                        r = backtest_model(session, s, sym, h, name)
                    except LookupError as exc:
                        typer.echo(f"  {name:<12}skipped: {exc}")
                        continue
                st, bh, bt = r.strategy, r.benchmark, r.bootstrap
                any_beats |= r.beats_benchmark and (bt["p_value"] or 1) < 0.05
                fmt = lambda v, f: format(v, f) if v is not None else "n/a"  # noqa: E731
                typer.echo(
                    f"  {name:<12}{st['total_return']:>9.1%}{bh['total_return']:>9.1%}"
                    f"{fmt(st['sharpe'], '.2f'):>8}{fmt(bh['sharpe'], '.2f'):>7}"
                    f"{st['max_drawdown']:>8.1%}{bh['max_drawdown']:>8.1%}{st['n_trades']:>7}"
                    f"{fmt(st['win_rate'] and st['win_rate'] * 100, '.0f'):>6}{st['exposure']:>6.0%}"
                    f"{fmt(bt['p_value'], '.2f'):>6}  {r.verdict}")
            if not any_beats:
                typer.secho("  => No model shows a statistically significant edge over buy-and-hold "
                            "for this horizon.", fg=typer.colors.YELLOW)


@app.command("serve")
def serve(
    host: str = typer.Option(None, help="Default: API_HOST (127.0.0.1)."),
    port: int = typer.Option(None, help="Default: API_PORT (8000)."),
) -> None:
    """Run the read-only REST API (docs at /docs)."""
    import uvicorn

    from forecaster.api.app import create_app

    s = get_settings()
    uvicorn.run(create_app(s), host=host or s.api_host, port=port or s.api_port, log_config=None)


@app.command("dashboard")
def dashboard(port: int = typer.Option(8501, help="Streamlit port.")) -> None:
    """Open the Streamlit dashboard (binds to localhost)."""
    import subprocess
    import sys

    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path),
                                      "--server.port", str(port), "--server.address", "127.0.0.1"]))


@app.command("snapshot")
def snapshot(out: Path = typer.Option(PROJECT_ROOT / "deploy" / "snapshot", help="Output directory.")) -> None:
    """Export a compact SQLite + model snapshot for the static cloud dashboard (deploy/)."""
    from forecaster.snapshot import build_snapshot

    s = get_settings()
    counts = build_snapshot(s.sqlalchemy_url, s.model_dir, out)
    typer.echo(f"Snapshot written to {out}: {counts}")


def _print_reports(reports) -> None:
    for r in reports:
        typer.echo(f"{r.symbol:<10} {r.interval:<3} dump={r.rows_dump:<6} api={r.rows_api:<6} "
                   f"gap_fill={r.rows_gap_fill:<4} invalid={r.invalid_dropped:<3} "
                   f"unfilled_gaps={len(r.unfilled_gaps)}")


if __name__ == "__main__":
    app()
