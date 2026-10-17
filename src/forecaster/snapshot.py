"""Build a compact, self-contained snapshot (SQLite + active model artifacts) for a static
dashboard deployment, e.g. Streamlit Community Cloud, which cannot reach MySQL or Binance.

Copied: symbols; daily klines for every symbol and 4h klines for targets (what features need);
market context; ACTIVE model runs, their predictions and latest backtests. Not copied: the
features table (rebuilt on the fly), inactive model versions, 4h klines of context symbols.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, func, insert, select, text

from forecaster.db.models import BacktestResult, Kline, MarketContext, ModelRun, Prediction, Symbol
from forecaster.logging_setup import get_logger
from forecaster.modeling.registry import resolve_artifact_path

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DB_NAME = "forecast.db"
CHUNK = 5000


def _migrate(url: str) -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


def _copy(src: Engine, dst: Engine, query, table) -> int:
    n = 0
    with src.connect() as s, dst.begin() as d:
        result = s.execution_options(stream_results=True).execute(query)
        while rows := result.fetchmany(CHUNK):
            d.execute(insert(table), [dict(r._mapping) for r in rows])
            n += len(rows)
    return n


def build_snapshot(source_url: str, model_dir: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "models").mkdir(parents=True)
    dst_url = f"sqlite:///{(out_dir / DB_NAME).as_posix()}"
    _migrate(dst_url)
    src, dst = create_engine(source_url), create_engine(dst_url)

    with src.connect() as c:
        targets = [r.id for r in c.execute(select(Symbol.id).where(Symbol.is_target.is_(True)))]
        runs = c.execute(select(ModelRun).where(ModelRun.is_active.is_(True))).all()
    run_ids = [r.id for r in runs]
    latest_bt = (
        select(func.max(BacktestResult.id))
        .where(BacktestResult.model_run_id.in_(run_ids))
        .group_by(BacktestResult.strategy, BacktestResult.horizon_days)
    )
    counts = {
        "symbols": _copy(src, dst, select(Symbol), Symbol.__table__),
        "klines": _copy(src, dst, select(Kline).where(
            (Kline.interval == "1d") | ((Kline.interval == "4h") & Kline.symbol_id.in_(targets))), Kline.__table__),
        "market_context": _copy(src, dst, select(MarketContext), MarketContext.__table__),
        "model_runs": _copy(src, dst, select(ModelRun).where(ModelRun.id.in_(run_ids)), ModelRun.__table__),
        "predictions": _copy(src, dst, select(Prediction).where(Prediction.model_run_id.in_(run_ids)),
                             Prediction.__table__),
        "backtest_results": _copy(src, dst, select(BacktestResult).where(BacktestResult.id.in_(latest_bt)),
                                  BacktestResult.__table__),
    }

    # Copy each active artifact and store its path relative to the snapshot's model dir.
    with dst.begin() as d:
        for run in runs:
            if not run.artifact_path:
                continue
            src_file = resolve_artifact_path(run.artifact_path, model_dir)
            rel = Path(run.model_name) / f"h{run.horizon_days}" / run.version
            target_dir = out_dir / "models" / str(run.symbol_id) / rel
            shutil.copytree(src_file.parent, target_dir)
            new_path = (Path(str(run.symbol_id)) / rel / src_file.name).as_posix()
            d.execute(text("UPDATE model_runs SET artifact_path = :p WHERE id = :i"), {"p": new_path, "i": run.id})
    with dst.connect() as c:
        c.execute(text("VACUUM"))
    src.dispose()
    dst.dispose()
    counts["size_mb"] = round(sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6, 1)
    log.info("snapshot built", extra={"out": str(out_dir), **counts})
    return counts
