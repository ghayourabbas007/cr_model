"""Load raw inputs from the database, build features/targets, persist features."""

from __future__ import annotations

from sqlalchemy.orm import Session

from forecaster.config import Settings
from forecaster.data.context import FUNDING_RATE, OPEN_INTEREST
from forecaster.data.store import load_context, load_klines
from forecaster.db.repository import get_symbol, split_symbol
from forecaster.db.session import session_scope
from forecaster.features.build import (
    FEATURE_SET_VERSION,
    FeatureInputs,
    build_features,
    check_no_lookahead,
    default_cutoffs,
)
from forecaster.features.store import save_features
from forecaster.logging_setup import get_logger

log = get_logger(__name__)


def load_inputs(session: Session, s: Settings, symbol: str) -> FeatureInputs:
    daily = load_klines(session, symbol, "1d")
    if daily.empty:
        raise ValueError(f"No 1d klines stored for {symbol}; run `forecaster data backfill` first")
    cross = {}
    for other in s.cross_asset_symbols:
        if other == symbol:
            continue
        prefix = split_symbol(other)[0].lower()
        cross[prefix] = load_klines(session, other, "1d")
    intraday = load_klines(session, symbol, "4h") if "4h" in s.intervals else None
    funding = load_context(session, symbol, FUNDING_RATE) if s.include_funding_features else None
    oi = load_context(session, symbol, OPEN_INTEREST) if s.include_oi_features else None
    return FeatureInputs(daily=daily, cross=cross, intraday=intraday, funding=funding, open_interest=oi)


def run_build_features(s: Settings, symbols: list[str] | None = None) -> dict[str, int]:
    out = {}
    with session_scope() as session:
        for name in symbols or s.target_symbols:
            sym = get_symbol(session, name)
            if sym is None:
                raise ValueError(f"{name} is not in the symbols table; run `forecaster db init`")
            frame = build_features(load_inputs(session, s, name))
            out[name] = save_features(session, sym.id, "1d", frame, FEATURE_SET_VERSION)
            session.commit()
            log.info("features stored", extra={"symbol": name, "rows": out[name],
                                               "version": FEATURE_SET_VERSION,
                                               "n_features": frame.shape[1] - 2})
    return out


def run_leakage_check(s: Settings, symbol: str, n_cutoffs: int = 5) -> list:
    """Truncation test on real stored data. Raises LeakageError on failure."""
    with session_scope() as session:
        inp = load_inputs(session, s, symbol)
    cutoffs = default_cutoffs(inp, n_cutoffs)
    check_no_lookahead(inp, cutoffs)
    return cutoffs
