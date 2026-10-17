"""Small data-access helpers shared across phases."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from forecaster.db.models import Symbol

KNOWN_QUOTES = ("USDT", "USDC", "FDUSD", "BUSD", "BTC", "ETH")


def split_symbol(symbol: str) -> tuple[str, str]:
    """'SOLUSDT' -> ('SOL', 'USDT')."""
    for quote in KNOWN_QUOTES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[: -len(quote)], quote
    raise ValueError(f"Cannot determine quote asset for symbol {symbol!r}")


def sync_symbols(session: Session, symbols: list[str], targets: list[str]) -> list[Symbol]:
    """Ensure every configured symbol exists and is active; update target flags. Idempotent."""
    existing = {s.symbol: s for s in session.scalars(select(Symbol))}
    out = []
    for sym in symbols:
        row = existing.get(sym)
        if row is None:
            base, quote = split_symbol(sym)
            row = Symbol(symbol=sym, base_asset=base, quote_asset=quote)
            session.add(row)
        row.is_active = True
        row.is_target = sym in targets
        out.append(row)
    session.flush()
    return out


def get_symbol(session: Session, symbol: str) -> Symbol | None:
    return session.scalar(select(Symbol).where(Symbol.symbol == symbol.upper()))
