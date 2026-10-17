from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from forecaster.config import get_settings


def get_engine(url: str | None = None) -> Engine:
    # Resolve the URL before caching so a settings change never returns a stale engine.
    return _engine(url or get_settings().sqlalchemy_url)


@lru_cache
def _engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def get_sessionmaker(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = get_sessionmaker(url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
