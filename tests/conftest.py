from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep tests independent of a developer's .env and clear cached settings."""
    from forecaster.config import get_settings

    monkeypatch.chdir(ROOT / "tests")  # no .env here
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def migrated_engine(tmp_path: Path) -> Engine:
    """SQLite database built by running the real Alembic migrations."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    return create_engine(url)


@pytest.fixture
def session(migrated_engine: Engine) -> Iterator[Session]:
    with Session(migrated_engine) as s:
        yield s
