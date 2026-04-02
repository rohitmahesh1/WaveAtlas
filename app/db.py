# app/db.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from sqlmodel import SQLModel, create_engine
from sqlalchemy.engine import Engine


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def get_database_url() -> str:
    """
    Set DATABASE_URL to:
      - sqlite:///./dev.db   (local)
      - postgresql+psycopg2://user:pass@host:5432/dbname   (prod)
    """
    url = os.getenv("DATABASE_URL")
    if url and url.strip():
        return url.strip()
    return "sqlite:///./dev.db"


def build_engine(database_url: Optional[str] = None) -> Engine:
    url = database_url or get_database_url()
    echo = _env_bool("DB_ECHO", False)

    # Cloud Run / server defaults
    pool_size = _env_int("DB_POOL_SIZE", 5)
    max_overflow = _env_int("DB_MAX_OVERFLOW", 10)
    pool_timeout = _env_int("DB_POOL_TIMEOUT", 30)
    pool_recycle = _env_int("DB_POOL_RECYCLE", 1800)

    connect_args: Dict[str, Any] = {}

    if url.startswith("sqlite"):
        # SQLite needs this for multi-threaded FastAPI usage.
        connect_args["check_same_thread"] = False
        return create_engine(url, echo=echo, connect_args=connect_args)

    # Postgres/MySQL/etc: use pool settings
    return create_engine(
        url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


# Engine used across the app (imported by api/deps.py)
engine: Engine = build_engine()


def init_db(*, create_all: bool = False) -> None:
    """
    For MVP you can use create_all=True to bootstrap tables.
    In production you should use migrations instead.
    """
    # Ensure models are imported so SQLModel.metadata is populated
    from . import models  # noqa: F401

    if create_all or _env_bool("DB_CREATE_ALL", False):
        SQLModel.metadata.create_all(engine)
