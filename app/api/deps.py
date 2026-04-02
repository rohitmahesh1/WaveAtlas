from __future__ import annotations

import os
from functools import lru_cache
from uuid import UUID, uuid4

from fastapi import Depends, Request, Response
from sqlmodel import Session

from ..artifact_store import GCSArtifactStore, LocalArtifactStore, ArtifactStore
from ..job_store import JobStore

# Assume you have an engine available.
# Implement this in app/db.py (or equivalent).
from ..db import engine  # type: ignore


SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "sid")


def get_db_session() -> Session:
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()


def get_job_store(session: Session = Depends(get_db_session)) -> JobStore:
    return JobStore(session=session)


def get_owner_session_id(request: Request, response: Response) -> UUID:
    """
    Ensures a stable cookie-backed session id exists.
    """
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    sid: UUID
    try:
        sid = UUID(raw) if raw else uuid4()
    except Exception:
        sid = uuid4()

    if not raw or raw != str(sid):
        _set_session_cookie(response, sid)

    return sid


def get_owner_session_id_ws(request: Request) -> UUID:
    """
    WebSocket connections don't have a Response to set cookies; require an existing cookie.
    """
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw:
        raise ValueError("Missing session cookie")
    return UUID(raw)


def _set_session_cookie(response: Response, sid: UUID) -> None:
    secure = os.getenv("COOKIE_SECURE", "0").strip() in ("1", "true", "True")
    samesite = os.getenv("COOKIE_SAMESITE", "lax").strip().lower()
    if samesite not in ("lax", "strict", "none"):
        samesite = "lax"

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=str(sid),
        httponly=True,
        secure=secure,
        samesite=samesite,
        path="/",
        max_age=60 * 60 * 24 * 365,
    )


@lru_cache(maxsize=1)
def _artifact_store_singleton() -> ArtifactStore:
    store = os.getenv("ARTIFACT_STORE", "local").strip().lower()
    if store == "gcs":
        bucket = os.environ["GCS_BUCKET"]
        prefix = os.getenv("GCS_PREFIX", "")
        public = os.getenv("GCS_PUBLIC", "0").strip() in ("1", "true", "True")
        return GCSArtifactStore(bucket=bucket, prefix=prefix, public=public)
    root_dir = os.getenv("ARTIFACT_ROOT_DIR", "./data")
    return LocalArtifactStore(root_dir=root_dir)


def get_artifact_store() -> ArtifactStore:
    return _artifact_store_singleton()
