from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return a DB-compatible naive UTC timestamp."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_isoformat(value: datetime) -> str:
    """
    Serialize a datetime as explicit UTC.

    Existing database rows are naive UTC values, so naive inputs are treated as
    UTC instead of local server time.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def utc_now_iso() -> str:
    return utc_isoformat(utc_now())
