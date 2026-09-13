from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, Optional


def _positive_rate(value: Any, *, field: str) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive number") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return rate


def _mapping_section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def normalize_sampling_rate_config(
    config: Optional[Dict[str, Any]],
    *,
    default: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a copy with one canonical sampling rate at io.sampling_rate."""

    normalized = deepcopy(config or {})
    io_cfg = _mapping_section(normalized, "io")
    period_cfg = _mapping_section(normalized, "period")

    canonical_raw = io_cfg.get("sampling_rate")
    legacy_raw = period_cfg.get("sampling_rate")
    canonical = (
        _positive_rate(canonical_raw, field="io.sampling_rate")
        if canonical_raw is not None
        else None
    )
    legacy = (
        _positive_rate(legacy_raw, field="period.sampling_rate")
        if legacy_raw is not None
        else None
    )
    if canonical is not None and legacy is not None and not math.isclose(
        canonical,
        legacy,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Conflicting sampling rates: use io.sampling_rate; "
            f"got io.sampling_rate={canonical:g} and period.sampling_rate={legacy:g}"
        )

    rate = canonical if canonical is not None else legacy
    if rate is None and default is not None:
        rate = _positive_rate(default, field="sampling rate")
    if rate is not None:
        io_cfg["sampling_rate"] = rate
        normalized["io"] = io_cfg

    if "sampling_rate" in period_cfg:
        period_cfg.pop("sampling_rate")
        normalized["period"] = period_cfg
    return normalized


def resolve_sampling_rate(config: Dict[str, Any], *, default: float = 1.0) -> float:
    normalized = normalize_sampling_rate_config(config, default=default)
    return float((normalized.get("io") or {})["sampling_rate"])
