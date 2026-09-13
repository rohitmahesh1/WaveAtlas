from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from .analysis_mode import normalize_analysis_mode
from .sampling import normalize_sampling_rate_config


_TABLE_MODES = {
    "auto",
    "area",
    "continuous",
    "raw",
    "binary",
    "extreme",
    "extremes",
    "extreme_mask",
    "intensity",
    "legacy",
}
_EVENT_POLARITIES = {
    "max",
    "maximum",
    "maxima",
    "positive",
    "peak",
    "peaks",
    "min",
    "minimum",
    "minima",
    "negative",
    "trough",
    "troughs",
    "both",
    "all",
}
_FIT_TARGETS = {
    "residual",
    "res",
    "residuals",
    "raw",
    "raw_position",
    "raw_wave",
    "wave",
    "base",
    "base_wave",
    "both",
    "comparison",
    "compare",
}


def _section(parent: Mapping[str, Any], key: str, *, path: str = "") -> Dict[str, Any]:
    value = parent.get(key)
    if value is None:
        return {}
    field = f"{path}.{key}" if path else key
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _finite(value: Any, *, field: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum:g}")
    return number


def _positive(value: Any, *, field: str) -> float:
    number = _finite(value, field=field)
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return int(value)


def _validate_bounds(
    section: Mapping[str, Any],
    low_key: str,
    high_key: str,
    *,
    path: str,
    positive: bool = False,
) -> None:
    converter = _positive if positive else _finite
    low = (
        converter(section[low_key], field=f"{path}.{low_key}")
        if section.get(low_key) is not None
        else None
    )
    high = (
        converter(section[high_key], field=f"{path}.{high_key}")
        if section.get(high_key) is not None
        else None
    )
    if low is not None and high is not None and low > high:
        raise ValueError(f"{path}.{low_key} must not exceed {path}.{high_key}")


def _validate_peaks(section: Mapping[str, Any], *, path: str) -> None:
    polarity_key = "event_polarity" if "event_polarity" in section else "polarity"
    if polarity_key in section:
        value = str(section[polarity_key]).strip().lower()
        if value not in _EVENT_POLARITIES:
            raise ValueError(f"{path}.{polarity_key} has unknown value {value!r}")
    if "minimum_per_track" in section:
        _integer(section["minimum_per_track"], field=f"{path}.minimum_per_track")
    if "prominence" in section:
        _finite(section["prominence"], field=f"{path}.prominence", minimum=0.0)
    for key in ("width", "distance"):
        if key in section:
            _positive(section[key], field=f"{path}.{key}")


def normalize_and_validate_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize and fail fast on settings that can silently change measurements."""

    normalized = normalize_sampling_rate_config(config)

    analysis = _section(normalized, "analysis")
    if "mode" in analysis:
        analysis["mode"] = normalize_analysis_mode(analysis["mode"])
        normalized["analysis"] = analysis

    heatmap = _section(normalized, "heatmap")
    if heatmap:
        if "table_mode" in heatmap:
            mode = str(heatmap["table_mode"]).strip().lower()
            if mode not in _TABLE_MODES:
                raise ValueError(f"heatmap.table_mode has unknown value {mode!r}")
        if "origin" in heatmap:
            origin = str(heatmap["origin"]).strip().lower()
            if origin not in {"lower", "upper"}:
                raise ValueError("heatmap.origin must be 'lower' or 'upper'")
        if "non_finite_policy" in heatmap:
            policy = str(heatmap["non_finite_policy"]).strip().lower()
            if policy not in {"reject", "zero"}:
                raise ValueError("heatmap.non_finite_policy must be 'reject' or 'zero'")
        _validate_bounds(heatmap, "lower", "upper", path="heatmap")
        _validate_bounds(heatmap, "vmin", "vmax", path="heatmap")
        for mode_name in ("continuous", "area"):
            mode_config = _section(heatmap, mode_name, path="heatmap")
            if "origin" in mode_config:
                origin = str(mode_config["origin"]).strip().lower()
                if origin not in {"lower", "upper"}:
                    raise ValueError(
                        f"heatmap.{mode_name}.origin must be 'lower' or 'upper'"
                    )
            _validate_bounds(mode_config, "vmin", "vmax", path=f"heatmap.{mode_name}")

    period = _section(normalized, "period")
    _validate_bounds(period, "min_freq", "max_freq", path="period", positive=True)

    kymo = _section(normalized, "kymo")
    if "backend" in kymo:
        backend = str(kymo["backend"]).strip().lower()
        if backend not in {"onnx", "wolfram"}:
            raise ValueError("kymo.backend must be 'onnx' or 'wolfram'")
    if "track_xy_order" in kymo:
        order = str(kymo["track_xy_order"]).strip().lower()
        if order not in {"auto", "yx", "xy"}:
            raise ValueError("kymo.track_xy_order must be 'auto', 'yx', or 'xy'")

    detrend = _section(normalized, "detrend")
    if "degree" in detrend:
        _integer(detrend["degree"], field="detrend.degree")
    if "min_samples" in detrend:
        min_samples = detrend["min_samples"]
        if isinstance(min_samples, int) and not isinstance(min_samples, bool):
            _integer(min_samples, field="detrend.min_samples", minimum=1)
        else:
            fraction = _positive(min_samples, field="detrend.min_samples")
            if fraction > 1:
                raise ValueError("fractional detrend.min_samples must not exceed 1")
    if detrend.get("residual_threshold") is not None:
        _positive(detrend["residual_threshold"], field="detrend.residual_threshold")

    peaks = _section(normalized, "peaks")
    _validate_peaks(peaks, path="peaks")
    large_wave = _section(analysis, "large_wave", path="analysis")
    large_wave_peaks = _section(large_wave, "peaks", path="analysis.large_wave")
    _validate_peaks(large_wave_peaks, path="analysis.large_wave.peaks")

    features = _section(normalized, "features")
    if "fit_target" in features:
        fit_target = str(features["fit_target"]).strip().lower().replace("-", "_")
        if fit_target not in _FIT_TARGETS:
            raise ValueError(f"features.fit_target has unknown value {fit_target!r}")
    if "fit_window_period_frac" in features:
        _positive(features["fit_window_period_frac"], field="features.fit_window_period_frac")

    return normalized
