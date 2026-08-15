from __future__ import annotations

from typing import Any, Dict


STANDARD_ANALYSIS_MODE = "standard"
RIPPLE_ANALYSIS_MODE = "ripple_family"
LARGE_WAVE_ANALYSIS_MODE = "large_wave"

_RIPPLE_ALIASES = {"ripple", "ripple_family", "ripple_waves", "ripple-wave", "ripple-waves"}
_LARGE_WAVE_ALIASES = {
    "large",
    "large_wave",
    "large_waves",
    "large-wave",
    "large-waves",
    "curved_wave",
    "curved_waves",
}


def resolve_analysis_mode(config: Dict[str, Any] | None) -> str:
    analysis = ((config or {}).get("analysis") or {})
    raw = str(analysis.get("mode", STANDARD_ANALYSIS_MODE)).strip().lower()
    if raw in _RIPPLE_ALIASES:
        return RIPPLE_ANALYSIS_MODE
    if raw in _LARGE_WAVE_ALIASES:
        return LARGE_WAVE_ANALYSIS_MODE
    return STANDARD_ANALYSIS_MODE
