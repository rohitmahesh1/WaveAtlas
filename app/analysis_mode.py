from __future__ import annotations

from typing import Any, Dict


STANDARD_ANALYSIS_MODE = "standard"
RIPPLE_ANALYSIS_MODE = "ripple_family"

_RIPPLE_ALIASES = {"ripple", "ripple_family", "ripple_waves", "ripple-wave", "ripple-waves"}


def resolve_analysis_mode(config: Dict[str, Any] | None) -> str:
    analysis = ((config or {}).get("analysis") or {})
    raw = str(analysis.get("mode", STANDARD_ANALYSIS_MODE)).strip().lower()
    return RIPPLE_ANALYSIS_MODE if raw in _RIPPLE_ALIASES else STANDARD_ANALYSIS_MODE
