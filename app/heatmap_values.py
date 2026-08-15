from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def load_heatmap_values(
    *,
    heatmap_path: Path,
    value_bytes: Optional[bytes],
    value_meta: Dict[str, Any],
) -> Tuple[np.ndarray, str]:
    image = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unable to read heatmap image: {heatmap_path}")
    if value_bytes is None:
        return image.astype(np.float32) / 255.0, "rendered_heatmap_luminance"

    rows = int(value_meta.get("source_rows") or value_meta.get("rows") or image.shape[0])
    cols = int(value_meta.get("source_cols") or value_meta.get("cols") or image.shape[1])
    values = np.frombuffer(value_bytes, dtype="<f4")
    if values.size != rows * cols:
        raise RuntimeError(
            f"Continuous heatmap value count mismatch: got {values.size}, expected {rows * cols}"
        )
    values = values.reshape((rows, cols)).copy()
    origin = str(value_meta.get("coord_origin", value_meta.get("origin", "upper"))).lower()
    row_order = str(value_meta.get("value_row_order", "top_to_bottom_source")).lower()
    if origin == "lower" and row_order == "top_to_bottom_source":
        values = np.flipud(values)
    if values.shape != image.shape:
        values = cv2.resize(values, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return values.astype(np.float32, copy=False), "continuous_table_values"


def normalize_heatmap_values(
    values: np.ndarray,
    *,
    percentiles: Sequence[float] = (1.0, 99.0),
    context: str = "Heatmap extraction",
) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError(f"{context} received no finite heatmap values")
    if len(percentiles) != 2:
        percentiles = (1.0, 99.0)
    low, high = np.percentile(finite, [float(percentiles[0]), float(percentiles[1])])
    span = max(float(high - low), 1e-6)
    normalized = np.nan_to_num((values - low) / span, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)
