from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


VALUE_ENCODING = "float32_le"


def encode_heatmap_values(values: np.ndarray) -> bytes:
    """Encode the authoritative scalar analysis raster."""

    return np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")


def decode_heatmap_values(value_bytes: bytes, value_meta: Dict[str, Any]) -> np.ndarray:
    """Decode values and orient their rows like the analysis/display raster."""

    encoding = str(value_meta.get("value_encoding") or VALUE_ENCODING).lower()
    if encoding != VALUE_ENCODING:
        raise RuntimeError(f"Unsupported heatmap value encoding: {encoding}")
    rows = int(
        value_meta.get("analysis_rows")
        or value_meta.get("source_rows")
        or value_meta.get("rows")
        or 0
    )
    cols = int(
        value_meta.get("analysis_cols")
        or value_meta.get("source_cols")
        or value_meta.get("cols")
        or 0
    )
    if rows <= 0 or cols <= 0:
        raise RuntimeError("Heatmap value metadata is missing positive raster dimensions")
    values = np.frombuffer(value_bytes, dtype="<f4")
    if values.size != rows * cols:
        raise RuntimeError(
            f"Continuous heatmap value count mismatch: got {values.size}, expected {rows * cols}"
        )
    values = values.reshape((rows, cols)).copy()

    # Table rows are stored in source order and may be flipped for a lower-origin
    # rendering. Image values are already stored in top-to-bottom image order.
    render_origin = str(
        value_meta.get(
            "render_origin",
            value_meta.get("origin", value_meta.get("coord_origin", "upper")),
        )
    ).lower()
    row_order = str(value_meta.get("value_row_order", "top_to_bottom_source")).lower()
    if render_origin == "lower" and row_order == "top_to_bottom_source":
        values = np.flipud(values)
    return values.astype(np.float32, copy=False)


def analysis_image_bytes(value_bytes: bytes, value_meta: Dict[str, Any]) -> bytes:
    """Create the deterministic grayscale extractor input from analysis values."""

    values = decode_heatmap_values(value_bytes, value_meta)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("Analysis raster contains no finite values")

    # Extractor normalization is part of the method contract. Display vmin,
    # vmax, and colormap settings intentionally have no effect here.
    low_value = float(np.min(finite))
    high_value = float(np.max(finite))

    if high_value > low_value:
        normalized = (values - low_value) / (high_value - low_value)
    else:
        normalized = np.zeros_like(values, dtype=np.float32)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    pixels = np.asarray(np.clip(normalized, 0.0, 1.0) * 255.0, dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def read_cv_image(path: Path, flags: int) -> Optional[np.ndarray]:
    """Read an image without OpenCV's Windows Unicode-path limitation."""
    try:
        encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, flags)


def write_cv_image(path: Path, image: np.ndarray) -> None:
    """Write an image without OpenCV's Windows Unicode-path limitation."""
    path = Path(path)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise RuntimeError(f"Unable to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def load_heatmap_values(
    *,
    heatmap_path: Path,
    value_bytes: Optional[bytes],
    value_meta: Dict[str, Any],
) -> Tuple[np.ndarray, str]:
    image = read_cv_image(heatmap_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unable to read heatmap image: {heatmap_path}")
    if value_bytes is None:
        raise RuntimeError("Authoritative scalar analysis values are required")
    values = decode_heatmap_values(value_bytes, value_meta)
    if values.shape != image.shape:
        raise RuntimeError(
            "Analysis raster dimensions do not match the extractor image: "
            f"values={values.shape}, image={image.shape}. Scientific runs do not resize "
            "analysis values implicitly."
        )
    source = str(value_meta.get("analysis_value_source") or "authoritative_analysis_values")
    return values.astype(np.float32, copy=False), source


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
