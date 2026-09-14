from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


TRACK_SCHEMA_VERSION = 1
TRACK_COORDINATE_ORDER = "image_row_position"
TRACK_INDEX_BASE = 0
SCIENTIFIC_COORDINATE_ORIGIN = "lower"

_COORDINATE_METADATA_FIELDS = (
    "coordinate_transform_version",
    "native_resolution",
    "source_rows",
    "source_cols",
    "analysis_rows",
    "analysis_cols",
    "source_coordinate_space",
    "analysis_coordinate_space",
    "analysis_to_source_x_scale",
    "analysis_to_source_frame_scale",
    "spatial_calibration_um_per_px",
    "spatial_distance_unit",
    "spatial_pixel_space",
)


def _coordinate_metadata(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    source = meta or {}
    return {key: source.get(key) for key in _COORDINATE_METADATA_FIELDS if key in source}


@dataclass(frozen=True)
class TrackCoordinates:
    """One track represented in both image and scientific coordinates."""

    image_row: np.ndarray
    frame: np.ndarray
    position: np.ndarray


def coordinate_height(meta: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not meta:
        return None
    for key in ("output_height", "source_rows", "nrows", "image_height"):
        value = meta.get(key)
        if value in (None, ""):
            continue
        try:
            height = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(height) and height > 0:
            return height
    return None


def coordinate_origin(meta: Optional[Mapping[str, Any]], *, default: Optional[str] = None) -> str:
    """Return the invariant origin of WaveAtlas scientific coordinates.

    ``origin`` and ``render_origin`` describe how source data was rendered. They
    do not change the scientific frame axis, whose zero is always at the bottom.
    The validation below still catches malformed legacy metadata.
    """

    source = meta or {}
    value = source.get("coord_origin")
    if value in (None, ""):
        value = default or SCIENTIFIC_COORDINATE_ORIGIN
    declared_origin = str(value).strip().lower()
    if declared_origin not in {"lower", "upper"}:
        raise ValueError(f"Unknown coordinate origin {value!r}; expected 'lower' or 'upper'")
    return SCIENTIFIC_COORDINATE_ORIGIN


def frame_from_image_row(
    image_row: Any,
    meta: Optional[Mapping[str, Any]],
) -> np.ndarray:
    rows = np.asarray(image_row, dtype=float)
    height = coordinate_height(meta)
    if height is None:
        # Legacy helper calls may already supply scientific frames and have no
        # image metadata. Production track artifacts always include the height.
        return rows.astype(float, copy=True)
    return (height - 1.0) - rows


def image_row_from_frame(
    frame: Any,
    meta: Optional[Mapping[str, Any]],
) -> np.ndarray:
    values = np.asarray(frame, dtype=float)
    height = coordinate_height(meta)
    if height is None:
        return values.astype(float, copy=True)
    return (height - 1.0) - values


def point_order_from_config(config: Mapping[str, Any]) -> str:
    kymo = config.get("kymo") or {}
    value = str(kymo.get("track_xy_order", "auto")).strip().lower()
    if value == "auto":
        # All current extractors persist canonical (image_row, position) points.
        return "yx"
    if value not in {"yx", "xy"}:
        raise ValueError(f"Unknown track coordinate order {value!r}; expected 'yx', 'xy', or 'auto'")
    return value


def point_order_from_artifact(
    artifact_meta: Optional[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> str:
    meta = artifact_meta or {}
    schema_version = meta.get("track_schema_version")
    if schema_version is not None:
        try:
            version = int(schema_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid track schema version {schema_version!r}") from exc
        if version != TRACK_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported track schema version {version}; expected {TRACK_SCHEMA_VERSION}"
            )

    value = str(meta.get("coordinate_order", "")).strip().lower()
    if schema_version is not None:
        if value != TRACK_COORDINATE_ORDER:
            raise ValueError(f"Unknown track artifact coordinate order {value!r}")
        return "yx"

    if value in {TRACK_COORDINATE_ORDER, "row_column", "yx"}:
        return "yx"
    if value in {"position_image_row", "column_row", "xy"}:
        return "xy"
    return point_order_from_config(config)


def index_base_from_artifact(artifact_meta: Optional[Mapping[str, Any]]) -> int:
    meta = artifact_meta or {}
    value = meta.get("coordinate_index_base", TRACK_INDEX_BASE)
    try:
        index_base = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid track coordinate index base {value!r}") from exc
    if meta.get("track_schema_version") is not None and index_base != TRACK_INDEX_BASE:
        raise ValueError(
            f"Track schema {TRACK_SCHEMA_VERSION} requires coordinate index base {TRACK_INDEX_BASE}"
        )
    return index_base


def track_coordinates_from_points(
    data: np.ndarray,
    *,
    order: str = "yx",
    heatmap_meta: Optional[Mapping[str, Any]] = None,
    index_base: int = TRACK_INDEX_BASE,
) -> TrackCoordinates:
    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        if order == "yx":
            image_row = arr[:, 0].astype(float, copy=False)
            position = arr[:, 1].astype(float, copy=False)
        elif order == "xy":
            position = arr[:, 0].astype(float, copy=False)
            image_row = arr[:, 1].astype(float, copy=False)
        else:
            raise ValueError(f"Unknown track coordinate order {order!r}")
    elif arr.ndim == 1:
        image_row = np.arange(arr.shape[0], dtype=float)
        position = arr.astype(float, copy=False)
    else:
        raise ValueError(f"Unsupported track array shape: {arr.shape}")

    if int(index_base) != 0:
        image_row = image_row - float(index_base)
        position = position - float(index_base)

    frame = frame_from_image_row(image_row, heatmap_meta)
    order_idx = np.argsort(frame, kind="stable")
    return TrackCoordinates(
        image_row=np.asarray(image_row[order_idx], dtype=float),
        frame=np.asarray(frame[order_idx], dtype=float),
        position=np.asarray(position[order_idx], dtype=float),
    )


def load_track_coordinates(
    track_path: Path,
    *,
    order: str = "yx",
    heatmap_meta: Optional[Mapping[str, Any]] = None,
    index_base: int = TRACK_INDEX_BASE,
) -> TrackCoordinates:
    return track_coordinates_from_points(
        np.load(track_path, allow_pickle=False),
        order=order,
        heatmap_meta=heatmap_meta,
        index_base=index_base,
    )


def track_artifact_metadata(
    track_index: int,
    heatmap_meta: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    meta = heatmap_meta or {}
    return {
        "track_index": int(track_index),
        "track_schema_version": TRACK_SCHEMA_VERSION,
        "coordinate_order": TRACK_COORDINATE_ORDER,
        "coordinate_index_base": TRACK_INDEX_BASE,
        "coord_origin": coordinate_origin(meta),
        "image_height": coordinate_height(meta),
        **_coordinate_metadata(meta),
    }


def track_manifest_metadata(
    *,
    total_tracks: int,
    analysis_mode: str,
    extractor: str,
    heatmap_meta: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "total_tracks": int(total_tracks),
        "analysis_mode": analysis_mode,
        "extractor": extractor,
        "track_schema_version": TRACK_SCHEMA_VERSION,
        "coordinate_order": TRACK_COORDINATE_ORDER,
        "coordinate_index_base": TRACK_INDEX_BASE,
        "coord_origin": coordinate_origin(heatmap_meta),
        "image_height": coordinate_height(heatmap_meta),
        **_coordinate_metadata(heatmap_meta),
    }
