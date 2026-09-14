from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class SpatialCalibration:
    """Horizontal source-pixel scale used by kymograph measurements."""

    micrometers_per_pixel: float

    def metadata(self) -> Dict[str, Any]:
        return {
            "spatial_calibration_um_per_px": self.micrometers_per_pixel,
            "spatial_distance_unit": "um",
            "spatial_pixel_space": "source_pixel",
        }


def resolve_spatial_calibration(config: Mapping[str, Any]) -> Optional[SpatialCalibration]:
    io_config = config.get("io") if isinstance(config, Mapping) else None
    if not isinstance(io_config, Mapping):
        return None
    raw_calibration = io_config.get("spatial_calibration")
    if raw_calibration is None:
        return None
    if not isinstance(raw_calibration, Mapping):
        raise ValueError("io.spatial_calibration must be a mapping")
    raw_value = raw_calibration.get("micrometers_per_pixel")
    if raw_value in (None, ""):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "io.spatial_calibration.micrometers_per_pixel must be a finite positive number"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "io.spatial_calibration.micrometers_per_pixel must be a finite positive number"
        )
    return SpatialCalibration(value)


_SUFFIXES = (
    ("_px_per_frame2", "_um_per_frame2"),
    ("_px_per_frame", "_um_per_frame"),
    ("_px_per_s", "_um_per_s"),
    ("_px_s", "_um_s"),
    ("_px", "_um"),
)

_LABEL_REPLACEMENTS = (
    ("pixels/sec", "µm/sec"),
    ("px/frame^2", "µm/frame^2"),
    ("px/frame", "µm/frame"),
    ("pixel-seconds", "µm-seconds"),
    ("Pixels", "µm"),
    ("pixels", "µm"),
    ("px", "µm"),
)


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def apply_spatial_calibration(
    record: MutableMapping[str, Any],
    calibration: Optional[SpatialCalibration],
) -> MutableMapping[str, Any]:
    """Add micrometer equivalents for explicitly pixel-valued fields."""

    if calibration is None:
        return record
    additions: Dict[str, Any] = {
        **calibration.metadata(),
        "Spatial Calibration (µm/pixel)": calibration.micrometers_per_pixel,
    }
    for key, raw_value in list(record.items()):
        value = _finite_number(raw_value)
        if value is None:
            continue
        for pixel_suffix, calibrated_suffix in _SUFFIXES:
            if key.endswith(pixel_suffix):
                additions[f"{key[:-len(pixel_suffix)]}{calibrated_suffix}"] = (
                    value * calibration.micrometers_per_pixel
                )
                break
        for pixel_text, calibrated_text in _LABEL_REPLACEMENTS:
            if pixel_text in key:
                additions[key.replace(pixel_text, calibrated_text)] = (
                    value * calibration.micrometers_per_pixel
                )
                break
    record.update(additions)
    return record


def spatial_calibration_from_record(record: Mapping[str, Any]) -> Optional[SpatialCalibration]:
    value = _finite_number(record.get("spatial_calibration_um_per_px"))
    if value is None or value <= 0:
        return None
    return SpatialCalibration(value)


def apply_spatial_calibration_many(
    records: Iterable[MutableMapping[str, Any]],
    calibration: Optional[SpatialCalibration],
) -> None:
    if calibration is None:
        return
    for record in records:
        apply_spatial_calibration(record, calibration)


def coordinate_contract_metadata(
    heatmap_meta: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    source_rows = int(heatmap_meta.get("source_rows") or heatmap_meta.get("output_height") or 0)
    source_cols = int(heatmap_meta.get("source_cols") or heatmap_meta.get("output_width") or 0)
    analysis_rows = int(heatmap_meta.get("analysis_rows") or heatmap_meta.get("output_height") or 0)
    analysis_cols = int(heatmap_meta.get("analysis_cols") or heatmap_meta.get("output_width") or 0)
    if min(source_rows, source_cols, analysis_rows, analysis_cols) <= 0:
        raise ValueError("Coordinate metadata requires positive source and analysis dimensions")
    if (source_rows, source_cols) != (analysis_rows, analysis_cols):
        raise ValueError("Scientific runs require native-resolution source and analysis rasters")
    metadata: Dict[str, Any] = {
        "coordinate_transform_version": 1,
        "native_resolution": True,
        "source_coordinate_space": "bottom_left_source_frame_position",
        "analysis_coordinate_space": "bottom_left_source_frame_position",
        "analysis_to_source_x_scale": 1.0,
        "analysis_to_source_frame_scale": 1.0,
    }
    calibration = resolve_spatial_calibration(config)
    if calibration is not None:
        metadata.update(calibration.metadata())
    return metadata
