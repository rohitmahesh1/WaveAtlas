from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from typing import Any, Dict, Literal, Sequence, Tuple

from .analysis_mode import LARGE_WAVE_ANALYSIS_MODE


ColumnLabelProfile = Literal["familiar", "descriptive"]

MEASUREMENT_SCHEMA_VERSION = 1
DEFAULT_COLUMN_LABEL_PROFILE: ColumnLabelProfile = "familiar"
COLUMN_LABEL_PROFILES: Tuple[ColumnLabelProfile, ...] = ("familiar", "descriptive")


@dataclass(frozen=True)
class MeasurementDefinition:
    key: str
    familiar_label: str
    quantity: str
    operational_definition: str
    algorithm: str
    unit: str
    coordinate_space: str
    aggregation_level: str
    modes: Tuple[str, ...]
    validity: str
    quality_fields: Tuple[str, ...] = ()


_DEFINITIONS = (
    MeasurementDefinition(
        key="standard_track_spectral_frequency_hz",
        familiar_label="Frequency",
        quantity="Dominant track oscillation frequency",
        operational_definition="Frequency of the strongest accepted spectral component in the detrended track position trace.",
        algorithm="Detrend the track, estimate its dominant spectral frequency, and constrain the result to the configured frequency range.",
        unit="Hz",
        coordinate_space="time in seconds; position in processed-image pixels",
        aggregation_level="track",
        modes=("standard",),
        validity="Valid when the spectral estimator returns a finite positive frequency within the configured range.",
        quality_fields=("spectral_snr", "frequency_agreement_error"),
    ),
    MeasurementDefinition(
        key="standard_track_spectral_period_s",
        familiar_label="Period",
        quantity="Dominant track oscillation period",
        operational_definition="Reciprocal of the dominant spectral frequency for the detrended track position trace.",
        algorithm="Compute 1 / standard_track_spectral_frequency_hz.",
        unit="s",
        coordinate_space="time in seconds",
        aggregation_level="track",
        modes=("standard",),
        validity="Valid when the corresponding spectral frequency is finite and positive.",
        quality_fields=("spectral_snr", "frequency_agreement_error"),
    ),
    MeasurementDefinition(
        key="standard_event_boundary_period_s",
        familiar_label="Period in Seconds",
        quantity="Local standard-event period",
        operational_definition="Time between the fitted left and right boundaries surrounding one detected event.",
        algorithm="Fit the local event waveform and subtract its fitted boundary times.",
        unit="s",
        coordinate_space="time in seconds",
        aggregation_level="detected event on one track",
        modes=("standard",),
        validity="Valid when both fitted boundaries are finite, ordered, and accepted by the event fit.",
        quality_fields=("fit_r2", "fit_rmse_px", "period_boundary_error_fraction"),
    ),
    MeasurementDefinition(
        key="standard_event_boundary_frequency_hz",
        familiar_label="Frequency (Hz)",
        quantity="Local standard-event frequency",
        operational_definition="Reciprocal of the fitted boundary period for one detected event.",
        algorithm="Compute 1 / standard_event_boundary_period_s.",
        unit="Hz",
        coordinate_space="time in seconds",
        aggregation_level="detected event on one track",
        modes=("standard",),
        validity="Valid when the corresponding boundary period is finite and positive.",
        quality_fields=("fit_r2", "fit_rmse_px", "period_boundary_error_fraction"),
    ),
    MeasurementDefinition(
        key="ripple_intertrack_interval_s",
        familiar_label="Period in Seconds",
        quantity="Ripple intertrack arrival interval",
        operational_definition="Median time separation between a neighboring pair of fitted ripple tracks over their shared horizontal range.",
        algorithm="Evaluate both fitted track lines at shared horizontal samples and take the median positive frame gap, converted by the sampling rate.",
        unit="s",
        coordinate_space="time in seconds; horizontal position in processed-image pixels",
        aggregation_level="neighboring track pair",
        modes=("ripple_family",),
        validity="Valid when the tracks overlap, the median gap is within configured limits, and gap variability passes its threshold.",
        quality_fields=("ripple_intertrack_gap_mad_frames", "ripple_intertrack_gap_cv"),
    ),
    MeasurementDefinition(
        key="ripple_intertrack_arrival_rate_hz",
        familiar_label="Frequency",
        quantity="Ripple intertrack arrival rate",
        operational_definition="Reciprocal of the ripple intertrack arrival interval.",
        algorithm="Compute 1 / ripple_intertrack_interval_s.",
        unit="Hz",
        coordinate_space="time in seconds",
        aggregation_level="neighboring track pair",
        modes=("ripple_family",),
        validity="Valid when the corresponding intertrack interval is finite and positive.",
        quality_fields=("ripple_intertrack_gap_mad_frames", "ripple_intertrack_gap_cv"),
    ),
    MeasurementDefinition(
        key="ripple_track_median_neighbor_arrival_rate_hz",
        familiar_label="Frequency",
        quantity="Track-level ripple arrival rate",
        operational_definition="Arrival rate derived from the median accepted interval between this track and its neighboring tracks.",
        algorithm="Take the median neighboring-track frame gap and convert its reciprocal using the sampling rate.",
        unit="Hz",
        coordinate_space="time in seconds",
        aggregation_level="track",
        modes=("ripple_family",),
        validity="Valid when the track participates in at least one accepted neighboring-track interval.",
        quality_fields=("ripple_neighbor_interval_count",),
    ),
    MeasurementDefinition(
        key="ripple_family_median_arrival_rate_hz",
        familiar_label="Avg frequency",
        quantity="Family-level ripple arrival rate",
        operational_definition="Median arrival rate across accepted neighboring-track intervals in one ripple family.",
        algorithm="Take the median of accepted ripple_intertrack_arrival_rate_hz values within a family.",
        unit="Hz",
        coordinate_space="time in seconds",
        aggregation_level="track family",
        modes=("ripple_family",),
        validity="Valid when the family contains at least one accepted neighboring-track interval.",
        quality_fields=("ripple_family_arrival_rate_iqr_hz", "ripple_family_interval_count"),
    ),
    MeasurementDefinition(
        key="ripple_propagation_velocity_px_per_s",
        familiar_label="Velocity",
        quantity="Signed ripple propagation velocity",
        operational_definition="Signed horizontal displacement per second along a fitted ripple track.",
        algorithm="Multiply fitted track slope in pixels per frame by the sampling rate.",
        unit="processed px/s",
        coordinate_space="bottom-left frame/processed-image-position coordinates",
        aggregation_level="track or neighboring track pair",
        modes=("ripple_family",),
        validity="Valid when the fitted track slope and sampling rate are finite.",
        quality_fields=("ripple_line_fit_rmse_px", "ripple_line_fit_r2"),
    ),
    MeasurementDefinition(
        key="ripple_angle_from_time_axis_deg",
        familiar_label="Angle",
        quantity="Ripple-track angle from the time axis",
        operational_definition="Angle of the fitted ripple track measured from the positive time axis.",
        algorithm="Convert the fitted pixels-per-frame slope to an angle using arctangent.",
        unit="degrees",
        coordinate_space="bottom-left frame/processed-image-position coordinates",
        aggregation_level="track or neighboring track pair",
        modes=("ripple_family",),
        validity="Valid when the track line fit is finite.",
        quality_fields=("ripple_line_fit_rmse_px", "ripple_line_fit_r2"),
    ),
    MeasurementDefinition(
        key="large_wave_equivalent_lobe_period_s",
        familiar_label="Period",
        quantity="Large-wave equivalent-lobe period",
        operational_definition="Equivalent full-cycle duration inferred from the fitted asymmetric large-wave lobe; it is not the interval between large-wave events.",
        algorithm="Fit the asymmetric local lobe and combine its fitted left and right quarter-period terms into an equivalent full-cycle period.",
        unit="s",
        coordinate_space="time in seconds",
        aggregation_level="large-wave measurement on one track",
        modes=("large_wave",),
        validity="Valid when the asymmetric fit and its period boundaries pass configured checks.",
        quality_fields=("fit_r2", "fit_rmse_px", "period_boundary_error_fraction"),
    ),
    MeasurementDefinition(
        key="large_wave_equivalent_lobe_frequency_hz",
        familiar_label="Frequency",
        quantity="Large-wave equivalent-lobe frequency",
        operational_definition="Reciprocal of the fitted equivalent-lobe period; it is not the rate at which large waves recur.",
        algorithm="Compute 1 / large_wave_equivalent_lobe_period_s.",
        unit="Hz",
        coordinate_space="time in seconds",
        aggregation_level="large-wave measurement on one track",
        modes=("large_wave",),
        validity="Valid when the corresponding equivalent-lobe period is finite and positive.",
        quality_fields=("fit_r2", "fit_rmse_px", "period_boundary_error_fraction"),
    ),
    MeasurementDefinition(
        key="large_wave_local_recurrence_rate_hz",
        familiar_label="Recurrence Frequency (Hz)",
        quantity="Local large-wave recurrence rate",
        operational_definition="Reciprocal of the local interval to a neighboring accepted large-wave peak of the same polarity on one track.",
        algorithm="Take the adjacent same-polarity peak-frame gap and convert its reciprocal using the sampling rate.",
        unit="Hz",
        coordinate_space="time in seconds",
        aggregation_level="large-wave measurement on one track",
        modes=("large_wave",),
        validity="Valid when another accepted same-polarity large-wave peak supplies a positive neighboring interval.",
    ),
    MeasurementDefinition(
        key="large_wave_track_median_recurrence_rate_hz",
        familiar_label="Recurrence freq",
        quantity="Track-level large-wave recurrence rate",
        operational_definition="Median local recurrence rate across accepted large-wave measurements on one track, with grouped-event recurrence used when local intervals are unavailable.",
        algorithm="Take the median valid local same-polarity recurrence rate, or the median associated grouped-event recurrence rate as a fallback.",
        unit="Hz",
        coordinate_space="time in seconds",
        aggregation_level="track",
        modes=("large_wave",),
        validity="Valid when at least one local or associated grouped-event recurrence interval is finite and positive.",
    ),
    MeasurementDefinition(
        key="large_wave_global_event_recurrence_rate_hz",
        familiar_label="Frequency (Hz)",
        quantity="Grouped large-wave event recurrence rate",
        operational_definition="Reciprocal of the interval from the preceding grouped large-wave event of the same polarity.",
        algorithm="Group coincident per-track measurements, order groups by center time, and invert consecutive same-polarity time gaps.",
        unit="Hz",
        coordinate_space="time in seconds",
        aggregation_level="cross-track grouped event",
        modes=("large_wave",),
        validity="Valid for grouped events that have a preceding same-polarity event with a positive time gap.",
        quality_fields=("large_wave_peak_frame_mad", "large_wave_peak_frame_span"),
    ),
)

MEASUREMENT_DEFINITIONS: Dict[str, MeasurementDefinition] = {
    definition.key: definition for definition in _DEFINITIONS
}
if len(MEASUREMENT_DEFINITIONS) != len(_DEFINITIONS):
    raise RuntimeError("Measurement keys must be unique")


def _canonical_definition_payload() -> Dict[str, Any]:
    return {
        "version": MEASUREMENT_SCHEMA_VERSION,
        "definitions": [asdict(definition) for definition in _DEFINITIONS],
        "exports": measurement_export_contract(),
    }


def measurement_schema_sha256() -> str:
    encoded = json.dumps(
        _canonical_definition_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def measurement_schema_identity() -> Dict[str, Any]:
    return {
        "version": MEASUREMENT_SCHEMA_VERSION,
        "sha256": measurement_schema_sha256(),
        "default_column_labels": DEFAULT_COLUMN_LABEL_PROFILE,
        "available_column_labels": list(COLUMN_LABEL_PROFILES),
    }


def measurement_schema_payload() -> Dict[str, Any]:
    return {
        **measurement_schema_identity(),
        "definitions": [asdict(definition) for definition in _DEFINITIONS],
        "exports": measurement_export_contract(),
    }


WAVE_EXPORT_FAMILIAR_HEADERS = (
    "wave_id", "track_id", "wave_index",
    "Frame position 1 (y-axis)", "Frame position 2 (y-axis)",
    "Period In Frames (Frame 1- Frame 2)", "Period in Seconds", "Frequency (Hertz)",
    "Period Source", "Pixel Position 1 (x-axis)", "Pixel Position 2 (x-axis)",
    "Amplitude (Pixels)", "Signed Amplitude (Pixels)", "Position 1 (x-axis)",
    "Position 2 (x-axis)", "Frame 1 (y-axis)", "Frame 2 (y-axis)",
    "Frame 1 (seconds)", "Frame 2 (seconds)", "Seconds 2 - Seconds 1",
    "Position2 -Position 1", "Velocity (pixels/sec)", "Frequency (Hz)",
    "Wavelength (Pixels)", "Peak Frame (y-axis)", "Peak Position (x-axis)",
    "Event Kind", "Event Polarity", "Event Value", "Peak Value Original", "Fit Target",
    "Compare Fit Targets", "Peak Frame Raw", "Peak Position Raw", "Frame 1 Raw", "Frame 2 Raw",
    "Fit Error (VNMSE)", "Fit Passes Peak", "Fit R2", "Fit RMSE (px)", "Fit NRMSE",
    "Fit MAE (px)", "Fit Points", "Residual Fit Error (VNMSE)", "Residual Fit R2",
    "Residual Fit RMSE (px)", "Raw Fit Error (VNMSE)", "Raw Fit R2", "Raw Fit RMSE (px)",
    "Track Fit Error Median", "Track Fit R2 Median", "Period Consistency CV",
    "Frequency Agreement Error", "Spectral SNR", "Peak Prominence SNR",
    "Config Event Polarity", "Endpoint Linking Enabled", "Endpoint Linking Level",
    "Fit Start Frame Raw", "Fit End Frame Raw", "Fit Duration (frames)",
    "Fit Duration (seconds)", "Period Asymmetry", "Period Boundary Error (fraction)",
    "Period Estimate Valid", "Recurrence Period (frames)", "Recurrence Period (seconds)",
    "Recurrence Frequency (Hz)", "Wave Type", "Type Score", "Detrend Method",
    "Detrend Fallback Used", "Detrend Fallback Reason", "Detrend Inlier Fraction",
)

_WAVE_EXPORT_BASE_KEYS = (
    "wave_id", "track_id", "wave_index",
    "event_boundary_start_frame", "event_boundary_end_frame", "event_period_frames",
    "event_period_s", "event_frequency_hz", "event_period_method",
    "event_boundary_start_position_px", "event_boundary_end_position_px",
    "event_amplitude_px", "event_signed_amplitude_px",
    "event_boundary_start_position_px", "event_boundary_end_position_px",
    "event_boundary_start_frame", "event_boundary_end_frame",
    "event_boundary_start_time_s", "event_boundary_end_time_s", "event_period_s",
    "event_displacement_px", "event_mean_velocity_px_per_s", "event_frequency_hz",
    "event_spatial_span_px", "event_peak_frame", "event_peak_position_px", "event_kind",
    "event_polarity", "event_value_px", "event_peak_residual_px", "fit_target",
    "fit_targets_compared", "event_peak_image_row", "event_peak_position_raw_px",
    "event_boundary_start_image_row", "event_boundary_end_image_row", "fit_error_vnmse",
    "fit_passes_peak", "fit_r2", "fit_rmse_px", "fit_nrmse", "fit_mae_px", "fit_point_count",
    "residual_fit_error_vnmse", "residual_fit_r2", "residual_fit_rmse_px",
    "raw_fit_error_vnmse", "raw_fit_r2", "raw_fit_rmse_px", "track_fit_error_median",
    "track_fit_r2_median", "track_period_consistency_cv", "track_frequency_agreement_error",
    "track_spectral_snr", "event_peak_prominence_snr", "configured_event_polarity",
    "endpoint_linking_enabled", "endpoint_linking_level", "fit_start_frame",
    "fit_end_frame", "fit_duration_frames", "fit_duration_s", "period_asymmetry",
    "period_boundary_error_fraction", "period_estimate_valid", "event_recurrence_interval_frames",
    "event_recurrence_interval_s", "event_recurrence_rate_hz", "wave_type", "wave_type_score",
    "detrend_method", "detrend_fallback_used", "detrend_fallback_reason", "detrend_inlier_fraction",
)

if len(WAVE_EXPORT_FAMILIAR_HEADERS) != len(_WAVE_EXPORT_BASE_KEYS):
    raise RuntimeError("Wave export headers and canonical keys are out of sync")


def wave_export_descriptive_keys(analysis_mode: str) -> Tuple[str, ...]:
    prefix = "large_wave_equivalent_lobe" if analysis_mode == LARGE_WAVE_ANALYSIS_MODE else "standard_event_boundary"
    replacements = {
        "event_boundary_start_frame": f"{prefix}_start_frame",
        "event_boundary_end_frame": f"{prefix}_end_frame",
        "event_period_frames": f"{prefix}_period_frames",
        "event_period_s": f"{prefix}_period_s",
        "event_frequency_hz": f"{prefix}_frequency_hz",
        "event_period_method": f"{prefix}_period_method",
        "event_boundary_start_position_px": f"{prefix}_start_position_px",
        "event_boundary_end_position_px": f"{prefix}_end_position_px",
        "event_boundary_start_time_s": f"{prefix}_start_time_s",
        "event_boundary_end_time_s": f"{prefix}_end_time_s",
        "event_displacement_px": f"{prefix}_displacement_px",
        "event_mean_velocity_px_per_s": f"{prefix}_mean_velocity_px_per_s",
        "event_spatial_span_px": f"{prefix}_spatial_span_px",
        "event_recurrence_interval_frames": "large_wave_local_recurrence_interval_frames",
        "event_recurrence_interval_s": "large_wave_local_recurrence_interval_s",
        "event_recurrence_rate_hz": "large_wave_local_recurrence_rate_hz",
    }
    return tuple(replacements.get(key, key) for key in _WAVE_EXPORT_BASE_KEYS)


def profile_csv_columns(
    familiar_headers: Sequence[str],
    descriptive_headers: Sequence[str],
    values: Sequence[Any],
    profile: ColumnLabelProfile,
) -> Tuple[list[str], list[Any]]:
    if len(familiar_headers) != len(descriptive_headers) or len(values) != len(familiar_headers):
        raise ValueError("CSV headers and values must have the same length")
    if profile == "familiar":
        return list(familiar_headers), list(values)

    headers: list[str] = []
    projected_values: list[Any] = []
    seen: set[str] = set()
    for header, value in zip(descriptive_headers, values):
        if header in seen:
            continue
        seen.add(header)
        headers.append(header)
        projected_values.append(value)
    return headers, projected_values


_RIPPLE_CANONICAL_FIELDS = {
    "tracks": (
        "track_index", "family_id", "family_label", "direction", "point_count",
        "slope_px_per_frame", "velocity_px_per_s", "speed_px_per_s", "angle_deg",
        "angle_from_time_axis_deg", "line_intercept_px", "line_rmse_px", "line_fit_rmse_px",
        "line_r2", "duration_frames", "duration_s", "spatial_span_px", "x_start_px", "x_end_px",
        "y_start_frame", "y_end_frame", "neighbor_interval_count", "period_frames", "period_s",
        "frequency_hz", "frequency_method", "eligible",
    ),
    "intervals": (
        "interval_index", "family_id", "family_label", "direction", "earlier_track_index",
        "later_track_index", "x_overlap_start_px", "x_overlap_end_px", "sample_count",
        "slope_px_per_frame", "velocity_px_per_s", "speed_px_per_s", "angle_deg",
        "angle_from_time_axis_deg", "period_frames", "period_s", "frequency_hz",
        "gap_mad_frames", "gap_cv", "measurement_method",
    ),
    "families": (
        "family_id", "family_label", "direction", "track_count", "interval_count", "track_indices",
        "median_slope_px_per_frame", "median_velocity_px_per_s", "median_speed_px_per_s",
        "median_angle_deg", "median_angle_from_time_axis_deg", "median_period_frames",
        "median_period_s", "median_frequency_hz", "frequency_iqr_hz", "x_min_px", "x_max_px",
        "y_min_frame", "y_max_frame", "frequency_method",
    ),
}

_RIPPLE_RENAMES = {
    "tracks": {
        "slope_px_per_frame": "ripple_track_slope_px_per_frame",
        "velocity_px_per_s": "ripple_propagation_velocity_px_per_s",
        "speed_px_per_s": "ripple_propagation_speed_px_per_s",
        "angle_deg": "ripple_angle_from_time_axis_deg",
        "angle_from_time_axis_deg": "ripple_angle_from_time_axis_deg",
        "line_intercept_px": "ripple_line_intercept_px",
        "line_rmse_px": "ripple_line_fit_rmse_px",
        "line_fit_rmse_px": "ripple_line_fit_rmse_px",
        "line_r2": "ripple_line_fit_r2",
        "neighbor_interval_count": "ripple_neighbor_interval_count",
        "period_frames": "ripple_track_median_neighbor_interval_frames",
        "period_s": "ripple_track_median_neighbor_interval_s",
        "frequency_hz": "ripple_track_median_neighbor_arrival_rate_hz",
        "frequency_method": "ripple_arrival_rate_method",
    },
    "intervals": {
        "interval_index": "ripple_interval_id",
        "slope_px_per_frame": "ripple_pair_median_slope_px_per_frame",
        "velocity_px_per_s": "ripple_propagation_velocity_px_per_s",
        "speed_px_per_s": "ripple_propagation_speed_px_per_s",
        "angle_deg": "ripple_angle_from_time_axis_deg",
        "angle_from_time_axis_deg": "ripple_angle_from_time_axis_deg",
        "period_frames": "ripple_intertrack_interval_frames",
        "period_s": "ripple_intertrack_interval_s",
        "frequency_hz": "ripple_intertrack_arrival_rate_hz",
        "gap_mad_frames": "ripple_intertrack_gap_mad_frames",
        "gap_cv": "ripple_intertrack_gap_cv",
        "measurement_method": "ripple_interval_method",
    },
    "families": {
        "interval_count": "ripple_family_interval_count",
        "median_slope_px_per_frame": "ripple_family_median_slope_px_per_frame",
        "median_velocity_px_per_s": "ripple_family_median_velocity_px_per_s",
        "median_speed_px_per_s": "ripple_family_median_speed_px_per_s",
        "median_angle_deg": "ripple_family_median_angle_from_time_axis_deg",
        "median_angle_from_time_axis_deg": "ripple_family_median_angle_from_time_axis_deg",
        "median_period_frames": "ripple_family_median_intertrack_interval_frames",
        "median_period_s": "ripple_family_median_intertrack_interval_s",
        "median_frequency_hz": "ripple_family_median_arrival_rate_hz",
        "frequency_iqr_hz": "ripple_family_arrival_rate_iqr_hz",
        "frequency_method": "ripple_family_arrival_rate_method",
    },
}


def measurement_export_contract() -> Dict[str, Any]:
    waves = {
        mode: [
            {"familiar": familiar, "descriptive": descriptive}
            for familiar, descriptive in zip(
                WAVE_EXPORT_FAMILIAR_HEADERS,
                wave_export_descriptive_keys(mode),
            )
        ]
        for mode in ("standard", "large_wave")
    }
    ripple: Dict[str, list[Dict[str, str]]] = {}
    for export_name, source_fields in _RIPPLE_CANONICAL_FIELDS.items():
        seen: set[str] = set()
        columns: list[Dict[str, str]] = []
        for source_field in source_fields:
            descriptive = _RIPPLE_RENAMES[export_name].get(source_field, source_field)
            if descriptive in seen:
                continue
            seen.add(descriptive)
            columns.append({"stored": source_field, "descriptive": descriptive})
        ripple[export_name] = columns
    return {"waves": waves, "ripple": ripple}


def descriptive_ripple_csv(data: bytes, export_name: str) -> bytes:
    name = export_name.strip().lower()
    source_fields = _RIPPLE_CANONICAL_FIELDS.get(name)
    if source_fields is None:
        raise ValueError(f"Unknown ripple export {export_name!r}")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    available = set(reader.fieldnames or ())
    missing = [field for field in source_fields if field not in available]
    if missing:
        raise ValueError(f"Ripple {name} CSV is missing canonical columns: {', '.join(missing)}")

    rename = _RIPPLE_RENAMES[name]
    output_fields: list[str] = []
    selected_fields: list[str] = []
    seen: set[str] = set()
    for source_field in source_fields:
        output_field = rename.get(source_field, source_field)
        if output_field in seen:
            continue
        seen.add(output_field)
        selected_fields.append(source_field)
        output_fields.append(output_field)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(output_fields)
    for row in reader:
        writer.writerow([row.get(field, "") for field in selected_fields])
    return output.getvalue().encode("utf-8")
