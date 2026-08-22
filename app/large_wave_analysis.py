from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cancel import CancellationRequested
from .extract_core import load_track_frame_position
from .large_wave_fit import LargeWaveFit, fit_large_wave
from .signal.detrend import detrend_residual


CancelCallback = Optional[Callable[[], bool]]
ProgressCallback = Optional[Callable[[str, int, int], None]]


@dataclass(frozen=True)
class LargeWaveAnalysisResult:
    wave_rows: List[Dict[str, Any]]
    measurements: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    track_summaries: Dict[int, Dict[str, Any]]
    tracks_csv: bytes
    measurements_csv: bytes
    events_csv: bytes


LARGE_WAVE_TRACK_FIELDS = [
    "Track ID",
    "Points",
    "Broad Peaks",
    "Grouped Large Waves",
    "Large Wave IDs",
    "Median Amplitude (px)",
    "Maximum Amplitude (px)",
    "Median Peak Width (frames)",
    "Median Peak Width (seconds)",
    "Maximum Speed (pixels/sec)",
    "Median Apex Curvature (px/frame^2)",
    "Median Frequency (Hz)",
    "Median Recurrence Frequency (Hz)",
    "track_index",
    "point_count",
    "large_wave_measurement_count",
    "large_wave_event_count",
    "large_wave_event_ids",
    "mean_large_wave_amplitude_px",
    "max_large_wave_amplitude_px",
    "mean_peak_width_frames",
    "mean_peak_width_s",
    "max_speed_px_per_s",
    "mean_apex_curvature_px_per_frame2",
    "large_wave_frequency_hz",
    "large_wave_recurrence_frequency_hz",
]

STANDARD_WAVE_FIELDS = [
    "wave_id",
    "track_id",
    "wave_index",
    "Frame position 1 (y-axis)",
    "Frame position 2 (y-axis)",
    "Period In Frames (Frame 1- Frame 2)",
    "Period in Seconds",
    "Frequency (Hertz)",
    "Period Source",
    "Pixel Position 1 (x-axis)",
    "Pixel Position 2 (x-axis)",
    "Amplitude (Pixels)",
    "Signed Amplitude (Pixels)",
    "Position 1 (x-axis)",
    "Position 2 (x-axis)",
    "Frame 1 (y-axis)",
    "Frame 2 (y-axis)",
    "Frame 1 (seconds)",
    "Frame 2 (seconds)",
    "Seconds 2 - Seconds 1",
    "Position2 -Position 1",
    "Velocity (pixels/sec)",
    "Frequency (Hz)",
    "Wavelength (Pixels)",
    "Peak Frame (y-axis)",
    "Peak Position (x-axis)",
    "Event Kind",
    "Event Polarity",
    "Event Value",
    "Peak Value Original",
    "Fit Target",
    "Compare Fit Targets",
    "Peak Frame Raw",
    "Peak Position Raw",
    "Frame 1 Raw",
    "Frame 2 Raw",
    "Fit Error (VNMSE)",
    "Fit Passes Peak",
    "Fit R2",
    "Fit RMSE (px)",
    "Fit NRMSE",
    "Fit MAE (px)",
    "Fit Points",
    "Residual Fit Error (VNMSE)",
    "Residual Fit R2",
    "Residual Fit RMSE (px)",
    "Raw Fit Error (VNMSE)",
    "Raw Fit R2",
    "Raw Fit RMSE (px)",
    "Track Fit Error Median",
    "Track Fit R2 Median",
    "Period Consistency CV",
    "Frequency Agreement Error",
    "Spectral SNR",
    "Peak Prominence SNR",
    "Config Event Polarity",
    "Endpoint Linking Enabled",
    "Endpoint Linking Level",
    "Fit Start Frame Raw",
    "Fit End Frame Raw",
    "Fit Duration (frames)",
    "Fit Duration (seconds)",
    "Period Asymmetry",
    "Period Boundary Error (fraction)",
    "Period Estimate Valid",
    "Recurrence Period (frames)",
    "Recurrence Period (seconds)",
    "Recurrence Frequency (Hz)",
    "Wave Type",
    "Type Score",
]

LARGE_WAVE_MEASUREMENT_FIELDS = [
    *STANDARD_WAVE_FIELDS,
    "Measurement ID",
    "Large Wave ID",
    "Track ID",
    "Direction",
    "Peak Frame",
    "Peak Time (seconds)",
    "Peak Position (px)",
    "Signed Amplitude (px)",
    "Amplitude (px)",
    "Peak Prominence (px)",
    "Width at Half Prominence (frames)",
    "Width at Half Prominence (seconds)",
    "Rise Time (seconds)",
    "Recovery Time (seconds)",
    "Maximum Approach Speed (pixels/sec)",
    "Maximum Recovery Speed (pixels/sec)",
    "Maximum Speed (pixels/sec)",
    "Apex Curvature (px/frame^2)",
    "Integrated Displacement (pixel-seconds)",
    "Left Quarter Period (frames)",
    "Right Quarter Period (frames)",
    "Period from Left Side (frames)",
    "Period from Right Side (frames)",
    "Baseline Start Position (px)",
    "Baseline End Position (px)",
    "Baseline Slope (px/frame)",
    "Baseline Velocity (pixels/sec)",
    "Left Shape Power",
    "Right Shape Power",
    "Fit Window Source",
    "Boundary Extrapolated",
    "Fit Boundary Extrapolated",
    "Equivalent Cycle Boundary Extrapolated",
    "Grouped Event",
    "measurement_id",
    "event_id",
    "track_index",
    "direction",
    "event_kind",
    "peak_frame",
    "peak_time_s",
    "peak_position_px",
    "signed_amplitude_px",
    "amplitude_px",
    "prominence_px",
    "width_half_prominence_frames",
    "width_half_prominence_s",
    "rise_time_s",
    "recovery_time_s",
    "max_approach_speed_px_per_s",
    "max_recovery_speed_px_per_s",
    "max_speed_px_per_s",
    "apex_curvature_px_per_frame2",
    "integrated_displacement_px_s",
    "grouped_event",
]

LARGE_WAVE_EVENT_FIELDS = [
    "Large Wave ID",
    "Direction",
    "Tracks",
    "Track IDs",
    "Center Frame",
    "Center Time (seconds)",
    "Median Amplitude (px)",
    "Maximum Amplitude (px)",
    "Amplitude IQR (px)",
    "Median Prominence (px)",
    "Median Width (frames)",
    "Median Width (seconds)",
    "Median Rise Time (seconds)",
    "Median Recovery Time (seconds)",
    "Maximum Speed (pixels/sec)",
    "Median Apex Curvature (px/frame^2)",
    "Peak Frame MAD",
    "Peak Frame Span",
    "Spatial Coverage (px)",
    "Period from Previous Event (frames)",
    "Period from Previous Event (seconds)",
    "Frequency (Hz)",
    "event_id",
    "direction",
    "event_kind",
    "track_count",
    "track_indices",
    "center_frame",
    "center_time_s",
    "median_amplitude_px",
    "max_amplitude_px",
    "amplitude_iqr_px",
    "median_prominence_px",
    "median_width_frames",
    "median_width_s",
    "median_rise_time_s",
    "median_recovery_time_s",
    "max_speed_px_per_s",
    "median_apex_curvature_px_per_frame2",
    "peak_frame_mad",
    "peak_frame_span",
    "spatial_coverage_px",
    "period_from_previous_frames",
    "period_from_previous_s",
    "frequency_hz",
]


def build_large_wave_track_config(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(config)
    large_cfg = (((resolved.get("analysis") or {}).get("large_wave") or {}))
    peak_overrides = large_cfg.get("peaks") or {}
    resolved["peaks"] = {**(resolved.get("peaks") or {}), **peak_overrides}
    return resolved


def analyze_large_wave_events(
    *,
    track_paths: Sequence[Path],
    track_rows: List[Dict[str, Any]],
    wave_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    cancel_cb: CancelCallback = None,
    progress_cb: ProgressCallback = None,
) -> LargeWaveAnalysisResult:
    large_cfg = (((config.get("analysis") or {}).get("large_wave") or {}))
    event_cfg = large_cfg.get("events") or {}
    sampling_rate = float((config.get("io") or {}).get("sampling_rate", 1.0))
    track_order = _track_order(config)
    by_track: Dict[int, List[Dict[str, Any]]] = {}
    for row in wave_rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        try:
            track_index = int(metrics.get("track_index"))
        except Exception:
            continue
        by_track.setdefault(track_index, []).append(row)

    measurements: List[Dict[str, Any]] = []
    kept_wave_rows: List[Dict[str, Any]] = []
    total_tracks = len(track_paths)
    for track_index, track_path in enumerate(track_paths):
        _check_cancel(cancel_cb)
        rows = by_track.get(track_index, [])
        if rows:
            frame, position = load_track_frame_position(track_path, order=track_order)
            residual = detrend_residual(frame, position, **(config.get("detrend") or {}))
            global_baseline = np.asarray(position, dtype=float) - residual
            track_candidates: List[Dict[str, Any]] = []
            for row in rows:
                metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                try:
                    seed_peak_i = int(metrics.get("peak_i"))
                except (TypeError, ValueError):
                    continue
                event_kind = "min" if str(metrics.get("event_kind", row.get("event_kind"))) == "min" else "max"
                fallback_peak = bool(metrics.get("fallback_peak", False))
                minimum_width_frames = _optional_float(metrics.get("bulge_width_frames"))
                if fallback_peak:
                    fallback_width = max(
                        3.0,
                        float(event_cfg.get("fallback_minimum_width_frames", 30.0)),
                    )
                    minimum_width_frames = max(minimum_width_frames or 0.0, fallback_width)
                fit = fit_large_wave(
                    frame=frame,
                    position=position,
                    global_baseline=global_baseline,
                    center_idx=seed_peak_i,
                    event_kind=event_kind,
                    sampling_rate=sampling_rate,
                    minimum_width_frames=minimum_width_frames,
                    width_multiplier=float(event_cfg.get("fit_window_width_multiplier", 1.0)),
                    boundary_smoothing_sigma_rows=float(
                        event_cfg.get("fit_boundary_smoothing_sigma_rows", 4.0)
                    ),
                    boundary_smoothing_scales=event_cfg.get(
                        "fit_boundary_smoothing_scales", [1.0]
                    ),
                    boundary_max_sigma_fraction=float(
                        event_cfg.get("fit_boundary_max_sigma_fraction", 0.125)
                    ),
                    boundary_baseline_tolerance_fraction=float(
                        event_cfg.get("fit_boundary_baseline_tolerance_fraction", 0.02)
                    ),
                    fit_error_tolerance=float(
                        event_cfg.get("fit_candidate_error_tolerance", 0.25)
                    ),
                    max_fit_error_vnmse=float(
                        event_cfg.get("fit_candidate_max_error_vnmse", 0.8)
                    ),
                    endpoint_anchor_rows=int(event_cfg.get("endpoint_anchor_rows", 7)),
                    curvature_half_window_rows=int(event_cfg.get("curvature_half_window_rows", 8)),
                    max_period_boundary_error_fraction=float(
                        event_cfg.get("max_period_boundary_error_fraction", 0.5)
                    ),
                )
                if fit is None:
                    continue
                measurement = _measure_fit(
                    row=row,
                    track_index=track_index,
                    frame=frame,
                    fit=fit,
                    sampling_rate=sampling_rate,
                    cfg=event_cfg,
                )
                if measurement is None:
                    continue
                measurement["_source_row"] = row
                track_candidates.append(measurement)

            for measurement in _dedupe_track_measurements(
                track_candidates,
                cfg=event_cfg,
                track_frame_min=float(frame[0]),
                track_frame_max=float(frame[-1]),
            ):
                measurement_id = f"LWM{len(measurements) + 1:04d}"
                measurement["measurement_id"] = measurement_id
                row = measurement.pop("_source_row")
                row_metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                row_metrics["large_wave_measurement_id"] = measurement_id
                row["metrics"] = row_metrics
                measurements.append(measurement)
                kept_wave_rows.append(row)
        if progress_cb is not None:
            progress_cb("large_wave_peak_measurement", track_index + 1, total_tracks)

    _assign_recurrence_periods(measurements, sampling_rate=sampling_rate)
    events = _group_measurements(measurements, sampling_rate=sampling_rate, cfg=event_cfg)
    if progress_cb is not None:
        progress_cb("large_wave_event_grouping", len(measurements), len(measurements))

    track_summaries = _summarize_tracks(measurements, events, track_paths)
    for measurement in measurements:
        summary = track_summaries.get(int(measurement["track_index"]), {})
        measurement.update({
            "track_fit_error_median": summary.get("track_fit_error_median"),
            "track_fit_r2_median": summary.get("track_fit_r2_median"),
            "period_consistency_cv": summary.get("period_consistency_cv"),
            "frequency_agreement_error": summary.get("frequency_agreement_error"),
        })

    measurement_by_id = {row["measurement_id"]: row for row in measurements}
    for row in kept_wave_rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        measurement_id = metrics.get("large_wave_measurement_id")
        measurement = measurement_by_id.get(str(measurement_id))
        if measurement:
            _apply_measurement_to_wave_row(row, measurement)
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            row["metrics"] = metrics

    for track_index in range(len(track_paths)):
        track_summaries.setdefault(track_index, _empty_track_summary(track_index, track_paths))
    for track_row in track_rows:
        track_index = int(track_row.get("track_index", -1))
        summary = track_summaries.get(track_index, _empty_track_summary(track_index, track_paths))
        metrics = track_row.get("metrics") if isinstance(track_row.get("metrics"), dict) else {}
        metrics.update({
            "analysis_mode": "large_wave",
            "num_peaks": summary.get("large_wave_measurement_count", 0),
            **summary,
        })
        track_row["metrics"] = metrics
        track_row["amplitude"] = summary.get("mean_large_wave_amplitude_px")
        track_row["frequency"] = summary.get("large_wave_frequency_hz")

    return LargeWaveAnalysisResult(
        wave_rows=kept_wave_rows,
        measurements=measurements,
        events=events,
        track_summaries=track_summaries,
        tracks_csv=_csv_bytes(
            [_track_csv_row(row, track_summaries.get(int(row.get("track_index", -1)), {})) for row in track_rows],
            LARGE_WAVE_TRACK_FIELDS,
        ),
        measurements_csv=_csv_bytes([_measurement_csv_row(row) for row in measurements], LARGE_WAVE_MEASUREMENT_FIELDS),
        events_csv=_csv_bytes([_event_csv_row(row) for row in events], LARGE_WAVE_EVENT_FIELDS),
    )


def _measure_fit(
    *,
    row: Dict[str, Any],
    track_index: int,
    frame: np.ndarray,
    fit: LargeWaveFit,
    sampling_rate: float,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    source_metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    fallback_peak = bool(source_metrics.get("fallback_peak", False))
    fit_metrics = dict(fit.metrics)
    peak_i = int(fit.center_idx)
    event_kind = fit.event_kind
    signed_amplitude = float(fit_metrics["signed_amplitude_px"])
    amplitude = float(fit_metrics["amplitude_px"])
    # With a baseline-return fit, fitted prominence and fitted amplitude are equivalent.
    prominence = amplitude
    width_frames = float(fit_metrics["width_half_prominence_frames"])

    if not fallback_peak and amplitude < float(cfg.get("min_amplitude_px", 3.0)):
        return None
    if not fallback_peak and prominence < float(cfg.get("min_prominence_px", 2.0)):
        return None
    if not fallback_peak and width_frames < float(cfg.get("min_width_frames", 5.0)):
        return None

    measurement = {
        "event_id": None,
        "track_index": track_index,
        "wave_index": row.get("wave_index"),
        "direction": fit_metrics["direction"],
        "event_kind": event_kind,
        "peak_frame": fit_metrics["peak_frame"],
        "peak_time_s": float(frame[peak_i]) / sampling_rate if sampling_rate > 0 else None,
        "peak_position_px": fit_metrics["peak_position_px"],
        "signed_amplitude_px": signed_amplitude,
        "amplitude_px": amplitude,
        "prominence_px": prominence,
        "width_half_prominence_frames": width_frames,
        "observed_prominence_px": _optional_float(source_metrics.get("bulge_prominence_px")),
        "fit_start_time_s": (
            float(fit_metrics["fit_start_frame"]) / sampling_rate if sampling_rate > 0 else None
        ),
        "fit_end_time_s": (
            float(fit_metrics["fit_end_frame"]) / sampling_rate if sampling_rate > 0 else None
        ),
        "frame1_time_s": (
            float(fit_metrics["equivalent_cycle_frame1"]) / sampling_rate
            if sampling_rate > 0 and fit_metrics.get("equivalent_cycle_frame1") is not None
            else None
        ),
        "frame2_time_s": (
            float(fit_metrics["equivalent_cycle_frame2"]) / sampling_rate
            if sampling_rate > 0 and fit_metrics.get("equivalent_cycle_frame2") is not None
            else None
        ),
        "grouped_event": False,
        "fallback_peak": fallback_peak,
        **fit_metrics,
    }
    return measurement


def _dedupe_track_measurements(
    candidates: List[Dict[str, Any]],
    *,
    cfg: Optional[Dict[str, Any]] = None,
    track_frame_min: Optional[float] = None,
    track_frame_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    cfg = cfg or {}
    frame_values = [_finite(item.get("peak_frame"), np.nan) for item in candidates]
    finite_frames = [value for value in frame_values if np.isfinite(value)]
    frame_min = _finite(track_frame_min, min(finite_frames) if finite_frames else 0.0)
    frame_max = _finite(track_frame_max, max(finite_frames) if finite_frames else frame_min)
    track_span = max(0.0, frame_max - frame_min)
    edge_margin = max(
        0.0,
        float(cfg.get("edge_margin_frames", 0.0)),
        track_span * max(0.0, float(cfg.get("edge_margin_fraction", 0.0))),
    )
    edge_floor = float(np.clip(float(cfg.get("edge_score_floor", 0.2)), 0.0, 1.0))
    min_separation = max(0.0, float(cfg.get("min_peak_separation_frames", 0.0)))
    width_separation_fraction = max(
        0.0,
        float(cfg.get("peak_separation_width_fraction", 0.0)),
    )
    overlap_limit = float(np.clip(
        float(cfg.get("duplicate_window_overlap_fraction", 0.8)),
        0.0,
        1.0,
    ))
    support_overlap_limit = float(np.clip(
        float(cfg.get("duplicate_support_overlap_fraction", 0.8)),
        0.0,
        1.0,
    ))

    def base_score(item: Dict[str, Any]) -> float:
        width = max(1.0, _finite(item.get("fit_window_width_frames"), 1.0))
        error = max(0.0, _finite(item.get("fit_error_vnmse"), 1.0))
        return width * max(1.0, _finite(item.get("amplitude_px"), 1.0)) / (1.0 + error)

    def selection_score(item: Dict[str, Any]) -> float:
        peak_frame = _finite(item.get("peak_frame"), frame_min)
        edge_distance = max(0.0, min(peak_frame - frame_min, frame_max - peak_frame))
        edge_weight = (
            edge_floor + (1.0 - edge_floor) * min(1.0, edge_distance / edge_margin)
            if edge_margin > 0
            else 1.0
        )
        score = base_score(item) * edge_weight
        item["peak_selection_edge_distance_frames"] = edge_distance
        item["peak_selection_edge_weight"] = edge_weight
        item["peak_selection_score"] = score
        return score

    def conflicts(candidate: Dict[str, Any], existing: Dict[str, Any]) -> bool:
        candidate_frame = _finite(candidate.get("peak_frame"), np.nan)
        existing_frame = _finite(existing.get("peak_frame"), np.nan)
        candidate_width = max(0.0, _finite(candidate.get("fit_window_width_frames"), 0.0))
        existing_width = max(0.0, _finite(existing.get("fit_window_width_frames"), 0.0))
        adaptive_separation = width_separation_fraction * min(candidate_width, existing_width)
        required_separation = max(min_separation, adaptive_separation)
        if (
            np.isfinite(candidate_frame)
            and np.isfinite(existing_frame)
            and abs(candidate_frame - existing_frame) <= required_separation
        ):
            return True
        if _window_overlap_fraction(candidate, existing) >= overlap_limit:
            return True
        return _window_overlap_fraction(
            candidate,
            existing,
            lo_key="fit_support_window_lo",
            hi_key="fit_support_window_hi",
        ) >= support_overlap_limit

    kept: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=selection_score, reverse=True):
        if not any(conflicts(candidate, existing) for existing in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda item: float(item["peak_frame"]))


def _window_overlap_fraction(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    lo_key: str = "fit_window_lo",
    hi_key: str = "fit_window_hi",
) -> float:
    a_lo = int(a.get(lo_key, a.get("fit_window_lo", 0)))
    a_hi = int(a.get(hi_key, a.get("fit_window_hi", a_lo)))
    b_lo = int(b.get(lo_key, b.get("fit_window_lo", 0)))
    b_hi = int(b.get(hi_key, b.get("fit_window_hi", b_lo)))
    overlap = max(0, min(a_hi, b_hi) - max(a_lo, b_lo))
    smaller = max(1, min(a_hi - a_lo, b_hi - b_lo))
    return float(overlap) / float(smaller)


def _assign_recurrence_periods(
    measurements: List[Dict[str, Any]], *, sampling_rate: float
) -> None:
    by_track_kind: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for measurement in measurements:
        key = (int(measurement["track_index"]), str(measurement["event_kind"]))
        by_track_kind.setdefault(key, []).append(measurement)
    for rows in by_track_kind.values():
        rows.sort(key=lambda item: float(item["peak_frame"]))
        for index, row in enumerate(rows):
            gaps: List[float] = []
            if index > 0:
                gaps.append(float(row["peak_frame"]) - float(rows[index - 1]["peak_frame"]))
            if index + 1 < len(rows):
                gaps.append(float(rows[index + 1]["peak_frame"]) - float(row["peak_frame"]))
            positive = [gap for gap in gaps if gap > 0]
            recurrence_period_frames = float(np.median(positive)) if positive else None
            recurrence_period_s = (
                recurrence_period_frames / sampling_rate
                if recurrence_period_frames is not None and sampling_rate > 0
                else None
            )
            row["recurrence_period_frames"] = recurrence_period_frames
            row["recurrence_period_s"] = recurrence_period_s
            row["recurrence_frequency_hz"] = (
                1.0 / recurrence_period_s
                if recurrence_period_s is not None and recurrence_period_s > 0
                else None
            )


def _apply_measurement_to_wave_row(row: Dict[str, Any], measurement: Dict[str, Any]) -> None:
    source = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    metrics = dict(source)
    map_frame = _coordinate_mapper(
        source,
        (("peak_frame_raw", "peak_frame_y_axis"), ("frame1_raw", "frame1"), ("frame2_raw", "frame2")),
    )
    map_position = _coordinate_mapper(
        source,
        (("peak_position_raw", "peak_position_x_axis"),),
    )

    frame1_raw = _optional_float(measurement.get("equivalent_cycle_frame1"))
    frame2_raw = _optional_float(measurement.get("equivalent_cycle_frame2"))
    peak_frame_raw = float(measurement["peak_frame"])
    cycle_position1 = _optional_float(measurement.get("equivalent_cycle_position1_px"))
    cycle_position2 = _optional_float(measurement.get("equivalent_cycle_position2_px"))
    pos1 = map_position(cycle_position1) if cycle_position1 is not None else None
    pos2 = map_position(cycle_position2) if cycle_position2 is not None else None
    peak_position = map_position(float(measurement["peak_position_px"]))
    frame1 = map_frame(frame1_raw) if frame1_raw is not None else None
    frame2 = map_frame(frame2_raw) if frame2_raw is not None else None
    peak_frame = map_frame(peak_frame_raw)

    measurement.update({
        "frame1": frame1,
        "frame2": frame2,
        "frame1_raw": frame1_raw,
        "frame2_raw": frame2_raw,
        "pos1_px": pos1,
        "pos2_px": pos2,
        "peak_frame_y_axis": peak_frame,
        "peak_position_x_axis": peak_position,
    })

    metrics.update({key: value for key, value in measurement.items() if not key.startswith("_")})
    metrics.update({
        "analysis_mode": "large_wave",
        "large_wave_event_id": measurement.get("event_id"),
        "large_wave_direction": measurement.get("direction"),
        "large_wave_amplitude_px": measurement.get("amplitude_px"),
        "large_wave_width_frames": measurement.get("width_half_prominence_frames"),
        "large_wave_width_s": measurement.get("width_half_prominence_s"),
        "large_wave_rise_time_s": measurement.get("rise_time_s"),
        "large_wave_recovery_time_s": measurement.get("recovery_time_s"),
        "large_wave_max_speed_px_per_s": measurement.get("max_speed_px_per_s"),
        "large_wave_apex_curvature_px_per_frame2": measurement.get(
            "apex_curvature_px_per_frame2"
        ),
        "frame1_raw": frame1_raw,
        "frame2_raw": frame2_raw,
        "frame1": frame1,
        "frame2": frame2,
        "pos1_px": pos1,
        "pos2_px": pos2,
        "peak_frame_raw": peak_frame_raw,
        "peak_position_raw": float(measurement["peak_position_px"]),
        "peak_frame_y_axis": peak_frame,
        "peak_position_x_axis": peak_position,
        "amplitude_px": measurement.get("amplitude_px"),
        "event_value": measurement.get("amplitude_px"),
        "peak_value_original": measurement.get("signed_amplitude_px"),
        "fit_target": "large_wave_local_chord",
        "compare_fit_targets": False,
        "wave_type": "large_wave",
    })
    row["metrics"] = metrics
    row["event_kind"] = measurement.get("event_kind")
    row["event_polarity"] = "minima" if measurement.get("event_kind") == "min" else "maxima"
    row["fit_target"] = "large_wave_local_chord"
    row["amplitude"] = measurement.get("amplitude_px")
    row["frequency"] = measurement.get("frequency_hz")
    row["period"] = measurement.get("period_s")
    row["error"] = measurement.get("fit_error_vnmse")
    row["t_start"] = measurement.get("frame1_time_s")
    row["t_end"] = measurement.get("frame2_time_s")
    row["x"] = int(round(peak_position)) if np.isfinite(peak_position) else None
    row["y"] = int(round(peak_frame)) if np.isfinite(peak_frame) else None


def _coordinate_mapper(
    metrics: Dict[str, Any],
    pairs: Sequence[Tuple[str, str]],
) -> Callable[[float], float]:
    raw_values: List[float] = []
    mapped_values: List[float] = []
    for raw_key, mapped_key in pairs:
        raw = _optional_float(metrics.get(raw_key))
        mapped = _optional_float(metrics.get(mapped_key))
        if raw is not None and mapped is not None:
            raw_values.append(raw)
            mapped_values.append(mapped)
    if len(raw_values) >= 2 and float(np.ptp(raw_values)) > 1e-9:
        slope, intercept = np.polyfit(np.asarray(raw_values), np.asarray(mapped_values), 1)
        return lambda value: float(slope * value + intercept)
    if raw_values:
        offset = mapped_values[0] - raw_values[0]
        return lambda value: float(value + offset)
    return lambda value: float(value)


def _group_measurements(
    measurements: List[Dict[str, Any]],
    *,
    sampling_rate: float,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    max_gap = float(cfg.get("max_peak_frame_gap", 20.0))
    min_tracks = max(1, int(cfg.get("min_tracks", 2)))
    clusters: List[List[Dict[str, Any]]] = []
    for measurement in sorted(measurements, key=lambda item: (item["event_kind"], item["peak_frame"])):
        matching: List[Tuple[float, List[Dict[str, Any]]]] = []
        for cluster in clusters:
            if cluster[0]["event_kind"] != measurement["event_kind"]:
                continue
            center = float(np.median([item["peak_frame"] for item in cluster]))
            distance = abs(float(measurement["peak_frame"]) - center)
            if distance <= max_gap:
                matching.append((distance, cluster))
        if matching:
            min(matching, key=lambda item: item[0])[1].append(measurement)
        else:
            clusters.append([measurement])

    accepted: List[List[Dict[str, Any]]] = []
    for cluster in clusters:
        strongest_by_track: Dict[int, Dict[str, Any]] = {}
        for measurement in cluster:
            track_index = int(measurement["track_index"])
            current = strongest_by_track.get(track_index)
            if current is None or float(measurement["amplitude_px"]) > float(current["amplitude_px"]):
                strongest_by_track[track_index] = measurement
        unique = list(strongest_by_track.values())
        if len(unique) >= min_tracks:
            accepted.append(unique)

    accepted.sort(key=lambda cluster: float(np.median([item["peak_frame"] for item in cluster])))
    events: List[Dict[str, Any]] = []
    for index, cluster in enumerate(accepted, start=1):
        event_id = f"LW{index:03d}"
        for measurement in cluster:
            measurement["event_id"] = event_id
            measurement["grouped_event"] = True
        events.append(_event_summary(event_id, cluster, sampling_rate))

    previous_frame_by_kind: Dict[str, float] = {}
    for event in events:
        event_kind = str(event["event_kind"])
        previous_frame = previous_frame_by_kind.get(event_kind)
        period_frames = None if previous_frame is None else float(event["center_frame"]) - previous_frame
        period_s = period_frames / sampling_rate if period_frames is not None and sampling_rate > 0 else None
        event["period_from_previous_frames"] = period_frames
        event["period_from_previous_s"] = period_s
        event["frequency_hz"] = 1.0 / period_s if period_s is not None and period_s > 0 else None
        previous_frame_by_kind[event_kind] = float(event["center_frame"])
    return events


def _event_summary(event_id: str, measurements: List[Dict[str, Any]], sampling_rate: float) -> Dict[str, Any]:
    frames = _values(measurements, "peak_frame")
    amplitudes = _values(measurements, "amplitude_px")
    positions = _values(measurements, "peak_position_px")
    return {
        "event_id": event_id,
        "direction": measurements[0]["direction"],
        "event_kind": measurements[0]["event_kind"],
        "track_count": len(measurements),
        "track_indices": ";".join(str(int(item["track_index"])) for item in sorted(measurements, key=lambda x: x["track_index"])),
        "center_frame": _median(frames),
        "center_time_s": _median(frames) / sampling_rate if sampling_rate > 0 else None,
        "median_amplitude_px": _median(amplitudes),
        "max_amplitude_px": float(np.max(amplitudes)),
        "amplitude_iqr_px": float(np.percentile(amplitudes, 75) - np.percentile(amplitudes, 25)),
        "median_prominence_px": _median(_values(measurements, "prominence_px")),
        "median_width_frames": _median(_values(measurements, "width_half_prominence_frames")),
        "median_width_s": _median(_values(measurements, "width_half_prominence_s")),
        "median_rise_time_s": _median(_values(measurements, "rise_time_s")),
        "median_recovery_time_s": _median(_values(measurements, "recovery_time_s")),
        "max_speed_px_per_s": float(np.max(_values(measurements, "max_speed_px_per_s"))),
        "median_apex_curvature_px_per_frame2": _median(
            _values(measurements, "apex_curvature_px_per_frame2")
        ),
        "peak_frame_mad": _mad(frames),
        "peak_frame_span": float(np.max(frames) - np.min(frames)),
        "spatial_coverage_px": float(np.max(positions) - np.min(positions)),
        "period_from_previous_frames": None,
        "period_from_previous_s": None,
        "frequency_hz": None,
    }


def _summarize_tracks(
    measurements: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    track_paths: Sequence[Path],
) -> Dict[int, Dict[str, Any]]:
    frequency_by_event = {str(event["event_id"]): event.get("frequency_hz") for event in events}
    by_track: Dict[int, List[Dict[str, Any]]] = {}
    for measurement in measurements:
        by_track.setdefault(int(measurement["track_index"]), []).append(measurement)
    summaries: Dict[int, Dict[str, Any]] = {}
    for track_index, rows in by_track.items():
        event_ids = sorted({str(row["event_id"]) for row in rows if row.get("event_id")})
        equivalent_frequencies = _values(rows, "frequency_hz")
        local_recurrence_frequencies = _values(rows, "recurrence_frequency_hz")
        grouped_frequencies = [
            frequency_by_event[event_id] for event_id in event_ids if frequency_by_event.get(event_id)
        ]
        frequency = _median(equivalent_frequencies)
        recurrence_frequency = (
            _median(local_recurrence_frequencies)
            if local_recurrence_frequencies.size
            else _median(np.asarray(grouped_frequencies, dtype=float)) if grouped_frequencies else None
        )
        periods = _values(rows, "period_s")
        fit_errors = _values(rows, "fit_error_vnmse")
        fit_r2 = _values(rows, "fit_r2")
        period_consistency = (
            float(np.std(periods) / np.mean(periods))
            if periods.size >= 2 and float(np.mean(periods)) > 0
            else None
        )
        frequency_agreement = None
        if equivalent_frequencies.size >= 2 and frequency is not None and frequency > 0:
            frequency_agreement = float(
                np.median(np.abs(equivalent_frequencies - frequency)) / frequency
            )
        summaries[track_index] = {
            "point_count": _point_count(track_paths, track_index),
            "large_wave_measurement_count": len(rows),
            "large_wave_event_count": len(event_ids),
            "large_wave_event_ids": event_ids,
            "mean_large_wave_amplitude_px": _median(_values(rows, "amplitude_px")),
            "max_large_wave_amplitude_px": float(np.max(_values(rows, "amplitude_px"))),
            "mean_peak_width_frames": _median(_values(rows, "width_half_prominence_frames")),
            "mean_peak_width_s": _median(_values(rows, "width_half_prominence_s")),
            "max_speed_px_per_s": float(np.max(_values(rows, "max_speed_px_per_s"))),
            "mean_apex_curvature_px_per_frame2": _median(
                _values(rows, "apex_curvature_px_per_frame2")
            ),
            "large_wave_frequency_hz": frequency,
            "large_wave_recurrence_frequency_hz": recurrence_frequency,
            "track_fit_error_median": _median(fit_errors),
            "track_fit_r2_median": _median(fit_r2),
            "period_consistency_cv": period_consistency,
            "frequency_agreement_error": frequency_agreement,
        }
    return summaries


def _empty_track_summary(
    track_index: int,
    track_paths: Sequence[Path],
) -> Dict[str, Any]:
    return {
        "point_count": _point_count(track_paths, track_index),
        "large_wave_measurement_count": 0,
        "large_wave_event_count": 0,
        "large_wave_event_ids": [],
        "mean_large_wave_amplitude_px": None,
        "max_large_wave_amplitude_px": None,
        "mean_peak_width_frames": None,
        "mean_peak_width_s": None,
        "max_speed_px_per_s": None,
        "mean_apex_curvature_px_per_frame2": None,
        "large_wave_frequency_hz": None,
        "large_wave_recurrence_frequency_hz": None,
        "track_fit_error_median": None,
        "track_fit_r2_median": None,
        "period_consistency_cv": None,
        "frequency_agreement_error": None,
    }


def _track_order(config: Dict[str, Any]) -> str:
    kymo_cfg = config.get("kymo") or {}
    order = str(kymo_cfg.get("track_xy_order", "auto")).lower()
    if order == "auto":
        order = "yx" if str(kymo_cfg.get("backend", "onnx")).lower() == "onnx" else "xy"
    return order


def _point_count(track_paths: Sequence[Path], track_index: int) -> int:
    if track_index < 0 or track_index >= len(track_paths):
        return 0
    try:
        return int(np.load(track_paths[track_index], mmap_mode="r").shape[0])
    except Exception:
        return 0


def _track_csv_row(track_row: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    track_index = int(track_row.get("track_index", -1))
    event_ids = summary.get("large_wave_event_ids") or []
    raw = {"track_index": track_index, **summary, "large_wave_event_ids": ";".join(event_ids)}
    return {
        "Track ID": track_index,
        "Points": summary.get("point_count"),
        "Broad Peaks": summary.get("large_wave_measurement_count"),
        "Grouped Large Waves": summary.get("large_wave_event_count"),
        "Large Wave IDs": ";".join(event_ids),
        "Median Amplitude (px)": summary.get("mean_large_wave_amplitude_px"),
        "Maximum Amplitude (px)": summary.get("max_large_wave_amplitude_px"),
        "Median Peak Width (frames)": summary.get("mean_peak_width_frames"),
        "Median Peak Width (seconds)": summary.get("mean_peak_width_s"),
        "Maximum Speed (pixels/sec)": summary.get("max_speed_px_per_s"),
        "Median Apex Curvature (px/frame^2)": summary.get("mean_apex_curvature_px_per_frame2"),
        "Median Frequency (Hz)": summary.get("large_wave_frequency_hz"),
        "Median Recurrence Frequency (Hz)": summary.get(
            "large_wave_recurrence_frequency_hz"
        ),
        **raw,
    }


def _measurement_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    standard = {
        "wave_id": row.get("measurement_id"),
        "track_id": row.get("track_index"),
        "wave_index": row.get("wave_index"),
        "Frame position 1 (y-axis)": row.get("frame1"),
        "Frame position 2 (y-axis)": row.get("frame2"),
        "Period In Frames (Frame 1- Frame 2)": row.get("period_frames"),
        "Period in Seconds": row.get("period_s"),
        "Frequency (Hertz)": row.get("frequency_hz"),
        "Period Source": row.get("period_source"),
        "Pixel Position 1 (x-axis)": row.get("pos1_px"),
        "Pixel Position 2 (x-axis)": row.get("pos2_px"),
        "Amplitude (Pixels)": row.get("amplitude_px"),
        "Signed Amplitude (Pixels)": row.get("signed_amplitude_px"),
        "Position 1 (x-axis)": row.get("pos1_px"),
        "Position 2 (x-axis)": row.get("pos2_px"),
        "Frame 1 (y-axis)": row.get("frame1"),
        "Frame 2 (y-axis)": row.get("frame2"),
        "Frame 1 (seconds)": row.get("frame1_time_s"),
        "Frame 2 (seconds)": row.get("frame2_time_s"),
        "Seconds 2 - Seconds 1": row.get("period_s"),
        "Position2 -Position 1": row.get("delta_pos_px"),
        "Velocity (pixels/sec)": row.get("velocity_px_per_s"),
        "Frequency (Hz)": row.get("frequency_hz"),
        "Wavelength (Pixels)": row.get("wavelength_px"),
        "Peak Frame (y-axis)": row.get("peak_frame_y_axis", row.get("peak_frame")),
        "Peak Position (x-axis)": row.get("peak_position_x_axis", row.get("peak_position_px")),
        "Event Kind": row.get("event_kind"),
        "Event Polarity": "minima" if row.get("event_kind") == "min" else "maxima",
        "Event Value": row.get("amplitude_px"),
        "Peak Value Original": row.get("signed_amplitude_px"),
        "Fit Target": "large_wave_local_chord",
        "Compare Fit Targets": False,
        "Peak Frame Raw": row.get("peak_frame"),
        "Peak Position Raw": row.get("peak_position_px"),
        "Frame 1 Raw": row.get("frame1_raw"),
        "Frame 2 Raw": row.get("frame2_raw"),
        "Fit Error (VNMSE)": row.get("fit_error_vnmse"),
        "Fit Passes Peak": row.get("fit_passes_peak"),
        "Fit R2": row.get("fit_r2"),
        "Fit RMSE (px)": row.get("fit_rmse_px"),
        "Fit NRMSE": row.get("fit_nrmse"),
        "Fit MAE (px)": row.get("fit_mae_px"),
        "Fit Points": row.get("fit_points"),
        "Residual Fit Error (VNMSE)": row.get("fit_error_vnmse"),
        "Residual Fit R2": row.get("fit_r2"),
        "Residual Fit RMSE (px)": row.get("fit_rmse_px"),
        "Track Fit Error Median": row.get("track_fit_error_median"),
        "Track Fit R2 Median": row.get("track_fit_r2_median"),
        "Period Consistency CV": row.get("period_consistency_cv"),
        "Frequency Agreement Error": row.get("frequency_agreement_error"),
        "Config Event Polarity": "both",
        "Wave Type": "large_wave",
    }
    friendly = {
        "Measurement ID": row.get("measurement_id"),
        "Large Wave ID": row.get("event_id") or "Unassigned",
        "Track ID": row.get("track_index"),
        "Direction": row.get("direction"),
        "Peak Frame": row.get("peak_frame"),
        "Peak Time (seconds)": row.get("peak_time_s"),
        "Peak Position (px)": row.get("peak_position_px"),
        "Signed Amplitude (px)": row.get("signed_amplitude_px"),
        "Amplitude (px)": row.get("amplitude_px"),
        "Peak Prominence (px)": row.get("prominence_px"),
        "Width at Half Prominence (frames)": row.get("width_half_prominence_frames"),
        "Width at Half Prominence (seconds)": row.get("width_half_prominence_s"),
        "Rise Time (seconds)": row.get("rise_time_s"),
        "Recovery Time (seconds)": row.get("recovery_time_s"),
        "Maximum Approach Speed (pixels/sec)": row.get("max_approach_speed_px_per_s"),
        "Maximum Recovery Speed (pixels/sec)": row.get("max_recovery_speed_px_per_s"),
        "Maximum Speed (pixels/sec)": row.get("max_speed_px_per_s"),
        "Apex Curvature (px/frame^2)": row.get("apex_curvature_px_per_frame2"),
        "Integrated Displacement (pixel-seconds)": row.get("integrated_displacement_px_s"),
        "Fit Start Frame Raw": row.get("fit_start_frame"),
        "Fit End Frame Raw": row.get("fit_end_frame"),
        "Fit Duration (frames)": row.get("fit_duration_frames"),
        "Fit Duration (seconds)": row.get("fit_duration_s"),
        "Period Source": row.get("period_source"),
        "Left Quarter Period (frames)": row.get("left_quarter_period_frames"),
        "Right Quarter Period (frames)": row.get("right_quarter_period_frames"),
        "Period from Left Side (frames)": row.get("period_from_left_frames"),
        "Period from Right Side (frames)": row.get("period_from_right_frames"),
        "Period Asymmetry": row.get("period_asymmetry"),
        "Period Boundary Error (fraction)": row.get("period_boundary_error_fraction"),
        "Period Estimate Valid": row.get("period_estimate_valid"),
        "Recurrence Period (frames)": row.get("recurrence_period_frames"),
        "Recurrence Period (seconds)": row.get("recurrence_period_s"),
        "Recurrence Frequency (Hz)": row.get("recurrence_frequency_hz"),
        "Baseline Start Position (px)": row.get("baseline_start_position_px"),
        "Baseline End Position (px)": row.get("baseline_end_position_px"),
        "Baseline Slope (px/frame)": row.get("baseline_slope_px_per_frame"),
        "Baseline Velocity (pixels/sec)": row.get("baseline_velocity_px_per_s"),
        "Left Shape Power": row.get("fit_left_shape_power"),
        "Right Shape Power": row.get("fit_right_shape_power"),
        "Fit Window Source": row.get("fit_window_source"),
        "Boundary Extrapolated": row.get("boundary_extrapolated"),
        "Fit Boundary Extrapolated": row.get("fit_boundary_extrapolated"),
        "Equivalent Cycle Boundary Extrapolated": row.get(
            "equivalent_cycle_boundary_extrapolated"
        ),
        "Grouped Event": row.get("grouped_event"),
    }
    return {**standard, **friendly, **row}


def _event_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Large Wave ID": row.get("event_id"),
        "Direction": row.get("direction"),
        "Tracks": row.get("track_count"),
        "Track IDs": row.get("track_indices"),
        "Center Frame": row.get("center_frame"),
        "Center Time (seconds)": row.get("center_time_s"),
        "Median Amplitude (px)": row.get("median_amplitude_px"),
        "Maximum Amplitude (px)": row.get("max_amplitude_px"),
        "Amplitude IQR (px)": row.get("amplitude_iqr_px"),
        "Median Prominence (px)": row.get("median_prominence_px"),
        "Median Width (frames)": row.get("median_width_frames"),
        "Median Width (seconds)": row.get("median_width_s"),
        "Median Rise Time (seconds)": row.get("median_rise_time_s"),
        "Median Recovery Time (seconds)": row.get("median_recovery_time_s"),
        "Maximum Speed (pixels/sec)": row.get("max_speed_px_per_s"),
        "Median Apex Curvature (px/frame^2)": row.get("median_apex_curvature_px_per_frame2"),
        "Peak Frame MAD": row.get("peak_frame_mad"),
        "Peak Frame Span": row.get("peak_frame_span"),
        "Spatial Coverage (px)": row.get("spatial_coverage_px"),
        "Period from Previous Event (frames)": row.get("period_from_previous_frames"),
        "Period from Previous Event (seconds)": row.get("period_from_previous_s"),
        "Frequency (Hz)": row.get("frequency_hz"),
        **row,
    }


def _csv_bytes(rows: List[Dict[str, Any]], fields: List[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return buf.getvalue().encode("utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}" if np.isfinite(float(value)) else ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return "" if value is None else value


def _values(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    values = [_finite(row.get(key), np.nan) for row in rows]
    return np.asarray([value for value in values if np.isfinite(value)], dtype=float)


def _median(values: np.ndarray) -> Optional[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _mad(values: np.ndarray) -> Optional[float]:
    median = _median(values)
    if median is None:
        return None
    return float(np.median(np.abs(np.asarray(values, dtype=float) - median)))


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if np.isfinite(number) else default


def _optional_float(value: Any) -> Optional[float]:
    number = _finite(value, np.nan)
    return float(number) if np.isfinite(number) else None


def _check_cancel(cancel_cb: CancelCallback) -> None:
    if cancel_cb is not None and cancel_cb():
        raise CancellationRequested("cancel_requested")
