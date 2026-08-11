from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

import networkx as nx
import numpy as np

from .cancel import CancellationRequested
from .extract_core import load_track_frame_position


CancelCallback = Optional[Callable[[], bool]]
ProgressCallback = Optional[Callable[[str, int, int], None]]


@dataclass
class _TrackGeometry:
    track_index: int
    track_path: Path
    frame: np.ndarray
    position: np.ndarray
    slope: float
    intercept: float
    line_rmse: float
    line_r2: Optional[float]
    direction: str
    angle_deg: float
    y_start: float
    y_end: float
    x_min: float
    x_max: float
    duration_frames: float
    spatial_span_px: float
    eligible: bool
    family_id: Optional[str] = None


@dataclass(frozen=True)
class RippleAnalysisResult:
    track_rows: List[Dict[str, Any]]
    overlay_events: List[Dict[str, Any]]
    intervals: List[Dict[str, Any]]
    families: List[Dict[str, Any]]
    tracks_csv: bytes
    intervals_csv: bytes
    families_csv: bytes


RIPPLE_TRACK_FIELDS = [
    "Track ID",
    "Family",
    "Direction",
    "Points",
    "Duration (frames)",
    "Duration (seconds)",
    "Start Frame",
    "End Frame",
    "Start X (px)",
    "End X (px)",
    "Spatial Span (px)",
    "Slope (px/frame)",
    "Velocity (pixels/sec)",
    "Speed (pixels/sec)",
    "Angle from Time Axis (degrees)",
    "Line Fit Error (RMSE px)",
    "Line Fit R2",
    "Neighbor Intervals",
    "Ripple Period (frames)",
    "Ripple Period (seconds)",
    "Ripple Frequency (Hz)",
    "Frequency Method",
    "Eligible for Family",
    "track_index",
    "family_id",
    "family_label",
    "direction",
    "point_count",
    "slope_px_per_frame",
    "velocity_px_per_s",
    "speed_px_per_s",
    "angle_deg",
    "angle_from_time_axis_deg",
    "line_intercept_px",
    "line_rmse_px",
    "line_fit_rmse_px",
    "line_r2",
    "duration_frames",
    "duration_s",
    "spatial_span_px",
    "x_start_px",
    "x_end_px",
    "y_start_frame",
    "y_end_frame",
    "neighbor_interval_count",
    "period_frames",
    "period_s",
    "frequency_hz",
    "frequency_method",
    "eligible",
]

RIPPLE_INTERVAL_FIELDS = [
    "Ripple Wave ID",
    "Family",
    "Direction",
    "Track 1",
    "Track 2",
    "Frame Gap / Period (frames)",
    "Period in Seconds",
    "Frequency (Hertz)",
    "Slope (px/frame)",
    "Velocity (pixels/sec)",
    "Speed (pixels/sec)",
    "Angle from Time Axis (degrees)",
    "X Overlap Start (px)",
    "X Overlap End (px)",
    "Samples Used",
    "Gap MAD (frames)",
    "Gap CV",
    "Measurement Method",
    "interval_index",
    "family_id",
    "family_label",
    "direction",
    "earlier_track_index",
    "later_track_index",
    "x_overlap_start_px",
    "x_overlap_end_px",
    "sample_count",
    "slope_px_per_frame",
    "velocity_px_per_s",
    "speed_px_per_s",
    "angle_deg",
    "angle_from_time_axis_deg",
    "period_frames",
    "period_s",
    "frequency_hz",
    "gap_mad_frames",
    "gap_cv",
    "measurement_method",
]

RIPPLE_FAMILY_FIELDS = [
    "Family",
    "Direction",
    "Tracks",
    "Intervals",
    "Track IDs",
    "Median Slope (px/frame)",
    "Median Velocity (pixels/sec)",
    "Median Speed (pixels/sec)",
    "Median Angle from Time Axis (degrees)",
    "Median Ripple Period (frames)",
    "Median Ripple Period (seconds)",
    "Median Ripple Frequency (Hz)",
    "Frequency IQR (Hz)",
    "X Min (px)",
    "X Max (px)",
    "Start Frame",
    "End Frame",
    "Frequency Method",
    "family_id",
    "family_label",
    "direction",
    "track_count",
    "interval_count",
    "track_indices",
    "median_slope_px_per_frame",
    "median_velocity_px_per_s",
    "median_speed_px_per_s",
    "median_angle_deg",
    "median_angle_from_time_axis_deg",
    "median_period_frames",
    "median_period_s",
    "median_frequency_hz",
    "frequency_iqr_hz",
    "x_min_px",
    "x_max_px",
    "y_min_frame",
    "y_max_frame",
    "frequency_method",
]


def analyze_ripple_tracks(
    *,
    job_id: UUID,
    track_paths: Sequence[Path],
    config: Dict[str, Any],
    cancel_cb: CancelCallback = None,
    progress_cb: ProgressCallback = None,
) -> RippleAnalysisResult:
    _check_cancel(cancel_cb)
    ripple_cfg = (((config.get("analysis") or {}).get("ripple") or {}))
    family_cfg = (ripple_cfg.get("family") or {})
    frequency_cfg = (ripple_cfg.get("frequency") or {})
    sampling_rate = float((config.get("io") or {}).get("sampling_rate", 1.0))
    min_track_rows = int(ripple_cfg.get("min_track_rows", 30))
    min_abs_slope = float(ripple_cfg.get("min_abs_slope", 0.05))
    max_line_rmse = float(ripple_cfg.get("max_line_rmse_px", 12.0))
    track_order = _track_order(config)

    geometries: List[_TrackGeometry] = []
    total = len(track_paths)
    for track_index, track_path in enumerate(track_paths):
        _check_cancel(cancel_cb)
        frame, position = load_track_frame_position(track_path, order=track_order)
        geometry = _fit_track_geometry(
            track_index=track_index,
            track_path=track_path,
            frame=frame,
            position=position,
            min_track_rows=min_track_rows,
            min_abs_slope=min_abs_slope,
            max_line_rmse=max_line_rmse,
        )
        geometries.append(geometry)
        if progress_cb is not None:
            progress_cb("ripple_track_geometry", track_index + 1, total)

    family_components = _group_track_families(
        geometries,
        family_cfg,
        cancel_cb=cancel_cb,
        progress_cb=progress_cb,
    )
    family_groups: List[Tuple[str, List[_TrackGeometry]]] = []
    families: List[Dict[str, Any]] = []
    intervals: List[Dict[str, Any]] = []
    intervals_by_track: Dict[int, List[Dict[str, Any]]] = {g.track_index: [] for g in geometries}

    family_number = 0
    for component in family_components:
        _check_cancel(cancel_cb)
        if len(component) < int(family_cfg.get("min_tracks", 2)):
            continue
        family_number += 1
        family_id = f"RF{family_number:03d}"
        members = [geometries[index] for index in component]
        for member in members:
            member.family_id = family_id
        family_groups.append((family_id, members))

    if not family_groups and progress_cb is not None:
        progress_cb("ripple_interval_analysis", 0, 0)
    for family_index, (family_id, members) in enumerate(family_groups):
        _check_cancel(cancel_cb)
        family_intervals = _measure_family_intervals(
            members,
            family_id=family_id,
            sampling_rate=sampling_rate,
            cfg=frequency_cfg,
            interval_start=len(intervals) + 1,
            cancel_cb=cancel_cb,
        )
        intervals.extend(family_intervals)
        for interval in family_intervals:
            intervals_by_track[int(interval["earlier_track_index"])].append(interval)
            intervals_by_track[int(interval["later_track_index"])].append(interval)
        families.append(_family_row(family_id, members, family_intervals, sampling_rate))
        if progress_cb is not None:
            progress_cb("ripple_interval_analysis", family_index + 1, len(family_groups))

    track_rows: List[Dict[str, Any]] = []
    overlay_events: List[Dict[str, Any]] = []
    track_csv_rows: List[Dict[str, Any]] = []
    max_overlay_points = int((config.get("overlay") or {}).get("max_points", 300))
    for geometry in geometries:
        _check_cancel(cancel_cb)
        track_intervals = intervals_by_track[geometry.track_index]
        period_frames = _median_or_none([row["period_frames"] for row in track_intervals])
        period_s = (period_frames / sampling_rate) if period_frames is not None and sampling_rate > 0 else None
        frequency_hz = (sampling_rate / period_frames) if period_frames is not None and period_frames > 0 else None
        frequency_method = "median_neighbor_gap" if track_intervals else None
        metrics = _track_metrics(
            geometry,
            sampling_rate=sampling_rate,
            interval_count=len(track_intervals),
            period_frames=period_frames,
            period_s=period_s,
            frequency_hz=frequency_hz,
            frequency_method=frequency_method,
        )
        track_rows.append({
            "track_index": geometry.track_index,
            "amplitude": None,
            "frequency": frequency_hz,
            "error": geometry.line_rmse,
            "x0": int(round(geometry.position[0])) if geometry.position.size else None,
            "y0": int(round(geometry.frame[0])) if geometry.frame.size else None,
            "metrics": metrics,
            "overlay": {},
        })
        overlay_events.append(_overlay_event(
            job_id=job_id,
            geometry=geometry,
            metrics=metrics,
            max_points=max_overlay_points,
        ))
        track_csv_rows.append({
            "Track ID": geometry.track_index,
            "Family": _family_label(geometry.family_id),
            "Direction": geometry.direction,
            "Points": int(geometry.frame.size),
            "Duration (frames)": geometry.duration_frames,
            "Duration (seconds)": _seconds(geometry.duration_frames, sampling_rate),
            "Start Frame": geometry.y_start,
            "End Frame": geometry.y_end,
            "Start X (px)": float(geometry.position[0]) if geometry.position.size else None,
            "End X (px)": float(geometry.position[-1]) if geometry.position.size else None,
            "Spatial Span (px)": geometry.spatial_span_px,
            "Slope (px/frame)": geometry.slope,
            "Velocity (pixels/sec)": _velocity_px_per_s(geometry.slope, sampling_rate),
            "Speed (pixels/sec)": _speed_px_per_s(geometry.slope, sampling_rate),
            "Angle from Time Axis (degrees)": geometry.angle_deg,
            "Line Fit Error (RMSE px)": geometry.line_rmse,
            "Line Fit R2": geometry.line_r2,
            "Neighbor Intervals": len(track_intervals),
            "Ripple Period (frames)": period_frames,
            "Ripple Period (seconds)": period_s,
            "Ripple Frequency (Hz)": frequency_hz,
            "Frequency Method": frequency_method,
            "Eligible for Family": geometry.eligible,
            "track_index": geometry.track_index,
            "family_id": geometry.family_id,
            "family_label": _family_label(geometry.family_id),
            "direction": geometry.direction,
            "point_count": int(geometry.frame.size),
            "slope_px_per_frame": geometry.slope,
            "velocity_px_per_s": _velocity_px_per_s(geometry.slope, sampling_rate),
            "speed_px_per_s": _speed_px_per_s(geometry.slope, sampling_rate),
            "angle_deg": geometry.angle_deg,
            "angle_from_time_axis_deg": geometry.angle_deg,
            "line_intercept_px": geometry.intercept,
            "line_rmse_px": geometry.line_rmse,
            "line_fit_rmse_px": geometry.line_rmse,
            "line_r2": geometry.line_r2,
            "duration_frames": geometry.duration_frames,
            "duration_s": _seconds(geometry.duration_frames, sampling_rate),
            "spatial_span_px": geometry.spatial_span_px,
            "x_start_px": float(geometry.position[0]) if geometry.position.size else None,
            "x_end_px": float(geometry.position[-1]) if geometry.position.size else None,
            "y_start_frame": geometry.y_start,
            "y_end_frame": geometry.y_end,
            "neighbor_interval_count": len(track_intervals),
            "period_frames": period_frames,
            "period_s": period_s,
            "frequency_hz": frequency_hz,
            "frequency_method": frequency_method,
            "eligible": geometry.eligible,
        })

    return RippleAnalysisResult(
        track_rows=track_rows,
        overlay_events=overlay_events,
        intervals=intervals,
        families=families,
        tracks_csv=_csv_bytes(track_csv_rows, RIPPLE_TRACK_FIELDS),
        intervals_csv=_csv_bytes(intervals, RIPPLE_INTERVAL_FIELDS),
        families_csv=_csv_bytes(families, RIPPLE_FAMILY_FIELDS),
    )


def _fit_track_geometry(
    *,
    track_index: int,
    track_path: Path,
    frame: np.ndarray,
    position: np.ndarray,
    min_track_rows: int,
    min_abs_slope: float,
    max_line_rmse: float,
) -> _TrackGeometry:
    frame, position = _clean_track(frame, position)
    slope, intercept, rmse, r2 = _robust_line_fit(frame, position)
    direction = "positive" if slope > min_abs_slope else "negative" if slope < -min_abs_slope else "stationary"
    y_start = float(frame[0]) if frame.size else 0.0
    y_end = float(frame[-1]) if frame.size else 0.0
    x_min = float(np.min(position)) if position.size else 0.0
    x_max = float(np.max(position)) if position.size else 0.0
    duration = max(0.0, y_end - y_start)
    eligible = bool(
        frame.size >= 2
        and duration + 1.0 >= float(min_track_rows)
        and direction != "stationary"
        and np.isfinite(rmse)
        and rmse <= max_line_rmse
    )
    return _TrackGeometry(
        track_index=track_index,
        track_path=track_path,
        frame=frame,
        position=position,
        slope=slope,
        intercept=intercept,
        line_rmse=rmse,
        line_r2=r2,
        direction=direction,
        angle_deg=float(np.degrees(np.arctan(slope))),
        y_start=y_start,
        y_end=y_end,
        x_min=x_min,
        x_max=x_max,
        duration_frames=duration,
        spatial_span_px=max(0.0, x_max - x_min),
        eligible=eligible,
    )


def _clean_track(frame: np.ndarray, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(frame, dtype=float).reshape(-1)
    x = np.asarray(position, dtype=float).reshape(-1)
    n = min(y.size, x.size)
    y = y[:n]
    x = x[:n]
    finite = np.isfinite(y) & np.isfinite(x)
    y = y[finite]
    x = x[finite]
    if y.size == 0:
        return y, x
    order = np.argsort(y, kind="stable")
    y = y[order]
    x = x[order]
    unique_y, inverse = np.unique(y, return_inverse=True)
    if unique_y.size != y.size:
        medians = np.asarray([np.median(x[inverse == i]) for i in range(unique_y.size)], dtype=float)
        return unique_y, medians
    return y, x


def _robust_line_fit(frame: np.ndarray, position: np.ndarray) -> Tuple[float, float, float, Optional[float]]:
    if frame.size < 2:
        x0 = float(position[0]) if position.size else 0.0
        return 0.0, x0, float("inf"), None
    slope, intercept = np.polyfit(frame, position, deg=1)
    residual = position - (slope * frame + intercept)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(1.0, 3.0 * robust_sigma)
    inliers = np.abs(residual - median) <= threshold
    if int(np.count_nonzero(inliers)) >= 2:
        slope, intercept = np.polyfit(frame[inliers], position[inliers], deg=1)
    fitted = slope * frame + intercept
    error = position - fitted
    rmse = float(np.sqrt(np.mean(error * error)))
    total = float(np.sum((position - float(np.mean(position))) ** 2))
    r2 = 1.0 - float(np.sum(error * error)) / total if total > 1e-12 else None
    return float(slope), float(intercept), rmse, float(r2) if r2 is not None else None


def _group_track_families(
    geometries: Sequence[_TrackGeometry],
    cfg: Dict[str, Any],
    *,
    cancel_cb: CancelCallback,
    progress_cb: ProgressCallback,
) -> List[List[int]]:
    graph = nx.Graph()
    eligible = [index for index, geometry in enumerate(geometries) if geometry.eligible]
    graph.add_nodes_from(eligible)
    max_angle_delta = float(cfg.get("max_angle_delta_deg", 10.0))
    min_x_overlap = float(cfg.get("min_x_overlap_px", 30.0))
    min_overlap_fraction = float(cfg.get("min_x_overlap_fraction", 0.15))
    max_reference_gap = float(cfg.get("max_reference_gap_frames", 300.0))

    if progress_cb is not None:
        progress_cb("ripple_family_grouping", 0, len(eligible))
    for pos, left_index in enumerate(eligible):
        left = geometries[left_index]
        for right_index in eligible[pos + 1 :]:
            _check_cancel(cancel_cb)
            right = geometries[right_index]
            if left.direction != right.direction:
                continue
            if abs(left.angle_deg - right.angle_deg) > max_angle_delta:
                continue
            overlap_start = max(left.x_min, right.x_min)
            overlap_end = min(left.x_max, right.x_max)
            overlap = overlap_end - overlap_start
            shorter_span = max(1e-9, min(left.spatial_span_px, right.spatial_span_px))
            if overlap < min_x_overlap or overlap / shorter_span < min_overlap_fraction:
                continue
            reference_x = 0.5 * (overlap_start + overlap_end)
            gap = abs(_frame_at_x(left, reference_x) - _frame_at_x(right, reference_x))
            if not np.isfinite(gap) or gap > max_reference_gap:
                continue
            graph.add_edge(left_index, right_index, gap=gap)
        if progress_cb is not None:
            progress_cb("ripple_family_grouping", pos + 1, len(eligible))

    components = [sorted(component) for component in nx.connected_components(graph)]
    components.sort(key=lambda indices: (
        geometries[indices[0]].direction,
        min(geometries[index].y_start for index in indices),
        min(indices),
    ))
    return components


def _measure_family_intervals(
    members: Sequence[_TrackGeometry],
    *,
    family_id: str,
    sampling_rate: float,
    cfg: Dict[str, Any],
    interval_start: int,
    cancel_cb: CancelCallback,
) -> List[Dict[str, Any]]:
    if len(members) < 2:
        return []
    reference_x = float(np.median([(member.x_min + member.x_max) * 0.5 for member in members]))
    ordered = sorted(members, key=lambda member: (_frame_at_x(member, reference_x), member.track_index))
    sample_count = max(3, int(cfg.get("sample_count", 25)))
    min_period = float(cfg.get("min_period_frames", 3.0))
    max_period = float(cfg.get("max_period_frames", 300.0))
    max_gap_cv = float(cfg.get("max_gap_cv", 0.75))
    out: List[Dict[str, Any]] = []

    for earlier, later in zip(ordered, ordered[1:]):
        _check_cancel(cancel_cb)
        overlap_start = max(earlier.x_min, later.x_min)
        overlap_end = min(earlier.x_max, later.x_max)
        if overlap_end <= overlap_start:
            continue
        xs = np.linspace(overlap_start, overlap_end, sample_count)
        gaps = np.abs(
            np.asarray([_frame_at_x(later, x) for x in xs])
            - np.asarray([_frame_at_x(earlier, x) for x in xs])
        )
        gaps = gaps[np.isfinite(gaps)]
        if gaps.size == 0:
            continue
        period_frames = float(np.median(gaps))
        if period_frames < min_period or period_frames > max_period:
            continue
        gap_mad = float(np.median(np.abs(gaps - period_frames)))
        gap_cv = gap_mad / period_frames if period_frames > 0 else float("inf")
        if gap_cv > max_gap_cv:
            continue
        period_s = period_frames / sampling_rate if sampling_rate > 0 else None
        frequency_hz = sampling_rate / period_frames if sampling_rate > 0 and period_frames > 0 else None
        slope = _median_or_none([earlier.slope, later.slope])
        velocity = _velocity_px_per_s(slope, sampling_rate) if slope is not None else None
        speed = abs(velocity) if velocity is not None else None
        angle = _median_or_none([earlier.angle_deg, later.angle_deg])
        measurement_method = "median_gap_between_neighbor_tracks"
        interval_index = interval_start + len(out)
        out.append({
            "Ripple Wave ID": interval_index,
            "Family": _family_label(family_id),
            "Direction": earlier.direction,
            "Track 1": earlier.track_index,
            "Track 2": later.track_index,
            "Frame Gap / Period (frames)": period_frames,
            "Period in Seconds": period_s,
            "Frequency (Hertz)": frequency_hz,
            "Slope (px/frame)": slope,
            "Velocity (pixels/sec)": velocity,
            "Speed (pixels/sec)": speed,
            "Angle from Time Axis (degrees)": angle,
            "X Overlap Start (px)": overlap_start,
            "X Overlap End (px)": overlap_end,
            "Samples Used": int(gaps.size),
            "Gap MAD (frames)": gap_mad,
            "Gap CV": gap_cv,
            "Measurement Method": measurement_method,
            "interval_index": interval_index,
            "family_id": family_id,
            "family_label": _family_label(family_id),
            "direction": earlier.direction,
            "earlier_track_index": earlier.track_index,
            "later_track_index": later.track_index,
            "x_overlap_start_px": overlap_start,
            "x_overlap_end_px": overlap_end,
            "sample_count": int(gaps.size),
            "slope_px_per_frame": slope,
            "velocity_px_per_s": velocity,
            "speed_px_per_s": speed,
            "angle_deg": angle,
            "angle_from_time_axis_deg": angle,
            "period_frames": period_frames,
            "period_s": period_s,
            "frequency_hz": frequency_hz,
            "gap_mad_frames": gap_mad,
            "gap_cv": gap_cv,
            "measurement_method": measurement_method,
        })
    return out


def _family_row(
    family_id: str,
    members: Sequence[_TrackGeometry],
    intervals: Sequence[Dict[str, Any]],
    sampling_rate: float,
) -> Dict[str, Any]:
    frequencies = [float(row["frequency_hz"]) for row in intervals if row.get("frequency_hz") is not None]
    periods = [float(row["period_frames"]) for row in intervals if row.get("period_frames") is not None]
    frequency_iqr = None
    if frequencies:
        q25, q75 = np.percentile(np.asarray(frequencies, dtype=float), [25.0, 75.0])
        frequency_iqr = float(q75 - q25)
    median_period = _median_or_none(periods)
    median_slope = _median_or_none([member.slope for member in members])
    median_velocity = _velocity_px_per_s(median_slope, sampling_rate) if median_slope is not None else None
    median_speed = abs(median_velocity) if median_velocity is not None else None
    median_angle = _median_or_none([member.angle_deg for member in members])
    frequency_method = "median_of_family_intervals" if intervals else None
    track_indices = ";".join(str(member.track_index) for member in members)
    return {
        "Family": _family_label(family_id),
        "Direction": members[0].direction,
        "Tracks": len(members),
        "Intervals": len(intervals),
        "Track IDs": track_indices,
        "Median Slope (px/frame)": median_slope,
        "Median Velocity (pixels/sec)": median_velocity,
        "Median Speed (pixels/sec)": median_speed,
        "Median Angle from Time Axis (degrees)": median_angle,
        "Median Ripple Period (frames)": median_period,
        "Median Ripple Period (seconds)": _seconds(median_period, sampling_rate),
        "Median Ripple Frequency (Hz)": _median_or_none(frequencies),
        "Frequency IQR (Hz)": frequency_iqr,
        "X Min (px)": min(member.x_min for member in members),
        "X Max (px)": max(member.x_max for member in members),
        "Start Frame": min(member.y_start for member in members),
        "End Frame": max(member.y_end for member in members),
        "Frequency Method": frequency_method,
        "family_id": family_id,
        "family_label": _family_label(family_id),
        "direction": members[0].direction,
        "track_count": len(members),
        "interval_count": len(intervals),
        "track_indices": track_indices,
        "median_slope_px_per_frame": median_slope,
        "median_velocity_px_per_s": median_velocity,
        "median_speed_px_per_s": median_speed,
        "median_angle_deg": median_angle,
        "median_angle_from_time_axis_deg": median_angle,
        "median_period_frames": median_period,
        "median_period_s": _seconds(median_period, sampling_rate),
        "median_frequency_hz": _median_or_none(frequencies),
        "frequency_iqr_hz": frequency_iqr,
        "x_min_px": min(member.x_min for member in members),
        "x_max_px": max(member.x_max for member in members),
        "y_min_frame": min(member.y_start for member in members),
        "y_max_frame": max(member.y_end for member in members),
        "frequency_method": frequency_method,
    }


def _track_metrics(
    geometry: _TrackGeometry,
    *,
    sampling_rate: float,
    interval_count: int,
    period_frames: Optional[float],
    period_s: Optional[float],
    frequency_hz: Optional[float],
    frequency_method: Optional[str],
) -> Dict[str, Any]:
    velocity = _velocity_px_per_s(geometry.slope, sampling_rate)
    speed = abs(velocity) if velocity is not None else None
    return {
        "analysis_mode": "ripple_family",
        "family_id": geometry.family_id,
        "family_label": _family_label(geometry.family_id),
        "direction": geometry.direction,
        "point_count": int(geometry.frame.size),
        "slope_px_per_frame": geometry.slope,
        "velocity_px_per_s": velocity,
        "speed_px_per_s": speed,
        "angle_deg": geometry.angle_deg,
        "angle_from_time_axis_deg": geometry.angle_deg,
        "line_intercept_px": geometry.intercept,
        "line_rmse_px": geometry.line_rmse,
        "line_fit_rmse_px": geometry.line_rmse,
        "line_r2": geometry.line_r2,
        "duration_frames": geometry.duration_frames,
        "duration_s": _seconds(geometry.duration_frames, sampling_rate),
        "spatial_span_px": geometry.spatial_span_px,
        "eligible": geometry.eligible,
        "neighbor_interval_count": interval_count,
        "period_frames": period_frames,
        "ripple_period_frames": period_frames,
        "period": period_s,
        "ripple_period_s": period_s,
        "dominant_frequency": frequency_hz,
        "frequency_hz": frequency_hz,
        "ripple_frequency_hz": frequency_hz,
        "frequency_method": frequency_method,
        "sampling_rate": sampling_rate,
    }


def _overlay_event(
    *,
    job_id: UUID,
    geometry: _TrackGeometry,
    metrics: Dict[str, Any],
    max_points: int,
) -> Dict[str, Any]:
    indices = _decimation_indices(geometry.frame.size, max_points)
    return {
        "job_id": str(job_id),
        "track_index": geometry.track_index,
        "sample": _sample_name(geometry.track_path),
        "poly": [
            {"x": float(geometry.position[index]), "y": float(geometry.frame[index])}
            for index in indices
        ],
        "peaks": [],
        "freq_hz": metrics.get("dominant_frequency"),
        "period": metrics.get("period"),
        "metrics": metrics,
    }


def _track_order(config: Dict[str, Any]) -> str:
    kymo = (config.get("kymo") or {})
    order = str(kymo.get("track_xy_order", "auto")).lower()
    if order == "auto":
        return "yx" if str(kymo.get("backend", "onnx")).lower() == "onnx" else "xy"
    return order


def _frame_at_x(track: _TrackGeometry, x: float) -> float:
    if abs(track.slope) <= 1e-12:
        return float("nan")
    return (float(x) - track.intercept) / track.slope


def _median_or_none(values: Sequence[Any]) -> Optional[float]:
    finite: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            finite.append(number)
    return float(np.median(np.asarray(finite, dtype=float))) if finite else None


def _decimation_indices(size: int, max_points: int) -> np.ndarray:
    if size <= 0:
        return np.asarray([], dtype=int)
    if max_points <= 0 or size <= max_points:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, max_points, dtype=int))


def _sample_name(track_path: Path) -> str:
    base = track_path.parent.parent.name
    return base[:-8] if base.endswith("_heatmap") else base


def _family_label(family_id: Optional[str]) -> str:
    return str(family_id) if family_id else "Unassigned"


def _velocity_px_per_s(slope_px_per_frame: Optional[float], sampling_rate: float) -> Optional[float]:
    if slope_px_per_frame is None or not np.isfinite(float(slope_px_per_frame)) or sampling_rate <= 0:
        return None
    return float(slope_px_per_frame) * float(sampling_rate)


def _speed_px_per_s(slope_px_per_frame: Optional[float], sampling_rate: float) -> Optional[float]:
    velocity = _velocity_px_per_s(slope_px_per_frame, sampling_rate)
    return abs(velocity) if velocity is not None else None


def _seconds(frame_count: Optional[float], sampling_rate: float) -> Optional[float]:
    if frame_count is None or not np.isfinite(float(frame_count)) or sampling_rate <= 0:
        return None
    return float(frame_count) / float(sampling_rate)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return ""
        return f"{number:.8g}"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _csv_bytes(rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return buffer.getvalue().encode("utf-8")


def _check_cancel(cancel_cb: CancelCallback) -> None:
    if cancel_cb is not None and cancel_cb():
        raise CancellationRequested("cancel_requested_during_ripple_analysis")
