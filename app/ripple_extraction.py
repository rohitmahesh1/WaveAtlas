from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from skimage.transform import probabilistic_hough_line

from .cancel import CancellationRequested
from .heatmap_values import load_heatmap_values, normalize_heatmap_values


CancelCallback = Optional[Callable[[], bool]]
ProgressCallback = Optional[Callable[[str, Dict[str, Any]], None]]


@dataclass
class _Trace:
    phase: str
    points: np.ndarray
    support: float
    source: str
    phase_contrast: float = 0.0


@dataclass(frozen=True)
class _OverlapStats:
    rows: float
    short_fraction: float
    distances: np.ndarray
    median: float
    p90: float


@dataclass(frozen=True)
class RippleExtractionResult:
    image_id: str
    base_dir: Path
    track_paths: List[Path]
    track_metadata: Dict[int, Dict[str, Any]]


DEFAULT_SCALES = (
    {"spatial_sigma_px": 6.0, "prominence": 0.018},
    {"spatial_sigma_px": 9.0, "prominence": 0.018},
    {"spatial_sigma_px": 13.0, "prominence": 0.015},
    {"spatial_sigma_px": 18.0, "prominence": 0.010},
)


def run_ripple_extraction(
    *,
    heatmap_path: Path,
    scratch_dir: Path,
    config: Dict[str, Any],
    heatmap_value_bytes: Optional[bytes] = None,
    heatmap_value_meta: Optional[Dict[str, Any]] = None,
    progress_cb: ProgressCallback = None,
    cancel_cb: CancelCallback = None,
) -> RippleExtractionResult:
    _check_cancel(cancel_cb)
    image_id = heatmap_path.stem
    base_dir = scratch_dir / image_id
    output_dir = base_dir / "kymobutler_output"
    debug_dir = base_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    extraction_cfg = (
        (((config.get("analysis") or {}).get("ripple") or {}).get("extraction") or {})
    )
    _progress(progress_cb, "load_values", {})
    values, value_source = load_heatmap_values(
        heatmap_path=heatmap_path,
        value_bytes=heatmap_value_bytes,
        value_meta=heatmap_value_meta or {},
    )
    bounds = extraction_cfg.get("normalize_percentiles", [1.0, 99.0])
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        bounds = [1.0, 99.0]
    normalized = normalize_heatmap_values(
        values,
        percentiles=bounds,
        context="Ripple extraction",
    )
    height, width = normalized.shape

    min_slope = float(extraction_cfg.get("min_abs_slope", 0.22))
    max_slope = float(extraction_cfg.get("max_abs_slope", 1.2))
    temporal_sigma = float(extraction_cfg.get("temporal_sigma_rows", 2.0))
    peak_distance = int(extraction_cfg.get("peak_min_distance_px", 12))
    continuation_fraction = float(extraction_cfg.get("continuation_prominence_fraction", 0.3))
    scales = _resolve_scales(extraction_cfg.get("scales"))
    theta_samples = max(24, int(extraction_cfg.get("theta_samples", 180)))
    theta_positive = np.linspace(np.arctan(min_slope), np.arctan(max_slope), theta_samples)
    theta = np.concatenate((-theta_positive[::-1], theta_positive))

    raw_traces: List[_Trace] = []
    seed_count = 0
    aggregate_candidates = np.zeros((height, width), dtype=bool)
    aggregate_seeds = np.zeros((height, width), dtype=np.uint8)
    probability_debug: Optional[np.ndarray] = None

    for scale_index, scale in enumerate(scales):
        _check_cancel(cancel_cb)
        sigma = float(scale["spatial_sigma_px"])
        prominence = float(scale["prominence"])
        _progress(
            progress_cb,
            "smoothing",
            {"scale": scale_index + 1, "scales": len(scales), "spatial_sigma_px": sigma},
        )
        smooth = gaussian_filter(normalized, sigma=(temporal_sigma, sigma))
        probability_debug = smooth
        seed_candidates = _make_candidates(
            smooth,
            prominence=prominence,
            distance=peak_distance,
            cancel_cb=cancel_cb,
        )
        trace_candidates = _make_candidates(
            smooth,
            prominence=prominence * continuation_fraction,
            distance=peak_distance,
            cancel_cb=cancel_cb,
        )
        orientation, coherence = _orientation_field(
            smooth,
            integration_sigma=float(extraction_cfg.get("orientation_window_sigma_px", 12.0)),
        )

        _progress(
            progress_cb,
            "seeding",
            {"scale": scale_index + 1, "scales": len(scales)},
        )
        masks: Dict[str, np.ndarray] = {}
        for phase in ("bright", "dark"):
            mask = np.zeros((height, width), dtype=bool)
            for y, row in enumerate(seed_candidates[phase]):
                if y % 64 == 0:
                    _check_cancel(cancel_cb)
                mask[y, row.astype(int)] = True
            masks[phase] = mask
            aggregate_candidates |= mask

        seeds = _hough_seeds(
            masks,
            theta=theta,
            cfg=extraction_cfg,
            aggregate=aggregate_seeds,
            cancel_cb=cancel_cb,
        )
        seed_count += len(seeds)
        _progress(
            progress_cb,
            "tracking",
            {
                "scale": scale_index + 1,
                "scales": len(scales),
                "seeds": len(seeds),
                "seeds_total": seed_count,
            },
        )
        for seed_index, (phase, p0, p1) in enumerate(seeds):
            if seed_index % 16 == 0:
                _check_cancel(cancel_cb)
            guided = _trace_seed_guided(
                phase,
                p0,
                p1,
                trace_candidates,
                orientation,
                coherence,
                min_slope=min_slope,
                max_slope=max_slope,
                cfg=extraction_cfg,
                cancel_cb=cancel_cb,
            )
            if guided is not None:
                raw_traces.append(guided)
            linear = _trace_seed_linear(
                phase,
                p0,
                p1,
                trace_candidates,
                orientation,
                coherence,
                min_slope=min_slope,
                max_slope=max_slope,
                cfg=extraction_cfg,
                cancel_cb=cancel_cb,
            )
            if linear is not None:
                raw_traces.append(linear)

    _check_cancel(cancel_cb)
    _progress(progress_cb, "deduping", {"candidate_traces": len(raw_traces)})
    raw_traces.sort(key=lambda trace: (len(trace.points), trace.support), reverse=True)
    traces = _dedupe_and_extend(raw_traces, extraction_cfg, cancel_cb=cancel_cb)
    if bool((extraction_cfg.get("endpoint_bridge") or {}).get("enabled", True)):
        bridge_cfg = extraction_cfg.get("endpoint_bridge") or {}
        bridge_smooth = gaussian_filter(
            normalized,
            sigma=(temporal_sigma, float(bridge_cfg.get("spatial_sigma_px", 9.0))),
        )
        bridge_orientation, bridge_coherence = _orientation_field(
            bridge_smooth,
            integration_sigma=float(extraction_cfg.get("orientation_window_sigma_px", 12.0)),
        )
        bridge_candidates = _make_candidates(
            bridge_smooth,
            prominence=float(bridge_cfg.get("prominence", 0.005)),
            distance=peak_distance,
            cancel_cb=cancel_cb,
        )
        traces = _bridge_traces(
            traces,
            extraction_cfg,
            candidates=bridge_candidates,
            orientation=bridge_orientation,
            coherence=bridge_coherence,
            min_slope=min_slope,
            max_slope=max_slope,
            cancel_cb=cancel_cb,
        )
        traces = _dedupe_and_extend(traces, extraction_cfg, cancel_cb=cancel_cb)
    traces = [
        trace
        for trace in traces
        if _valid_final_trace(trace, min_slope=min_slope, max_slope=max_slope, cfg=extraction_cfg)
    ]
    traces = _dedupe_and_extend(traces, extraction_cfg, cancel_cb=cancel_cb)
    validation_smooth = gaussian_filter(
        normalized,
        sigma=(temporal_sigma, float(extraction_cfg.get("validation_spatial_sigma_px", 8.0))),
    )
    min_phase_contrast = float(extraction_cfg.get("min_phase_contrast", 0.025))
    contrast_offset = int(extraction_cfg.get("phase_contrast_offset_px", 12))
    contrast_filtered: List[_Trace] = []
    for trace in traces:
        trace.phase_contrast = _phase_contrast(
            trace,
            validation_smooth,
            offset=contrast_offset,
        )
        if trace.phase_contrast >= min_phase_contrast:
            contrast_filtered.append(trace)
    traces = _dedupe_and_extend(contrast_filtered, extraction_cfg, cancel_cb=cancel_cb)
    traces.sort(key=lambda trace: (len(trace.points), trace.support), reverse=True)
    max_tracks = max(1, int(extraction_cfg.get("max_tracks", 500)))
    traces = traces[:max_tracks]

    _progress(progress_cb, "saving", {"tracks": len(traces)})
    track_paths: List[Path] = []
    track_metadata: Dict[int, Dict[str, Any]] = {}
    for track_index, trace in enumerate(traces):
        _check_cancel(cancel_cb)
        path = output_dir / f"track_{track_index:04d}.npy"
        np.save(path, trace.points.astype(np.float32, copy=False))
        slope = float(np.polyfit(trace.points[:, 0], trace.points[:, 1], 1)[0])
        metadata = {
            "track_index": track_index,
            "phase": trace.phase,
            "source": trace.source,
            "point_count": int(len(trace.points)),
            "duration_frames": float(np.ptp(trace.points[:, 0])),
            "slope_px_per_frame": slope,
            "support_fraction": float(trace.support),
            "phase_contrast": float(trace.phase_contrast),
        }
        track_paths.append(path)
        track_metadata[track_index] = metadata

    manifest = {
        "extractor": "ripple_multiscale_hough",
        "value_source": value_source,
        "seed_count": seed_count,
        "candidate_trace_count": len(raw_traces),
        "track_count": len(track_paths),
        "tracks": [track_metadata[index] for index in range(len(track_paths))],
    }
    (base_dir / "ripple_track_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_debug_artifacts(
        heatmap_path=heatmap_path,
        debug_dir=debug_dir,
        base_dir=base_dir,
        probability=probability_debug,
        candidates=aggregate_candidates,
        seeds=aggregate_seeds,
        traces=traces,
        manifest=manifest,
    )
    return RippleExtractionResult(
        image_id=image_id,
        base_dir=base_dir,
        track_paths=track_paths,
        track_metadata=track_metadata,
    )


def _resolve_scales(value: Any) -> List[Dict[str, float]]:
    raw = value if isinstance(value, list) and value else DEFAULT_SCALES
    scales: List[Dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sigma = float(item.get("spatial_sigma_px", 0.0))
        prominence = float(item.get("prominence", 0.0))
        if sigma > 0 and prominence > 0:
            scales.append({"spatial_sigma_px": sigma, "prominence": prominence})
    return scales or [dict(item) for item in DEFAULT_SCALES]


def _make_candidates(
    smooth: np.ndarray,
    *,
    prominence: float,
    distance: int,
    cancel_cb: CancelCallback,
) -> Dict[str, List[np.ndarray]]:
    result: Dict[str, List[np.ndarray]] = {"bright": [], "dark": []}
    for y, row in enumerate(smooth):
        if y % 64 == 0:
            _check_cancel(cancel_cb)
        bright, _ = find_peaks(row, distance=distance, prominence=prominence)
        dark, _ = find_peaks(-row, distance=distance, prominence=prominence)
        result["bright"].append(bright.astype(np.float32))
        result["dark"].append(dark.astype(np.float32))
    return result


def _orientation_field(smooth: np.ndarray, *, integration_sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(smooth)
    jxx = gaussian_filter(grad_x * grad_x, sigma=integration_sigma)
    jxy = gaussian_filter(grad_x * grad_y, sigma=integration_sigma)
    jyy = gaussian_filter(grad_y * grad_y, sigma=integration_sigma)
    delta = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2)
    normal_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    orientation = -np.tan(normal_angle)
    coherence = delta / np.maximum(jxx + jyy, 1e-8)
    return orientation, coherence


def _hough_seeds(
    masks: Dict[str, np.ndarray],
    *,
    theta: np.ndarray,
    cfg: Dict[str, Any],
    aggregate: np.ndarray,
    cancel_cb: CancelCallback,
) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    hough_cfg = cfg.get("hough") or {}
    height = next(iter(masks.values())).shape[0]
    window_rows = int(hough_cfg.get("window_rows", 256))
    window_step = int(hough_cfg.get("window_step_rows", 112))
    seeds: List[Tuple[str, Tuple[int, int], Tuple[int, int]]] = []
    for phase, mask in masks.items():
        _check_cancel(cancel_cb)
        global_lines = probabilistic_hough_line(
            mask.copy(),
            threshold=int(hough_cfg.get("global_threshold", 18)),
            line_length=int(hough_cfg.get("global_line_length", 65)),
            line_gap=int(hough_cfg.get("global_line_gap", 20)),
            theta=theta,
            rng=int(hough_cfg.get("random_seed", 42)),
        )
        seeds.extend((phase, p0, p1) for p0, p1 in global_lines)
        for start in range(0, height, window_step):
            _check_cancel(cancel_cb)
            stop = min(height, start + window_rows)
            if stop - start < max(60, window_rows // 2):
                continue
            local_lines = probabilistic_hough_line(
                mask[start:stop].copy(),
                threshold=int(hough_cfg.get("window_threshold", 8)),
                line_length=int(hough_cfg.get("window_line_length", 30)),
                line_gap=int(hough_cfg.get("window_line_gap", 16)),
                theta=theta,
                rng=int(hough_cfg.get("random_seed", 42)) + start,
            )
            seeds.extend(
                (phase, (p0[0], p0[1] + start), (p1[0], p1[1] + start))
                for p0, p1 in local_lines
            )
    for _phase, p0, p1 in seeds:
        cv2.line(aggregate, p0, p1, 255, 1, cv2.LINE_8)
    return seeds


def _local_candidate(
    row: np.ndarray,
    *,
    y: int,
    predicted_x: float,
    slope: float,
    initial_slope: float,
    orientation: np.ndarray,
    coherence: np.ndarray,
    min_slope: float,
    max_slope: float,
    cfg: Dict[str, Any],
) -> Optional[Tuple[int, float]]:
    radius = float(cfg.get("trace_search_radius_px", 9.0))
    min_coherence = float(cfg.get("min_orientation_coherence", 0.055))
    max_orientation_delta = float(cfg.get("max_orientation_slope_delta", 0.65))
    choices: List[Tuple[float, int, float]] = []
    for index in np.flatnonzero(np.abs(row - predicted_x) <= radius):
        x = int(round(float(row[index])))
        local_slope = float(orientation[y, x])
        if not np.isfinite(local_slope) or float(coherence[y, x]) < min_coherence:
            continue
        if np.sign(local_slope) != np.sign(initial_slope):
            continue
        if not min_slope * 0.65 <= abs(local_slope) <= max_slope * 1.4:
            continue
        if abs(local_slope - slope) > max_orientation_delta:
            continue
        distance = abs(float(row[index]) - predicted_x)
        choices.append((distance + 1.5 * abs(local_slope - slope), int(index), local_slope))
    if not choices:
        return None
    _, index, local_slope = min(choices)
    return index, local_slope


def _follow(
    candidates: Sequence[np.ndarray],
    orientation: np.ndarray,
    coherence: np.ndarray,
    *,
    start_y: int,
    start_x: float,
    initial_slope: float,
    direction: int,
    min_slope: float,
    max_slope: float,
    cfg: Dict[str, Any],
    cancel_cb: CancelCallback,
) -> List[Tuple[float, float]]:
    height, width = orientation.shape
    points = [(float(start_y), float(start_x))]
    slope = float(initial_slope)
    gap = 0
    predicted_x = float(start_x)
    y = int(start_y)
    max_gap = int(cfg.get("trace_max_gap_rows", 32))
    fit_window = int(cfg.get("trace_fit_window_rows", 45))
    while True:
        y += direction
        if y < 0 or y >= height:
            break
        if y % 64 == 0:
            _check_cancel(cancel_cb)
        predicted_x += slope * direction
        if predicted_x < 0 or predicted_x >= width:
            break
        row = candidates[y]
        choice = _local_candidate(
            row,
            y=y,
            predicted_x=predicted_x,
            slope=slope,
            initial_slope=initial_slope,
            orientation=orientation,
            coherence=coherence,
            min_slope=min_slope,
            max_slope=max_slope,
            cfg=cfg,
        )
        if choice is None:
            gap += 1
            if gap > max_gap:
                break
            continue
        index, local_slope = choice
        x = float(row[index])
        points.append((float(y), x))
        gap = 0
        recent = points[-fit_window:]
        if len(recent) >= 12:
            recent_y = np.asarray([point[0] for point in recent])
            recent_x = np.asarray([point[1] for point in recent])
            fitted = float(np.polyfit(recent_y, recent_x, 1)[0])
            if (
                np.sign(fitted) == np.sign(initial_slope)
                and min_slope * 0.75 <= abs(fitted) <= max_slope * 1.2
            ):
                slope = 0.7 * slope + 0.3 * fitted
        slope = 0.85 * slope + 0.15 * local_slope
        predicted_x = 0.65 * predicted_x + 0.35 * x
    return points


def _trace_seed_guided(
    phase: str,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    candidates: Dict[str, List[np.ndarray]],
    orientation: np.ndarray,
    coherence: np.ndarray,
    *,
    min_slope: float,
    max_slope: float,
    cfg: Dict[str, Any],
    cancel_cb: CancelCallback,
) -> Optional[_Trace]:
    x0, y0 = p0
    x1, y1 = p1
    if y0 == y1:
        return None
    slope = (x1 - x0) / (y1 - y0)
    if not min_slope <= abs(slope) <= max_slope:
        return None
    center_y = int(round((y0 + y1) / 2))
    center_x = float(x0 + slope * (center_y - y0))
    center_row = candidates[phase][center_y]
    if not center_row.size:
        return None
    center_index = int(np.argmin(np.abs(center_row - center_x)))
    if abs(float(center_row[center_index]) - center_x) > 10.0:
        return None
    center_x = float(center_row[center_index])
    backward = _follow(
        candidates[phase],
        orientation,
        coherence,
        start_y=center_y,
        start_x=center_x,
        initial_slope=slope,
        direction=-1,
        min_slope=min_slope,
        max_slope=max_slope,
        cfg=cfg,
        cancel_cb=cancel_cb,
    )
    forward = _follow(
        candidates[phase],
        orientation,
        coherence,
        start_y=center_y,
        start_x=center_x,
        initial_slope=slope,
        direction=1,
        min_slope=min_slope,
        max_slope=max_slope,
        cfg=cfg,
        cancel_cb=cancel_cb,
    )
    points = np.asarray(backward[::-1][:-1] + forward, dtype=np.float32)
    return _trace_if_valid(
        phase,
        points,
        source="guided",
        min_slope=min_slope,
        max_slope=max_slope,
        cfg=cfg,
    )


def _trace_seed_linear(
    phase: str,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    candidates: Dict[str, List[np.ndarray]],
    orientation: np.ndarray,
    coherence: np.ndarray,
    *,
    min_slope: float,
    max_slope: float,
    cfg: Dict[str, Any],
    cancel_cb: CancelCallback,
) -> Optional[_Trace]:
    x0, y0 = p0
    x1, y1 = p1
    if y0 == y1:
        return None
    slope = (x1 - x0) / (y1 - y0)
    if not min_slope <= abs(slope) <= max_slope:
        return None
    center_y = int(round((y0 + y1) / 2))
    intercept = float(x0 - slope * y0)
    observed = np.empty((0, 2), dtype=float)
    for _iteration in range(int(cfg.get("linear_refit_iterations", 3))):
        points: List[Tuple[float, float]] = []
        for y, row in enumerate(candidates[phase]):
            if y % 64 == 0:
                _check_cancel(cancel_cb)
            predicted = slope * y + intercept
            if predicted < 0 or predicted >= orientation.shape[1] or not row.size:
                continue
            choice = _local_candidate(
                row,
                y=y,
                predicted_x=predicted,
                slope=slope,
                initial_slope=slope,
                orientation=orientation,
                coherence=coherence,
                min_slope=min_slope,
                max_slope=max_slope,
                cfg=cfg,
            )
            if choice is not None:
                points.append((float(y), float(row[choice[0]])))
        if not points:
            return None
        observed = np.asarray(points, dtype=float)
        center_index = int(np.argmin(np.abs(observed[:, 0] - center_y)))
        if abs(float(observed[center_index, 0]) - center_y) > 30:
            return None
        max_gap = int(cfg.get("linear_max_gap_rows", 45))
        start = center_index
        while start > 0 and observed[start, 0] - observed[start - 1, 0] <= max_gap:
            start -= 1
        stop = center_index
        while stop + 1 < len(observed) and observed[stop + 1, 0] - observed[stop, 0] <= max_gap:
            stop += 1
        observed = observed[start : stop + 1]
        if len(observed) < 30:
            return None
        fitted_slope, fitted_intercept = np.polyfit(observed[:, 0], observed[:, 1], 1)
        if np.sign(fitted_slope) != np.sign(slope) or not min_slope <= abs(fitted_slope) <= max_slope:
            return None
        slope = 0.5 * slope + 0.5 * float(fitted_slope)
        intercept = 0.5 * intercept + 0.5 * float(fitted_intercept)

    y_start = int(observed[0, 0])
    y_stop = int(observed[-1, 0])
    if y_stop - y_start + 1 < int(cfg.get("linear_min_track_rows", 140)):
        return None
    support = len(observed) / (y_stop - y_start + 1)
    if support < float(cfg.get("linear_min_support_fraction", 0.28)):
        return None
    fitted = slope * observed[:, 0] + intercept
    rmse = float(np.sqrt(np.mean((observed[:, 1] - fitted) ** 2)))
    if rmse > float(cfg.get("linear_max_rmse_px", 7.0)):
        return None
    dense_y = np.arange(y_start, y_stop + 1, dtype=np.float32)
    dense_x = np.interp(dense_y, observed[:, 0], observed[:, 1]).astype(np.float32)
    return _Trace(phase, np.column_stack([dense_y, dense_x]), support, "linear")


def _trace_if_valid(
    phase: str,
    points: np.ndarray,
    *,
    source: str,
    min_slope: float,
    max_slope: float,
    cfg: Dict[str, Any],
) -> Optional[_Trace]:
    if len(points) < int(cfg.get("min_observed_points", 80)):
        return None
    span = float(points[-1, 0] - points[0, 0] + 1)
    support = len(points) / max(span, 1.0)
    if span < float(cfg.get("min_track_rows", 100)) or support < float(cfg.get("min_support_fraction", 0.55)):
        return None
    slope = float(np.polyfit(points[:, 0], points[:, 1], 1)[0])
    if not min_slope <= abs(slope) <= max_slope:
        return None
    total_motion = float(np.sum(np.abs(np.diff(points[:, 1]))))
    net_motion = abs(float(points[-1, 1] - points[0, 1]))
    if total_motion > 0 and net_motion / total_motion < float(cfg.get("min_directionality", 0.42)):
        return None
    return _Trace(phase, points, support, source)


def _same_track(left: _Trace, right: _Trace, cfg: Dict[str, Any]) -> bool:
    dedupe_cfg = cfg.get("dedupe") or {}
    same_phase = left.phase == right.phase
    stats = _overlap_stats(left, right)
    if stats is None:
        return False

    close_distance = float(dedupe_cfg.get("close_distance_px", 7.0))
    min_close_fraction = float(dedupe_cfg.get("min_close_fraction", 0.68))
    partial_max_median = float(dedupe_cfg.get("partial_max_median_distance_px", 7.0))
    min_spatial_overlap_rows = float(dedupe_cfg.get("min_spatial_overlap_rows", 12.0))
    if not same_phase:
        close_distance = min(close_distance, float(dedupe_cfg.get("cross_phase_close_distance_px", 5.0)))
        min_close_fraction = max(
            min_close_fraction,
            float(dedupe_cfg.get("cross_phase_min_close_fraction", 0.75)),
        )
        partial_max_median = min(
            partial_max_median,
            float(dedupe_cfg.get("cross_phase_partial_max_median_distance_px", 5.0)),
        )
        min_spatial_overlap_rows = max(
            min_spatial_overlap_rows,
            float(dedupe_cfg.get("cross_phase_min_spatial_overlap_rows", 16.0)),
        )

    if _longest_close_run_rows(stats, close_distance) >= min_spatial_overlap_rows:
        return True

    if not same_phase and not bool(dedupe_cfg.get("allow_cross_phase", True)):
        return False
    left_slope = float(np.polyfit(left.points[:, 0], left.points[:, 1], 1)[0])
    right_slope = float(np.polyfit(right.points[:, 0], right.points[:, 1], 1)[0])
    if (
        np.sign(left_slope) != np.sign(right_slope)
        or abs(left_slope - right_slope) > float(dedupe_cfg.get("max_slope_delta", 0.35))
    ):
        return False
    max_median = float(dedupe_cfg.get("max_median_distance_px", 6.0))
    max_p90 = float(dedupe_cfg.get("max_p90_distance_px", 10.0))
    if not same_phase:
        max_median = min(max_median, float(dedupe_cfg.get("cross_phase_max_median_distance_px", 4.0)))
        max_p90 = min(max_p90, float(dedupe_cfg.get("cross_phase_max_p90_distance_px", 7.0)))

    left_span = _trace_span_rows(left)
    right_span = _trace_span_rows(right)
    relative_min = float(dedupe_cfg.get("min_overlap_fraction", 0.45)) * min(left_span, right_span)
    full_overlap_required = max(float(dedupe_cfg.get("min_overlap_rows", 35.0)), min(70.0, relative_min))
    if stats.rows >= full_overlap_required and stats.median <= max_median and stats.p90 <= max_p90:
        return True

    partial_overlap_required = float(dedupe_cfg.get("min_partial_overlap_rows", 18.0))
    partial_short_fraction = float(dedupe_cfg.get("min_short_overlap_fraction", 0.55))
    if stats.rows < partial_overlap_required or stats.short_fraction < partial_short_fraction:
        return False

    close_fraction = float(np.mean(stats.distances <= close_distance))
    return stats.median <= partial_max_median and close_fraction >= min_close_fraction


def _trace_span_rows(trace: _Trace) -> float:
    return float(np.ptp(trace.points[:, 0]) + 1.0)


def _overlap_stats(left: _Trace, right: _Trace) -> Optional[_OverlapStats]:
    overlap_start = max(float(left.points[0, 0]), float(right.points[0, 0]))
    overlap_end = min(float(left.points[-1, 0]), float(right.points[-1, 0]))
    if overlap_end < overlap_start:
        return None

    overlap_rows = float(overlap_end - overlap_start + 1.0)
    short_span = min(_trace_span_rows(left), _trace_span_rows(right))
    sample_count = min(101, max(3, int(round(overlap_rows))))
    ys = np.linspace(overlap_start, overlap_end, sample_count)
    distances = np.abs(
        np.interp(ys, left.points[:, 0], left.points[:, 1])
        - np.interp(ys, right.points[:, 0], right.points[:, 1])
    )
    return _OverlapStats(
        rows=overlap_rows,
        short_fraction=overlap_rows / max(short_span, 1.0),
        distances=distances,
        median=float(np.median(distances)),
        p90=float(np.percentile(distances, 90)),
    )


def _longest_close_run_rows(stats: _OverlapStats, close_distance: float) -> float:
    close = stats.distances <= close_distance
    if not np.any(close):
        return 0.0
    longest = 0
    current = 0
    for value in close:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    row_step = stats.rows / max(1, len(stats.distances))
    return float(longest) * row_step


def _dedupe_rank(trace: _Trace) -> Tuple[float, float, float, int]:
    return (trace.support, trace.phase_contrast, _trace_span_rows(trace), len(trace.points))


def _merge_pair(left: _Trace, right: _Trace, *, source_tag: Optional[str] = None) -> _Trace:
    combined = np.vstack([left.points, right.points])
    combined = combined[np.argsort(combined[:, 0], kind="stable")]
    unique_y = np.unique(combined[:, 0])
    points = np.asarray(
        [(y, np.median(combined[combined[:, 0] == y, 1])) for y in unique_y],
        dtype=np.float32,
    )
    dense_y = np.arange(int(points[0, 0]), int(points[-1, 0]) + 1, dtype=np.float32)
    dense_x = np.interp(dense_y, points[:, 0], points[:, 1]).astype(np.float32)
    sources = set(left.source.split("+")) | set(right.source.split("+"))
    if source_tag:
        sources.add(source_tag)
    return _Trace(
        left.phase,
        np.column_stack([dense_y, dense_x]),
        min(left.support, right.support),
        "+".join(sorted(sources)),
        phase_contrast=max(left.phase_contrast, right.phase_contrast),
    )


def _dedupe_and_extend(
    items: Sequence[_Trace],
    cfg: Dict[str, Any],
    *,
    cancel_cb: CancelCallback,
) -> List[_Trace]:
    traces: List[_Trace] = []
    ordered = sorted(
        items,
        key=_dedupe_rank,
        reverse=True,
    )
    for candidate_index, candidate in enumerate(ordered):
        if candidate_index % 32 == 0:
            _check_cancel(cancel_cb)
        matches = [index for index, trace in enumerate(traces) if _same_track(candidate, trace, cfg)]
        if not matches:
            traces.append(candidate)
    return traces


def _endpoint_slope(trace: _Trace, *, tail: bool, fit_rows: int) -> float:
    points = trace.points[-fit_rows:] if tail else trace.points[:fit_rows]
    return float(np.polyfit(points[:, 0], points[:, 1], 1)[0])


def _bridge_traces(
    items: Sequence[_Trace],
    cfg: Dict[str, Any],
    *,
    candidates: Dict[str, List[np.ndarray]],
    orientation: np.ndarray,
    coherence: np.ndarray,
    min_slope: float,
    max_slope: float,
    cancel_cb: CancelCallback,
) -> List[_Trace]:
    bridge_cfg = cfg.get("endpoint_bridge") or {}
    traces = list(items)
    while True:
        _check_cancel(cancel_cb)
        proposals: List[Tuple[float, int, int]] = []
        for left_index, left in enumerate(traces):
            for right_index, right in enumerate(traces):
                if left_index == right_index or left.phase != right.phase:
                    continue
                gap = float(right.points[0, 0] - left.points[-1, 0])
                if gap < 1 or gap > float(bridge_cfg.get("max_gap_rows", 90)):
                    continue
                fit_rows = int(bridge_cfg.get("fit_rows", 40))
                left_slope = _endpoint_slope(left, tail=True, fit_rows=fit_rows)
                right_slope = _endpoint_slope(right, tail=False, fit_rows=fit_rows)
                bridge_slope = float(right.points[0, 1] - left.points[-1, 1]) / gap
                if (
                    np.sign(left_slope) != np.sign(right_slope)
                    or np.sign(left_slope) != np.sign(bridge_slope)
                    or not min_slope * 0.75 <= abs(bridge_slope) <= max_slope * 1.2
                ):
                    continue
                bridge_angle = np.arctan(bridge_slope)
                angle_error = max(
                    abs(np.arctan(left_slope) - bridge_angle),
                    abs(np.arctan(right_slope) - bridge_angle),
                )
                max_angle = np.deg2rad(float(bridge_cfg.get("max_angle_delta_deg", 14.0)))
                if angle_error > max_angle:
                    continue

                support = 0
                missing = 0
                longest_missing = 0
                search_radius = float(bridge_cfg.get("search_radius_px", 8.0))
                min_coherence = float(bridge_cfg.get("min_orientation_coherence", 0.06))
                for offset in range(1, int(gap)):
                    if offset % 32 == 0:
                        _check_cancel(cancel_cb)
                    y = int(round(float(left.points[-1, 0]))) + offset
                    predicted_x = float(left.points[-1, 1]) + bridge_slope * offset
                    row = candidates[left.phase][y]
                    matched = False
                    if row.size:
                        candidate_index = int(np.argmin(np.abs(row - predicted_x)))
                        candidate_x = int(round(float(row[candidate_index])))
                        local_slope = float(orientation[y, candidate_x])
                        matched = (
                            abs(float(row[candidate_index]) - predicted_x) <= search_radius
                            and float(coherence[y, candidate_x]) >= min_coherence
                            and np.sign(local_slope) == np.sign(bridge_slope)
                            and abs(np.arctan(local_slope) - bridge_angle)
                            <= max_angle + np.deg2rad(4.0)
                        )
                    if matched:
                        support += 1
                        missing = 0
                    else:
                        missing += 1
                        longest_missing = max(longest_missing, missing)
                support_fraction = support / max(1, int(gap) - 1)
                if gap > 24 and (
                    support_fraction < float(bridge_cfg.get("min_support_fraction", 0.14))
                    or longest_missing > int(bridge_cfg.get("max_unsupported_rows", 36))
                ):
                    continue
                score = (
                    20.0 * angle_error
                    + 0.02 * gap
                    - 2.0 * support_fraction
                    - 0.001 * (len(left.points) + len(right.points))
                )
                proposals.append((score, left_index, right_index))
        if not proposals:
            return traces
        _, left_index, right_index = min(proposals)
        merged = _merge_pair(traces[left_index], traces[right_index], source_tag="bridge")
        traces = [
            trace for index, trace in enumerate(traces) if index not in (left_index, right_index)
        ]
        traces.append(merged)


def _valid_final_trace(
    trace: _Trace,
    *,
    min_slope: float,
    max_slope: float,
    cfg: Dict[str, Any],
) -> bool:
    if len(trace.points) < int(cfg.get("min_track_rows", 100)):
        return False
    slope = float(np.polyfit(trace.points[:, 0], trace.points[:, 1], 1)[0])
    if not min_slope <= abs(slope) <= max_slope:
        return False
    total_motion = float(np.sum(np.abs(np.diff(trace.points[:, 1]))))
    net_motion = abs(float(trace.points[-1, 1] - trace.points[0, 1]))
    return total_motion <= 0 or net_motion / total_motion >= float(cfg.get("min_directionality", 0.42))


def _phase_contrast(trace: _Trace, smooth: np.ndarray, *, offset: int) -> float:
    if offset <= 0 or trace.points.size == 0:
        return 0.0
    y = np.clip(np.rint(trace.points[:, 0]).astype(int), 0, smooth.shape[0] - 1)
    x = np.rint(trace.points[:, 1]).astype(int)
    keep = (x >= offset) & (x < smooth.shape[1] - offset)
    if not np.any(keep):
        return 0.0
    center = smooth[y[keep], x[keep]]
    flanks = 0.5 * (
        smooth[y[keep], x[keep] - offset]
        + smooth[y[keep], x[keep] + offset]
    )
    signed = center - flanks if trace.phase == "bright" else flanks - center
    return float(np.median(signed))


def _write_debug_artifacts(
    *,
    heatmap_path: Path,
    debug_dir: Path,
    base_dir: Path,
    probability: Optional[np.ndarray],
    candidates: np.ndarray,
    seeds: np.ndarray,
    traces: Sequence[_Trace],
    manifest: Dict[str, Any],
) -> None:
    image = cv2.imread(str(heatmap_path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((*candidates.shape, 3), dtype=np.uint8)
    if image.shape[:2] != candidates.shape:
        image = cv2.resize(image, (candidates.shape[1], candidates.shape[0]))
    final_mask = np.zeros(candidates.shape, dtype=np.uint8)
    overlay = image.copy()
    for trace in traces:
        points = np.rint(trace.points[:, [1, 0]]).astype(np.int32)
        color = (60, 255, 60) if trace.phase == "bright" else (255, 220, 40)
        cv2.polylines(overlay, [points], False, color, 2, cv2.LINE_AA)
        cv2.polylines(final_mask, [points], False, 255, 1, cv2.LINE_8)
    if probability is not None:
        scaled = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(debug_dir / "prob.png"), scaled)
    cv2.imwrite(str(debug_dir / "mask_raw.png"), candidates.astype(np.uint8) * 255)
    cv2.imwrite(str(debug_dir / "mask_filtered.png"), seeds)
    cv2.imwrite(str(debug_dir / "skeleton.png"), final_mask)
    cv2.imwrite(str(base_dir / "overlay_tracks.png"), overlay)
    (debug_dir / "stats.txt").write_text(
        "\n".join(
            [
                "extractor=ripple_multiscale_hough",
                f"value_source={manifest['value_source']}",
                f"seeds={manifest['seed_count']}",
                f"candidate_traces={manifest['candidate_trace_count']}",
                f"tracks={manifest['track_count']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _progress(callback: ProgressCallback, stage: str, data: Dict[str, Any]) -> None:
    if callback is not None:
        callback(stage, data)


def _check_cancel(callback: CancelCallback) -> None:
    if callback is not None and callback():
        raise CancellationRequested("cancel_requested_during_ripple_extraction")
