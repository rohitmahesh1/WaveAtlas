from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from .cancel import CancellationRequested
from .heatmap_values import load_heatmap_values, normalize_heatmap_values
from .modules.kb_adapter import link_track_endpoints
from .modules.tracker import Track


CancelCallback = Optional[Callable[[], bool]]
ProgressCallback = Optional[Callable[[str, Dict[str, Any]], None]]


@dataclass(frozen=True)
class LargeWaveExtractionResult:
    image_id: str
    base_dir: Path
    track_paths: List[Path]
    track_metadata: Dict[int, Dict[str, Any]]


@dataclass
class _RidgeTrace:
    phase: str
    track: Track
    ridge_strength: float
    support_fraction: float
    detector: str = "broad"
    cusp_score: float = 0.0
    intensity_score: float = 0.0


def run_large_wave_extraction(
    *,
    heatmap_path: Path,
    scratch_dir: Path,
    config: Dict[str, Any],
    heatmap_value_bytes: Optional[bytes] = None,
    heatmap_value_meta: Optional[Dict[str, Any]] = None,
    progress_cb: ProgressCallback = None,
    cancel_cb: CancelCallback = None,
) -> LargeWaveExtractionResult:
    _check_cancel(cancel_cb)
    image_id = heatmap_path.stem
    if image_id.endswith("_heatmap"):
        image_id = image_id[:-8]
    base_dir = scratch_dir / image_id
    output_dir = base_dir / "kymobutler_output"
    debug_dir = base_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    large_cfg = ((config.get("analysis") or {}).get("large_wave") or {})
    extraction_cfg = large_cfg.get("extraction") or {}
    ensemble_cfg = extraction_cfg.get("ensemble") or {}
    intensity_bias_cfg = extraction_cfg.get("intensity_bias") or {}
    bridge_cfg = extraction_cfg.get("endpoint_bridge") or {}
    link_cfg = large_cfg.get("endpoint_link") or {}

    _progress(progress_cb, "load_values", {})
    values, value_source = load_heatmap_values(
        heatmap_path=heatmap_path,
        value_bytes=heatmap_value_bytes,
        value_meta=heatmap_value_meta or {},
    )
    percentiles = extraction_cfg.get("normalize_percentiles", [1.0, 99.0])
    if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2:
        percentiles = [1.0, 99.0]
    normalized = normalize_heatmap_values(
        values,
        percentiles=percentiles,
        context="Large-wave extraction",
    )
    height, width = normalized.shape

    spatial_sigma = _resolved_spatial_sigma(width, extraction_cfg)
    temporal_sigma = max(0.0, float(extraction_cfg.get("temporal_sigma_rows", 2.0)))
    background_value = extraction_cfg.get("background_sigma_px")
    background_sigma = max(
        spatial_sigma + 1.0,
        float(background_value) if background_value is not None else 2.15 * spatial_sigma,
    )
    prominence = max(0.0, float(extraction_cfg.get("prominence", 0.010)))
    peak_distance_value = extraction_cfg.get("peak_min_distance_px")
    peak_distance = max(
        2,
        int(peak_distance_value) if peak_distance_value is not None else int(round(spatial_sigma)),
    )

    _progress(
        progress_cb,
        "smoothing",
        {
            "spatial_sigma_px": spatial_sigma,
            "temporal_sigma_rows": temporal_sigma,
        },
    )
    smooth = gaussian_filter(normalized, sigma=(temporal_sigma, spatial_sigma))
    baseline = gaussian_filter(smooth, sigma=(max(1.0, temporal_sigma * 2.5), background_sigma))
    ridge_response = _normalized_response(
        smooth,
        baseline,
        percentile=float(extraction_cfg.get("response_percentile", 99.5)),
    )

    _progress(progress_cb, "tracing", {})
    broad_masks, candidate_counts = _ridge_candidate_masks(
        smooth,
        prominence=prominence,
        distance=peak_distance,
        intensity_bias=intensity_bias_cfg,
        cancel_cb=cancel_cb,
    )
    min_component_rows = max(2, int(extraction_cfg.get("min_component_rows", 50)))
    min_component_support = float(extraction_cfg.get("min_component_support_fraction", 0.65))
    initial_by_phase: Dict[str, List[Track]] = {}
    for phase, mask in broad_masks.items():
        initial_by_phase[phase] = _component_tracks(
            mask,
            phase=phase,
            min_rows=min_component_rows,
            min_support_fraction=min_component_support,
            cancel_cb=cancel_cb,
        )

    _progress(
        progress_cb,
        "bridging",
        {"component_tracks": sum(len(items) for items in initial_by_phase.values())},
    )
    broad_traces: List[_RidgeTrace] = []
    link_summaries: Dict[str, Dict[str, Any]] = {}
    link_manifests: Dict[str, Dict[str, Any]] = {}
    for phase, initial_tracks in initial_by_phase.items():
        _check_cancel(cancel_cb)
        linked, stats = link_track_endpoints(
            initial_tracks,
            ridge_response,
            max_gap_rows=int(bridge_cfg.get("max_gap_rows", 40)),
            min_bridge_prob=float(bridge_cfg.get("min_bridge_response", 0.01)),
            max_conflict_fraction=float(bridge_cfg.get("max_conflict_fraction", 0.05)),
            overlap_enabled=False,
            prefer_smooth_curves=True,
            curve_length_weight=float(link_cfg.get("length_weight", 0.30)),
            curve_tangent_weight=float(link_cfg.get("tangent_weight", 0.25)),
            curve_curvature_weight=float(link_cfg.get("curvature_weight", 0.35)),
            curve_max_turn_deg=float(bridge_cfg.get("max_turn_deg", 150.0)),
            curve_max_curvature=float(bridge_cfg.get("max_curvature_px_per_row2", 1.0)),
            max_chord_slope_px_per_row=float(bridge_cfg.get("max_chord_slope_px_per_row", 2.0)),
            max_step_dx_px_per_row=float(bridge_cfg.get("max_step_dx_px_per_row", 4.0)),
            max_manifest_rejections=int(bridge_cfg.get("max_manifest_rejections", 250)),
            cancel_cb=cancel_cb,
        )
        manifest = stats.pop("manifest", {})
        link_summaries[phase] = stats
        link_manifests[phase] = manifest
        for track in linked:
            _check_cancel(cancel_cb)
            smoothed = _smooth_track(
                track,
                sigma_rows=float(extraction_cfg.get("track_smoothing_sigma_rows", 1.25)),
            )
            if len(smoothed.points) < int(extraction_cfg.get("min_track_rows", 50)):
                continue
            strength = _mean_response(ridge_response, smoothed.points)
            span = int(smoothed.points[-1][0]) - int(smoothed.points[0][0]) + 1
            support = float(len(smoothed.points)) / float(max(1, span))
            broad_traces.append(
                _RidgeTrace(
                    phase=phase,
                    track=smoothed,
                    ridge_strength=strength,
                    support_fraction=support,
                    detector="broad",
                    intensity_score=_mean_response(normalized, smoothed.points),
                )
            )

    cusp_traces: List[_RidgeTrace] = []
    cusp_masks = {
        phase: np.zeros_like(mask)
        for phase, mask in broad_masks.items()
    }
    cusp_candidate_counts = {"bright": 0, "dark": 0}
    cusp_sigma: Optional[float] = None
    if bool(ensemble_cfg.get("enabled", True)):
        _progress(progress_cb, "tracing_cusps", {})
        cusp_sigma = _resolved_cusp_sigma(spatial_sigma, ensemble_cfg)
        cusp_temporal_sigma = max(
            0.0,
            float(ensemble_cfg.get("temporal_sigma_rows", 1.0)),
        )
        cusp_prominence = max(
            0.0,
            float(ensemble_cfg.get("prominence", 0.012)),
        )
        cusp_distance_value = ensemble_cfg.get("peak_min_distance_px")
        cusp_distance = max(
            2,
            int(cusp_distance_value)
            if cusp_distance_value is not None
            else int(round(max(2.0, peak_distance * 0.52))),
        )
        cusp_smooth = gaussian_filter(
            normalized,
            sigma=(cusp_temporal_sigma, cusp_sigma),
        )
        cusp_background_sigma = max(cusp_sigma + 1.0, 2.15 * cusp_sigma)
        cusp_baseline = gaussian_filter(
            cusp_smooth,
            sigma=(max(1.0, cusp_temporal_sigma * 2.5), cusp_background_sigma),
        )
        cusp_response = _normalized_response(
            cusp_smooth,
            cusp_baseline,
            percentile=float(extraction_cfg.get("response_percentile", 99.5)),
        )
        ridge_response = np.maximum(ridge_response, cusp_response)
        cusp_masks, cusp_candidate_counts = _ridge_candidate_masks(
            cusp_smooth,
            prominence=cusp_prominence,
            distance=cusp_distance,
            intensity_bias=intensity_bias_cfg,
            cancel_cb=cancel_cb,
        )
        cusp_min_rows = max(
            3,
            int(ensemble_cfg.get("min_track_rows", 80)),
        )
        cusp_half_window = max(
            2,
            int(ensemble_cfg.get("cusp_half_window_rows", 16)),
        )
        min_cusp_score = max(
            0.0,
            float(ensemble_cfg.get("min_cusp_arm_displacement_px", 5.0)),
        )
        max_cusp_step = max(
            1.0,
            float(ensemble_cfg.get("max_step_dx_px_per_row", 12.0)),
        )
        for phase, mask in cusp_masks.items():
            initial_cusp_tracks = _rowwise_tracks(
                mask,
                phase=phase,
                max_step_dx_px=max_cusp_step,
                min_rows=cusp_min_rows,
                cancel_cb=cancel_cb,
            )
            for track in initial_cusp_tracks:
                smoothed = _smooth_track(
                    track,
                    sigma_rows=float(extraction_cfg.get("track_smoothing_sigma_rows", 1.25)),
                )
                score = _cusp_score(smoothed, half_window_rows=cusp_half_window)
                if score < min_cusp_score:
                    continue
                cusp_traces.append(
                    _RidgeTrace(
                        phase=phase,
                        track=smoothed,
                        ridge_strength=_mean_response(cusp_response, smoothed.points),
                        support_fraction=1.0,
                        detector="cusp",
                        cusp_score=score,
                        intensity_score=_mean_response(normalized, smoothed.points),
                    )
                )

    traces, ensemble_summary = _reconcile_ensemble(
        broad_traces,
        cusp_traces,
        config=ensemble_cfg,
    )
    _progress(
        progress_cb,
        "deduping",
        {"candidate_tracks": len(traces)},
    )
    traces, conflict_summary = _resolve_track_conflicts(
        traces,
        config=ensemble_cfg.get("conflicts") or {},
        intensity_weight=_intensity_selection_weight(intensity_bias_cfg),
    )
    ensemble_summary["conflicts"] = conflict_summary

    intensity_weight = _intensity_selection_weight(intensity_bias_cfg)
    traces.sort(
        key=lambda trace: _trace_priority(trace, intensity_weight=intensity_weight),
        reverse=True,
    )
    max_tracks = max(1, int(extraction_cfg.get("max_tracks", 150)))
    traces = traces[:max_tracks]

    _progress(progress_cb, "saving", {"tracks": len(traces)})
    track_paths: List[Path] = []
    track_metadata: Dict[int, Dict[str, Any]] = {}
    for track_index, trace in enumerate(traces):
        _check_cancel(cancel_cb)
        path = output_dir / f"track_{track_index:04d}.npy"
        points = np.asarray(trace.track.points, dtype=np.float32)
        np.save(path, points)
        metadata = {
            "track_index": track_index,
            "phase": trace.phase,
            "source": "large_wave_multiscale_ensemble",
            "detector": trace.detector,
            "cusp_score": trace.cusp_score,
            "point_count": int(len(points)),
            "duration_frames": float(np.ptp(points[:, 0])) if len(points) else 0.0,
            "ridge_strength": trace.ridge_strength,
            "intensity_score": trace.intensity_score,
            "support_fraction": trace.support_fraction,
            "spatial_sigma_px": spatial_sigma,
        }
        track_paths.append(path)
        track_metadata[track_index] = metadata

    manifest = {
        "schema_version": 2,
        "extractor": "large_wave_multiscale_ensemble",
        "value_source": value_source,
        "spatial_sigma_px": spatial_sigma,
        "temporal_sigma_rows": temporal_sigma,
        "background_sigma_px": background_sigma,
        "prominence": prominence,
        "peak_min_distance_px": peak_distance,
        "intensity_bias": _intensity_bias_manifest(intensity_bias_cfg),
        "candidate_counts": candidate_counts,
        "cusp_spatial_sigma_px": cusp_sigma,
        "cusp_candidate_counts": cusp_candidate_counts,
        "component_track_counts": {
            phase: len(items) for phase, items in initial_by_phase.items()
        },
        "endpoint_link_summaries": link_summaries,
        "endpoint_link_manifests": link_manifests,
        "ensemble": ensemble_summary,
        "track_count": len(track_paths),
        "tracks": [track_metadata[index] for index in range(len(track_paths))],
    }
    (debug_dir / "large_wave_ridges.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_debug_artifacts(
        heatmap_path=heatmap_path,
        base_dir=base_dir,
        debug_dir=debug_dir,
        ridge_response=ridge_response,
        masks={
            phase: np.maximum(broad_masks[phase], cusp_masks[phase])
            for phase in broad_masks
        },
        traces=traces,
        manifest=manifest,
    )
    return LargeWaveExtractionResult(
        image_id=image_id,
        base_dir=base_dir,
        track_paths=track_paths,
        track_metadata=track_metadata,
    )


def _resolved_spatial_sigma(width: int, cfg: Dict[str, Any]) -> float:
    explicit = cfg.get("spatial_sigma_px")
    if explicit is not None:
        return max(1.0, float(explicit))
    fraction = max(0.0, float(cfg.get("spatial_sigma_fraction", 0.02)))
    lower = max(1.0, float(cfg.get("min_spatial_sigma_px", 12.0)))
    upper = max(lower, float(cfg.get("max_spatial_sigma_px", 36.0)))
    return float(np.clip(float(width) * fraction, lower, upper))


def _resolved_cusp_sigma(broad_sigma: float, cfg: Dict[str, Any]) -> float:
    ratio = max(0.05, float(cfg.get("cusp_spatial_sigma_ratio", 0.35)))
    lower = max(1.0, float(cfg.get("min_spatial_sigma_px", 6.0)))
    upper = max(lower, float(cfg.get("max_spatial_sigma_px", 16.0)))
    return float(np.clip(broad_sigma * ratio, min(lower, broad_sigma), min(upper, broad_sigma)))


def _normalized_response(
    smooth: np.ndarray,
    baseline: np.ndarray,
    *,
    percentile: float,
) -> np.ndarray:
    response = np.abs(smooth - baseline)
    scale = max(float(np.percentile(response, percentile)), 1e-6)
    return np.clip(response / scale, 0.0, 1.0).astype(np.float32)


def _resolved_intensity_bias(config: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool(config.get("enabled", False))
    strength = float(np.clip(float(config.get("strength", 0.0)), 0.0, 1.0))
    min_factor = max(0.05, float(config.get("min_prominence_factor", 0.65)))
    max_factor = max(min_factor, float(config.get("max_prominence_factor", 1.60)))
    effective_min = 1.0 + strength * (min_factor - 1.0)
    effective_max = 1.0 + strength * (max_factor - 1.0)
    if not enabled:
        effective_min = 1.0
        effective_max = 1.0
    return {
        "enabled": enabled,
        "strength": strength,
        "min_prominence_factor": min_factor,
        "max_prominence_factor": max_factor,
        "effective_min_prominence_factor": effective_min,
        "effective_max_prominence_factor": effective_max,
    }


def _intensity_biased_peaks(
    signal: np.ndarray,
    *,
    intensity_row: np.ndarray,
    prominence: float,
    distance: int,
    bias: Dict[str, Any],
) -> np.ndarray:
    if not bias["enabled"] or bias["strength"] <= 0.0:
        peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)
        return peaks

    minimum_factor = float(bias["effective_min_prominence_factor"])
    maximum_factor = float(bias["effective_max_prominence_factor"])
    peaks, properties = find_peaks(
        signal,
        prominence=prominence * minimum_factor,
        distance=distance,
    )
    if not len(peaks):
        return peaks
    intensity = np.clip(intensity_row[peaks], 0.0, 1.0)
    required_factor = maximum_factor - intensity * (maximum_factor - minimum_factor)
    required_prominence = prominence * required_factor
    return peaks[properties["prominences"] + 1e-12 >= required_prominence]


def _intensity_selection_weight(config: Dict[str, Any]) -> float:
    bias = _resolved_intensity_bias(config)
    if not bias["enabled"]:
        return 0.0
    return max(0.0, float(config.get("selection_weight", 0.75)))


def _intensity_bias_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    manifest = _resolved_intensity_bias(config)
    manifest["selection_weight"] = _intensity_selection_weight(config)
    return manifest


def _ridge_candidate_masks(
    smooth: np.ndarray,
    *,
    prominence: float,
    distance: int,
    intensity_bias: Optional[Dict[str, Any]] = None,
    cancel_cb: CancelCallback,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    height, width = smooth.shape
    masks = {
        "bright": np.zeros((height, width), dtype=np.uint8),
        "dark": np.zeros((height, width), dtype=np.uint8),
    }
    counts = {"bright": 0, "dark": 0}
    bias = _resolved_intensity_bias(intensity_bias or {})
    for y, row in enumerate(smooth):
        if y % 64 == 0:
            _check_cancel(cancel_cb)
        bright = _intensity_biased_peaks(
            row,
            intensity_row=row,
            prominence=prominence,
            distance=distance,
            bias=bias,
        )
        dark = _intensity_biased_peaks(
            -row,
            intensity_row=row,
            prominence=prominence,
            distance=distance,
            bias=bias,
        )
        masks["bright"][y, bright] = 1
        masks["dark"][y, dark] = 1
        counts["bright"] += int(len(bright))
        counts["dark"] += int(len(dark))
    return masks, counts


def _component_tracks(
    mask: np.ndarray,
    *,
    phase: str,
    min_rows: int,
    min_support_fraction: float,
    cancel_cb: CancelCallback,
) -> List[Track]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    tracks: List[Track] = []
    for component in range(1, count):
        if component % 64 == 0:
            _check_cancel(cancel_cb)
        pixel_count = int(stats[component, cv2.CC_STAT_AREA])
        if pixel_count < min_rows:
            continue
        y0 = int(stats[component, cv2.CC_STAT_TOP])
        span = int(stats[component, cv2.CC_STAT_HEIGHT])
        if span < min_rows:
            continue
        ys, xs = np.nonzero(labels == component)
        rows = np.unique(ys)
        support = float(len(rows)) / float(max(1, span))
        if support < min_support_fraction:
            continue
        points = [
            (int(y), int(round(float(np.median(xs[ys == y])))))
            for y in rows
        ]
        tracks.append(Track(points=points, id=f"{phase}:{component}:{y0}"))
    tracks.sort(
        key=lambda track: (
            track.points[0][0],
            track.points[0][1],
            -len(track.points),
        )
    )
    return tracks


def _rowwise_tracks(
    mask: np.ndarray,
    *,
    phase: str,
    max_step_dx_px: float,
    min_rows: int,
    cancel_cb: CancelCallback,
) -> List[Track]:
    tracks: List[List[Tuple[int, int]]] = []
    active: List[int] = []
    for y in range(mask.shape[0]):
        if y % 64 == 0:
            _check_cancel(cancel_cb)
        xs = np.flatnonzero(mask[y]).astype(np.int32)
        current: List[Optional[int]] = [None] * len(xs)
        if active and len(xs):
            previous_x = np.asarray(
                [tracks[index][-1][1] for index in active],
                dtype=np.float64,
            )
            costs = np.abs(previous_x[:, None] - xs[None, :]).astype(np.float64)
            forbidden = max_step_dx_px + 1e6
            assignment_costs = np.where(costs <= max_step_dx_px, costs, forbidden)
            source_rows, target_cols = linear_sum_assignment(assignment_costs)
            for source_row, target_col in zip(source_rows, target_cols):
                if costs[source_row, target_col] > max_step_dx_px:
                    continue
                track_index = active[int(source_row)]
                tracks[track_index].append((int(y), int(xs[target_col])))
                current[int(target_col)] = track_index
        for candidate_index, x in enumerate(xs):
            if current[candidate_index] is not None:
                continue
            tracks.append([(int(y), int(x))])
            current[candidate_index] = len(tracks) - 1
        active = [int(index) for index in current if index is not None]

    output = [
        Track(points=points, id=f"{phase}:cusp:{index}")
        for index, points in enumerate(tracks)
        if len(points) >= min_rows
    ]
    output.sort(
        key=lambda track: (
            -len(track.points),
            track.points[0][0],
            track.points[0][1],
        )
    )
    return output


def _cusp_score(track: Track, *, half_window_rows: int) -> float:
    if len(track.points) <= 2 * half_window_rows:
        return 0.0
    points = sorted(track.points, key=lambda point: (point[0], point[1]))
    ys = np.asarray([point[0] for point in points], dtype=np.int32)
    xs = gaussian_filter1d(
        np.asarray([point[1] for point in points], dtype=np.float64),
        sigma=2.0,
        mode="nearest",
    )
    best = 0.0
    window = int(half_window_rows)
    for center in range(window, len(points) - window):
        if ys[center] - ys[center - window] != window:
            continue
        if ys[center + window] - ys[center] != window:
            continue
        left = float(xs[center] - xs[center - window])
        right = float(xs[center] - xs[center + window])
        if left * right <= 0.0:
            continue
        best = max(best, min(abs(left), abs(right)))
    return best


def _reconcile_ensemble(
    broad: List[_RidgeTrace],
    cusp: List[_RidgeTrace],
    *,
    config: Dict[str, Any],
) -> Tuple[List[_RidgeTrace], Dict[str, Any]]:
    min_overlap_rows = max(1, int(config.get("dedupe_min_overlap_rows", 30)))
    max_distance = max(0.0, float(config.get("dedupe_max_distance_px", 8.0)))
    min_close_fraction = float(config.get("dedupe_min_close_fraction", 0.65))
    min_broad_coverage = float(config.get("dedupe_min_broad_coverage", 0.35))

    def duplicates(broad_trace: _RidgeTrace, cusp_trace: _RidgeTrace) -> bool:
        if broad_trace.phase != cusp_trace.phase:
            return False
        broad_rows = {int(y): float(x) for y, x in broad_trace.track.points}
        cusp_rows = {int(y): float(x) for y, x in cusp_trace.track.points}
        shared_rows = sorted(broad_rows.keys() & cusp_rows.keys())
        if len(shared_rows) < min_overlap_rows:
            return False
        distances = np.asarray(
            [abs(broad_rows[y] - cusp_rows[y]) for y in shared_rows],
            dtype=np.float64,
        )
        close_fraction = float(np.mean(distances <= max_distance))
        broad_coverage = float(len(shared_rows)) / float(max(1, len(broad_rows)))
        return (
            close_fraction >= min_close_fraction
            and broad_coverage >= min_broad_coverage
        )

    kept_broad = [
        broad_trace
        for broad_trace in broad
        if not any(duplicates(broad_trace, cusp_trace) for cusp_trace in cusp)
    ]
    combined = [*kept_broad, *cusp]
    return combined, {
        "broad_track_count": len(broad),
        "cusp_track_count": len(cusp),
        "replaced_broad_track_count": len(broad) - len(kept_broad),
        "reconciled_track_count": len(combined),
    }


def _resolve_track_conflicts(
    traces: List[_RidgeTrace],
    *,
    config: Dict[str, Any],
    intensity_weight: float = 0.0,
) -> Tuple[List[_RidgeTrace], Dict[str, Any]]:
    min_overlap_rows = max(2, int(config.get("min_overlap_rows", 12)))
    max_overlap_distance = max(0.0, float(config.get("max_overlap_distance_px", 3.0)))
    min_close_fraction = float(config.get("min_close_fraction", 0.60))
    min_close_run_rows = max(2, int(config.get("min_close_run_rows", 8)))
    intersection_tolerance = max(
        0.0,
        float(config.get("intersection_tolerance_px", 0.5)),
    )

    ordered = sorted(
        traces,
        key=lambda trace: _trace_priority(
            trace,
            intensity_weight=max(0.0, float(intensity_weight)),
        ),
        reverse=True,
    )
    kept: List[_RidgeTrace] = []
    rejected: List[Dict[str, Any]] = []
    counts = {"overlap": 0, "intersection": 0}
    for candidate in ordered:
        rejection: Optional[Tuple[str, _RidgeTrace, Dict[str, Any]]] = None
        for survivor in kept:
            conflict = _track_conflict(
                candidate.track,
                survivor.track,
                min_overlap_rows=min_overlap_rows,
                max_overlap_distance=max_overlap_distance,
                min_close_fraction=min_close_fraction,
                min_close_run_rows=min_close_run_rows,
                intersection_tolerance=intersection_tolerance,
            )
            if conflict is not None:
                rejection = (conflict[0], survivor, conflict[1])
                break
        if rejection is None:
            kept.append(candidate)
            continue
        reason, survivor, details = rejection
        counts[reason] += 1
        if len(rejected) < 250:
            rejected.append(
                {
                    "reason": reason,
                    "rejected_track_id": str(candidate.track.id),
                    "rejected_detector": candidate.detector,
                    "rejected_point_count": len(candidate.track.points),
                    "rejected_intensity_score": candidate.intensity_score,
                    "survivor_track_id": str(survivor.track.id),
                    "survivor_detector": survivor.detector,
                    "survivor_point_count": len(survivor.track.points),
                    "survivor_intensity_score": survivor.intensity_score,
                    **details,
                }
            )
    return kept, {
        "input_track_count": len(traces),
        "output_track_count": len(kept),
        "rejected_track_count": len(traces) - len(kept),
        "rejection_counts": counts,
        "intensity_selection_weight": max(0.0, float(intensity_weight)),
        "rejected_examples": rejected,
    }


def _trace_priority(
    trace: _RidgeTrace,
    *,
    intensity_weight: float,
) -> Tuple[float, int, float, float, float, bool]:
    point_count = len(trace.track.points)
    intensity = float(np.clip(trace.intensity_score, 0.0, 1.0))
    return (
        float(np.log1p(point_count)) + intensity_weight * intensity,
        point_count,
        trace.cusp_score,
        trace.ridge_strength,
        trace.support_fraction,
        trace.detector == "cusp",
    )


def _track_conflict(
    first: Track,
    second: Track,
    *,
    min_overlap_rows: int,
    max_overlap_distance: float,
    min_close_fraction: float,
    min_close_run_rows: int,
    intersection_tolerance: float,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    first_rows = {int(y): float(x) for y, x in first.points}
    second_rows = {int(y): float(x) for y, x in second.points}
    shared_rows = sorted(first_rows.keys() & second_rows.keys())
    if len(shared_rows) < 2:
        return None

    signed_distance = np.asarray(
        [first_rows[y] - second_rows[y] for y in shared_rows],
        dtype=np.float64,
    )
    for index in range(len(shared_rows) - 1):
        if shared_rows[index + 1] != shared_rows[index] + 1:
            continue
        left = float(signed_distance[index])
        right = float(signed_distance[index + 1])
        if abs(left) <= intersection_tolerance or abs(right) <= intersection_tolerance:
            return "intersection", {"intersection_row": shared_rows[index]}
        if left * right < 0.0:
            return "intersection", {"intersection_row": shared_rows[index]}

    if len(shared_rows) < min_overlap_rows:
        return None
    close = np.abs(signed_distance) <= max_overlap_distance
    close_fraction = float(np.mean(close))
    longest_run = 0
    current_run = 0
    previous_row: Optional[int] = None
    for y, is_close in zip(shared_rows, close):
        if is_close and (previous_row is None or y == previous_row + 1):
            current_run += 1
        elif is_close:
            current_run = 1
        else:
            current_run = 0
        longest_run = max(longest_run, current_run)
        previous_row = y
    if close_fraction >= min_close_fraction or longest_run >= min_close_run_rows:
        return "overlap", {
            "shared_rows": len(shared_rows),
            "close_fraction": close_fraction,
            "longest_close_run_rows": longest_run,
        }
    return None


def _smooth_track(track: Track, *, sigma_rows: float) -> Track:
    if len(track.points) < 3 or sigma_rows <= 0:
        return track
    points = sorted(track.points, key=lambda point: (point[0], point[1]))
    ys = np.asarray([point[0] for point in points], dtype=np.int32)
    xs = np.asarray([point[1] for point in points], dtype=np.float64)
    smooth_x = gaussian_filter1d(xs, sigma=sigma_rows, mode="nearest")
    return Track(
        points=[(int(y), int(round(x))) for y, x in zip(ys, smooth_x)],
        id=track.id,
    )


def _mean_response(response: np.ndarray, points: List[Tuple[int, int]]) -> float:
    if not points:
        return 0.0
    height, width = response.shape
    values = [
        float(response[y, x])
        for y, x in points
        if 0 <= int(y) < height and 0 <= int(x) < width
    ]
    return float(np.mean(values)) if values else 0.0


def _write_debug_artifacts(
    *,
    heatmap_path: Path,
    base_dir: Path,
    debug_dir: Path,
    ridge_response: np.ndarray,
    masks: Dict[str, np.ndarray],
    traces: List[_RidgeTrace],
    manifest: Dict[str, Any],
) -> None:
    combined = np.maximum(masks["bright"], masks["dark"])
    response_image = np.asarray(np.clip(ridge_response * 255.0, 0, 255), dtype=np.uint8)
    mask_image = combined.astype(np.uint8) * 255
    cv2.imwrite(str(debug_dir / "prob.png"), response_image)
    cv2.imwrite(str(debug_dir / "mask_raw.png"), mask_image)
    cv2.imwrite(str(debug_dir / "mask_clean.png"), mask_image)
    cv2.imwrite(str(debug_dir / "mask_filtered.png"), mask_image)
    cv2.imwrite(str(debug_dir / "skeleton.png"), mask_image)

    image = cv2.imread(str(heatmap_path), cv2.IMREAD_COLOR)
    if image is not None:
        overlay = image.copy()
        for trace in traces:
            points = np.asarray(
                [(int(x), int(y)) for y, x in trace.track.points],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            if len(points) >= 2:
                cv2.polylines(overlay, [points], False, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(base_dir / "overlay_tracks.png"), overlay)

    with open(debug_dir / "stats.txt", "w") as handle:
        handle.write("extractor=large_wave_multiscale_ensemble\n")
        handle.write(f"value_source={manifest['value_source']}\n")
        handle.write(f"spatial_sigma_px={manifest['spatial_sigma_px']:.4f}\n")
        handle.write(f"candidate_counts={json.dumps(manifest['candidate_counts'], sort_keys=True)}\n")
        handle.write(
            f"component_track_counts={json.dumps(manifest['component_track_counts'], sort_keys=True)}\n"
        )
        handle.write(f"track_count={manifest['track_count']}\n")


def _progress(callback: ProgressCallback, stage: str, data: Dict[str, Any]) -> None:
    if callback is not None:
        callback(stage, data)


def _check_cancel(callback: CancelCallback) -> None:
    if callback is not None and callback():
        raise CancellationRequested("cancel_requested")
