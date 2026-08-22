from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class LargeWaveFit:
    """A fitted broad excursion and the local baseline that defines it."""

    center_idx: int
    window_lo: int
    window_hi: int
    event_kind: str
    baseline: np.ndarray
    residual_fit: np.ndarray
    fitted_position: np.ndarray
    metrics: Dict[str, Any]


@dataclass(frozen=True)
class LargeWaveBasinCandidate:
    """A same-side residual basin observed at one smoothing scale."""

    window_lo: int
    window_hi: int
    smoothing_sigma_rows: float


def large_wave_basin_window(
    residual: np.ndarray,
    *,
    center_idx: int,
    minimum_width_frames: Optional[float],
    width_multiplier: float,
    smoothing_sigma_rows: float,
    baseline_tolerance_fraction: float = 0.0,
) -> Optional[Tuple[int, int]]:
    values = np.asarray(residual, dtype=float)
    if center_idx < 0 or center_idx >= len(values) or len(values) < 3:
        return None
    sigma = max(0.0, float(smoothing_sigma_rows))
    smooth = gaussian_filter1d(values, sigma=sigma, mode="nearest") if sigma > 0 else values
    center_value = float(smooth[center_idx])
    if not math.isfinite(center_value) or abs(center_value) <= 1e-9:
        return None
    sign = 1.0 if center_value > 0 else -1.0
    baseline_tolerance = abs(center_value) * max(0.0, float(baseline_tolerance_fraction))

    lo = center_idx
    while lo > 0 and float(smooth[lo - 1]) * sign > baseline_tolerance:
        lo -= 1
    hi = center_idx
    while hi < len(smooth) - 1 and float(smooth[hi + 1]) * sign > baseline_tolerance:
        hi += 1

    try:
        minimum_width = max(0.0, float(minimum_width_frames or 0.0))
    except (TypeError, ValueError):
        minimum_width = 0.0
    basin_width = float(hi - lo)
    if max(basin_width, minimum_width) <= 1:
        return None
    if basin_width >= minimum_width:
        base_lo, base_hi = lo, hi
    else:
        half_span = minimum_width / 2.0
        base_lo = max(0, int(math.floor(float(center_idx) - half_span)))
        base_hi = min(len(values) - 1, int(math.ceil(float(center_idx) + half_span)))

    multiplier = max(0.1, float(width_multiplier))
    target_lo = max(
        0,
        int(math.floor(float(center_idx) - float(center_idx - base_lo) * multiplier)),
    )
    target_hi = min(
        len(values) - 1,
        int(math.ceil(float(center_idx) + float(base_hi - center_idx) * multiplier)),
    )
    return target_lo, target_hi


def large_wave_basin_candidates(
    residual: np.ndarray,
    *,
    center_idx: int,
    minimum_width_frames: Optional[float],
    width_multiplier: float,
    smoothing_sigma_rows: float,
    smoothing_scale_multipliers: Sequence[float],
    max_sigma_fraction: float = 0.125,
    baseline_tolerance_fraction: float = 0.02,
) -> List[LargeWaveBasinCandidate]:
    """Return unique nested basins from local through broad smoothing scales."""

    values = np.asarray(residual, dtype=float)
    if values.size < 3:
        return []
    base_sigma = max(0.0, float(smoothing_sigma_rows))
    multipliers = [1.0]
    for value in smoothing_scale_multipliers:
        try:
            multiplier = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
        if multiplier not in multipliers:
            multipliers.append(multiplier)
    max_sigma = max(base_sigma, float(values.size) * max(0.0, float(max_sigma_fraction)))
    unique: Dict[Tuple[int, int], LargeWaveBasinCandidate] = {}
    for multiplier in sorted(multipliers):
        sigma = min(base_sigma * multiplier, max_sigma)
        window = large_wave_basin_window(
            values,
            center_idx=center_idx,
            minimum_width_frames=minimum_width_frames,
            width_multiplier=width_multiplier,
            smoothing_sigma_rows=sigma,
            baseline_tolerance_fraction=baseline_tolerance_fraction,
        )
        if window is None:
            continue
        lo, hi = int(window[0]), int(window[1])
        candidate = LargeWaveBasinCandidate(lo, hi, sigma)
        previous = unique.get((lo, hi))
        if previous is None or sigma < previous.smoothing_sigma_rows:
            unique[(lo, hi)] = candidate
    return sorted(
        unique.values(),
        key=lambda candidate: (
            candidate.window_hi - candidate.window_lo,
            candidate.smoothing_sigma_rows,
        ),
    )


def fit_asymmetric_basin_residual(
    residual: np.ndarray,
    frame: np.ndarray,
    center_idx: int,
    *,
    window_lo: int,
    window_hi: int,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Fit an apex-anchored asymmetric half-cosine to baseline-relative data."""

    values = np.asarray(residual, dtype=np.float64)
    frame_values = np.asarray(frame, dtype=np.float64)
    if center_idx < 0 or center_idx >= len(frame_values):
        return None
    lo = max(0, min(int(window_lo), center_idx - 1))
    hi = min(len(frame_values) - 1, max(int(window_hi), center_idx + 1))
    if not (lo < center_idx < hi):
        return None

    shape = np.zeros(len(frame_values), dtype=np.float64)
    left_span = max(float(frame_values[center_idx] - frame_values[lo]), 1e-9)
    right_span = max(float(frame_values[hi] - frame_values[center_idx]), 1e-9)
    left_phase = np.clip(
        (float(frame_values[center_idx]) - frame_values[lo : center_idx + 1]) / left_span,
        0.0,
        1.0,
    )
    right_phase = np.clip(
        (frame_values[center_idx : hi + 1] - float(frame_values[center_idx])) / right_span,
        0.0,
        1.0,
    )
    peak_value = float(values[center_idx])
    observed_window = values[lo : hi + 1]
    left_base = np.clip(np.cos(0.5 * math.pi * left_phase), 0.0, 1.0)
    right_base = np.clip(np.cos(0.5 * math.pi * right_phase), 0.0, 1.0)

    def fit_power(base_shape: np.ndarray, observed: np.ndarray) -> float:
        if len(base_shape) < 4 or not math.isfinite(peak_value):
            return 1.0

        def error(power: float) -> float:
            predicted = peak_value * np.power(base_shape, power)
            return float(np.mean((observed - predicted) ** 2))

        result = minimize_scalar(error, bounds=(0.25, 16.0), method="bounded")
        return float(result.x) if result.success and math.isfinite(float(result.x)) else 1.0

    left_power = fit_power(left_base, values[lo : center_idx + 1])
    right_power = fit_power(right_base, values[center_idx : hi + 1])
    shape[lo : center_idx + 1] = np.power(left_base, left_power)
    shape[center_idx : hi + 1] = np.power(right_base, right_power)
    shape[lo] = 0.0
    shape[center_idx] = 1.0
    shape[hi] = 0.0
    fitted_residual = peak_value * shape

    quality = _fit_quality(observed_window, fitted_residual[lo : hi + 1])
    return fitted_residual, {
        "fit_method": "asymmetric_half_cosine_basin",
        "fit_amp_A": peak_value,
        "fit_phase_phi": None,
        "fit_offset_c": 0.0,
        "fit_freq_hz": None,
        "fit_left_shape_power": left_power,
        "fit_right_shape_power": right_power,
        "fit_window_lo": lo,
        "fit_window_hi": hi,
        "fit_peak_value": peak_value,
        "fit_peak_error": 0.0,
        "fit_passes_peak": True,
        **quality,
    }


def fit_large_wave(
    *,
    frame: np.ndarray,
    position: np.ndarray,
    global_baseline: np.ndarray,
    center_idx: int,
    event_kind: str,
    sampling_rate: float,
    minimum_width_frames: Optional[float] = None,
    width_multiplier: float = 1.0,
    boundary_smoothing_sigma_rows: float = 4.0,
    boundary_smoothing_scales: Optional[Sequence[float]] = None,
    boundary_max_sigma_fraction: float = 0.125,
    boundary_baseline_tolerance_fraction: float = 0.02,
    fit_error_tolerance: float = 0.25,
    max_fit_error_vnmse: float = 0.8,
    endpoint_anchor_rows: int = 7,
    curvature_half_window_rows: int = 8,
    max_period_boundary_error_fraction: float = 0.5,
    fixed_window: Optional[Tuple[int, int]] = None,
    refine_apex: bool = True,
) -> Optional[LargeWaveFit]:
    frame_values = np.asarray(frame, dtype=np.float64)
    positions = np.asarray(position, dtype=np.float64)
    global_base = np.asarray(global_baseline, dtype=np.float64)
    if not (len(frame_values) == len(positions) == len(global_base)) or len(frame_values) < 3:
        return None
    if center_idx < 0 or center_idx >= len(frame_values):
        return None

    global_residual = positions - global_base
    if fixed_window is None and boundary_smoothing_scales:
        basin_candidates = large_wave_basin_candidates(
            global_residual,
            center_idx=center_idx,
            minimum_width_frames=minimum_width_frames,
            width_multiplier=width_multiplier,
            smoothing_sigma_rows=boundary_smoothing_sigma_rows,
            smoothing_scale_multipliers=boundary_smoothing_scales,
            max_sigma_fraction=boundary_max_sigma_fraction,
            baseline_tolerance_fraction=boundary_baseline_tolerance_fraction,
        )
        candidate_fits: List[LargeWaveFit] = []
        for basin in basin_candidates:
            candidate_fit = fit_large_wave(
                frame=frame_values,
                position=positions,
                global_baseline=global_base,
                center_idx=center_idx,
                event_kind=event_kind,
                sampling_rate=sampling_rate,
                minimum_width_frames=minimum_width_frames,
                width_multiplier=width_multiplier,
                boundary_smoothing_sigma_rows=boundary_smoothing_sigma_rows,
                boundary_smoothing_scales=None,
                boundary_max_sigma_fraction=boundary_max_sigma_fraction,
                boundary_baseline_tolerance_fraction=boundary_baseline_tolerance_fraction,
                fit_error_tolerance=fit_error_tolerance,
                max_fit_error_vnmse=max_fit_error_vnmse,
                endpoint_anchor_rows=endpoint_anchor_rows,
                curvature_half_window_rows=curvature_half_window_rows,
                max_period_boundary_error_fraction=max_period_boundary_error_fraction,
                fixed_window=(basin.window_lo, basin.window_hi),
                refine_apex=refine_apex,
            )
            if candidate_fit is None:
                continue
            candidate_metrics = {
                **candidate_fit.metrics,
                "fit_boundary_sigma_rows": basin.smoothing_sigma_rows,
            }
            candidate_fits.append(replace(candidate_fit, metrics=candidate_metrics))
        if not candidate_fits:
            return None
        selected = _select_multiscale_fit(
            candidate_fits,
            error_tolerance=fit_error_tolerance,
            max_error_vnmse=max_fit_error_vnmse,
        )
        parent = max(
            basin_candidates,
            key=lambda basin: basin.window_hi - basin.window_lo,
        )
        selected_metrics = {
            **selected.metrics,
            "fit_window_source": "large_wave_multiscale_local_chord",
            "fit_window_selection": "longest_coherent_multiscale_basin",
            "fit_window_candidate_count": len(candidate_fits),
            "fit_support_window_lo": parent.window_lo,
            "fit_support_window_hi": parent.window_hi,
            "fit_support_window_width_frames": float(parent.window_hi - parent.window_lo),
            "fit_support_boundary_sigma_rows": parent.smoothing_sigma_rows,
        }
        return replace(selected, metrics=selected_metrics)

    window = fixed_window or large_wave_basin_window(
        global_residual,
        center_idx=center_idx,
        minimum_width_frames=minimum_width_frames,
        width_multiplier=width_multiplier,
        smoothing_sigma_rows=boundary_smoothing_sigma_rows,
    )
    if window is None:
        return None
    lo = max(0, min(int(window[0]), center_idx - 1))
    hi = min(len(frame_values) - 1, max(int(window[1]), center_idx + 1))
    if not (lo < center_idx < hi):
        return None

    anchor_radius = max(1, int(endpoint_anchor_rows))
    left_anchor = _robust_endpoint_anchor(positions, global_base, lo, anchor_radius)
    right_anchor = _robust_endpoint_anchor(positions, global_base, hi, anchor_radius)
    frame_span = float(frame_values[hi] - frame_values[lo])
    if not math.isfinite(frame_span) or frame_span <= 0:
        return None
    baseline_slope = (right_anchor - left_anchor) / frame_span
    local_baseline = left_anchor + baseline_slope * (frame_values - float(frame_values[lo]))
    local_residual = positions - local_baseline

    normalized_kind = "min" if str(event_kind).lower() == "min" else "max"
    sign = -1.0 if normalized_kind == "min" else 1.0
    if refine_apex:
        sigma = max(0.0, float(boundary_smoothing_sigma_rows))
        oriented = local_residual * sign
        smooth = gaussian_filter1d(oriented, sigma=sigma, mode="nearest") if sigma > 0 else oriented
        interior = smooth[lo + 1 : hi]
        if interior.size:
            center_idx = int(lo + 1 + np.nanargmax(interior))
    signed_amplitude = float(local_residual[center_idx])
    if not math.isfinite(signed_amplitude) or signed_amplitude * sign <= 0:
        return None

    fit_result = fit_asymmetric_basin_residual(
        local_residual,
        frame_values,
        center_idx,
        window_lo=lo,
        window_hi=hi,
    )
    if fit_result is None:
        return None
    fitted_residual, fit_meta = fit_result
    fitted_position = local_baseline + fitted_residual

    peak_frame = float(frame_values[center_idx])
    peak_position = float(fitted_position[center_idx])
    duration_frames = frame_span
    duration_s = duration_frames / sampling_rate if sampling_rate > 0 else None
    left_lobe_frames = peak_frame - float(frame_values[lo])
    right_lobe_frames = float(frame_values[hi]) - peak_frame
    equivalent_period_frames = 2.0 * duration_frames
    equivalent_period_s = (
        equivalent_period_frames / sampling_rate if sampling_rate > 0 else None
    )
    equivalent_frequency_hz = (
        1.0 / equivalent_period_s
        if equivalent_period_s is not None and equivalent_period_s > 0
        else None
    )
    left_power = float(fit_meta["fit_left_shape_power"])
    right_power = float(fit_meta["fit_right_shape_power"])
    half_left = _half_height_frame(
        boundary_frame=float(frame_values[lo]),
        apex_frame=peak_frame,
        power=left_power,
        left=True,
    )
    half_right = _half_height_frame(
        boundary_frame=float(frame_values[hi]),
        apex_frame=peak_frame,
        power=right_power,
        left=False,
    )
    half_width_frames = max(0.0, half_right - half_left)

    time_window = frame_values[lo : hi + 1] / max(float(sampling_rate), 1e-9)
    fit_window = fitted_residual[lo : hi + 1]
    integrated = float(np.trapz(np.abs(fit_window), time_window)) if hi > lo else 0.0
    wave_velocity = _gradient(fitted_residual, frame_values) * float(sampling_rate)
    approach = np.abs(wave_velocity[lo : center_idx + 1])
    recovery = np.abs(wave_velocity[center_idx : hi + 1])
    max_approach = float(np.max(approach)) if approach.size else 0.0
    max_recovery = float(np.max(recovery)) if recovery.size else 0.0
    curvature = _fit_curvature(
        frame_values,
        fitted_position,
        center_idx,
        half_window=max(2, int(curvature_half_window_rows)),
    )
    fit_boundary_extrapolated = bool(lo == 0 or hi == len(frame_values) - 1)
    left_boundary_ratio = abs(float(local_residual[lo])) / max(abs(signed_amplitude), 1e-9)
    right_boundary_ratio = abs(float(local_residual[hi])) / max(abs(signed_amplitude), 1e-9)
    period_boundary_error = max(left_boundary_ratio, right_boundary_ratio)
    period_valid = bool(
        not fit_boundary_extrapolated
        and period_boundary_error <= max(0.0, float(max_period_boundary_error_fraction))
    )

    cycle_frame1 = peak_frame - equivalent_period_frames / 2.0 if period_valid else None
    cycle_frame2 = peak_frame + equivalent_period_frames / 2.0 if period_valid else None
    cycle_boundary_extrapolated = bool(
        period_valid
        and cycle_frame1 is not None
        and cycle_frame2 is not None
        and (
            cycle_frame1 < float(frame_values[0])
            or cycle_frame2 > float(frame_values[-1])
        )
    )

    cycle_position1 = None
    cycle_position2 = None
    cycle_delta_position = None
    if period_valid and cycle_frame1 is not None and cycle_frame2 is not None:
        baseline1 = left_anchor + baseline_slope * (cycle_frame1 - float(frame_values[lo]))
        baseline2 = left_anchor + baseline_slope * (cycle_frame2 - float(frame_values[lo]))
        # Both full-cycle endpoints have the same phase, opposite the selected apex.
        cycle_position1 = float(baseline1 - signed_amplitude)
        cycle_position2 = float(baseline2 - signed_amplitude)
        cycle_delta_position = float(cycle_position2 - cycle_position1)

    baseline_velocity = (
        cycle_delta_position / equivalent_period_s
        if cycle_delta_position is not None
        and equivalent_period_s is not None
        and equivalent_period_s > 0
        else None
    )
    angle = (
        float(np.degrees(np.arctan2(cycle_delta_position, equivalent_period_frames)))
        if cycle_delta_position is not None
        else None
    )

    metrics: Dict[str, Any] = {
        **fit_meta,
        "fit_method": "asymmetric_half_cosine_local_chord",
        "event_kind": normalized_kind,
        "direction": "negative" if normalized_kind == "min" else "positive",
        "peak_i": int(center_idx),
        "peak_frame": peak_frame,
        "peak_position_px": peak_position,
        "signed_amplitude_px": signed_amplitude,
        "amplitude_px": abs(signed_amplitude),
        "fit_window_source": "large_wave_local_endpoint_chord",
        "fit_window_width_frames": duration_frames,
        "fit_duration_frames": duration_frames,
        "fit_duration_s": duration_s,
        "fit_start_frame": float(frame_values[lo]),
        "fit_end_frame": float(frame_values[hi]),
        "fit_start_position_px": float(left_anchor),
        "fit_end_position_px": float(right_anchor),
        "baseline_start_position_px": float(left_anchor),
        "baseline_end_position_px": float(right_anchor),
        "baseline_slope_px_per_frame": float(baseline_slope),
        "baseline_velocity_px_per_s": baseline_velocity,
        "period_source": "equivalent_sinusoid_from_lobe" if period_valid else None,
        "period_estimate_valid": period_valid,
        "period_boundary_error_fraction": period_boundary_error,
        "left_quarter_period_frames": left_lobe_frames,
        "right_quarter_period_frames": right_lobe_frames,
        "period_from_left_frames": 4.0 * left_lobe_frames,
        "period_from_right_frames": 4.0 * right_lobe_frames,
        "period_asymmetry": abs(left_lobe_frames - right_lobe_frames) / duration_frames,
        "period_frames": equivalent_period_frames if period_valid else None,
        "period_s": equivalent_period_s if period_valid else None,
        "frequency_hz": equivalent_frequency_hz if period_valid else None,
        "fit_freq_hz": equivalent_frequency_hz if period_valid else None,
        "equivalent_cycle_frame1": cycle_frame1,
        "equivalent_cycle_frame2": cycle_frame2,
        "equivalent_cycle_position1_px": cycle_position1,
        "equivalent_cycle_position2_px": cycle_position2,
        "delta_pos_px": cycle_delta_position,
        "seconds_delta": equivalent_period_s if period_valid else None,
        "velocity_px_per_s": baseline_velocity,
        "wavelength_px": abs(cycle_delta_position) if cycle_delta_position is not None else None,
        "angle_from_time_axis_deg": angle,
        "width_half_prominence_frames": half_width_frames,
        "width_half_prominence_s": (
            half_width_frames / sampling_rate if sampling_rate > 0 else None
        ),
        "rise_time_s": (
            (peak_frame - float(frame_values[lo])) / sampling_rate if sampling_rate > 0 else None
        ),
        "recovery_time_s": (
            (float(frame_values[hi]) - peak_frame) / sampling_rate if sampling_rate > 0 else None
        ),
        "max_approach_speed_px_per_s": max_approach,
        "max_recovery_speed_px_per_s": max_recovery,
        "max_speed_px_per_s": max(max_approach, max_recovery),
        "apex_curvature_px_per_frame2": curvature,
        "integrated_displacement_px_s": integrated,
        "fit_boundary_extrapolated": fit_boundary_extrapolated,
        "equivalent_cycle_boundary_extrapolated": cycle_boundary_extrapolated,
        "boundary_extrapolated": cycle_boundary_extrapolated,
        "endpoint_anchor_rows": anchor_radius,
        "observed_peak_position_px": float(positions[center_idx]),
        "observed_peak_residual_px": signed_amplitude,
    }
    return LargeWaveFit(
        center_idx=int(center_idx),
        window_lo=lo,
        window_hi=hi,
        event_kind=normalized_kind,
        baseline=local_baseline,
        residual_fit=fitted_residual,
        fitted_position=fitted_position,
        metrics=metrics,
    )


def _select_multiscale_fit(
    fits: Sequence[LargeWaveFit],
    *,
    error_tolerance: float,
    max_error_vnmse: float,
) -> LargeWaveFit:
    """Choose maximum coverage among fits that remain close to the best error."""

    finite_errors = [
        float(fit.metrics["fit_error_vnmse"])
        for fit in fits
        if fit.metrics.get("fit_error_vnmse") is not None
        and math.isfinite(float(fit.metrics["fit_error_vnmse"]))
    ]
    best_error = min(finite_errors) if finite_errors else math.inf
    tolerance_limit = best_error + max(0.0, float(error_tolerance))
    absolute_limit = max(0.0, float(max_error_vnmse))
    coherent = [
        fit
        for fit in fits
        if fit.metrics.get("fit_error_vnmse") is not None
        and math.isfinite(float(fit.metrics["fit_error_vnmse"]))
        and float(fit.metrics["fit_error_vnmse"]) <= tolerance_limit
        and float(fit.metrics["fit_error_vnmse"]) <= absolute_limit
    ]
    if not coherent:
        return min(
            fits,
            key=lambda fit: (
                float(fit.metrics["fit_error_vnmse"])
                if fit.metrics.get("fit_error_vnmse") is not None
                and math.isfinite(float(fit.metrics["fit_error_vnmse"]))
                else math.inf,
                -float(fit.metrics.get("fit_window_width_frames", 0.0)),
            ),
        )
    return max(
        coherent,
        key=lambda fit: (
            float(fit.metrics.get("fit_window_width_frames", 0.0)),
            -float(fit.metrics["fit_error_vnmse"]),
        ),
    )


def _robust_endpoint_anchor(
    position: np.ndarray,
    global_baseline: np.ndarray,
    index: int,
    radius: int,
) -> float:
    lo = max(0, int(index) - radius)
    hi = min(len(position), int(index) + radius + 1)
    residual = np.asarray(position[lo:hi], dtype=float) - np.asarray(global_baseline[lo:hi], dtype=float)
    finite = residual[np.isfinite(residual)]
    offset = float(np.median(finite)) if finite.size else 0.0
    return float(global_baseline[index]) + offset


def _half_height_frame(*, boundary_frame: float, apex_frame: float, power: float, left: bool) -> float:
    safe_power = max(0.25, float(power))
    phase = (2.0 / math.pi) * math.acos(float(np.power(0.5, 1.0 / safe_power)))
    if left:
        return apex_frame - phase * (apex_frame - boundary_frame)
    return apex_frame + phase * (boundary_frame - apex_frame)


def _gradient(values: np.ndarray, frame: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=float)
    return np.gradient(np.asarray(values, dtype=float), np.asarray(frame, dtype=float))


def _fit_curvature(
    frame: np.ndarray,
    fitted_position: np.ndarray,
    center_idx: int,
    *,
    half_window: int,
) -> float:
    lo = max(0, center_idx - half_window)
    hi = min(len(frame), center_idx + half_window + 1)
    if hi - lo < 3:
        return 0.0
    x = np.asarray(frame[lo:hi], dtype=float) - float(frame[center_idx])
    y = np.asarray(fitted_position[lo:hi], dtype=float)
    try:
        coeff = np.polyfit(x, y, 2)
        return float(abs(2.0 * coeff[0]))
    except Exception:
        return 0.0


def _fit_quality(observed: np.ndarray, fitted: np.ndarray) -> Dict[str, Any]:
    observed_values = np.asarray(observed, dtype=float)
    fitted_values = np.asarray(fitted, dtype=float)
    valid = np.isfinite(observed_values) & np.isfinite(fitted_values)
    observed_valid = observed_values[valid]
    fitted_valid = fitted_values[valid]
    points = int(observed_valid.size)
    if points == 0:
        return {
            "fit_error_vnmse": None,
            "fit_r2": None,
            "fit_rmse_px": None,
            "fit_nrmse": None,
            "fit_mae_px": None,
            "fit_points": 0,
        }
    error = observed_valid - fitted_valid
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    variance = float(np.var(observed_valid))
    if points >= 2 and variance > 0:
        vnmse = mse / variance
        r2 = 1.0 - vnmse
        nrmse = math.sqrt(vnmse)
    else:
        vnmse = r2 = nrmse = None
    return {
        "fit_error_vnmse": vnmse,
        "fit_r2": r2,
        "fit_rmse_px": rmse,
        "fit_nrmse": nrmse,
        "fit_mae_px": mae,
        "fit_points": points,
    }
