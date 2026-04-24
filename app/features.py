# app/features.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------
# ID helpers
# -----------------------

def sample_id_from_name(name: str) -> int:
    h = 2166136261
    for ch in name:
        h ^= ord(ch)
        h *= 16777619
        h &= 0xFFFFFFFF
    return int(h % 10_000_000)


def coerce_track_id(stem: str) -> Optional[int]:
    try:
        return int(stem)
    except Exception:
        return None


# -----------------------
# JSON sanitization
# -----------------------

def _is_nan(x: Any) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def json_sanitize(obj: Any) -> Any:
    """
    Convert numpy scalars/arrays and NaN to JSON-friendly Python types.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        if isinstance(obj, float) and _is_nan(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if _is_nan(v) else v
    if isinstance(obj, (np.ndarray,)):
        return [json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    return str(obj)


def _finite_or_none(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _coord_height(coord_meta: Optional[dict]) -> Optional[float]:
    if not coord_meta:
        return None
    for key in ("output_height", "source_rows", "nrows"):
        value = coord_meta.get(key)
        if value in (None, ""):
            continue
        try:
            height = float(value)
        except Exception:
            continue
        if np.isfinite(height) and height > 0:
            return height
    return None


def map_heatmap_x(x: float, coord_meta: Optional[dict] = None) -> float:
    _ = coord_meta
    return float(x)


def map_heatmap_y(y: float, coord_meta: Optional[dict] = None) -> float:
    if not np.isfinite(y):
        return float(y)
    height = _coord_height(coord_meta)
    origin = str((coord_meta or {}).get("coord_origin", (coord_meta or {}).get("origin", "upper"))).lower()
    if height is None:
        return float(y)
    if origin == "lower":
        return float((height - 1.0) - float(y))
    return float(y)


# -----------------------
# Geometry & descriptors
# -----------------------

def segment_bbox(x: np.ndarray, y: np.ndarray, i: int, j: int) -> Tuple[float, float, float, float]:
    if j < i:
        i, j = j, i
    xs = x[i : j + 1]
    ys = y[i : j + 1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def orientation_deg(x_seg: np.ndarray, y_seg: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x_seg).ravel()
    y = np.asarray(y_seg).ravel()
    if x.size < 2:
        return 0.0, 0.0
    try:
        a, _b = np.polyfit(x, y, deg=1)
        slope = float(a)
    except Exception:
        slope = 0.0
    angle = float(np.degrees(np.arctan(np.abs(slope))))
    dx = np.diff(x)
    dy = np.diff(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_slopes = np.where(np.abs(dx) > 0, dy / dx, 0.0)
    angle_std = float(np.degrees(np.std(np.arctan(np.abs(local_slopes))))) if local_slopes.size else 0.0
    return angle, angle_std


# -----------------------
# Bulge metrics from find_peaks props
# -----------------------

def _peak_prop_at_index(peaks_idx: np.ndarray, props: Dict[str, np.ndarray], peak_i: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if peaks_idx is None or props is None:
        return out
    peaks_idx = np.asarray(peaks_idx, dtype=int)
    try:
        pos = int(np.where(peaks_idx == peak_i)[0][0])
    except Exception:
        return out
    for k, v in props.items():
        try:
            out[k] = float(np.asarray(v)[pos])
        except Exception:
            pass
    return out


def bulge_from_props(
    peak_i: int,
    peaks_idx: np.ndarray,
    props: Dict[str, np.ndarray],
    sampling_rate: float,
) -> Dict[str, float]:
    md = _peak_prop_at_index(peaks_idx, props, peak_i)
    prom = float(md.get("prominences", np.nan))
    width_frames = float(md.get("widths", np.nan))
    width_s = (width_frames / sampling_rate) if (sampling_rate and np.isfinite(width_frames)) else np.nan
    return {
        "bulge_prominence_px": prom,
        "bulge_width_frames": width_frames,
        "bulge_width_s": width_s,
    }


# -----------------------
# Anchored sine fit (around a peak)
# -----------------------

def _fit_anchored_sine(
    residual: np.ndarray,
    t: np.ndarray,
    freq: float,
    sampling_rate: float,
    center_idx: int,
    fit_lo: int,
    fit_hi: int,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Fit y ~= A*sin(omega t + phi) + c with the sine maximum anchored at
    center_idx and constrained to pass through the detected peak.
    """
    omega = 2.0 * np.pi * float(freq) / float(sampling_rate)
    t0 = float(t[int(center_idx)])
    phi = (np.pi / 2.0) - omega * t0
    s = np.sin(omega * t + phi).astype(np.float64)

    peak_value = float(residual[int(center_idx)])
    lo = int(max(0, fit_lo))
    hi = int(min(len(t) - 1, fit_hi))
    z = s[lo : hi + 1] - 1.0
    target = residual[lo : hi + 1].astype(np.float64) - peak_value
    denom = float(np.dot(z, z))
    if denom > 0 and np.isfinite(denom):
        A = float(np.dot(z, target) / denom)
    else:
        A = peak_value
    c = float(peak_value - A)
    yfit = (A * s + c).astype(np.float64)
    return yfit, float(A), float(phi), float(c)


def anchored_sine_params(
    residual: np.ndarray,
    x: np.ndarray,
    sampling_rate: float,
    freq: float,
    center_idx: int,
    period_frac: float = 0.5,
) -> Dict[str, float]:
    out = {
        "fit_amp_A": np.nan,
        "fit_phase_phi": np.nan,
        "fit_offset_c": np.nan,
        "fit_freq_hz": float(freq) if freq is not None else np.nan,
        "fit_error_vnmse": np.nan,
        "fit_window_lo": np.nan,
        "fit_window_hi": np.nan,
    }
    if sampling_rate is None or sampling_rate <= 0 or freq is None or freq <= 0 or center_idx < 0 or center_idx >= len(x):
        return out

    frames_per_period = sampling_rate / float(freq)
    half_span = max(1, int(round((period_frac * frames_per_period) / 2.0)))
    lo = max(0, int(center_idx) - half_span)
    hi = min(len(x) - 1, int(center_idx) + half_span)

    yfit_res, A, phi, c = _fit_anchored_sine(
        residual=residual,
        t=x,
        freq=float(freq),
        sampling_rate=sampling_rate,
        center_idx=int(center_idx),
        fit_lo=lo,
        fit_hi=hi,
    )

    y_slice = residual[lo : hi + 1]
    y_fit = yfit_res[lo : hi + 1]
    if y_slice.size >= 2 and np.var(y_slice) > 0:
        vnmse = float(np.mean((y_slice - y_fit) ** 2) / np.var(y_slice))
    else:
        vnmse = np.nan

    out.update({
        "fit_amp_A": float(A),
        "fit_phase_phi": float(phi),
        "fit_offset_c": float(c),
        "fit_error_vnmse": float(vnmse),
        "fit_window_lo": float(lo),
        "fit_window_hi": float(hi),
        "fit_peak_value": float(residual[int(center_idx)]),
        "fit_peak_error": float(yfit_res[int(center_idx)] - residual[int(center_idx)]),
        "fit_passes_peak": True,
    })
    return out


# -----------------------
# Type heuristic
# -----------------------

def classify_wave_type(angle_deg: float, prominence_px: float, cfg: Optional[dict] = None) -> Tuple[str, float]:
    cfg = cfg or {}
    ripple_max = float(cfg.get("ripple_max_deg", 10.0))
    surf_min = float(cfg.get("surf_min_deg", 20.0))
    prom_min = float(cfg.get("prominence_min_px", 1.0))

    if np.isfinite(angle_deg) and np.isfinite(prominence_px):
        if angle_deg <= ripple_max and prominence_px >= prom_min:
            score = float(max(0.0, min(1.0, (prominence_px / (prom_min + 1e-6)) * (1.0 - angle_deg / (ripple_max + 1e-6)))))
            return "ripple", min(1.0, score)
        if angle_deg >= surf_min:
            score = float(max(0.0, min(1.0, (angle_deg - surf_min) / (90.0 - surf_min))))
            return "surf", score

    return "ambiguous", 0.5


# -----------------------
# Row builders
# -----------------------

def _local_period_frames_from_peaks(peaks_idx: np.ndarray, k: int, frame: Optional[np.ndarray] = None) -> Optional[float]:
    p = np.asarray(peaks_idx, dtype=int)
    if p.size == 0 or k < 0 or k >= p.size:
        return None
    frame_values = np.asarray(frame, dtype=float) if frame is not None else None

    def peak_frame(idx: int) -> float:
        if frame_values is None:
            return float(p[idx])
        return float(frame_values[p[idx]])

    gaps: List[float] = []
    if k - 1 >= 0:
        gaps.append(abs(peak_frame(k) - peak_frame(k - 1)))
    if k + 1 < p.size:
        gaps.append(abs(peak_frame(k + 1) - peak_frame(k)))
    if not gaps:
        return None
    return float(np.median(gaps))


def _interp_position_at_frame(frame: np.ndarray, position: np.ndarray, target_frame: float) -> float:
    if not np.isfinite(target_frame) or frame.size == 0:
        return np.nan
    order = np.argsort(frame, kind="stable")
    frame_sorted = np.asarray(frame, dtype=float)[order]
    position_sorted = np.asarray(position, dtype=float)[order]
    return float(np.interp(float(target_frame), frame_sorted, position_sorted))


def build_peak_rows(
    *,
    frame: np.ndarray,
    position: np.ndarray,
    residual: np.ndarray,
    peaks_idx: np.ndarray,
    peak_props: dict,
    sampling_rate: float,
    sample: str,
    track_stem: str,
    features_cfg: Optional[dict] = None,
    global_freq_hz: float | None = None,
    period_frac_for_fit: float = 0.5,
    coord_meta: Optional[dict] = None,
) -> List[dict]:
    rows: List[dict] = []
    features_cfg = features_cfg or {}

    p = np.asarray(peaks_idx, dtype=int)
    if p.size == 0:
        return rows

    sample_id = sample_id_from_name(sample)
    maybe_track_id = coerce_track_id(track_stem)

    global_fpp = (sampling_rate / float(global_freq_hz)) if (sampling_rate and global_freq_hz and global_freq_hz > 0) else None

    for idx_in_list, peak_i in enumerate(p):
        frame_value_img = float(frame[peak_i])
        pos_px_img = float(position[peak_i])
        frame_value = map_heatmap_y(frame_value_img, coord_meta)
        pos_px = map_heatmap_x(pos_px_img, coord_meta)
        amp = float(residual[peak_i])

        local_fpp = _local_period_frames_from_peaks(p, idx_in_list, frame)
        frames_per_period = local_fpp if (local_fpp and local_fpp > 0) else (global_fpp if (global_fpp and global_fpp > 0) else np.nan)
        period_frames = float(frames_per_period) if np.isfinite(frames_per_period) else np.nan
        period_s = (period_frames / sampling_rate) if (sampling_rate and np.isfinite(period_frames)) else np.nan
        freq_hz = (1.0 / period_s) if (np.isfinite(period_s) and period_s > 0) else (float(global_freq_hz) if (global_freq_hz and global_freq_hz > 0) else np.nan)

        bulge = bulge_from_props(int(peak_i), p, peak_props or {}, sampling_rate)

        fit = anchored_sine_params(
            residual=residual,
            x=frame,
            sampling_rate=sampling_rate,
            freq=freq_hz if (np.isfinite(freq_hz) and freq_hz > 0) else (global_freq_hz or np.nan),
            center_idx=int(peak_i),
            period_frac=float(features_cfg.get("fit_window_period_frac", period_frac_for_fit)),
        )

        lo = fit.get("fit_window_lo", np.nan)
        hi = fit.get("fit_window_hi", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            i0, i1 = int(max(0, lo)), int(min(len(frame) - 1, hi))
            ang_mean, ang_std = orientation_deg(frame[i0 : i1 + 1], position[i0 : i1 + 1])
        else:
            ang_mean, ang_std = (np.nan, np.nan)

        x_px = int(round(pos_px)) if np.isfinite(pos_px) else None
        y_px = int(round(frame_value)) if np.isfinite(frame_value) else None

        metrics = {
            "sample": sample,
            "sample_id": int(sample_id),
            "track_stem": track_stem,
            "track_id_hint": int(maybe_track_id) if maybe_track_id is not None else None,
            "peak_index": int(idx_in_list + 1),
            "peak_i": int(peak_i),
            "frame": frame_value,
            "pos_px": pos_px,
            "x_px": x_px,
            "y_px": y_px,
            "local_period_frames": period_frames,
            "local_period_s": period_s,
            "local_freq_hz": freq_hz,
            "orientation_deg": ang_mean,
            "orientation_std_deg": ang_std,
            **bulge,
            **fit,
        }

        rows.append({
            "pos": frame_value,                   # Peak.pos
            "value": amp,                         # Peak.value
            "metrics": json_sanitize(metrics),    # Peak.metrics (JSONB)
        })

    return rows


def build_wave_rows(
    *,
    frame: np.ndarray,
    position: np.ndarray,
    residual: np.ndarray,
    peaks_idx: np.ndarray,
    peak_props: dict,
    sampling_rate: float,
    sample: str,
    track_stem: str,
    features_cfg: Optional[dict] = None,
    freq_hz: float | None = None,
    period_frac_for_fit: float = 0.5,
    coord_meta: Optional[dict] = None,
) -> List[dict]:
    features_cfg = features_cfg or {}
    rows: List[dict] = []

    p = np.asarray(peaks_idx, dtype=int)
    if p.size == 0:
        return rows

    sample_id = sample_id_from_name(sample)
    maybe_track_id = coerce_track_id(track_stem)
    global_fpp = (sampling_rate / float(freq_hz)) if (sampling_rate and freq_hz and freq_hz > 0) else None
    frame_min = float(np.nanmin(frame)) if frame.size else np.nan
    frame_max = float(np.nanmax(frame)) if frame.size else np.nan

    for k, peak_i_raw in enumerate(p):
        peak_i = int(peak_i_raw)
        prev_i = int(p[k - 1]) if k - 1 >= 0 else None
        next_i = int(p[k + 1]) if k + 1 < p.size else None

        peak_frame_img = float(frame[peak_i])
        peak_pos_img = float(position[peak_i])
        peak_frame = map_heatmap_y(peak_frame_img, coord_meta)
        peak_pos = map_heatmap_x(peak_pos_img, coord_meta)
        period_est = _local_period_frames_from_peaks(p, k, frame)
        if not (period_est and period_est > 0) and global_fpp and global_fpp > 0:
            period_est = float(global_fpp)

        if period_est and np.isfinite(period_est) and period_est > 0:
            if prev_i is not None:
                frame1 = (float(frame[prev_i]) + peak_frame) / 2.0
            else:
                frame1 = peak_frame - (float(period_est) / 2.0)

            if next_i is not None:
                frame2 = (peak_frame + float(frame[next_i])) / 2.0
            else:
                frame2 = peak_frame + (float(period_est) / 2.0)
        else:
            frame1 = peak_frame
            frame2 = peak_frame

        if frame2 < frame1:
            frame1, frame2 = frame2, frame1

        period_frames = frame2 - frame1
        period_s = (period_frames / sampling_rate) if sampling_rate else float("nan")
        freq = (1.0 / period_s) if (np.isfinite(period_s) and period_s > 0) else (float(freq_hz) if (freq_hz and freq_hz > 0) else np.nan)

        pos1 = map_heatmap_x(_interp_position_at_frame(frame, position, frame1), coord_meta)
        pos2 = map_heatmap_x(_interp_position_at_frame(frame, position, frame2), coord_meta)

        frame1_coord = map_heatmap_y(frame1, coord_meta)
        frame2_coord = map_heatmap_y(frame2, coord_meta)

        amp = float(residual[peak_i])

        dpos = pos2 - pos1
        vel = (dpos / period_s) if (np.isfinite(period_s) and period_s != 0) else float("nan")
        wavelength = float(abs(dpos))

        mask = (frame >= frame1) & (frame <= frame2)
        if int(np.count_nonzero(mask)) >= 2:
            ang_mean, ang_std = orientation_deg(frame[mask], position[mask])
            frame_seg = frame[mask]
            pos_seg = position[mask]
        else:
            lo_i = max(0, peak_i - 1)
            hi_i = min(len(frame) - 1, peak_i + 1)
            ang_mean, ang_std = orientation_deg(frame[lo_i : hi_i + 1], position[lo_i : hi_i + 1])
            frame_seg = frame[lo_i : hi_i + 1]
            pos_seg = position[lo_i : hi_i + 1]
        xmin = float(np.nanmin(pos_seg)) if pos_seg.size else np.nan
        xmax = float(np.nanmax(pos_seg)) if pos_seg.size else np.nan
        ymin = float(np.nanmin(frame_seg)) if frame_seg.size else np.nan
        ymax = float(np.nanmax(frame_seg)) if frame_seg.size else np.nan
        ymin_coord = map_heatmap_y(ymax, coord_meta)
        ymax_coord = map_heatmap_y(ymin, coord_meta)

        bulge = bulge_from_props(peak_i, p, peak_props or {}, sampling_rate)

        fit = anchored_sine_params(
            residual=residual,
            x=frame,
            sampling_rate=sampling_rate,
            freq=freq if np.isfinite(freq) else (freq_hz or np.nan),
            center_idx=peak_i,
            period_frac=float(features_cfg.get("fit_window_period_frac", period_frac_for_fit)),
        )

        wlabel, wscore = classify_wave_type(
            angle_deg=ang_mean,
            prominence_px=float(bulge.get("bulge_prominence_px", np.nan)),
            cfg=features_cfg.get("classify", {}),
        )

        # Click point in heatmap coords (x_px = column, y_px = row).
        x_px = int(round(peak_pos)) if np.isfinite(peak_pos) else None
        y_px = int(round(peak_frame)) if np.isfinite(peak_frame) else None

        # Time window for this wave (best-effort)
        t_start = (frame1 / sampling_rate) if (sampling_rate and np.isfinite(frame1)) else None
        t_end = (frame2 / sampling_rate) if (sampling_rate and np.isfinite(frame2)) else None
        seconds_delta = (t_end - t_start) if (t_start is not None and t_end is not None) else float("nan")
        boundary_extrapolated = bool(
            np.isfinite(frame_min)
            and np.isfinite(frame_max)
            and (frame1 < frame_min or frame2 > frame_max)
        )

        metrics = {
            "sample": sample,
            "sample_id": int(sample_id),
            "track_stem": track_stem,
            "track_id_hint": int(maybe_track_id) if maybe_track_id is not None else None,
            "wave_index": int(k + 1),
            "peak_i": int(peak_i),
            "peak_index": int(k + 1),
            "peak_count": 1,
            "has_peak": True,
            "previous_peak_i": int(prev_i) if prev_i is not None else None,
            "next_peak_i": int(next_i) if next_i is not None else None,
            "peak_frame_y_axis": peak_frame,
            "peak_position_x_axis": peak_pos,
            "frame1": frame1_coord,
            "frame2": frame2_coord,
            "period_frames": period_frames,
            "period_s": period_s,
            "frequency_hz": freq,
            "pos1_px": pos1,
            "pos2_px": pos2,
            "amplitude_px": amp,
            "delta_pos_px": dpos,
            "seconds_delta": seconds_delta,
            "velocity_px_per_s": vel,
            "wavelength_px": wavelength,
            "bbox": {"xmin": xmin, "xmax": xmax, "ymin": min(ymin_coord, ymax_coord), "ymax": max(ymin_coord, ymax_coord)},
            "boundary_extrapolated": boundary_extrapolated,
            "orientation_deg": ang_mean,
            "orientation_std_deg": ang_std,
            "wave_type": wlabel,
            "type_score": float(wscore),
            **bulge,
            **fit,
            "legacy": {
                "Sample": sample,
                "Track": maybe_track_id if maybe_track_id is not None else track_stem,
                "Wave number": int(k + 1),
                "Frame position 1": frame1_coord,
                "Frame position 2": frame2_coord,
                "Period (frames)": period_frames,
                "Period (s)": period_s,
                "Frequency (Hz)": freq,
                "Pixel position 1": pos1,
                "Pixel position 2": pos2,
                "Amplitude (pixels)": amp,
                "Peak frame": peak_frame,
                "Peak position": peak_pos,
                "Frame 1 (seconds)": t_start,
                "Frame 2 (seconds)": t_end,
                "Seconds 2 - Seconds 1": seconds_delta,
                "Δposition (px)": dpos,
                "Velocity (px/s)": vel,
                "Wavelength (px)": wavelength,
            },
        }

        rows.append({
            "wave_index": int(k + 1),                         # Wave.wave_index
            "x": x_px,                                         # Wave.x (heatmap col)
            "y": y_px,                                         # Wave.y (heatmap row)
            "amplitude": _finite_or_none(amp),                 # Wave.amplitude
            "frequency": _finite_or_none(freq),                # Wave.frequency
            "period": _finite_or_none(period_s),               # Wave.period
            "error": _finite_or_none(fit.get("fit_error_vnmse")),  # Wave.error
            "t_start": _finite_or_none(t_start),               # Wave.t_start
            "t_end": _finite_or_none(t_end),                   # Wave.t_end
            "metrics": json_sanitize(metrics),                 # Wave.metrics (JSONB)
        })

    return rows
