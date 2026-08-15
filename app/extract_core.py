# app/extract_core.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple
from uuid import UUID

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .signal.detrend import detrend_residual
from .signal.peaks import detect_peaks, detect_peaks_adaptive, ensure_minimum_peaks
from .signal.period import estimate_dominant_frequency, frequency_to_period, resolve_positive_frequency
from .features import build_wave_rows, build_peak_rows, map_heatmap_x, map_heatmap_y


PEAK_POLARITY_ALIASES = {
    "max": "maxima",
    "maximum": "maxima",
    "maxima": "maxima",
    "positive": "maxima",
    "peak": "maxima",
    "peaks": "maxima",
    "min": "minima",
    "minimum": "minima",
    "minima": "minima",
    "negative": "minima",
    "trough": "minima",
    "troughs": "minima",
    "both": "both",
    "all": "both",
}


ENDPOINT_LINK_LEVELS: Dict[str, Dict[str, Any]] = {
    "minimal": {
        "max_gap_rows": 12,
        "max_dx": 3.0,
        "min_bridge_prob": 0.16,
        "max_slope_delta": 0.25,
        "fit_rows": 10,
        "max_conflict_fraction": 0.02,
        "insert_bridge_points": True,
        "overlap_enabled": True,
        "min_overlap_rows": 8,
        "max_overlap_rows": 20,
        "overlap_dx_tol": 1.5,
    },
    "conservative": {
        "max_gap_rows": 22,
        "max_dx": 4.0,
        "min_bridge_prob": 0.14,
        "max_slope_delta": 0.32,
        "fit_rows": 10,
        "max_conflict_fraction": 0.06,
        "insert_bridge_points": True,
        "overlap_enabled": True,
        "min_overlap_rows": 7,
        "max_overlap_rows": 30,
        "overlap_dx_tol": 2.0,
    },
    "balanced": {
        "max_gap_rows": 28,
        "max_dx": 5.0,
        "min_bridge_prob": 0.12,
        "max_slope_delta": 0.38,
        "fit_rows": 12,
        "max_conflict_fraction": 0.10,
        "insert_bridge_points": True,
        "overlap_enabled": True,
        "min_overlap_rows": 6,
        "max_overlap_rows": 38,
        "overlap_dx_tol": 2.5,
    },
    "maximal": {
        "max_gap_rows": 35,
        "max_dx": 6.0,
        "min_bridge_prob": 0.10,
        "max_slope_delta": 0.45,
        "fit_rows": 12,
        "max_conflict_fraction": 0.15,
        "insert_bridge_points": True,
        "overlap_enabled": True,
        "min_overlap_rows": 5,
        "max_overlap_rows": 45,
        "overlap_dx_tol": 3.0,
    },
    "aggressive": {
        "max_gap_rows": 50,
        "max_dx": 8.0,
        "min_bridge_prob": 0.08,
        "max_slope_delta": 0.60,
        "fit_rows": 14,
        "max_conflict_fraction": 0.22,
        "insert_bridge_points": True,
        "overlap_enabled": True,
        "min_overlap_rows": 4,
        "max_overlap_rows": 65,
        "overlap_dx_tol": 4.0,
    },
    "experimental": {
        "max_gap_rows": 75,
        "max_dx": 10.0,
        "min_bridge_prob": 0.06,
        "max_slope_delta": 0.80,
        "fit_rows": 16,
        "max_conflict_fraction": 0.35,
        "insert_bridge_points": True,
        "overlap_enabled": True,
        "min_overlap_rows": 3,
        "max_overlap_rows": 90,
        "overlap_dx_tol": 5.0,
    },
}

ENDPOINT_LINK_LEVEL_ALIASES = {
    "low": "minimal",
    "small": "minimal",
    "medium": "balanced",
    "normal": "balanced",
    "default": "maximal",
    "high": "maximal",
    "max": "maximal",
    "extra": "aggressive",
    "very_high": "aggressive",
    "very-high": "aggressive",
    "beyond": "aggressive",
    "extreme": "experimental",
    "maximum": "experimental",
}

ENDPOINT_LINK_SHARED_DEFAULTS: Dict[str, Any] = {
    "max_chord_slope_px_per_row": 2.0,
    "max_step_dx_px_per_row": 4.0,
    "max_manifest_rejections": 500,
}


# -----------------------------
# Types
# -----------------------------

@dataclass(frozen=True)
class KymoOutput:
    image_id: str
    base_dir: Path
    track_paths: List[Path]


class KymoRunner(Protocol):
    def run(
        self,
        *,
        heatmap_path: Path,
        scratch_dir: Path,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> KymoOutput: ...


# -----------------------------
# Kymo runner selection
# -----------------------------

def select_kymo_runner(*, config: Dict[str, Any]) -> KymoRunner:
    kymo_cfg = (config.get("kymo") or {})
    backend = str(kymo_cfg.get("backend", "onnx")).lower()
    if backend == "wolfram":
        return WolframKymoRunner(config=config)
    return OnnxKymoRunner(config=config)


@dataclass
class OnnxKymoRunner:
    config: Dict[str, Any]

    def run(
        self,
        *,
        heatmap_path: Path,
        scratch_dir: Path,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> KymoOutput:
        from .modules.kb_adapter import run_kymobutler as run_kymo

        image_id = _image_id_from_path(heatmap_path)
        base_dir = scratch_dir / image_id
        base_dir.mkdir(parents=True, exist_ok=True)

        onnx_cfg = ((self.config.get("kymo") or {}).get("onnx") or {})
        analysis_cfg = (self.config.get("analysis") or {})

        export_dir = onnx_cfg.get("export_dir", None)
        providers = _parse_providers(onnx_cfg.get("providers", None))

        debug_cfg = (onnx_cfg.get("debug") or {})
        debug_save_images = bool(debug_cfg.get("save_debug_images", True))
        save_overlay_tracks = bool(debug_cfg.get("save_overlay_tracks", True))

        run_kymo(
            str(heatmap_path),
            output_dir=str(base_dir),
            export_dir=export_dir,
            providers=providers,
            debug_save_images=debug_save_images,
            save_overlay_tracks=save_overlay_tracks,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            **_flatten_onnx_cfg_for_runner(onnx_cfg, analysis_cfg=analysis_cfg),
        )

        track_paths = _discover_tracks(base_dir)
        return KymoOutput(image_id=image_id, base_dir=base_dir, track_paths=track_paths)


@dataclass
class WolframKymoRunner:
    config: Dict[str, Any]

    def run(
        self,
        *,
        heatmap_path: Path,
        scratch_dir: Path,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> KymoOutput:
        from .modules.kymo_interface import run_kymobutler as run_kymo

        image_id = _image_id_from_path(heatmap_path)
        base_dir = scratch_dir / image_id
        base_dir.mkdir(parents=True, exist_ok=True)

        kymo_cfg = (self.config.get("kymo") or {})
        wolfram_cfg = (kymo_cfg.get("wolfram") or {})
        min_length = int(wolfram_cfg.get("min_length", kymo_cfg.get("min_length", 30)))
        verbose = bool(wolfram_cfg.get("verbose", kymo_cfg.get("verbose", False)))
        scripts_dir = wolfram_cfg.get("scripts_dir", None)
        executable = str(wolfram_cfg.get("executable", "wolframscript"))
        script_name = str(wolfram_cfg.get("script_name", "Run_Kymobutler.wls"))

        run_kymo(
            str(heatmap_path),
            output_dir=str(base_dir),
            scripts_dir=scripts_dir,
            executable=executable,
            script_name=script_name,
            min_length=min_length,
            verbose=verbose,
            progress_cb=progress_cb,
        )

        track_paths = _discover_tracks(base_dir)
        return KymoOutput(image_id=image_id, base_dir=base_dir, track_paths=track_paths)


def _parse_providers(value: Any) -> Optional[Iterable[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        return [p.strip() for p in s.split(",") if p.strip()]
    return [str(value)]


def _discover_tracks(base_dir: Path) -> List[Path]:
    out_dir = base_dir / "kymobutler_output"
    if not out_dir.exists():
        return []
    return sorted(out_dir.glob("*.npy"))


def _image_id_from_path(p: Path) -> str:
    stem = p.stem
    if stem.endswith("_heatmap"):
        return stem[:-8]
    return stem


def _flatten_onnx_cfg_for_runner(
    onnx_cfg: Dict[str, Any],
    *,
    analysis_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    thresholds = (onnx_cfg.get("thresholds") or {})
    hyst = (onnx_cfg.get("hysteresis") or {})
    auto = (onnx_cfg.get("auto_threshold") or {})
    morph = (onnx_cfg.get("morphology") or {})
    comp = (onnx_cfg.get("components") or {})
    skel = (onnx_cfg.get("skeleton") or {})
    post = (onnx_cfg.get("postproc") or {})
    endpoint_link = (post.get("endpoint_link") or {})
    endpoint_link_resolved = _resolve_endpoint_link_cfg(endpoint_link)
    dedupe = (post.get("dedupe") or {})
    tracking = (onnx_cfg.get("tracking") or {})
    analysis_cfg = analysis_cfg or {}
    ripple_cfg = (analysis_cfg.get("ripple") or {})
    ripple_linking = (ripple_cfg.get("endpoint_link") or {})
    large_wave_cfg = (analysis_cfg.get("large_wave") or {})
    large_wave_linking = (large_wave_cfg.get("endpoint_link") or {})
    analysis_mode = str(analysis_cfg.get("mode", "standard")).strip().lower()
    ripple_mode = analysis_mode in {"ripple", "ripple_family", "ripple_waves", "ripple-wave", "ripple-waves"}
    large_wave_mode = analysis_mode in {
        "large",
        "large_wave",
        "large_waves",
        "large-wave",
        "large-waves",
        "curved_wave",
        "curved_waves",
    }

    return {
        "min_length": int(tracking.get("min_length", 30)),
        "seg_size": int(onnx_cfg.get("seg_size", 256)),
        "force_mode": onnx_cfg.get("force_mode", None),
        "thr": float(thresholds.get("thr_default", 0.20)),
        "thr_bi": thresholds.get("thr_bi", None),
        "thr_uni": thresholds.get("thr_uni", None),
        "auto_threshold": bool(auto.get("enabled", True)),
        "auto_sweep": tuple(auto.get("sweep", [0.12, 0.30, 19])),
        "auto_target_pct": tuple(auto.get("target_mask_pct", [15.0, 25.0])),
        "auto_trigger_pct": tuple(auto.get("trigger_pct", [5.0, 35.0])),
        "hysteresis_enable": bool(hyst.get("enabled", True)),
        "hysteresis_low": float(hyst.get("low", 0.08)),
        "hysteresis_high": float(hyst.get("high", 0.18)),
        "morph_mode": str(morph.get("mode", "directional")),
        "classic_kernel": int(morph.get("classic_kernel", 3)),
        "dir_kv": int(morph.get("dir_kv", 5)),
        "dir_kh": int(morph.get("dir_kh", 5)),
        "diag_bridge": bool(morph.get("diag_bridge", True)),
        "weak_shave_enable": bool(morph.get("weak_shave_enable", True)),
        "weak_shave_p": float(morph.get("weak_shave_p", 0.12)),
        "comp_min_px": int(comp.get("min_px", 5)),
        "comp_min_rows": int(comp.get("min_rows", 5)),
        "skel_keep_ratio": float(skel.get("keep_ratio", 0.60)),
        "skel_keep_min_px": int(skel.get("keep_min_px", 2000)),
        "skel_prob_floor_min": float(skel.get("prob_floor_min", 0.06)),
        "skel_prob_floor_max": float(skel.get("prob_floor_max", 0.10)),
        "prune_iters": int(skel.get("prune_iters", 0)),
        "extend_rows": int(post.get("extend_rows", 22)),
        "dx_win": int(post.get("dx_win", 4)),
        "refine_prob_min": float(post.get("prob_min", 0.11)),
        "max_gap_rows": int(post.get("max_gap_rows", 13)),
        "max_dx": int(post.get("max_dx", 6)),
        "prob_bridge_min": float(post.get("prob_bridge_min", 0.11)),
        "endpoint_link_enable": bool(endpoint_link_resolved["enabled"]),
        "endpoint_link_max_gap_rows": int(endpoint_link_resolved["max_gap_rows"]),
        "endpoint_link_max_dx": float(endpoint_link_resolved["max_dx"]),
        "endpoint_link_min_bridge_prob": float(endpoint_link_resolved["min_bridge_prob"]),
        "endpoint_link_max_slope_delta": float(endpoint_link_resolved["max_slope_delta"]),
        "endpoint_link_fit_rows": int(endpoint_link_resolved["fit_rows"]),
        "endpoint_link_max_conflict_fraction": float(endpoint_link_resolved["max_conflict_fraction"]),
        "endpoint_link_insert_bridge_points": bool(endpoint_link_resolved["insert_bridge_points"]),
        "endpoint_link_overlap_enabled": bool(endpoint_link_resolved["overlap_enabled"]),
        "endpoint_link_min_overlap_rows": int(endpoint_link_resolved["min_overlap_rows"]),
        "endpoint_link_max_overlap_rows": int(endpoint_link_resolved["max_overlap_rows"]),
        "endpoint_link_overlap_dx_tol": float(endpoint_link_resolved["overlap_dx_tol"]),
        "endpoint_link_max_chord_slope_px_per_row": float(
            endpoint_link_resolved["max_chord_slope_px_per_row"]
        ),
        "endpoint_link_max_step_dx_px_per_row": float(
            endpoint_link_resolved["max_step_dx_px_per_row"]
        ),
        "endpoint_link_max_manifest_rejections": int(
            endpoint_link_resolved["max_manifest_rejections"]
        ),
        "endpoint_link_prefer_long_linear": bool(
            ripple_mode and ripple_linking.get("prefer_long_linear", True)
        ),
        "endpoint_link_length_weight": float(ripple_linking.get("length_weight", 0.25)),
        "endpoint_link_linearity_weight": float(ripple_linking.get("linearity_weight", 0.25)),
        "endpoint_link_min_abs_slope": float(ripple_linking.get("min_abs_slope", 0.05)),
        "endpoint_link_prefer_smooth_curves": bool(
            large_wave_mode and large_wave_linking.get("prefer_smooth_curves", True)
        ),
        "endpoint_link_curve_length_weight": float(large_wave_linking.get("length_weight", 0.30)),
        "endpoint_link_curve_tangent_weight": float(large_wave_linking.get("tangent_weight", 0.25)),
        "endpoint_link_curve_curvature_weight": float(large_wave_linking.get("curvature_weight", 0.35)),
        "endpoint_link_curve_max_turn_deg": float(large_wave_linking.get("max_turn_deg", 120.0)),
        "endpoint_link_curve_max_curvature": float(
            large_wave_linking.get("max_curvature_px_per_row2", 0.35)
        ),
        "dedupe_enable": bool(dedupe.get("enabled", True)),
        "dedupe_min_rows": int(dedupe.get("min_rows", 30)),
        "dedupe_min_score": float(dedupe.get("min_score", 0.11)),
        "dedupe_overlap_iou": float(dedupe.get("overlap_iou", 0.80)),
        "dedupe_dx_tol": float(dedupe.get("dx_tol", 2.5)),
        "fuse_uni_into_bi": bool(onnx_cfg.get("fuse_uni_into_bi", True)),
        "fuse_uni_weight": float(onnx_cfg.get("fuse_uni_weight", 0.7)),
    }


def _resolve_endpoint_link_level(value: Any) -> str:
    key = str(value or "maximal").strip().lower().replace(" ", "_")
    key = ENDPOINT_LINK_LEVEL_ALIASES.get(key, key)
    if key not in ENDPOINT_LINK_LEVELS:
        allowed = ", ".join(sorted(ENDPOINT_LINK_LEVELS))
        raise ValueError(f"Unknown endpoint_link.level {value!r}. Expected one of: {allowed}")
    return key


def _resolve_endpoint_link_cfg(endpoint_link: Dict[str, Any]) -> Dict[str, Any]:
    level = _resolve_endpoint_link_level(endpoint_link.get("level", "maximal"))
    resolved = dict(ENDPOINT_LINK_LEVELS[level])
    resolved["level"] = level
    resolved["enabled"] = bool(endpoint_link.get("enabled", False))
    for key in ENDPOINT_LINK_LEVELS["maximal"]:
        if key in endpoint_link:
            resolved[key] = endpoint_link[key]
    for key, default in ENDPOINT_LINK_SHARED_DEFAULTS.items():
        resolved[key] = endpoint_link.get(key, default)
    return resolved


# -----------------------------
# Track processing
# -----------------------------

def process_track(
    *,
    job_id: UUID,
    track_index: int,
    track_path: Path,
    config: Dict[str, Any],
    heatmap_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    kymo_cfg = (config.get("kymo") or {})
    backend = str(kymo_cfg.get("backend", "onnx")).lower()
    track_xy_order = str(kymo_cfg.get("track_xy_order", "auto")).lower()
    if track_xy_order == "auto":
        # ONNX kymobutler saves (y, x) points; Wolfram typically outputs (x, y).
        track_xy_order = "yx" if backend == "onnx" else "xy"

    frame, position = load_track_frame_position(track_path, order=track_xy_order)
    return process_track_arrays(
        job_id=job_id,
        track_index=track_index,
        track_stem=track_path.stem,
        sample=_infer_sample(track_path),
        frame=frame,
        position=position,
        config=config,
        heatmap_meta=heatmap_meta,
    )


def process_track_arrays(
    *,
    job_id: UUID,
    track_index: int,
    track_stem: str,
    sample: str,
    frame: np.ndarray,
    position: np.ndarray,
    config: Dict[str, Any],
    heatmap_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    frame = np.asarray(frame, dtype=float)
    position = np.asarray(position, dtype=float)

    io_cfg = (config.get("io") or {})
    sampling_rate = float(io_cfg.get("sampling_rate", 1.0))

    detrend_cfg = (config.get("detrend") or {})
    peaks_cfg = (config.get("peaks") or {})
    period_cfg = dict(config.get("period") or {})
    period_cfg.setdefault("sampling_rate", sampling_rate)

    features_cfg = (config.get("features") or {})
    overlay_cfg = (config.get("overlay") or {})

    residual = detrend_residual(frame, position, **detrend_cfg)

    try:
        estimated_freq_hz = float(estimate_dominant_frequency(residual, **period_cfg))
    except Exception:
        estimated_freq_hz = float("nan")
    freq_hz = resolve_positive_frequency(
        estimated_freq_hz,
        frame=frame,
        sampling_rate=float(period_cfg.get("sampling_rate", 1.0)),
        min_freq=period_cfg.get("min_freq"),
        max_freq=period_cfg.get("max_freq"),
    )
    period_s = float(frequency_to_period(freq_hz))
    frames_per_period = (sampling_rate / freq_hz) if (np.isfinite(freq_hz) and freq_hz > 0) else None

    peak_sets = _detect_peak_sets(residual, peaks_cfg, frames_per_period)

    wave_rows: List[Dict[str, Any]] = []
    peak_rows: List[Dict[str, Any]] = []
    for peak_set in peak_sets:
        wave_rows.extend(build_wave_rows(
            frame=frame,
            position=position,
            residual=residual,
            fit_residual=peak_set["signal"],
            peaks_idx=peak_set["peaks_idx"],
            peak_props=peak_set["peak_props"],
            sampling_rate=sampling_rate,
            sample=sample,
            track_stem=track_stem,
            features_cfg=features_cfg,
            event_polarity=peak_set["event_polarity"],
            event_kind=peak_set["event_kind"],
            fit_signal_sign=peak_set["sign"],
            freq_hz=freq_hz if np.isfinite(freq_hz) else None,
            period_frac_for_fit=float(features_cfg.get("fit_window_period_frac", 0.5)),
            coord_meta=heatmap_meta,
        ))

        peak_rows.extend(build_peak_rows(
            frame=frame,
            position=position,
            residual=residual,
            fit_residual=peak_set["signal"],
            peaks_idx=peak_set["peaks_idx"],
            peak_props=peak_set["peak_props"],
            sampling_rate=sampling_rate,
            sample=sample,
            track_stem=track_stem,
            features_cfg=features_cfg,
            event_polarity=peak_set["event_polarity"],
            event_kind=peak_set["event_kind"],
            fit_signal_sign=peak_set["sign"],
            global_freq_hz=freq_hz if np.isfinite(freq_hz) else None,
            period_frac_for_fit=float(features_cfg.get("fit_window_period_frac", 0.5)),
            coord_meta=heatmap_meta,
        ))

    wave_rows = _sort_and_renumber_event_rows(wave_rows, index_key="wave_index")
    peak_rows = _sort_and_renumber_event_rows(peak_rows)
    peak_props = _combine_peak_props([peak_set["peak_props"] for peak_set in peak_sets])
    event_indices = _event_indices_from_rows(peak_rows)
    event_kinds = _event_kinds_from_rows(peak_rows)
    event_polarity = _normalize_event_polarity(peaks_cfg.get("event_polarity", peaks_cfg.get("polarity", "both")))

    track_quality = _track_quality_metrics(
        residual=residual,
        wave_rows=wave_rows,
        peak_rows=peak_rows,
        peak_props=peak_props,
        sampling_rate=sampling_rate,
        dominant_freq_hz=freq_hz,
        period_cfg=period_cfg,
    )
    _attach_quality_metric_columns(wave_rows, peak_rows, track_quality)

    amps = np.abs(residual[event_indices]) if len(event_indices) else np.array([], dtype=float)
    num_maxima = sum(1 for kind in event_kinds if kind == "max")
    num_minima = sum(1 for kind in event_kinds if kind == "min")

    track_row: Dict[str, Any] = {
        "track_index": int(track_index),
        "amplitude": float(np.nanmean(amps)) if amps.size else None,
        "frequency": float(freq_hz) if np.isfinite(freq_hz) else None,
        "error": _quality_cell_value(track_quality.get("track_fit_error_median")),
        "x0": int(round(map_heatmap_x(position[0], heatmap_meta))) if position.size else None,
        "y0": int(round(map_heatmap_y(frame[0], heatmap_meta))) if frame.size else None,
        "metrics": {
            "num_peaks": int(len(event_indices)),
            "num_events": int(len(event_indices)),
            "num_maxima": int(num_maxima),
            "num_minima": int(num_minima),
            "event_polarity": event_polarity,
            "period": float(period_s) if np.isfinite(period_s) else None,
            "sampling_rate": sampling_rate,
            "track_stem": track_stem,
            "sample": sample,
            "coord_origin": (heatmap_meta or {}).get("coord_origin"),
            "pixel_mapping": (heatmap_meta or {}).get("pixel_mapping"),
            **track_quality,
        },
        "overlay": {},
    }
    track_row.update({key: _quality_cell_value(value) for key, value in track_quality.items()})

    overlay_track_event = _build_overlay_track_event(
        job_id=job_id,
        track_index=track_index,
        frame=frame,
        position=position,
        residual=residual,
        peaks_idx=event_indices,
        peak_kinds=event_kinds,
        freq_hz=freq_hz,
        period_s=period_s,
        cfg=overlay_cfg,
        track_stem=track_stem,
        sample=sample,
    )

    return track_row, wave_rows, peak_rows, overlay_track_event


def load_track_frame_position(track_path: Path, *, order: str = "xy") -> Tuple[np.ndarray, np.ndarray]:
    return track_frame_position_from_points(np.load(track_path), order=order)


def track_frame_position_from_points(data: np.ndarray, *, order: str = "xy") -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        if order == "yx":
            frame = arr[:, 0].astype(float, copy=False)
            position = arr[:, 1].astype(float, copy=False)
        else:
            position = arr[:, 0].astype(float, copy=False)
            frame = arr[:, 1].astype(float, copy=False)
        order_idx = np.argsort(frame, kind="stable")
        return frame[order_idx], position[order_idx]
    if arr.ndim == 1:
        return np.arange(arr.shape[0], dtype=float), arr.astype(float, copy=False)
    raise ValueError(f"Unsupported track array shape: {arr.shape}")


def _load_track_frame_position(track_path: Path, *, order: str = "xy") -> Tuple[np.ndarray, np.ndarray]:
    return load_track_frame_position(track_path, order=order)


FIT_QUALITY_KEYS = [
    "fit_error_vnmse",
    "fit_r2",
    "fit_rmse_px",
    "fit_nrmse",
    "fit_mae_px",
    "fit_points",
    "raw_fit_error_vnmse",
    "raw_fit_r2",
    "raw_fit_rmse_px",
    "raw_fit_nrmse",
    "raw_fit_mae_px",
    "raw_fit_points",
]
TRACK_CONTEXT_QUALITY_KEYS = [
    "period_consistency_cv",
    "frequency_agreement_error",
    "spectral_snr",
    "peak_prominence_snr",
]
TRACK_QUALITY_KEYS = [
    "track_fit_error_median",
    "track_fit_error_p90",
    "track_fit_r2_median",
    "track_fit_rmse_px_median",
    "track_fit_nrmse_median",
    "track_fit_mae_px_median",
    "track_fit_points_median",
    *TRACK_CONTEXT_QUALITY_KEYS,
]


def _track_quality_metrics(
    *,
    residual: np.ndarray,
    wave_rows: List[Dict[str, Any]],
    peak_rows: List[Dict[str, Any]],
    peak_props: Dict[str, Any],
    sampling_rate: float,
    dominant_freq_hz: float,
    period_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    periods = _metric_values(wave_rows, "period_s", positive=True)
    local_freqs = _metric_values(wave_rows, "frequency_hz", positive=True)
    fit_errors = _metric_values(wave_rows, "fit_error_vnmse")

    out: Dict[str, Any] = {
        "track_fit_error_median": _nanmedian(fit_errors),
        "track_fit_error_p90": _nanpercentile(fit_errors, 90.0),
        "track_fit_r2_median": _nanmedian(_metric_values(wave_rows, "fit_r2")),
        "track_fit_rmse_px_median": _nanmedian(_metric_values(wave_rows, "fit_rmse_px")),
        "track_fit_nrmse_median": _nanmedian(_metric_values(wave_rows, "fit_nrmse")),
        "track_fit_mae_px_median": _nanmedian(_metric_values(wave_rows, "fit_mae_px")),
        "track_fit_points_median": _nanmedian(_metric_values(wave_rows, "fit_points", positive=True)),
        "period_consistency_cv": _coefficient_of_variation(periods),
        "frequency_agreement_error": _frequency_agreement_error(local_freqs, dominant_freq_hz),
        "spectral_snr": _spectral_snr(
            residual,
            sampling_rate=sampling_rate,
            min_freq=period_cfg.get("min_freq"),
            max_freq=period_cfg.get("max_freq"),
        ),
        "peak_prominence_snr": _peak_prominence_snr(residual, peak_props),
    }
    return out


def _attach_quality_metric_columns(
    wave_rows: List[Dict[str, Any]],
    peak_rows: List[Dict[str, Any]],
    track_quality: Dict[str, Any],
) -> None:
    for row in [*wave_rows, *peak_rows]:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        for key in FIT_QUALITY_KEYS:
            row[key] = _quality_cell_value(metrics.get(key))
        for key in TRACK_CONTEXT_QUALITY_KEYS:
            row[key] = _quality_cell_value(track_quality.get(key))


def _metric_values(rows: List[Dict[str, Any]], key: str, *, positive: bool = False) -> np.ndarray:
    values: List[float] = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(key)
        try:
            number = float(value)
        except Exception:
            continue
        if not np.isfinite(number):
            continue
        if positive and number <= 0:
            continue
        values.append(number)
    return np.asarray(values, dtype=float)


def _quality_cell_value(value: Any) -> Any:
    try:
        number = float(value)
    except Exception:
        return value
    if not np.isfinite(number):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return number


def _nanmedian(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def _nanpercentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _coefficient_of_variation(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size < 2:
        return float("nan")
    mean = float(np.mean(finite))
    if mean <= 0:
        return float("nan")
    return float(np.std(finite) / mean)


def _frequency_agreement_error(local_freqs: np.ndarray, dominant_freq_hz: float) -> float:
    try:
        dominant = float(dominant_freq_hz)
    except Exception:
        dominant = float("nan")
    if not np.isfinite(dominant) or dominant <= 0:
        return float("nan")
    finite = np.asarray(local_freqs, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return float("nan")
    return float(abs(np.median(finite) - dominant) / dominant)


def _spectral_snr(
    residual: np.ndarray,
    *,
    sampling_rate: float,
    min_freq: Any = None,
    max_freq: Any = None,
) -> float:
    if sampling_rate is None or sampling_rate <= 0:
        return float("nan")
    y = np.asarray(residual, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return float("nan")
    y = y - float(np.mean(y))
    magnitude = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=1.0 / float(sampling_rate))
    if magnitude.size:
        magnitude[0] = 0.0

    mask = freqs > 0
    min_value = _optional_float(min_freq)
    max_value = _optional_float(max_freq)
    if min_value is not None:
        mask &= freqs >= min_value
    if max_value is not None:
        mask &= freqs <= max_value

    candidates = magnitude[mask]
    candidates = candidates[np.isfinite(candidates)]
    if candidates.size < 2:
        return float("nan")
    peak_pos = int(np.argmax(candidates))
    peak = float(candidates[peak_pos])
    background = np.delete(candidates, peak_pos)
    background = background[np.isfinite(background) & (background > 0)]
    if peak <= 0 or background.size == 0:
        return float("nan")
    noise_floor = float(np.median(background))
    return float(peak / noise_floor) if noise_floor > 0 else float("nan")


def _peak_prominence_snr(residual: np.ndarray, peak_props: Dict[str, Any]) -> float:
    prominences = np.asarray((peak_props or {}).get("prominences", []), dtype=float)
    prominences = prominences[np.isfinite(prominences) & (prominences > 0)]
    if prominences.size == 0:
        return float("nan")
    noise = _mad(residual)
    if not np.isfinite(noise) or noise <= 0:
        return float("nan")
    return float(np.median(prominences) / noise)


def _mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    med = float(np.median(finite))
    return float(np.median(np.abs(finite - med)))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _normalize_event_polarity(value: Any) -> str:
    key = str(value or "both").strip().lower()
    return PEAK_POLARITY_ALIASES.get(key, "both")


def _peak_polarity_specs(value: Any) -> List[Dict[str, Any]]:
    polarity = _normalize_event_polarity(value)
    specs: List[Dict[str, Any]] = []
    if polarity in {"maxima", "both"}:
        specs.append({"event_polarity": "maxima", "event_kind": "max", "sign": 1})
    if polarity in {"minima", "both"}:
        specs.append({"event_polarity": "minima", "event_kind": "min", "sign": -1})
    return specs


def _detect_peak_sets(
    residual: np.ndarray,
    peaks_cfg: Dict[str, Any],
    frames_per_period: Optional[float],
) -> List[Dict[str, Any]]:
    specs = _peak_polarity_specs(peaks_cfg.get("event_polarity", peaks_cfg.get("polarity", "both")))
    base_signal = np.asarray(residual, dtype=float)
    smoothing_sigma = float(peaks_cfg.get("smoothing_sigma_rows", 0.0) or 0.0)
    if smoothing_sigma > 0 and base_signal.size > 1:
        base_signal = gaussian_filter1d(base_signal, sigma=smoothing_sigma, mode="nearest")
    out: List[Dict[str, Any]] = []
    for spec in specs:
        sign = int(spec["sign"])
        signal = base_signal * float(sign)
        peaks_idx, peak_props = _detect_peaks(signal, peaks_cfg, frames_per_period)
        out.append({
            **spec,
            "signal": signal,
            "peaks_idx": np.asarray(peaks_idx, dtype=int),
            "peak_props": peak_props,
        })
    return _suppress_cross_polarity_peak_sets(out, peaks_cfg)


def _peak_set_strengths(peak_set: Dict[str, Any]) -> np.ndarray:
    peaks_idx = np.asarray(peak_set.get("peaks_idx", []), dtype=int)
    peak_props = peak_set.get("peak_props") or {}
    prominences = np.asarray(peak_props.get("prominences", []), dtype=float)
    if prominences.ndim == 1 and prominences.shape[0] == peaks_idx.shape[0]:
        strengths = prominences.astype(float, copy=True)
    else:
        signal = np.asarray(peak_set.get("signal", []), dtype=float)
        strengths = np.full(peaks_idx.shape[0], np.nan, dtype=float)
        valid = (peaks_idx >= 0) & (peaks_idx < signal.shape[0])
        strengths[valid] = signal[peaks_idx[valid]]
    return np.where(np.isfinite(strengths), strengths, -np.inf)


def _filter_peak_props(peak_props: Dict[str, Any], keep_mask: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, values in (peak_props or {}).items():
        arr = np.asarray(values)
        if arr.ndim >= 1 and arr.shape[0] == keep_mask.shape[0]:
            out[key] = arr[keep_mask]
        else:
            out[key] = values
    return out


def _suppress_cross_polarity_peak_sets(
    peak_sets: List[Dict[str, Any]],
    peaks_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    try:
        min_distance = int(peaks_cfg.get("cross_polarity_min_distance", 0))
    except Exception:
        min_distance = 0
    policy = str(peaks_cfg.get("cross_polarity_policy", "stronger")).strip().lower()
    if min_distance <= 0 or policy in {"none", "off", "false", "disabled"}:
        return peak_sets

    events: List[Dict[str, Any]] = []
    for set_i, peak_set in enumerate(peak_sets):
        peaks_idx = np.asarray(peak_set.get("peaks_idx", []), dtype=int)
        strengths = _peak_set_strengths(peak_set)
        for peak_pos, peak_i in enumerate(peaks_idx.tolist()):
            events.append({
                "set_i": set_i,
                "peak_pos": int(peak_pos),
                "peak_i": int(peak_i),
                "event_kind": str(peak_set.get("event_kind", "")),
                "strength": float(strengths[peak_pos]) if peak_pos < strengths.shape[0] else float("-inf"),
            })

    if len(events) < 2:
        return peak_sets

    keep = np.ones(len(events), dtype=bool)
    order = sorted(
        range(len(events)),
        key=lambda i: (-events[i]["strength"], events[i]["peak_i"], 0 if events[i]["event_kind"] == "max" else 1),
    )
    for event_i in order:
        if not keep[event_i]:
            continue
        event = events[event_i]
        for other_i, other in enumerate(events):
            if other_i == event_i or not keep[other_i]:
                continue
            if other["event_kind"] == event["event_kind"]:
                continue
            if abs(int(other["peak_i"]) - int(event["peak_i"])) <= min_distance:
                keep[other_i] = False

    per_set_keep = [
        np.ones(np.asarray(peak_set.get("peaks_idx", []), dtype=int).shape[0], dtype=bool)
        for peak_set in peak_sets
    ]
    for event_i, event in enumerate(events):
        if not keep[event_i]:
            per_set_keep[int(event["set_i"])][int(event["peak_pos"])] = False

    filtered: List[Dict[str, Any]] = []
    for peak_set, keep_mask in zip(peak_sets, per_set_keep):
        peaks_idx = np.asarray(peak_set.get("peaks_idx", []), dtype=int)
        next_set = dict(peak_set)
        next_set["peaks_idx"] = peaks_idx[keep_mask]
        next_set["peak_props"] = _filter_peak_props(peak_set.get("peak_props") or {}, keep_mask)
        filtered.append(next_set)
    return filtered


def _event_sort_key(row: Dict[str, Any]) -> Tuple[int, int]:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    try:
        peak_i = int(metrics.get("peak_i", 0))
    except Exception:
        peak_i = 0
    kind = str(metrics.get("event_kind", row.get("event_kind", "max")))
    return peak_i, 1 if kind == "min" else 0


def _sort_and_renumber_event_rows(rows: List[Dict[str, Any]], *, index_key: Optional[str] = None) -> List[Dict[str, Any]]:
    sorted_rows = sorted(rows, key=_event_sort_key)
    for idx, row in enumerate(sorted_rows, start=1):
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            row["metrics"] = metrics
        metrics["event_index"] = int(idx)
        metrics["peak_index"] = int(idx)
        if index_key:
            row[index_key] = int(idx)
            metrics[index_key] = int(idx)
    return sorted_rows


def _event_indices_from_rows(rows: List[Dict[str, Any]]) -> np.ndarray:
    indices: List[int] = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        try:
            indices.append(int(metrics["peak_i"]))
        except Exception:
            continue
    return np.asarray(indices, dtype=int)


def _event_kinds_from_rows(rows: List[Dict[str, Any]]) -> List[str]:
    kinds: List[str] = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        kind = str(metrics.get("event_kind", row.get("event_kind", "max")))
        kinds.append("min" if kind == "min" else "max")
    return kinds


def _combine_peak_props(prop_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted({key for props in prop_sets for key in (props or {}).keys()})
    out: Dict[str, Any] = {}
    for key in keys:
        values = []
        for props in prop_sets:
            if key not in (props or {}):
                continue
            arr = np.asarray(props[key])
            if arr.ndim == 1:
                values.append(arr)
        if values:
            out[key] = np.concatenate(values)
    return out


def _detect_peaks(
    residual: np.ndarray,
    peaks_cfg: Dict[str, Any],
    frames_per_period: Optional[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if bool(peaks_cfg.get("adaptive", False)):
        peaks, props = detect_peaks_adaptive(
            residual,
            frames_per_period=frames_per_period,
            distance_frac=float(peaks_cfg.get("distance_frac", 0.6)),
            width_frac=float(peaks_cfg.get("width_frac", 0.2)),
            rel_mad_k=float(peaks_cfg.get("rel_mad_k", 2.0)),
            abs_min_prom_px=float(peaks_cfg.get("abs_min_prom_px", 1.0)),
            nms_enable=bool(peaks_cfg.get("nms_enable", True)),
            nms_dominance_frac=float(peaks_cfg.get("nms_dominance_frac", 0.55)),
        )
        return ensure_minimum_peaks(residual, peaks, props, minimum=int(peaks_cfg.get("minimum_per_track", 1)))
    legacy_kwargs: Dict[str, Any] = {
        "prominence": float(peaks_cfg.get("prominence", 1.0)),
        "width": float(peaks_cfg.get("width", 1.0)),
    }
    if peaks_cfg.get("distance", None) is not None:
        legacy_kwargs["distance"] = int(peaks_cfg["distance"])
    peaks, props = detect_peaks(residual, **legacy_kwargs)
    return ensure_minimum_peaks(residual, peaks, props, minimum=int(peaks_cfg.get("minimum_per_track", 1)))


def _infer_sample(track_path: Path) -> str:
    base = track_path.parent.parent.name
    if base.endswith("_heatmap"):
        return base[:-8]
    return base


def _build_overlay_track_event(
    *,
    job_id: UUID,
    track_index: int,
    frame: np.ndarray,
    position: np.ndarray,
    residual: np.ndarray,
    peaks_idx: np.ndarray,
    freq_hz: float,
    period_s: float,
    cfg: Dict[str, Any],
    track_stem: str,
    sample: str,
    peak_kinds: Optional[List[str]] = None,
) -> Dict[str, Any]:
    max_points = int(cfg.get("max_points", 300))
    xs, ys = _decimate_polyline(position, frame, max_points=max_points)

    peak_pts: List[Dict[str, Any]] = []
    kinds = peak_kinds or []
    for event_pos, i in enumerate(peaks_idx.tolist()):
        if 0 <= i < len(frame):
            kind = kinds[event_pos] if event_pos < len(kinds) else "max"
            peak_pts.append({
                "i": int(i),
                "kind": kind,
                "event_polarity": "minima" if kind == "min" else "maxima",
                "x": float(position[i]),
                "y": float(frame[i]),
                "frame": float(frame[i]),
                "position": float(position[i]),
                "amp": float(residual[i]),
            })

    return {
        "job_id": str(job_id),
        "sample": sample,
        "track_index": int(track_index),
        "track_stem": track_stem,
        "poly": [{"x": float(a), "y": float(b)} for a, b in zip(xs, ys)],
        "peaks": peak_pts,
        "freq_hz": float(freq_hz) if np.isfinite(freq_hz) else None,
        "period": float(period_s) if np.isfinite(period_s) else None,
    }


def _decimate_polyline(x: np.ndarray, y: np.ndarray, *, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(min(len(x), len(y)))
    if n <= max_points:
        return x[:n], y[:n]
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return x[idx], y[idx]
