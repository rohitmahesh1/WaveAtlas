# app/extract_core.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple
from uuid import UUID

import numpy as np

from .signal.detrend import detrend_residual
from .signal.peaks import detect_peaks, detect_peaks_adaptive
from .signal.period import estimate_dominant_frequency, frequency_to_period
from .features import build_wave_rows, build_peak_rows


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
    ) -> KymoOutput:
        from .modules.kb_adapter import run_kymobutler as run_kymo

        image_id = _image_id_from_path(heatmap_path)
        base_dir = scratch_dir / image_id
        base_dir.mkdir(parents=True, exist_ok=True)

        onnx_cfg = ((self.config.get("kymo") or {}).get("onnx") or {})

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
            **_flatten_onnx_cfg_for_runner(onnx_cfg),
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
    ) -> KymoOutput:
        from .modules.kymo_interface import run_kymobutler as run_kymo

        image_id = _image_id_from_path(heatmap_path)
        base_dir = scratch_dir / image_id
        base_dir.mkdir(parents=True, exist_ok=True)

        kymo_cfg = (self.config.get("kymo") or {})
        min_length = int(kymo_cfg.get("min_length", 30))
        verbose = bool(kymo_cfg.get("verbose", False))

        run_kymo(
            str(heatmap_path),
            output_dir=str(base_dir),
            min_length=min_length,
            verbose=verbose,
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


def _flatten_onnx_cfg_for_runner(onnx_cfg: Dict[str, Any]) -> Dict[str, Any]:
    thresholds = (onnx_cfg.get("thresholds") or {})
    hyst = (onnx_cfg.get("hysteresis") or {})
    auto = (onnx_cfg.get("auto_threshold") or {})
    morph = (onnx_cfg.get("morphology") or {})
    comp = (onnx_cfg.get("components") or {})
    skel = (onnx_cfg.get("skeleton") or {})
    post = (onnx_cfg.get("postproc") or {})
    dedupe = (post.get("dedupe") or {})
    tracking = (onnx_cfg.get("tracking") or {})

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
        "dedupe_enable": bool(dedupe.get("enabled", True)),
        "dedupe_min_rows": int(dedupe.get("min_rows", 30)),
        "dedupe_min_score": float(dedupe.get("min_score", 0.11)),
        "dedupe_overlap_iou": float(dedupe.get("overlap_iou", 0.80)),
        "dedupe_dx_tol": float(dedupe.get("dx_tol", 2.5)),
        "fuse_uni_into_bi": bool(onnx_cfg.get("fuse_uni_into_bi", True)),
        "fuse_uni_weight": float(onnx_cfg.get("fuse_uni_weight", 0.7)),
    }


# -----------------------------
# Track processing
# -----------------------------

def process_track(
    *,
    job_id: UUID,
    track_index: int,
    track_path: Path,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    kymo_cfg = (config.get("kymo") or {})
    backend = str(kymo_cfg.get("backend", "onnx")).lower()
    track_xy_order = str(kymo_cfg.get("track_xy_order", "auto")).lower()
    if track_xy_order == "auto":
        # ONNX kymobutler saves (y, x) points; Wolfram typically outputs (x, y).
        track_xy_order = "yx" if backend == "onnx" else "xy"

    io_cfg = (config.get("io") or {})
    sampling_rate = float(io_cfg.get("sampling_rate", 1.0))

    detrend_cfg = (config.get("detrend") or {})
    peaks_cfg = (config.get("peaks") or {})
    period_cfg = dict(config.get("period") or {})
    period_cfg.setdefault("sampling_rate", sampling_rate)

    features_cfg = (config.get("features") or {})
    overlay_cfg = (config.get("overlay") or {})

    x, y = _load_track_xy(track_path, order=track_xy_order)

    residual = detrend_residual(x, y, **detrend_cfg)

    freq_hz = float(estimate_dominant_frequency(residual, **period_cfg))
    period_s = float(frequency_to_period(freq_hz))
    frames_per_period = (sampling_rate / freq_hz) if (np.isfinite(freq_hz) and freq_hz > 0) else None

    peaks_idx, peak_props = _detect_peaks(residual, peaks_cfg, frames_per_period)

    wave_rows = build_wave_rows(
        x=x,
        y=y,
        residual=residual,
        peaks_idx=peaks_idx,
        peak_props=peak_props,
        sampling_rate=sampling_rate,
        sample=_infer_sample(track_path),
        track_stem=track_path.stem,
        features_cfg=features_cfg,
        freq_hz=freq_hz if np.isfinite(freq_hz) else None,
        period_frac_for_fit=float(features_cfg.get("fit_window_period_frac", 0.5)),
    )

    peak_rows = build_peak_rows(
        x=x,
        y=y,
        residual=residual,
        peaks_idx=peaks_idx,
        peak_props=peak_props,
        sampling_rate=sampling_rate,
        sample=_infer_sample(track_path),
        track_stem=track_path.stem,
        features_cfg=features_cfg,
        global_freq_hz=freq_hz if np.isfinite(freq_hz) else None,
        period_frac_for_fit=float(features_cfg.get("fit_window_period_frac", 0.5)),
    )

    amps = residual[peaks_idx] if len(peaks_idx) else np.array([], dtype=float)

    track_row: Dict[str, Any] = {
        "track_index": int(track_index),
        "amplitude": float(np.nanmean(amps)) if amps.size else None,
        "frequency": float(freq_hz) if np.isfinite(freq_hz) else None,
        "error": None,
        "x0": int(x[0]) if x.size else None,
        "y0": int(y[0]) if y.size else None,
        "metrics": {
            "num_peaks": int(len(peaks_idx)),
            "period": float(period_s) if np.isfinite(period_s) else None,
            "sampling_rate": sampling_rate,
            "track_stem": track_path.stem,
            "sample": _infer_sample(track_path),
        },
        "overlay": {},
    }

    overlay_track_event = _build_overlay_track_event(
        job_id=job_id,
        track_index=track_index,
        x=x,
        y=y,
        residual=residual,
        peaks_idx=peaks_idx,
        freq_hz=freq_hz,
        period_s=period_s,
        cfg=overlay_cfg,
        track_stem=track_path.stem,
        sample=_infer_sample(track_path),
    )

    return track_row, wave_rows, peak_rows, overlay_track_event


def _load_track_xy(track_path: Path, *, order: str = "xy") -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(track_path)
    if data.ndim == 2 and data.shape[1] >= 2:
        if order == "yx":
            return data[:, 1].astype(float, copy=False), data[:, 0].astype(float, copy=False)
        return data[:, 0].astype(float, copy=False), data[:, 1].astype(float, copy=False)
    if data.ndim == 1:
        return np.arange(data.shape[0], dtype=float), data.astype(float, copy=False)
    raise ValueError(f"Unsupported track array shape: {data.shape}")


def _detect_peaks(
    residual: np.ndarray,
    peaks_cfg: Dict[str, Any],
    frames_per_period: Optional[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if bool(peaks_cfg.get("adaptive", True)):
        return detect_peaks_adaptive(
            residual,
            frames_per_period=frames_per_period,
            distance_frac=float(peaks_cfg.get("distance_frac", 0.6)),
            width_frac=float(peaks_cfg.get("width_frac", 0.2)),
            rel_mad_k=float(peaks_cfg.get("rel_mad_k", 2.0)),
            abs_min_prom_px=float(peaks_cfg.get("abs_min_prom_px", 1.0)),
            nms_enable=bool(peaks_cfg.get("nms_enable", True)),
            nms_dominance_frac=float(peaks_cfg.get("nms_dominance_frac", 0.55)),
        )
    legacy_kwargs: Dict[str, Any] = {
        "prominence": float(peaks_cfg.get("prominence", 1.0)),
        "width": float(peaks_cfg.get("width", 1.0)),
    }
    if peaks_cfg.get("distance", None) is not None:
        legacy_kwargs["distance"] = int(peaks_cfg["distance"])
    return detect_peaks(residual, **legacy_kwargs)


def _infer_sample(track_path: Path) -> str:
    base = track_path.parent.parent.name
    if base.endswith("_heatmap"):
        return base[:-8]
    return base


def _build_overlay_track_event(
    *,
    job_id: UUID,
    track_index: int,
    x: np.ndarray,
    y: np.ndarray,
    residual: np.ndarray,
    peaks_idx: np.ndarray,
    freq_hz: float,
    period_s: float,
    cfg: Dict[str, Any],
    track_stem: str,
    sample: str,
) -> Dict[str, Any]:
    max_points = int(cfg.get("max_points", 300))
    xs, ys = _decimate_polyline(x, y, max_points=max_points)

    peak_pts: List[Dict[str, Any]] = []
    for i in peaks_idx.tolist():
        if 0 <= i < len(x):
            peak_pts.append({"i": int(i), "x": float(x[i]), "y": float(y[i]), "amp": float(residual[i])})

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
