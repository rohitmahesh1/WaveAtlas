from __future__ import annotations

import csv
import io
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

import yaml
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel, Field
from sqlmodel import Session, select, delete

from ..artifact_store import ArtifactStore
from ..db import engine
from ..job_store import JobStore
from ..models import (
    Artifact,
    ArtifactKind,
    ArtifactRead,
    Job,
    JobCreate,
    JobRead,
    JobStatus,
    JobEvent,
    Peak,
    Track,
    Wave,
)
from ..pipeline import PipelineSettings, run_job
from ..analysis_mode import LARGE_WAVE_ANALYSIS_MODE, RIPPLE_ANALYSIS_MODE, resolve_analysis_mode
from ..time_utils import utc_now_iso
from ..extract_core import PEAK_POLARITY_ALIASES, _suppress_cross_polarity_peak_sets
from ..large_wave_fit import (
    fit_asymmetric_basin_residual,
    fit_large_wave,
    large_wave_basin_window,
)
from ..signal.detrend import fit_baseline_ransac
from ..signal.peaks import detect_peaks, detect_peaks_adaptive, ensure_minimum_peaks
from ..signal.period import estimate_dominant_frequency, frequency_to_period, resolve_positive_frequency

from .deps import get_artifact_store, get_db_session, get_owner_session_id


router = APIRouter(tags=["jobs"])


# -----------------------------
# Helpers
# -----------------------------

def _get_job_owned(session: Session, job_id: UUID, owner_session_id: UUID) -> Job:
    q = select(Job).where(Job.id == job_id, Job.owner_session_id == owner_session_id)
    job = session.exec(q).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _get_artifact_owned(session: Session, job_id: UUID, artifact_id: UUID, owner_session_id: UUID) -> Artifact:
    # Ensures job exists and ownership is correct
    _get_job_owned(session, job_id, owner_session_id)
    q = select(Artifact).where(Artifact.id == artifact_id, Artifact.job_id == job_id)
    art = session.exec(q).first()
    if not art:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return art


def _get_upload_filename(session: Session, job_id: UUID) -> Optional[str]:
    q = (
        select(Artifact)
        .where(Artifact.job_id == job_id, Artifact.kind.in_((ArtifactKind.upload_csv, ArtifactKind.upload_image)))
        .order_by(Artifact.created_at.desc())
        .limit(1)
    )
    art = session.exec(q).first()
    if not art or not getattr(art, "meta", None):
        return None
    filename = (art.meta or {}).get("filename")
    return str(filename) if filename else None


def _job_read_with_filename(session: Session, job: Job) -> JobRead:
    out = JobRead.model_validate(job)  # type: ignore
    out.input_filename = _get_upload_filename(session, job.id)
    out.analysis_mode = resolve_analysis_mode(dict(job.config or {}))
    return out


def _pipeline_settings_from_env() -> PipelineSettings:
    scratch_root = Path(os.getenv("SCRATCH_ROOT", "/tmp/mlapp_scratch"))
    batch_size = int(os.getenv("DB_BATCH_SIZE", "50"))
    progress_every = float(os.getenv("PROGRESS_EVERY_SECS", "2.0"))
    overlay_every = int(os.getenv("EMIT_OVERLAY_EVERY_TRACKS", "1"))

    return PipelineSettings(
        scratch_root=scratch_root,
        db_batch_size=batch_size,
        progress_every_secs=progress_every,
        emit_overlay_every_tracks=overlay_every,
    )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dicts, with override values taking precedence.
    """
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _pipeline_config_from_env() -> Dict[str, Any]:
    """
    Read YAML config from PIPELINE_CONFIG_PATH or configs/default.yaml.
    Returns {} if the config file is empty.
    """
    p = _pipeline_config_path()
    if not p.exists():
        raise HTTPException(status_code=500, detail=f"Pipeline config not found: {p}")
    data = yaml.safe_load(p.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Pipeline config must be a YAML mapping")
    return data


def _pipeline_config_path() -> Path:
    """
    Resolve the config path used for defaults.
    Falls back to ./configs/default.yaml when env var is unset.
    """
    raw_path = os.getenv("PIPELINE_CONFIG_PATH", "").strip()
    if raw_path:
        return Path(raw_path)
    return Path("configs/default.yaml")


def _config_docs_path() -> Path:
    raw_path = os.getenv("CONFIG_DOCS_PATH", "").strip()
    if raw_path:
        return Path(raw_path)
    return Path(__file__).resolve().parents[2] / "docs" / "config.md"


def _parse_config_value(config_value: Any) -> Dict[str, Any]:
    if isinstance(config_value, str):
        if not config_value.strip():
            return {}
        try:
            parsed = yaml.safe_load(config_value)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid YAML/JSON config: {exc}") from exc
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="Config must be a YAML/JSON mapping")
        return parsed
    if config_value is None:
        return {}
    if not isinstance(config_value, dict):
        raise HTTPException(status_code=400, detail="Config must be an object or YAML mapping")
    return config_value


def _effective_pipeline_config(job: Job) -> Dict[str, Any]:
    """
    Resolve the exact config a job should run with.

    The stored job config starts as user overrides. When a job is claimed for
    execution, routes persist this merged snapshot back to job.config so detail
    views and resumed work use the same settings the pipeline used.
    """
    return _deep_merge(_pipeline_config_from_env(), dict(job.config or {}))


def _artifact_prefix() -> str:
    pfx = os.getenv("GCS_PREFIX", "").strip().strip("/")
    return pfx


def _gcs_key_for_upload(job_id: UUID, filename: str) -> str:
    safe = filename.strip().replace("\\", "/").split("/")[-1]
    safe = safe or "upload"
    pfx = _artifact_prefix()
    if pfx:
        return f"{pfx}/jobs/{job_id}/upload/{safe}"
    return f"jobs/{job_id}/upload/{safe}"


def _is_image_upload(*, filename: Optional[str], content_type: Optional[str]) -> bool:
    ctype = (content_type or "").strip().lower()
    if ctype.startswith("image/"):
        return True
    suffix = Path(filename or "").suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _upload_artifact_kind(*, filename: Optional[str], content_type: Optional[str]) -> ArtifactKind:
    return ArtifactKind.upload_image if _is_image_upload(filename=filename, content_type=content_type) else ArtifactKind.upload_csv


def _ensure_upload_exists(job_store: JobStore, job_id: UUID) -> None:
    arts = [
        *job_store.list_artifacts(job_id, kind=ArtifactKind.upload_csv, limit=1),
        *job_store.list_artifacts(job_id, kind=ArtifactKind.upload_image, limit=1),
    ]
    if not arts:
        raise HTTPException(status_code=400, detail="No upload found for job")


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


def _track_xy_order_from_config(config: Dict[str, Any]) -> str:
    kymo_cfg = (config.get("kymo") or {})
    backend = str(kymo_cfg.get("backend", "onnx")).lower()
    order = str(kymo_cfg.get("track_xy_order", "auto")).lower()
    if order == "auto":
        return "yx" if backend == "onnx" else "xy"
    return order


def _load_track_frame_position_from_bytes(data: bytes, *, order: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(io.BytesIO(data), allow_pickle=False)
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
    raise HTTPException(status_code=500, detail="Unsupported track array shape")


def _parse_index_range(value: str, n: int) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="range must be 'lo:hi'")
    lo = int(parts[0]) if parts[0] else 0
    hi = int(parts[1]) if parts[1] else (n - 1)
    lo = max(0, min(lo, n - 1))
    hi = max(0, min(hi, n - 1))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _detect_peaks_for_detail(
    residual: np.ndarray,
    peaks_cfg: Dict[str, Any],
    frames_per_period: Optional[float],
) -> tuple[np.ndarray, Dict[str, Any]]:
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


def _normalize_detail_event_polarity(value: Any) -> str:
    key = str(value or "both").strip().lower()
    return PEAK_POLARITY_ALIASES.get(key, "both")


def _detail_peak_polarity_specs(value: Any) -> List[Dict[str, Any]]:
    polarity = _normalize_detail_event_polarity(value)
    specs: List[Dict[str, Any]] = []
    if polarity in {"maxima", "both"}:
        specs.append({"event_polarity": "maxima", "event_kind": "max", "sign": 1})
    if polarity in {"minima", "both"}:
        specs.append({"event_polarity": "minima", "event_kind": "min", "sign": -1})
    return specs


def _detect_peak_sets_for_detail(
    residual: np.ndarray,
    peaks_cfg: Dict[str, Any],
    frames_per_period: Optional[float],
) -> List[Dict[str, Any]]:
    specs = _detail_peak_polarity_specs(peaks_cfg.get("event_polarity", peaks_cfg.get("polarity", "both")))
    out: List[Dict[str, Any]] = []
    for spec in specs:
        sign = int(spec["sign"])
        signal = np.asarray(residual, dtype=float) * float(sign)
        peaks_idx, peak_props = _detect_peaks_for_detail(signal, peaks_cfg, frames_per_period)
        out.append({
            **spec,
            "signal": signal,
            "peaks_idx": np.asarray(peaks_idx, dtype=int),
            "peak_props": peak_props,
        })
    return _suppress_cross_polarity_peak_sets(out, peaks_cfg)


def _large_wave_peak_events_for_detail(
    waves: Iterable[Wave],
    residual: np.ndarray,
    *,
    width_multiplier: float = 1.0,
    boundary_smoothing_sigma_rows: float = 4.0,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for wave in waves:
        metrics = dict(wave.metrics or {})
        try:
            peak_i = int(metrics.get("peak_i"))
        except (TypeError, ValueError):
            continue
        if peak_i < 0 or peak_i >= len(residual):
            continue
        event_kind = "min" if str(metrics.get("event_kind", wave.event_kind)) == "min" else "max"
        sign = -1 if event_kind == "min" else 1
        signal = np.asarray(residual, dtype=float) * float(sign)
        try:
            event_amplitude = float(metrics.get("amplitude_px"))
        except (TypeError, ValueError):
            event_amplitude = float(signal[peak_i])
        if not math.isfinite(event_amplitude):
            event_amplitude = float(signal[peak_i])
        try:
            fit_window_frames = float(
                metrics.get("large_wave_width_frames", metrics.get("bulge_width_frames"))
            )
        except (TypeError, ValueError):
            fit_window_frames = float("nan")
        stored_lo = metrics.get("fit_window_lo")
        stored_hi = metrics.get("fit_window_hi")
        try:
            basin_window = (int(stored_lo), int(stored_hi))
            if not (0 <= basin_window[0] < peak_i < basin_window[1] < len(residual)):
                basin_window = None
        except (TypeError, ValueError):
            basin_window = None
        if basin_window is None:
            basin_window = _large_wave_basin_window(
                residual,
                center_idx=peak_i,
                minimum_width_frames=(
                    fit_window_frames if math.isfinite(fit_window_frames) else None
                ),
                width_multiplier=width_multiplier,
                smoothing_sigma_rows=boundary_smoothing_sigma_rows,
            )
        events.append({
            "peak_i": peak_i,
            "event_kind": event_kind,
            "event_polarity": "minima" if event_kind == "min" else "maxima",
            "fit_signal_sign": sign,
            "event_amplitude": event_amplitude,
            "fit_window_frames": (
                float(basin_window[1] - basin_window[0])
                if basin_window is not None
                else fit_window_frames if math.isfinite(fit_window_frames) else None
            ),
            "fit_window_lo": basin_window[0] if basin_window is not None else None,
            "fit_window_hi": basin_window[1] if basin_window is not None else None,
            "fit_window_source": (
                metrics.get("fit_window_source")
                if basin_window is not None and stored_lo is not None and stored_hi is not None
                else "large_wave_baseline_basin" if basin_window is not None else None
            ),
            "wave_index": wave.wave_index,
        })
    events.sort(key=lambda event: (int(event["peak_i"]), str(event["event_kind"])))
    return events


def _large_wave_basin_window(
    residual: np.ndarray,
    *,
    center_idx: int,
    minimum_width_frames: Optional[float],
    width_multiplier: float,
    smoothing_sigma_rows: float,
) -> Optional[tuple[int, int]]:
    return large_wave_basin_window(
        residual,
        center_idx=center_idx,
        minimum_width_frames=minimum_width_frames,
        width_multiplier=width_multiplier,
        smoothing_sigma_rows=smoothing_sigma_rows,
    )


def _large_wave_fit_frequency(
    *,
    width_frames: Any,
    sampling_rate: float,
    period_frac: float,
    width_multiplier: float,
) -> Optional[float]:
    try:
        width = float(width_frames) * max(0.1, float(width_multiplier))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(width) or width <= 0 or sampling_rate <= 0 or period_frac <= 0:
        return None
    return float(period_frac) * float(sampling_rate) / width


def _fit_anchored_wave_basin(
    residual: np.ndarray,
    t: np.ndarray,
    center_idx: int,
    *,
    window_lo: int,
    window_hi: int,
) -> Optional[tuple[np.ndarray, Dict[str, Any]]]:
    return fit_asymmetric_basin_residual(
        residual,
        t,
        center_idx,
        window_lo=window_lo,
        window_hi=window_hi,
    )


def _fit_anchored_sine(
    residual: np.ndarray,
    t: np.ndarray,
    freq: float,
    center_idx: Optional[int],
    *,
    sampling_rate: float = 1.0,
    period_frac: float = 0.5,
) -> Optional[tuple[np.ndarray, Dict[str, Any]]]:
    if center_idx is None or center_idx < 0 or center_idx >= len(t):
        return None
    if not (isinstance(freq, float) and math.isfinite(freq) and freq > 0):
        return None
    if sampling_rate <= 0:
        return None
    omega = 2.0 * math.pi * float(freq) / float(sampling_rate)
    t0 = float(t[int(center_idx)])
    phi = (math.pi / 2.0) - omega * t0
    s = np.sin(omega * t + phi).astype(np.float64)

    frames_per_period = (float(sampling_rate) / float(freq)) if sampling_rate else (1.0 / float(freq))
    half_span = max(1, int(round((float(period_frac) * frames_per_period) / 2.0)))
    lo = max(0, int(center_idx) - half_span)
    hi = min(len(t) - 1, int(center_idx) + half_span)
    peak_value = float(residual[int(center_idx)])
    z = s[lo : hi + 1] - 1.0
    y_slice = residual[lo : hi + 1].astype(np.float64)
    target = y_slice - peak_value
    denom = float(np.dot(z, z))
    A = float(np.dot(z, target) / denom) if denom > 0 and math.isfinite(denom) else peak_value
    c = float(peak_value - A)
    yfit = (A * s + c).astype(np.float64)
    y_fit = yfit[lo : hi + 1]
    if y_slice.size >= 2 and float(np.var(y_slice)) > 0:
        vnmse = float(np.mean((y_slice - y_fit) ** 2) / np.var(y_slice))
    else:
        vnmse = float("nan")
    return yfit, {
        "fit_amp_A": A,
        "fit_phase_phi": phi,
        "fit_offset_c": c,
        "fit_freq_hz": float(freq),
        "fit_error_vnmse": vnmse if math.isfinite(vnmse) else None,
        "fit_window_lo": lo,
        "fit_window_hi": hi,
        "fit_peak_value": peak_value,
        "fit_peak_error": float(yfit[int(center_idx)] - peak_value),
        "fit_passes_peak": True,
    }


def _detail_fit_meta_for_original_polarity(
    fit_meta: Dict[str, Any],
    *,
    sign: int,
    original_peak_value: float,
) -> Dict[str, Any]:
    out = dict(fit_meta)
    out["fit_signal_sign"] = int(sign)
    out["fit_event_value"] = out.get("fit_peak_value")
    out["fit_peak_value"] = float(original_peak_value)
    if int(sign) < 0:
        for key in ("fit_amp_A", "fit_offset_c", "fit_peak_error"):
            try:
                out[key] = float(out[key]) * -1.0
            except Exception:
                pass
    return out


# -----------------------------
# Upload flow payloads
# -----------------------------

class UploadSessionResponse(BaseModel):
    upload_url: str
    blob_path: str
    content_type: str
    object_key: str


class UploadCompletePayload(BaseModel):
    blob_path: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    byte_size: Optional[int] = None


class ConfigValidatePayload(BaseModel):
    config: Any = Field(default_factory=dict)


class JobRenamePayload(BaseModel):
    run_name: str = Field(min_length=1, max_length=120)


# -----------------------------
# Artifact views (frontend-friendly)
# -----------------------------

class ArtifactView(ArtifactRead):
    download_url: str


def _artifact_download_url(job_id: UUID, artifact_id: UUID) -> str:
    return f"/api/jobs/{job_id}/artifacts/{artifact_id}/download"


def _artifact_to_view(art: Artifact, *, job_id: UUID, artifact_store: ArtifactStore) -> ArtifactView:
    signed = artifact_store.signed_url(art.blob_path, expires_in=int(os.getenv("SIGNED_URL_EXPIRES_SECS", "3600")))
    url = signed or _artifact_download_url(job_id, art.id)
    return ArtifactView(**ArtifactRead.model_validate(art).model_dump(), download_url=url)  # type: ignore


# -----------------------------
# Routes
# -----------------------------

@router.post("/jobs", response_model=JobRead)
def create_job(
    payload: JobCreate,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
) -> JobRead:
    store = JobStore(session=session)
    config_value = _parse_config_value(payload.config)
    job = store.create_job(owner_session_id=owner_session_id, run_name=payload.run_name, config=config_value)
    return _job_read_with_filename(session, job)


@router.get("/jobs", response_model=List[JobRead])
def list_jobs(
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[JobRead]:
    store = JobStore(session=session)
    jobs = store.list_jobs_for_owner(owner_session_id, limit=limit, offset=offset, newest_first=True)
    return [_job_read_with_filename(session, j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobRead)
def get_job(
    job_id: UUID,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
) -> JobRead:
    job = _get_job_owned(session, job_id, owner_session_id)
    return _job_read_with_filename(session, job)


@router.patch("/jobs/{job_id}/name", response_model=JobRead)
def rename_job(
    job_id: UUID,
    payload: JobRenamePayload,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
) -> JobRead:
    _get_job_owned(session, job_id, owner_session_id)
    store = JobStore(session=session)
    run_name = payload.run_name.strip()
    if not run_name:
        raise HTTPException(status_code=422, detail="run_name must not be empty")
    job = store.update_run_name(job_id, run_name)
    return _job_read_with_filename(session, job)


@router.get("/config/default")
def get_default_config_text() -> Response:
    """
    Return the default pipeline config text from disk.
    """
    path = _pipeline_config_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {path}")
    try:
        raw = path.read_text()
        parsed = yaml.safe_load(raw)
        if parsed is not None and not isinstance(parsed, dict):
            raise HTTPException(status_code=500, detail="Default config must be a YAML mapping")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {exc}") from exc
    return Response(content=raw, media_type="text/plain")


@router.get("/docs/config")
def get_config_docs() -> Response:
    path = _config_docs_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Config docs not found")
    return Response(content=path.read_text(), media_type="text/markdown")


@router.post("/config/validate")
def validate_config(payload: ConfigValidatePayload) -> Dict[str, Any]:
    _ = _parse_config_value(payload.config)
    return {"ok": True}


@router.delete("/jobs/{job_id}")
def delete_job(
    job_id: UUID,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
) -> Dict[str, Any]:
    job = _get_job_owned(session, job_id, owner_session_id)
    if job.status in (JobStatus.queued, JobStatus.in_progress, JobStatus.cancel_requested):
        raise HTTPException(status_code=409, detail="Job is still running. Cancel it before deleting.")

    # Delete blobs first (best-effort)
    arts = session.exec(select(Artifact).where(Artifact.job_id == job_id)).all()
    blob_errors: List[str] = []
    for art in arts:
        try:
            artifact_store.delete_blob(art.blob_path)
        except Exception as exc:
            blob_errors.append(str(exc))

    # Delete DB rows (order matters due to FKs)
    session.exec(delete(Artifact).where(Artifact.job_id == job_id))
    session.exec(delete(Peak).where(Peak.job_id == job_id))
    session.exec(delete(Wave).where(Wave.job_id == job_id))
    session.exec(delete(Track).where(Track.job_id == job_id))
    session.exec(delete(JobEvent).where(JobEvent.job_id == job_id))
    session.exec(delete(Job).where(Job.id == job_id))
    session.commit()

    return {
        "ok": True,
        "deleted": {
            "artifacts": len(arts),
        },
        "blob_errors": blob_errors,
    }


@router.post("/jobs/{job_id}/upload-session", response_model=UploadSessionResponse)
def create_upload_session(
    job_id: UUID,
    request: Request,
    response: Response,
    filename: str = Query("upload.csv"),
    content_type: str = Query("text/csv"),
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
) -> UploadSessionResponse:
    """
    Returns a Google Cloud Storage resumable upload URL so the client can upload large files directly to GCS.
    """
    _get_job_owned(session, job_id, owner_session_id)

    if os.getenv("ARTIFACT_STORE", "local").strip().lower() != "gcs":
        raise HTTPException(status_code=400, detail="Resumable upload requires ARTIFACT_STORE=gcs")

    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail="google-cloud-storage is required for upload-session") from e

    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="GCS_BUCKET is not configured")

    key = _gcs_key_for_upload(job_id, filename)
    blob_path = f"gs://{bucket_name}/{key}"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    origin = request.headers.get("origin") or None

    upload_url = blob.create_resumable_upload_session(content_type=content_type, origin=origin)

    return UploadSessionResponse(
        upload_url=upload_url,
        blob_path=blob_path,
        content_type=content_type,
        object_key=key,
    )


@router.post("/jobs/{job_id}/upload-complete", response_model=ArtifactRead)
def upload_complete(
    job_id: UUID,
    payload: UploadCompletePayload,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
) -> ArtifactRead:
    """
    Records the uploaded blob as the job's upload artifact.
    The client should call this after finishing the resumable upload to GCS.
    """
    _get_job_owned(session, job_id, owner_session_id)
    store = JobStore(session=session)
    kind = _upload_artifact_kind(filename=payload.filename or payload.blob_path, content_type=payload.content_type)

    art = store.create_artifact(
        job_id=job_id,
        kind=kind,
        blob_path=payload.blob_path,
        label="upload",
        content_type=payload.content_type or ("image/png" if kind == ArtifactKind.upload_image else "text/csv"),
        byte_size=payload.byte_size,
        meta={
            "filename": payload.filename,
            "input_type": "image" if kind == ArtifactKind.upload_image else "table",
            "uploaded_at": utc_now_iso(),
            "upload_method": "gcs_resumable",
        },
    )
    return ArtifactRead.model_validate(art)  # type: ignore


@router.post("/jobs/{job_id}/upload", response_model=ArtifactRead)
async def upload_table(
    job_id: UUID,
    response: Response,
    file: UploadFile = File(...),
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
) -> ArtifactRead:
    """
    Direct upload to the API (OK for local dev / small files).
    Streams to a temp file to avoid loading the entire upload into memory.
    For large files on Cloud Run, prefer /upload-session + /upload-complete.
    """
    _get_job_owned(session, job_id, owner_session_id)
    store = JobStore(session=session)

    tmp_dir = Path(os.getenv("SCRATCH_ROOT", "/tmp/mlapp_scratch")) / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "upload.csv"
    content_type = file.content_type or "application/octet-stream"
    kind = _upload_artifact_kind(filename=filename, content_type=content_type)

    with tempfile.NamedTemporaryFile(dir=str(tmp_dir), delete=False) as tf:
        tmp_path = Path(tf.name)

    byte_size = 0
    try:
        chunk_size = int(os.getenv("UPLOAD_CHUNK_BYTES", str(1024 * 1024)))
        with tmp_path.open("ab") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                byte_size += len(chunk)
                out.write(chunk)

        if byte_size == 0:
            raise HTTPException(status_code=400, detail="Empty upload")

        blob_path, stored_size = artifact_store.put_file(
            job_id=job_id,
            kind=kind.value,
            filename=filename,
            local_path=str(tmp_path),
            content_type=content_type,
            label="upload",
        )

        art = store.create_artifact(
            job_id=job_id,
            kind=kind,
            blob_path=blob_path,
            label="upload",
            content_type=content_type,
            byte_size=stored_size,
            meta={
                "filename": filename,
                "input_type": "image" if kind == ArtifactKind.upload_image else "table",
                "uploaded_at": utc_now_iso(),
                "upload_method": "api_stream",
            },
        )
        return ArtifactRead.model_validate(art)  # type: ignore
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass


@router.post("/jobs/{job_id}/start", response_model=JobRead)
def start_job(
    job_id: UUID,
    background: BackgroundTasks,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
) -> JobRead:
    job = _get_job_owned(session, job_id, owner_session_id)
    store = JobStore(session=session)

    if job.status != JobStatus.queued:
        return _job_read_with_filename(session, job)

    _ensure_upload_exists(store, job_id)
    settings = _pipeline_settings_from_env()
    config = _effective_pipeline_config(job)

    job, claimed = store.claim_start(job_id, config=config)
    if not claimed:
        return _job_read_with_filename(session, job)

    def _run() -> None:
        with Session(engine) as bg_session:
            bg_store = JobStore(session=bg_session)
            run_job(
                job_id,
                job_store=bg_store,
                artifact_store=artifact_store,
                config=config,
                settings=settings,
            )

    background.add_task(_run)
    return _job_read_with_filename(session, job)


@router.post("/jobs/{job_id}/resume", response_model=JobRead)
def resume_job(
    job_id: UUID,
    background: BackgroundTasks,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
) -> JobRead:
    job = _get_job_owned(session, job_id, owner_session_id)
    store = JobStore(session=session)

    if job.status in (JobStatus.queued, JobStatus.in_progress, JobStatus.cancel_requested, JobStatus.completed):
        return _job_read_with_filename(session, job)

    _ensure_upload_exists(store, job_id)
    settings = _pipeline_settings_from_env()
    config = _effective_pipeline_config(job)

    job, claimed = store.claim_resume(job_id, config=config)
    if not claimed:
        return _job_read_with_filename(session, job)

    def _run() -> None:
        with Session(engine) as bg_session:
            bg_store = JobStore(session=bg_session)
            run_job(
                job_id,
                job_store=bg_store,
                artifact_store=artifact_store,
                config=config,
                settings=settings,
                resume=True,
            )

    background.add_task(_run)
    return _job_read_with_filename(session, job)


@router.post("/jobs/{job_id}/cancel", response_model=JobRead)
def cancel_job(
    job_id: UUID,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
) -> JobRead:
    _get_job_owned(session, job_id, owner_session_id)
    store = JobStore(session=session)
    job = store.request_cancel(job_id)
    return _job_read_with_filename(session, job)


@router.get("/jobs/{job_id}/artifacts", response_model=List[ArtifactView])
def list_artifacts(
    job_id: UUID,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    kind: Optional[ArtifactKind] = Query(None),
    label: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
) -> List[ArtifactView]:
    _get_job_owned(session, job_id, owner_session_id)

    q = select(Artifact).where(Artifact.job_id == job_id)
    if kind is not None:
        q = q.where(Artifact.kind == kind)
    if label is not None:
        q = q.where(Artifact.label == label)
    q = q.order_by(Artifact.created_at.asc()).limit(limit)

    arts = session.exec(q).all()
    return [_artifact_to_view(a, job_id=job_id, artifact_store=artifact_store) for a in arts]


@router.get("/jobs/{job_id}/tracks/{track_index}/detail")
def get_track_detail(
    job_id: UUID,
    track_index: int,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    include_sine: bool = Query(False),
    include_residual: bool = Query(False),
    index_range: Optional[str] = Query(None, alias="range"),
):
    _get_job_owned(session, job_id, owner_session_id)
    store = JobStore(session=session)

    label = f"track:{int(track_index)}"
    arts = store.list_artifacts(job_id, kind=ArtifactKind.track_npy, label=label, limit=1)
    if not arts:
        raise HTTPException(
            status_code=404,
            detail="Track data not stored. Enable track_detail.store_npy in config.",
        )

    track_bytes = artifact_store.get_bytes(arts[0].blob_path)

    job = store.get_job(job_id)
    config = _effective_pipeline_config(job)
    order = _track_xy_order_from_config(config)
    frame, position = _load_track_frame_position_from_bytes(track_bytes, order=order)
    analysis_mode = resolve_analysis_mode(config)
    track_model = session.exec(
        select(Track).where(Track.job_id == job_id, Track.track_index == int(track_index))
    ).first()

    if analysis_mode == RIPPLE_ANALYSIS_MODE:
        metrics = dict((track_model.metrics if track_model else {}) or {})
        slope = float(metrics.get("slope_px_per_frame", 0.0) or 0.0)
        intercept = float(metrics.get("line_intercept_px", position[0] if position.size else 0.0) or 0.0)
        baseline = (slope * frame + intercept).astype(float)
        residual = (position - baseline).astype(float)
        lo, hi = 0, len(frame) - 1
        if index_range:
            lo, hi = _parse_index_range(index_range, len(frame))
        return {
            "track_index": int(track_index),
            "analysis_mode": RIPPLE_ANALYSIS_MODE,
            "coords": {"poly_format": "[x, y]", "x_name": "time_index", "y_name": "position_px"},
            "time_index": frame[lo : hi + 1].tolist(),
            "position": position[lo : hi + 1].tolist(),
            "baseline": baseline[lo : hi + 1].tolist(),
            "residual": residual[lo : hi + 1].tolist() if include_residual else None,
            "sine_fit": None,
            "regression": {"method": "robust_linear", "slope": slope, "intercept": intercept},
            "peaks": [],
            "peaks_in_slice": [],
            "peak_points": [],
            "peak_regressions": [],
            "strongest_peak_idx": None,
            "metrics": metrics,
        }

    detrend_cfg = (config.get("detrend") or {}).copy()
    degree = int(detrend_cfg.pop("degree", 1))
    model = fit_baseline_ransac(frame, position, degree=degree, **detrend_cfg)
    baseline = model.predict(frame.reshape(-1, 1)).astype(float)
    residual = (position - baseline).astype(float)

    peaks_cfg = (config.get("peaks") or {})
    period_cfg = dict(config.get("period") or {})
    io_cfg = (config.get("io") or {})
    sampling_rate = float(io_cfg.get("sampling_rate", period_cfg.get("sampling_rate", 1.0)))
    period_cfg.setdefault("sampling_rate", sampling_rate)

    try:
        freq = float(estimate_dominant_frequency(residual, **period_cfg))
    except Exception:
        freq = float("nan")
    freq = resolve_positive_frequency(
        freq,
        frame=frame,
        sampling_rate=sampling_rate,
        min_freq=period_cfg.get("min_freq"),
        max_freq=period_cfg.get("max_freq"),
    )
    period = float(frequency_to_period(freq)) if (isinstance(freq, float) and math.isfinite(freq) and freq > 0) else float("nan")

    frames_per_period = (sampling_rate / float(freq)) if (sampling_rate and math.isfinite(freq) and freq > 0) else None
    large_wave_event_cfg = (
        (((config.get("analysis") or {}).get("large_wave") or {}).get("events") or {})
    )
    peak_events: List[Dict[str, Any]] = []
    if analysis_mode == LARGE_WAVE_ANALYSIS_MODE and track_model is not None:
        large_wave_rows = session.exec(
            select(Wave)
            .where(Wave.job_id == job_id, Wave.track_id == track_model.id)
            .order_by(Wave.wave_index.asc())
        ).all()
        peak_events = _large_wave_peak_events_for_detail(
            large_wave_rows,
            residual,
            width_multiplier=float(
                large_wave_event_cfg.get("fit_window_width_multiplier", 1.0)
            ),
            boundary_smoothing_sigma_rows=float(
                large_wave_event_cfg.get("fit_boundary_smoothing_sigma_rows", 4.0)
            ),
        )
        track_metrics = dict(track_model.metrics or {})
        try:
            stored_frequency = float(track_metrics.get("large_wave_frequency_hz"))
        except (TypeError, ValueError):
            stored_frequency = float("nan")
        if math.isfinite(stored_frequency) and stored_frequency > 0:
            freq = stored_frequency
            period = float(frequency_to_period(freq))
    if not peak_events:
        peak_sets = _detect_peak_sets_for_detail(residual, peaks_cfg, frames_per_period)
        for peak_set in peak_sets:
            signal = np.asarray(peak_set["signal"], dtype=float)
            sign = int(peak_set["sign"])
            for peak_i_raw in np.asarray(peak_set["peaks_idx"], dtype=int).tolist():
                peak_i = int(peak_i_raw)
                if peak_i < 0 or peak_i >= len(frame):
                    continue
                peak_events.append({
                    "peak_i": peak_i,
                    "event_kind": str(peak_set["event_kind"]),
                    "event_polarity": str(peak_set["event_polarity"]),
                    "fit_signal_sign": sign,
                    "event_amplitude": float(signal[peak_i]),
                })
    peak_events.sort(key=lambda event: (int(event["peak_i"]), 0 if event["event_kind"] == "max" else 1))
    peaks_idx = np.asarray([int(event["peak_i"]) for event in peak_events], dtype=int)

    lo, hi = 0, len(frame) - 1
    if index_range:
        lo, hi = _parse_index_range(index_range, len(frame))

    strongest_peak_idx: Optional[int] = None
    if peak_events:
        try:
            strongest_event = max(
                peak_events,
                key=lambda event: (
                    float(event["event_amplitude"])
                    if math.isfinite(float(event["event_amplitude"]))
                    else float("-inf")
                ),
            )
            strongest_peak_idx = int(strongest_event["peak_i"])
        except Exception:
            strongest_peak_idx = int(peak_events[0]["peak_i"])

    def peak_point(ordinal: int, event: Dict[str, Any]) -> Dict[str, Any]:
        peak_i = int(event["peak_i"])
        in_slice = bool(lo <= peak_i <= hi)
        return {
            "peak_index": int(ordinal),
            "peak_i": int(peak_i),
            "frame": float(frame[peak_i]),
            "position": float(position[peak_i]),
            "amplitude": float(residual[peak_i]),
            "event_amplitude": float(event["event_amplitude"]),
            "event_kind": str(event["event_kind"]),
            "event_polarity": str(event["event_polarity"]),
            "fit_signal_sign": int(event["fit_signal_sign"]),
            "fit_window_frames": event.get("fit_window_frames"),
            "fit_window_lo": event.get("fit_window_lo"),
            "fit_window_hi": event.get("fit_window_hi"),
            "fit_window_source": event.get("fit_window_source"),
            "in_slice": in_slice,
            "slice_index": int(peak_i - lo) if in_slice else None,
            "is_strongest": bool(strongest_peak_idx is not None and int(peak_i) == strongest_peak_idx),
        }

    peak_points = [peak_point(i + 1, event) for i, event in enumerate(peak_events)]
    peak_regressions: List[Dict[str, Any]] = []
    sine_fit = None
    if include_sine:
        fit_freq = float(freq) if math.isfinite(freq) else float("nan")
        period_frac = float((config.get("features") or {}).get("fit_window_period_frac", 0.5))
        for point in peak_points:
            peak_i = int(point["peak_i"])
            sign = int(point.get("fit_signal_sign", 1))
            fit_signal = residual.astype(float, copy=False) * float(sign)
            fit_result = None
            shared_large_fit = None
            if (
                analysis_mode == LARGE_WAVE_ANALYSIS_MODE
                and point.get("fit_window_lo") is not None
                and point.get("fit_window_hi") is not None
            ):
                shared_large_fit = fit_large_wave(
                    frame=frame,
                    position=position,
                    global_baseline=baseline,
                    center_idx=peak_i,
                    event_kind=str(point.get("event_kind", "max")),
                    sampling_rate=sampling_rate,
                    endpoint_anchor_rows=int(large_wave_event_cfg.get("endpoint_anchor_rows", 7)),
                    curvature_half_window_rows=int(
                        large_wave_event_cfg.get("curvature_half_window_rows", 8)
                    ),
                    max_period_boundary_error_fraction=float(
                        large_wave_event_cfg.get("max_period_boundary_error_fraction", 0.5)
                    ),
                    fixed_window=(int(point["fit_window_lo"]), int(point["fit_window_hi"])),
                    refine_apex=False,
                )
            if shared_large_fit is None:
                point_fit_freq = fit_freq
                if analysis_mode == LARGE_WAVE_ANALYSIS_MODE:
                    width_frequency = _large_wave_fit_frequency(
                        width_frames=point.get("fit_window_frames"),
                        sampling_rate=sampling_rate,
                        period_frac=period_frac,
                        width_multiplier=1.0,
                    )
                    if width_frequency is not None:
                        point_fit_freq = width_frequency
                fit_result = _fit_anchored_sine(
                    fit_signal,
                    frame,
                    point_fit_freq,
                    peak_i,
                    sampling_rate=sampling_rate,
                    period_frac=period_frac,
                )
            regression = dict(point)
            regression["sine_fit"] = None
            if shared_large_fit is not None:
                regression.update(shared_large_fit.metrics)
                regression["fit_window_source"] = point.get("fit_window_source")
                regression["sine_fit"] = shared_large_fit.fitted_position[lo : hi + 1].tolist()
                regression["fit_baseline"] = shared_large_fit.baseline[lo : hi + 1].tolist()
                if strongest_peak_idx is not None and peak_i == strongest_peak_idx:
                    sine_fit = shared_large_fit.fitted_position
            elif fit_result is not None:
                yfit_signed, fit_meta = fit_result
                yfit_res = yfit_signed * float(sign)
                full_fit = (baseline + yfit_res).astype(float)
                regression.update(_detail_fit_meta_for_original_polarity(
                    fit_meta,
                    sign=sign,
                    original_peak_value=float(residual[peak_i]),
                ))
                regression["fit_window_source"] = point.get("fit_window_source")
                regression["fit_window_width_frames"] = float(
                    int(fit_meta["fit_window_hi"]) - int(fit_meta["fit_window_lo"])
                )
                regression["sine_fit"] = full_fit[lo : hi + 1].tolist()
                if strongest_peak_idx is not None and peak_i == strongest_peak_idx:
                    sine_fit = full_fit
            peak_regressions.append(regression)

    frame_view = frame[lo : hi + 1]
    baseline_view = baseline[lo : hi + 1]
    residual_view = residual[lo : hi + 1] if include_residual else None
    sine_view = sine_fit[lo : hi + 1] if sine_fit is not None else None
    peaks_in_slice = [int(event["peak_i"]) for event in peak_events if lo <= int(event["peak_i"]) <= hi]

    event_amps = np.asarray([float(event["event_amplitude"]) for event in peak_events], dtype=float)
    event_amps = event_amps[np.isfinite(event_amps)]
    if event_amps.size > 0:
        mean_amp = float(event_amps.mean())
    else:
        mean_amp = float("nan")

    return {
        "track_index": int(track_index),
        "analysis_mode": analysis_mode,
        "coords": {"poly_format": "[x, y]", "x_name": "time_index", "y_name": "position_px"},
        "time_index": frame_view.tolist(),
        "position": position[lo : hi + 1].tolist(),
        "baseline": baseline_view.tolist(),
        "residual": (residual_view.tolist() if residual_view is not None else None),
        "sine_fit": (sine_view.tolist() if sine_view is not None else None),
        "regression": {"method": "ransac_poly", "degree": degree, "params": detrend_cfg},
        "peaks": [int(i) for i in peaks_idx.tolist()],
        "peaks_in_slice": peaks_in_slice,
        "peak_points": peak_points,
        "peak_regressions": peak_regressions,
        "strongest_peak_idx": strongest_peak_idx,
        "metrics": {
            "dominant_frequency": freq if math.isfinite(freq) else None,
            "period": period if math.isfinite(period) else None,
            "num_peaks": int(len(peaks_idx)),
            "num_maxima": int(sum(1 for event in peak_events if event["event_kind"] == "max")),
            "num_minima": int(sum(1 for event in peak_events if event["event_kind"] == "min")),
            "event_polarity": (
                "both"
                if analysis_mode == LARGE_WAVE_ANALYSIS_MODE
                else _normalize_detail_event_polarity(peaks_cfg.get("event_polarity", peaks_cfg.get("polarity", "both")))
            ),
            "mean_amplitude": mean_amp if math.isfinite(mean_amp) else None,
        },
    }


@router.get("/jobs/{job_id}/artifacts/{artifact_id}/download")
def download_artifact(
    job_id: UUID,
    artifact_id: UUID,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
):
    """
    Streams artifact bytes through the backend (works for LocalArtifactStore and as fallback for GCS).
    Prefer signed URLs when available.
    """
    art = _get_artifact_owned(session, job_id, artifact_id, owner_session_id)
    data = artifact_store.get_bytes(art.blob_path)

    media = art.content_type or "application/octet-stream"
    filename = (art.label or art.kind.value or "artifact").replace(":", "_")
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
    }
    return Response(content=data, media_type=media, headers=headers)


@router.get("/jobs/{job_id}/ripple/{export_name}.csv")
def export_ripple_csv(
    job_id: UUID,
    export_name: str,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
) -> Response:
    _get_job_owned(session, job_id, owner_session_id)
    labels = {
        "tracks": "ripple_tracks",
        "intervals": "ripple_intervals",
        "families": "ripple_families",
    }
    label = labels.get(export_name.strip().lower())
    if label is None:
        raise HTTPException(status_code=404, detail="Unknown ripple export")
    artifact = session.exec(
        select(Artifact)
        .where(Artifact.job_id == job_id, Artifact.kind == ArtifactKind.other, Artifact.label == label)
        .order_by(Artifact.created_at.desc())
        .limit(1)
    ).first()
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Ripple {export_name} export is not available")
    data = artifact_store.get_bytes(artifact.blob_path)
    filename = ((artifact.meta or {}).get("filename") if isinstance(artifact.meta, dict) else None) or {
        "tracks": "tracks.csv",
        "intervals": "waves.csv",
        "families": "families.csv",
    }[export_name.strip().lower()]
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=data, media_type="text/csv", headers=headers)


@router.get("/jobs/{job_id}/waves.csv")
def export_waves_csv(
    job_id: UUID,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
) -> StreamingResponse:
    job = _get_job_owned(session, job_id, owner_session_id)
    job_config = job.config or {}
    peaks_cfg = (job_config.get("peaks") or {}) if isinstance(job_config, dict) else {}
    endpoint_cfg = (
        (((job_config.get("kymo") or {}).get("onnx") or {}).get("postproc") or {}).get("endpoint_link") or {}
    ) if isinstance(job_config, dict) else {}
    config_event_polarity = peaks_cfg.get("event_polarity", peaks_cfg.get("polarity", ""))
    endpoint_link_enabled = endpoint_cfg.get("enabled", "")
    endpoint_link_level = endpoint_cfg.get("level", "")
    analysis_mode = resolve_analysis_mode(job_config)

    q = select(Wave).where(Wave.job_id == job_id).order_by(Wave.created_at.asc())
    rows = session.exec(q).all()

    headers = [
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

    def metric(row: Wave, key: str, default=None):
        metrics = row.metrics or {}
        value = metrics.get(key, default)
        return "" if value is None else value

    def attr_or_metric(row: Wave, attr: str, key: Optional[str] = None):
        value = getattr(row, attr, None)
        if value is not None:
            return value
        return metric(row, key or attr)

    def gen():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(headers)
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        for r in rows:
            frame1 = metric(r, "frame1")
            frame2 = metric(r, "frame2")
            pos1 = metric(r, "pos1_px")
            pos2 = metric(r, "pos2_px")
            freq = metric(r, "frequency_hz", r.frequency)
            signed_amplitude = metric(
                r,
                "signed_amplitude_px",
                metric(r, "amplitude_px", r.amplitude),
            )
            try:
                amplitude = abs(float(signed_amplitude))
            except (TypeError, ValueError):
                amplitude = signed_amplitude
            period_source = metric(r, "period_source")
            if period_source == "" and analysis_mode != LARGE_WAVE_ANALYSIS_MODE:
                period_source = "sine_fit"
            w.writerow([
                r.id,
                r.track_id or "",
                r.wave_index,
                frame1,
                frame2,
                metric(r, "period_frames"),
                metric(r, "period_s", r.period),
                freq,
                period_source,
                pos1,
                pos2,
                amplitude,
                signed_amplitude,
                pos1,
                pos2,
                frame1,
                frame2,
                r.t_start if r.t_start is not None else metric(r, "frame1_seconds"),
                r.t_end if r.t_end is not None else metric(r, "frame2_seconds"),
                metric(r, "seconds_delta"),
                metric(r, "delta_pos_px"),
                metric(r, "velocity_px_per_s"),
                freq,
                metric(r, "wavelength_px"),
                metric(r, "peak_frame_y_axis"),
                metric(r, "peak_position_x_axis"),
                attr_or_metric(r, "event_kind"),
                attr_or_metric(r, "event_polarity"),
                metric(r, "event_value"),
                metric(r, "peak_value_original"),
                attr_or_metric(r, "fit_target"),
                metric(r, "compare_fit_targets"),
                metric(r, "peak_frame_raw"),
                metric(r, "peak_position_raw"),
                metric(r, "frame1_raw"),
                metric(r, "frame2_raw"),
                r.error if r.error is not None else metric(r, "fit_error_vnmse"),
                metric(r, "fit_passes_peak"),
                metric(r, "fit_r2"),
                metric(r, "fit_rmse_px"),
                metric(r, "fit_nrmse"),
                metric(r, "fit_mae_px"),
                metric(r, "fit_points"),
                metric(r, "residual_fit_error_vnmse"),
                metric(r, "residual_fit_r2"),
                metric(r, "residual_fit_rmse_px"),
                metric(r, "raw_fit_error_vnmse"),
                metric(r, "raw_fit_r2"),
                metric(r, "raw_fit_rmse_px"),
                metric(r, "track_fit_error_median"),
                metric(r, "track_fit_r2_median"),
                metric(r, "period_consistency_cv"),
                metric(r, "frequency_agreement_error"),
                metric(r, "spectral_snr"),
                metric(r, "peak_prominence_snr"),
                config_event_polarity,
                endpoint_link_enabled,
                endpoint_link_level,
                metric(r, "fit_start_frame"),
                metric(r, "fit_end_frame"),
                metric(r, "fit_duration_frames"),
                metric(r, "fit_duration_s"),
                metric(r, "period_asymmetry"),
                metric(r, "period_boundary_error_fraction"),
                metric(r, "period_estimate_valid"),
                metric(r, "recurrence_period_frames"),
                metric(r, "recurrence_period_s"),
                metric(r, "recurrence_frequency_hz"),
                metric(r, "wave_type"),
                metric(r, "type_score"),
            ])
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    return StreamingResponse(gen(), media_type="text/csv")
