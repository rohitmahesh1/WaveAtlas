from __future__ import annotations

import csv
import io
import math
import os
import tempfile
from datetime import datetime
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
from ..signal.detrend import fit_baseline_ransac
from ..signal.peaks import detect_peaks, detect_peaks_adaptive
from ..signal.period import estimate_dominant_frequency, frequency_to_period

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
        .where(Artifact.job_id == job_id, Artifact.kind == ArtifactKind.upload_csv)
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
    safe = safe or "upload.csv"
    pfx = _artifact_prefix()
    if pfx:
        return f"{pfx}/jobs/{job_id}/upload/{safe}"
    return f"jobs/{job_id}/upload/{safe}"


def _ensure_upload_exists(job_store: JobStore, job_id: UUID) -> None:
    arts = job_store.list_artifacts(job_id, kind=ArtifactKind.upload_csv, limit=1)
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


def _load_track_xy_from_bytes(data: bytes, *, order: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        if order == "yx":
            return arr[:, 1].astype(float, copy=False), arr[:, 0].astype(float, copy=False)
        return arr[:, 0].astype(float, copy=False), arr[:, 1].astype(float, copy=False)
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


def _fit_anchored_sine(
    residual: np.ndarray,
    t: np.ndarray,
    freq: float,
    center_idx: Optional[int],
) -> Optional[np.ndarray]:
    if center_idx is None or center_idx < 0 or center_idx >= len(t):
        return None
    if not (isinstance(freq, float) and math.isfinite(freq) and freq > 0):
        return None
    omega = 2.0 * math.pi * float(freq)
    t0 = float(t[int(center_idx)])
    phi = (math.pi / 2.0) - omega * t0
    s = np.sin(omega * t + phi).astype(np.float64)
    X = np.vstack([s, np.ones_like(s)]).T
    y = residual.astype(np.float64)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    A = float(beta[0])
    c = float(beta[1])
    yfit = (A * s + c).astype(np.float64)
    return yfit


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
    path = Path("docs/config.md")
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

    upload_url = blob.create_resumable_upload_session(content_type=content_type)

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

    art = store.create_artifact(
        job_id=job_id,
        kind=ArtifactKind.upload_csv,
        blob_path=payload.blob_path,
        label="upload",
        content_type=payload.content_type or "text/csv",
        byte_size=payload.byte_size,
        meta={
            "filename": payload.filename,
            "uploaded_at": datetime.utcnow().isoformat(),
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
            kind=ArtifactKind.upload_csv.value,
            filename=filename,
            local_path=str(tmp_path),
            content_type=content_type,
            label="upload",
        )

        art = store.create_artifact(
            job_id=job_id,
            kind=ArtifactKind.upload_csv,
            blob_path=blob_path,
            label="upload",
            content_type=content_type,
            byte_size=stored_size,
            meta={"filename": filename, "uploaded_at": datetime.utcnow().isoformat(), "upload_method": "api_stream"},
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
    x, y = _load_track_xy_from_bytes(track_bytes, order=order)

    detrend_cfg = (config.get("detrend") or {}).copy()
    degree = int(detrend_cfg.pop("degree", 1))
    model = fit_baseline_ransac(x, y, degree=degree, **detrend_cfg)
    baseline = model.predict(x.reshape(-1, 1)).astype(float)
    residual = (y - baseline).astype(float)

    peaks_cfg = (config.get("peaks") or {})
    period_cfg = dict(config.get("period") or {})
    io_cfg = (config.get("io") or {})
    sampling_rate = float(io_cfg.get("sampling_rate", period_cfg.get("sampling_rate", 1.0)))
    period_cfg.setdefault("sampling_rate", sampling_rate)

    try:
        freq = float(estimate_dominant_frequency(residual, **period_cfg))
    except Exception:
        freq = float("nan")
    period = float(frequency_to_period(freq)) if (isinstance(freq, float) and math.isfinite(freq) and freq > 0) else float("nan")

    frames_per_period = (sampling_rate / float(freq)) if (sampling_rate and math.isfinite(freq) and freq > 0) else None
    peaks_idx, _props = _detect_peaks_for_detail(residual, peaks_cfg, frames_per_period)
    peaks_idx = np.asarray(peaks_idx, dtype=int)

    strongest_peak_idx: Optional[int] = None
    if peaks_idx.size > 0:
        try:
            strongest_peak_idx = int(peaks_idx[int(np.argmax(residual[peaks_idx]))])
        except Exception:
            strongest_peak_idx = int(peaks_idx[0])

    sine_fit = None
    if include_sine:
        yfit_res = _fit_anchored_sine(residual, x, float(freq) if math.isfinite(freq) else float("nan"), strongest_peak_idx)
        if yfit_res is not None:
            sine_fit = (baseline + yfit_res).astype(float)

    lo, hi = 0, len(x) - 1
    if index_range:
        lo, hi = _parse_index_range(index_range, len(x))

    x_view = x[lo : hi + 1]
    baseline_view = baseline[lo : hi + 1]
    residual_view = residual[lo : hi + 1] if include_residual else None
    sine_view = sine_fit[lo : hi + 1] if sine_fit is not None else None
    peaks_in_slice = [int(i) for i in peaks_idx.tolist() if lo <= int(i) <= hi]

    if peaks_idx.size > 0:
        try:
            mean_amp = float(residual[peaks_idx].mean())
        except Exception:
            mean_amp = float("nan")
    else:
        mean_amp = float("nan")

    return {
        "track_index": int(track_index),
        "coords": {"poly_format": "[x, y]", "x_name": "time_index", "y_name": "position_px"},
        "time_index": x_view.tolist(),
        "position": y[lo : hi + 1].tolist(),
        "baseline": baseline_view.tolist(),
        "residual": (residual_view.tolist() if residual_view is not None else None),
        "sine_fit": (sine_view.tolist() if sine_view is not None else None),
        "regression": {"method": "ransac_poly", "degree": degree, "params": detrend_cfg},
        "peaks": [int(i) for i in peaks_idx.tolist()],
        "peaks_in_slice": peaks_in_slice,
        "strongest_peak_idx": strongest_peak_idx,
        "metrics": {
            "dominant_frequency": freq if math.isfinite(freq) else None,
            "period": period if math.isfinite(period) else None,
            "num_peaks": int(len(peaks_idx)),
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


@router.get("/jobs/{job_id}/waves.csv")
def export_waves_csv(
    job_id: UUID,
    response: Response,
    owner_session_id: UUID = Depends(get_owner_session_id),
    session: Session = Depends(get_db_session),
) -> StreamingResponse:
    _get_job_owned(session, job_id, owner_session_id)

    q = select(Wave).where(Wave.job_id == job_id).order_by(Wave.created_at.asc())
    rows = session.exec(q).all()

    def gen():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["wave_id", "wave_index", "x", "y", "amplitude", "frequency", "period", "error", "t_start", "t_end"])
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        for r in rows:
            w.writerow([r.id, r.wave_index, r.x, r.y, r.amplitude, r.frequency, r.period, r.error, r.t_start, r.t_end])
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    return StreamingResponse(gen(), media_type="text/csv")
