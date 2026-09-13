from __future__ import annotations

from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence
from uuid import UUID

from sqlmodel import Session, select

from .artifact_store import ArtifactStore
from .cancel import CancellationRequested
from .job_store import JobStore
from .models import Artifact, ArtifactKind, Peak, Track, Wave
from .time_utils import utc_now_iso


RUN_MANIFEST_SCHEMA_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return None
    return int(value) if isinstance(value, int) else number


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=canonical_json_bytes)
    return {
        "count": len(ordered),
        "sha256": sha256_bytes(canonical_json_bytes(ordered)),
    }


def database_result_digests(session: Session, job_id: UUID) -> Dict[str, Dict[str, Any]]:
    tracks = list(
        session.exec(
            select(Track).where(Track.job_id == job_id).order_by(Track.track_index.asc())
        ).all()
    )
    track_index_by_id = {track.id: int(track.track_index) for track in tracks}
    track_rows = [
        {
            "track_index": int(track.track_index),
            "x0": track.x0,
            "y0": track.y0,
            "amplitude": track.amplitude,
            "frequency": track.frequency,
            "error": track.error,
            "metrics": track.metrics or {},
        }
        for track in tracks
    ]

    waves = list(session.exec(select(Wave).where(Wave.job_id == job_id)).all())
    wave_rows = [
        {
            "track_index": track_index_by_id.get(wave.track_id),
            "wave_index": wave.wave_index,
            "event_polarity": wave.event_polarity,
            "event_kind": wave.event_kind,
            "fit_target": wave.fit_target,
            "x": wave.x,
            "y": wave.y,
            "amplitude": wave.amplitude,
            "frequency": wave.frequency,
            "period": wave.period,
            "error": wave.error,
            "t_start": wave.t_start,
            "t_end": wave.t_end,
            "metrics": wave.metrics or {},
        }
        for wave in waves
    ]

    peaks = list(session.exec(select(Peak).where(Peak.job_id == job_id)).all())
    peak_rows = [
        {
            "track_index": track_index_by_id.get(peak.track_id),
            "pos": peak.pos,
            "value": peak.value,
            "event_polarity": peak.event_polarity,
            "event_kind": peak.event_kind,
            "fit_target": peak.fit_target,
            "metrics": peak.metrics or {},
        }
        for peak in peaks
    ]
    return {
        "tracks": _digest_rows(track_rows),
        "waves": _digest_rows(wave_rows),
        "peaks": _digest_rows(peak_rows),
    }


def software_identity(config: Dict[str, Any]) -> Dict[str, Any]:
    desktop_build = dict(config.get("desktop_build") or {})
    if desktop_build:
        return {
            "version": desktop_build.get("version"),
            "commit": desktop_build.get("commit"),
            "python": desktop_build.get("python"),
            "model_release": desktop_build.get("model_release"),
            "models": desktop_build.get("models") or {},
            "packages": desktop_build.get("packages") or {},
        }
    return {
        "version": os.getenv("WAVEATLAS_VERSION", "development"),
        "commit": os.getenv("WAVEATLAS_BUILD_COMMIT"),
        "python": platform.python_version(),
        "model_release": None,
        "models": {},
        "packages": {},
    }


def build_run_manifest(
    *,
    job_id: UUID,
    analysis_mode: str,
    config: Dict[str, Any],
    input_identity: Dict[str, Any],
    database_outputs: Dict[str, Dict[str, Any]],
    artifact_outputs: Iterable[Dict[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": str(job_id),
        "generated_at": generated_at,
        "analysis_mode": analysis_mode,
        "input": _json_safe(input_identity),
        "method": {"config": _json_safe(config)},
        "software": software_identity(config),
        "outputs": {
            "database": database_outputs,
            "artifacts": sorted(
                (_json_safe(row) for row in artifact_outputs),
                key=lambda row: (str(row.get("kind")), str(row.get("label"))),
            ),
        },
    }


def publish_run_manifest_artifact(
    *,
    job_id: UUID,
    analysis_mode: str,
    config: Dict[str, Any],
    heatmap_meta: Optional[Mapping[str, Any]],
    job_store: JobStore,
    artifact_store: ArtifactStore,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Artifact:
    """Build, stage, and atomically replace the canonical run manifest."""

    def check_cancel() -> None:
        if cancel_cb is not None and cancel_cb():
            raise CancellationRequested("cancel_requested_during_result_manifest")

    check_cancel()
    uploads = [
        *job_store.list_artifacts(job_id, kind=ArtifactKind.upload_image, limit=10),
        *job_store.list_artifacts(job_id, kind=ArtifactKind.upload_csv, limit=10),
    ]
    if not uploads:
        raise RuntimeError("Cannot create result manifest without an input artifact")
    uploads.sort(key=lambda artifact: artifact.created_at)
    source_id = str((heatmap_meta or {}).get("source_artifact_id") or "")
    upload = next((item for item in uploads if str(item.id) == source_id), uploads[0])
    source_sha256 = str((heatmap_meta or {}).get("source_sha256") or "")
    if not source_sha256:
        source_sha256 = sha256_bytes(artifact_store.get_bytes(upload.blob_path))

    artifact_outputs: list[Dict[str, Any]] = []
    for artifact in job_store.list_artifacts(job_id, limit=10000):
        check_cancel()
        if artifact.kind in {ArtifactKind.upload_csv, ArtifactKind.upload_image}:
            continue
        if artifact.label == "result_manifest":
            continue
        artifact_meta = dict(artifact.meta or {})
        digest = str(artifact_meta.get("sha256") or "")
        if not digest:
            digest = sha256_bytes(artifact_store.get_bytes(artifact.blob_path))
        artifact_outputs.append({
            "kind": artifact.kind.value,
            "label": artifact.label,
            "filename": artifact_meta.get("filename"),
            "content_type": artifact.content_type,
            "byte_size": artifact.byte_size,
            "sha256": digest,
        })

    input_meta = dict(upload.meta or {})
    manifest = build_run_manifest(
        job_id=job_id,
        analysis_mode=analysis_mode,
        config=config,
        input_identity={
            "artifact_kind": upload.kind.value,
            "filename": input_meta.get("filename"),
            "content_type": upload.content_type,
            "byte_size": upload.byte_size,
            "sha256": source_sha256,
        },
        database_outputs=database_result_digests(job_store.session, job_id),
        artifact_outputs=artifact_outputs,
        generated_at=utc_now_iso(),
    )
    data = canonical_json_bytes(manifest)
    manifest_sha256 = sha256_bytes(data)
    blob_path, byte_size = artifact_store.put_bytes(
        job_id=job_id,
        kind=ArtifactKind.other.value,
        filename="run-manifest.json",
        data=data,
        content_type="application/json",
        label=f"result_manifest-staging-{os.urandom(16).hex()}",
    )
    committed = False
    try:
        artifact, old_blob_paths = job_store.replace_artifact(
            job_id=job_id,
            kind=ArtifactKind.other,
            label="result_manifest",
            blob_path=blob_path,
            content_type="application/json",
            byte_size=byte_size,
            meta={
                "filename": "run-manifest.json",
                "schema_version": manifest["schema_version"],
                "sha256": manifest_sha256,
            },
        )
        committed = True
    finally:
        if not committed:
            artifact_store.delete_blob(blob_path)
    for old_blob_path in old_blob_paths:
        artifact_store.delete_blob(old_blob_path)
    return artifact
