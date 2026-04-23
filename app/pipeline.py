# app/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from .artifact_store import ArtifactStore
from .job_store import JobStore
from .models import ArtifactKind, EventType, JobStatus

from .io.image_to_heatmap import image_to_heatmap_bytes
from .io.table_to_heatmap import table_to_heatmap_bytes
from .extract_core import select_kymo_runner, process_track


@dataclass(frozen=True)
class PipelineSettings:
    scratch_root: Path
    db_batch_size: int = 50
    progress_every_secs: float = 2.0
    emit_overlay_every_tracks: int = 1  # overlay_track JobEvent cadence


class PipelineError(RuntimeError):
    pass


def run_job(
    job_id: UUID,
    *,
    job_store: JobStore,
    artifact_store: ArtifactStore,
    config: Dict[str, Any],
    settings: PipelineSettings,
    resume: bool = False,
) -> None:
    started_at = datetime.utcnow()

    scratch_dir = settings.scratch_root / str(job_id)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    def cancelled() -> bool:
        return job_store.is_cancel_requested(job_id)

    def emit(event_type: EventType, payload: Dict[str, Any]) -> None:
        job_store.append_event(job_id, event_type, payload)

    def user_log(message: str, *, stage: Optional[str] = None, level: str = "info") -> None:
        payload: Dict[str, Any] = {"message": message, "level": level}
        if stage:
            payload["stage"] = stage
        emit(EventType.user_log, payload)

    def set_progress(
        stage: str,
        *,
        processed: int = 0,
        total: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        progress: Dict[str, Any] = {
            "stage": stage,
            "processed": int(processed),
            "total": int(total),
            "pct": (float(processed) / float(total)) if total > 0 else 0.0,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if extra:
            progress.update(extra)
        job_store.update_progress(job_id, progress, replace=False, emit_event=True)

    def _coerce_meta(
        *,
        meta: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Support both names during refactors.
        if meta is not None:
            return dict(meta)
        if metadata is not None:
            return dict(metadata)
        return {}

    def publish_bytes(
        *,
        kind: ArtifactKind,
        filename: str,
        data: bytes,
        content_type: str,
        label: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta_dict = _coerce_meta(meta=meta, metadata=metadata)

        blob_path, byte_size = artifact_store.put_bytes(
            job_id=job_id,
            kind=kind.value,
            filename=filename,
            data=data,
            content_type=content_type,
            label=label,
        )
        art = job_store.create_artifact(
            job_id=job_id,
            kind=kind,
            blob_path=blob_path,
            label=label,
            content_type=content_type,
            byte_size=byte_size,
            meta=meta_dict,
        )

        if kind == ArtifactKind.overlay:
            signed = artifact_store.signed_url(
                blob_path, expires_in=int(os.getenv("SIGNED_URL_EXPIRES_SECS", "3600"))
            )
            download_url = signed or f"/api/jobs/{job_id}/artifacts/{art.id}/download"
            emit(
                EventType.overlay_ready,
                {
                    "artifact": {
                        "id": str(art.id),
                        "kind": kind.value,
                        "label": label,
                        "content_type": content_type,
                        "download_url": download_url,
                    }
                },
            )
        else:
            emit(EventType.progress, {"artifact": {"kind": kind.value, "label": label, "blob_path": blob_path}})

    def publish_file(
        *,
        kind: ArtifactKind,
        filename: str,
        local_path: Path,
        content_type: str,
        label: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta_dict = _coerce_meta(meta=meta, metadata=metadata)

        blob_path, byte_size = artifact_store.put_file(
            job_id=job_id,
            kind=kind.value,
            filename=filename,
            local_path=str(local_path),
            content_type=content_type,
            label=label,
        )
        art = job_store.create_artifact(
            job_id=job_id,
            kind=kind,
            blob_path=blob_path,
            label=label,
            content_type=content_type,
            byte_size=byte_size,
            meta=meta_dict,
        )

        if kind == ArtifactKind.overlay:
            signed = artifact_store.signed_url(
                blob_path, expires_in=int(os.getenv("SIGNED_URL_EXPIRES_SECS", "3600"))
            )
            download_url = signed or f"/api/jobs/{job_id}/artifacts/{art.id}/download"
            emit(
                EventType.overlay_ready,
                {
                    "artifact": {
                        "id": str(art.id),
                        "kind": kind.value,
                        "label": label,
                        "content_type": content_type,
                        "download_url": download_url,
                    }
                },
            )
        else:
            emit(EventType.progress, {"artifact": {"kind": kind.value, "label": label, "blob_path": blob_path}})

    def publish_debug_overlays(image_id: str, base_dir: Path) -> None:
        dbg = base_dir / "debug"
        if not dbg.exists():
            return

        file_map: List[Tuple[str, str, str]] = [
            ("prob", "prob.png", "image/png"),
            ("mask_raw", "mask_raw.png", "image/png"),
            ("mask_clean", "mask_clean.png", "image/png"),
            ("mask_filtered", "mask_filtered.png", "image/png"),
            ("skeleton", "skeleton.png", "image/png"),
            ("mask_hysteresis", "mask_hysteresis.png", "image/png"),
            ("stats", "stats.txt", "text/plain"),
        ]

        for overlay_name, fname, ctype in file_map:
            p = dbg / fname
            if p.exists():
                label = f"{image_id}:{overlay_name}"
                publish_file(
                    kind=ArtifactKind.overlay,
                    filename=f"{image_id}_{fname}",
                    local_path=p,
                    content_type=ctype,
                    label=label,
                    meta={"image_id": image_id, "overlay": overlay_name},
                )

        ot = base_dir / "overlay_tracks.png"
        if ot.exists():
            label = f"{image_id}:overlay_tracks"
            publish_file(
                kind=ArtifactKind.overlay,
                filename=f"{image_id}_overlay_tracks.png",
                local_path=ot,
                content_type="image/png",
                label=label,
                meta={"image_id": image_id, "overlay": "overlay_tracks"},
            )

    resume_cfg = (config.get("service") or {}).get("resume") or {}
    resume_enabled = bool(resume_cfg.get("enabled", False)) or bool(resume)

    try:
        # -----------------------------
        # Job init
        # -----------------------------
        job_store.set_status(job_id, JobStatus.in_progress, emit_event=True)
        set_progress("init")
        user_log("Starting analysis", stage="init")

        if resume_enabled:
            job_store.recompute_counts(job_id)

        if cancelled():
            job_store.set_status(job_id, JobStatus.cancelled, emit_event=True)
            emit(EventType.cancelled, {"reason": "cancel_requested_before_start"})
            return

        # -----------------------------
        # Base heatmap (resume-aware)
        # -----------------------------
        heatmap_png: Optional[bytes] = None
        heatmap_meta: Optional[Dict[str, Any]] = None
        if resume_enabled:
            existing = job_store.list_artifacts(
                job_id, kind=ArtifactKind.base_heatmap, label="base_heatmap", limit=1
            )
            if existing:
                heatmap_png = artifact_store.get_bytes(existing[0].blob_path)
                set_progress("heatmap_ready", extra={"resume": True})
                user_log("Using existing heatmap", stage="heatmap_ready")

        if heatmap_png is None:
            # -----------------------------
            # Load uploaded input from artifact store
            # -----------------------------
            user_log("Loading input", stage="load_input")
            uploads = [
                *job_store.list_artifacts(job_id, kind=ArtifactKind.upload_image, limit=10),
                *job_store.list_artifacts(job_id, kind=ArtifactKind.upload_csv, limit=10),
            ]
            if not uploads:
                raise PipelineError("No upload artifact found (expected table or image upload)")
            uploads.sort(key=lambda art: art.created_at)

            upload = uploads[0]
            input_bytes = artifact_store.get_bytes(upload.blob_path)
            input_filename = (upload.meta or {}).get("filename")

            loaded_stage = "image_loaded" if upload.kind == ArtifactKind.upload_image else "table_loaded"
            set_progress(loaded_stage, extra={"upload_blob": upload.blob_path, "input_kind": upload.kind.value})
            emit(EventType.progress, {"stage": loaded_stage, "bytes": len(input_bytes), "input_kind": upload.kind.value})

            if cancelled():
                job_store.set_status(job_id, JobStatus.cancelled, emit_event=True)
                emit(EventType.cancelled, {"reason": "cancel_requested_after_input_loaded"})
                return

            # -----------------------------
            # Input -> base heatmap
            # -----------------------------
            user_log("Generating heatmap", stage="heatmap")
            if upload.kind == ArtifactKind.upload_image:
                heatmap_png, heatmap_meta = image_to_heatmap_bytes(
                    input_bytes,
                    config=config,
                    filename_hint=str(input_filename) if input_filename else None,
                )
            else:
                heatmap_png, heatmap_meta = table_to_heatmap_bytes(
                    input_bytes,
                    config=config,
                    filename_hint=str(input_filename) if input_filename else None,
                )
            heatmap_meta = {
                **(heatmap_meta or {}),
                "source_artifact_id": str(upload.id),
                "source_artifact_kind": upload.kind.value,
            }
            publish_bytes(
                kind=ArtifactKind.base_heatmap,
                filename="base_heatmap.png",
                data=heatmap_png,
                content_type="image/png",
                label="base_heatmap",
                meta=heatmap_meta,
            )
            set_progress("heatmap_ready")
            user_log("Heatmap ready", stage="heatmap_ready")

        if heatmap_png is None:
            raise PipelineError("Heatmap bytes missing (resume or generation failed)")
        heatmap_path = scratch_dir / "base_heatmap.png"
        heatmap_path.write_bytes(heatmap_png)

        if cancelled():
            job_store.set_status(job_id, JobStatus.cancelled, emit_event=True)
            emit(EventType.cancelled, {"reason": "cancel_requested_after_heatmap"})
            return

        # -----------------------------
        # Heatmap -> tracks (kymo runner) or resume from artifacts
        # -----------------------------
        track_paths: List[Path] = []

        def _parse_track_index(meta: Dict[str, Any], label: Optional[str]) -> Optional[int]:
            idx = meta.get("track_index") if isinstance(meta, dict) else None
            if idx is not None:
                try:
                    return int(idx)
                except Exception:
                    return None
            if label and label.startswith("track:"):
                try:
                    return int(label.split(":", 1)[1])
                except Exception:
                    return None
            return None

        def _load_tracks_from_artifacts(total_tracks: int) -> Optional[List[Path]]:
            if total_tracks <= 0:
                return None
            arts = job_store.list_artifacts(
                job_id, kind=ArtifactKind.track_npy, limit=max(2000, total_tracks + 10)
            )
            mapping: Dict[int, Any] = {}
            for art in arts:
                idx = _parse_track_index(getattr(art, "meta", {}) or {}, art.label)
                if idx is None:
                    continue
                mapping[idx] = art

            if any(i not in mapping for i in range(total_tracks)):
                return None

            dest_dir = scratch_dir / "tracks_resumed"
            dest_dir.mkdir(parents=True, exist_ok=True)
            paths: List[Path] = []
            for i in range(total_tracks):
                art = mapping[i]
                data = artifact_store.get_bytes(art.blob_path)
                p = dest_dir / f"track_{i}.npy"
                p.write_bytes(data)
                paths.append(p)
            return paths

        def _load_track_manifest() -> Optional[Dict[str, Any]]:
            arts = job_store.list_artifacts(job_id, kind=ArtifactKind.track_manifest, label="tracks_manifest", limit=1)
            if not arts:
                return None
            try:
                raw = artifact_store.get_bytes(arts[0].blob_path)
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return None

        if resume_enabled:
            manifest = _load_track_manifest()
            if manifest and isinstance(manifest.get("total_tracks"), int):
                maybe_paths = _load_tracks_from_artifacts(int(manifest["total_tracks"]))
                if maybe_paths:
                    track_paths = maybe_paths
                    emit(EventType.progress, {"stage": "kymo_done", "tracks_found": len(track_paths), "resume": True})
                    user_log(f"Using cached tracks ({len(track_paths)})", stage="kymo_done")

        if not track_paths:
            user_log("Extracting tracks (KymoButler)", stage="kymo_start")
            set_progress("kymo_start", extra={"detail": "Extracting tracks"})
            runner = select_kymo_runner(config=config)

            kymo_stage_labels = {
                "load_image": "Loading heatmap",
                "segmenting": "Segmenting heatmap",
                "masking": "Cleaning mask",
                "skeletonizing": "Skeletonizing tracks",
                "tracking": "Tracing tracks",
                "refining": "Refining tracks",
                "deduping": "Removing duplicates",
                "scaling": "Scaling to original size",
                "saving": "Saving tracks",
            }
            last_kymo_stage: Optional[str] = None

            def kymo_progress(stage: str, data: Dict[str, Any]) -> None:
                nonlocal last_kymo_stage
                label = kymo_stage_labels.get(stage, stage)
                extra = dict(data) if data else None
                set_progress(f"kymo_{stage}", extra=extra)
                if stage != last_kymo_stage and stage in kymo_stage_labels:
                    user_log(label, stage=f"kymo_{stage}")
                    last_kymo_stage = stage

            kymo_out = runner.run(heatmap_path=heatmap_path, scratch_dir=scratch_dir, progress_cb=kymo_progress)

            image_id: str = kymo_out.image_id
            base_dir: Path = kymo_out.base_dir
            track_paths = list(kymo_out.track_paths)

            publish_debug_overlays(image_id, base_dir)
            emit(EventType.progress, {"stage": "kymo_done", "tracks_found": len(track_paths), "image_id": image_id})
            user_log(f"Found {len(track_paths)} tracks", stage="kymo_done")

            if not track_paths:
                raise PipelineError("Kymo runner produced no tracks")

            if resume_enabled:
                # Persist all tracks for resume
                existing = job_store.list_artifacts(
                    job_id, kind=ArtifactKind.track_npy, limit=max(2000, len(track_paths) + 10)
                )
                existing_idx = set(
                    idx for idx in (_parse_track_index(getattr(a, "meta", {}) or {}, a.label) for a in existing) if idx is not None
                )
                for track_index, track_path in enumerate(track_paths):
                    if track_index in existing_idx:
                        continue
                    label = f"track:{track_index}"
                    blob_path, byte_size = artifact_store.put_file(
                        job_id=job_id,
                        kind=ArtifactKind.track_npy.value,
                        filename=f"track_{track_index}.npy",
                        local_path=str(track_path),
                        content_type="application/octet-stream",
                        label=label,
                    )
                    job_store.create_artifact(
                        job_id=job_id,
                        kind=ArtifactKind.track_npy,
                        blob_path=blob_path,
                        label=label,
                        content_type="application/octet-stream",
                        byte_size=byte_size,
                        meta={"track_index": int(track_index)},
                    )

                manifest_payload = {"total_tracks": len(track_paths)}
                publish_bytes(
                    kind=ArtifactKind.track_manifest,
                    filename="tracks_manifest.json",
                    data=json.dumps(manifest_payload).encode("utf-8"),
                    content_type="application/json",
                    label="tracks_manifest",
                    meta=manifest_payload,
                )
        
        # -----------------------------
        # Process tracks -> DB rows + overlay events
        # -----------------------------
        user_log("Analyzing tracks", stage="processing_tracks")
        job_store.bump_counts(job_id, tracks_total=len(track_paths))

        processed_set = set(job_store.get_processed_track_indices(job_id)) if resume_enabled else set()
        processed = len(processed_set)
        set_progress("processing_tracks", processed=processed, total=len(track_paths))

        waves_buf: List[Dict[str, Any]] = []
        peaks_buf: List[Dict[str, Any]] = []
        new_processed = 0
        batch_new_processed = 0
        last_progress_ts = datetime.utcnow()
        processing_started_at = datetime.utcnow()
        last_rate_ts = processing_started_at
        last_processed_for_rate = processed
        ema_rate_tps: Optional[float] = None
        ema_alpha = 0.2

        for track_index, track_path in enumerate(track_paths):
            if cancelled():
                job_store.set_status(job_id, JobStatus.cancelled, emit_event=True)
                emit(EventType.cancelled, {"reason": "cancel_requested_mid_run", "processed": processed})
                set_progress("cancelled", processed=processed, total=len(track_paths))
                return

            if resume_enabled and track_index in processed_set:
                processed += 1
                continue

            # Optional: persist raw track for detail view
            track_detail_cfg = (config.get("track_detail") or {})
            store_track_npy = bool(track_detail_cfg.get("store_npy", True)) or resume_enabled
            if store_track_npy:
                label = f"track:{track_index}"
                existing = job_store.list_artifacts(job_id, kind=ArtifactKind.track_npy, label=label, limit=1)
                if not existing:
                    blob_path, byte_size = artifact_store.put_file(
                        job_id=job_id,
                        kind=ArtifactKind.track_npy.value,
                        filename=f"track_{track_index}.npy",
                        local_path=str(track_path),
                        content_type="application/octet-stream",
                        label=label,
                    )
                    job_store.create_artifact(
                        job_id=job_id,
                        kind=ArtifactKind.track_npy,
                        blob_path=blob_path,
                        label=label,
                        content_type="application/octet-stream",
                        byte_size=byte_size,
                        meta={"track_index": int(track_index)},
                    )

            track_row, wave_rows, peak_rows, overlay_track = process_track(
                job_id=job_id,
                track_index=track_index,
                track_path=track_path,
                config=config,
            )

            # print("Processed track #", track_row, " Wave", wave_rows)

            track = job_store.upsert_track_by_index(
                job_id,
                track_index,
                processed_at=datetime.utcnow(),
                amplitude=track_row.get("amplitude"),
                frequency=track_row.get("frequency"),
                error=track_row.get("error"),
                x0=track_row.get("x0"),
                y0=track_row.get("y0"),
                metrics=track_row.get("metrics") or {},
                overlay=track_row.get("overlay") or {},
            )

            for row in wave_rows or []:
                row["track_id"] = track.id
            for row in peak_rows or []:
                row["track_id"] = track.id

            waves_buf.extend(wave_rows or [])
            peaks_buf.extend(peak_rows or [])

            processed += 1
            new_processed += 1
            batch_new_processed += 1

            if settings.emit_overlay_every_tracks > 0 and (new_processed % settings.emit_overlay_every_tracks == 0):
                emit(EventType.overlay_track, overlay_track)
                #print("emitting overlay")

            if settings.db_batch_size > 0 and (new_processed % settings.db_batch_size == 0):
                if waves_buf:
                    job_store.insert_waves_batch(job_id, waves_buf)
                    job_store.bump_counts(job_id, waves_done_delta=len(waves_buf))
                    waves_buf.clear()
                if peaks_buf:
                    job_store.insert_peaks_batch(job_id, peaks_buf)
                    job_store.bump_counts(job_id, peaks_done_delta=len(peaks_buf))
                    peaks_buf.clear()

                job_store.bump_counts(job_id, tracks_done_delta=batch_new_processed)
                batch_new_processed = 0

            now = datetime.utcnow()
            if (now - last_progress_ts).total_seconds() >= settings.progress_every_secs:
                rate_elapsed = (now - last_rate_ts).total_seconds()
                delta = processed - last_processed_for_rate
                inst_rate = (delta / rate_elapsed) if rate_elapsed > 0 else 0.0
                if inst_rate > 0:
                    ema_rate_tps = inst_rate if ema_rate_tps is None else (
                        ema_alpha * inst_rate + (1.0 - ema_alpha) * ema_rate_tps
                    )
                last_rate_ts = now
                last_processed_for_rate = processed

                remaining = max(0, len(track_paths) - processed)
                rate_use = ema_rate_tps or 0.0
                eta_secs = (remaining / rate_use) if rate_use > 0 else None
                extra = {
                    "rate_tps": rate_use,
                    "eta_secs": eta_secs,
                }
                set_progress("processing_tracks", processed=processed, total=len(track_paths), extra=extra)
                last_progress_ts = now

        if waves_buf:
            job_store.insert_waves_batch(job_id, waves_buf)
            job_store.bump_counts(job_id, waves_done_delta=len(waves_buf))
            waves_buf.clear()

        if peaks_buf:
            job_store.insert_peaks_batch(job_id, peaks_buf)
            job_store.bump_counts(job_id, peaks_done_delta=len(peaks_buf))
            peaks_buf.clear()

        # Final counts
        if batch_new_processed:
            job_store.bump_counts(job_id, tracks_done_delta=batch_new_processed)

        set_progress("completed", processed=len(track_paths), total=len(track_paths), extra={"eta_secs": 0.0})

        user_log("Completed", stage="completed")
        job_store.set_status(job_id, JobStatus.completed, emit_event=True)
        emit(EventType.done, {"ok": True, "duration_s": (datetime.utcnow() - started_at).total_seconds()})

    except Exception as e:
        user_log("Run failed", stage="failed", level="error")
        job_store.set_status(job_id, JobStatus.failed, error=str(e), emit_event=True)
        emit(EventType.error, {"error": str(e)})
        raise
