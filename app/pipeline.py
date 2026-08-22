# app/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from .artifact_store import ArtifactStore
from .cancel import CancellationRequested
from .job_store import JobStore
from .models import ArtifactKind, EventType, JobStatus

from .io.image_to_heatmap import image_to_heatmap_bytes
from .io.table_to_heatmap import table_to_heatmap_payload
from .extract_core import select_kymo_runner, process_track
from .analysis_mode import LARGE_WAVE_ANALYSIS_MODE, RIPPLE_ANALYSIS_MODE, resolve_analysis_mode
from .large_wave_analysis import analyze_large_wave_events, build_large_wave_track_config
from .large_wave_extraction import run_large_wave_extraction
from .ripple_analysis import analyze_ripple_tracks
from .ripple_extraction import run_ripple_extraction
from .time_utils import utc_now, utc_now_iso


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
    started_at = utc_now()

    scratch_dir = settings.scratch_root / str(job_id)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    def cancelled() -> bool:
        return job_store.is_cancel_requested(job_id)

    def check_cancel(reason: str = "cancel_requested") -> None:
        if cancelled():
            raise CancellationRequested(reason)

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
            "updated_at": utc_now_iso(),
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
        check_cancel("cancel_requested_before_debug_overlays")
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
            check_cancel("cancel_requested_during_debug_overlays")
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

        endpoint_links = dbg / "endpoint_links.json"
        if endpoint_links.exists():
            check_cancel("cancel_requested_during_debug_overlays")
            publish_file(
                kind=ArtifactKind.debug_text,
                filename=f"{image_id}_endpoint_links.json",
                local_path=endpoint_links,
                content_type="application/json",
                label=f"{image_id}:endpoint_links",
                meta={"image_id": image_id, "debug": "endpoint_links"},
            )

        large_wave_ridges = dbg / "large_wave_ridges.json"
        if large_wave_ridges.exists():
            check_cancel("cancel_requested_during_debug_overlays")
            publish_file(
                kind=ArtifactKind.debug_text,
                filename=f"{image_id}_large_wave_ridges.json",
                local_path=large_wave_ridges,
                content_type="application/json",
                label=f"{image_id}:large_wave_ridges",
                meta={"image_id": image_id, "debug": "large_wave_ridges"},
            )

        ot = base_dir / "overlay_tracks.png"
        if ot.exists():
            check_cancel("cancel_requested_during_debug_overlays")
            label = f"{image_id}:overlay_tracks"
            publish_file(
                kind=ArtifactKind.overlay,
                filename=f"{image_id}_overlay_tracks.png",
                local_path=ot,
                content_type="image/png",
                label=label,
                meta={"image_id": image_id, "overlay": "overlay_tracks"},
            )

    def publish_ripple_manifest(image_id: str, base_dir: Path) -> None:
        manifest_path = base_dir / "ripple_track_manifest.json"
        if not manifest_path.exists():
            return
        check_cancel("cancel_requested_before_ripple_manifest_publish")
        publish_file(
            kind=ArtifactKind.debug_text,
            filename=f"{image_id}_ripple_track_manifest.json",
            local_path=manifest_path,
            content_type="application/json",
            label=f"{image_id}:ripple_track_manifest",
            meta={
                "image_id": image_id,
                "analysis_mode": RIPPLE_ANALYSIS_MODE,
                "extractor": "ripple_multiscale_hough",
            },
        )

    def load_heatmap_values_artifact() -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        check_cancel("cancel_requested_before_heatmap_values_resume")
        arts = job_store.list_artifacts(job_id, kind=ArtifactKind.other, label="base_heatmap_values", limit=1)
        if not arts:
            return None, None
        data = artifact_store.get_bytes(arts[0].blob_path)
        check_cancel("cancel_requested_after_heatmap_values_resume")
        return data, dict(getattr(arts[0], "meta", {}) or {})

    resume_cfg = (config.get("service") or {}).get("resume") or {}
    resume_enabled = bool(resume_cfg.get("enabled", False)) or bool(resume)
    analysis_mode = resolve_analysis_mode(config)

    try:
        # -----------------------------
        # Job init
        # -----------------------------
        job_store.set_status(job_id, JobStatus.in_progress, emit_event=True)
        set_progress("init")
        user_log("Starting analysis", stage="init")
        check_cancel("cancel_requested_before_start")

        if resume_enabled:
            check_cancel("cancel_requested_before_resume_counts")
            job_store.recompute_counts(job_id)

        check_cancel("cancel_requested_before_start")

        # -----------------------------
        # Base heatmap (resume-aware)
        # -----------------------------
        heatmap_png: Optional[bytes] = None
        heatmap_meta: Optional[Dict[str, Any]] = None
        heatmap_value_bytes: Optional[bytes] = None
        heatmap_value_meta: Optional[Dict[str, Any]] = None
        if resume_enabled:
            check_cancel("cancel_requested_before_heatmap_resume")
            existing = job_store.list_artifacts(
                job_id, kind=ArtifactKind.base_heatmap, label="base_heatmap", limit=1
            )
            if existing:
                check_cancel("cancel_requested_before_heatmap_resume")
                heatmap_png = artifact_store.get_bytes(existing[0].blob_path)
                heatmap_meta = dict(getattr(existing[0], "meta", {}) or {})
                check_cancel("cancel_requested_after_heatmap_resume")
                set_progress("heatmap_ready", extra={"resume": True})
                user_log("Using existing heatmap", stage="heatmap_ready")

        if heatmap_png is None:
            # -----------------------------
            # Load uploaded input from artifact store
            # -----------------------------
            user_log("Loading input", stage="load_input")
            check_cancel("cancel_requested_before_input_load")
            uploads = [
                *job_store.list_artifacts(job_id, kind=ArtifactKind.upload_image, limit=10),
                *job_store.list_artifacts(job_id, kind=ArtifactKind.upload_csv, limit=10),
            ]
            if not uploads:
                raise PipelineError("No upload artifact found (expected table or image upload)")
            uploads.sort(key=lambda art: art.created_at)

            upload = uploads[0]
            check_cancel("cancel_requested_before_input_download")
            input_bytes = artifact_store.get_bytes(upload.blob_path)
            input_filename = (upload.meta or {}).get("filename")
            check_cancel("cancel_requested_after_input_download")

            loaded_stage = "image_loaded" if upload.kind == ArtifactKind.upload_image else "table_loaded"
            set_progress(loaded_stage, extra={"upload_blob": upload.blob_path, "input_kind": upload.kind.value})
            emit(EventType.progress, {"stage": loaded_stage, "bytes": len(input_bytes), "input_kind": upload.kind.value})

            check_cancel("cancel_requested_after_input_loaded")

            # -----------------------------
            # Input -> base heatmap
            # -----------------------------
            user_log("Generating heatmap", stage="heatmap")
            check_cancel("cancel_requested_before_heatmap")
            if upload.kind == ArtifactKind.upload_image:
                heatmap_png, heatmap_meta = image_to_heatmap_bytes(
                    input_bytes,
                    config=config,
                    filename_hint=str(input_filename) if input_filename else None,
                    cancel_cb=cancelled,
                )
            else:
                heatmap_png, heatmap_meta, heatmap_value_bytes, heatmap_value_meta = table_to_heatmap_payload(
                    input_bytes,
                    config=config,
                    filename_hint=str(input_filename) if input_filename else None,
                    cancel_cb=cancelled,
                )
            check_cancel("cancel_requested_after_heatmap")
            heatmap_meta = {
                **(heatmap_meta or {}),
                "source_artifact_id": str(upload.id),
                "source_artifact_kind": upload.kind.value,
            }
            check_cancel("cancel_requested_before_heatmap_publish")
            publish_bytes(
                kind=ArtifactKind.base_heatmap,
                filename="base_heatmap.png",
                data=heatmap_png,
                content_type="image/png",
                label="base_heatmap",
                meta=heatmap_meta,
            )
            if heatmap_value_bytes is not None:
                check_cancel("cancel_requested_before_heatmap_values_publish")
                publish_bytes(
                    kind=ArtifactKind.other,
                    filename="base_heatmap_values.f32",
                    data=heatmap_value_bytes,
                    content_type="application/octet-stream",
                    label="base_heatmap_values",
                    meta={
                        **(heatmap_value_meta or {}),
                        "source_artifact_id": str(upload.id),
                        "source_artifact_kind": upload.kind.value,
                    },
                )
            check_cancel("cancel_requested_after_heatmap_publish")
            set_progress("heatmap_ready")
            user_log("Heatmap ready", stage="heatmap_ready")

        if heatmap_png is None:
            raise PipelineError("Heatmap bytes missing (resume or generation failed)")
        heatmap_path = scratch_dir / "base_heatmap.png"
        check_cancel("cancel_requested_before_heatmap_write")
        heatmap_path.write_bytes(heatmap_png)
        check_cancel("cancel_requested_after_heatmap_write")

        check_cancel("cancel_requested_after_heatmap")

        if analysis_mode in {RIPPLE_ANALYSIS_MODE, LARGE_WAVE_ANALYSIS_MODE} and heatmap_value_bytes is None:
            cached_values, cached_meta = load_heatmap_values_artifact()
            if cached_values is not None:
                heatmap_value_bytes = cached_values
                heatmap_value_meta = cached_meta or {}

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
            check_cancel("cancel_requested_before_track_resume")
            arts = job_store.list_artifacts(
                job_id, kind=ArtifactKind.track_npy, limit=max(2000, total_tracks + 10)
            )
            mapping: Dict[int, Any] = {}
            for art in arts:
                check_cancel("cancel_requested_during_track_resume")
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
                check_cancel("cancel_requested_during_track_resume")
                art = mapping[i]
                data = artifact_store.get_bytes(art.blob_path)
                check_cancel("cancel_requested_during_track_resume")
                p = dest_dir / f"track_{i}.npy"
                p.write_bytes(data)
                paths.append(p)
            return paths

        def _load_track_manifest() -> Optional[Dict[str, Any]]:
            check_cancel("cancel_requested_before_track_manifest")
            arts = job_store.list_artifacts(job_id, kind=ArtifactKind.track_manifest, label="tracks_manifest", limit=1)
            if not arts:
                return None
            try:
                check_cancel("cancel_requested_before_track_manifest")
                raw = artifact_store.get_bytes(arts[0].blob_path)
                check_cancel("cancel_requested_after_track_manifest")
                return json.loads(raw.decode("utf-8"))
            except CancellationRequested:
                raise
            except Exception:
                return None

        if resume_enabled:
            check_cancel("cancel_requested_before_track_resume")
            manifest = _load_track_manifest()
            if manifest and isinstance(manifest.get("total_tracks"), int):
                maybe_paths = _load_tracks_from_artifacts(int(manifest["total_tracks"]))
                if maybe_paths:
                    track_paths = maybe_paths
                    cached_extractor = str(manifest.get("extractor") or "")
                    cached_stage = "ripple_extract_done" if cached_extractor == "ripple_multiscale_hough" else "kymo_done"
                    emit(
                        EventType.progress,
                        {
                            "stage": cached_stage,
                            "tracks_found": len(track_paths),
                            "resume": True,
                            "extractor": cached_extractor or None,
                        },
                    )
                    user_log(f"Using cached tracks ({len(track_paths)})", stage=cached_stage)

        if not track_paths:
            check_cancel("cancel_requested_before_track_extraction")
            image_id: str
            base_dir: Path
            ripple_extraction_cfg = (((config.get("analysis") or {}).get("ripple") or {}).get("extraction") or {})
            large_wave_extraction_cfg = (
                (((config.get("analysis") or {}).get("large_wave") or {}).get("extraction") or {})
            )
            use_ripple_extractor = (
                analysis_mode == RIPPLE_ANALYSIS_MODE
                and bool(ripple_extraction_cfg.get("enabled", True))
            )
            use_large_wave_extractor = (
                analysis_mode == LARGE_WAVE_ANALYSIS_MODE
                and bool(large_wave_extraction_cfg.get("enabled", True))
            )
            if use_ripple_extractor:
                extractor_name = "ripple_multiscale_hough"
            elif use_large_wave_extractor:
                extractor_name = "large_wave_multiscale_ensemble"
            else:
                extractor_name = "kymobutler"
            if use_ripple_extractor:
                user_log("Extracting tracks", stage="ripple_extract_start")
                set_progress("ripple_extract_start", extra={"detail": "Extracting tracks"})
                check_cancel("cancel_requested_before_ripple_extraction")
                ripple_stage_labels = {
                    "load_values": "Loading heatmap",
                    "smoothing": "Segmenting heatmap",
                    "seeding": "Tracing tracks",
                    "tracking": "Tracing tracks",
                    "deduping": "Removing duplicates",
                    "saving": "Saving tracks",
                }
                last_ripple_extract_stage: Optional[str] = None

                def ripple_extract_progress(stage: str, data: Dict[str, Any]) -> None:
                    nonlocal last_ripple_extract_stage
                    label = ripple_stage_labels.get(stage, stage)
                    extra = dict(data) if data else {}
                    if stage == "load_values" and heatmap_value_bytes is not None:
                        extra["value_source"] = "base_heatmap_values"
                    set_progress(f"ripple_extract_{stage}", extra=extra or None)
                    if stage != last_ripple_extract_stage and stage in ripple_stage_labels:
                        user_log(label, stage=f"ripple_extract_{stage}")
                        last_ripple_extract_stage = stage

                ripple_out = run_ripple_extraction(
                    heatmap_path=heatmap_path,
                    scratch_dir=scratch_dir,
                    config=config,
                    heatmap_value_bytes=heatmap_value_bytes,
                    heatmap_value_meta=heatmap_value_meta,
                    progress_cb=ripple_extract_progress,
                    cancel_cb=cancelled,
                )
                check_cancel("cancel_requested_after_ripple_extraction")
                image_id = ripple_out.image_id
                base_dir = ripple_out.base_dir
                track_paths = list(ripple_out.track_paths)
                publish_ripple_manifest(image_id, base_dir)
            elif use_large_wave_extractor:
                user_log("Extracting tracks", stage="kymo_start")
                set_progress("kymo_start", extra={"detail": "Extracting tracks"})
                check_cancel("cancel_requested_before_large_wave_extraction")
                large_wave_stage_map = {
                    "load_values": ("load_image", "Loading heatmap"),
                    "smoothing": ("segmenting", "Segmenting heatmap"),
                    "tracing": ("tracking", "Tracing tracks"),
                    "tracing_cusps": ("tracking", "Tracing tracks"),
                    "bridging": ("endpoint_linking", "Bridging track gaps"),
                    "deduping": ("deduping", "Removing duplicates"),
                    "saving": ("saving", "Saving tracks"),
                }
                last_large_wave_extract_stage: Optional[str] = None

                def large_wave_extract_progress(stage: str, data: Dict[str, Any]) -> None:
                    nonlocal last_large_wave_extract_stage
                    mapped_stage, label = large_wave_stage_map.get(stage, (stage, stage))
                    extra = dict(data) if data else {}
                    if stage == "load_values" and heatmap_value_bytes is not None:
                        extra["value_source"] = "base_heatmap_values"
                    set_progress(f"kymo_{mapped_stage}", extra=extra or None)
                    if mapped_stage != last_large_wave_extract_stage:
                        user_log(label, stage=f"kymo_{mapped_stage}")
                        last_large_wave_extract_stage = mapped_stage

                large_wave_out = run_large_wave_extraction(
                    heatmap_path=heatmap_path,
                    scratch_dir=scratch_dir,
                    config=config,
                    heatmap_value_bytes=heatmap_value_bytes,
                    heatmap_value_meta=heatmap_value_meta,
                    progress_cb=large_wave_extract_progress,
                    cancel_cb=cancelled,
                )
                check_cancel("cancel_requested_after_large_wave_extraction")
                image_id = large_wave_out.image_id
                base_dir = large_wave_out.base_dir
                track_paths = list(large_wave_out.track_paths)
            else:
                user_log("Extracting tracks (KymoButler)", stage="kymo_start")
                set_progress("kymo_start", extra={"detail": "Extracting tracks"})
                check_cancel("cancel_requested_before_kymo")
                runner = select_kymo_runner(config=config)

                kymo_stage_labels = {
                    "load_image": "Loading heatmap",
                    "segmenting": "Segmenting heatmap",
                    "masking": "Cleaning mask",
                    "skeletonizing": "Skeletonizing tracks",
                    "tracking": "Tracing tracks",
                    "refining": "Refining tracks",
                    "endpoint_linking": "Bridging track gaps",
                    "endpoint_linking_done": "Track gaps bridged",
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

                kymo_out = runner.run(
                    heatmap_path=heatmap_path,
                    scratch_dir=scratch_dir,
                    progress_cb=kymo_progress,
                    cancel_cb=cancelled,
                )
                check_cancel("cancel_requested_after_kymo")

                image_id = kymo_out.image_id
                base_dir = kymo_out.base_dir
                track_paths = list(kymo_out.track_paths)

            check_cancel("cancel_requested_before_debug_overlays")
            publish_debug_overlays(image_id, base_dir)
            check_cancel("cancel_requested_after_debug_overlays")
            done_stage = "ripple_extract_done" if use_ripple_extractor else "kymo_done"
            emit(
                EventType.progress,
                {
                    "stage": done_stage,
                    "tracks_found": len(track_paths),
                    "image_id": image_id,
                    "extractor": extractor_name,
                },
            )
            user_log(f"Found {len(track_paths)} tracks", stage=done_stage)

            if not track_paths:
                raise PipelineError(f"{extractor_name} produced no tracks")

            if resume_enabled:
                # Persist all tracks for resume
                check_cancel("cancel_requested_before_track_persist")
                existing = job_store.list_artifacts(
                    job_id, kind=ArtifactKind.track_npy, limit=max(2000, len(track_paths) + 10)
                )
                existing_idx = set(
                    idx for idx in (_parse_track_index(getattr(a, "meta", {}) or {}, a.label) for a in existing) if idx is not None
                )
                for track_index, track_path in enumerate(track_paths):
                    check_cancel("cancel_requested_during_track_persist")
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

                manifest_payload = {
                    "total_tracks": len(track_paths),
                    "analysis_mode": analysis_mode,
                    "extractor": extractor_name,
                }
                check_cancel("cancel_requested_before_track_manifest_write")
                publish_bytes(
                    kind=ArtifactKind.track_manifest,
                    filename="tracks_manifest.json",
                    data=json.dumps(manifest_payload).encode("utf-8"),
                    content_type="application/json",
                    label="tracks_manifest",
                    meta=manifest_payload,
                )

        if analysis_mode == RIPPLE_ANALYSIS_MODE:
            # Ripple analysis intentionally separates track geometry, family
            # assignment, and inter-track interval measurement.
            check_cancel("cancel_requested_before_ripple_analysis")
            user_log("Analyzing tracks", stage="ripple_track_geometry")
            job_store.bump_counts(job_id, tracks_total=len(track_paths))
            last_ripple_stage: Optional[str] = None
            last_ripple_progress_at = utc_now()
            ripple_stage_labels = {
                "ripple_track_geometry": "Analyzing tracks",
                "ripple_family_grouping": "Grouping tracks",
                "ripple_interval_analysis": "Measuring waves",
            }

            def ripple_progress(stage: str, processed: int, total: int) -> None:
                nonlocal last_ripple_stage, last_ripple_progress_at
                check_cancel("cancel_requested_during_ripple_analysis")
                now = utc_now()
                stage_changed = stage != last_ripple_stage
                should_publish = (
                    stage_changed
                    or total <= 0
                    or processed >= total
                    or (now - last_ripple_progress_at).total_seconds() >= settings.progress_every_secs
                )
                if should_publish:
                    set_progress(stage, processed=processed, total=total)
                    last_ripple_progress_at = now
                if stage_changed:
                    user_log(ripple_stage_labels.get(stage, stage), stage=stage)
                    last_ripple_stage = stage

            ripple_result = analyze_ripple_tracks(
                job_id=job_id,
                track_paths=track_paths,
                config=config,
                cancel_cb=cancelled,
                progress_cb=ripple_progress,
            )
            check_cancel("cancel_requested_after_ripple_analysis")

            track_detail_cfg = (config.get("track_detail") or {})
            store_track_npy = bool(track_detail_cfg.get("store_npy", True)) or resume_enabled
            for track_path, track_row, overlay_track in zip(
                track_paths,
                ripple_result.track_rows,
                ripple_result.overlay_events,
            ):
                check_cancel("cancel_requested_during_ripple_persistence")
                track_index = int(track_row["track_index"])
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
                            meta={"track_index": track_index},
                        )

                job_store.upsert_track_by_index(
                    job_id,
                    track_index,
                    processed_at=utc_now(),
                    amplitude=track_row.get("amplitude"),
                    frequency=track_row.get("frequency"),
                    error=track_row.get("error"),
                    x0=track_row.get("x0"),
                    y0=track_row.get("y0"),
                    metrics=track_row.get("metrics") or {},
                    overlay=track_row.get("overlay") or {},
                )
                emit(EventType.overlay_track, overlay_track)

            job_store.recompute_counts(job_id)

            ripple_exports = [
                ("ripple_tracks", "tracks.csv", ripple_result.tracks_csv, len(ripple_result.track_rows)),
                ("ripple_intervals", "waves.csv", ripple_result.intervals_csv, len(ripple_result.intervals)),
                ("ripple_families", "families.csv", ripple_result.families_csv, len(ripple_result.families)),
            ]
            for label, filename, data, row_count in ripple_exports:
                check_cancel("cancel_requested_during_ripple_export")
                existing = job_store.list_artifacts(job_id, kind=ArtifactKind.other, label=label, limit=1)
                if existing:
                    continue
                publish_bytes(
                    kind=ArtifactKind.other,
                    filename=filename,
                    data=data,
                    content_type="text/csv",
                    label=label,
                    meta={
                        "analysis_mode": RIPPLE_ANALYSIS_MODE,
                        "row_count": row_count,
                        "filename": filename,
                    },
                )

            check_cancel("cancel_requested_before_completion")
            completion_extra = {
                "analysis_mode": RIPPLE_ANALYSIS_MODE,
                "families_found": len(ripple_result.families),
                "intervals_found": len(ripple_result.intervals),
                "eta_secs": 0.0,
            }
            set_progress("completed", processed=len(track_paths), total=len(track_paths), extra=completion_extra)
            user_log("Completed", stage="completed")
            job_store.set_status(job_id, JobStatus.completed, emit_event=True)
            emit(EventType.done, {
                "ok": True,
                "analysis_mode": RIPPLE_ANALYSIS_MODE,
                "duration_s": (utc_now() - started_at).total_seconds(),
                "families_found": len(ripple_result.families),
                "intervals_found": len(ripple_result.intervals),
            })
            return

        # -----------------------------
        # Standard and large-wave tracks -> DB rows + overlay events
        # -----------------------------
        check_cancel("cancel_requested_before_processing_tracks")
        user_log("Analyzing tracks", stage="processing_tracks")
        job_store.bump_counts(job_id, tracks_total=len(track_paths))

        processed_set = set(job_store.get_processed_track_indices(job_id)) if resume_enabled else set()
        check_cancel("cancel_requested_before_processing_tracks")
        processed = len(processed_set)
        set_progress("processing_tracks", processed=processed, total=len(track_paths))

        waves_buf: List[Dict[str, Any]] = []
        peaks_buf: List[Dict[str, Any]] = []
        new_processed = 0
        batch_new_processed = 0
        last_progress_ts = utc_now()
        processing_started_at = utc_now()
        last_rate_ts = processing_started_at
        last_processed_for_rate = processed
        ema_rate_tps: Optional[float] = None
        ema_alpha = 0.2
        large_wave_mode = analysis_mode == LARGE_WAVE_ANALYSIS_MODE
        track_analysis_config = build_large_wave_track_config(config) if large_wave_mode else config
        large_wave_track_rows: List[Dict[str, Any]] = []
        large_wave_overlay_events: List[Dict[str, Any]] = []

        for track_index, track_path in enumerate(track_paths):
            check_cancel("cancel_requested_during_processing_tracks")

            if resume_enabled and track_index in processed_set:
                processed += 1
                continue

            # Optional: persist raw track for detail view
            track_detail_cfg = (config.get("track_detail") or {})
            store_track_npy = bool(track_detail_cfg.get("store_npy", True)) or resume_enabled
            if store_track_npy:
                check_cancel("cancel_requested_before_track_artifact")
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

            check_cancel("cancel_requested_before_track_analysis")
            track_row, wave_rows, peak_rows, overlay_track = process_track(
                job_id=job_id,
                track_index=track_index,
                track_path=track_path,
                config=track_analysis_config,
                heatmap_meta=heatmap_meta,
            )
            check_cancel("cancel_requested_after_track_analysis")

            if large_wave_mode:
                track_metrics = track_row.get("metrics") if isinstance(track_row.get("metrics"), dict) else {}
                track_metrics["analysis_mode"] = LARGE_WAVE_ANALYSIS_MODE
                track_row["metrics"] = track_metrics
                overlay_track["metrics"] = {
                    "analysis_mode": LARGE_WAVE_ANALYSIS_MODE,
                    "mean_amplitude": track_row.get("amplitude"),
                    "dominant_frequency": track_row.get("frequency"),
                    "period": track_metrics.get("period"),
                    "num_peaks": track_metrics.get("num_peaks"),
                }
                for row in wave_rows or []:
                    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                    metrics["track_index"] = int(track_index)
                    row["metrics"] = metrics
                large_wave_track_rows.append(track_row)
                large_wave_overlay_events.append(overlay_track)

            # print("Processed track #", track_row, " Wave", wave_rows)

            track = job_store.upsert_track_by_index(
                job_id,
                track_index,
                processed_at=utc_now(),
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

            if (
                not large_wave_mode
                and settings.emit_overlay_every_tracks > 0
                and new_processed % settings.emit_overlay_every_tracks == 0
            ):
                check_cancel("cancel_requested_before_overlay_event")
                emit(EventType.overlay_track, overlay_track)

            if settings.db_batch_size > 0 and (new_processed % settings.db_batch_size == 0):
                check_cancel("cancel_requested_before_batch_write")
                if waves_buf and not large_wave_mode:
                    job_store.insert_waves_batch(job_id, waves_buf)
                    job_store.bump_counts(job_id, waves_done_delta=len(waves_buf))
                    waves_buf.clear()
                if peaks_buf:
                    job_store.insert_peaks_batch(job_id, peaks_buf)
                    job_store.bump_counts(job_id, peaks_done_delta=len(peaks_buf))
                    peaks_buf.clear()

                job_store.bump_counts(job_id, tracks_done_delta=batch_new_processed)
                batch_new_processed = 0

            now = utc_now()
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

        large_wave_result = None
        if large_wave_mode:
            last_large_wave_stage: Optional[str] = None
            large_wave_stage_labels = {
                "large_wave_peak_measurement": "Measuring broad peaks",
                "large_wave_event_grouping": "Grouping large waves",
            }

            def large_wave_progress(stage: str, processed_count: int, total_count: int) -> None:
                nonlocal last_large_wave_stage
                check_cancel("cancel_requested_during_large_wave_analysis")
                set_progress(stage, processed=processed_count, total=total_count)
                if stage != last_large_wave_stage:
                    user_log(large_wave_stage_labels.get(stage, stage), stage=stage)
                    last_large_wave_stage = stage

            large_wave_result = analyze_large_wave_events(
                track_paths=track_paths,
                track_rows=large_wave_track_rows,
                wave_rows=waves_buf,
                config=config,
                cancel_cb=cancelled,
                progress_cb=large_wave_progress,
            )
            waves_buf = list(large_wave_result.wave_rows)

            for track_row, overlay_track in zip(large_wave_track_rows, large_wave_overlay_events):
                track_index = int(track_row["track_index"])
                summary = large_wave_result.track_summaries.get(track_index, {})
                event_frequency = summary.get("large_wave_frequency_hz")
                overlay_track["metrics"] = {
                    "analysis_mode": LARGE_WAVE_ANALYSIS_MODE,
                    "mean_amplitude": summary.get("mean_large_wave_amplitude_px"),
                    "dominant_frequency": event_frequency,
                    "period": (1.0 / float(event_frequency)) if event_frequency else None,
                    "recurrence_frequency": summary.get(
                        "large_wave_recurrence_frequency_hz"
                    ),
                    "num_peaks": summary.get("large_wave_measurement_count", 0),
                    **summary,
                }
                overlay_track["peaks"] = [
                    {
                        "x": measurement["peak_position_px"],
                        "y": measurement["peak_frame"],
                        "amp": measurement["signed_amplitude_px"],
                    }
                    for measurement in large_wave_result.measurements
                    if int(measurement["track_index"]) == track_index
                ]
                job_store.upsert_track_by_index(
                    job_id,
                    track_index,
                    processed_at=utc_now(),
                    amplitude=track_row.get("amplitude"),
                    frequency=track_row.get("frequency"),
                    error=track_row.get("error"),
                    x0=track_row.get("x0"),
                    y0=track_row.get("y0"),
                    metrics=track_row.get("metrics") or {},
                    overlay=track_row.get("overlay") or {},
                )
                emit(EventType.overlay_track, overlay_track)

            large_wave_exports = [
                ("large_wave_tracks", "tracks.csv", large_wave_result.tracks_csv, len(large_wave_track_rows)),
                (
                    "large_wave_measurements",
                    "waves.csv",
                    large_wave_result.measurements_csv,
                    len(large_wave_result.measurements),
                ),
                ("large_wave_events", "events.csv", large_wave_result.events_csv, len(large_wave_result.events)),
            ]
            for label, filename, data, row_count in large_wave_exports:
                check_cancel("cancel_requested_during_large_wave_export")
                existing = job_store.list_artifacts(job_id, kind=ArtifactKind.other, label=label, limit=1)
                if existing:
                    continue
                publish_bytes(
                    kind=ArtifactKind.other,
                    filename=filename,
                    data=data,
                    content_type="text/csv",
                    label=label,
                    meta={
                        "analysis_mode": LARGE_WAVE_ANALYSIS_MODE,
                        "row_count": row_count,
                        "filename": filename,
                    },
                )

        if waves_buf:
            check_cancel("cancel_requested_before_final_batch_write")
            job_store.insert_waves_batch(job_id, waves_buf)
            job_store.bump_counts(job_id, waves_done_delta=len(waves_buf))
            waves_buf.clear()

        if peaks_buf:
            check_cancel("cancel_requested_before_final_batch_write")
            job_store.insert_peaks_batch(job_id, peaks_buf)
            job_store.bump_counts(job_id, peaks_done_delta=len(peaks_buf))
            peaks_buf.clear()

        # Final counts
        if batch_new_processed:
            check_cancel("cancel_requested_before_final_counts")
            job_store.bump_counts(job_id, tracks_done_delta=batch_new_processed)

        check_cancel("cancel_requested_before_completion")
        completion_extra: Dict[str, Any] = {"eta_secs": 0.0, "analysis_mode": analysis_mode}
        if large_wave_result is not None:
            completion_extra.update({
                "large_wave_events_found": len(large_wave_result.events),
                "large_wave_measurements_found": len(large_wave_result.measurements),
            })
        set_progress("completed", processed=len(track_paths), total=len(track_paths), extra=completion_extra)

        user_log("Completed", stage="completed")
        job_store.set_status(job_id, JobStatus.completed, emit_event=True)
        done_payload: Dict[str, Any] = {
            "ok": True,
            "analysis_mode": analysis_mode,
            "duration_s": (utc_now() - started_at).total_seconds(),
        }
        if large_wave_result is not None:
            done_payload.update({
                "large_wave_events_found": len(large_wave_result.events),
                "large_wave_measurements_found": len(large_wave_result.measurements),
            })
        emit(EventType.done, done_payload)

    except CancellationRequested as e:
        try:
            job_store.session.rollback()
        except Exception:
            pass

        reason = str(e) or "cancel_requested"
        try:
            job_store.set_status(job_id, JobStatus.cancelled, emit_event=True)
            set_progress("cancelled")
            emit(EventType.cancelled, {"reason": reason})
            user_log("Run cancelled", stage="cancelled", level="warn")
        except Exception:
            try:
                job_store.session.rollback()
            except Exception:
                pass
        return

    except Exception as e:
        error_text = str(e)
        try:
            job_store.session.rollback()
        except Exception:
            pass

        try:
            job_store.set_status(job_id, JobStatus.failed, error=error_text, emit_event=True)
            user_log("Run failed", stage="failed", level="error")
            emit(EventType.error, {"error": error_text})
        except Exception:
            try:
                job_store.session.rollback()
            except Exception:
                pass
        raise
