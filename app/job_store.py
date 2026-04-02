# app/job_store.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy import func
from sqlmodel import Session, select

from .models import (
    Artifact,
    ArtifactKind,
    EventType,
    Job,
    JobEvent,
    JobStatus,
    Peak,
    Track,
    Wave,
)


@dataclass
class JobStore:
    """
    Thin DB access layer. All durable job state lives here:
    - Job status/progress/cancel flag
    - Tracks/Waves/Peaks rows
    - JobEvent append-only log (websocket replay)
    - Artifact rows (pointers to blobs in object storage)

    This store assumes you pass a live SQLModel Session (scoped to request/worker unit of work).
    """
    session: Session

    # -------------------------
    # Jobs
    # -------------------------

    def create_job(
        self,
        *,
        owner_session_id: UUID,
        run_name: str = "untitled",
        config: Optional[Dict[str, Any]] = None,
    ) -> Job:
        now = datetime.utcnow()
        job = Job(
            owner_session_id=owner_session_id,
            run_name=run_name,
            status=JobStatus.queued,
            cancel_requested=False,
            created_at=now,
            updated_at=now,
            config=config or {},
            progress={},
        )
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)

        # Optional: emit an initial status event
        self.append_event(job.id, EventType.status, {"status": job.status})
        return job

    def get_job(self, job_id: UUID) -> Job:
        job = self.session.get(Job, job_id)
        if not job:
            raise NoResultFound(f"Job not found: {job_id}")
        return job

    def list_jobs_for_owner(
        self,
        owner_session_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
        newest_first: bool = True,
    ) -> List[Job]:
        q = select(Job).where(Job.owner_session_id == owner_session_id)
        q = q.order_by(Job.created_at.desc() if newest_first else Job.created_at.asc())
        q = q.offset(offset).limit(limit)
        return list(self.session.exec(q).all())

    def set_status(
        self,
        job_id: UUID,
        status: JobStatus,
        *,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        emit_event: bool = True,
    ) -> Job:
        job = self.get_job(job_id)
        now = datetime.utcnow()

        # Lifecycle timestamps
        if status == JobStatus.in_progress and job.started_at is None:
            job.started_at = now
        if status in (JobStatus.completed, JobStatus.cancelled, JobStatus.failed):
            job.finished_at = now

        job.status = status
        job.updated_at = now
        if error is not None:
            job.error = error
        if error_code is not None:
            job.error_code = error_code

        self.session.add(job)
        self.session.commit()

        if emit_event:
            payload: Dict[str, Any] = {"status": status}
            if error:
                payload["error"] = error
            if error_code:
                payload["error_code"] = error_code
            self.append_event(job_id, EventType.status, payload)

        return job

    def request_cancel(self, job_id: UUID, *, emit_event: bool = True) -> Job:
        job = self.get_job(job_id)
        if job.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
            return job

        job.cancel_requested = True
        job.updated_at = datetime.utcnow()

        # Optionally move status -> cancel_requested (your pipeline can treat either flag as canonical)
        if job.status not in (JobStatus.cancel_requested,):
            job.status = JobStatus.cancel_requested

        self.session.add(job)
        self.session.commit()

        if emit_event:
            self.append_event(job_id, EventType.status, {"status": JobStatus.cancel_requested})

        return job

    def is_cancel_requested(self, job_id: UUID) -> bool:
        job = self.get_job(job_id)
        return bool(job.cancel_requested) or job.status == JobStatus.cancel_requested

    def clear_cancel(self, job_id: UUID, *, emit_event: bool = True) -> Job:
        job = self.get_job(job_id)
        job.cancel_requested = False
        if job.status in (JobStatus.cancel_requested, JobStatus.cancelled):
            job.status = JobStatus.queued
        job.updated_at = datetime.utcnow()
        self.session.add(job)
        self.session.commit()
        if emit_event:
            self.append_event(job_id, EventType.status, {"status": job.status})
        return job

    def get_processed_track_indices(self, job_id: UUID) -> List[int]:
        q = select(Track.track_index).where(Track.job_id == job_id, Track.processed_at.is_not(None))
        return [int(r) for r in self.session.exec(q).all()]

    def recompute_counts(self, job_id: UUID) -> Job:
        job = self.get_job(job_id)
        tracks_done = self.session.exec(
            select(func.count()).select_from(Track).where(Track.job_id == job_id, Track.processed_at.is_not(None))
        ).one()
        waves_done = self.session.exec(
            select(func.count()).select_from(Wave).where(Wave.job_id == job_id)
        ).one()
        peaks_done = self.session.exec(
            select(func.count()).select_from(Peak).where(Peak.job_id == job_id)
        ).one()

        if isinstance(tracks_done, (tuple, list)):
            tracks_done = tracks_done[0]
        if isinstance(waves_done, (tuple, list)):
            waves_done = waves_done[0]
        if isinstance(peaks_done, (tuple, list)):
            peaks_done = peaks_done[0]

        job.tracks_done = int(tracks_done or 0)
        job.waves_done = int(waves_done or 0)
        job.peaks_done = int(peaks_done or 0)
        job.updated_at = datetime.utcnow()
        self.session.add(job)
        self.session.commit()
        return job

    def update_progress(
        self,
        job_id: UUID,
        progress: Dict[str, Any],
        *,
        replace: bool = False,
        emit_event: bool = False,
    ) -> Job:
        """
        Progress snapshot for quick polling or "download progress json".
        - replace=False merges keys into existing progress
        - replace=True replaces the whole progress dict
        """
        job = self.get_job(job_id)
        now = datetime.utcnow()

        if replace:
            job.progress = dict(progress)
        else:
            merged = dict(job.progress or {})
            merged.update(progress)
            job.progress = merged

        job.updated_at = now
        self.session.add(job)
        self.session.commit()

        if emit_event:
            self.append_event(job_id, EventType.progress, job.progress)

        return job

    # -------------------------
    # Events (websocket)
    # -------------------------

    def append_event(self, job_id: UUID, event_type: EventType, payload: Dict[str, Any]) -> JobEvent:
        """
        Append a JobEvent with a per-job monotonic seq.
        We assign seq by "max(seq)+1" with retry on unique constraint collision.
        This is safe enough for MVP; if you later run multiple writers per job concurrently,
        you can replace with a 'next_seq' counter on Job row + SELECT FOR UPDATE.
        """
        max_retries = 5
        for attempt in range(max_retries):
            seq = self._compute_next_seq(job_id)
            ev = JobEvent(
                job_id=job_id,
                seq=seq,
                type=event_type,
                payload=payload,
                created_at=datetime.utcnow(),
            )
            self.session.add(ev)
            try:
                self.session.commit()
                self.session.refresh(ev)
                return ev
            except IntegrityError:
                self.session.rollback()
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError("Failed to append event after retries")

    def _compute_next_seq(self, job_id: UUID) -> int:
        q = select(JobEvent.seq).where(JobEvent.job_id == job_id).order_by(JobEvent.seq.desc()).limit(1)
        row = self.session.exec(q).first()
        return (row or 0) + 1

    def get_events_after(
        self,
        job_id: UUID,
        *,
        after_seq: int = 0,
        limit: int = 500,
    ) -> List[JobEvent]:
        q = (
            select(JobEvent)
            .where(JobEvent.job_id == job_id, JobEvent.seq > after_seq)
            .order_by(JobEvent.seq.asc())
            .limit(limit)
        )
        return list(self.session.exec(q).all())

    # -------------------------
    # Tracks / Waves / Peaks
    # -------------------------

    def upsert_track_by_index(
        self,
        job_id: UUID,
        track_index: int,
        *,
        processed_at: Optional[datetime] = None,
        amplitude: Optional[float] = None,
        frequency: Optional[float] = None,
        error: Optional[float] = None,
        x0: Optional[int] = None,
        y0: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        overlay: Optional[Dict[str, Any]] = None,
    ) -> Track:
        """
        Ensures (job_id, track_index) exists. Useful for resume/cancellation.
        Uses a read-then-write pattern that works across SQLite/Postgres.
        """
        q = select(Track).where(Track.job_id == job_id, Track.track_index == track_index)
        track = self.session.exec(q).first()

        if not track:
            track = Track(
                job_id=job_id,
                track_index=track_index,
                processed_at=processed_at,
                amplitude=amplitude,
                frequency=frequency,
                error=error,
                x0=x0,
                y0=y0,
                metrics=metrics or {},
                overlay=overlay or {},
            )
            self.session.add(track)
        else:
            if processed_at is not None:
                track.processed_at = processed_at
            if amplitude is not None:
                track.amplitude = amplitude
            if frequency is not None:
                track.frequency = frequency
            if error is not None:
                track.error = error
            if x0 is not None:
                track.x0 = x0
            if y0 is not None:
                track.y0 = y0
            if metrics is not None:
                merged = dict(track.metrics or {})
                merged.update(metrics)
                track.metrics = merged
            if overlay is not None:
                track.overlay = overlay
            self.session.add(track)

        self.session.commit()
        self.session.refresh(track)
        return track

    def insert_tracks_batch(self, job_id: UUID, rows: Sequence[Dict[str, Any]]) -> int:
        objs = [Track(job_id=job_id, **r) for r in rows]
        self.session.add_all(objs)
        self.session.commit()
        return len(objs)

    def insert_waves_batch(self, job_id: UUID, rows: Sequence[Dict[str, Any]]) -> int:
        objs = [Wave(job_id=job_id, **r) for r in rows]
        self.session.add_all(objs)
        self.session.commit()
        return len(objs)

    def insert_peaks_batch(self, job_id: UUID, rows: Sequence[Dict[str, Any]]) -> int:
        objs = [Peak(job_id=job_id, **r) for r in rows]
        self.session.add_all(objs)
        self.session.commit()
        return len(objs)

    def bump_counts(
        self,
        job_id: UUID,
        *,
        tracks_done_delta: int = 0,
        waves_done_delta: int = 0,
        peaks_done_delta: int = 0,
        tracks_total: Optional[int] = None,
    ) -> Job:
        job = self.get_job(job_id)
        job.tracks_done = int(job.tracks_done or 0) + int(tracks_done_delta)
        job.waves_done = int(job.waves_done or 0) + int(waves_done_delta)
        job.peaks_done = int(job.peaks_done or 0) + int(peaks_done_delta)
        if tracks_total is not None:
            job.tracks_total = tracks_total
        job.updated_at = datetime.utcnow()
        self.session.add(job)
        self.session.commit()
        return job

    def update_run_name(self, job_id: UUID, run_name: str) -> Job:
        job = self.get_job(job_id)
        job.run_name = run_name
        job.updated_at = datetime.utcnow()
        self.session.add(job)
        self.session.commit()
        return job

    # -------------------------
    # Artifacts (DB index only)
    # -------------------------

    def create_artifact(
        self,
        *,
        job_id: UUID,
        kind: ArtifactKind,
        blob_path: str,
        label: Optional[str] = None,
        track_id: Optional[UUID] = None,
        wave_id: Optional[UUID] = None,
        content_type: Optional[str] = None,
        byte_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """
        Back-compat:
          - callers can pass `metadata=...` (old name)
          - callers can pass `meta=...` (new name)
        Model attribute is `Artifact.meta` to avoid SQLAlchemy reserved name conflict.
        """
        meta_in = meta if meta is not None else metadata

        art = Artifact(
            job_id=job_id,
            kind=kind,
            label=label,
            track_id=track_id,
            wave_id=wave_id,
            blob_path=blob_path,
            content_type=content_type,
            byte_size=byte_size,
            meta=meta_in or {},  # IMPORTANT: use `meta`, not `metadata`
            created_at=datetime.utcnow(),
        )
        self.session.add(art)
        self.session.commit()
        self.session.refresh(art)
        return art

    def list_artifacts(
        self,
        job_id: UUID,
        *,
        kind: Optional[ArtifactKind] = None,
        label: Optional[str] = None,
        limit: int = 200,
    ) -> List[Artifact]:
        q = select(Artifact).where(Artifact.job_id == job_id)
        if kind is not None:
            q = q.where(Artifact.kind == kind)
        if label is not None:
            q = q.where(Artifact.label == label)
        q = q.order_by(Artifact.created_at.asc()).limit(limit)
        return list(self.session.exec(q).all())
