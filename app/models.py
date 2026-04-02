# app/models.py

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, UniqueConstraint, Index, JSON


# -----------------------------
# Enums
# -----------------------------

class JobStatus(str, Enum):
    queued = "queued"
    in_progress = "in_progress"
    cancel_requested = "cancel_requested"
    cancelled = "cancelled"
    completed = "completed"
    failed = "failed"


class ArtifactKind(str, Enum):
    # uploads / primary outputs
    upload_csv = "upload_csv"
    base_heatmap = "base_heatmap"

    # overlays (use label for variants like mask_clean, skeleton, etc.)
    overlay = "overlay"

    # exports
    export_waves_csv = "export_waves_csv"
    export_tracks_csv = "export_tracks_csv"
    export_peaks_csv = "export_peaks_csv"
    export_progress_json = "export_progress_json"

    # debug / misc
    debug_text = "debug_text"
    # track resume helpers
    track_npy = "track_npy"
    track_manifest = "track_manifest"
    other = "other"


class EventType(str, Enum):
    # job lifecycle
    status = "status"
    progress = "progress"
    error = "error"
    done = "done"
    cancelled = "cancelled"
    user_log = "user_log"

    # streaming overlay updates / incremental computation
    overlay_track = "overlay_track"
    overlay_ready = "overlay_ready"
    waves_batch = "waves_batch"


# -----------------------------
# DB Tables
# -----------------------------

class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)

    owner_session_id: UUID = Field(index=True)
    run_name: str = Field(default="untitled", index=True)

    status: JobStatus = Field(default=JobStatus.queued, index=True)
    cancel_requested: bool = Field(default=False, index=True)

    error: Optional[str] = Field(default=None)
    error_code: Optional[str] = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    started_at: Optional[datetime] = Field(default=None, index=True)
    finished_at: Optional[datetime] = Field(default=None, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    progress: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))

    tracks_total: Optional[int] = Field(default=None)
    tracks_done: int = Field(default=0, index=True)
    waves_done: int = Field(default=0, index=True)
    peaks_done: int = Field(default=0, index=True)

    artifacts: List["Artifact"] = Relationship(back_populates="job")
    tracks: List["Track"] = Relationship(back_populates="job")
    waves: List["Wave"] = Relationship(back_populates="job")
    peaks: List["Peak"] = Relationship(back_populates="job")
    events: List["JobEvent"] = Relationship(back_populates="job")


class Artifact(SQLModel, table=True):
    """
    Any durable blob stored in object storage (GCS/local).
    NOTE: 'metadata' is reserved in SQLAlchemy; use `meta` instead.
    """
    __tablename__ = "artifacts"
    __table_args__ = (
        Index("ix_artifacts_job_kind", "job_id", "kind"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    job_id: UUID = Field(foreign_key="jobs.id", index=True)

    kind: ArtifactKind = Field(index=True)

    label: Optional[str] = Field(default=None, index=True)

    track_id: Optional[UUID] = Field(default=None, foreign_key="tracks.id", index=True)
    wave_id: Optional[UUID] = Field(default=None, foreign_key="waves.id", index=True)

    blob_path: str = Field(index=True)
    content_type: Optional[str] = None
    byte_size: Optional[int] = None

    # Keep DB column name "metadata" but Python attribute name "meta"
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSON, nullable=False),
    )

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    job: Job = Relationship(back_populates="artifacts")


class Track(SQLModel, table=True):
    __tablename__ = "tracks"
    __table_args__ = (
        Index("ix_tracks_job_track_index", "job_id", "track_index"),
        UniqueConstraint("job_id", "track_index", name="uq_tracks_job_track_index"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    job_id: UUID = Field(foreign_key="jobs.id", index=True)

    track_index: int = Field(index=True)
    processed_at: Optional[datetime] = Field(default=None, index=True)

    x0: Optional[int] = Field(default=None, index=True)
    y0: Optional[int] = Field(default=None, index=True)

    amplitude: Optional[float] = Field(default=None, index=True)
    frequency: Optional[float] = Field(default=None, index=True)
    error: Optional[float] = Field(default=None, index=True)

    metrics: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    overlay: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))

    job: Job = Relationship(back_populates="tracks")
    waves: List["Wave"] = Relationship(back_populates="track")
    peaks: List["Peak"] = Relationship(back_populates="track")


class Wave(SQLModel, table=True):
    __tablename__ = "waves"
    __table_args__ = (
        Index("ix_waves_job_track", "job_id", "track_id"),
        Index("ix_waves_job_metrics", "job_id", "amplitude", "frequency", "error"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    job_id: UUID = Field(foreign_key="jobs.id", index=True)
    track_id: Optional[UUID] = Field(default=None, foreign_key="tracks.id", index=True)

    wave_index: Optional[int] = Field(default=None, index=True)

    x: Optional[int] = Field(default=None, index=True)
    y: Optional[int] = Field(default=None, index=True)

    amplitude: Optional[float] = Field(default=None, index=True)
    frequency: Optional[float] = Field(default=None, index=True)
    period: Optional[float] = Field(default=None, index=True)
    error: Optional[float] = Field(default=None, index=True)

    t_start: Optional[float] = None
    t_end: Optional[float] = None

    metrics: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    job: Job = Relationship(back_populates="waves")
    track: Optional[Track] = Relationship(back_populates="waves")


class Peak(SQLModel, table=True):
    __tablename__ = "peaks"
    __table_args__ = (
        Index("ix_peaks_job_track", "job_id", "track_id"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    job_id: UUID = Field(foreign_key="jobs.id", index=True)
    track_id: Optional[UUID] = Field(default=None, foreign_key="tracks.id", index=True)
    wave_id: Optional[UUID] = Field(default=None, foreign_key="waves.id", index=True)

    pos: Optional[float] = Field(default=None, index=True)
    value: Optional[float] = None

    metrics: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    job: Job = Relationship(back_populates="peaks")
    track: Optional[Track] = Relationship(back_populates="peaks")


class JobEvent(SQLModel, table=True):
    __tablename__ = "job_events"
    __table_args__ = (
        UniqueConstraint("job_id", "seq", name="uq_job_events_job_seq"),
        Index("ix_job_events_job_seq", "job_id", "seq"),
        Index("ix_job_events_job_created", "job_id", "created_at"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    job_id: UUID = Field(foreign_key="jobs.id", index=True)

    seq: int = Field(index=True)
    type: EventType = Field(index=True)

    payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    job: Job = Relationship(back_populates="events")


# -----------------------------
# API Schemas (MVP)
# -----------------------------

class JobCreate(SQLModel):
    run_name: str = "untitled"
    config: Union[Dict[str, Any], str] = Field(default_factory=dict)


class JobRead(SQLModel):
    id: UUID
    owner_session_id: UUID
    run_name: str
    status: JobStatus
    cancel_requested: bool
    error: Optional[str]
    error_code: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    updated_at: datetime
    progress: Dict[str, Any]
    tracks_total: Optional[int]
    tracks_done: int
    waves_done: int
    peaks_done: int
    input_filename: Optional[str] = None


class ArtifactRead(SQLModel):
    id: UUID
    job_id: UUID
    kind: ArtifactKind
    label: Optional[str]
    track_id: Optional[UUID]
    wave_id: Optional[UUID]
    blob_path: str
    content_type: Optional[str]
    byte_size: Optional[int]
    meta: Dict[str, Any]
    created_at: datetime


class WaveRead(SQLModel):
    id: UUID
    job_id: UUID
    track_id: Optional[UUID]
    wave_index: Optional[int]
    x: Optional[int]
    y: Optional[int]
    amplitude: Optional[float]
    frequency: Optional[float]
    period: Optional[float]
    error: Optional[float]
    t_start: Optional[float]
    t_end: Optional[float]
    metrics: Dict[str, Any]
    created_at: datetime


class JobEventRead(SQLModel):
    job_id: UUID
    seq: int
    type: EventType
    payload: Dict[str, Any]
    created_at: datetime
