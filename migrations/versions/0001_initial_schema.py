"""Create the initial WaveAtlas schema.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-21
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


job_status = sa.Enum(
    "queued",
    "in_progress",
    "cancel_requested",
    "cancelled",
    "completed",
    "failed",
    name="jobstatus",
)
artifact_kind = sa.Enum(
    "upload_csv",
    "base_heatmap",
    "overlay",
    "export_waves_csv",
    "export_tracks_csv",
    "export_peaks_csv",
    "export_progress_json",
    "debug_text",
    "track_npy",
    "track_manifest",
    "other",
    name="artifactkind",
)
event_type = sa.Enum(
    "status",
    "progress",
    "error",
    "done",
    "cancelled",
    "user_log",
    "overlay_track",
    "overlay_ready",
    "waves_batch",
    name="eventtype",
)


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_session_id", sa.Uuid(), nullable=False),
        sa.Column("run_name", sa.String(), nullable=False),
        sa.Column("status", job_status, nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("progress", sa.JSON(), nullable=False),
        sa.Column("tracks_total", sa.Integer(), nullable=True),
        sa.Column("tracks_done", sa.Integer(), nullable=False),
        sa.Column("waves_done", sa.Integer(), nullable=False),
        sa.Column("peaks_done", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_cancel_requested", "jobs", ["cancel_requested"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])
    op.create_index("ix_jobs_error_code", "jobs", ["error_code"])
    op.create_index("ix_jobs_finished_at", "jobs", ["finished_at"])
    op.create_index("ix_jobs_id", "jobs", ["id"])
    op.create_index("ix_jobs_owner_session_id", "jobs", ["owner_session_id"])
    op.create_index("ix_jobs_peaks_done", "jobs", ["peaks_done"])
    op.create_index("ix_jobs_run_name", "jobs", ["run_name"])
    op.create_index("ix_jobs_started_at", "jobs", ["started_at"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_tracks_done", "jobs", ["tracks_done"])
    op.create_index("ix_jobs_updated_at", "jobs", ["updated_at"])
    op.create_index("ix_jobs_waves_done", "jobs", ["waves_done"])

    op.create_table(
        "job_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("type", event_type, nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "seq", name="uq_job_events_job_seq"),
    )
    op.create_index("ix_job_events_created_at", "job_events", ["created_at"])
    op.create_index("ix_job_events_id", "job_events", ["id"])
    op.create_index("ix_job_events_job_created", "job_events", ["job_id", "created_at"])
    op.create_index("ix_job_events_job_id", "job_events", ["job_id"])
    op.create_index("ix_job_events_job_seq", "job_events", ["job_id", "seq"])
    op.create_index("ix_job_events_seq", "job_events", ["seq"])
    op.create_index("ix_job_events_type", "job_events", ["type"])

    op.create_table(
        "tracks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("track_index", sa.Integer(), nullable=False),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        sa.Column("x0", sa.Integer(), nullable=True),
        sa.Column("y0", sa.Integer(), nullable=True),
        sa.Column("amplitude", sa.Float(), nullable=True),
        sa.Column("frequency", sa.Float(), nullable=True),
        sa.Column("error", sa.Float(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("overlay", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "track_index", name="uq_tracks_job_track_index"),
    )
    op.create_index("ix_tracks_amplitude", "tracks", ["amplitude"])
    op.create_index("ix_tracks_error", "tracks", ["error"])
    op.create_index("ix_tracks_frequency", "tracks", ["frequency"])
    op.create_index("ix_tracks_id", "tracks", ["id"])
    op.create_index("ix_tracks_job_id", "tracks", ["job_id"])
    op.create_index("ix_tracks_job_track_index", "tracks", ["job_id", "track_index"])
    op.create_index("ix_tracks_processed_at", "tracks", ["processed_at"])
    op.create_index("ix_tracks_track_index", "tracks", ["track_index"])
    op.create_index("ix_tracks_x0", "tracks", ["x0"])
    op.create_index("ix_tracks_y0", "tracks", ["y0"])

    op.create_table(
        "waves",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("track_id", sa.Uuid(), nullable=True),
        sa.Column("wave_index", sa.Integer(), nullable=True),
        sa.Column("x", sa.Integer(), nullable=True),
        sa.Column("y", sa.Integer(), nullable=True),
        sa.Column("amplitude", sa.Float(), nullable=True),
        sa.Column("frequency", sa.Float(), nullable=True),
        sa.Column("period", sa.Float(), nullable=True),
        sa.Column("error", sa.Float(), nullable=True),
        sa.Column("t_start", sa.Float(), nullable=True),
        sa.Column("t_end", sa.Float(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["track_id"], ["tracks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_waves_amplitude", "waves", ["amplitude"])
    op.create_index("ix_waves_created_at", "waves", ["created_at"])
    op.create_index("ix_waves_error", "waves", ["error"])
    op.create_index("ix_waves_frequency", "waves", ["frequency"])
    op.create_index("ix_waves_id", "waves", ["id"])
    op.create_index("ix_waves_job_id", "waves", ["job_id"])
    op.create_index("ix_waves_job_metrics", "waves", ["job_id", "amplitude", "frequency", "error"])
    op.create_index("ix_waves_job_track", "waves", ["job_id", "track_id"])
    op.create_index("ix_waves_period", "waves", ["period"])
    op.create_index("ix_waves_track_id", "waves", ["track_id"])
    op.create_index("ix_waves_wave_index", "waves", ["wave_index"])
    op.create_index("ix_waves_x", "waves", ["x"])
    op.create_index("ix_waves_y", "waves", ["y"])

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("kind", artifact_kind, nullable=False),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("track_id", sa.Uuid(), nullable=True),
        sa.Column("wave_id", sa.Uuid(), nullable=True),
        sa.Column("blob_path", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=True),
        sa.Column("byte_size", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["track_id"], ["tracks.id"]),
        sa.ForeignKeyConstraint(["wave_id"], ["waves.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_artifacts_blob_path", "artifacts", ["blob_path"])
    op.create_index("ix_artifacts_created_at", "artifacts", ["created_at"])
    op.create_index("ix_artifacts_id", "artifacts", ["id"])
    op.create_index("ix_artifacts_job_id", "artifacts", ["job_id"])
    op.create_index("ix_artifacts_job_kind", "artifacts", ["job_id", "kind"])
    op.create_index("ix_artifacts_kind", "artifacts", ["kind"])
    op.create_index("ix_artifacts_label", "artifacts", ["label"])
    op.create_index("ix_artifacts_track_id", "artifacts", ["track_id"])
    op.create_index("ix_artifacts_wave_id", "artifacts", ["wave_id"])

    op.create_table(
        "peaks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("track_id", sa.Uuid(), nullable=True),
        sa.Column("wave_id", sa.Uuid(), nullable=True),
        sa.Column("pos", sa.Float(), nullable=True),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["track_id"], ["tracks.id"]),
        sa.ForeignKeyConstraint(["wave_id"], ["waves.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_peaks_created_at", "peaks", ["created_at"])
    op.create_index("ix_peaks_id", "peaks", ["id"])
    op.create_index("ix_peaks_job_id", "peaks", ["job_id"])
    op.create_index("ix_peaks_job_track", "peaks", ["job_id", "track_id"])
    op.create_index("ix_peaks_pos", "peaks", ["pos"])
    op.create_index("ix_peaks_track_id", "peaks", ["track_id"])
    op.create_index("ix_peaks_wave_id", "peaks", ["wave_id"])


def downgrade() -> None:
    op.drop_index("ix_peaks_wave_id", table_name="peaks")
    op.drop_index("ix_peaks_track_id", table_name="peaks")
    op.drop_index("ix_peaks_pos", table_name="peaks")
    op.drop_index("ix_peaks_job_track", table_name="peaks")
    op.drop_index("ix_peaks_job_id", table_name="peaks")
    op.drop_index("ix_peaks_id", table_name="peaks")
    op.drop_index("ix_peaks_created_at", table_name="peaks")
    op.drop_table("peaks")

    op.drop_index("ix_artifacts_wave_id", table_name="artifacts")
    op.drop_index("ix_artifacts_track_id", table_name="artifacts")
    op.drop_index("ix_artifacts_label", table_name="artifacts")
    op.drop_index("ix_artifacts_kind", table_name="artifacts")
    op.drop_index("ix_artifacts_job_kind", table_name="artifacts")
    op.drop_index("ix_artifacts_job_id", table_name="artifacts")
    op.drop_index("ix_artifacts_id", table_name="artifacts")
    op.drop_index("ix_artifacts_created_at", table_name="artifacts")
    op.drop_index("ix_artifacts_blob_path", table_name="artifacts")
    op.drop_table("artifacts")

    op.drop_index("ix_waves_y", table_name="waves")
    op.drop_index("ix_waves_x", table_name="waves")
    op.drop_index("ix_waves_wave_index", table_name="waves")
    op.drop_index("ix_waves_track_id", table_name="waves")
    op.drop_index("ix_waves_period", table_name="waves")
    op.drop_index("ix_waves_job_track", table_name="waves")
    op.drop_index("ix_waves_job_metrics", table_name="waves")
    op.drop_index("ix_waves_job_id", table_name="waves")
    op.drop_index("ix_waves_id", table_name="waves")
    op.drop_index("ix_waves_frequency", table_name="waves")
    op.drop_index("ix_waves_error", table_name="waves")
    op.drop_index("ix_waves_created_at", table_name="waves")
    op.drop_index("ix_waves_amplitude", table_name="waves")
    op.drop_table("waves")

    op.drop_index("ix_tracks_y0", table_name="tracks")
    op.drop_index("ix_tracks_x0", table_name="tracks")
    op.drop_index("ix_tracks_track_index", table_name="tracks")
    op.drop_index("ix_tracks_processed_at", table_name="tracks")
    op.drop_index("ix_tracks_job_track_index", table_name="tracks")
    op.drop_index("ix_tracks_job_id", table_name="tracks")
    op.drop_index("ix_tracks_id", table_name="tracks")
    op.drop_index("ix_tracks_frequency", table_name="tracks")
    op.drop_index("ix_tracks_error", table_name="tracks")
    op.drop_index("ix_tracks_amplitude", table_name="tracks")
    op.drop_table("tracks")

    op.drop_index("ix_job_events_type", table_name="job_events")
    op.drop_index("ix_job_events_seq", table_name="job_events")
    op.drop_index("ix_job_events_job_seq", table_name="job_events")
    op.drop_index("ix_job_events_job_id", table_name="job_events")
    op.drop_index("ix_job_events_job_created", table_name="job_events")
    op.drop_index("ix_job_events_id", table_name="job_events")
    op.drop_index("ix_job_events_created_at", table_name="job_events")
    op.drop_table("job_events")

    op.drop_index("ix_jobs_waves_done", table_name="jobs")
    op.drop_index("ix_jobs_updated_at", table_name="jobs")
    op.drop_index("ix_jobs_tracks_done", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_started_at", table_name="jobs")
    op.drop_index("ix_jobs_run_name", table_name="jobs")
    op.drop_index("ix_jobs_peaks_done", table_name="jobs")
    op.drop_index("ix_jobs_owner_session_id", table_name="jobs")
    op.drop_index("ix_jobs_id", table_name="jobs")
    op.drop_index("ix_jobs_finished_at", table_name="jobs")
    op.drop_index("ix_jobs_error_code", table_name="jobs")
    op.drop_index("ix_jobs_created_at", table_name="jobs")
    op.drop_index("ix_jobs_cancel_requested", table_name="jobs")
    op.drop_table("jobs")

    bind = op.get_bind()
    event_type.drop(bind, checkfirst=True)
    artifact_kind.drop(bind, checkfirst=True)
    job_status.drop(bind, checkfirst=True)
