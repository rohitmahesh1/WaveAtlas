from __future__ import annotations

import logging
import secrets
import threading
from typing import Callable
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session, select

from ..job_store import JobStore
from ..models import Job, JobStatus
from .workspace import Workspace


class DesktopRuntime:
    def __init__(self, workspace: Workspace, owner: UUID, engine, manifest: dict):
        self.workspace = workspace
        self.owner = owner
        self.engine = engine
        self.manifest = manifest
        self.version = str(manifest.get("version") or "development")
        self.token = secrets.token_urlsafe(32)
        self.launch_token = secrets.token_urlsafe(32)
        self.origin = ""
        self.cookie_name = ""
        self._lock = threading.Lock()
        self.active_job: UUID | None = None
        self.stopping = False
        self.shutdown: Callable[[], None] = lambda: None

    def reserve(self, job_id: UUID) -> None:
        with self._lock:
            if self.stopping:
                raise HTTPException(
                    503,
                    "WaveAtlas is closing. Reopen the app to start another analysis.",
                )
            if self.active_job is not None:
                raise HTTPException(
                    409,
                    "An analysis is already running. Wait for it to finish or stop it first.",
                )
            self.active_job = job_id

    def release(self) -> None:
        with self._lock:
            self.active_job = None

    def status(self) -> dict:
        with self._lock:
            return {
                "active_job": str(self.active_job) if self.active_job else None,
                "stopping": self.stopping,
            }

    def request_stop(self) -> None:
        with self._lock:
            self.stopping = True
            active = self.active_job
        if active:
            with Session(self.engine) as session:
                JobStore(session).request_cancel(active)
        self.shutdown()

    def recover(self) -> None:
        # The caller holds the process-wide workspace file lock, so no live worker
        # can own these rows. Persist checkpoints, but mark abandoned runs resumable.
        with Session(self.engine) as session:
            jobs = session.exec(
                select(Job).where(
                    Job.status.in_((JobStatus.in_progress, JobStatus.cancel_requested))
                )
            ).all()
            store = JobStore(session)
            for job in jobs:
                job.cancel_requested = False
                session.add(job)
                store.set_status(
                    job.id,
                    JobStatus.failed,
                    error="WaveAtlas closed before this analysis finished. Resume to reuse available checkpoints.",
                    error_code="desktop_interrupted",
                )
        logging.getLogger(__name__).info("Recovered %d interrupted analyses", len(jobs))

    def constrain_config(self, config: dict) -> dict:
        from copy import deepcopy

        config = deepcopy(config)
        kymo = config.setdefault("kymo", {})
        if kymo.get("backend", "onnx") != "onnx":
            raise HTTPException(
                422, "The Windows app supports the bundled ONNX backend only."
            )
        onnx = kymo.setdefault("onnx", {})
        onnx["export_dir"] = str(self.workspace.resources / "export")
        onnx["providers"] = ["CPUExecutionProvider"]
        config["desktop_build"] = {"version": self.version, **self.manifest}
        return config

    def validate_resume(self, job, store, artifacts) -> None:
        import io
        import json
        import numpy as np
        from ..models import ArtifactKind

        previous = (job.config or {}).get("desktop_build", {})
        if previous and (
            previous.get("version") != self.version
            or previous.get("models") != self.manifest.get("models")
        ):
            raise HTTPException(
                409,
                "This analysis used a different app or model version. Start a new run with the original input.",
            )
        manifests = store.list_artifacts(
            job.id, kind=ArtifactKind.track_manifest, label="tracks_manifest", limit=1
        )
        tracks = store.list_artifacts(job.id, kind=ArtifactKind.track_npy, limit=100000)
        if not manifests and not tracks and not job.tracks_done:
            return  # No extraction checkpoint exists yet; recompute from the input.
        try:
            manifest = json.loads(artifacts.get_bytes(manifests[0].blob_path))
            total = int(manifest["total_tracks"])
            mapping = {int(a.meta["track_index"]): a for a in tracks}
            if total <= 0 or any(i not in mapping for i in range(total)):
                raise ValueError("Incomplete track set")
            for i in range(total):
                points = np.load(
                    io.BytesIO(artifacts.get_bytes(mapping[i].blob_path)),
                    allow_pickle=False,
                )
                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or not np.isfinite(points).all()
                ):
                    raise ValueError("Invalid track checkpoint")
        except Exception as exc:
            raise HTTPException(
                409,
                "This run has an incomplete or damaged checkpoint. Start a new run with the original input; existing results have been preserved.",
            ) from exc
