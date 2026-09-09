from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient
from filelock import Timeout
from sqlmodel import Session, create_engine, select

from app.api.app_factory import create_app
from app.api.deps import get_db_session
from app.api.job_execution import schedule_job
from app.artifact_store import LocalArtifactStore
from app.desktop.runtime import DesktopRuntime
from app.desktop.workspace import Workspace
from app.job_store import JobStore
from app.models import Job, JobStatus

ROOT = Path(__file__).resolve().parents[1]


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ)
        self.env.start()
        self.workspace = Workspace(
            Path(self.tmp.name) / "Workspace ü with spaces", ROOT
        )
        self.workspace.lock.acquire()
        self.owner = self.workspace.configure()
        self.workspace.migrate()
        self.engine = create_engine(
            f"sqlite:///{self.workspace.database.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        self.desktop = DesktopRuntime(
            self.workspace, self.owner, self.engine, {"test": True}
        )
        self.desktop.origin = "http://127.0.0.1:54321"
        self.desktop.cookie_name = "waveatlas_54321"

    def tearDown(self):
        self.engine.dispose()
        self.workspace.lock.release()
        self.env.stop()
        self.tmp.cleanup()

    def client(self):
        app = create_app(desktop=self.desktop)

        def session():
            with Session(self.engine) as db:
                yield db

        app.dependency_overrides[get_db_session] = session
        return TestClient(app, base_url=self.desktop.origin)

    def test_bootstrap_is_single_use_and_cookie_is_separate_from_owner(self):
        with self.client() as client:
            self.assertEqual(client.get("/api/runtime").status_code, 401)
            launch = self.desktop.launch_token
            response = client.get(
                f"/desktop/launch?token={launch}", follow_redirects=False
            )
            self.assertEqual(response.status_code, 303)
            self.assertIn("HttpOnly", response.headers["set-cookie"])
            self.assertIn("SameSite=strict", response.headers["set-cookie"])
            self.assertEqual(client.get("/api/runtime").json()["mode"], "desktop")
            self.assertEqual(
                client.get(f"/desktop/launch?token={launch}").status_code, 403
            )
            job = client.post("/api/jobs", json={"run_name": "Saved locally"}).json()
            self.assertEqual(job["owner_session_id"], str(self.owner))
            client.cookies.clear()
            # A new authenticated launch recovers the same workspace, not a new owner.
            client.headers["Authorization"] = f"Bearer {self.desktop.token}"
            jobs = client.get("/api/jobs").json()
            self.assertEqual([j["id"] for j in jobs], [job["id"]])

    def test_local_auth_blocks_remote_origins_hosts_and_websockets(self):
        with self.client() as client:
            headers = {"Authorization": f"Bearer {self.desktop.token}"}
            for extra in (
                {"Origin": "https://example.com"},
                {"Origin": "null"},
                {"Host": "evil.example"},
                {"Sec-Fetch-Site": "cross-site"},
            ):
                self.assertEqual(
                    client.get(
                        "/api/runtime", headers={**headers, **extra}
                    ).status_code,
                    403,
                )
            self.assertEqual(
                client.get("/api/runtime", headers=headers).status_code, 200
            )
            from starlette.websockets import WebSocketDisconnect

            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect(f"/api/ws/jobs/{uuid4()}"):
                    pass

    def test_upload_uses_local_relative_artifacts(self):
        with self.client() as client:
            client.headers["Authorization"] = f"Bearer {self.desktop.token}"
            job = client.post("/api/jobs", json={"run_name": "Input"}).json()
            response = client.post(
                f"/api/jobs/{job['id']}/upload",
                files={"file": ("test.csv", b"1,2\n3,4\n", "text/csv")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            artifact = response.json()
            self.assertFalse(Path(artifact["blob_path"]).is_absolute())
            response = client.get(
                f"/api/jobs/{job['id']}/artifacts/{artifact['id']}/download"
            )
            self.assertEqual(response.content, b"1,2\n3,4\n")

    def test_workspace_lock_identity_and_schema_upgrade_backup(self):
        other = Workspace(self.workspace.root, ROOT)
        with self.assertRaises(Timeout):
            other.lock.acquire()
        self.assertEqual(self.workspace.configure(), self.owner)
        with closing(sqlite3.connect(self.workspace.database)) as db:
            # Roll back only the migration metadata and its new schema columns.
            for table in ("waves", "peaks"):
                db.execute(f"DROP INDEX ix_{table}_job_event_kind")
                for column in ("event_polarity", "event_kind", "fit_target"):
                    db.execute(f"DROP INDEX ix_{table}_{column}")
                    db.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            db.execute(
                "UPDATE alembic_version SET version_num='0002_image_upload_kind'"
            )
            db.commit()
        self.workspace.migrate()
        self.assertEqual(
            len(list((self.workspace.root / "backups").glob("*.sqlite"))), 1
        )
        self.workspace.migrate()
        self.assertEqual(
            len(list((self.workspace.root / "backups").glob("*.sqlite"))), 1
        )

    def test_newer_schema_is_rejected_without_mutating_database(self):
        with closing(sqlite3.connect(self.workspace.database)) as db:
            db.execute("UPDATE alembic_version SET version_num='future_release'")
            db.commit()
        before = self.workspace.database.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "newer WaveAtlas"):
            self.workspace.migrate()
        self.assertEqual(before, self.workspace.database.read_bytes())

    def test_recovery_makes_abandoned_jobs_resumable(self):
        with Session(self.engine) as session:
            store = JobStore(session)
            job = store.create_job(owner_session_id=self.owner)
            store.claim_start(job.id)
            job_id = job.id
        self.desktop.recover()
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            self.assertEqual(job.status, JobStatus.failed)
            self.assertEqual(job.error_code, "desktop_interrupted")
            _, claimed = JobStore(session).claim_resume(job_id)
            self.assertTrue(claimed)

    def test_admission_is_atomic_and_released_on_worker_failure(self):
        successes, failures = [], []
        barrier = threading.Barrier(8)

        def reserve():
            barrier.wait()
            try:
                self.desktop.reserve(uuid4())
                successes.append(True)
            except HTTPException:
                failures.append(True)

        threads = [threading.Thread(target=reserve) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual((len(successes), len(failures)), (1, 7))
        self.desktop.release()
        with Session(self.engine) as session:
            store = JobStore(session)
            job = store.create_job(owner_session_id=self.owner)
            background = BackgroundTasks()
            schedule_job(
                store=store,
                job=job,
                config={},
                settings=None,
                artifacts=None,
                background=background,
                engine=self.engine,
                desktop=self.desktop,
            )
            with patch(
                "app.api.job_execution.run_job",
                side_effect=RuntimeError("worker failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    background.tasks[0].func()
            self.assertIsNone(self.desktop.active_job)
            session.expire_all()
            self.assertEqual(session.get(Job, job.id).status, JobStatus.failed)
        self.desktop.request_stop()
        with self.assertRaises(HTTPException) as error:
            self.desktop.reserve(uuid4())
        self.assertEqual(error.exception.status_code, 503)

    def test_desktop_config_locks_models_and_provider(self):
        config = {
            "kymo": {
                "onnx": {"export_dir": "other", "providers": ["CUDAExecutionProvider"]}
            }
        }
        local = self.desktop.constrain_config(config)
        self.assertEqual(local["kymo"]["onnx"]["providers"], ["CPUExecutionProvider"])
        self.assertEqual(config["kymo"]["onnx"]["export_dir"], "other")
        with self.assertRaises(HTTPException):
            self.desktop.constrain_config({"kymo": {"backend": "wolfram"}})

    def test_artifacts_survive_workspace_move_and_reject_traversal(self):
        source = self.workspace.root / "source"
        source.write_bytes(b"research data")
        store = LocalArtifactStore(
            str(self.workspace.root / "artifacts"), relative_keys=True
        )
        key, size = store.put_file(
            job_id=uuid4(),
            kind="upload_csv",
            filename="../input.csv",
            local_path=str(source),
        )
        self.assertEqual(size, 13)
        moved = Path(self.tmp.name) / "moved artifacts"
        shutil.copytree(store.root_dir, moved)
        restored = LocalArtifactStore(str(moved), relative_keys=True)
        self.assertEqual(restored.get_bytes(key), b"research data")
        for bad in ("../source", str(source)):
            with self.assertRaises(ValueError):
                restored.get_bytes(bad)

    def test_real_excel_input_engine_is_installed(self):
        import io
        import openpyxl
        from app.io.table_to_heatmap import _load_table_bytes

        workbook = openpyxl.Workbook()
        workbook.active.append([1, 2])
        workbook.active.append([3, 4])
        stream = io.BytesIO()
        workbook.save(stream)
        values, meta = _load_table_bytes(stream.getvalue(), filename_hint="input.xlsx")
        self.assertEqual(values.values.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(meta["format"], "xlsx")

    def test_resume_refuses_mixed_versions_and_incomplete_checkpoints(self):
        from app.models import ArtifactKind

        with Session(self.engine) as session:
            store = JobStore(session)
            job = store.create_job(
                owner_session_id=self.owner,
                config={"desktop_build": {"version": "future"}},
            )
            artifacts = LocalArtifactStore(
                str(self.workspace.root / "artifacts"), relative_keys=True
            )
            with self.assertRaisesRegex(HTTPException, "different app"):
                self.desktop.validate_resume(job, store, artifacts)
            job.config = {}
            session.add(job)
            session.commit()
            self.desktop.validate_resume(job, store, artifacts)
            key, size = artifacts.put_bytes(
                job_id=job.id,
                kind="track_manifest",
                filename="manifest.json",
                data=b'{"total_tracks": 2}',
            )
            store.create_artifact(
                job_id=job.id,
                kind=ArtifactKind.track_manifest,
                blob_path=key,
                byte_size=size,
                label="tracks_manifest",
                meta={},
            )
            with self.assertRaisesRegex(HTTPException, "incomplete or damaged"):
                self.desktop.validate_resume(job, store, artifacts)


if __name__ == "__main__":
    unittest.main()
