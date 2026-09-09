"""Exercise the actual packaged backend with real models, without a browser.

Used with the Windows executable or the source backend during development. Requests
stay on loopback. The private readiness message is never printed to build logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


class Backend:
    def __init__(self, command, workspace):
        self.process = subprocess.Popen(
            [*command, "--workspace", str(workspace)],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        messages = queue.Queue()

        def read():
            for line in self.process.stdout:
                try:
                    messages.put(json.loads(line))
                except ValueError:
                    pass
            messages.put(
                {"type": "error", "message": "Backend exited before readiness"}
            )

        threading.Thread(target=read, daemon=True).start()
        try:
            ready = messages.get(timeout=180)
            if ready["type"] != "ready":
                raise RuntimeError(ready.get("message"))
            self.origin, self.token = ready["origin"], ready["token"]
        except BaseException:
            self.process.kill()
            self.process.communicate()
            raise

    def request(self, path, data=None, content_type="application/json", raw=False):
        headers = {"Authorization": f"Bearer {self.token}"}
        if data is not None:
            headers["Content-Type"] = content_type
            if not isinstance(data, bytes):
                data = json.dumps(data).encode()
        request = urllib.request.Request(self.origin + path, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            return payload if raw else json.loads(payload)

    def stop(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"command":"shutdown"}\n')
            self.process.stdin.flush()
        self.process.communicate(timeout=25)
        if self.process.returncode != 0:
            raise RuntimeError("Backend exited unsuccessfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path)
    args = parser.parse_args()
    command = (
        [str(args.backend.resolve())]
        if args.backend
        else [sys.executable, "-m", "app.desktop.entry"]
    )
    with tempfile.TemporaryDirectory(prefix="waveatlas-smoke-") as temporary:
        workspace = Path(temporary) / "Lab workspace ü"
        backend = Backend(command, workspace)
        try:
            assert backend.request("/api/runtime")["upload_transport"] == "local"
            assert b"<html" in backend.request("/", raw=True).lower()
            assert backend.request("/health")["status"] == "ok"
            assert backend.request("/api/docs/config", raw=True)
            # A small synthetic image exercises the real ONNX extractor. Scientific
            # shape/metric correctness is covered by the core regression suite.
            import io
            import numpy as np
            from PIL import Image

            values = np.zeros((160, 128), dtype=np.uint8)
            for y in range(160):
                for base in (25, 65, 105):
                    x = base + int(8 * np.sin(y / 14))
                    values[y, max(0, x - 2) : min(128, x + 3)] = 255
            for mode in ("standard", "ripple_family", "large_wave"):
                image_values = values
                if mode != "standard":
                    rows = np.arange(240)[:, None]
                    cols = np.arange(240)[None, :]
                    offset = (
                        0.5 * rows
                        if mode == "ripple_family"
                        else 25 * np.sin(rows / 40)
                    )
                    image_values = np.asarray(
                        127 + 120 * np.cos(2 * np.pi * (cols - offset) / 48),
                        dtype=np.uint8,
                    )
                stream = io.BytesIO()
                Image.fromarray(image_values).save(stream, format="PNG")
                job = backend.request(
                    "/api/jobs",
                    {
                        "run_name": mode,
                        "config": {
                            "analysis": {"mode": mode},
                            "viz": {"enabled": False},
                        },
                    },
                )
                job_id = job["id"]
                boundary = uuid4().hex
                body = (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
                    + stream.getvalue()
                    + f"\r\n--{boundary}--\r\n".encode()
                )
                backend.request(
                    f"/api/jobs/{job_id}/upload",
                    body,
                    f"multipart/form-data; boundary={boundary}",
                )
                backend.request(f"/api/jobs/{job_id}/start", {})
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    job = backend.request(f"/api/jobs/{job_id}")
                    if job["status"] in ("completed", "failed", "cancelled"):
                        break
                    time.sleep(0.5)
                if job["status"] != "completed":
                    raise RuntimeError(
                        f"{mode} analysis failed: {job.get('error', job['status'])}"
                    )
                assert job["tracks_done"] > 0, f"No tracks were measured for {mode}"
                artifacts = backend.request(f"/api/jobs/{job_id}/artifacts")
                assert artifacts, f"No artifacts for {mode}"
                artifact = artifacts[0]
                assert backend.request(
                    f"/api/jobs/{job_id}/artifacts/{artifact['id']}/download", raw=True
                )
                print(
                    f"{mode}: completed; {job['tracks_done']} tracks; {len(artifacts)} artifacts"
                )
            original = {job["id"] for job in backend.request("/api/jobs")}
        finally:
            backend.stop()
        backend = Backend(command, workspace)
        try:
            assert {job["id"] for job in backend.request("/api/jobs")} == original
            print("Relaunch: workspace history preserved with a new launch session")
        finally:
            backend.stop()


if __name__ == "__main__":
    main()
