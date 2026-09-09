"""Frozen backend entry point; only stdlib imports until paths are configured."""

from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket
import sys
import threading
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate packaged models and dependencies, then exit",
    )
    args = parser.parse_args()

    from app.desktop.workspace import Workspace, default_workspace, resource_root

    workspace = Workspace(args.workspace or default_workspace(), resource_root())
    try:
        with workspace.lock:
            owner = workspace.configure()
            handler = RotatingFileHandler(
                workspace.root / "logs/backend.log",
                maxBytes=2_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            logging.basicConfig(
                level=logging.INFO,
                handlers=[handler],
                force=True,
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
            )
            manifest = workspace.validate_resources()
            import onnxruntime as ort

            ort.disable_telemetry_events()
            if args.check:
                import openpyxl  # noqa: F401 -- ensure spreadsheet engines survived freezing
                import xlrd  # noqa: F401

                for name in ("uni_seg", "bi_seg", "classifier", "decision"):
                    ort.InferenceSession(
                        str(workspace.resources / "export" / f"{name}.onnx"),
                        providers=["CPUExecutionProvider"],
                    )
                print(json.dumps({"type": "checked", "models": 4}), flush=True)
                return 0
            workspace.migrate()

            from app.db import engine
            from app.desktop.runtime import DesktopRuntime
            from app.api.app_factory import create_app
            import uvicorn

            desktop = DesktopRuntime(workspace, owner, engine, manifest)
            desktop.recover()
            # Durable checkpoints are stored as artifacts; scratch is disposable.
            import shutil

            for path in (workspace.root / "scratch").iterdir():
                try:
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                except OSError:
                    logging.warning(
                        "Could not remove a stale scratch file", exc_info=True
                    )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                # Bind once to an OS-selected port and give that socket to Uvicorn;
                # there is no close/rebind race or fixed-port collision.
                listener.bind(("127.0.0.1", 0))
                listener.listen(128)
                port = listener.getsockname()[1]
                desktop.origin = f"http://127.0.0.1:{port}"
                desktop.cookie_name = f"waveatlas_{port}"
                server = uvicorn.Server(
                    uvicorn.Config(
                        create_app(desktop=desktop),
                        log_config=None,
                        access_log=False,
                        host="127.0.0.1",
                        port=port,
                        timeout_graceful_shutdown=10,
                    )
                )

                def shutdown():
                    server.should_exit = True

                    # Native inference may not return promptly. The OS releases
                    # the workspace lock and startup recovers durable checkpoints.
                    def force_exit():
                        logging.shutdown()
                        os._exit(0)

                    timer = threading.Timer(15, force_exit)
                    timer.daemon = True
                    timer.start()

                desktop.shutdown = shutdown

                def control():
                    try:
                        for line in sys.stdin:
                            if json.loads(line).get("command") == "shutdown":
                                break
                    except Exception:
                        logging.exception("Desktop control pipe closed")
                    finally:
                        desktop.request_stop()

                def ready():
                    while not server.started and not server.should_exit:
                        time.sleep(0.05)
                    if server.started and not server.should_exit:
                        print(
                            json.dumps(
                                {
                                    "type": "ready",
                                    "origin": desktop.origin,
                                    "url": f"{desktop.origin}/desktop/launch?token={desktop.launch_token}",
                                    "token": desktop.token,
                                }
                            ),
                            flush=True,
                        )

                threading.Thread(target=control, daemon=True).start()
                threading.Thread(target=ready, daemon=True).start()
                server.run(sockets=[listener])
                engine.dispose()
        return 0
    except Exception as exc:
        from filelock import Timeout

        message = (
            "WaveAtlas is already using this workspace."
            if isinstance(exc, Timeout)
            else str(exc)
        )
        logging.exception("Desktop startup failed")
        print(json.dumps({"type": "error", "message": message}), flush=True)
        return 1


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
