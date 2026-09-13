from __future__ import annotations

from contextlib import closing
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from filelock import FileLock


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))


def default_workspace() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "WaveAtlas"
    # Useful for backend development/tests; Windows is the supported desktop target.
    return Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share")) / "WaveAtlas"


@dataclass
class Workspace:
    root: Path
    resources: Path

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        self.resources = self.resources.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.root / "workspace.lock"), timeout=0)

    @property
    def database(self) -> Path:
        return self.root / "waveatlas.sqlite"

    def configure(self) -> UUID:
        """Call with the workspace lock held, before importing app.db or the API."""
        for folder in ("artifacts", "scratch", "logs", "backups", "cache"):
            (self.root / folder).mkdir(exist_ok=True)
        identity_path = self.root / "workspace.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if identity.get("format") != 1:
                raise RuntimeError(
                    "This workspace was created by an unsupported WaveAtlas version."
                )
            owner = UUID(identity["owner_id"])
        else:
            if self.database.exists():
                raise RuntimeError(
                    "Workspace identity is missing. Restore workspace.json from your backup."
                )
            owner = uuid4()
            temporary = identity_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"format": 1, "owner_id": str(owner)}), encoding="utf-8"
            )
            temporary.replace(identity_path)

        os.environ.update(
            {
                "DATABASE_URL": f"sqlite:///{self.database.as_posix()}",
                "DB_CREATE_ALL": "0",
                "ARTIFACT_STORE": "local",
                "ARTIFACT_ROOT_DIR": str(self.root / "artifacts"),
                "SCRATCH_ROOT": str(self.root / "scratch"),
                "KYMO_EXPORT_DIR": str(self.resources / "export"),
                "FRONTEND_DIST_DIR": str(self.resources / "frontend/dist"),
                "PIPELINE_CONFIG_PATH": str(self.resources / "configs/default.yaml"),
                "CONFIG_DOCS_PATH": str(self.resources / "docs/config.md"),
                "MPLCONFIGDIR": str(self.root / "cache/matplotlib"),
                "MPLBACKEND": "Agg",
                "CORS_ORIGINS": "",
            }
        )
        return owner

    def migrate(self) -> None:
        from alembic import command
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        config = Config()
        config.set_main_option(
            "script_location", str(self.resources / "migrations").replace("%", "%%")
        )
        scripts = ScriptDirectory.from_config(config)
        current = None
        if self.database.exists():
            with closing(sqlite3.connect(self.database)) as db:
                tables = {
                    r[0]
                    for r in db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if "alembic_version" in tables:
                    row = db.execute(
                        "SELECT version_num FROM alembic_version"
                    ).fetchone()
                    current = row[0] if row else None
                if tables and not current:
                    raise RuntimeError(
                        "Database has no migration version. Restore a complete workspace backup."
                    )
            if current:
                try:
                    scripts.get_revision(current)
                except Exception as exc:
                    raise RuntimeError(
                        "This database needs a newer WaveAtlas release. No changes were made."
                    ) from exc
                if current != scripts.get_current_head():
                    from datetime import datetime, timezone

                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    backup = self.root / "backups" / f"before-upgrade-{stamp}.sqlite"
                    with (
                        closing(sqlite3.connect(self.database)) as src,
                        closing(sqlite3.connect(backup)) as dest,
                    ):
                        src.backup(dest)
        command.upgrade(config, "head")

    def validate_resources(self) -> dict:
        import hashlib

        required = [
            "frontend/dist/index.html",
            "configs/default.yaml",
            "docs/config.md",
        ]
        required += [
            f"export/{name}.onnx"
            for name in ("uni_seg", "bi_seg", "classifier", "decision")
        ]
        for name in required:
            if not (self.resources / name).is_file():
                raise RuntimeError(
                    f"Application file is missing: {name}. Reinstall WaveAtlas."
                )
        manifest = self.resources / "build-manifest.json"
        if not manifest.is_file():
            if getattr(sys, "frozen", False):
                raise RuntimeError("Build manifest is missing. Reinstall WaveAtlas.")
            return {"build": "development"}
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        for name in ("uni_seg", "bi_seg", "classifier", "decision"):
            filename = f"{name}.onnx"
            with (self.resources / "export" / filename).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != metadata["models"][filename]:
                raise RuntimeError(
                    f"Model integrity check failed: {filename}. Reinstall WaveAtlas."
                )
        return metadata
