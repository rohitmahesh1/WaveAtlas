"""Prepare verified, versioned inputs for a desktop build (never at app startup)."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_commit() -> str:
    commit = os.environ.get("WAVEATLAS_BUILD_COMMIT")
    if commit is None:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Unable to determine the source commit; set WAVEATLAS_BUILD_COMMIT "
                "when building from an exported source archive."
            ) from error
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise RuntimeError("The source commit must be a 40-character Git object ID.")
    return commit.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-models", action="store_true")
    args = parser.parse_args()
    models = json.loads((ROOT / "desktop/models.json").read_text(encoding="utf-8"))
    exports = ROOT / "export"
    exports.mkdir(exist_ok=True)
    for name, expected in models["files"].items():
        path = exports / name
        if not path.exists() and args.fetch_models:
            url = f"https://github.com/{models['repository']}/releases/download/{models['release']}/{name}"
            temporary = path.with_suffix(".download")
            try:
                with urllib.request.urlopen(url) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                if digest(temporary) != expected:
                    raise RuntimeError(f"Downloaded model checksum mismatch: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(
                f"Model missing or checksum mismatch: {path}; use the pinned release in desktop/models.json"
            )

    if not (ROOT / "frontend/dist/index.html").is_file():
        raise RuntimeError("Build the frontend before preparing the desktop bundle.")
    sys.path.insert(0, str(ROOT))
    from app.desktop import VERSION

    for file in ("desktop/src-tauri/tauri.conf.json", "desktop/package.json"):
        if json.loads((ROOT / file).read_text(encoding="utf-8"))["version"] != VERSION:
            raise RuntimeError(f"Version mismatch in {file}")
    packages = {
        d.metadata["Name"]: d.version
        for d in metadata.distributions()
        if d.metadata["Name"]
    }
    manifest = {
        "version": VERSION,
        "commit": build_commit(),
        "model_release": models["release"],
        "models": models["files"],
        "python": sys.version.split()[0],
        "packages": packages,
    }
    (ROOT / "desktop/build-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    notices = []
    for dist in sorted(
        metadata.distributions(), key=lambda d: d.metadata.get("Name", "")
    ):
        notices.append(f"\n{'=' * 72}\n{dist.metadata.get('Name')} {dist.version}\n")
        for file in dist.files or ():
            if any(
                p.lower().startswith(("license", "copying", "notice"))
                for p in file.parts
            ):
                path = dist.locate_file(file)
                if path.is_file():
                    notices.append(path.read_text(encoding="utf-8", errors="replace"))
    (ROOT / "desktop/THIRD_PARTY_NOTICES.txt").write_text(
        "\n".join(notices), encoding="utf-8"
    )

    # A small code-drawn application icon. No external artwork or network needed.
    from PIL import Image, ImageDraw
    import math

    icon = Image.new("RGBA", (256, 256), "#111827")
    draw = ImageDraw.Draw(icon)
    points = [
        (x, 128 - int(55 * math.sin((x - 28) * math.pi * 4 / 200)))
        for x in range(28, 229)
    ]
    draw.line(points, fill="#67e8f9", width=14)
    icons = ROOT / "desktop/src-tauri/icons"
    icons.mkdir(exist_ok=True)
    icon.save(
        icons / "icon.ico", sizes=[(32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    )
    print(f"Desktop inputs ready: {VERSION}, model release {models['release']}")


if __name__ == "__main__":
    main()
