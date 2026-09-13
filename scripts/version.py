"""Show, set, or validate the single WaveAtlas application version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CARGO_TOML = ROOT / "desktop/src-tauri/Cargo.toml"
CARGO_LOCK = ROOT / "desktop/src-tauri/Cargo.lock"
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def cargo_version() -> str:
    data = tomllib.loads(CARGO_TOML.read_text(encoding="utf-8"))
    return str(data["package"]["version"])


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    original = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, original, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not update the WaveAtlas version in {path}")
    path.write_text(updated, encoding="utf-8")


def set_version(version: str) -> None:
    if SEMVER.fullmatch(version) is None:
        raise SystemExit(f"Invalid semantic version: {version}")
    replace_once(
        CARGO_TOML,
        r'^(version\s*=\s*")[^"]+("\s*)$',
        rf"\g<1>{version}\g<2>",
    )
    replace_once(
        CARGO_LOCK,
        r'(\[\[package\]\]\nname = "waveatlas-desktop"\nversion = ")[^"]+("\s*)',
        rf"\g<1>{version}\g<2>",
    )


def check() -> str:
    version = cargo_version()
    if SEMVER.fullmatch(version) is None:
        raise SystemExit(f"Cargo package version is not valid SemVer: {version}")
    lock = tomllib.loads(CARGO_LOCK.read_text(encoding="utf-8"))
    locked = next(
        package["version"]
        for package in lock["package"]
        if package["name"] == "waveatlas-desktop"
    )
    if locked != version:
        raise SystemExit(
            f"Cargo.lock has WaveAtlas {locked}, but Cargo.toml has {version}"
        )
    tauri = json.loads(
        (ROOT / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8")
    )
    if "version" in tauri:
        raise SystemExit("Remove the duplicate application version from tauri.conf.json")
    if tauri.get("bundle", {}).get("windows", {}).get("allowDowngrades") is not False:
        raise SystemExit("Windows release builds must disable installer downgrades")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="new semantic version")
    parser.add_argument("--check", action="store_true", help="validate release metadata")
    args = parser.parse_args()
    if args.version:
        set_version(args.version)
    version = check()
    print(version)


if __name__ == "__main__":
    main()
