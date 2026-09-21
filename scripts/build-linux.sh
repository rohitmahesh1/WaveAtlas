#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--container" ]]; then
    if [[ "$(uname -m)" != "x86_64" ]]; then
        echo "The Linux x64 release must be built on an x86_64 host." >&2
        exit 1
    fi
    commit="$(git -C "$root" rev-parse HEAD)"
    podman_args=()
    if ! podman info >/dev/null 2>&1; then
        storage_root="${TMPDIR:-/tmp}/waveatlas-podman-root"
        run_root="${TMPDIR:-/tmp}/waveatlas-podman-runroot"
        podman_args=(
            --root "$storage_root"
            --runroot "$run_root"
            --storage-driver vfs
        )
    fi
    exec podman "${podman_args[@]}" run --rm --network host \
        -e WAVEATLAS_LINUX_CONTAINER=1 \
        -e WAVEATLAS_BUILD_COMMIT="$commit" \
        -v "$root:/workspace" -w /workspace ubuntu:22.04 \
        bash scripts/build-linux.sh
fi

if [[ "${WAVEATLAS_LINUX_CONTAINER:-}" != "1" ]]; then
    echo "Run scripts/build-linux.sh --container to use the supported Ubuntu 22.04 builder." >&2
    exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "Linux releases must be built inside Ubuntu 22.04." >&2
    exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This script currently produces only Linux x86_64 releases." >&2
    exit 1
fi
if [[ ! "${WAVEATLAS_BUILD_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "WAVEATLAS_BUILD_COMMIT must identify the exact source commit." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    build-essential ca-certificates curl file libayatana-appindicator3-dev \
    libfuse2 libssl-dev libwebkit2gtk-4.1-dev libxdo-dev patchelf \
    pkg-config python3-pip python3-venv librsvg2-dev wget xz-utils

python3 -m pip install --disable-pip-version-check --no-cache-dir "uv==0.12.17"
uv python install 3.11.16
rm -rf /opt/waveatlas-python
uv venv --python 3.11.16 /opt/waveatlas-python
uv pip install --python /opt/waveatlas-python/bin/python --require-hashes \
    -r desktop/requirements-linux.lock -r desktop/build-requirements-linux.lock

node_version="22.23.2"
node_archive="node-v${node_version}-linux-x64.tar.xz"
curl -fsSLo "/tmp/$node_archive" \
    "https://nodejs.org/download/release/v${node_version}/${node_archive}"
echo "d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307  /tmp/$node_archive" \
    | sha256sum --check --status
rm -rf /opt/node
mkdir -p /opt/node
tar -xJf "/tmp/$node_archive" --strip-components=1 -C /opt/node

curl --proto '=https' --tlsv1.2 -fsSLo /tmp/rustup-init \
    https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init
chmod +x /tmp/rustup-init
/tmp/rustup-init -y --profile minimal --default-toolchain stable --no-modify-path
export PATH="/root/.cargo/bin:$PATH"
rustup component add rustfmt

export PATH="/opt/waveatlas-python/bin:/opt/node/bin:$PATH"
cd "$root"

version="$(python scripts/version.py --check)"
python -c "import sys; assert sys.version_info[:2] == (3, 11)"
npm ci --prefix frontend
npm run build --prefix frontend
npm ci --prefix desktop
python scripts/prepare-desktop.py
python -m unittest discover -s tests -v
cargo fmt --manifest-path desktop/src-tauri/Cargo.toml -- --check

rm -rf desktop/payload desktop/build/backend
python -m PyInstaller --noconfirm --distpath desktop --workpath desktop/build \
    desktop/backend.spec
test -x desktop/payload/waveatlas-backend
file desktop/payload/waveatlas-backend | grep -q 'ELF 64-bit.*x86-64'

cargo test --manifest-path desktop/src-tauri/Cargo.toml --locked
desktop/payload/waveatlas-backend \
    --workspace "$root/desktop/build/check-workspace-linux" --check
python scripts/smoke-desktop.py --backend "$root/desktop/payload/waveatlas-backend"

rm -rf desktop/src-tauri/target/release/bundle/appimage \
    desktop/src-tauri/target/release/bundle/deb
npm run build --prefix desktop -- --bundles deb -- --locked
export APPIMAGE_EXTRACT_AND_RUN=1
export NO_STRIP=true
npm run build --prefix desktop -- --bundles appimage --verbose -- --locked

mapfile -t appimages < <(find desktop/src-tauri/target/release/bundle/appimage \
    -maxdepth 1 -type f -name '*.AppImage')
mapfile -t debs < <(find desktop/src-tauri/target/release/bundle/deb \
    -maxdepth 1 -type f -name '*.deb')
if [[ "${#appimages[@]}" != 1 || "${#debs[@]}" != 1 ]]; then
    echo "Expected one AppImage and one Debian package." >&2
    exit 1
fi

release_dir="$root/desktop/build/release-linux"
rm -rf "$release_dir"
mkdir -p "$release_dir"
appimage_name="WaveAtlas-${version}-linux-x86_64.AppImage"
deb_name="WaveAtlas-${version}-linux-x86_64.deb"
install -m 0755 "${appimages[0]}" "$release_dir/$appimage_name"
install -m 0644 "${debs[0]}" "$release_dir/$deb_name"
file "$release_dir/$appimage_name" | grep -q 'ELF 64-bit.*x86-64'
dpkg-deb --info "$release_dir/$deb_name" >/dev/null
(
    cd "$release_dir"
    sha256sum "$appimage_name" "$deb_name" > SHA256SUMS.txt
)
echo "Linux release ready under desktop/build/release-linux."
