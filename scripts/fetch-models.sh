#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash ./scripts/fetch-models.sh [--repo owner/repo] [--dir path] [--force] [--tag vX.Y.Z] [--pattern "*.onnx"]
# Examples:
#   bash ./scripts/fetch-models.sh
#   bash ./scripts/fetch-models.sh --force
#   bash ./scripts/fetch-models.sh --tag v0.1.0
#   bash ./scripts/fetch-models.sh --pattern "model_*.onnx"

REPO="rohitmahesh1/WaveAtlas"
DIR="export"
FORCE=0
TAG=""
PATTERN="*.onnx"   # default: download all .onnx assets from the release

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)    REPO="$2"; shift 2 ;;
    --dir)     DIR="$2";  shift 2 ;;
    --force)   FORCE=1;   shift ;;
    --tag)     TAG="$2";  shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) not found. Install from https://cli.github.com/"; exit 1
fi

mkdir -p "$DIR"

# Pick latest release tag unless --tag provided
if [[ -z "${TAG}" ]]; then
  echo "Checking latest release for $REPO..."
  TAG="$(gh release view --repo "$REPO" --json tagName -q '.tagName' || true)"
  if [[ -z "${TAG}" ]]; then
    echo "No release found for $REPO (or not accessible)."; exit 1
  fi
fi
echo "Using release tag: $TAG"

# List assets and filter by the pattern (default: *.onnx)
ASSETS="$(gh release view "$TAG" --repo "$REPO" --json assets -q '.assets[].name' || true)"
if [[ -z "${ASSETS//[$'\t\r\n ']/}" ]]; then
  echo "No assets found on release $TAG."; exit 1
fi

echo "Filtering assets with pattern: $PATTERN"
SELECTED=""
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  case "$name" in
    $PATTERN) SELECTED+="$name"$'\n' ;;
  esac
done <<< "$ASSETS"

if [[ -z "${SELECTED//[$'\t\r\n ']/}" ]]; then
  echo "No assets matched pattern '$PATTERN' on release $TAG."; exit 1
fi

echo "Model assets to download:"
printf '  - %s\n' $SELECTED

# Download only missing files unless --force was passed
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  if [[ $FORCE -eq 0 && -s "$DIR/$name" ]]; then
    echo "✓ $name already exists; skipping"
    continue
  fi
  echo "↓ Downloading $name"
  gh release download "$TAG" \
    --repo "$REPO" \
    --pattern "$name" \
    --dir "$DIR" \
    --clobber
done <<< "$SELECTED"

printf "%s\n" "$TAG" > "$DIR/.models_release_tag"
echo "Models are ready in '$DIR/'."
