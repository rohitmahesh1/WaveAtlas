#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash ./scripts/fetch-samples.sh [--repo owner/repo] [--dir path] [--force] [--tag vX.Y.Z]
# Examples:
#   bash ./scripts/fetch-samples.sh
#   bash ./scripts/fetch-samples.sh --force
#   bash ./scripts/fetch-samples.sh --repo rohitmahesh1/WaveCalling --dir samples
#   bash ./scripts/fetch-samples.sh --tag v0.1.0

REPO="rohitmahesh1/WaveAtlas"
DIR="samples"
FORCE=0
TAG=""

# Patterns considered "samples" (no .onnx here)
SAMPLE_PATTERNS="sample_* *.tif *.tiff *.csv *.tsv *.xlsx *.xls *.png *.jpg *.jpeg *.npz *.npy"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --dir)  DIR="$2";  shift 2 ;;
    --force) FORCE=1; shift ;;
    --tag) TAG="$2"; shift 2 ;;
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

ASSETS="$(gh release view "$TAG" --repo "$REPO" --json assets -q '.assets[].name' || true)"
if [[ -z "${ASSETS//[$'\t\r\n ']/}" ]]; then
  echo "No assets found on release $TAG."; exit 1
fi

echo "Selecting sample assets…"
SELECTED=""
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  # match against any of the sample patterns
  match=0
  for pat in $SAMPLE_PATTERNS; do
    case "$name" in
      $pat) match=1; break;;
    esac
  done
  [[ $match -eq 1 ]] && SELECTED+="$name"$'\n'
done <<< "$ASSETS"

if [[ -z "${SELECTED//[$'\t\r\n ']/}" ]]; then
  echo "No sample assets matched patterns."; exit 1
fi

echo "Samples to download:"
printf '  - %s\n' $SELECTED

# Download only missing files unless --force
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

printf "%s\n" "$TAG" > "$DIR/.samples_release_tag"
echo "Samples are ready in '$DIR/'."
