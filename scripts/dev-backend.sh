#!/usr/bin/env bash
set -e

RESET_LOCAL_DATA=0

usage() {
  cat <<'EOF'
Usage: bash ./scripts/dev-backend.sh [--reset-local-data]

Options:
  --reset-local-data  Delete ignored local backend state in ./data and the scratch directory before starting.
  -h, --help          Show this help.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --reset-local-data)
      RESET_LOCAL_DATA=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# ---------------------------
# Database
# ---------------------------
export DATABASE_URL="${DATABASE_URL:-sqlite:///./data/waveatlas.local.sqlite}"
export DB_CREATE_ALL=0

# ---------------------------
# Artifact storage (local only)
# ---------------------------
export ARTIFACT_STORE="local"
export ARTIFACT_ROOT_DIR="./data"

# ---------------------------
# Scratch space
# ---------------------------
export SCRATCH_ROOT="/tmp/mlapp_scratch"

if [[ "$RESET_LOCAL_DATA" == "1" ]]; then
  if [[ "$DATABASE_URL" != "sqlite:///./data/waveatlas.local.sqlite" ]]; then
    echo "Refusing to reset local data while DATABASE_URL is not the dev SQLite database: $DATABASE_URL" >&2
    exit 1
  fi
  if [[ "$ARTIFACT_STORE" != "local" || "$ARTIFACT_ROOT_DIR" != "./data" ]]; then
    echo "Refusing to reset local data while artifact storage is not ./data." >&2
    exit 1
  fi
  if [[ -L "$ARTIFACT_ROOT_DIR" ]]; then
    echo "Refusing to reset local data because $ARTIFACT_ROOT_DIR is a symlink." >&2
    exit 1
  fi

  echo "Clearing local WaveAtlas data in $ARTIFACT_ROOT_DIR and $SCRATCH_ROOT"
  rm -rf "$ARTIFACT_ROOT_DIR" "$SCRATCH_ROOT"
fi

mkdir -p ./data

# ---------------------------
# Frontend (Vite) → Backend CORS
# ---------------------------
export CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"

# ---------------------------
# Cookies (dev-friendly)
# ---------------------------
export COOKIE_SECURE=0
export COOKIE_SAMESITE="lax"

export KYMO_EXPORT_DIR="./export"

export EMIT_OVERLAY_EVERY_TRACKS=1

# ---------------------------
# Pipeline config (YAML)
# ---------------------------
export PIPELINE_CONFIG_PATH="./configs/default.yaml"


# ---------------------------
# ONNX / model exports (optional)
# ---------------------------
# export KYMO_EXPORT_DIR="./onnx_exports"

# ---------------------------
# Run API
# ---------------------------
alembic upgrade head

uvicorn app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
