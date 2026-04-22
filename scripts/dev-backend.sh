#!/usr/bin/env bash
set -e

# ---------------------------
# Database
# ---------------------------
export DATABASE_URL="${DATABASE_URL:-sqlite:///./data/waveatlas.local.sqlite}"
export DB_CREATE_ALL=0
mkdir -p ./data

# ---------------------------
# Artifact storage (local only)
# ---------------------------
export ARTIFACT_STORE="local"
export ARTIFACT_ROOT_DIR="./data"

# ---------------------------
# Scratch space
# ---------------------------
export SCRATCH_ROOT="/tmp/mlapp_scratch"

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
