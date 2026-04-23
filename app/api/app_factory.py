from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .router import api_router


def create_app() -> FastAPI:
    app = FastAPI(title="ML Webapp Backend")

    _add_cors(app)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api_router)
    _mount_frontend(app)

    return app


def _add_cors(app: FastAPI) -> None:
    origins_raw = os.getenv("CORS_ORIGINS", "")
    if origins_raw.strip() == "*":
        origins = ["*"]
        allow_credentials = False
    else:
        origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
        allow_credentials = True

    # If no origins specified, keep CORS closed by default (safe).
    if not origins:
        return

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )


def _frontend_dist_dir() -> Path:
    configured = os.getenv("FRONTEND_DIST_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "frontend" / "dist").resolve()


def _mount_frontend(app: FastAPI) -> None:
    dist_dir = _frontend_dist_dir()
    index_path = dist_dir / "index.html"
    if not index_path.is_file():
        return

    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/{path:path}", include_in_schema=False)
    def frontend_fallback(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        candidate = (dist_dir / path).resolve()
        if candidate.is_file() and (candidate == dist_dir or dist_dir in candidate.parents):
            return FileResponse(candidate)
        return FileResponse(index_path)
