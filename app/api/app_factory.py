from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import api_router


def create_app() -> FastAPI:
    app = FastAPI(title="ML Webapp Backend")

    _add_cors(app)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api_router)

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
