from __future__ import annotations

from fastapi import APIRouter

from .routes_jobs import router as jobs_router
from .ws import router as ws_router

# Put the /api prefix ONCE here (cleaner + avoids double-prefix bugs)
api_router = APIRouter(prefix="/api")

# The child routers should define paths like "/jobs", "/ws/...", etc.
api_router.include_router(jobs_router)
api_router.include_router(ws_router)
