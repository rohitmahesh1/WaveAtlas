# app/api/ws.py
from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from ..db import engine
from ..job_store import JobStore
from ..models import Job
from .deps import get_owner_session_id_ws

router = APIRouter(tags=["ws"])


def _job_owned(job_id: UUID, owner_session_id: UUID) -> bool:
    with Session(engine) as session:
        q = select(Job.id).where(Job.id == job_id, Job.owner_session_id == owner_session_id)
        return session.exec(q).first() is not None


def _job_snapshot(job_id: UUID) -> dict:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            return {}
        return {
            "id": str(job.id),
            "run_name": job.run_name,
            "status": job.status,
            "cancel_requested": job.cancel_requested,
            "error": job.error,
            "error_code": job.error_code,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "updated_at": job.updated_at.isoformat(),
            "progress": job.progress or {},
            "tracks_total": job.tracks_total,
            "tracks_done": job.tracks_done,
            "waves_done": job.waves_done,
            "peaks_done": job.peaks_done,
        }


@router.websocket("/ws/jobs/{job_id}")
async def ws_job_events(
    websocket: WebSocket,
    job_id: UUID,
    after_seq: int = Query(0, ge=0),
    poll_interval: float = Query(0.5, gt=0.05, le=5.0),
    ping_interval: float = Query(10.0, gt=1.0, le=60.0),
):
    await websocket.accept()

    # Auth: session cookie must already exist for WS
    try:
        owner_session_id = get_owner_session_id_ws(websocket)  # type: ignore[arg-type]
    except Exception:
        await websocket.close(code=4401)
        return

    if not _job_owned(job_id, owner_session_id):
        await websocket.close(code=4404)
        return

    # Initial snapshot
    await websocket.send_json(
        {
            "job_id": str(job_id),
            "seq": after_seq,
            "type": "snapshot",
            "payload": _job_snapshot(job_id),
        }
    )

    last = after_seq
    last_ping = asyncio.get_event_loop().time()

    try:
        while True:
            # IMPORTANT: short-lived DB session per poll (prevents pool exhaustion)
            with Session(engine) as session:
                store = JobStore(session=session)
                events = store.get_events_after(job_id, after_seq=last, limit=500)

                for ev in events:
                    await websocket.send_json(
                        {
                            "job_id": str(job_id),
                            "seq": ev.seq,
                            "type": ev.type,
                            "payload": ev.payload,
                            "created_at": ev.created_at.isoformat(),
                        }
                    )
                    last = ev.seq

            now = asyncio.get_event_loop().time()
            if now - last_ping >= ping_interval:
                await websocket.send_json({"job_id": str(job_id), "seq": last, "type": "ping", "payload": {}})
                last_ping = now

            await asyncio.sleep(poll_interval)

    except WebSocketDisconnect:
        return
