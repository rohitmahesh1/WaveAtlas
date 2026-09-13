from __future__ import annotations

from sqlmodel import Session

from ..job_store import JobStore
from ..pipeline import run_job


def schedule_job(
    *,
    store,
    job,
    config,
    settings,
    artifacts,
    background,
    engine,
    desktop=None,
    resume=False,
):
    """Claim and schedule once; local admission is held through the whole worker."""
    if desktop:
        if resume:
            desktop.validate_resume(job, store, artifacts)
        config = desktop.constrain_config(config)
        desktop.reserve(job.id)
    try:
        claim = store.claim_resume if resume else store.claim_start
        job, claimed = claim(job.id, config=config)
        if not claimed:
            if desktop:
                desktop.release()
            return job
    except BaseException:
        if desktop:
            desktop.release()
        raise

    def run():
        try:
            with Session(engine) as session:
                run_job(
                    job.id,
                    job_store=JobStore(session),
                    artifact_store=artifacts,
                    config=config,
                    settings=settings,
                    resume=resume,
                )
        except Exception as exc:
            # run_job normally records its own failures. Cover failures before
            # its error handler as well, so a live app never strands active rows.
            from ..models import JobStatus

            with Session(engine) as session:
                failed_store = JobStore(session)
                current = failed_store.get_job(job.id)
                if current.status in (
                    JobStatus.in_progress,
                    JobStatus.cancel_requested,
                ):
                    failed_store.set_status(
                        job.id,
                        JobStatus.failed,
                        error=str(exc),
                        error_code="worker_error",
                    )
            raise
        finally:
            if desktop:
                desktop.release()

    background.add_task(run)
    return job
