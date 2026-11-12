from uuid import UUID

from fastapi import APIRouter, Request

from ..jobs import Actor, JobInput
from .agents import _current_user
from .auth import _request_id

router = APIRouter(prefix="/api", tags=["scheduled tasks"])


def write_actor(request):
    user = _current_user(request)
    request.app.state.auth.require_origin(request.headers.get("origin"))
    request.app.state.auth.require_csrf(user, request.headers.get("x-csrf-token"))
    return Actor("user", user.id), _request_id(request)


@router.get("/agents/{agent_id}/jobs")
def list_jobs(agent_id: UUID, request: Request, cursor: str | None = None, limit: int = 20):
    return request.app.state.jobs.list(
        Actor("user", _current_user(request).id), str(agent_id), cursor=cursor, limit=limit
    )


@router.post("/agents/{agent_id}/jobs", status_code=201)
async def create_job(agent_id: UUID, request: Request, body: JobInput):
    actor, request_id = write_actor(request)
    return await request.app.state.jobs.create(actor, str(agent_id), request_id, body.model_dump())


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: UUID, request: Request):
    actor, request_id = write_actor(request)
    return await request.app.state.jobs.cancel(actor, str(job_id), request_id)


@router.get("/jobs/{job_id}/runs")
def job_runs(job_id: UUID, request: Request, cursor: str | None = None, limit: int = 50):
    return request.app.state.jobs.runs(
        Actor("user", _current_user(request).id), str(job_id), cursor=cursor, limit=limit
    )
