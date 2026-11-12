from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response, status

from ..api.auth import COOKIE_NAME, _request_id
from ..auth import AuthService
from ..ownership import require_owned_agent
from ..repository import Repository
from ..runner import Runner
from ..schemas import (
    ChatMessageAccepted,
    ChatMessageRequest,
    ChatRunResponse,
)

router = APIRouter(prefix="/api", tags=["chat"])


def _current_user(request: Request):
    auth: AuthService = request.app.state.auth
    return auth.authenticate(request.cookies.get(COOKIE_NAME))


def _serialize_run(run) -> ChatRunResponse:
    return ChatRunResponse(
        source=run.source,
        job_id=run.job_id,
        scheduled_for=(
            datetime.fromtimestamp(run.scheduled_for / 1000, tz=UTC).isoformat()
            if run.scheduled_for is not None else None
        ),
        coalesced_count=run.coalesced_count,
        id=run.id,
        agent_id=run.agent_id,
        retry_of=run.retry_of,
        prompt=run.prompt,
        status=run.status,
        model=run.model,
        chat_epoch=run.chat_epoch,
        created_at=datetime.fromtimestamp(run.created_at / 1_000, tz=UTC).isoformat(),
        started_at=(
            datetime.fromtimestamp(run.started_at / 1_000, tz=UTC).isoformat()
            if run.started_at is not None
            else None
        ),
        finished_at=(
            datetime.fromtimestamp(run.finished_at / 1_000, tz=UTC).isoformat()
            if run.finished_at is not None
            else None
        ),
        termination_reason=run.termination_reason,
        cleanup_state=run.cleanup_state,
        error_code=run.error_code,
        error_message=run.error_message,
        paused=run.paused,
    )


@router.post(
    "/agents/{agent_id}/messages",
    response_model=ChatMessageAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def send_message(
    agent_id: UUID,
    request: Request,
    response: Response,
    body: ChatMessageRequest,
) -> ChatMessageAccepted:
    user = _current_user(request)
    auth: AuthService = request.app.state.auth
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    request_id = _request_id(request)

    repository: Repository = request.app.state.repository
    agent = require_owned_agent(repository, user.id, str(agent_id))
    run, replayed = repository.create_chat_run(
        user.id,
        agent.id,
        request_id,
        body.prompt,
    )
    runner: Runner = request.app.state.runner
    runner.enqueue(agent.id)
    if replayed:
        response.status_code = status.HTTP_200_OK
    return ChatMessageAccepted(
        run_id=run.id,
        status=run.status,
        queued=run.status == "queued",
        paused=run.paused,
        replayed=replayed,
    )


@router.get(
    "/agents/{agent_id}/chat",
    response_model=list[ChatRunResponse],
)
def list_chat(
    agent_id: UUID,
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
) -> list[ChatRunResponse]:
    user = _current_user(request)
    repository: Repository = request.app.state.repository
    require_owned_agent(repository, user.id, str(agent_id))
    runs = repository.list_owned_chat_runs(user.id, str(agent_id), limit=limit)
    return [_serialize_run(run) for run in runs]
