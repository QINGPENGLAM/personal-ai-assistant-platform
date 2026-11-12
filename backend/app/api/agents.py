from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Request, Response, status

from ..agents import AgentService
from ..api.auth import COOKIE_NAME, _request_id
from ..auth import AuthError, AuthService
from ..ownership import require_owned_agent
from ..repository import Repository
from ..schemas import AgentCreateRequest, AgentResponse, NewChatAccepted, NewChatRequest

router = APIRouter(prefix="/api", tags=["assistants"])


def _current_user(request: Request):
    auth: AuthService = request.app.state.auth
    return auth.authenticate(request.cookies.get(COOKIE_NAME))


def _serialize_agent(agent, repository=None) -> AgentResponse:
    admission = None if repository is None else repository.run_start_usage()
    if admission is not None and admission["retry_at"] is not None:
        admission["retry_at"] = datetime.fromtimestamp(
            admission["retry_at"] / 1000,
            tz=UTC,
        ).isoformat()
    return AgentResponse(
        run_admission=admission,
        storage_protection=None if repository is None else repository.database.storage.status(),
        id=agent.id,
        name=agent.name,
        model=agent.model,
        status=agent.status,
        paused=agent.paused,
        chat_epoch=agent.chat_epoch,
        blockers=list(agent.blockers),
        container_state=agent.container_state,
        created_at=datetime.fromtimestamp(agent.created_at / 1_000, tz=UTC).isoformat(),
    )


@router.get("/models")
def list_models(request: Request) -> dict[str, list[str]]:
    _current_user(request)
    return {"models": sorted(request.app.state.verified_models)}


@router.get("/agents", response_model=list[AgentResponse])
def list_agents(request: Request) -> list[AgentResponse]:
    user = _current_user(request)
    repository: Repository = request.app.state.repository
    return [_serialize_agent(agent) for agent in repository.list_agents(user.id)]


@router.post(
    "/agents",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent(
    request: Request,
    response: Response,
    body: AgentCreateRequest,
) -> AgentResponse:
    user = _current_user(request)
    auth: AuthService = request.app.state.auth
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    request_id = _request_id(request)

    if body.model not in request.app.state.verified_models:
        raise AuthError(
            422,
            "model_unavailable",
            "The selected model has not passed the current runtime check.",
        )

    service: AgentService = request.app.state.agent_service
    agent, replayed = await service.create(
        user.id,
        body.name,
        body.model,
        request_id,
    )
    if replayed:
        response.status_code = status.HTTP_200_OK
    return _serialize_agent(agent)


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: UUID, request: Request) -> AgentResponse:
    user = _current_user(request)
    repository: Repository = request.app.state.repository
    agent = require_owned_agent(repository, user.id, str(agent_id))
    result = _serialize_agent(agent, repository)
    result.container_state = await request.app.state.agent_service.container_state(str(agent_id))
    return result


@router.post("/agents/{agent_id}/resume", response_model=AgentResponse)
@router.post("/agents/{agent_id}/continue", response_model=AgentResponse)
async def continue_agent(agent_id: UUID, request: Request) -> AgentResponse:
    user = _current_user(request)
    auth: AuthService = request.app.state.auth
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    repository: Repository = request.app.state.repository
    agent = repository.continue_agent(
        user.id,
        str(agent_id),
        verified_models=request.app.state.verified_models,
        request_id=_request_id(request),
    )
    request.app.state.runner.enqueue(agent.id)
    return _serialize_agent(agent)


@router.post("/agents/{agent_id}/prepare", response_model=AgentResponse)
async def prepare_agent(agent_id: UUID, request: Request) -> AgentResponse:
    user = _current_user(request)
    auth: AuthService = request.app.state.auth
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    _request_id(request)
    service: AgentService = request.app.state.agent_service
    return _serialize_agent(await service.prepare(user.id, str(agent_id)))


@router.post("/agents/{agent_id}/new-chat", response_model=NewChatAccepted)
async def new_chat(
    agent_id: UUID,
    request: Request,
    body: NewChatRequest,
) -> NewChatAccepted:
    user = _current_user(request)
    auth: AuthService = request.app.state.auth
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    request_id = _request_id(request)
    repository: Repository = request.app.state.repository
    async with request.app.state.coordination.agent_locks([str(agent_id)]):
        agent, replayed = repository.new_chat(user.id, str(agent_id), request_id)
    return NewChatAccepted(
        operation_id=request_id,
        replayed=replayed,
        agent=_serialize_agent(agent),
    )


async def _lifecycle(request, agent_id, action, confirmed=False):
    user = _current_user(request)
    request.app.state.auth.require_origin(request.headers.get("origin"))
    request.app.state.auth.require_csrf(user, request.headers.get("x-csrf-token"))
    request_id = _request_id(request)
    agent, replayed = await request.app.state.agent_service.control(
        user.id,
        str(agent_id),
        action,
        request_id,
        confirmed=confirmed,
    )
    return NewChatAccepted(
        operation_id=request_id, replayed=replayed, agent=_serialize_agent(agent)
    )


@router.post("/agents/{agent_id}/wake", response_model=NewChatAccepted)
async def wake_agent(agent_id: UUID, request: Request):
    return await _lifecycle(request, agent_id, "wake")


@router.post("/agents/{agent_id}/sleep", response_model=NewChatAccepted)
async def sleep_agent(agent_id: UUID, request: Request):
    return await _lifecycle(request, agent_id, "sleep")


@router.post("/agents/{agent_id}/restart", response_model=NewChatAccepted)
async def restart_agent(agent_id: UUID, request: Request):
    return await _lifecycle(request, agent_id, "restart")


@router.post("/agents/{agent_id}/reset-model-context", response_model=NewChatAccepted)
async def reset_model_context(agent_id: UUID, request: Request, body: NewChatRequest):
    return await _lifecycle(request, agent_id, "reset-model-context", body.confirmed)
