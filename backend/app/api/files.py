from __future__ import annotations

from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Request
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse

from ..repository import RepositoryError
from .agents import _current_user
from .auth import _request_id

router = APIRouter(prefix="/api/agents/{agent_id}/files", tags=["workspace"])


def mutation(request):
    user = _current_user(request)
    auth = request.app.state.auth
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    return user, _request_id(request)


@router.get("")
async def listing(agent_id: UUID, request: Request, path: str = ""):
    user = _current_user(request)
    return await request.app.state.workspace.list_files(user.id, str(agent_id), path)


@router.get("/metadata")
async def metadata(agent_id: UUID, request: Request, path: str):
    user = _current_user(request)
    return await request.app.state.workspace.metadata(user.id, str(agent_id), path)


@router.get("/preview")
async def preview(agent_id: UUID, request: Request, path: str):
    user = _current_user(request)
    return await request.app.state.workspace.preview(user.id, str(agent_id), path)


class FileStream(StreamingResponse):
    def __init__(self, download):
        self.download = download
        super().__init__(
            download.chunks(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": (
                    "attachment; filename*=UTF-8''" + quote(download.name, safe="")
                ),
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
            },
        )

    async def __call__(self, scope, receive, send):
        # Close the opened descriptor even when the client disconnects before iteration begins.
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.download.close()


@router.get("/download")
async def download(agent_id: UUID, request: Request, path: str):
    user = _current_user(request)
    opened = await request.app.state.workspace.download(user.id, str(agent_id), path)
    return FileStream(opened)


@router.post("")
async def upload(
    agent_id: UUID,
    request: Request,
    path: str,
    overwrite: bool = False,
    expected_sha256: str | None = None,
):
    user, request_id = mutation(request)
    try:
        return await request.app.state.workspace.upload(
            user.id,
            str(agent_id),
            request_id,
            path,
            request,
            overwrite=overwrite,
            expected_sha256=expected_sha256,
        )
    except ClientDisconnect:
        raise RepositoryError(400, "upload_disconnected", "The upload connection closed.") from None


@router.delete("")
async def delete(agent_id: UUID, request: Request, path: str, expected_sha256: str):
    user, request_id = mutation(request)
    return await request.app.state.workspace.delete(
        user.id,
        str(agent_id),
        request_id,
        path,
        expected_sha256,
    )
