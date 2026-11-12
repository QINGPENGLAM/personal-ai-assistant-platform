from __future__ import annotations

import json
from contextvars import ContextVar
from uuid import UUID, uuid4

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..jobs import Actor, JobInput
from ..network_gateway import GatewayError
from ..repository import RepositoryError
from ..schemas import StrictModel

bound_request: ContextVar = ContextVar("platform_mcp_request")


class CreateJob(JobInput):
    request_id: UUID


class CancelJob(StrictModel):
    request_id: UUID
    job_id: UUID


class ListJobs(StrictModel):
    cursor: str | None = Field(default=None, max_length=256)
    limit: int = Field(default=20, ge=1, le=50, strict=True)


class PlatformMCP:
    def __init__(self):
        self.server = Server("platform-scheduling")
        self.schemas = {"create_job": CreateJob, "list_jobs": ListJobs, "cancel_job": CancelJob}
        descriptions = {
            "create_job": (
                "Schedule one task, optionally recurring at a fixed UTC interval. "
                "Use an explicit numeric UTC offset."
            ),
            "list_jobs": "List this assistant's schedules, including user-created tasks.",
            "cancel_job": (
                "Cancel future and queued work. Already executing work requires a separate Stop."
            ),
        }

        @self.server.list_tools()
        async def list_tools():
            return [
                types.Tool(
                    name=name, description=descriptions[name], inputSchema=model.model_json_schema()
                )
                for name, model in self.schemas.items()
            ]

        @self.server.call_tool(validate_input=False)
        async def call_tool(name, arguments):
            request_id = str(uuid4())
            try:
                if name not in self.schemas:
                    raise RepositoryError(
                        422, "unknown_tool", "This scheduling tool is unavailable."
                    )
                parsed = self.schemas[name].model_validate(arguments)
                if hasattr(parsed, "request_id"):
                    request_id = str(parsed.request_id)
                app, raw, context, token_hash = bound_request.get()

                def validate(connection):
                    # Re-resolve the same token after lock acquisition; never read a new token file.
                    self.validate(
                        app, raw, context, token_hash, connection, writing=name != "list_jobs"
                    )

                actor = Actor("agent", context.agent_id, validate)
                jobs = app.state.jobs
                if name == "create_job":
                    result = await jobs.create(
                        actor,
                        context.agent_id,
                        request_id,
                        parsed.model_dump(exclude={"request_id"}),
                    )
                    result = {
                        key: result[key]
                        for key in (
                            "job_id",
                            "name",
                            "run_at",
                            "interval_seconds",
                            "enabled",
                            "created_by",
                        )
                    }
                elif name == "cancel_job":
                    result = await jobs.cancel(actor, str(parsed.job_id), request_id)
                else:
                    result = jobs.list(
                        actor, context.agent_id, cursor=parsed.cursor, limit=parsed.limit
                    )
                    keys = (
                        "job_id",
                        "name",
                        "next_run_at",
                        "interval_seconds",
                        "enabled",
                        "created_by",
                        "active_run_id",
                    )
                    result["items"] = [{key: item[key] for key in keys} for item in result["items"]]
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(result))]
                )
            except ValidationError:
                error = {
                    "code": "invalid_arguments",
                    "message": "Check the scheduling tool arguments.",
                    "request_id": request_id,
                }
            except RepositoryError as caught:
                error = {"code": caught.code, "message": caught.message, "request_id": request_id}
            except Exception:
                # Tool errors expose a stable repair path, never exception text or credentials.
                error = {
                    "code": "tool_unavailable",
                    "message": "Scheduling is temporarily unavailable.",
                    "request_id": request_id,
                }
            return types.CallToolResult(
                isError=True, content=[types.TextContent(type="text", text=json.dumps(error))]
            )

        self.manager = StreamableHTTPSessionManager(
            self.server,
            json_response=True,
            stateless=True,
            max_request_body_size=1024 * 1024,
        )

    @staticmethod
    def validate(app, raw, context, token_hash, connection, *, writing):
        try:
            gateway = app.state.network_gateway
            if gateway is None:
                raise RepositoryError(403, "tool_unauthorized", "The tool token is not authorized.")
            current_hash, current = gateway.authority.resolve(raw, "mcp")
            row = connection.execute(
                """SELECT runs.status, runs.run_token_hash, runs.agent_id, agents.generation
                   FROM runs JOIN agents ON agents.id=runs.agent_id WHERE runs.id=?""",
                (context.run_id,),
            ).fetchone()
            if (
                current != context
                or current_hash != token_hash
                or row is None
                or row["agent_id"] != context.agent_id
                or row["generation"] != context.generation
                or row["status"] not in (("running",) if writing else ("starting", "running"))
                or (row["status"] == "running" and row["run_token_hash"] != token_hash)
            ):
                raise RepositoryError(
                    403, "tool_unauthorized", "The run no longer authorizes this tool."
                )
        except GatewayError:
            raise RepositoryError(
                403, "tool_unauthorized", "The tool token is revoked or expired."
            ) from None

    async def __call__(self, scope, receive, send):
        request = Request(scope)
        app = scope["app"]
        raw = request.headers.get("authorization", "")
        try:
            if not raw.startswith("Bearer ") or app.state.network_gateway is None:
                raise RepositoryError(403, "tool_unauthorized", "A current run token is required.")
            raw = raw.removeprefix("Bearer ")
            token_hash, context = app.state.network_gateway.authority.resolve(raw, "mcp")
            with app.state.repository.database.read() as connection:
                self.validate(app, raw, context, token_hash, connection, writing=False)
        except (RepositoryError, GatewayError):
            response = JSONResponse(
                {"code": "tool_unauthorized", "message": "A current run token is required."},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        token = bound_request.set((app, raw, context, token_hash))
        try:
            await self.manager.handle_request(scope, receive, send)
        finally:
            bound_request.reset(token)
