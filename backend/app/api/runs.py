from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime
from functools import partial
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agents import AgentPreparationError
from ..api.auth import COOKIE_NAME, _request_id
from ..auth import AuthService
from ..db import StorageMaintenanceError, StorageWriteBlocked
from ..event_store import EventStore, EventStoreError
from ..repository import Repository, RepositoryError
from ..schemas import RetryAccepted, RunDetailsResponse, RunEventResponse

router = APIRouter(prefix="/api", tags=["runs"])
TERMINAL_RUN_STATES = frozenset(
    {"completed", "completed_with_errors", "failed", "interrupted", "cancelled"}
)


def _current_user(request: Request):
    auth: AuthService = request.app.state.auth
    return auth.authenticate(request.cookies.get(COOKIE_NAME))


def _serialize_run(run, last_event_sequence: int) -> RunDetailsResponse:
    return RunDetailsResponse(
        source=run.source,
        job_id=run.job_id,
        scheduled_for=(
            datetime.fromtimestamp(run.scheduled_for / 1000, tz=UTC).isoformat()
            if run.scheduled_for is not None
            else None
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
        last_event_sequence=last_event_sequence,
    )


def _serialize_event(event) -> RunEventResponse:
    return RunEventResponse(
        sequence=event.sequence,
        kind=event.kind,
        sdk_item_id=event.sdk_item_id,
        payload=event.payload,
        created_at=datetime.fromtimestamp(event.created_at / 1_000, tz=UTC).isoformat(),
    )


@router.get("/runs/{run_id}", response_model=RunDetailsResponse)
def get_run(run_id: UUID, request: Request) -> RunDetailsResponse:
    user = _current_user(request)
    repository: Repository = request.app.state.repository
    run, last_event_sequence = repository.get_owned_run(user.id, str(run_id))
    return _serialize_run(run, last_event_sequence)


@router.post("/runs/{run_id}/stop", response_model=RunDetailsResponse)
async def stop_run(run_id: UUID, request: Request) -> RunDetailsResponse:
    user = _current_user(request)
    auth: AuthService = request.app.state.auth
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    runner = request.app.state.runner
    await runner.stop_run(user.id, str(run_id))
    repository: Repository = request.app.state.repository
    run, last_event_sequence = repository.get_owned_run(user.id, str(run_id))
    return _serialize_run(run, last_event_sequence)


@router.post(
    "/runs/{run_id}/retry",
    response_model=RetryAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_run(run_id: UUID, request: Request, response: Response) -> RetryAccepted:
    user = _current_user(request)
    auth: AuthService = request.app.state.auth
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    request_id = _request_id(request)
    repository: Repository = request.app.state.repository
    original, _ = repository.get_owned_run(user.id, str(run_id))
    async with request.app.state.coordination.agent_locks([original.agent_id]):
        run, replayed, context_warning = repository.retry_chat_run(
            user.id,
            str(run_id),
            request_id,
        )
    request.app.state.runner.enqueue(run.agent_id)
    if replayed:
        response.status_code = status.HTTP_200_OK
    return RetryAccepted(
        run_id=run.id,
        status=run.status,
        queued=run.status == "queued",
        paused=run.paused,
        replayed=replayed,
        context_warning=context_warning,
    )


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: UUID,
    request: Request,
    after: int = Query(default=0),
) -> StreamingResponse:
    user = _current_user(request)
    repository: Repository = request.app.state.repository
    run, last_event_sequence = repository.get_owned_run(user.id, str(run_id))
    if after < 0:
        raise RepositoryError(409, "invalid_event_cursor", "The event cursor must be nonnegative.")
    if after > last_event_sequence:
        raise RepositoryError(409, "invalid_event_cursor", "The event cursor is ahead of this run.")
    event_store: EventStore = request.app.state.event_store

    async def stream():
        nonlocal run, last_event_sequence
        cursor = after
        last_heartbeat = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            try:
                events = event_store.list_after(str(run_id), after=cursor, limit=100)
            except EventStoreError as error:
                raise RepositoryError(422, "invalid_event_cursor", str(error)) from error
            for event in events:
                cursor = event.sequence
                data = _serialize_event(event).model_dump_json()
                yield f"id: {event.sequence}\nevent: {event.kind}\ndata: {data}\n\n"
            run, last_event_sequence = repository.get_owned_run(user.id, str(run_id))
            if run.status in TERMINAL_RUN_STATES and cursor >= last_event_sequence:
                return
            now = time.monotonic()
            if now - last_heartbeat >= 15:
                yield ": heartbeat\n\n"
                last_heartbeat = now
            await asyncio.sleep(0.25)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agents/{agent_id}/runs")
def list_agent_runs(
    agent_id: UUID,
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    archive_roots: bool = False,
    cursor: UUID | None = None,
):
    user = _current_user(request)
    repository: Repository = request.app.state.repository
    repository.get_owned_agent(user.id, str(agent_id))
    if archive_roots:
        runs, next_cursor = repository.list_archive_roots(
            user.id,
            str(agent_id),
            limit=limit,
            cursor=str(cursor) if cursor else None,
        )
        return {
            "items": [_serialize_run(*repository.get_owned_run(user.id, run.id)) for run in runs],
            "next_cursor": next_cursor,
        }
    runs = repository.list_owned_chat_runs(user.id, str(agent_id), limit=limit)
    return [_serialize_run(*repository.get_owned_run(user.id, run.id)) for run in runs]


class ArchiveExportRequest(BaseModel):
    run_ids: list[UUID] = Field(min_length=1, max_length=100)


class ArchiveDownload(StreamingResponse):
    """Release the unlinked staging file and capacity even on disconnect or backpressure."""

    def __init__(self, artifact, slot, digest, size):
        self.artifact, self.slot = artifact, slot

        async def chunks():
            while data := artifact.read(64 * 1024):
                yield data

        super().__init__(
            chunks(),
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="run-history.json"',
                "Content-Length": str(size),
                "X-Archive-SHA256": digest,
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def __call__(self, scope, receive, send):
        try:
            async with asyncio.timeout(60):
                await super().__call__(scope, receive, send)
        finally:
            self.artifact.close()
            self.slot.release()


@router.post("/runs/archive/export")
async def export_run_archive(request: Request, body: ArchiveExportRequest):
    user = _current_user(request)
    auth = request.app.state.auth
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    slot = await _archive_slot(request)
    workspace = request.app.state.workspace
    cancelled = threading.Event()

    def prepare():
        name = f"archive-{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=workspace._staging_fd,
        )
        try:
            os.unlink(name, dir_fd=workspace._staging_fd)
            artifact = os.fdopen(descriptor, "w+b")
        except BaseException:
            os.close(descriptor)
            raise
        try:
            digest, size = request.app.state.repository.export_run_archive(
                user.id,
                [str(value) for value in body.run_ids],
                artifact,
                cancelled=cancelled,
            )
            return artifact, digest, size
        except BaseException:
            artifact.close()
            raise

    try:
        artifact, digest, size = await workspace.io(
            prepare,
            on_cancel=cancelled.set,
            dispose_cancelled_result=lambda result: result[0].close(),
        )
        return ArchiveDownload(artifact, slot, digest, size)
    except BaseException as error:
        slot.release()
        if isinstance(error, OSError):
            raise RepositoryError(
                507,
                "archive_staging_full",
                "Temporary export space is unavailable; try a smaller selection.",
            ) from None
        if isinstance(error, sqlite3.OperationalError) and str(error) == "interrupted":
            raise RepositoryError(
                408, "archive_timeout", "History export was interrupted."
            ) from None
        raise


async def _archive_slot(request):
    # One archive operation at a time; do not queue requests holding large selected artifacts.
    if not hasattr(request.app.state, "archive_slot"):
        request.app.state.archive_slot = asyncio.Semaphore(1)
    slot = request.app.state.archive_slot
    if slot.locked():
        raise RepositoryError(429, "archive_busy", "Another archive operation is in progress.")
    await slot.acquire()
    return slot


class ArchiveCheckRequest(ArchiveExportRequest):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


ARCHIVE_CHECK_TIMEOUT_SECONDS = 35


@router.post("/runs/archive/check")
async def check_archive_purge(request: Request, body: ArchiveCheckRequest, response: Response):
    response.headers["Cache-Control"] = "no-store"
    user = _current_user(request)
    request.app.state.auth.require_origin(request.headers.get("origin"))
    request.app.state.auth.require_csrf(user, request.headers.get("x-csrf-token"))
    slot = await _archive_slot(request)
    cancelled = threading.Event()

    async def work():
        def inspect():
            return request.app.state.repository.check_archive_purge(
                user.id,
                [str(value) for value in body.run_ids],
                body.sha256,
                cancelled=cancelled,
            )

        plan = await request.app.state.workspace.io(inspect, on_cancel=cancelled.set)
        blockers = set(plan["blockers"])
        guard = request.app.state.repository.database.storage
        if guard.status()["active"]:
            blockers.update({"storage_repair_required", "container_state_unknown"})
        else:
            # Inventory once, regardless of the number of selected assistants.
            try:
                async with asyncio.timeout(5):
                    running = set(
                        await request.app.state.agent_service.preparer.running_agent_ids()
                    )
                if running:
                    blockers.add("sleep_required")
            except (AgentPreparationError, TimeoutError):
                blockers.add("container_state_unknown")
        if guard.status()["active"]:
            blockers.add("storage_repair_required")
        try:
            await request.app.state.workspace.io(
                request.app.state.repository.require_quiescent_archive_maintenance,
            )
        except RepositoryError:
            blockers.add("maintenance_busy")
        if guard.status()["maintenance_active"]:
            blockers.add("maintenance_busy")
        if not blockers:
            try:
                async with asyncio.timeout(5):
                    sample = await request.app.state.agent_service.preparer.platform_storage_usage()
                if guard.enabled:
                    guard.observe(sample)
                await request.app.state.workspace.io(
                    partial(
                        request.app.state.repository.database.reclaim_space,
                        sample,
                        cancelled=cancelled,
                        _check_only=True,
                    ),
                )
            except (AgentPreparationError, TimeoutError, OSError) as error:
                if guard.enabled:
                    guard.observe(None)
                blockers.add(getattr(error, "code", "quota_measurement_unavailable"))
            except StorageMaintenanceError as error:
                blockers.add(error.code)
            except StorageWriteBlocked:
                blockers.add(
                    "storage_repair_required" if guard.status()["active"] else "maintenance_busy"
                )
        return {
            **plan,
            "eligible": not blockers,
            "blockers": sorted(blockers),
            "deletion_available": not blockers,
        }

    try:
        return await _archive_request(request, work, cancelled, ARCHIVE_CHECK_TIMEOUT_SECONDS)
    finally:
        slot.release()


async def _archive_request(request, work, cancelled, timeout):
    disconnected = False
    worker = asyncio.create_task(work(), name="archive-operation")

    async def watch_disconnect():
        nonlocal disconnected
        while True:
            if await request.is_disconnected():
                disconnected = True
                cancelled.set()
                worker.cancel()
                return
            await asyncio.sleep(0.1)

    watcher = asyncio.create_task(watch_disconnect(), name="archive-disconnect")
    try:
        async with asyncio.timeout(timeout):
            return await worker
    except asyncio.CancelledError:
        if disconnected:
            raise RepositoryError(
                408, "archive_cancelled", "History check was interrupted."
            ) from None
        raise
    except TimeoutError:
        raise RepositoryError(408, "archive_timeout", "History check was interrupted.") from None
    except sqlite3.OperationalError as error:
        if str(error) == "interrupted":
            raise RepositoryError(
                408, "archive_timeout", "History check was interrupted."
            ) from None
        raise
    finally:
        cancelled.set()
        worker.cancel()
        watcher.cancel()
        await asyncio.gather(worker, watcher, return_exceptions=True)


class ArchivePurgeRequest(ArchiveCheckRequest):
    confirmed: bool = Field(strict=True)


@router.post("/runs/archive/purge")
async def purge_run_archive(request: Request, body: ArchivePurgeRequest, response: Response):
    response.headers["Cache-Control"] = "no-store"
    user = _current_user(request)
    request.app.state.auth.require_origin(request.headers.get("origin"))
    request.app.state.auth.require_csrf(user, request.headers.get("x-csrf-token"))
    request_id = _request_id(request)
    if body.confirmed is not True:
        raise RepositoryError(422, "confirmation_required", "Confirm permanent history deletion.")
    slot = await _archive_slot(request)
    cancelled = threading.Event()

    async def work():
        return await request.app.state.agent_service.purge_archive(
            user.id,
            [str(value) for value in body.run_ids],
            body.sha256,
            request_id,
            confirmed=True,
        )

    try:
        return await _archive_request(request, work, cancelled, 95)
    except StorageMaintenanceError as error:
        status = (
            507 if error.code in {"maintenance_space_required", "maintenance_measurement"} else 409
        )
        raise RepositoryError(status, error.code, str(error)) from None
    finally:
        slot.release()
