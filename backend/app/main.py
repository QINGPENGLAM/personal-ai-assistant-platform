from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

from .agents import AgentPreparationError, AgentService
from .api.agents import router as agents_router
from .api.auth import router as auth_router
from .api.chat import router as chat_router
from .api.files import router as files_router
from .api.jobs import router as jobs_router
from .api.mcp import PlatformMCP
from .api.runs import router as runs_router
from .auth import AuthError, AuthService
from .codex_gateway import CodexGateway
from .config import Settings
from .containers import ContainerManager
from .coordination import Coordination, RuntimeProcessLock
from .db import Database, StorageWriteBlocked
from .event_store import EventStore
from .jobs import JobService
from .network_gateway import NetworkGateway
from .recovery import HistoryInspector, StartupRecovery, VerifiedModelReports
from .repository import Repository, RepositoryError
from .runner import Runner
from .scheduler import Scheduler
from .workspace import WorkspaceService

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    value = request.headers.get("x-request-id")
    if value is not None:
        try:
            return str(uuid.UUID(value))
        except ValueError:
            pass
    return str(uuid.uuid4())


class PlatformStorageMonitor:
    def __init__(self, database, measure, stop, repair):
        self.database, self.measure, self.stop, self.repair = database, measure, stop, repair
        self._task = None

    async def check(self):
        guard = self.database.storage
        try:
            sample = await self.measure()
        except Exception:
            sample = None
        guard.observe(sample)
        if not guard.latched:
            return
        try:
            if not await self.stop() or not guard.can_recover:
                return
            with guard.repair():
                await self.repair()
            # Repair itself allocates storage; never clear using an earlier measurement.
            guard.observe(await self.measure())
            if guard.can_recover:
                guard.clear_after_repair()
        except Exception:
            guard.trip("storage_recovery_failed")
            logger.warning("Platform storage recovery remains blocked.")

    def start(self):
        self._task = asyncio.create_task(self._loop(), name="platform-storage-monitor")

    async def _loop(self):
        while True:
            await self.check()
            await asyncio.sleep(1)

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None


def create_app(
    settings: Settings | None = None,
    *,
    database_path: Path | None = None,
    test_mode: bool = False,
    verified_models: tuple[str, ...] = (),
    frontend_dist: Path | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    if database_path is not None:
        resolved_settings = Settings(
            runtime_root=resolved_settings.runtime_root,
            database_path=database_path,
            allowed_origin=resolved_settings.allowed_origin,
            cookie_secure=resolved_settings.cookie_secure,
            platform_image=resolved_settings.platform_image,
            agent_image=resolved_settings.agent_image,
            auth_mode=resolved_settings.auth_mode,
            auth_file_source=resolved_settings.auth_file_source,
            toolkit_api_key=resolved_settings.toolkit_api_key,
            model_report_path=resolved_settings.model_report_path,
            gateway_socket=resolved_settings.gateway_socket,
            quota_helper_socket=resolved_settings.quota_helper_socket,
            agent_data_root=resolved_settings.agent_data_root,
            platform_state_root=resolved_settings.platform_state_root,
            auth_file_validation_path=resolved_settings.auth_file_validation_path,
            platform_volume=resolved_settings.platform_volume,
            agent_volume=resolved_settings.agent_volume,
        )

    platform_mcp = PlatformMCP()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        process_lock = RuntimeProcessLock(resolved_settings.platform_state_root / "backend.lock")
        process_lock.acquire()
        network_gateway: NetworkGateway | None = None
        runner: Runner | None = None
        model_refresh_task: asyncio.Task[None] | None = None
        recovery: StartupRecovery | None = None
        recovery_complete = False
        workspace = None
        scheduler = None
        agent_service = None
        storage_monitor = None
        database = None
        container_manager = None
        mcp_stack = AsyncExitStack()
        try:
            database = Database(resolved_settings.database_path)
            database.storage.enabled = not test_mode
            known_instance = None
            identity_cache = resolved_settings.platform_state_root / "instance-id"
            if not test_mode:
                # Read ownership identity before bootstrap writes can fail on an exhausted database.
                database.storage.trip("startup_verification")
                try:
                    existing = database.instance_id() if database.path.exists() else None
                except (sqlite3.Error, RuntimeError):
                    existing = None
                try:
                    known_instance = database.cached_instance_id(identity_cache)
                except FileNotFoundError:
                    known_instance = None
                except (OSError, RuntimeError, ValueError, UnicodeError):
                    if existing is not None:
                        container_manager = ContainerManager(resolved_settings, existing)
                    raise RuntimeError("Runtime identity cache requires repair.") from None
                if existing is not None:
                    if known_instance is not None and existing != known_instance:
                        raise RuntimeError("Runtime identity sources do not match.")
                    known_instance = existing
                if known_instance is not None:
                    container_manager = ContainerManager(resolved_settings, known_instance)
            database.initialize()
            instance_id = database.instance_id()
            if known_instance is not None and known_instance != instance_id:
                raise RuntimeError("Runtime identity changed during bootstrap.")
            container_manager = ContainerManager(resolved_settings, instance_id)
            repository = Repository(database)
            coordination = Coordination()
            runtime_settings = resolved_settings
            if not test_mode:
                await container_manager.verify_runtime()
                image_id = await container_manager.resolve_agent_image()
                runtime_settings = replace(resolved_settings, agent_image=image_id)
                container_manager = ContainerManager(runtime_settings, instance_id)
                network_gateway = NetworkGateway(runtime_settings, instance_id)
                await network_gateway.start()

            if not test_mode:
                database.storage.enabled = True
                with database.read() as connection:
                    if connection.execute(
                        "SELECT protection_latched FROM platform_storage_state WHERE id=1"
                    ).fetchone()[0]:
                        database.storage.trip("persisted_storage_protection")
                try:
                    database.storage.observe(await container_manager.platform_storage_usage())
                except Exception:
                    database.storage.observe(None)

            class TestAgentPreparer:
                def __init__(self):
                    self.running = set()

                async def prepare_agent(self, agent):
                    for folder in ("workspace", "codex", "credentials"):
                        (runtime_settings.agent_data_root / agent.id / folder).mkdir(
                            parents=True,
                            exist_ok=True,
                            mode=0o700,
                        )
                    container_manager._write_agent_configuration(agent)

                async def start_agent(self, agent_id):
                    self.running.add(agent_id)

                async def stop_agent(self, agent_id):
                    self.running.discard(agent_id)

                async def running_agent_ids(self):
                    return sorted(self.running)

                def reset_model_context(self, *args):
                    return container_manager.reset_model_context(*args)

                async def platform_storage_usage(self):
                    # Fake mode measures local DB bytes with simulated quota/free-space limits.
                    return {
                        "quota": {
                            "used_bytes": database._allocated_bytes(),
                            "hard_bytes": 2 * 1024**3,
                        },
                        "free_bytes": 4 * 1024**3,
                    }

            app.state.settings = runtime_settings
            app.state.instance_id = instance_id
            app.state.database = database
            app.state.auth = AuthService(database, resolved_settings)
            app.state.repository = repository
            model_reports = (
                None
                if test_mode
                else VerifiedModelReports.for_instance(
                    runtime_settings,
                    instance_id,
                    runtime_settings.agent_image,
                )
            )
            report_result = None if model_reports is None else model_reports.refresh()
            available_models = (
                frozenset(verified_models)
                if test_mode
                else report_result.models
                if report_result is not None
                else frozenset()
            )
            app.state.model_reports = model_reports
            app.state.model_report_reason = None if report_result is None else report_result.reason
            app.state.verified_models = available_models
            if not test_mode and not database.storage.status()["active"]:
                repository.update_model_availability(available_models)
                if report_result is not None and report_result.reason is not None:
                    logger.info(
                        "Model dispatch is disabled: report_status=%s",
                        report_result.reason,
                    )
            app.state.network_gateway = network_gateway
            app.state.container_manager = container_manager
            app.state.coordination = coordination
            app.state.event_store = EventStore(database)
            agent_service = AgentService(
                repository,
                TestAgentPreparer() if test_mode else container_manager,
            )
            app.state.agent_service = agent_service

            def check_model_admission() -> frozenset[str]:
                assert model_reports is not None
                if database.storage.status()["active"]:
                    return frozenset()
                refreshed = model_reports.refresh()
                app.state.verified_models = refreshed.models
                app.state.model_report_reason = refreshed.reason
                repository.update_model_availability(refreshed.models)
                return refreshed.models

            runner = Runner(
                repository,
                app.state.event_store,
                coordination,
                container_manager,
                network_gateway or NetworkGateway(resolved_settings, instance_id),
                CodexGateway(instance_id),
                enabled=not test_mode,
                verified_models=available_models,
                model_admission=None if test_mode else check_model_admission,
            )
            app.state.runner = runner
            if network_gateway is not None:
                recovery = StartupRecovery(
                    repository,
                    app.state.event_store,
                    container_manager,
                    network_gateway,
                    inspector=HistoryInspector(
                        repository,
                        app.state.event_store,
                        container_manager,
                        CodexGateway(instance_id),
                    ),
                )
                if not database.storage.status()["active"]:
                    await recovery.recover()
                    recovery_complete = True

            async def check_file_quota(agent_id):
                try:
                    return await container_manager._request_quota_helper(
                        {"operation": "verify_agent", "agent_id": agent_id},
                    )
                except AgentPreparationError as error:
                    raise RepositoryError(
                        503,
                        "quota_unavailable",
                        "File quota verification is unavailable.",
                    ) from error

            workspace = WorkspaceService(
                runtime_settings,
                repository,
                coordination,
                test_mode=test_mode,
                quota_check=None if test_mode else check_file_quota,
            )
            workspace.initialize()
            if not database.storage.status()["active"]:
                await workspace.recover_operations()
            app.state.workspace = workspace
            agent_service.runner = runner
            agent_service.coordination = coordination
            agent_service.workspace = workspace
            runner.capacity_admission = agent_service.make_room
            if not database.storage.status()["active"]:
                repository.interrupt_agent_controls()
            app.state.jobs = JobService(repository, coordination)
            scheduler = Scheduler(
                app.state.jobs,
                lambda agent_id: app.state.runner.enqueue(agent_id),
            )
            app.state.scheduler = scheduler
            await mcp_stack.enter_async_context(platform_mcp.manager.run())
            affected_agents = set()

            async def storage_stop():
                affected_agents.update(
                    context.run.agent_id for context in runner._active_runs.values()
                )
                runner.revoke_for_storage()
                worker_cleanup = asyncio.create_task(runner.storage_stop())
                try:
                    inventory = await container_manager.running_agent_ids()
                except Exception:
                    await worker_cleanup
                    return False
                affected_agents.update(inventory)

                async def stop_one(agent_id):
                    async with coordination.lifecycle_lock:
                        await container_manager.stop_agent(agent_id)
                    container_manager.remove_run_token(agent_id)
                    return True

                try:
                    async with asyncio.timeout(35):
                        results = await asyncio.gather(
                            worker_cleanup,
                            *(stop_one(agent) for agent in inventory),
                            return_exceptions=True,
                        )
                    return all(result is True for result in results)
                except TimeoutError:
                    return False

            async def storage_repair():
                nonlocal recovery_complete
                database.preserve_instance_id(identity_cache, instance_id)
                # This callback runs only after a fresh low-water measurement and physical cleanup.
                with database.transaction() as connection:
                    for agent_id in sorted(affected_agents):
                        connection.execute(
                            "UPDATE agents SET paused=1, generation=generation+1 WHERE id=?",
                            (agent_id,),
                        )
                if recovery is not None:
                    await recovery.recover(inspect_history=not recovery_complete)
                    recovery_complete = True
                await workspace.recover_operations()
                repository.interrupt_agent_controls()
                with database.transaction() as connection:
                    check = str(uuid.uuid4())
                    connection.execute(
                        "INSERT INTO runtime_metadata(key,value) "
                        "VALUES ('storage_repair_probe', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (check,),
                    )
                    if (
                        connection.execute(
                            "SELECT value FROM runtime_metadata WHERE key='storage_repair_probe'"
                        ).fetchone()[0]
                        != check
                    ):
                        raise RuntimeError("Storage repair probe failed.")
                    connection.execute(
                        "UPDATE platform_storage_state SET protection_latched=0 WHERE id=1"
                    )
                affected_agents.clear()

            if not test_mode:
                loop = asyncio.get_running_loop()
                database.storage.on_trip = lambda: loop.call_soon_threadsafe(
                    runner.revoke_for_storage
                )
                storage_monitor = PlatformStorageMonitor(
                    database,
                    container_manager.platform_storage_usage,
                    storage_stop,
                    storage_repair,
                )
                await storage_monitor.check()
                storage_monitor.start()
            app.state.storage = database.storage
            runner.start()
            scheduler.start()
            agent_service.start_idle_monitor()
            if model_reports is not None:

                async def refresh_models() -> None:
                    while True:
                        await asyncio.sleep(1)
                        previous_models = app.state.verified_models
                        previous_reason = app.state.model_report_reason
                        try:
                            refreshed = model_reports.refresh()
                        except Exception:
                            logger.exception("Could not refresh the verified-model report.")
                            refreshed_models = frozenset()
                            refreshed_reason = "refresh_failed"
                        else:
                            refreshed_models = refreshed.models
                            refreshed_reason = refreshed.reason
                        app.state.verified_models = refreshed_models
                        app.state.model_report_reason = refreshed_reason
                        runner.verified_models = refreshed_models
                        if not database.storage.status()["active"] and (
                            refreshed_models != previous_models
                            or refreshed_reason != previous_reason
                        ):
                            try:
                                repository.update_model_availability(refreshed_models)
                            except Exception:
                                logger.exception("Could not persist model availability blockers.")
                            else:
                                logger.info(
                                    "Verified model availability changed: count=%s reason=%s",
                                    len(refreshed_models),
                                    refreshed_reason or "valid",
                                )

                model_refresh_task = asyncio.create_task(
                    refresh_models(),
                    name="verified-model-report-refresh",
                )
            yield
        finally:
            # Bootstrap failure must still stop old owned containers.
            if (
                not test_mode
                and database is not None
                and container_manager is not None
                and database.storage.status()["active"]
            ):
                try:
                    async with asyncio.timeout(35):
                        inventory = await container_manager.running_agent_ids()
                        await asyncio.gather(
                            *(container_manager.stop_agent(agent) for agent in inventory)
                        )
                except Exception:
                    logger.warning("Emergency container cleanup remains unconfirmed.")
            if database is not None:
                database.storage.on_trip = None
            try:
                if agent_service is not None:
                    await agent_service.close()
            finally:
                try:
                    if storage_monitor is not None:
                        await storage_monitor.close()
                finally:
                    try:
                        if scheduler is not None:
                            await scheduler.close()
                    finally:
                        try:
                            await mcp_stack.aclose()
                        finally:
                            if model_refresh_task is not None:
                                model_refresh_task.cancel()
                                await asyncio.gather(model_refresh_task, return_exceptions=True)
                            try:
                                if runner is not None:
                                    await runner.close()
                                if recovery is not None and recovery_complete:
                                    if database.storage.status()["active"]:
                                        await storage_stop()
                                    else:
                                        await recovery.recover(inspect_history=False)
                            finally:
                                try:
                                    if network_gateway is not None:
                                        await network_gateway.close()
                                finally:
                                    try:
                                        if workspace is not None:
                                            workspace.close()
                                    finally:
                                        process_lock.release()
                            for name in (
                                "storage",
                                "jobs",
                                "scheduler",
                                "workspace",
                                "network_gateway",
                                "model_reports",
                                "model_report_reason",
                                "container_manager",
                                "coordination",
                                "event_store",
                                "runner",
                                "auth",
                                "database",
                                "repository",
                                "agent_service",
                            ):
                                if hasattr(app.state, name):
                                    setattr(app.state, name, None)

    app = FastAPI(title="Personal Assistant Platform", lifespan=lifespan)
    app.include_router(auth_router)
    app.include_router(agents_router)
    app.include_router(chat_router)
    app.include_router(runs_router)
    app.include_router(files_router)
    app.include_router(jobs_router)
    app.router.routes.append(Route("/mcp", platform_mcp, methods=["POST"]))

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "instance_id": app.state.instance_id,
            "storage_protection": app.state.database.storage.status(),
            "agent_image": app.state.settings.agent_image,
            "dispatch_capable": (
                bool(app.state.verified_models)
                and not app.state.database.storage.status()["active"]
            ),
        }

    @app.exception_handler(StorageWriteBlocked)
    async def storage_write_blocked(request: Request, error: StorageWriteBlocked):
        status = request.app.state.database.storage.status()
        if status["maintenance_active"] and not status["active"]:
            return JSONResponse(
                status_code=503,
                content={
                    "code": "maintenance_busy",
                    "message": "Database maintenance is active. Retry this write shortly.",
                    "request_id": _request_id(request),
                },
            )
        return JSONResponse(
            status_code=507,
            content={
                "code": "storage_protected",
                "message": "Platform storage needs repair. Saved data remains available.",
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(sqlite3.Error)
    async def database_unavailable(request: Request, error: sqlite3.Error):
        if not isinstance(error, sqlite3.IntegrityError):
            request.app.state.database.storage.trip("database_write_failed")
        return JSONResponse(
            status_code=503,
            content={
                "code": "database_unavailable",
                "message": "Platform storage is unavailable; inspect saved state before retrying.",
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(AuthError)
    async def handle_auth_error(request: Request, error: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "code": error.code,
                "message": error.message,
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(RepositoryError)
    async def handle_repository_error(request: Request, error: RepositoryError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "code": error.code,
                "message": error.message,
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(AgentPreparationError)
    async def handle_preparation_error(
        request: Request,
        error: AgentPreparationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "code": error.code,
                "message": error.message,
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "invalid_request",
                "message": "The request contains invalid values.",
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.error(
            "Unhandled request error request_id=%s",
            request_id,
            exc_info=(type(error), error, error.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "The request could not be completed.",
                "request_id": request_id,
            },
        )

    bundle = frontend_dist or Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if (bundle / "index.html").is_file():
        if (bundle / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=bundle / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(request: Request, path: str):
            if path in {"", "login", "register", "agents"} or path.startswith("agents/"):
                return FileResponse(bundle / "index.html", headers={"Cache-Control": "no-cache"})
            return JSONResponse(
                status_code=404,
                content={
                    "code": "not_found",
                    "message": "Not found.",
                    "request_id": _request_id(request),
                },
            )

    return app


app = create_app()
