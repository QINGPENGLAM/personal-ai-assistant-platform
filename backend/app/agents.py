from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
import time
from contextvars import copy_context
from functools import partial
from typing import Protocol
from uuid import UUID

from .db import StorageMaintenanceError, StorageWriteBlocked
from .repository import AgentRecord, Repository, RepositoryError

logger = logging.getLogger(__name__)


class AgentPreparationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AgentPreparer(Protocol):
    async def prepare_agent(self, agent: AgentRecord) -> None: ...


class AgentService:
    def __init__(
        self,
        repository: Repository,
        preparer: AgentPreparer,
        *,
        runner=None,
        coordination=None,
        workspace=None,
        clock=None,
    ) -> None:
        self.repository = repository
        self.preparer = preparer
        self.runner, self.coordination, self.workspace = runner, coordination, workspace
        self.clock = clock or time.monotonic
        self._idle_since = {}
        self._idle_task = None

    async def create(
        self,
        owner_id: str,
        name: str,
        model: str,
        request_id: str,
    ) -> tuple[AgentRecord, bool]:
        agent, replayed = self.repository.create_agent(owner_id, name, model, request_id)
        if replayed:
            return agent, True
        await self._prepare(agent)
        self.repository.complete_agent_creation(owner_id, request_id, agent.id)
        return self.repository.get_owned_agent(owner_id, agent.id), False

    async def prepare(self, owner_id: str, agent_id: str) -> AgentRecord:
        agent = self.repository.get_owned_agent(owner_id, agent_id)
        if agent.status == "ready":
            return agent
        self.repository.set_agent_status(agent_id, "provisioning")
        await self._prepare(agent)
        return self.repository.get_owned_agent(owner_id, agent_id)

    async def _prepare(self, agent: AgentRecord) -> None:
        try:
            await self.preparer.prepare_agent(agent)
        except AgentPreparationError as error:
            self.repository.fail_agent_creation(agent.id, error.code)
            raise
        self.repository.set_agent_status(agent.id, "ready")

    async def control(self, owner, agent_id, action, request_id, *, confirmed=False):
        if action not in {"wake", "sleep", "restart", "reset-model-context"}:
            raise RepositoryError(404, "not_found", "Assistant control not found.")
        self.repository.get_owned_agent(owner, agent_id)
        if self.runner is None or self.coordination is None:
            raise RepositoryError(503, "control_unavailable", "Assistant controls are unavailable.")
        if action == "reset-model-context" and confirmed is not True:
            raise RepositoryError(
                422, "confirmation_required", "Confirm deletion of model history."
            )
        request_id = str(UUID(request_id))
        body_hash = hashlib.sha256(
            json.dumps({"confirmed": confirmed}, sort_keys=True).encode()
        ).hexdigest()
        replayed = self.repository.agent_control_receipt(
            owner, agent_id, action, request_id, body_hash
        )
        if replayed:
            return self.repository.get_owned_agent(owner, agent_id), True
        if action == "reset-model-context":
            async with self.coordination.lifecycle_lock:
                async with self.coordination.agent_locks([agent_id]):
                    if agent_id in await self.preparer.running_agent_ids():
                        raise RepositoryError(
                            409,
                            "sleep_required",
                            "Sleep the assistant before resetting model history.",
                        )
                    replayed = self.repository.reserve_agent_control(
                        owner, agent_id, action, request_id, body_hash
                    )
        else:
            async with self.coordination.agent_locks([agent_id]):
                replayed = self.repository.reserve_agent_control(
                    owner, agent_id, action, request_id, body_hash
                )
        if replayed:
            return self.repository.get_owned_agent(owner, agent_id), True
        try:
            if action in {"sleep", "restart"}:
                await self.runner.repair_agent(owner, agent_id, allow_repair=action == "restart")
                if self.runner.containers is not self.preparer:
                    await self.preparer.stop_agent(agent_id)
                self._idle_since.pop(agent_id, None)
            if action == "restart":
                await self.preparer.prepare_agent(self.repository.get_owned_agent(owner, agent_id))
                self.repository.set_recovery_blocker(agent_id, blocked=False)
            if action in {"wake", "restart"}:
                await self._wake_idle(owner, agent_id)
            if action == "reset-model-context":
                cancelled = threading.Event()
                async with self.coordination.lifecycle_lock:
                    async with self.coordination.agent_locks([agent_id]):
                        await self.preparer.stop_agent(agent_id)
                        agent = self.repository.get_owned_agent(owner, agent_id)
                        await self.workspace.io(
                            self.preparer.reset_model_context,
                            agent,
                            time.monotonic() + 30,
                            cancelled,
                            self.repository.database.storage.require_writable,
                            on_cancel=cancelled.set,
                        )
                try:
                    quota = await self.workspace._check_quota(agent_id)
                    if (
                        quota is not None
                        and quota["used_bytes"] < quota["hard_bytes"]
                        and quota["used_inodes"] < quota["hard_inodes"]
                    ):
                        self.repository.set_agent_storage_blocker(agent_id, blocked=False)
                except (RepositoryError, AgentPreparationError):
                    logger.warning("Context reset completed without verified quota headroom.")
            self.repository.finish_agent_control(
                owner, agent_id, action, request_id, completed=True
            )
            return self.repository.get_owned_agent(owner, agent_id), False
        except BaseException as error:
            if action == "restart":
                self.repository.set_recovery_blocker(agent_id, blocked=True)
            self.repository.finish_agent_control(
                owner, agent_id, action, request_id, completed=False
            )
            if isinstance(
                error,
                (
                    asyncio.CancelledError,
                    RepositoryError,
                    AgentPreparationError,
                    StorageWriteBlocked,
                ),
            ):
                raise
            logger.warning("Assistant control failed; inspect state before a new request.")
            raise RepositoryError(
                503,
                "control_failed",
                "The control failed. Inspect the assistant before a new request.",
            ) from None

    async def make_room(self, agent_id):
        """Reserve starting runs and idle containers under the caller's lifecycle lock."""
        running = set(await self.preparer.running_agent_ids())
        with self.repository.database.read() as connection:
            reserved = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT agent_id FROM runs WHERE status IN "
                    "('starting','running','stopping') "
                    "OR cleanup_state IN ('pending','failed')",
                ).fetchall()
            }
        while len((running | reserved) - {agent_id}) >= 2:
            idle = self.repository.idle_agent_ids(running - {agent_id})
            if not idle:
                return False
            victim = min(idle, key=lambda item: (self._idle_since.get(item, 0), item))
            async with self.coordination.agent_locks([victim]):
                if victim not in self.repository.idle_agent_ids([victim]):
                    return False
                await self.preparer.stop_agent(victim)
                self._idle_since.pop(victim, None)
                running.discard(victim)
        return True

    async def _wake_idle(self, owner, agent_id):
        if self.coordination.execution_slots.locked():
            raise RepositoryError(409, "container_capacity", "Both execution slots are busy.")
        async with self.coordination.execution_slot():
            async with self.coordination.lifecycle_lock:
                if not await self.make_room(agent_id):
                    raise RepositoryError(
                        409, "container_capacity", "Both assistant containers are busy."
                    )
                async with self.coordination.agent_locks([agent_id]):
                    self.repository.database.storage.require_writable()
                    if agent_id not in await self.preparer.running_agent_ids():
                        await self.preparer.start_agent(agent_id)
                    self._idle_since[agent_id] = self.clock()

    async def container_state(self, agent_id):
        if self.repository.database.storage.status()["active"]:
            return "unknown"
        try:
            return "running" if agent_id in await self.preparer.running_agent_ids() else "stopped"
        except AgentPreparationError:
            return "unknown"

    async def sleep_idle(self):
        if self.repository.database.storage.status()["active"]:
            return
        async with self.coordination.lifecycle_lock:
            running = set(await self.preparer.running_agent_ids())
            for agent in list(self._idle_since):
                if agent not in running:
                    self._idle_since.pop(agent, None)
                    continue
                if self.clock() - self._idle_since[agent] < 900:
                    continue
                async with self.coordination.agent_locks([agent]):
                    if agent in self.repository.idle_agent_ids([agent]):
                        await self.preparer.stop_agent(agent)
                        self._idle_since.pop(agent, None)

    def start_idle_monitor(self):
        self._idle_task = asyncio.create_task(self._watch_idle(), name="idle-container-sleep")

    async def _watch_idle(self):
        while True:
            await asyncio.sleep(1)
            try:
                await self.sleep_idle()
            except Exception:
                logger.warning(
                    "Idle container sleep could not be confirmed; the next check will retry."
                )

    async def close(self):
        if self._idle_task is not None:
            self._idle_task.cancel()
            await asyncio.gather(self._idle_task, return_exceptions=True)
            self._idle_task = None

    async def purge_archive(self, owner, roots, digest, request_id, *, confirmed=False):
        """Internal healthy/quiescent coordinator; destructive HTTP/UI remains unwired."""
        if confirmed is not True:
            raise RepositoryError(
                422, "confirmation_required", "Confirm permanent history deletion."
            )
        cancelled = threading.Event()

        async def io(function, *args, **kwargs):
            context = copy_context()
            return await self.workspace.io(
                context.run,
                partial(function, *args, **kwargs),
                on_cancel=cancelled.set,
            )

        async def measure():
            try:
                async with asyncio.timeout(5):
                    sample = await self.preparer.platform_storage_usage()
            except (AgentPreparationError, TimeoutError, OSError):
                if self.repository.database.storage.enabled:
                    self.repository.database.storage.observe(None)
                raise
            if self.repository.database.storage.enabled:
                self.repository.database.storage.observe(sample)
            self.repository.database.storage.require_writable()
            return sample

        async with asyncio.timeout(90):
            previous = await io(
                self.repository.archive_purge_result, owner, roots, digest, request_id
            )
            if previous is not None and not previous["reclamation_pending"]:
                return previous
            plan = None
            if previous is None:
                try:
                    plan = await io(
                        self.repository.check_archive_purge,
                        owner,
                        roots,
                        digest,
                        cancelled=cancelled,
                    )
                except RepositoryError:
                    # Another request may have deleted the roots after our first receipt read.
                    previous = await io(
                        self.repository.archive_purge_result, owner, roots, digest, request_id
                    )
                    if previous is None:
                        raise
            if previous is not None:
                if not previous["reclamation_pending"]:
                    return previous
                agent_ids = previous.get("agent_ids")
                if not agent_ids:
                    return {**previous, "maintenance_error": "maintenance_context_missing"}
            else:
                if not plan["eligible"]:
                    raise RepositoryError(
                        409, "archive_dependencies", "Resolve archive blockers before maintenance."
                    )
                agent_ids = plan["agent_ids"]
            for agent in agent_ids:
                self.repository.get_owned_agent(owner, agent)
            async with self.coordination.lifecycle_lock:
                async with self.coordination.agent_locks(agent_ids):
                    # A concurrent request may have completed while this one waited for locks.
                    current = await io(
                        self.repository.archive_purge_result, owner, roots, digest, request_id
                    )
                    if current is not None and not current["reclamation_pending"]:
                        return current
                    previous = current or previous
                    try:
                        async with asyncio.timeout(5):
                            running = await self.preparer.running_agent_ids()
                        if running:
                            raise RepositoryError(
                                409,
                                "maintenance_busy",
                                "Sleep running assistants before maintenance.",
                            )
                        await io(self.repository.require_quiescent_archive_maintenance)
                        with self.repository.database.storage.maintenance():
                            result = previous
                            try:
                                sample = await measure()
                                await io(
                                    self.repository.database.reclaim_space,
                                    sample,
                                    cancelled=cancelled,
                                    _within_maintenance=True,
                                    _check_only=True,
                                )
                                result = await io(
                                    self.repository.purge_archive_records,
                                    owner,
                                    roots,
                                    digest,
                                    request_id,
                                    confirmed=True,
                                    cancelled=cancelled,
                                )
                                replayed = result["replayed"]
                                await io(
                                    self.repository.record_archive_reclamation,
                                    owner,
                                    request_id,
                                    "running",
                                )
                                sample = await measure()
                                measurement = await io(
                                    self.repository.database.reclaim_space,
                                    sample,
                                    cancelled=cancelled,
                                    _within_maintenance=True,
                                )
                                result = await io(
                                    self.repository.record_archive_reclamation,
                                    owner,
                                    request_id,
                                    "completed",
                                    measurement=measurement,
                                )
                                return {**result, "replayed": replayed}
                            except BaseException as error:
                                # Consult the receipt if a cancelled await hid a committed deletion.
                                if result is None:
                                    result = await io(
                                        self.repository.archive_purge_result,
                                        owner,
                                        roots,
                                        digest,
                                        request_id,
                                    )
                                if result is None:
                                    raise
                                code = getattr(error, "code", "maintenance_failed")
                                if isinstance(error, StorageWriteBlocked):
                                    code = "storage_protection"
                                elif isinstance(error, asyncio.CancelledError):
                                    code = "maintenance_cancelled"
                                elif isinstance(error, TimeoutError):
                                    code = "maintenance_timeout"
                                if result["reclamation_pending"]:
                                    try:
                                        saved = await io(
                                            self.repository.record_archive_reclamation,
                                            owner,
                                            request_id,
                                            "pending",
                                            error=code,
                                        )
                                        result = {**saved, "replayed": result["replayed"]}
                                    except (StorageWriteBlocked, sqlite3.Error):
                                        result = {
                                            **result,
                                            "maintenance_error": code,
                                            "reclamation_state": "unconfirmed",
                                        }
                                if isinstance(error, asyncio.CancelledError):
                                    raise
                                if isinstance(
                                    error,
                                    (
                                        StorageMaintenanceError,
                                        StorageWriteBlocked,
                                        AgentPreparationError,
                                        TimeoutError,
                                        sqlite3.Error,
                                        OSError,
                                    ),
                                ):
                                    return result
                                raise
                    except (
                        RepositoryError,
                        StorageWriteBlocked,
                        AgentPreparationError,
                        TimeoutError,
                    ) as error:
                        if previous is not None:
                            code = (
                                "storage_protection"
                                if isinstance(error, StorageWriteBlocked)
                                else "maintenance_timeout"
                                if isinstance(error, TimeoutError)
                                else getattr(error, "code", "maintenance_busy")
                            )
                            return {**previous, "maintenance_error": code}
                        raise
