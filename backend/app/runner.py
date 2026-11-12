from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Protocol

from openai_codex.models import Notification

from .codex_gateway import CodexGateway
from .coordination import Coordination
from .event_store import EventStore
from .repository import Repository, RepositoryError, RunDispatchContext

logger = logging.getLogger(__name__)


class ContainerRuntime(Protocol):
    async def start_agent(self, agent_id: str) -> None: ...

    async def stop_agent(self, agent_id: str) -> None: ...

    def write_run_token(self, agent_id: str, token: str) -> None: ...

    def remove_run_token(self, agent_id: str) -> None: ...


class NetworkAuthority(Protocol):
    def issue_run_token(self, run_id: str, agent_id: str, generation: int) -> str: ...

    def revoke_run(self, run_id: str) -> None: ...

    async def wait_run_closed(self, run_id: str) -> None: ...

    def token_hash(self, raw_token: str) -> str: ...


class TurnHandle(Protocol):
    id: str

    def stream(self): ...

    async def interrupt(self) -> object: ...


class CodexSession(Protocol):
    async def prepare_thread(self, model: str, thread_id: str | None = None): ...

    async def start_turn(self, thread: Any, model: str, prompt: str) -> TurnHandle: ...

    async def close(self) -> None: ...


class CodexRuntime(Protocol):
    async def connect(self, agent_id: str) -> CodexSession: ...


class Runner:
    def __init__(
        self,
        repository: Repository,
        event_store: EventStore,
        coordination: Coordination,
        containers: ContainerRuntime,
        network: NetworkAuthority,
        codex: CodexRuntime | CodexGateway,
        *,
        enabled: bool = True,
        verified_models: frozenset[str] = frozenset(),
        model_admission: Callable[[], frozenset[str]] | None = None,
        notification_mapper: (
            Callable[[Notification], tuple[str, dict[str, object], str | None] | None] | None
        ) = None,
    ) -> None:
        self.repository = repository
        self.event_store = event_store
        self.coordination = coordination
        self.containers = containers
        self.network = network
        self.codex = codex
        self.capacity_admission = None
        self.enabled = enabled
        self.verified_models = verified_models
        self.model_admission = model_admission
        self.notification_mapper = notification_mapper or self._map_notification
        self._agent_tasks: dict[str, asyncio.Task[None]] = {}
        self._active_runs: dict[str, RunDispatchContext] = {}
        self._closing = False
        self._rescan_task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()

    def start(self) -> None:
        if not self.enabled:
            return
        if self._rescan_task is None:
            self._rescan_task = asyncio.create_task(
                self._rescan_queues(),
                name="run-admission-rescan",
            )
        for agent_id in self.repository.queued_agent_ids():
            self._kick_agent(agent_id)

    async def close(self) -> None:
        self._closing = True
        if self._rescan_task is not None:
            self._rescan_task.cancel()
            await asyncio.gather(self._rescan_task, return_exceptions=True)
            self._rescan_task = None
        for task in tuple(self._agent_tasks.values()):
            task.cancel()
        if self._agent_tasks:
            await asyncio.gather(*self._agent_tasks.values(), return_exceptions=True)
        self._agent_tasks.clear()

    def enqueue(self, agent_id: str) -> None:
        if self.enabled and not self._closing:
            self._kick_agent(agent_id)
            self._wake.set()

    def revoke_for_storage(self) -> None:
        self.event_store.drop_pending()
        for run_id in tuple(self._active_runs):
            self.network.revoke_run(run_id)
        for task in tuple(self._agent_tasks.values()):
            if not task.done():
                task.cancel()

    async def storage_stop(self) -> bool:
        # No database write is needed to revoke a run or initiate physical cleanup.
        tasks = set(self._agent_tasks.values())
        self.revoke_for_storage()
        if not tasks:
            return True
        _, pending = await asyncio.wait(tasks, timeout=35)
        return not pending

    async def repair_agent(self, owner_id: str, agent_id: str, *, allow_repair: bool) -> None:
        runs = self.repository.agent_unclean_runs(owner_id, agent_id)
        for run in runs:
            self.network.revoke_run(run["id"])
        worker = self._agent_tasks.get(agent_id)
        if worker is not None and not worker.done():
            worker.cancel()

        async def collect_worker():
            if worker is not None and not worker.done():
                _, pending = await asyncio.wait({worker}, timeout=35)
                if pending:
                    raise TimeoutError("Old assistant worker remains active.")

        try:
            async with asyncio.timeout(35):
                results = await asyncio.gather(
                    collect_worker(),
                    self._stop_container(agent_id),
                    *(self.network.wait_run_closed(run["id"]) for run in runs),
                    return_exceptions=True,
                )
            if any(isinstance(result, BaseException) for result in results):
                raise RuntimeError("Old assistant cleanup remains unconfirmed.")
            self.containers.remove_run_token(agent_id)
            remaining = self.repository.agent_unclean_runs(owner_id, agent_id)
            if remaining and not allow_repair:
                raise RuntimeError("Restart is required for unconfirmed cleanup records.")
            for run in remaining:
                intended, reason = self.repository.run_cleanup_intent(run["id"])
                self.event_store.finalize(
                    run["id"],
                    {
                        "status": intended,
                        "reason": reason,
                        "history_incomplete": True,
                    },
                )
                self.repository.complete_recovered_run(run["id"])
        except BaseException:
            self.repository.set_recovery_blocker(agent_id, blocked=True)
            raise

    async def stop_run(self, owner_id: str, run_id: str) -> None:
        run, _last_sequence = self.repository.get_owned_run(owner_id, run_id)
        async with self.coordination.agent_locks([run.agent_id]):
            stopping, stop_requested = self.repository.request_run_stop(owner_id, run_id)
            if stop_requested:
                self.network.revoke_run(run_id)
        if stopping.status != "stopping":
            return
        if stopping.cleanup_state == "failed":
            raise RepositoryError(
                503,
                "run_cleanup_unconfirmed",
                "The run could not be confirmed stopped; recovery is required.",
            )

        task = self._agent_tasks.get(run.agent_id)
        if task is None or task.done():
            self._mark_cleanup_failed(run_id)
            raise RepositoryError(
                503,
                "run_cleanup_unconfirmed",
                "The run could not be confirmed stopped; recovery is required.",
            )
        if stop_requested:
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=35)
        except TimeoutError as error:
            self._mark_cleanup_failed(run_id)
            raise RepositoryError(
                503,
                "run_cleanup_timeout",
                "The run did not stop within the cleanup deadline; recovery is required.",
            ) from error
        except Exception as error:
            self._mark_cleanup_failed(run_id)
            raise RepositoryError(
                503,
                "run_cleanup_unconfirmed",
                "The run could not be confirmed stopped; recovery is required.",
            ) from error
        stopped, _last_sequence = self.repository.get_owned_run(owner_id, run_id)
        if stopped.status == "stopping" or stopped.cleanup_state != "confirmed":
            raise RepositoryError(
                503,
                "run_cleanup_unconfirmed",
                "The run could not be confirmed stopped; recovery is required.",
            )

    def _kick_agent(self, agent_id: str) -> None:
        if self._closing:
            return
        current = self._agent_tasks.get(agent_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(self._run_agent_queue(agent_id))
        self._agent_tasks[agent_id] = task

        def finish(completed: asyncio.Task[None]) -> None:
            if self._agent_tasks.get(agent_id) is completed:
                self._agent_tasks.pop(agent_id, None)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                logger.error("Run queue worker failed for assistant %s", agent_id)

        task.add_done_callback(finish)

    async def _rescan_queues(self) -> None:
        # A retained queue must wake after a rolling window expires without another HTTP write.
        while not self._closing:
            self._wake.clear()
            try:
                for agent_id in self.repository.queued_agent_ids():
                    self._kick_agent(agent_id)
            except Exception:
                logger.exception("Queued work could not be reconsidered; the next scan will retry.")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=1)
            except TimeoutError:
                pass

    async def _run_agent_queue(self, agent_id: str) -> None:
        while not self._closing:
            if self.repository.database.storage.status()["active"]:
                return
            context: RunDispatchContext | None = None
            async with self.coordination.execution_slot():
                async with self.coordination.lifecycle_lock:
                    # Refresh after capacity waits, before selecting an eligible owner/assistant.
                    if self.model_admission is not None:
                        self.verified_models = frozenset()
                        try:
                            self.verified_models = self.model_admission()
                        except Exception:
                            logger.exception("Model admission refresh failed; dispatch blocked.")
                            return
                    if self.repository.next_runnable_agent(self.verified_models) != agent_id:
                        return
                    if self.capacity_admission is not None and not await self.capacity_admission(
                        agent_id
                    ):
                        return
                    async with self.coordination.agent_locks([agent_id]):
                        context = self.repository.claim_next_run(
                            agent_id,
                            verified_models=self.verified_models,
                        )
                if context is None:
                    return
                self._active_runs[context.run.id] = context
                try:
                    await self._execute_run(context)
                finally:
                    self._active_runs.pop(context.run.id, None)
            # Let an eligible different owner take the next slot before this agent loops.
            self._wake.set()

    async def _execute_run(self, context: RunDispatchContext) -> None:
        run = context.run
        raw_token: str | None = None
        session: CodexSession | None = None
        turn: TurnHandle | None = None
        intended_status = "completed"
        error_code: str | None = None
        error_message: str | None = None
        had_tool_errors = False
        kickoff_cancelled = False
        try:
            async with self.coordination.lifecycle_lock:
                if self.capacity_admission is not None and not await self.capacity_admission(
                    run.agent_id
                ):
                    raise RepositoryError(
                        409, "container_capacity", "Container capacity changed before startup."
                    )
                await self.containers.start_agent(run.agent_id)
            raw_token = self.network.issue_run_token(
                run.id,
                run.agent_id,
                context.generation,
            )
            self.containers.write_run_token(run.agent_id, raw_token)
            session = await self.codex.connect(run.agent_id)
            thread = await session.prepare_thread(run.model, context.thread_id)
            self.repository.update_run_thread(run.id, thread.id)
            async with self.coordination.agent_locks([run.agent_id]):
                allowed = self.repository.update_run_running(
                    run.id,
                    thread_id=thread.id,
                    generation=context.generation,
                    run_token_hash=self.network.token_hash(raw_token),
                )
            if not allowed:
                intended_status = "cancelled"
                error_code = "run_cancelled"
                error_message = "The run was cancelled before it started."
            else:
                turn = await session.start_turn(thread, run.model, run.prompt)
                self.repository.update_run_turn_id(run.id, turn.id)
                turn_status: str | None = None
                async for notification in turn.stream():
                    if notification.method == "turn/completed":
                        completed_payload = self._notification_payload(notification)
                        turn_data = completed_payload.get("turn")
                        if isinstance(turn_data, dict):
                            raw_status = turn_data.get("status")
                            if isinstance(raw_status, str):
                                turn_status = raw_status
                    mapped = self.notification_mapper(notification)
                    if mapped is None:
                        continue
                    kind, payload, item_id = mapped
                    event = self.event_store.append(
                        run.id,
                        kind,  # type: ignore[arg-type]
                        payload,
                        sdk_item_id=item_id,
                    )
                    if kind == "error" or (
                        event is not None
                        and kind == "activity_completed"
                        and payload.get("status") == "failed"
                    ):
                        had_tool_errors = True
                if turn_status == "failed" or turn_status is None:
                    intended_status = "failed"
                    error_code = (
                        "codex_turn_failed"
                        if turn_status == "failed"
                        else "turn_completion_missing"
                    )
                    error_message = "The assistant could not complete this run."
                elif turn_status == "interrupted":
                    intended_status = "interrupted"
                    error_code = "codex_turn_interrupted"
                    error_message = "The assistant run was interrupted."
                elif had_tool_errors:
                    intended_status = "completed_with_errors"
        except asyncio.CancelledError:
            try:
                requested_status, requested_reason = self.repository.run_cleanup_intent(run.id)
            except Exception:
                logger.exception("Could not read stop intent for run %s", run.id)
                requested_status, requested_reason = None, None
            intended_status = requested_status or "interrupted"
            error_code = (
                requested_reason
                if requested_reason in {"user_stop", "manual_sleep", "manual_restart"}
                else "backend_shutdown"
            )
            error_message = (
                "The run was stopped by the user."
                if requested_reason == "user_stop"
                else "The run was interrupted while the backend was stopping."
            )
            kickoff_cancelled = turn is None
        except Exception:
            intended_status = "failed"
            error_code = "codex_run_failed"
            error_message = "The assistant could not complete this run."
            logger.warning("Run execution failed for run %s", run.id)

        cleanup_task = asyncio.create_task(
            self._finalize_run(
                context,
                raw_token,
                session,
                turn,
                intended_status,
                error_code,
                error_message,
                had_tool_errors,
                kickoff_cancelled,
            )
        )
        try:
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            await cleanup_task
        except Exception:
            logger.exception("Run finalization failed for run %s", run.id)
            self._mark_cleanup_failed(run.id)

    async def _finalize_run(
        self,
        context: RunDispatchContext,
        raw_token: str | None,
        session: CodexSession | None,
        turn: TurnHandle | None,
        intended_status: str,
        error_code: str | None,
        error_message: str | None,
        had_tool_errors: bool,
        kickoff_cancelled: bool,
    ) -> None:
        run = context.run
        if raw_token is not None:
            self.network.revoke_run(run.id)
        cleanup_state_saved = True
        async with self.coordination.agent_locks([run.agent_id]):
            try:
                requested_status, requested_reason = self.repository.run_cleanup_intent(run.id)
                if requested_reason in {"user_stop", "manual_sleep", "manual_restart"}:
                    intended_status = requested_status or "interrupted"
                    error_code = requested_reason
                    error_message = "The run was stopped by an assistant control."
            except Exception:
                logger.exception("Could not read cleanup intent for run %s", run.id)
                cleanup_state_saved = False
            try:
                self.repository.begin_run_cleanup(
                    run.id,
                    intended_status,
                    (
                        "turn_complete"
                        if intended_status.startswith("completed")
                        else error_code or "run_failed"
                    ),
                )
            except Exception:
                logger.warning("Could not begin cleanup for run %s", run.id)
                cleanup_state_saved = False

        cleanup_operations = [
            session.close() if session is not None else self._no_op(),
            self._stop_container(run.agent_id),
            self.network.wait_run_closed(run.id),
        ]
        required_cleanup_count = len(cleanup_operations)
        if intended_status == "interrupted" and turn is not None:
            cleanup_operations.append(turn.interrupt())
        try:
            cleanup_results = await asyncio.wait_for(
                asyncio.gather(*cleanup_operations, return_exceptions=True),
                timeout=35,
            )
        except TimeoutError:
            cleanup_results = [TimeoutError("run cleanup deadline exceeded")]
        # Interrupt is advisory: SDK close and container exit can close its RPC transport.
        # Client, container and gateway termination remain mandatory, including timeouts.
        cleanup_failed = any(
            isinstance(result, BaseException)
            for result in cleanup_results[:required_cleanup_count]
        )
        if len(cleanup_results) > required_cleanup_count:
            interrupt_result = cleanup_results[required_cleanup_count]
            if isinstance(interrupt_result, BaseException):
                logger.info("Interrupt RPC ended during cleanup for run %s (%s)",
                            run.id, type(interrupt_result).__name__)
        try:
            self.containers.remove_run_token(run.agent_id)
        except Exception:
            cleanup_failed = True
        if cleanup_failed or not cleanup_state_saved:
            self._mark_cleanup_failed(run.id)
            return

        final_payload: dict[str, object] = {"status": intended_status}
        if error_code is not None:
            final_payload.update({"error_code": error_code, "message": error_message})
        if kickoff_cancelled:
            final_payload["history_incomplete"] = True
        try:
            if error_code is not None:
                self.event_store.append(
                    run.id,
                    "error",
                    {"code": error_code, "message": error_message or "The run failed."},
                )
            self.event_store.finalize(run.id, final_payload)
            self.repository.complete_run(
                run.id,
                intended_status,
                error_code=error_code,
                error_message=error_message,
                had_tool_errors=had_tool_errors,
            )
        except Exception:
            logger.warning("Could not persist final run state for run %s", run.id)
            self._mark_cleanup_failed(run.id)

    @staticmethod
    async def _no_op() -> None:
        return None

    async def _stop_container(self, agent_id: str) -> None:
        async with self.coordination.lifecycle_lock:
            await self.containers.stop_agent(agent_id)

    def _mark_cleanup_failed(self, run_id: str) -> None:
        try:
            self.repository.fail_run_cleanup(run_id)
        except Exception:
            logger.warning("Could not persist cleanup failure for run %s", run_id)

    @staticmethod
    def _notification_payload(notification: Notification) -> dict[str, object]:
        payload_object = notification.payload
        if isinstance(payload_object, dict):
            return payload_object
        if hasattr(payload_object, "model_dump"):
            payload = payload_object.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            )
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _map_notification(
        notification: Notification,
    ) -> tuple[str, dict[str, object], str | None] | None:
        payload = Runner._notification_payload(notification)
        method = notification.method
        if method == "item/agentMessage/delta":
            item_id = payload.get("itemId") or payload.get("item_id")
            text = payload.get("delta")
            if isinstance(item_id, str) and isinstance(text, str):
                return "message_delta", {"item_id": item_id, "text": text}, item_id
        if method == "item/started":
            item = payload.get("item")
            if isinstance(item, dict):
                item_id = item.get("id")
                if isinstance(item_id, str):
                    return (
                        "activity_started",
                        {
                            "item_id": item_id,
                            "name": item.get("type", "activity"),
                            "summary": Runner._safe_activity_summary(item),
                        },
                        item_id,
                    )
        if method == "item/completed":
            item = payload.get("item")
            if isinstance(item, dict):
                item_id = item.get("id")
                if isinstance(item_id, str):
                    return (
                        "activity_completed",
                        {
                            "item_id": item_id,
                            "name": item.get("type", "activity"),
                            "status": item.get("status", "completed"),
                            "summary": Runner._safe_activity_summary(item),
                        },
                        item_id,
                    )
        if method == "error":
            return "error", {"message": "Codex reported an error."}, None
        return None

    @staticmethod
    def _safe_activity_summary(item: dict[str, object]) -> str:
        item_type = item.get("type")
        if item_type == "commandExecution":
            command = item.get("command")
            if isinstance(command, str):
                return command[:500]
        return str(item_type or "activity")[:100]
