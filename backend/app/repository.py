from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass

from .db import Database

MAX_AGENTS_PER_USER = 5
MAX_AGENTS_PER_INSTANCE = 20


class RepositoryError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class AgentRecord:
    id: str
    owner_id: str
    name: str
    model: str
    status: str
    paused: bool
    blockers: tuple[str, ...]
    container_state: str
    created_at: int
    chat_epoch: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> AgentRecord:
        return cls(
            id=row["id"],
            owner_id=row["owner_id"],
            name=row["name"],
            model=row["model"],
            status=row["status"],
            paused=bool(row["paused"]),
            blockers=tuple(json.loads(row["blockers_json"])),
            container_state="unknown",
            created_at=row["created_at"],
            chat_epoch=row["chat_epoch"],
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    agent_id: str
    source: str
    job_id: str | None
    scheduled_for: int | None
    coalesced_count: int
    retry_of: str | None
    prompt: str
    status: str
    model: str
    chat_epoch: int
    created_at: int
    started_at: int | None
    finished_at: int | None
    thread_id: str | None
    sdk_turn_id: str | None
    termination_reason: str | None
    cleanup_state: str
    error_code: str | None
    error_message: str | None
    paused: bool

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RunRecord:
        return cls(
            id=row["id"],
            agent_id=row["agent_id"],
            source=row["source"],
            job_id=row["job_id"],
            scheduled_for=row["scheduled_for"],
            coalesced_count=row["coalesced_count"],
            retry_of=row["retry_of"],
            prompt=row["prompt"],
            status=row["status"],
            model=row["model"],
            chat_epoch=row["chat_epoch"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            thread_id=row["thread_id"],
            sdk_turn_id=row["sdk_turn_id"],
            termination_reason=row["termination_reason"],
            cleanup_state=row["cleanup_state"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            paused=bool(row["paused"]),
        )


@dataclass(frozen=True, slots=True)
class RunDispatchContext:
    run: RunRecord
    thread_id: str | None
    generation: int
    blockers: tuple[str, ...]
    current_chat_epoch: int


class Repository:
    def __init__(
        self,
        database: Database,
        *,
        clock=None,
        minute_start_limit: int = 5,
        day_start_limit: int = 20,
    ) -> None:
        if any(
            type(value) is not int or value < 1 for value in (minute_start_limit, day_start_limit)
        ):
            raise ValueError("Run-start limits must be positive integers.")
        self.database = database
        self.clock = clock or (lambda: time.time_ns() // 1_000_000)
        self.minute_start_limit = minute_start_limit
        self.day_start_limit = day_start_limit

    def create_agent(
        self,
        owner_id: str,
        name: str,
        model: str,
        request_id: str,
    ) -> tuple[AgentRecord, bool]:
        body_hash = hashlib.sha256(
            json.dumps({"model": model, "name": name}, sort_keys=True).encode()
        ).hexdigest()
        now_ms = time.time_ns() // 1_000_000
        with self.database.transaction() as connection:
            previous = connection.execute(
                """
                SELECT body_hash, entity_id
                FROM idempotency
                WHERE actor_kind = 'user' AND actor_id = ? AND operation = 'agent.create'
                  AND resource_id = 'collection:agents' AND request_id = ?
                """,
                (owner_id, request_id),
            ).fetchone()
            if previous is not None:
                if previous["body_hash"] != body_hash:
                    raise RepositoryError(
                        409,
                        "idempotency_conflict",
                        "This request ID was already used with different content.",
                    )
                if previous["entity_id"] is None:
                    raise RepositoryError(
                        503,
                        "operation_incomplete",
                        "The previous assistant creation has no saved result.",
                    )
                row = connection.execute(
                    "SELECT * FROM agents WHERE id = ? AND owner_id = ?",
                    (previous["entity_id"], owner_id),
                ).fetchone()
                if row is None:
                    raise RepositoryError(
                        503,
                        "operation_incomplete",
                        "The previous assistant creation result is unavailable.",
                    )
                return AgentRecord.from_row(row), True

            user_count = connection.execute(
                "SELECT COUNT(*) FROM agents WHERE owner_id = ?", (owner_id,)
            ).fetchone()[0]
            total_count = connection.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
            if user_count >= MAX_AGENTS_PER_USER or total_count >= MAX_AGENTS_PER_INSTANCE:
                raise RepositoryError(429, "agent_limit", "The assistant limit has been reached.")

            agent_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO agents(
                    id, owner_id, name, model, status, paused, blockers_json,
                    generation, created_at
                ) VALUES (?, ?, ?, ?, 'provisioning', 0, '[]', 0, ?)
                """,
                (agent_id, owner_id, name, model, now_ms),
            )
            connection.execute(
                """
                INSERT INTO idempotency(
                    actor_kind, actor_id, operation, resource_id, request_id,
                    body_hash, entity_id, operation_state, response_code,
                    response_json, created_at
                ) VALUES ('user', ?, 'agent.create', 'collection:agents', ?, ?, ?,
                          'accepted', 201, NULL, ?)
                """,
                (owner_id, request_id, body_hash, agent_id, now_ms),
            )
            row = connection.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                raise RuntimeError("New assistant row was not persisted")
            return AgentRecord.from_row(row), False

    def list_agents(self, owner_id: str) -> list[AgentRecord]:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT * FROM agents WHERE owner_id = ? ORDER BY created_at, id",
                (owner_id,),
            ).fetchall()
        return [AgentRecord.from_row(row) for row in rows]

    def get_owned_agent(self, owner_id: str, agent_id: str) -> AgentRecord:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT * FROM agents WHERE id = ? AND owner_id = ?",
                (agent_id, owner_id),
            ).fetchone()
        if row is None:
            raise RepositoryError(404, "not_found", "Assistant not found.")
        return AgentRecord.from_row(row)

    def create_chat_run(
        self,
        owner_id: str,
        agent_id: str,
        request_id: str,
        prompt: str,
    ) -> tuple[RunRecord, bool]:
        owner_id = self._canonical_uuid(owner_id, "owner_id")
        agent_id = self._canonical_uuid(agent_id, "agent_id")
        request_id = self._canonical_uuid(request_id, "request_id")
        if not prompt.strip() or len(prompt.encode("utf-8")) > 16 * 1024:
            raise RepositoryError(422, "invalid_prompt", "The message must be 1–16 KiB.")
        body_hash = hashlib.sha256(
            json.dumps({"prompt": prompt}, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        now_ms = time.time_ns() // 1_000_000
        with self.database.transaction() as connection:
            agent = connection.execute(
                "SELECT * FROM agents WHERE id = ? AND owner_id = ?",
                (agent_id, owner_id),
            ).fetchone()
            if agent is None:
                raise RepositoryError(404, "not_found", "Assistant not found.")

            previous = connection.execute(
                """
                SELECT body_hash, entity_id, operation_state FROM idempotency
                WHERE actor_kind = 'user' AND actor_id = ? AND operation = 'chat.create'
                  AND resource_id = ? AND request_id = ?
                """,
                (owner_id, agent_id, request_id),
            ).fetchone()
            if previous is not None:
                if previous["body_hash"] != body_hash:
                    raise RepositoryError(
                        409,
                        "idempotency_conflict",
                        "This request ID was already used with different content.",
                    )
                if previous["operation_state"] == "purged":
                    raise RepositoryError(
                        410, "run_purged", "This saved run was explicitly purged."
                    )
                row = connection.execute(
                    """
                    SELECT runs.*, agents.paused
                    FROM runs JOIN agents ON agents.id = runs.agent_id
                    WHERE runs.id = ? AND agents.owner_id = ? AND runs.source = 'chat'
                    """,
                    (previous["entity_id"], owner_id),
                ).fetchone()
                if row is None:
                    raise RepositoryError(
                        503,
                        "operation_incomplete",
                        "The previous message result is unavailable.",
                    )
                return RunRecord.from_row(row), True

            if agent["status"] != "ready":
                raise RepositoryError(
                    409,
                    "assistant_not_ready",
                    "Prepare the assistant before sending a message.",
                )
            queued_count = connection.execute(
                """
                SELECT COUNT(*) FROM runs
                WHERE agent_id = ? AND status = 'queued'
                """,
                (agent_id,),
            ).fetchone()[0]
            if queued_count >= 20:
                raise RepositoryError(429, "queue_full", "The assistant's run queue is full.")

            run_id = str(uuid.uuid4())
            chat_epoch = agent["chat_epoch"]
            connection.execute(
                """
                INSERT INTO runs(
                    id, agent_id, source, prompt, status, model, chat_epoch, created_at
                ) VALUES (?, ?, 'chat', ?, 'queued', ?, ?, ?)
                """,
                (run_id, agent_id, prompt, agent["model"], chat_epoch, now_ms),
            )
            user_item_id = f"user:{run_id}"
            user_payload = json.dumps(
                {
                    "item_id": user_item_id,
                    "full_text": prompt,
                    "role": "user",
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            connection.execute(
                """
                INSERT INTO events(run_id, sequence, kind, sdk_item_id, payload_json, created_at)
                VALUES (?, 1, 'message_snapshot', ?, ?, ?)
                """,
                (run_id, user_item_id, user_payload, now_ms),
            )
            connection.execute(
                """
                INSERT INTO idempotency(
                    actor_kind, actor_id, operation, resource_id, request_id,
                    body_hash, entity_id, operation_state, response_code,
                    response_json, created_at
                ) VALUES ('user', ?, 'chat.create', ?, ?, ?, ?, 'accepted', 202, NULL, ?)
                """,
                (owner_id, agent_id, request_id, body_hash, run_id, now_ms),
            )
            row = connection.execute(
                """
                SELECT runs.*, agents.paused
                FROM runs JOIN agents ON agents.id = runs.agent_id
                WHERE runs.id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("New run row was not persisted")
        return RunRecord.from_row(row), False

    def get_owned_run(self, owner_id: str, run_id: str) -> tuple[RunRecord, int]:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT runs.*, agents.paused,
                       COALESCE(MAX(events.sequence), 0) AS last_event_sequence
                FROM runs
                JOIN agents ON agents.id = runs.agent_id
                LEFT JOIN events ON events.run_id = runs.id
                WHERE agents.owner_id = ? AND runs.id = ?
                GROUP BY runs.id
                """,
                (owner_id, run_id),
            ).fetchone()
        if row is None:
            raise RepositoryError(404, "not_found", "Run not found.")
        return RunRecord.from_row(row), row["last_event_sequence"]

    def retry_chat_run(
        self,
        owner_id: str,
        run_id: str,
        request_id: str,
    ) -> tuple[RunRecord, bool, bool]:
        owner_id = self._canonical_uuid(owner_id, "owner_id")
        run_id = self._canonical_uuid(run_id, "run_id")
        request_id = self._canonical_uuid(request_id, "request_id")
        body_hash = hashlib.sha256(
            json.dumps({"action": "retry", "run_id": run_id}, sort_keys=True).encode()
        ).hexdigest()
        now_ms = time.time_ns() // 1_000_000
        with self.database.transaction() as connection:
            previous = connection.execute(
                """
                SELECT body_hash, entity_id, response_json, operation_state
                FROM idempotency
                WHERE actor_kind = 'user' AND actor_id = ? AND operation = 'run.retry'
                  AND resource_id = ? AND request_id = ?
                """,
                (owner_id, run_id, request_id),
            ).fetchone()
            if previous is not None:
                if previous["body_hash"] != body_hash:
                    raise RepositoryError(
                        409,
                        "idempotency_conflict",
                        "This request ID was already used with different content.",
                    )
                if previous["operation_state"] == "purged":
                    raise RepositoryError(
                        410, "run_purged", "This saved run was explicitly purged."
                    )
                row = connection.execute(
                    """
                    SELECT runs.*, agents.paused
                    FROM runs JOIN agents ON agents.id = runs.agent_id
                    WHERE runs.id = ? AND agents.owner_id = ?
                    """,
                    (previous["entity_id"], owner_id),
                ).fetchone()
                if row is None:
                    raise RepositoryError(
                        503,
                        "operation_incomplete",
                        "The previous retry result is unavailable.",
                    )
                response = json.loads(previous["response_json"] or "{}")
                return RunRecord.from_row(row), True, bool(response.get("context_warning"))

            original = connection.execute(
                """
                SELECT runs.*, agents.owner_id, agents.chat_epoch AS current_chat_epoch,
                       agents.chat_thread_id, agents.paused
                FROM runs JOIN agents ON agents.id = runs.agent_id
                WHERE runs.id = ? AND agents.owner_id = ?
                """,
                (run_id, owner_id),
            ).fetchone()
            if original is None:
                raise RepositoryError(404, "not_found", "Run not found.")
            if original["status"] not in {"failed", "interrupted", "completed_with_errors"}:
                raise RepositoryError(
                    409,
                    "run_not_retryable",
                    "Only failed, interrupted, or partially completed runs can be retried.",
                )
            if (
                original["source"] == "chat"
                and original["chat_epoch"] != original["current_chat_epoch"]
            ):
                raise RepositoryError(
                    409,
                    "stale_chat_run",
                    "This run belongs to an older conversation and cannot be retried.",
                )
            if original["source"] == "job":
                job = connection.execute(
                    "SELECT cancelled_at FROM jobs WHERE id = ?",
                    (original["job_id"],),
                ).fetchone()
                busy = connection.execute(
                    "SELECT 1 FROM runs WHERE job_id=? "
                    "AND status IN ('queued','starting','running','stopping')",
                    (original["job_id"],),
                ).fetchone()
                if job["cancelled_at"] is not None or busy is not None:
                    raise RepositoryError(
                        409,
                        "job_not_retryable",
                        "The job is cancelled or has waiting work.",
                    )
            queued_count = connection.execute(
                """
                SELECT COUNT(*) FROM runs
                WHERE agent_id = ? AND status = 'queued'
                """,
                (original["agent_id"],),
            ).fetchone()[0]
            if queued_count >= 20:
                raise RepositoryError(429, "queue_full", "The assistant's run queue is full.")

            retry_id = str(uuid.uuid4())
            thread_id = original["thread_id"] or (
                original["chat_thread_id"] if original["source"] == "chat" else None
            )
            context_warning = thread_id is None
            connection.execute(
                """
                INSERT INTO runs(
                    id, agent_id, source, job_id, retry_of, prompt, status, model, chat_epoch,
                    thread_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    retry_id,
                    original["agent_id"],
                    original["source"],
                    original["job_id"],
                    run_id,
                    original["prompt"],
                    original["model"],
                    original["chat_epoch"],
                    thread_id,
                    now_ms,
                ),
            )
            user_item_id = f"user:{retry_id}"
            user_payload = json.dumps(
                {"item_id": user_item_id, "full_text": original["prompt"], "role": "user"},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            connection.execute(
                """
                INSERT INTO events(run_id, sequence, kind, sdk_item_id, payload_json, created_at)
                VALUES (?, 1, 'message_snapshot', ?, ?, ?)
                """,
                (retry_id, user_item_id, user_payload, now_ms),
            )
            connection.execute(
                """
                INSERT INTO idempotency(
                    actor_kind, actor_id, operation, resource_id, request_id,
                    body_hash, entity_id, operation_state, response_code,
                    response_json, created_at
                ) VALUES ('user', ?, 'run.retry', ?, ?, ?, ?, 'accepted', 202, ?, ?)
                """,
                (
                    owner_id,
                    run_id,
                    request_id,
                    body_hash,
                    retry_id,
                    json.dumps({"context_warning": context_warning}, separators=(",", ":")),
                    now_ms,
                ),
            )
            row = connection.execute(
                """
                SELECT runs.*, agents.paused
                FROM runs JOIN agents ON agents.id = runs.agent_id
                WHERE runs.id = ?
                """,
                (retry_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("New retry run was not persisted")
        return RunRecord.from_row(row), False, context_warning

    def request_run_stop(self, owner_id: str, run_id: str) -> tuple[RunRecord, bool]:
        owner_id = self._canonical_uuid(owner_id, "owner_id")
        run_id = self._canonical_uuid(run_id, "run_id")
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT runs.*, agents.owner_id, agents.paused
                FROM runs JOIN agents ON agents.id = runs.agent_id
                WHERE runs.id = ? AND agents.owner_id = ?
                """,
                (run_id, owner_id),
            ).fetchone()
            if row is None:
                raise RepositoryError(404, "not_found", "Run not found.")
            if row["status"] in {
                "completed",
                "completed_with_errors",
                "failed",
                "interrupted",
                "cancelled",
            }:
                if row["termination_reason"] == "user_stop":
                    return RunRecord.from_row(row), False
                raise RepositoryError(409, "run_not_active", "The run is no longer active.")
            if row["status"] == "queued":
                raise RepositoryError(409, "run_not_active", "The run has not started yet.")
            if row["status"] == "stopping":
                connection.execute(
                    "UPDATE agents SET paused = 1 WHERE id = ?",
                    (row["agent_id"],),
                )
                return RunRecord.from_row(row), False
            connection.execute(
                """
                UPDATE agents SET paused = 1, generation = generation + 1
                WHERE id = ?
                """,
                (row["agent_id"],),
            )
            connection.execute(
                """
                UPDATE runs SET status = 'stopping', cleanup_state = 'pending',
                    intended_terminal_status = 'interrupted',
                    intended_termination_reason = 'user_stop'
                WHERE id = ? AND status IN ('starting', 'running')
                """,
                (run_id,),
            )
            updated = connection.execute(
                """
                SELECT runs.*, agents.paused
                FROM runs JOIN agents ON agents.id = runs.agent_id
                WHERE runs.id = ?
                """,
                (run_id,),
            ).fetchone()
        if updated is None:
            raise RuntimeError("Run row disappeared after requesting stop")
        return RunRecord.from_row(updated), True

    def continue_agent(
        self,
        owner_id: str,
        agent_id: str,
        *,
        verified_models: frozenset[str],
        request_id: str | None = None,
    ) -> AgentRecord:
        owner_id = self._canonical_uuid(owner_id, "owner_id")
        agent_id = self._canonical_uuid(agent_id, "agent_id")
        if request_id is not None:
            request_id = self._canonical_uuid(request_id, "request_id")
        with self.database.transaction() as connection:
            agent = connection.execute(
                "SELECT * FROM agents WHERE id = ? AND owner_id = ?",
                (agent_id, owner_id),
            ).fetchone()
            if agent is None:
                raise RepositoryError(404, "not_found", "Assistant not found.")
            if (
                request_id is not None
                and connection.execute(
                    "SELECT 1 FROM idempotency WHERE actor_kind='user' AND actor_id=? "
                    "AND operation='agent.control.continue' AND resource_id=? AND request_id=?",
                    (owner_id, agent_id, request_id),
                ).fetchone()
            ):
                return AgentRecord.from_row(agent)
            if agent["status"] != "ready":
                raise RepositoryError(409, "assistant_not_ready", "Prepare the assistant first.")
            if agent["model"] not in verified_models:
                raise RepositoryError(
                    409,
                    "model_unavailable",
                    "The assistant's model has not passed the current runtime check.",
                )
            blockers = json.loads(agent["blockers_json"])
            if blockers:
                raise RepositoryError(
                    409,
                    "assistant_blocked",
                    "Resolve the assistant's blockers before continuing tasks.",
                )
            if self.lifecycle_pending(connection, agent_id):
                raise RepositoryError(
                    409, "control_pending", "An assistant control is in progress."
                )
            active_run = connection.execute(
                """
                SELECT 1 FROM runs
                WHERE agent_id = ? AND status IN ('starting', 'running', 'stopping')
                LIMIT 1
                """,
                (agent_id,),
            ).fetchone()
            if active_run is not None:
                raise RepositoryError(
                    409,
                    "run_cleanup_pending",
                    "Wait for the current run to finish stopping before continuing.",
                )
            connection.execute("UPDATE agents SET paused = 0 WHERE id = ?", (agent_id,))
            if request_id is not None:
                connection.execute(
                    "INSERT INTO idempotency(actor_kind,actor_id,operation,resource_id,request_id,"
                    "body_hash,entity_id,operation_state,created_at) VALUES ('user',?,"
                    "'agent.control.continue',?,?,?,?,'completed',?)",
                    (
                        owner_id,
                        agent_id,
                        request_id,
                        hashlib.sha256(b"{}").hexdigest(),
                        agent_id,
                        self.clock(),
                    ),
                )
            updated = connection.execute(
                "SELECT * FROM agents WHERE id = ?",
                (agent_id,),
            ).fetchone()
        if updated is None:
            raise RuntimeError("Assistant row disappeared after continuing")
        return AgentRecord.from_row(updated)

    def update_model_availability(self, verified_models: frozenset[str]) -> None:
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT id, model, paused, blockers_json FROM agents WHERE status = 'ready'"
            ).fetchall()
            for row in rows:
                current_blockers = set(json.loads(row["blockers_json"]))
                blockers = set(current_blockers)
                if row["model"] in verified_models:
                    blockers.discard("model_setup_required")
                else:
                    blockers.add("model_setup_required")
                paused = bool(row["paused"]) or row["model"] not in verified_models
                if blockers != current_blockers or paused != bool(row["paused"]):
                    connection.execute(
                        "UPDATE agents SET paused = ?, blockers_json = ? WHERE id = ?",
                        (int(paused), json.dumps(sorted(blockers)), row["id"]),
                    )

    def list_owned_chat_runs(
        self,
        owner_id: str,
        agent_id: str,
        *,
        limit: int = 50,
    ) -> list[RunRecord]:
        if not 1 <= limit <= 100:
            raise RepositoryError(422, "invalid_limit", "limit must be between 1 and 100.")
        with self.database.read() as connection:
            rows = connection.execute(
                """
                SELECT runs.*, agents.paused
                FROM runs JOIN agents ON agents.id = runs.agent_id
                WHERE agents.owner_id = ? AND agents.id = ? AND runs.source = 'chat'
                ORDER BY runs.created_at DESC, runs.id DESC
                LIMIT ?
                """,
                (owner_id, agent_id, limit),
            ).fetchall()
        return [RunRecord.from_row(row) for row in rows]

    def update_run_starting(self, run_id: str) -> RunRecord:
        now_ms = time.time_ns() // 1_000_000
        with self.database.transaction() as connection:
            updated = connection.execute(
                """
                UPDATE runs SET status = 'starting', started_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now_ms, run_id),
            )
            if updated.rowcount != 1:
                raise RepositoryError(409, "run_not_queued", "The run is no longer queued.")
            row = connection.execute(
                """
                SELECT runs.*, agents.paused FROM runs
                JOIN agents ON agents.id = runs.agent_id WHERE runs.id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("Run row disappeared after starting")
        return RunRecord.from_row(row)

    def claim_next_run(
        self,
        agent_id: str,
        *,
        verified_models: frozenset[str],
    ) -> RunDispatchContext | None:
        agent_id = self._canonical_uuid(agent_id, "agent_id")
        now_ms = self.clock()
        with self.database.transaction() as connection:
            if self.lifecycle_pending(connection, agent_id):
                return None
            if self._run_start_usage(connection, now_ms)["limited"]:
                return None
            if connection.execute(
                "SELECT protection_latched FROM platform_storage_state WHERE id=1"
            ).fetchone()[0]:
                return None
            agent = connection.execute(
                "SELECT * FROM agents WHERE id = ?",
                (agent_id,),
            ).fetchone()
            if (
                agent is None
                or agent["status"] != "ready"
                or agent["paused"]
                or json.loads(agent["blockers_json"])
            ):
                return None
            active_run = connection.execute(
                """
                SELECT 1 FROM runs
                WHERE agent_id = ?
                  AND (status IN ('starting', 'running', 'stopping')
                       OR cleanup_state IN ('pending', 'failed'))
                LIMIT 1
                """,
                (agent_id,),
            ).fetchone()
            if active_run is not None:
                return None
            row = connection.execute(
                """
                SELECT * FROM runs
                WHERE agent_id = ? AND status = 'queued'
                ORDER BY created_at, id
                LIMIT 1
                """,
                (agent_id,),
            ).fetchone()
            if row is None:
                return None
            if row["model"] not in verified_models:
                return None
            if row["source"] == "chat" and row["chat_epoch"] != agent["chat_epoch"]:
                connection.execute(
                    """
                    UPDATE runs SET status = 'cancelled', finished_at = ?,
                        termination_reason = 'new_chat'
                    WHERE id = ? AND status = 'queued'
                    """,
                    (now_ms, row["id"]),
                )
                return None
            connection.execute(
                """
                UPDATE runs SET status = 'starting', started_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now_ms, row["id"]),
            )
            # Reserve before any container/SDK work; failed starts still consume this slot.
            connection.execute(
                "DELETE FROM model_run_starts WHERE started_at <= ?", (now_ms - 86_400_000,)
            )
            connection.execute("INSERT INTO model_run_starts(started_at) VALUES (?)", (now_ms,))
            connection.execute(
                "INSERT INTO runtime_metadata(key,value) VALUES ('last_dispatch_owner', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (agent["owner_id"],),
            )
            updated = connection.execute(
                """
                SELECT runs.*, agents.paused
                FROM runs JOIN agents ON agents.id = runs.agent_id
                WHERE runs.id = ?
                """,
                (row["id"],),
            ).fetchone()
        if updated is None:
            raise RuntimeError("Claimed run row disappeared")
        return RunDispatchContext(
            run=RunRecord.from_row(updated),
            thread_id=updated["thread_id"]
            or (agent["chat_thread_id"] if updated["source"] == "chat" else None),
            generation=agent["generation"],
            blockers=tuple(json.loads(agent["blockers_json"])),
            current_chat_epoch=agent["chat_epoch"],
        )

    def update_run_thread(self, run_id: str, thread_id: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                UPDATE runs SET thread_id = ?
                WHERE id = ? AND status IN ('starting', 'running')
                """,
                (thread_id, run_id),
            )
            connection.execute(
                """
                UPDATE agents SET chat_thread_id = ?
                WHERE id = (
                    SELECT agent_id FROM runs WHERE id = ? AND source = 'chat'
                )
                """,
                (thread_id, run_id),
            )

    def update_run_running(
        self,
        run_id: str,
        *,
        thread_id: str,
        generation: int,
        run_token_hash: str,
    ) -> bool:
        now_ms = time.time_ns() // 1_000_000
        with self.database.transaction() as connection:
            run = connection.execute(
                """
                SELECT runs.agent_id, runs.status, runs.source, runs.chat_epoch, agents.generation,
                       agents.paused, agents.blockers_json,
                       agents.chat_epoch AS current_chat_epoch
                FROM runs JOIN agents ON agents.id = runs.agent_id
                WHERE runs.id = ?
                """,
                (run_id,),
            ).fetchone()
            if (
                run is None
                or run["status"] != "starting"
                or run["generation"] != generation
                or run["paused"]
                or json.loads(run["blockers_json"])
                or (run["source"] == "chat" and run["chat_epoch"] != run["current_chat_epoch"])
            ):
                return False
            cursor = connection.execute(
                """
                UPDATE runs SET status = 'running', thread_id = ?,
                    run_token_hash = ?, agent_generation = ?
                WHERE id = ? AND status = 'starting'
                """,
                (thread_id, run_token_hash, generation, run_id),
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                "UPDATE agents SET chat_thread_id = ? WHERE id = ? AND ? = 'chat'",
                (thread_id, run["agent_id"], run["source"]),
            )
            connection.execute(
                """
                INSERT INTO events(run_id, sequence, kind, payload_json, created_at)
                VALUES (?, (SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE run_id = ?),
                        'state', ?, ?)
                """,
                (
                    run_id,
                    run_id,
                    json.dumps({"status": "running"}, separators=(",", ":"), sort_keys=True),
                    now_ms,
                ),
            )
            return True

    def update_run_turn_id(self, run_id: str, turn_id: str) -> None:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET sdk_turn_id = ?
                WHERE id = ? AND status = 'running'
                """,
                (turn_id, run_id),
            )
            if cursor.rowcount != 1:
                raise RepositoryError(409, "run_not_running", "The run is no longer active.")

    def begin_run_cleanup(
        self,
        run_id: str,
        intended_status: str,
        reason: str,
    ) -> None:
        if intended_status not in {
            "completed",
            "completed_with_errors",
            "failed",
            "interrupted",
            "cancelled",
        }:
            raise ValueError("Invalid intended terminal status")
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status = 'stopping', cleanup_state = 'pending',
                    intended_terminal_status = COALESCE(intended_terminal_status, ?),
                    intended_termination_reason = COALESCE(intended_termination_reason, ?)
                WHERE id = ? AND status IN ('starting', 'running', 'stopping')
                  AND cleanup_state != 'failed'
                """,
                (intended_status, reason, run_id),
            )
            if cursor.rowcount != 1:
                raise RepositoryError(409, "run_not_active", "The run is no longer active.")

    def run_cleanup_intent(self, run_id: str) -> tuple[str | None, str | None]:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT intended_terminal_status, intended_termination_reason
                FROM runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise RepositoryError(404, "not_found", "Run not found.")
        return row["intended_terminal_status"], row["intended_termination_reason"]

    def fail_run_cleanup(self, run_id: str) -> None:
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT runs.agent_id, agents.blockers_json
                FROM runs JOIN agents ON agents.id = runs.agent_id
                WHERE runs.id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise RepositoryError(404, "not_found", "Run not found.")
            blockers = set(json.loads(row["blockers_json"]))
            blockers.add("recovery_error")
            connection.execute(
                """
                UPDATE runs SET status = 'stopping', cleanup_state = 'failed'
                WHERE id = ?
                """,
                (run_id,),
            )
            connection.execute(
                "UPDATE agents SET paused = 1, blockers_json = ? WHERE id = ?",
                (json.dumps(sorted(blockers)), row["agent_id"]),
            )

    def complete_run(
        self,
        run_id: str,
        status: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        had_tool_errors: bool = False,
    ) -> None:
        if status not in {
            "completed",
            "completed_with_errors",
            "failed",
            "interrupted",
            "cancelled",
        }:
            raise ValueError("Invalid terminal run status")
        now_ms = time.time_ns() // 1_000_000
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status = ?, finished_at = ?, error_code = ?,
                    error_message = ?, had_tool_errors = ?, run_token_hash = NULL,
                    cleanup_state = 'confirmed', termination_reason = intended_termination_reason
                WHERE id = ? AND status IN ('starting', 'running', 'stopping')
                """,
                (
                    status,
                    now_ms,
                    error_code,
                    error_message,
                    int(had_tool_errors),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RepositoryError(409, "run_not_active", "The run is no longer active.")

    def queued_agent_ids(self) -> list[str]:
        with self.database.read() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT agent_id FROM runs
                WHERE status = 'queued'
                ORDER BY agent_id
                """
            ).fetchall()
        return [row["agent_id"] for row in rows]

    @staticmethod
    def _canonical_uuid(value: str, name: str) -> str:
        try:
            canonical = str(uuid.UUID(value))
        except (ValueError, TypeError, AttributeError) as error:
            raise RepositoryError(422, "invalid_identifier", f"{name} must be a UUID.") from error
        if canonical != value:
            raise RepositoryError(
                422,
                "invalid_identifier",
                f"{name} must be a canonical UUID.",
            )
        return canonical

    def set_agent_status(
        self,
        agent_id: str,
        status: str,
        *,
        error_blocker: str | None = None,
    ) -> AgentRecord:
        if status not in {"provisioning", "provisioning_error", "ready"}:
            raise ValueError("Invalid assistant preparation status")
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT blockers_json FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if row is None:
                raise RepositoryError(404, "not_found", "Assistant not found.")
            blockers = set(json.loads(row["blockers_json"]))
            if error_blocker is None:
                blockers.discard("provisioning_error")
            else:
                blockers.add(error_blocker)
            connection.execute(
                "UPDATE agents SET status = ?, blockers_json = ? WHERE id = ?",
                (status, json.dumps(sorted(blockers)), agent_id),
            )
            updated = connection.execute(
                "SELECT * FROM agents WHERE id = ?",
                (agent_id,),
            ).fetchone()
        if updated is None:
            raise RuntimeError("Assistant row disappeared after update")
        return AgentRecord.from_row(updated)

    def complete_agent_creation(self, owner_id: str, request_id: str, agent_id: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                UPDATE idempotency SET operation_state = 'completed'
                WHERE actor_kind = 'user' AND actor_id = ? AND operation = 'agent.create'
                  AND resource_id = 'collection:agents' AND request_id = ? AND entity_id = ?
                """,
                (owner_id, request_id, agent_id),
            )

    def fail_agent_creation(self, agent_id: str, error_code: str) -> None:
        with self.database.transaction() as connection:
            exists = connection.execute("SELECT 1 FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if exists is None:
                raise RepositoryError(404, "not_found", "Assistant not found.")
            connection.execute(
                """
                UPDATE agents SET status = 'provisioning_error' WHERE id = ?
                """,
                (agent_id,),
            )
            connection.execute(
                """
                UPDATE idempotency
                SET operation_state = 'failed',
                    response_json = ?
                WHERE entity_id = ? AND operation = 'agent.create'
                """,
                (json.dumps({"code": error_code}), agent_id),
            )

    def prepare_startup_recovery(self) -> list[RunRecord]:
        """Invalidate old authority durably without declaring old processes terminated."""
        with self.database.transaction() as connection:
            agent_ids = [
                row["agent_id"]
                for row in connection.execute(
                    "SELECT DISTINCT agent_id FROM runs "
                    "WHERE status IN ('starting', 'running', 'stopping')"
                )
            ]
            for agent_id in agent_ids:
                row = connection.execute(
                    "SELECT blockers_json, chat_thread_id FROM agents WHERE id = ?",
                    (agent_id,),
                ).fetchone()
                blockers = set(json.loads(row["blockers_json"])) | {"recovery_error"}
                if row["chat_thread_id"] is not None:
                    blockers.add("history_error")
                connection.execute(
                    """UPDATE agents SET paused = 1, generation = generation + 1,
                       blockers_json = ? WHERE id = ?""",
                    (json.dumps(sorted(blockers)), agent_id),
                )
            connection.execute(
                """UPDATE runs SET status = 'stopping', cleanup_state = 'pending',
                   run_token_hash = NULL,
                   intended_terminal_status = COALESCE(intended_terminal_status, 'interrupted'),
                   intended_termination_reason = COALESCE(
                       intended_termination_reason, 'backend_restart')
                   WHERE status IN ('starting', 'running', 'stopping')""",
            )
            rows = connection.execute(
                """SELECT runs.*, agents.paused FROM runs
                   JOIN agents ON agents.id = runs.agent_id
                   WHERE runs.status = 'stopping'
                   ORDER BY runs.agent_id, runs.created_at, runs.id""",
            ).fetchall()
        return [RunRecord.from_row(row) for row in rows]

    def set_recovery_blocker(self, agent_id: str, *, blocked: bool) -> bool:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT blockers_json FROM agents WHERE id = ?",
                (agent_id,),
            ).fetchone()
            if row is None:
                return False
            blockers = set(json.loads(row["blockers_json"]))
            if blocked:
                blockers.add("recovery_error")
            else:
                active = connection.execute(
                    """SELECT 1 FROM runs WHERE agent_id = ?
                       AND status IN ('starting', 'running', 'stopping') LIMIT 1""",
                    (agent_id,),
                ).fetchone()
                if active is not None:
                    return False
                blockers.discard("recovery_error")
            connection.execute(
                "UPDATE agents SET paused = CASE WHEN ? THEN 1 ELSE paused END, "
                "blockers_json = ? WHERE id = ?",
                (int(blocked), json.dumps(sorted(blockers)), agent_id),
            )
        return True

    def complete_recovered_run(self, run_id: str) -> None:
        """Commit the saved outcome only after the recovery service confirms cleanup."""
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """UPDATE runs SET status = intended_terminal_status,
                   termination_reason = intended_termination_reason,
                   cleanup_state = 'confirmed', run_token_hash = NULL, finished_at = ?,
                   error_code = CASE WHEN intended_terminal_status = 'interrupted'
                       THEN COALESCE(error_code, intended_termination_reason) ELSE error_code END
                   WHERE id = ? AND status = 'stopping'
                     AND intended_terminal_status IS NOT NULL""",
                (time.time_ns() // 1_000_000, run_id),
            )
            if cursor.rowcount != 1:
                raise RepositoryError(
                    409, "run_not_active", "The run is no longer awaiting recovery."
                )

    def history_states(self) -> list[dict[str, object]]:
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT agents.id, agents.model, agents.chat_thread_id, agents.blockers_json,
                   EXISTS(SELECT 1 FROM runs WHERE runs.agent_id = agents.id
                       AND runs.source = 'chat' AND runs.chat_epoch = agents.chat_epoch
                       AND (runs.started_at IS NOT NULL OR
                            (runs.thread_id IS NOT NULL AND runs.status != 'queued'))) AS used
                   FROM agents WHERE agents.status = 'ready' ORDER BY agents.id""",
            ).fetchall()
        return [dict(row) for row in rows]

    def set_history_blocker(self, agent_id: str, *, blocked: bool) -> None:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT blockers_json FROM agents WHERE id = ?",
                (agent_id,),
            ).fetchone()
            if row is None:
                raise RepositoryError(404, "not_found", "Assistant not found.")
            blockers = set(json.loads(row["blockers_json"]))
            if blocked:
                blockers.add("history_error")
            else:
                blockers.discard("history_error")
            connection.execute(
                "UPDATE agents SET paused = CASE WHEN ? THEN 1 ELSE paused END, "
                "blockers_json = ? WHERE id = ?",
                (int(blocked), json.dumps(sorted(blockers)), agent_id),
            )

    def new_chat(self, owner_id: str, agent_id: str, request_id: str) -> tuple[AgentRecord, bool]:
        owner_id = self._canonical_uuid(owner_id, "owner_id")
        agent_id = self._canonical_uuid(agent_id, "agent_id")
        request_id = self._canonical_uuid(request_id, "request_id")
        now_ms = time.time_ns() // 1_000_000
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM agents WHERE id = ? AND owner_id = ?",
                (agent_id, owner_id),
            ).fetchone()
            if row is None:
                raise RepositoryError(404, "not_found", "Assistant not found.")
            previous = connection.execute(
                """SELECT 1 FROM idempotency WHERE actor_kind = 'user' AND actor_id = ?
                   AND operation = 'agent.new_chat' AND resource_id = ? AND request_id = ?""",
                (owner_id, agent_id, request_id),
            ).fetchone()
            if previous is not None:
                return AgentRecord.from_row(row), True
            if not row["paused"]:
                raise RepositoryError(
                    409, "pause_required", "Pause the assistant before starting a new chat."
                )
            if self.lifecycle_pending(connection, agent_id):
                raise RepositoryError(
                    409, "control_pending", "An assistant control is in progress."
                )
            active = connection.execute(
                """SELECT 1 FROM runs WHERE agent_id = ? AND
                   (status IN ('starting', 'running', 'stopping')
                    OR cleanup_state IN ('pending', 'failed')) LIMIT 1""",
                (agent_id,),
            ).fetchone()
            if active is not None:
                raise RepositoryError(
                    409,
                    "run_cleanup_pending",
                    "Confirm old run cleanup before starting a new chat.",
                )
            blockers = set(json.loads(row["blockers_json"])) - {"history_error"}
            connection.execute(
                """UPDATE agents SET chat_epoch = chat_epoch + 1, chat_thread_id = NULL,
                   generation = generation + 1, blockers_json = ? WHERE id = ?""",
                (json.dumps(sorted(blockers)), agent_id),
            )
            connection.execute(
                """UPDATE runs SET status = 'cancelled', termination_reason = 'new_chat',
                   finished_at = ? WHERE agent_id = ? AND source = 'chat' AND status = 'queued'""",
                (now_ms, agent_id),
            )
            body_hash = hashlib.sha256(b'{"confirmed":true}').hexdigest()
            connection.execute(
                """INSERT INTO idempotency(actor_kind, actor_id, operation, resource_id,
                   request_id, body_hash, entity_id, operation_state, response_code, created_at)
                   VALUES ('user', ?, 'agent.new_chat', ?, ?, ?, ?, 'completed', 200, ?)""",
                (owner_id, agent_id, request_id, body_hash, agent_id, now_ms),
            )
            updated = connection.execute(
                "SELECT * FROM agents WHERE id = ?",
                (agent_id,),
            ).fetchone()
        return AgentRecord.from_row(updated), False

    def workspace_receipt(self, owner, agent, request_id, operation, body_hash):
        self.get_owned_agent(owner, agent)
        with self.database.read() as connection:
            row = connection.execute(
                """SELECT * FROM idempotency WHERE actor_kind = 'user' AND actor_id = ?
                   AND operation = ? AND resource_id = ? AND request_id = ?""",
                (owner, operation, agent, request_id),
            ).fetchone()
        if row is None:
            return None
        if row["body_hash"] != body_hash:
            raise RepositoryError(
                409, "idempotency_conflict", "This request ID has different content."
            )
        if row["operation_state"] != "completed":
            raise RepositoryError(
                409,
                "file_operation_uncertain",
                "The previous result is uncertain. Inspect the file before a new request.",
            )
        return {**json.loads(row["response_json"]), "replayed": True}

    def check_workspace_mutation(self, owner, agent, *, upload):
        record = self.get_owned_agent(owner, agent)
        with self.database.read() as connection:
            if self.lifecycle_pending(connection, agent):
                raise RepositoryError(
                    409, "control_pending", "An assistant control is in progress."
                )
            active = connection.execute(
                """SELECT 1 FROM runs WHERE agent_id = ? AND
                   (status IN ('starting', 'running', 'stopping')
                    OR cleanup_state IN ('pending', 'failed'))
                   LIMIT 1""",
                (agent,),
            ).fetchone()
            protection = connection.execute(
                "SELECT protection_latched FROM platform_storage_state WHERE id = 1"
            ).fetchone()[0]
        if active is not None or "recovery_error" in record.blockers:
            raise RepositoryError(
                409, "workspace_busy", "The assistant is handling files; please try later."
            )
        if upload and (protection or "agent_storage_full" in record.blockers):
            raise RepositoryError(
                507, "storage_full", "Storage is full. Remove files before uploading."
            )
        if upload and record.status != "ready":
            raise RepositoryError(
                409, "assistant_not_ready", "Prepare the assistant before uploading."
            )

    def reserve_workspace_operation(self, owner, agent, request_id, operation, body_hash, payload):
        self.check_workspace_mutation(owner, agent, upload=operation == "workspace.upload")
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO idempotency(actor_kind, actor_id, operation, resource_id,
                   request_id, body_hash, entity_id, operation_state, response_json, created_at)
                   VALUES ('user', ?, ?, ?, ?, ?, ?, 'accepted', ?, ?)""",
                (
                    owner,
                    operation,
                    agent,
                    request_id,
                    body_hash,
                    agent,
                    json.dumps(payload),
                    time.time_ns() // 1_000_000,
                ),
            )

    def finish_workspace_operation(self, owner, agent, request_id, operation, response=None):
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE idempotency SET operation_state = ?, response_json =
                   COALESCE(?, response_json), response_code = ? WHERE actor_kind = 'user'
                   AND actor_id = ? AND operation = ? AND resource_id = ? AND request_id = ?""",
                (
                    "completed" if response is not None else "interrupted_unknown",
                    json.dumps(response) if response is not None else None,
                    200 if response is not None else 409,
                    owner,
                    operation,
                    agent,
                    request_id,
                ),
            )

    def interrupted_workspace_operations(self):
        with self.database.transaction() as connection:
            rows = connection.execute(
                """SELECT * FROM idempotency
                   WHERE operation IN ('workspace.upload', 'workspace.delete')
                   AND operation_state IN ('accepted', 'interrupted_unknown')""",
            ).fetchall()
            connection.execute(
                """UPDATE idempotency SET operation_state = 'interrupted_unknown',
                   response_code = 409
                   WHERE operation IN ('workspace.upload', 'workspace.delete')
                   AND operation_state = 'accepted'""",
            )
        return [dict(row) for row in rows]

    def set_agent_storage_blocker(self, agent, *, blocked):
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT blockers_json FROM agents WHERE id = ?", (agent,)
            ).fetchone()
            blockers = set(json.loads(row["blockers_json"]))
            if blocked:
                blockers.add("agent_storage_full")
            else:
                blockers.discard("agent_storage_full")
            connection.execute(
                "UPDATE agents SET blockers_json = ?, "
                "paused = CASE WHEN ? THEN 1 ELSE paused END WHERE id = ?",
                (json.dumps(sorted(blockers)), int(blocked), agent),
            )

    def _run_start_usage(self, connection, now):
        starts = [
            row[0]
            for row in connection.execute(
                "SELECT started_at FROM model_run_starts WHERE started_at > ? ORDER BY started_at",
                (now - 86_400_000,),
            ).fetchall()
        ]
        minute = [started for started in starts if started > now - 60_000]
        release = []
        if len(minute) >= self.minute_start_limit:
            release.append(minute[-self.minute_start_limit] + 60_000)
        if len(starts) >= self.day_start_limit:
            release.append(starts[-self.day_start_limit] + 86_400_000)
        return {
            "limited": bool(release),
            "retry_at": max(release) if release else None,
            "minute_count": len(minute),
            "day_count": len(starts),
            "minute_limit": self.minute_start_limit,
            "day_limit": self.day_start_limit,
        }

    def run_start_usage(self):
        with self.database.read() as connection:
            return self._run_start_usage(connection, self.clock())

    def next_runnable_agent(self, verified_models: frozenset[str]) -> str | None:
        if not verified_models or self.database.storage.status()["active"]:
            return None
        with self.database.read() as connection:
            if self._run_start_usage(connection, self.clock())["limited"]:
                return None
            if connection.execute(
                "SELECT protection_latched FROM platform_storage_state WHERE id=1"
            ).fetchone()[0]:
                return None
            # Select only each assistant's FIFO head; ineligible owners do not enter rotation.
            rows = connection.execute(
                """SELECT agents.id, agents.owner_id, agents.model, runs.created_at,
                          runs.model AS run_model
                   FROM agents JOIN runs ON runs.agent_id=agents.id
                   WHERE agents.status='ready' AND agents.paused=0
                     AND json_array_length(agents.blockers_json)=0 AND runs.status='queued'
                     AND NOT EXISTS(SELECT 1 FROM idempotency op WHERE op.resource_id=agents.id
                         AND op.operation LIKE 'agent.control.%' AND op.operation_state='accepted')
                     AND runs.id=(SELECT r.id FROM runs r WHERE r.agent_id=agents.id
                         AND r.status='queued' ORDER BY r.created_at,r.id LIMIT 1)
                     AND NOT EXISTS(SELECT 1 FROM runs r WHERE r.agent_id=agents.id AND
                         (r.status IN ('starting','running','stopping')
                          OR r.cleanup_state IN ('pending','failed')))
                   ORDER BY agents.owner_id,runs.created_at,agents.id""",
            ).fetchall()
            previous = connection.execute(
                "SELECT value FROM runtime_metadata WHERE key='last_dispatch_owner'"
            ).fetchone()
        eligible = [
            row
            for row in rows
            if row["model"] in verified_models and row["run_model"] in verified_models
        ]
        if not eligible:
            return None
        owners = sorted({row["owner_id"] for row in eligible})
        last = previous[0] if previous else ""
        owner = next((identity for identity in owners if identity > last), owners[0])
        return min(
            (row for row in eligible if row["owner_id"] == owner),
            key=lambda row: (row["created_at"], row["id"]),
        )["id"]

    @staticmethod
    def lifecycle_pending(connection, agent_id):
        return (
            connection.execute(
                "SELECT 1 FROM idempotency WHERE resource_id=? AND operation LIKE "
                "'agent.control.%' "
                "AND operation_state='accepted' LIMIT 1",
                (agent_id,),
            ).fetchone()
            is not None
        )

    def agent_unclean_runs(self, owner, agent):
        self.get_owned_agent(owner, agent)
        with self.database.read() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM runs WHERE agent_id=? AND (status IN "
                    "('starting','running','stopping') "
                    "OR cleanup_state IN ('pending','failed'))",
                    (agent,),
                ).fetchall()
            ]

    def agent_control_receipt(self, owner, agent, action, request_id, body_hash):
        self.get_owned_agent(owner, agent)
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT body_hash,operation_state FROM idempotency WHERE actor_kind='user' "
                "AND actor_id=? AND operation=? AND resource_id=? AND request_id=?",
                (owner, "agent.control." + action, agent, request_id),
            ).fetchone()
        if row is None:
            return False
        if row["body_hash"] != body_hash:
            raise RepositoryError(409, "idempotency_conflict", "The request content changed.")
        if row["operation_state"] == "accepted":
            raise RepositoryError(
                409, "control_pending", "The assistant control is still in progress."
            )
        if row["operation_state"] != "completed":
            raise RepositoryError(
                409, "control_uncertain", "Inspect the assistant before a new control request."
            )
        return True

    def reserve_agent_control(self, owner, agent, action, request_id, body_hash):
        self.get_owned_agent(owner, agent)
        with self.database.transaction() as connection:
            previous = connection.execute(
                "SELECT * FROM idempotency WHERE actor_kind='user' AND actor_id=? "
                "AND operation=? AND resource_id=? AND request_id=?",
                (owner, "agent.control." + action, agent, request_id),
            ).fetchone()
            if previous is not None:
                if previous["body_hash"] != body_hash:
                    raise RepositoryError(
                        409, "idempotency_conflict", "The request content changed."
                    )
                if previous["operation_state"] != "completed":
                    raise RepositoryError(
                        409,
                        "control_uncertain",
                        "Inspect the assistant before a new control request.",
                    )
                return True
            record = connection.execute("SELECT * FROM agents WHERE id=?", (agent,)).fetchone()
            if action != "sleep" and record["status"] != "ready":
                raise RepositoryError(
                    409, "prepare_required", "Retry preparation before using this control."
                )
            if self.lifecycle_pending(connection, agent):
                raise RepositoryError(
                    409, "control_pending", "Another assistant control is still in progress."
                )
            if "recovery_error" in json.loads(record["blockers_json"]) and action != "restart":
                raise RepositoryError(
                    409, "restart_required", "Restart is required to repair the failed cleanup."
                )
            active = connection.execute(
                "SELECT 1 FROM runs WHERE agent_id=? AND (status IN "
                "('starting','running','stopping') "
                "OR cleanup_state IN ('pending','failed')) LIMIT 1",
                (agent,),
            ).fetchone()
            if action in {"wake", "reset-model-context"} and active:
                raise RepositoryError(
                    409, "assistant_busy", "Wait for current work and cleanup to finish."
                )
            if action == "reset-model-context" and not record["paused"]:
                raise RepositoryError(
                    409, "pause_required", "Pause the assistant before resetting model context."
                )
            if action in {"sleep", "restart"}:
                reason = "manual_sleep" if action == "sleep" else "manual_restart"
                connection.execute(
                    "UPDATE agents SET paused=1,generation=generation+1 WHERE id=?", (agent,)
                )
                connection.execute(
                    "UPDATE runs SET "
                    "status='stopping',cleanup_state='pending',run_token_hash=NULL, "
                    "intended_terminal_status=COALESCE(intended_terminal_status,'interrupted'), "
                    "intended_termination_reason=COALESCE(intended_termination_reason,?) "
                    "WHERE agent_id=? AND status IN ('starting','running')",
                    (reason, agent),
                )
            if action == "restart":
                connection.execute(
                    "UPDATE runs SET status='stopping',intended_terminal_status=COALESCE("
                    "intended_terminal_status,CASE WHEN status IN "
                    "('completed','completed_with_errors',"
                    "'failed','interrupted','cancelled') THEN status ELSE 'interrupted' END), "
                    "intended_termination_reason=COALESCE("
                    "intended_termination_reason,'manual_restart') "
                    "WHERE agent_id=? AND cleanup_state IN ('pending','failed')",
                    (agent,),
                )
            if action == "reset-model-context":
                blockers = set(json.loads(record["blockers_json"])) | {"context_reset_error"}
                connection.execute(
                    "UPDATE agents SET blockers_json=? WHERE id=?",
                    (json.dumps(sorted(blockers)), agent),
                )
                # A crash during cleanup must never resume the partially deleted old thread.
                connection.execute(
                    "UPDATE agents SET chat_epoch=chat_epoch+1,chat_thread_id=NULL, "
                    "generation=generation+1 WHERE id=?",
                    (agent,),
                )
                connection.execute(
                    "UPDATE runs SET "
                    "status='cancelled',termination_reason='context_reset',finished_at=? "
                    "WHERE agent_id=? AND source='chat' AND status='queued'",
                    (self.clock(), agent),
                )
            connection.execute(
                "INSERT INTO "
                "idempotency(actor_kind,actor_id,operation,resource_id,request_id,body_hash, "
                "entity_id,operation_state,created_at) VALUES ('user',?,?,?,?,?,?,'accepted',?)",
                (
                    owner,
                    "agent.control." + action,
                    agent,
                    request_id,
                    body_hash,
                    agent,
                    self.clock(),
                ),
            )
        return False

    def finish_agent_control(self, owner, agent, action, request_id, *, completed):
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE idempotency SET operation_state=? WHERE actor_kind='user' AND actor_id=? "
                "AND operation=? AND resource_id=? AND request_id=?",
                (
                    "completed" if completed else "interrupted_unknown",
                    owner,
                    "agent.control." + action,
                    agent,
                    request_id,
                ),
            )
            if completed and action == "reset-model-context":
                row = connection.execute(
                    "SELECT blockers_json FROM agents WHERE id=?", (agent,)
                ).fetchone()
                blockers = set(json.loads(row[0])) - {"history_error", "context_reset_error"}
                connection.execute(
                    "UPDATE agents SET blockers_json=? WHERE id=?",
                    (json.dumps(sorted(blockers)), agent),
                )

    def interrupt_agent_controls(self):
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE idempotency SET operation_state='interrupted_unknown' "
                "WHERE operation LIKE 'agent.control.%' AND operation_state='accepted'",
            )

    def idle_agent_ids(self, agent_ids):
        with self.database.read() as connection:
            return [
                agent
                for agent in agent_ids
                if connection.execute(
                    "SELECT 1 FROM agents WHERE id=? AND NOT EXISTS(SELECT 1 FROM "
                    "runs WHERE agent_id=? "
                    "AND (status IN ('queued','starting','running','stopping') OR "
                    "cleanup_state IN ('pending','failed')))",
                    (agent, agent),
                ).fetchone()
                and not self.lifecycle_pending(connection, agent)
            ]

    def export_run_archive(
        self,
        owner,
        root_ids,
        output,
        *,
        max_bytes=32 * 1024**2,
        cancelled=None,
        _inspect=None,
        _connection=None,
    ):
        """Write one canonical read snapshot without granting ordinary DB write authority."""
        roots = sorted(set(self._canonical_uuid(value, "run_id") for value in root_ids))
        if not roots or len(roots) > 100:
            raise RepositoryError(422, "archive_selection", "Select between 1 and 100 run roots.")
        deadline = time.monotonic() + 30
        digest, size = hashlib.sha256(), 0

        def interrupted():
            return time.monotonic() >= deadline or (cancelled is not None and cancelled.is_set())

        def emit(value):
            nonlocal size
            if interrupted():
                raise RepositoryError(408, "archive_timeout", "History export was interrupted.")
            encoded = value.encode("utf-8")
            size += len(encoded)
            if size > max_bytes:
                raise RepositoryError(413, "archive_too_large", "Select a smaller history export.")
            output.write(encoded)
            digest.update(encoded)

        encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        with (
            self.database.read() if _connection is None else nullcontext(_connection)
        ) as connection:
            if _connection is None:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA temp_store=MEMORY")
                connection.execute("BEGIN")
            connection.set_progress_handler(lambda: int(interrupted()), 1000)
            for root in roots:
                row = connection.execute(
                    "SELECT runs.retry_of FROM runs JOIN agents ON agents.id=runs.agent_id "
                    "WHERE runs.id=? AND agents.owner_id=?",
                    (root, owner),
                ).fetchone()
                if row is None:
                    raise RepositoryError(404, "not_found", "Run not found.")
                if row["retry_of"] is not None:
                    raise RepositoryError(
                        409, "archive_root", "Select the original run of a retry tree."
                    )
            placeholders = ",".join("?" for _ in roots)
            tree = (
                "WITH RECURSIVE tree(id) AS (SELECT id FROM runs WHERE id IN ("
                + placeholders
                + ") UNION SELECT runs.id FROM runs JOIN tree ON runs.retry_of=tree.id) "
            )
            rows = connection.execute(
                tree + "SELECT runs.*,agents.owner_id FROM runs "
                "JOIN tree ON runs.id=tree.id JOIN agents "
                "ON agents.id=runs.agent_id ORDER BY runs.id LIMIT 10001",
                roots,
            )
            # Bound tree metadata; event payloads remain row-streamed into the staged artifact.
            runs = rows.fetchall()
            if len(runs) > 10000:
                raise RepositoryError(413, "archive_too_large", "Select a smaller history export.")
            terminal = {"completed", "completed_with_errors", "failed", "interrupted", "cancelled"}
            for run in runs:
                if run["owner_id"] != owner:
                    raise RepositoryError(404, "not_found", "Run not found.")
                if run["status"] not in terminal:
                    raise RepositoryError(409, "archive_active", "Every retry must be terminal.")
                if run["cleanup_state"] in {"pending", "failed"}:
                    raise RepositoryError(
                        409, "archive_cleanup", "Confirm run cleanup before export."
                    )
            if _inspect is not None:
                _inspect(connection, runs)
            emit('{"format":"qpl-run-archive-v1","root_ids":')
            for chunk in encoder.iterencode(roots):
                emit(chunk)
            emit(',"runs":[')
            for index, run in enumerate(runs):
                if index:
                    emit(",")
                record = dict(run)
                for key in ("owner_id", "run_token_hash"):
                    record.pop(key, None)
                emit('{"events":[')
                events = connection.execute(
                    "SELECT sequence,kind,sdk_item_id,payload_json,created_at FROM events "
                    "WHERE run_id=? ORDER BY sequence",
                    (run["id"],),
                )
                for event_index, event in enumerate(events):
                    if event_index:
                        emit(",")
                    item = dict(event)
                    item["payload"] = json.loads(item.pop("payload_json"))
                    for chunk in encoder.iterencode(item):
                        emit(chunk)
                emit('],"run":')
                for chunk in encoder.iterencode(record):
                    emit(chunk)
                emit("}")
            emit("]}\n")
        output.flush()
        output.seek(0)
        return digest.hexdigest(), size

    def list_archive_roots(self, owner, agent, *, cursor=None, limit=20):
        self.get_owned_agent(owner, agent)
        if not 1 <= limit <= 100:
            raise RepositoryError(422, "invalid_limit", "limit must be between 1 and 100.")
        with self.database.read() as connection:
            anchor = None
            if cursor is not None:
                cursor = self._canonical_uuid(cursor, "cursor")
                anchor = connection.execute(
                    "SELECT created_at,id FROM runs WHERE id=? AND agent_id=?",
                    (cursor, agent),
                ).fetchone()
                if anchor is None:
                    raise RepositoryError(
                        409, "archive_cursor", "History changed; return to the first page."
                    )
            rows = connection.execute(
                "SELECT runs.*,agents.paused FROM runs JOIN agents ON agents.id=runs.agent_id "
                "WHERE agents.owner_id=? AND runs.agent_id=? AND runs.retry_of IS NULL "
                "AND runs.status IN ('completed','completed_with_errors',"
                "'failed','interrupted','cancelled') "
                "AND runs.cleanup_state IN ('not_required','confirmed') "
                + ("AND (runs.created_at,runs.id)<(?,?) " if anchor else "")
                + "ORDER BY runs.created_at DESC,runs.id DESC LIMIT ?",
                (
                    owner,
                    agent,
                    *((anchor["created_at"], anchor["id"]) if anchor else ()),
                    limit + 1,
                ),
            ).fetchall()
        page = [RunRecord.from_row(row) for row in rows[:limit]]
        return page, page[-1].id if len(rows) > limit else None

    def check_archive_purge(
        self,
        owner,
        root_ids,
        expected_digest,
        *,
        cancelled=None,
        _connection=None,
        _selected=None,
    ):
        """Informational snapshot only; future deletion must revalidate every condition."""
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(value not in "0123456789abcdef" for value in expected_digest)
        ):
            raise RepositoryError(422, "archive_digest", "Provide the downloaded SHA-256 digest.")
        blockers = set()
        summary = {"run_count": 0, "event_count": 0, "agent_ids": []}

        class DigestOnly:
            def write(self, data):
                return len(data)

            def flush(self):
                pass

            def seek(self, offset):
                pass

        def inspect(connection, runs):
            selected = {run["id"] for run in runs}
            if _selected is not None:
                _selected.extend(sorted(selected))
            agents = sorted({run["agent_id"] for run in runs})
            summary["run_count"] = len(runs)
            summary["agent_ids"] = agents
            for run in runs:
                summary["event_count"] += connection.execute(
                    "SELECT COUNT(*) FROM events WHERE run_id=?",
                    (run["id"],),
                ).fetchone()[0]
            for agent in agents:
                row = connection.execute(
                    "SELECT paused,chat_thread_id,chat_epoch FROM agents WHERE id=?",
                    (agent,),
                ).fetchone()
                if not row["paused"]:
                    blockers.add("pause_required")
                if self.lifecycle_pending(connection, agent):
                    blockers.add("control_pending")
                if connection.execute(
                    "SELECT 1 FROM runs WHERE agent_id=? AND "
                    "(status IN ('starting','running','stopping') OR "
                    "cleanup_state IN ('pending','failed')) LIMIT 1",
                    (agent,),
                ).fetchone():
                    blockers.add("run_cleanup_pending")
                threads = {
                    run["thread_id"]
                    for run in runs
                    if run["agent_id"] == agent and run["thread_id"] is not None
                }
                if row["chat_thread_id"] is not None and (
                    row["chat_thread_id"] in threads
                    or any(
                        run["agent_id"] == agent
                        and run["source"] == "chat"
                        and run["chat_epoch"] == row["chat_epoch"]
                        for run in runs
                    )
                ):
                    blockers.add("current_model_context")
                for reference in connection.execute(
                    "SELECT id,thread_id FROM runs WHERE agent_id=? AND thread_id IS NOT NULL",
                    (agent,),
                ):
                    if reference["id"] not in selected and reference["thread_id"] in threads:
                        blockers.add("unselected_thread_reference")

        digest, size = self.export_run_archive(
            owner,
            root_ids,
            DigestOnly(),
            cancelled=cancelled,
            _inspect=inspect,
            _connection=_connection,
        )
        if digest != expected_digest:
            raise RepositoryError(
                409,
                "archive_changed",
                "History changed; download and inspect a fresh export.",
            )
        if self.database.storage.status()["active"]:
            blockers.add("storage_repair_required")
        return {
            **summary,
            "sha256": digest,
            "export_bytes": size,
            "eligible": not blockers,
            "blockers": sorted(blockers),
            "deletion_available": False,
        }

    @staticmethod
    def _archive_purge_receipt(connection, owner, request_id, body_hash):
        receipt = connection.execute(
            "SELECT body_hash,response_json FROM idempotency WHERE actor_kind='user' "
            "AND actor_id=? AND operation='archive.purge' "
            "AND resource_id='collection:run-archives' AND request_id=?",
            (owner, request_id),
        ).fetchone()
        if receipt is None:
            return None
        if receipt["body_hash"] != body_hash:
            raise RepositoryError(409, "idempotency_conflict", "The purge request content changed.")
        return {**json.loads(receipt["response_json"]), "replayed": True}

    def purge_archive_records(
        self, owner, root_ids, digest, request_id, *, confirmed=False, cancelled=None
    ):
        """Internal transaction engine; maintenance/physical admission is not yet exposed."""
        if confirmed is not True:
            raise RepositoryError(
                422, "confirmation_required", "Confirm permanent history deletion."
            )
        owner = self._canonical_uuid(owner, "owner_id")
        request_id = self._canonical_uuid(request_id, "request_id")
        roots = sorted({self._canonical_uuid(value, "run_id") for value in root_ids})
        body_hash = hashlib.sha256(
            json.dumps(
                {"root_ids": roots, "sha256": digest},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.database.read() as connection:
            replay = self._archive_purge_receipt(connection, owner, request_id, body_hash)
        if replay is not None:
            return replay
        with self.database.transaction() as connection:
            replay = self._archive_purge_receipt(connection, owner, request_id, body_hash)
            if replay is not None:
                return replay
            selected = []
            try:
                plan = self.check_archive_purge(
                    owner,
                    roots,
                    digest,
                    cancelled=cancelled,
                    _connection=connection,
                    _selected=selected,
                )
                if not plan["eligible"]:
                    raise RepositoryError(
                        409,
                        "archive_dependencies",
                        "Resolve archive blockers before deleting history.",
                    )
                # Defer self-references until the entire retry tree is removed in this transaction.
                connection.execute("PRAGMA defer_foreign_keys=ON")
                for run_id in selected:
                    if cancelled is not None and cancelled.is_set():
                        raise RepositoryError(
                            408, "archive_cancelled", "History purge was interrupted."
                        )
                    connection.execute(
                        "UPDATE idempotency SET operation_state='purged' WHERE entity_id=? "
                        "AND operation IN ('chat.create','run.retry')",
                        (run_id,),
                    )
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
                result = {
                    "root_ids": roots,
                    "agent_ids": plan["agent_ids"],
                    "sha256": digest,
                    "run_count": plan["run_count"],
                    "event_count": plan["event_count"],
                    "export_bytes": plan["export_bytes"],
                    "purged": True,
                    "replayed": False,
                    "reclaimed_bytes": 0,
                    "reclamation_pending": True,
                    "reclamation_state": "pending",
                }
                connection.execute(
                    "INSERT INTO idempotency(actor_kind,actor_id,operation,resource_id,request_id,"
                    "body_hash,operation_state,response_code,response_json,created_at) "
                    "VALUES ('user',?,'archive.purge','collection:run-archives',?,?,"
                    "'completed',200,?,?)",
                    (
                        owner,
                        request_id,
                        body_hash,
                        json.dumps(result, sort_keys=True),
                        self.clock(),
                    ),
                )
                # Cancellation before commit cannot leave deleted rows without their receipt.
                if cancelled is not None and cancelled.is_set():
                    raise RepositoryError(
                        408, "archive_cancelled", "History purge was interrupted."
                    )
            except sqlite3.OperationalError as error:
                # Expected progress-handler cancellation must not become a storage failure.
                if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
                    raise RepositoryError(
                        408,
                        "archive_cancelled",
                        "History purge was interrupted.",
                    ) from None
                raise
            finally:
                connection.set_progress_handler(None, 0)
        return result


    def archive_purge_result(self, owner, root_ids, digest, request_id):
        owner = self._canonical_uuid(owner, "owner_id")
        request_id = self._canonical_uuid(request_id, "request_id")
        roots = sorted({self._canonical_uuid(value, "run_id") for value in root_ids})
        body_hash = hashlib.sha256(
            json.dumps(
                {"root_ids": roots, "sha256": digest},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.database.read() as connection:
            return self._archive_purge_receipt(connection, owner, request_id, body_hash)

    def record_archive_reclamation(self, owner, request_id, phase, *, measurement=None, error=None):
        owner = self._canonical_uuid(owner, "owner_id")
        request_id = self._canonical_uuid(request_id, "request_id")
        if phase not in {"running", "completed", "pending"}:
            raise ValueError("Invalid reclamation phase.")
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT response_json FROM idempotency WHERE actor_kind='user' AND actor_id=? "
                "AND operation='archive.purge' AND resource_id='collection:run-archives' "
                "AND request_id=?",
                (owner, request_id),
            ).fetchone()
            if row is None:
                raise RepositoryError(404, "not_found", "Archive cleanup receipt not found.")
            result = json.loads(row[0])
            if not result["reclamation_pending"]:
                return result
            if phase == "running" and result.get("reclamation_state") == "running":
                result["previous_reclamation_unconfirmed"] = True
            result.update(
                reclamation_state=phase,
                maintenance_error=error,
                reclamation_pending=phase != "completed",
            )
            if phase == "completed":
                result["reclamation_measurement"] = measurement
                result["reclaimed_bytes"] = measurement["reclaimed_bytes"]
                result["reclamation_scope"] = "database_files_compaction_interval"
            connection.execute(
                "UPDATE idempotency SET response_json=? WHERE actor_kind='user' AND actor_id=? "
                "AND operation='archive.purge' AND resource_id='collection:run-archives' "
                "AND request_id=?",
                (json.dumps(result, sort_keys=True), owner, request_id),
            )
        return result

    def require_quiescent_archive_maintenance(self):
        with self.database.read() as connection:
            if connection.execute(
                "SELECT 1 FROM runs WHERE status IN ('starting','running','stopping') "
                "OR cleanup_state IN ('pending','failed') LIMIT 1",
            ).fetchone():
                raise RepositoryError(
                    409, "maintenance_busy", "Wait for active work and cleanup before maintenance."
                )
