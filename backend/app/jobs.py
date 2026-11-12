from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, ValidationError, field_validator

from .coordination import Coordination
from .repository import Repository, RepositoryError
from .schemas import StrictModel

NONTERMINAL = ("queued", "starting", "running", "stopping")
OFFSET_TIME = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?"
    r"[+-](?:(?:0\d|1[0-3]):[0-5]\d|14:00)$"
)


def iso(value):
    return datetime.fromtimestamp(value / 1000, UTC).isoformat() if value is not None else None


class JobInput(StrictModel):
    name: str = Field(min_length=1, max_length=80, strict=True)
    prompt: str = Field(min_length=1, max_length=16 * 1024, strict=True)
    run_at: str = Field(strict=True)
    interval_seconds: int | None = Field(default=None, ge=60, le=31536000, strict=True)

    @field_validator("name", "prompt")
    @classmethod
    def nonempty(cls, value):
        if not value.strip() or len(value.encode()) > 16 * 1024:
            raise ValueError("Use nonempty text within the size limit.")
        return value

    @field_validator("run_at")
    @classmethod
    def explicit_offset(cls, value):
        if not OFFSET_TIME.fullmatch(value) or value.endswith("-00:00"):
            raise ValueError("Use a full timestamp with an explicit numeric UTC offset.")
        datetime.fromisoformat(value)
        return value


@dataclass(frozen=True)
class Actor:
    kind: Literal["user", "agent"]
    id: str
    # Agent callers supply immutable token/run validation; it runs inside each DB transaction.
    validate: Callable | None = None


class JobService:
    def __init__(self, repository: Repository, coordination: Coordination, *, clock=None):
        self.repository = repository
        self.coordination = coordination
        self.clock = clock or (lambda: time.time_ns() // 1_000_000)

    def authorize(self, connection, actor, agent):
        row = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
        if row is None or (actor.kind == "user" and row["owner_id"] != actor.id):
            raise RepositoryError(404, "not_found", "Assistant not found.")
        if actor.kind == "agent":
            if actor.id != agent or actor.validate is None:
                raise RepositoryError(403, "tool_unauthorized", "The tool token is not authorized.")
            actor.validate(connection)
        elif actor.kind != "user":
            raise RepositoryError(403, "tool_unauthorized", "The caller is not authorized.")
        return row

    @staticmethod
    def cursor(value):
        if value is None:
            return None
        try:
            if not isinstance(value, str) or len(value) > 256:
                raise ValueError
            timestamp, identity = json.loads(base64.urlsafe_b64decode(value))
            if (
                type(timestamp) is not int
                or not 0 <= timestamp < 2**63
                or not isinstance(identity, str)
                or str(UUID(identity)) != identity
            ):
                raise ValueError
            return timestamp, identity
        except (ValueError, TypeError, UnicodeError):
            raise RepositoryError(422, "invalid_cursor", "The page cursor is invalid.") from None

    @staticmethod
    def page(rows, limit, serializer):
        more = len(rows) > limit
        selected = rows[:limit]
        cursor = None
        if more:
            last = selected[-1]
            cursor = base64.urlsafe_b64encode(
                json.dumps([last["created_at"], last["id"]]).encode()
            ).decode()
        return {"items": [serializer(row) for row in selected], "next_cursor": cursor}

    @staticmethod
    def pagination(cursor, limit):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise RepositoryError(422, "invalid_limit", "Use a page size between 1 and 100.")
        key = JobService.cursor(cursor)
        return (" AND (created_at, id) > (?, ?)" if key else ""), (key or ())

    def serialize(self, connection, row):
        active = connection.execute(
            "SELECT id FROM runs WHERE job_id = ? AND status IN ('starting','running','stopping')",
            (row["id"],),
        ).fetchone()
        return {
            "job_id": row["id"],
            "agent_id": row["agent_id"],
            "name": row["name"],
            "prompt": row["prompt"],
            "created_by": row["created_by"],
            "run_at": iso(row["run_at"]),
            "interval_seconds": row["interval_seconds"],
            "next_run_at": iso(row["next_run_at"]),
            "enabled": bool(row["enabled"]),
            "cancelled_at": iso(row["cancelled_at"]),
            "created_at": iso(row["created_at"]),
            "active_run_id": active["id"] if active else None,
            "lateness_ms": max(0, self.clock() - row["next_run_at"]) if row["next_run_at"] else 0,
        }

    def list(self, actor, agent, *, cursor=None, limit=20):
        clause, parameters = self.pagination(cursor, limit)
        with self.repository.database.read() as connection:
            self.authorize(connection, actor, agent)
            rows = connection.execute(
                f"SELECT * FROM jobs WHERE agent_id = ?{clause} ORDER BY created_at, id LIMIT ?",
                (agent, *parameters, limit + 1),
            ).fetchall()
            return self.page(rows, limit, lambda row: self.serialize(connection, row))

    @staticmethod
    def receipt(connection, actor, operation, resource, request_id, body_hash):
        row = connection.execute(
            """SELECT * FROM idempotency WHERE actor_kind = ? AND actor_id = ?
               AND operation = ? AND resource_id = ? AND request_id = ?""",
            (actor.kind, actor.id, operation, resource, request_id),
        ).fetchone()
        if row and row["body_hash"] != body_hash:
            raise RepositoryError(
                409, "idempotency_conflict", "This request ID has different content."
            )
        return row

    def save_receipt(self, connection, actor, operation, resource, request_id, body_hash, entity):
        connection.execute(
            """INSERT INTO idempotency(actor_kind, actor_id, operation, resource_id, request_id,
               body_hash, entity_id, operation_state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?,
               'completed', ?)""",
            (
                actor.kind,
                actor.id,
                operation,
                resource,
                request_id,
                body_hash,
                entity,
                self.clock(),
            ),
        )

    async def create(self, actor, agent, request_id, body):
        try:
            parsed = JobInput.model_validate(body)
            request_id = str(UUID(request_id))
            run_at = int(datetime.fromisoformat(parsed.run_at).timestamp() * 1000)
        except (ValidationError, ValueError, TypeError, OverflowError):
            raise RepositoryError(
                422, "invalid_job", "Check the job text, time and interval."
            ) from None
        canonical = parsed.model_dump()
        canonical["run_at"] = run_at
        body_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        async with self.coordination.agent_locks([agent]):
            with self.repository.database.transaction() as connection:
                self.authorize(connection, actor, agent)
                previous = self.receipt(
                    connection, actor, "job.create", agent, request_id, body_hash
                )
                if previous:
                    row = connection.execute(
                        "SELECT * FROM jobs WHERE id = ?", (previous["entity_id"],)
                    ).fetchone()
                    return self.serialize(connection, row)
                now = self.clock()
                if not now <= run_at <= now + 365 * 86400 * 1000:
                    raise RepositoryError(422, "invalid_time", "Schedule within the next 365 days.")
                if connection.execute(
                    "SELECT protection_latched FROM platform_storage_state WHERE id=1"
                ).fetchone()[0]:
                    raise RepositoryError(507, "storage_full", "Platform storage needs attention.")
                count = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE agent_id = ? AND enabled = 1", (agent,)
                ).fetchone()[0]
                if count >= 50:
                    raise RepositoryError(
                        429, "job_limit", "This assistant already has 50 enabled jobs."
                    )
                job_id = str(uuid4())
                connection.execute(
                    """INSERT INTO jobs(id, agent_id, name, prompt, created_by, run_at,
                       interval_seconds, next_run_at, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        job_id,
                        agent,
                        parsed.name,
                        parsed.prompt,
                        actor.kind,
                        run_at,
                        parsed.interval_seconds,
                        run_at,
                        now,
                    ),
                )
                self.save_receipt(
                    connection, actor, "job.create", agent, request_id, body_hash, job_id
                )
                row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                return self.serialize(connection, row)

    def owned_job(self, connection, actor, job_id):
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or (actor.kind == "agent" and row["agent_id"] != actor.id):
            raise RepositoryError(404, "not_found", "Scheduled task not found.")
        self.authorize(connection, actor, row["agent_id"])
        return row

    async def cancel(self, actor, job_id, request_id):
        with self.repository.database.read() as connection:
            job = self.owned_job(connection, actor, job_id)
        async with self.coordination.agent_locks([job["agent_id"]]):
            with self.repository.database.transaction() as connection:
                self.owned_job(connection, actor, job_id)
                request_id = str(UUID(request_id))
                body_hash = hashlib.sha256(b"cancel").hexdigest()
                previous = self.receipt(
                    connection, actor, "job.cancel", job_id, request_id, body_hash
                )
                if previous is None:
                    now = self.clock()
                    connection.execute(
                        """UPDATE jobs SET enabled = 0, next_run_at = NULL, cancelled_at =
                           COALESCE(cancelled_at, ?) WHERE id = ?""",
                        (now, job_id),
                    )
                    connection.execute(
                        """UPDATE runs SET status = 'cancelled', finished_at = ?,
                           termination_reason = 'job_cancel'
                           WHERE job_id = ? AND status = 'queued'""",
                        (now, job_id),
                    )
                    self.save_receipt(
                        connection, actor, "job.cancel", job_id, request_id, body_hash, job_id
                    )
                row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                result = self.serialize(connection, row)
                return {
                    key: result[key]
                    for key in ("job_id", "enabled", "cancelled_at", "active_run_id")
                }

    def runs(self, actor, job_id, *, cursor=None, limit=50):
        clause, parameters = self.pagination(cursor, limit)
        clause = clause.replace("created_at", "runs.created_at").replace(", id)", ", runs.id)")
        with self.repository.database.read() as connection:
            self.owned_job(connection, actor, job_id)
            rows = connection.execute(
                f"""SELECT runs.*, agents.paused FROM runs JOIN agents ON agents.id = runs.agent_id
                    WHERE job_id = ?{clause} ORDER BY runs.created_at, runs.id LIMIT ?""",
                (job_id, *parameters, limit + 1),
            ).fetchall()

            def serialize(row):
                result = dict(row)
                for key in (
                    "run_token_hash",
                    "agent_generation",
                    "intended_terminal_status",
                    "intended_termination_reason",
                ):
                    result.pop(key, None)
                for key in ("scheduled_for", "created_at", "started_at", "finished_at"):
                    result[key] = iso(result[key])
                result["paused"] = bool(result["paused"])
                return result

            return self.page(rows, limit, serialize)
