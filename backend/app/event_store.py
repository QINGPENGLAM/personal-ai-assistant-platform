from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from .db import Database

EventKind = Literal[
    "message_delta",
    "message_snapshot",
    "activity_started",
    "activity_completed",
    "error",
    "state",
]
EVENT_KINDS = frozenset(
    {
        "message_delta",
        "message_snapshot",
        "activity_started",
        "activity_completed",
        "error",
        "state",
    }
)
MAX_EVENTS_PER_RUN = 20_000
MAX_EVENT_BYTES_PER_RUN = 4 * 1024 * 1024
RESERVED_EVENTS = 16
RESERVED_EVENT_BYTES = 16 * 1024
MAX_BUFFERED_DELTA_BYTES = 16 * 1024
DELTA_FLUSH_SECONDS = 0.05
MAX_PAGE_SIZE = 100
TRUNCATION_MESSAGE = "Displayed output was truncated by the platform budget."


class EventStoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    kind: EventKind
    sdk_item_id: str | None
    payload: dict[str, object]
    created_at: int


@dataclass(slots=True)
class _RunUsage:
    event_count: int
    payload_bytes: int
    last_sequence: int
    truncated: bool
    active_items: set[str]


class EventStore:
    def __init__(self, database: Database) -> None:
        self.database = database
        self._usage: dict[str, _RunUsage] = {}
        self._delta_buffers: dict[str, dict[str, list[str]]] = {}
        self._delta_bytes: dict[str, int] = {}
        self._flush_handles: dict[str, asyncio.TimerHandle] = {}

    def drop_pending(self) -> None:
        for handle in self._flush_handles.values():
            handle.cancel()
        self._flush_handles.clear()
        self._delta_buffers.clear()
        self._delta_bytes.clear()

    def append(
        self,
        run_id: str,
        kind: EventKind,
        payload: dict[str, object],
        *,
        sdk_item_id: str | None = None,
        terminal: bool = False,
    ) -> RunEvent | None:
        run_id = self._canonical_uuid(run_id, "run_id")
        self.flush_deltas(run_id)
        return self._append(run_id, kind, payload, sdk_item_id, terminal=terminal)

    def append_delta(self, run_id: str, item_id: str, text: str) -> None:
        self.database.storage.require_writable()
        run_id = self._canonical_uuid(run_id, "run_id")
        if not item_id or not text:
            return
        loop = asyncio.get_running_loop()
        buffers = self._delta_buffers.setdefault(run_id, {})
        buffers.setdefault(item_id, []).append(text)
        byte_count = self._delta_bytes.get(run_id, 0) + len(text.encode("utf-8"))
        self._delta_bytes[run_id] = byte_count
        if byte_count >= MAX_BUFFERED_DELTA_BYTES:
            self.flush_deltas(run_id)
            return
        if run_id not in self._flush_handles:
            self._flush_handles[run_id] = loop.call_later(
                DELTA_FLUSH_SECONDS,
                self.flush_deltas,
                run_id,
            )

    def flush_deltas(self, run_id: str) -> None:
        handle = self._flush_handles.pop(run_id, None)
        if handle is not None:
            handle.cancel()
        buffers = self._delta_buffers.pop(run_id, {})
        self._delta_bytes.pop(run_id, None)
        for item_id, chunks in buffers.items():
            self._append(
                run_id,
                "message_delta",
                {"item_id": item_id, "text": "".join(chunks)},
                item_id,
                terminal=False,
            )

    def finalize(self, run_id: str, payload: dict[str, object]) -> RunEvent | None:
        run_id = self._canonical_uuid(run_id, "run_id")
        self.flush_deltas(run_id)
        usage = self._get_usage(run_id)
        if usage.truncated:
            payload = {
                "reason": "output_truncated",
                "terminal_summary": True,
                "incomplete_activity_count": len(usage.active_items),
                "activities_closed": True,
            }
        return self._append(run_id, "state", payload, None, terminal=True)

    def list_after(self, run_id: str, after: int = 0, limit: int = 50) -> list[RunEvent]:
        run_id = self._canonical_uuid(run_id, "run_id")
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise EventStoreError("after must be a nonnegative event sequence")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE:
            raise EventStoreError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        with self.database.read() as connection:
            rows = connection.execute(
                """
                SELECT run_id, sequence, kind, sdk_item_id, payload_json, created_at
                FROM events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (run_id, after, limit),
            ).fetchall()
        return [
            RunEvent(
                run_id=row["run_id"],
                sequence=row["sequence"],
                kind=row["kind"],
                sdk_item_id=row["sdk_item_id"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _append(
        self,
        run_id: str,
        kind: EventKind,
        payload: dict[str, object],
        sdk_item_id: str | None,
        *,
        terminal: bool,
    ) -> RunEvent | None:
        self.database.storage.require_writable()
        if kind not in EVENT_KINDS:
            raise EventStoreError(f"Unsupported event kind: {kind}")
        if terminal and kind not in {"error", "state"}:
            raise EventStoreError("Only error and state events may use the terminal budget")
        payload_json = self._encode_payload(payload)
        payload_size = len(payload_json.encode("utf-8"))
        usage = self._get_usage(run_id)
        if usage.truncated and not terminal:
            return None
        if self._is_duplicate(run_id, kind, payload_json, sdk_item_id):
            return None

        if not terminal and (
            usage.event_count >= MAX_EVENTS_PER_RUN - RESERVED_EVENTS
            or usage.payload_bytes + payload_size
            > MAX_EVENT_BYTES_PER_RUN - RESERVED_EVENT_BYTES
        ):
            self._append_truncation(run_id, usage)
            return None
        if (
            usage.event_count >= MAX_EVENTS_PER_RUN
            or usage.payload_bytes + payload_size > MAX_EVENT_BYTES_PER_RUN
        ):
            raise EventStoreError("The reserved event budget is exhausted")

        now_ms = time.time_ns() // 1_000_000
        sequence = usage.last_sequence + 1
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO events(run_id, sequence, kind, sdk_item_id, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, sequence, kind, sdk_item_id, payload_json, now_ms),
            )
        usage.event_count += 1
        usage.payload_bytes += payload_size
        usage.last_sequence = sequence
        if kind == "activity_started" and sdk_item_id is not None:
            usage.active_items.add(sdk_item_id)
        elif kind == "activity_completed" and sdk_item_id is not None:
            usage.active_items.discard(sdk_item_id)
        return RunEvent(run_id, sequence, kind, sdk_item_id, payload, now_ms)

    def _append_truncation(self, run_id: str, usage: _RunUsage) -> None:
        usage.truncated = True
        self._append(
            run_id,
            "state",
            {
                "reason": "output_truncated",
                "message": TRUNCATION_MESSAGE,
                "incomplete_activity_count": len(usage.active_items),
            },
            None,
            terminal=True,
        )

    def _get_usage(self, run_id: str) -> _RunUsage:
        cached = self._usage.get(run_id)
        if cached is not None:
            return cached
        with self.database.read() as connection:
            rows = connection.execute(
                """
                SELECT sequence, kind, sdk_item_id, payload_json
                FROM events WHERE run_id = ? ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
        usage = _RunUsage(
            event_count=len(rows),
            payload_bytes=sum(len(row["payload_json"].encode("utf-8")) for row in rows),
            last_sequence=rows[-1]["sequence"] if rows else 0,
            truncated=False,
            active_items=set(),
        )
        for row in rows:
            payload = json.loads(row["payload_json"])
            if row["kind"] == "activity_started" and row["sdk_item_id"] is not None:
                usage.active_items.add(row["sdk_item_id"])
            elif row["kind"] == "activity_completed" and row["sdk_item_id"] is not None:
                usage.active_items.discard(row["sdk_item_id"])
            if row["kind"] == "state" and payload.get("reason") == "output_truncated":
                usage.truncated = True
        self._usage[run_id] = usage
        return usage

    def _is_duplicate(
        self,
        run_id: str,
        kind: EventKind,
        payload_json: str,
        sdk_item_id: str | None,
    ) -> bool:
        if sdk_item_id is None or kind not in {
            "message_snapshot",
            "activity_started",
            "activity_completed",
        }:
            return False
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM events
                WHERE run_id = ? AND sdk_item_id = ? AND kind = ? AND payload_json = ?
                LIMIT 1
                """,
                (run_id, sdk_item_id, kind, payload_json),
            ).fetchone()
        return row is not None

    @staticmethod
    def _encode_payload(payload: dict[str, object]) -> str:
        if not isinstance(payload, dict):
            raise EventStoreError("Event payload must be an object")
        try:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise EventStoreError("Event payload must contain JSON values") from error

    @staticmethod
    def _canonical_uuid(value: str, name: str) -> str:
        try:
            canonical = str(UUID(value))
        except (ValueError, TypeError, AttributeError) as error:
            raise EventStoreError(f"{name} must be a UUID") from error
        if canonical != value:
            raise EventStoreError(f"{name} must be a canonical UUID")
        return canonical
