from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

import backend.app.event_store as event_store_module
from backend.app.coordination import Coordination, CoordinationError, RuntimeProcessLock
from backend.app.db import Database
from backend.app.event_store import EventStore, EventStoreError
from backend.app.repository import Repository, RepositoryError
from backend.app.runner import Runner


class FakeContainers:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.tokens: dict[str, str] = {}
        self.fail_stop = False

    async def start_agent(self, agent_id: str) -> None:
        self.started.append(agent_id)

    async def stop_agent(self, agent_id: str) -> None:
        self.stopped.append(agent_id)
        if self.fail_stop:
            raise RuntimeError("container stop failed")

    def write_run_token(self, agent_id: str, token: str) -> None:
        self.tokens[agent_id] = token

    def remove_run_token(self, agent_id: str) -> None:
        self.tokens.pop(agent_id, None)


class FakeNetwork:
    def __init__(self) -> None:
        self.revoked: list[str] = []

    def issue_run_token(self, run_id: str, agent_id: str, generation: int) -> str:
        return f"token-{run_id}-{agent_id}-{generation}"

    def revoke_run(self, run_id: str) -> None:
        self.revoked.append(run_id)

    async def wait_run_closed(self, run_id: str) -> None:
        return None

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()


class FakeTurn:
    id = "turn-1"

    def __init__(self, status: str | None = "completed") -> None:
        self.status = status
        self.interrupted = False

    async def stream(self) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(method="item/agentMessage/delta", payload={"delta": "Hello"})
        if self.status is not None:
            yield SimpleNamespace(
                method="turn/completed",
                payload={"turn": {"status": self.status}},
            )

    async def interrupt(self) -> SimpleNamespace:
        self.interrupted = True
        return SimpleNamespace()


class FakeSession:
    def __init__(self, turn_status: str | None = "completed") -> None:
        self.closed = False
        self.prepared_thread_id: str | None = None
        self.turn_status = turn_status
        self.turn: FakeTurn | None = None

    async def prepare_thread(self, model: str, thread_id: str | None = None) -> SimpleNamespace:
        self.prepared_thread_id = thread_id or "thread-1"
        return SimpleNamespace(id=self.prepared_thread_id)

    async def start_turn(
        self,
        thread: SimpleNamespace,
        model: str,
        prompt: str,
    ) -> FakeTurn:
        if self.turn is None:
            self.turn = FakeTurn(self.turn_status)
        return self.turn

    async def close(self) -> None:
        self.closed = True


class FakeCodex:
    def __init__(self, turn_status: str | None = "completed") -> None:
        self.session = FakeSession(turn_status)

    async def connect(self, agent_id: str) -> FakeSession:
        return self.session


def _create_run(database: Database) -> str:
    user_id = str(uuid.uuid4())
    agent_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    now_ms = time.time_ns() // 1_000_000
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO users(id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, f"{user_id}@example.test", "test-hash", now_ms),
        )
        connection.execute(
            """
            INSERT INTO agents(id, owner_id, name, model, status, created_at)
            VALUES (?, ?, 'Test assistant', 'gpt-6-luna', 'ready', ?)
            """,
            (agent_id, user_id, now_ms),
        )
        connection.execute(
            """
            INSERT INTO runs(id, agent_id, source, prompt, status, model, created_at)
            VALUES (?, ?, 'chat', 'test prompt', 'queued', 'gpt-6-luna', ?)
            """,
            (run_id, agent_id, now_ms),
        )
    return run_id


def test_retry_chat_run_preserves_thread_and_request_idempotency(database: Database) -> None:
    run_id = _create_run(database)
    with database.transaction() as connection:
        identity = connection.execute(
            """
            SELECT agents.owner_id, runs.agent_id
            FROM runs JOIN agents ON agents.id = runs.agent_id
            WHERE runs.id = ?
            """,
            (run_id,),
        ).fetchone()
        connection.execute(
            "UPDATE runs SET status = 'failed', thread_id = 'thread-original' WHERE id = ?",
            (run_id,),
        )
        connection.execute(
            "UPDATE agents SET chat_thread_id = 'thread-original' WHERE id = ?",
            (identity["agent_id"],),
        )

    repository = Repository(database)
    request_id = str(uuid.uuid4())
    retried, replayed, context_warning = repository.retry_chat_run(
        identity["owner_id"],
        run_id,
        request_id,
    )
    replay, replayed_again, warning_again = repository.retry_chat_run(
        identity["owner_id"],
        run_id,
        request_id,
    )

    assert retried.status == "queued"
    assert retried.retry_of == run_id
    assert retried.thread_id == "thread-original"
    assert not replayed and replayed_again
    assert retried.id == replay.id
    assert not context_warning and not warning_again


def test_retry_chat_run_discloses_missing_context_and_rejects_nonterminal(
    database: Database,
) -> None:
    run_id = _create_run(database)
    with database.read() as connection:
        identity = connection.execute(
            """
            SELECT agents.owner_id
            FROM runs JOIN agents ON agents.id = runs.agent_id
            WHERE runs.id = ?
            """,
            (run_id,),
        ).fetchone()
    repository = Repository(database)
    with pytest.raises(RepositoryError) as error:
        repository.retry_chat_run(
            identity["owner_id"],
            run_id,
            str(uuid.uuid4()),
        )
    assert error.value.code == "run_not_retryable"

    with database.transaction() as connection:
        connection.execute("UPDATE runs SET status = 'interrupted' WHERE id = ?", (run_id,))
    retried, _replayed, context_warning = repository.retry_chat_run(
        identity["owner_id"],
        run_id,
        str(uuid.uuid4()),
    )
    assert retried.thread_id is None
    assert context_warning


@pytest.mark.asyncio
async def test_event_store_buffers_deltas_deduplicates_snapshots_and_replays(
    database: Database,
) -> None:
    run_id = _create_run(database)
    store = EventStore(database)
    item_id = "message-1"
    store.append_delta(run_id, item_id, "Hello")
    store.append_delta(run_id, item_id, " world")
    await asyncio.sleep(event_store_module.DELTA_FLUSH_SECONDS * 1.2)

    snapshot = {"item_id": item_id, "full_text": "Hello world"}
    store.append(run_id, "message_snapshot", snapshot, sdk_item_id=item_id)
    duplicate = store.append(run_id, "message_snapshot", snapshot, sdk_item_id=item_id)
    store.append(
        run_id,
        "activity_started",
        {"item_id": "tool-1", "name": "command"},
        sdk_item_id="tool-1",
    )
    store.append(
        run_id,
        "activity_completed",
        {"item_id": "tool-1", "status": "completed"},
        sdk_item_id="tool-1",
    )

    assert duplicate is None
    events = store.list_after(run_id)
    assert [event.sequence for event in events] == [1, 2, 3, 4]
    assert events[0].kind == "message_delta"
    assert events[0].payload == {"item_id": item_id, "text": "Hello world"}
    assert [event.kind for event in events[1:]] == [
        "message_snapshot",
        "activity_started",
        "activity_completed",
    ]
    assert store.list_after(run_id, after=2, limit=1)[0].sequence == 3

    resumed_store = EventStore(database)
    resumed_store.append(run_id, "state", {"status": "running"})
    assert resumed_store.list_after(run_id)[-1].sequence == 5


def test_event_store_reserves_terminal_events_and_closes_truncated_activities(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(event_store_module, "MAX_EVENTS_PER_RUN", 5)
    monkeypatch.setattr(event_store_module, "RESERVED_EVENTS", 2)
    run_id = _create_run(database)
    store = EventStore(database)

    store.append(
        run_id,
        "activity_started",
        {"item_id": "tool-1", "name": "command"},
        sdk_item_id="tool-1",
    )
    store.append(run_id, "message_snapshot", {"full_text": "a"})
    store.append(run_id, "message_snapshot", {"full_text": "b"})
    assert store.append(run_id, "message_snapshot", {"full_text": "c"}) is None
    assert store.append(run_id, "message_delta", {"item_id": "message-1", "text": "d"}) is None

    store.finalize(run_id, {"status": "completed"})
    events = store.list_after(run_id, limit=10)

    assert len(events) == 5
    assert events[3].payload["reason"] == "output_truncated"
    assert events[3].payload["incomplete_activity_count"] == 1
    assert events[4].payload == {
        "activities_closed": True,
        "incomplete_activity_count": 1,
        "reason": "output_truncated",
        "terminal_summary": True,
    }


def test_event_store_truncates_at_payload_byte_budget(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(event_store_module, "MAX_EVENTS_PER_RUN", 100)
    monkeypatch.setattr(event_store_module, "RESERVED_EVENTS", 2)
    monkeypatch.setattr(event_store_module, "MAX_EVENT_BYTES_PER_RUN", 1_200)
    monkeypatch.setattr(event_store_module, "RESERVED_EVENT_BYTES", 400)
    run_id = _create_run(database)
    store = EventStore(database)

    for _ in range(10):
        store.append(run_id, "message_delta", {"text": "x" * 90})
    store.finalize(run_id, {"status": "completed"})

    events = store.list_after(run_id, limit=100)
    assert events[-2].payload["reason"] == "output_truncated"
    assert events[-1].payload["terminal_summary"] is True
    assert sum(
        len(event_store_module.json.dumps(event.payload, ensure_ascii=False).encode("utf-8"))
        for event in events
    ) <= event_store_module.MAX_EVENT_BYTES_PER_RUN


def test_event_store_rejects_invalid_replay_cursors(database: Database) -> None:
    store = EventStore(database)
    run_id = _create_run(database)

    with pytest.raises(EventStoreError, match="nonnegative event sequence"):
        store.list_after(run_id, after=-1)
    with pytest.raises(EventStoreError, match="limit"):
        store.list_after(run_id, limit=101)


def test_runtime_process_lock_excludes_a_second_backend(tmp_path) -> None:
    lock_path = tmp_path / "platform-state" / "backend.lock"
    first = RuntimeProcessLock(lock_path)
    second = RuntimeProcessLock(lock_path)

    first.acquire()
    with pytest.raises(CoordinationError, match="Another backend"):
        second.acquire()
    first.release()
    second.acquire()
    second.release()


@pytest.mark.asyncio
async def test_coordination_limits_execution_and_orders_agent_locks() -> None:
    coordination = Coordination(execution_capacity=1)
    first_agent = str(uuid.uuid4())
    second_agent = str(uuid.uuid4())
    active = 0
    maximum_active = 0

    async def occupy(order: list[str]) -> None:
        nonlocal active, maximum_active
        async with coordination.admission(order):
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.wait_for(
        asyncio.gather(
            occupy([first_agent, second_agent]),
            occupy([second_agent, first_agent]),
        ),
        timeout=1,
    )

    assert maximum_active == 1


@pytest.mark.asyncio
async def test_runner_executes_and_persists_one_chat_turn(database: Database) -> None:
    run_id = _create_run(database)
    with database.read() as connection:
        identity = connection.execute(
            """
            SELECT agents.id AS agent_id, agents.owner_id
            FROM runs JOIN agents ON agents.id = runs.agent_id
            WHERE runs.id = ?
            """,
            (run_id,),
        ).fetchone()
    agent_id = identity["agent_id"]
    repository = Repository(database)
    event_store = EventStore(database)
    containers = FakeContainers()
    network = FakeNetwork()
    codex = FakeCodex()
    runner = Runner(
        repository,
        event_store,
        Coordination(),
        containers,
        network,
        codex,
        verified_models=frozenset({"gpt-6-luna"}),
        notification_mapper=lambda notification: (
            (
                "message_delta",
                {"item_id": "assistant-item", "text": "Hello"},
                "assistant-item",
            )
            if notification.method == "item/agentMessage/delta"
            else None
        ),
    )

    runner.enqueue(agent_id)
    worker = runner._agent_tasks[agent_id]
    await asyncio.wait_for(worker, timeout=2)
    run, _last_sequence = repository.get_owned_run(identity["owner_id"], run_id)
    events = event_store.list_after(run_id, limit=20)

    assert run.status == "completed"
    assert run.thread_id == "thread-1"
    assert run.sdk_turn_id == "turn-1"
    assert containers.started == [agent_id]
    assert containers.stopped == [agent_id]
    assert containers.tokens == {}
    assert network.revoked == [run_id]
    assert codex.session.closed
    assert [event.kind for event in events] == [
        "state",
        "message_delta",
        "state",
    ]
    assert events[1].payload["text"] == "Hello"


@pytest.mark.asyncio
async def test_runner_does_not_report_failed_sdk_turn_as_success(database: Database) -> None:
    run_id = _create_run(database)
    with database.read() as connection:
        identity = connection.execute(
            """
            SELECT agents.id AS agent_id, agents.owner_id
            FROM runs JOIN agents ON agents.id = runs.agent_id
            WHERE runs.id = ?
            """,
            (run_id,),
        ).fetchone()
    repository = Repository(database)
    event_store = EventStore(database)
    runner = Runner(
        repository,
        event_store,
        Coordination(),
        FakeContainers(),
        FakeNetwork(),
        FakeCodex("failed"),
        verified_models=frozenset({"gpt-6-luna"}),
        notification_mapper=lambda _notification: None,
    )

    runner.enqueue(identity["agent_id"])
    await asyncio.wait_for(runner._agent_tasks[identity["agent_id"]], timeout=2)
    run, _last_sequence = repository.get_owned_run(identity["owner_id"], run_id)
    events = event_store.list_after(run_id, limit=20)

    assert run.status == "failed"
    assert run.error_code == "codex_turn_failed"
    assert any(event.kind == "error" for event in events)


@pytest.mark.asyncio
async def test_runner_keeps_runs_queued_without_a_verified_model(database: Database) -> None:
    run_id = _create_run(database)
    with database.read() as connection:
        identity = connection.execute(
            """
            SELECT agents.id AS agent_id, agents.owner_id
            FROM runs JOIN agents ON agents.id = runs.agent_id
            WHERE runs.id = ?
            """,
            (run_id,),
        ).fetchone()
    containers = FakeContainers()
    runner = Runner(
        Repository(database),
        EventStore(database),
        Coordination(),
        containers,
        FakeNetwork(),
        FakeCodex(),
    )

    runner.enqueue(identity["agent_id"])
    await asyncio.wait_for(runner._agent_tasks[identity["agent_id"]], timeout=2)
    run, _last_sequence = Repository(database).get_owned_run(identity["owner_id"], run_id)

    assert run.status == "queued"
    assert not containers.started


class BlockingFakeTurn(FakeTurn):
    def __init__(self) -> None:
        super().__init__(status=None)
        self.started = asyncio.Event()

    async def stream(self) -> AsyncIterator[SimpleNamespace]:
        self.started.set()
        await asyncio.Event().wait()
        if False:
            yield SimpleNamespace()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt_fails", [False, True])
async def test_stop_revokes_run_and_waits_for_confirmed_cleanup(
    database: Database, interrupt_fails: bool,
) -> None:
    run_id = _create_run(database)
    with database.read() as connection:
        identity = connection.execute(
            """
            SELECT runs.agent_id, agents.owner_id
            FROM runs JOIN agents ON agents.id = runs.agent_id
            WHERE runs.id = ?
            """,
            (run_id,),
        ).fetchone()
    containers = FakeContainers()
    network = FakeNetwork()
    codex = FakeCodex()
    turn = BlockingFakeTurn()
    if interrupt_fails:
        async def closed_transport_interrupt():
            turn.interrupted = True
            # Closing the SDK/terminating its container can race the advisory RPC.
            await asyncio.sleep(0)
            raise EOFError("SDK transport closed during interrupt")
        turn.interrupt = closed_transport_interrupt
    codex.session.turn = turn
    runner = Runner(
        Repository(database),
        EventStore(database),
        Coordination(),
        containers,
        network,
        codex,
        verified_models=frozenset({"gpt-6-luna"}),
    )

    runner.enqueue(identity["agent_id"])
    await asyncio.wait_for(turn.started.wait(), timeout=2)
    await runner.stop_run(identity["owner_id"], run_id)
    run, _last_sequence = Repository(database).get_owned_run(identity["owner_id"], run_id)
    with database.read() as connection:
        agent = connection.execute(
            "SELECT paused FROM agents WHERE id = ?",
            (identity["agent_id"],),
        ).fetchone()

    assert run.status == "interrupted"
    assert run.termination_reason == "user_stop"
    assert run.cleanup_state == "confirmed"
    assert agent["paused"] == 1
    assert run_id in network.revoked
    assert containers.stopped == [identity["agent_id"]]
    assert turn.interrupted
    assert codex.session.closed


@pytest.mark.asyncio
async def test_stop_keeps_cleanup_failure_blocked(database: Database) -> None:
    run_id = _create_run(database)
    with database.read() as connection:
        identity = connection.execute(
            """
            SELECT runs.agent_id, agents.owner_id
            FROM runs JOIN agents ON agents.id = runs.agent_id
            WHERE runs.id = ?
            """,
            (run_id,),
        ).fetchone()
    containers = FakeContainers()
    containers.fail_stop = True
    runner = Runner(
        Repository(database),
        EventStore(database),
        Coordination(),
        containers,
        FakeNetwork(),
        FakeCodex(),
        verified_models=frozenset({"gpt-6-luna"}),
    )
    run, _last_sequence = Repository(database).get_owned_run(identity["owner_id"], run_id)
    runner.enqueue(run.agent_id)
    while run.status == "queued":
        await asyncio.sleep(0)
        run, _last_sequence = Repository(database).get_owned_run(identity["owner_id"], run_id)

    with pytest.raises(RepositoryError) as error:
        await runner.stop_run(identity["owner_id"], run_id)
    run, _last_sequence = Repository(database).get_owned_run(identity["owner_id"], run_id)
    with database.read() as connection:
        agent = connection.execute(
            "SELECT paused, blockers_json FROM agents WHERE id = ?",
            (identity["agent_id"],),
        ).fetchone()

    assert error.value.status_code == 503
    assert run.status == "stopping"
    assert run.cleanup_state == "failed"
    assert agent["paused"] == 1
    assert "recovery_error" in agent["blockers_json"]


@pytest.mark.asyncio
async def test_dispatch_rechecks_credentials_before_claiming_run(database, settings) -> None:
    from dataclasses import replace

    from backend.app.recovery import VerifiedModelReports

    original = replace(settings, auth_mode="toolkit", toolkit_api_key="original-test-key")
    reports = VerifiedModelReports(original)
    reports.write_report({"gpt-6.1-sol": "passed", "gpt-6-luna": "passed"})
    cached_models = reports.refresh().models
    rotated_reports = VerifiedModelReports(replace(original, toolkit_api_key="rotated-test-key"))
    run_id = _create_run(database)
    with database.read() as connection:
        agent_id = connection.execute(
            "SELECT agent_id FROM runs WHERE id = ?", (run_id,),
        ).fetchone()["agent_id"]
    containers = FakeContainers()
    runner = Runner(
        Repository(database), EventStore(database), Coordination(), containers,
        FakeNetwork(), FakeCodex(), verified_models=cached_models,
        model_admission=lambda: rotated_reports.refresh().models,
    )
    await runner._run_agent_queue(agent_id)
    with database.read() as connection:
        run = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "queued"
    assert containers.started == []


def test_http_chat_retry_and_continue_dispatch_on_the_application_loop(settings):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    app = create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    with TestClient(app) as client:
        runner = app.state.runner
        runner.enabled = True
        runner.containers = FakeContainers()
        runner.network = FakeNetwork()
        runner.codex = FakeCodex()
        headers = {"Origin": settings.allowed_origin, "X-Request-ID": str(uuid.uuid4())}
        registered = client.post("/api/auth/register", headers=headers, json={
            "email": "dispatch-http@example.test", "password": "correct horse battery",
        })
        headers["X-CSRF-Token"] = registered.json()["csrf_token"]
        created = client.post("/api/agents", headers=headers, json={
            "name": "HTTP assistant", "model": "gpt-6-luna",
        })
        agent_id = created.json()["id"]
        sent = client.post(
            f"/api/agents/{agent_id}/messages", headers=headers, json={"prompt": "Hello"},
        )
        assert sent.status_code == 202
        original_id = sent.json()["run_id"]

        def await_completed(run_id):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if client.get(f"/api/runs/{run_id}").json()["status"] == "completed":
                    return
                time.sleep(0.01)
            raise AssertionError("The HTTP-submitted run did not complete.")

        await_completed(original_id)
        with app.state.database.transaction() as connection:
            connection.execute("UPDATE runs SET status = 'failed' WHERE id = ?", (original_id,))
            connection.execute("UPDATE agents SET paused = 1 WHERE id = ?", (agent_id,))
        headers["X-Request-ID"] = str(uuid.uuid4())
        retried = client.post(f"/api/runs/{original_id}/retry", headers=headers)
        assert retried.status_code == 202
        assert retried.json()["paused"] is True
        continued = client.post(f"/api/agents/{agent_id}/continue", headers=headers)
        assert continued.status_code == 200
        await_completed(retried.json()["run_id"])


def test_product_run_limit_reservation_survives_restart_and_exact_expiry(database):
    run_id = _create_run(database)
    with database.read() as connection:
        agent = connection.execute("SELECT agent_id FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    now = [2_000_000_000_000]
    repository = Repository(database, clock=lambda: now[0])
    for index in range(6):
        if index:
            with database.transaction() as connection:
                connection.execute(
                    "UPDATE runs SET status='queued', started_at=NULL WHERE id=?", (run_id,)
                )
        claimed = repository.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"}))
        assert (claimed is not None) == (index < 5)
    usage = repository.run_start_usage()
    assert usage["limited"] is True and usage["minute_count"] == 5
    assert usage["retry_at"] == now[0] + 60_000
    restarted = Repository(database, clock=lambda: now[0])
    assert restarted.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"})) is None
    now[0] += 59_999
    assert restarted.run_start_usage()["limited"]
    now[0] += 1
    assert restarted.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"})) is not None
    assert restarted.run_start_usage()["day_count"] == 6


def test_daily_limit_uses_rolling_window_and_clock_rollback_does_not_reset_it(database):
    run_id = _create_run(database)
    now = [2_000_000_000_000]
    with database.read() as connection:
        agent = connection.execute("SELECT agent_id FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    repository = Repository(database, clock=lambda: now[0])
    for _index in range(20):
        with database.transaction() as connection:
            connection.execute("UPDATE runs SET status='queued' WHERE id=?", (run_id,))
        assert repository.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"}))
        now[0] += 60_000
    with database.transaction() as connection:
        connection.execute("UPDATE runs SET status='queued' WHERE id=?", (run_id,))
    assert repository.run_start_usage()["limited"]
    now[0] -= 120_000
    assert repository.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"})) is None
    now[0] = 2_000_000_000_000 + 86_400_000
    assert repository.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"})) is not None
    assert repository.run_start_usage()["day_count"] == 20


def test_fair_rotation_skips_blocked_agent_and_selects_oldest_eligible_head(database):
    first, second, third = [_create_run(database) for _ in range(3)]
    with database.transaction() as connection:
        rows = {row["id"]: row for row in connection.execute("SELECT * FROM runs").fetchall()}
        owner = connection.execute(
            "SELECT owner_id FROM agents WHERE id=?", (rows[first]["agent_id"],)
        ).fetchone()[0]
        connection.execute(
            "UPDATE agents SET owner_id=? WHERE id=?", (owner, rows[second]["agent_id"])
        )
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (rows[first]["agent_id"],))
        connection.execute("UPDATE runs SET created_at=1 WHERE id=?", (first,))
        connection.execute("UPDATE runs SET created_at=2 WHERE id=?", (second,))
        connection.execute("UPDATE runs SET created_at=3 WHERE id=?", (third,))
    repository = Repository(database)
    selected = []
    for _ in range(2):
        agent = repository.next_runnable_agent(frozenset({"gpt-6-luna"}))
        selected.append(agent)
        assert repository.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"}))
    assert set(selected) == {rows[second]["agent_id"], rows[third]["agent_id"]}
    assert repository.next_runnable_agent(frozenset({"gpt-6-luna"})) is None


async def test_runner_reconsiders_rate_limited_queue_without_new_browser_request(database):
    run_id = _create_run(database)
    now = [2_000_000_000_000]
    repository = Repository(database, clock=lambda: now[0])
    with database.transaction() as connection:
        connection.executemany(
            "INSERT INTO model_run_starts(started_at) VALUES (?)", [(now[0],)] * 5
        )
        agent = connection.execute("SELECT agent_id FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    containers = FakeContainers()
    runner = Runner(
        repository,
        EventStore(database),
        Coordination(),
        containers,
        FakeNetwork(),
        FakeCodex(),
        verified_models=frozenset({"gpt-6-luna"}),
    )
    runner.start()
    try:
        await asyncio.sleep(0.05)
        assert containers.started == []
        now[0] += 60_000
        async with asyncio.timeout(2):
            while not containers.stopped:
                await asyncio.sleep(0.02)
        assert containers.started == [agent]
    finally:
        await runner.close()


def test_owner_agent_detail_exposes_durable_run_limit_without_pausing(settings):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    with TestClient(
        create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    ) as client:
        headers = {"Origin": settings.allowed_origin, "X-Request-ID": str(uuid.uuid4())}
        user = client.post(
            "/api/auth/register",
            headers=headers,
            json={
                "email": "usage@example.test",
                "password": "correct horse battery",
            },
        ).json()
        headers["X-CSRF-Token"] = user["csrf_token"]
        agent = client.post(
            "/api/agents", headers=headers, json={"name": "Usage", "model": "gpt-6-luna"}
        ).json()["id"]
        now = time.time_ns() // 1_000_000
        with client.app.state.database.transaction() as connection:
            connection.executemany(
                "INSERT INTO model_run_starts(started_at) VALUES (?)", [(now,)] * 5
            )
        details = client.get(f"/api/agents/{agent}").json()
        assert details["run_admission"]["limited"] is True
        assert details["run_admission"]["minute_count"] == 5
        assert details["run_admission"]["minute_limit"] == 5
        assert details["run_admission"]["day_limit"] == 20
        assert details["run_admission"]["retry_at"] is not None
        assert details["paused"] is False


async def test_two_claims_cannot_both_take_the_last_global_start(database):
    identities = [_create_run(database) for _ in range(2)]
    now = 2_000_000_000_000
    with database.transaction() as connection:
        connection.executemany("INSERT INTO model_run_starts(started_at) VALUES (?)", [(now,)] * 4)
        agents = [
            connection.execute("SELECT agent_id FROM runs WHERE id=?", (run,)).fetchone()[0]
            for run in identities
        ]
    repositories = [Repository(database, clock=lambda: now) for _ in range(2)]
    results = await asyncio.gather(
        *[
            asyncio.to_thread(repo.claim_next_run, agent, verified_models=frozenset({"gpt-6-luna"}))
            for repo, agent in zip(repositories, agents, strict=True)
        ]
    )
    assert sum(result is not None for result in results) == 1
    with database.read() as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_run_starts").fetchone()[0] == 5
        assert (
            connection.execute("SELECT COUNT(*) FROM runs WHERE status='queued'").fetchone()[0] == 1
        )


def test_round_robin_owner_pointer_persists_and_assistant_fifo_is_preserved(database):
    runs = [_create_run(database) for _ in range(4)]
    with database.transaction() as connection:
        agents = [
            connection.execute("SELECT agent_id FROM runs WHERE id=?", (run,)).fetchone()[0]
            for run in runs
        ]
        owners = [
            connection.execute("SELECT owner_id FROM agents WHERE id=?", (agent,)).fetchone()[0]
            for agent in agents
        ]
        connection.execute("UPDATE agents SET owner_id=? WHERE id=?", (owners[0], agents[1]))
        connection.execute("UPDATE agents SET owner_id=? WHERE id=?", (owners[2], agents[3]))
        connection.executemany("UPDATE runs SET created_at=? WHERE id=?", enumerate(runs, 1))
    served = []
    for _ in range(4):
        # Reconstruct the repository to prove fairness does not rely on process-local pointers.
        repo = Repository(database)
        chosen = repo.next_runnable_agent(frozenset({"gpt-6-luna"}))
        context = repo.claim_next_run(chosen, verified_models=frozenset({"gpt-6-luna"}))
        with database.transaction() as connection:
            served.append(
                connection.execute("SELECT owner_id FROM agents WHERE id=?", (chosen,)).fetchone()[
                    0
                ]
            )
            connection.execute("UPDATE runs SET status='completed' WHERE id=?", (context.run.id,))
    assert served[0] != served[1] and served[0] == served[2] and served[1] == served[3]


async def test_live_runner_rotates_owners_with_multiple_competing_assistants(database):
    runs = [_create_run(database) for _ in range(4)]
    with database.transaction() as connection:
        agents = [
            connection.execute("SELECT agent_id FROM runs WHERE id=?", (run,)).fetchone()[0]
            for run in runs
        ]
        owner = connection.execute(
            "SELECT owner_id FROM agents WHERE id=?", (agents[0],)
        ).fetchone()[0]
        connection.executemany(
            "UPDATE agents SET owner_id=? WHERE id=?", [(owner, agent) for agent in agents[:3]]
        )
        owners = {
            row["id"]: row["owner_id"]
            for row in connection.execute("SELECT id,owner_id FROM agents")
        }
    containers = FakeContainers()
    runner = Runner(
        Repository(database),
        EventStore(database),
        Coordination(execution_capacity=1),
        containers,
        FakeNetwork(),
        FakeCodex(),
        verified_models=frozenset({"gpt-6-luna"}),
    )
    runner.start()
    try:
        async with asyncio.timeout(4):
            while len(containers.stopped) < 4:
                await asyncio.sleep(0.02)
        assert set(containers.started) == set(agents)
        assert owners[containers.started[0]] != owners[containers.started[1]]
    finally:
        await runner.close()


def test_claim_cannot_bypass_unclean_terminal_barrier(database):
    run_id = _create_run(database)
    with database.transaction() as connection:
        row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        connection.execute(
            "UPDATE runs SET status='failed', cleanup_state='failed' WHERE id=?", (run_id,)
        )
        connection.execute(
            "INSERT INTO runs(id,agent_id,source,prompt,status,model,created_at) "
            "VALUES (?,?,'chat','Next','queued','gpt-6-luna',?)",
            (str(uuid.uuid4()), row["agent_id"], row["created_at"] + 1),
        )
    repo = Repository(database)
    assert repo.claim_next_run(row["agent_id"], verified_models=frozenset({"gpt-6-luna"})) is None
    assert repo.run_start_usage()["day_count"] == 0


async def test_storage_emergency_revokes_and_stops_when_database_writes_are_frozen(database):
    run_id = _create_run(database)
    with database.read() as connection:
        agent = connection.execute("SELECT agent_id FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    containers, network, codex = FakeContainers(), FakeNetwork(), FakeCodex()
    turn = BlockingFakeTurn()
    codex.session.turn = turn
    runner = Runner(
        Repository(database),
        EventStore(database),
        Coordination(),
        containers,
        network,
        codex,
        verified_models=frozenset({"gpt-6-luna"}),
    )
    runner.enqueue(agent)
    await asyncio.wait_for(turn.started.wait(), 2)
    database.storage.enabled = True
    database.storage.trip("simulated_quota_full")
    try:
        assert await runner.storage_stop()
        assert run_id in network.revoked
        assert agent in containers.stopped and codex.session.closed
        with database.read() as connection:
            assert (
                connection.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()[0]
                != "completed"
            )
    finally:
        await runner.close()
