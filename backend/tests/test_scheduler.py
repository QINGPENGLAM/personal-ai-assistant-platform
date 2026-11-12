from datetime import UTC, datetime
from uuid import uuid4

import pytest

from backend.app.auth import AuthService
from backend.app.coordination import Coordination
from backend.app.repository import Repository, RepositoryError


@pytest.fixture
def jobs(settings, database):
    from backend.app.jobs import Actor, JobService
    from backend.app.scheduler import Scheduler

    owner, _ = AuthService(database, settings).register(
        "jobs@example.test",
        "correct horse battery",
        "127.0.0.1",
    )
    repository = Repository(database)
    agent, _ = repository.create_agent(owner.id, "Tasks", "gpt-6-luna", str(uuid4()))
    repository.set_agent_status(agent.id, "ready")
    now = [int(datetime(2026, 11, 6, tzinfo=UTC).timestamp() * 1000)]
    coordination = Coordination()
    service = JobService(repository, coordination, clock=lambda: now[0])
    scheduler = Scheduler(service, lambda agent: None)
    return service, scheduler, Actor("user", owner.id), agent.id, now


def body(**kwargs):
    return {
        "name": "Review notes",
        "prompt": "Summarize workspace notes",
        "run_at": "2026-11-06T00:01:00+00:00",
        **kwargs,
    }


async def test_due_once_is_durable_and_does_not_repeat_after_restart(jobs):
    service, scheduler, actor, agent, now = jobs
    job = await service.create(actor, agent, str(uuid4()), body())
    now[0] += 60_000
    await scheduler.tick()
    # A fresh service/clock/coordination instance must rely on persisted occurrence state.
    from backend.app.jobs import JobService
    from backend.app.scheduler import Scheduler

    restarted = JobService(
        Repository(service.repository.database), Coordination(), clock=lambda: now[0]
    )
    await Scheduler(restarted, lambda agent: None).tick()
    runs = service.runs(actor, job["job_id"])["items"]
    assert len(runs) == 1 and runs[0]["status"] == "queued"
    assert service.list(actor, agent)["items"][0]["enabled"] is False
    assert runs[0]["scheduled_for"] == "2026-11-06T00:01:00+00:00"


async def test_recurrence_coalesces_without_reanchoring_or_duplicate_nonterminal(jobs):
    service, scheduler, actor, agent, now = jobs
    job = await service.create(actor, agent, str(uuid4()), body(interval_seconds=60))
    now[0] += 305_000
    await scheduler.tick()
    first = service.runs(actor, job["job_id"])["items"][0]
    assert first["scheduled_for"] == "2026-11-06T00:05:00+00:00"
    assert first["coalesced_count"] == 4
    now[0] += 180_000
    await scheduler.tick()
    assert len(service.runs(actor, job["job_id"])["items"]) == 1
    with service.repository.database.transaction() as connection:
        connection.execute("UPDATE runs SET status = 'completed' WHERE id = ?", (first["id"],))
    await scheduler.tick()
    runs = service.runs(actor, job["job_id"])["items"]
    assert len(runs) == 2
    assert runs[-1]["scheduled_for"] == "2026-11-06T00:08:00+00:00"
    assert runs[-1]["coalesced_count"] == 2
    now[0] -= 500_000
    await scheduler.tick()
    assert len(service.runs(actor, job["job_id"])["items"]) == 2


async def test_job_create_replays_even_when_time_has_passed(jobs):
    service, scheduler, actor, agent, now = jobs
    request_id = str(uuid4())
    first = await service.create(actor, agent, request_id, body())
    now[0] += 120_000
    assert (await service.create(actor, agent, request_id, body()))["job_id"] == first["job_id"]
    with pytest.raises(RepositoryError) as conflict:
        await service.create(actor, agent, request_id, body(name="Different"))
    assert conflict.value.status_code == 409


@pytest.mark.parametrize(
    "run_at",
    [
        "2026-11-06T00:01:00Z",
        "2026-11-06T00:01:00",
        "2026-11-06T00:01:00-00:00",
        "2026-11-06T00:01:00+14:01",
        "2026-02-30T00:01:00+00:00",
        "2026-11-06T00:01:00+01:60",
    ],
)
async def test_invalid_explicit_times_rejected(jobs, run_at):
    service, scheduler, actor, agent, now = jobs
    with pytest.raises(RepositoryError) as rejected:
        await service.create(actor, agent, str(uuid4()), body(run_at=run_at))
    assert rejected.value.status_code == 422


async def test_cancel_queued_job_retains_history_and_actor_isolation(jobs):
    from backend.app.jobs import Actor

    service, scheduler, actor, agent, now = jobs
    request_id = str(uuid4())
    job = await service.create(actor, agent, request_id, body())
    now[0] += 60_000
    await scheduler.tick()
    with pytest.raises(RepositoryError) as denied:
        await service.cancel(Actor("user", str(uuid4())), job["job_id"], str(uuid4()))
    assert denied.value.status_code == 404
    receipt = await service.cancel(actor, job["job_id"], str(uuid4()))
    assert receipt["enabled"] is False and receipt["active_run_id"] is None
    assert service.runs(actor, job["job_id"])["items"][0]["termination_reason"] == "job_cancel"


async def test_job_dispatch_keeps_chat_thread_and_ignores_chat_epoch(jobs):
    service, scheduler, actor, agent, now = jobs
    await service.create(actor, agent, str(uuid4()), body())
    with service.repository.database.transaction() as connection:
        connection.execute(
            "UPDATE agents SET chat_epoch=3, chat_thread_id='chat-thread' WHERE id=?", (agent,)
        )
    now[0] += 60000
    await scheduler.tick()
    context = service.repository.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"}))
    assert context is not None and context.thread_id is None
    assert service.repository.update_run_running(
        context.run.id,
        thread_id="job-thread",
        generation=context.generation,
        run_token_hash="token",
    )
    with service.repository.database.read() as connection:
        assert (
            connection.execute("SELECT chat_thread_id FROM agents WHERE id=?", (agent,)).fetchone()[
                0
            ]
            == "chat-thread"
        )


@pytest.mark.parametrize("cursor", ["WzAsIDBd", "WzAsIHt9XQ=="])
async def test_malformed_cursor_returns_validation_error(jobs, cursor):
    service, scheduler, actor, agent, now = jobs
    with pytest.raises(RepositoryError) as invalid:
        service.list(actor, agent, cursor=cursor)
    assert invalid.value.code == "invalid_cursor"


async def test_pause_queue_capacity_and_cancel_running_semantics(jobs):
    service, scheduler, actor, agent, now = jobs
    job = await service.create(actor, agent, str(uuid4()), body(interval_seconds=60))
    with service.repository.database.transaction() as connection:
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    now[0] += 60_000
    await scheduler.tick()
    assert (
        service.repository.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"})) is None
    )
    run = service.runs(actor, job["job_id"])["items"][0]
    with service.repository.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='starting' WHERE id=?", (run["id"],))
    cancelled = await service.cancel(actor, job["job_id"], str(uuid4()))
    assert cancelled["active_run_id"] == run["id"]
    assert service.runs(actor, job["job_id"])["items"][0]["status"] == "starting"


async def test_full_queue_does_not_advance_schedule(jobs):
    service, scheduler, actor, agent, now = jobs
    job = await service.create(actor, agent, str(uuid4()), body())
    for _ in range(20):
        service.repository.create_chat_run(actor.id, agent, str(uuid4()), "Waiting")
    now[0] += 60_000
    await scheduler.tick()
    assert service.runs(actor, job["job_id"])["items"] == []
    assert service.list(actor, agent)["items"][0]["next_run_at"] == job["run_at"]
    with service.repository.database.transaction() as connection:
        connection.execute(
            "DELETE FROM events WHERE run_id IN (SELECT id FROM runs WHERE agent_id=?)", (agent,)
        )
        connection.execute("DELETE FROM runs WHERE agent_id=?", (agent,))
    await scheduler.tick()
    assert len(service.runs(actor, job["job_id"])["items"]) == 1


async def test_job_retry_keeps_original_context_and_blocks_next_occurrence(jobs):
    service, scheduler, actor, agent, now = jobs
    job = await service.create(actor, agent, str(uuid4()), body(interval_seconds=60))
    now[0] += 60_000
    await scheduler.tick()
    original = service.runs(actor, job["job_id"])["items"][0]
    with service.repository.database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET status='failed', thread_id='job-thread' WHERE id=?", (original["id"],)
        )
    retry, replay, warning = service.repository.retry_chat_run(
        actor.id, original["id"], str(uuid4())
    )
    assert retry.source == "job" and retry.job_id == job["job_id"]
    assert retry.thread_id == "job-thread" and warning is False
    now[0] += 120_000
    await scheduler.tick()
    assert len(service.runs(actor, job["job_id"])["items"]) == 2
    with pytest.raises(RepositoryError):
        service.repository.retry_chat_run(actor.id, original["id"], str(uuid4()))
    await service.cancel(actor, job["job_id"], str(uuid4()))
    assert service.runs(actor, job["job_id"])["items"][-1]["status"] == "cancelled"
    with pytest.raises(RepositoryError):
        service.repository.retry_chat_run(actor.id, original["id"], str(uuid4()))


async def test_repeated_wall_clock_offsets_store_different_instants(jobs):
    service, scheduler, actor, agent, now = jobs
    now[0] = int(datetime(2026, 10, 31, tzinfo=UTC).timestamp() * 1000)
    one = await service.create(actor, agent, str(uuid4()), body(run_at="2026-11-01T01:30:00-04:00"))
    two = await service.create(actor, agent, str(uuid4()), body(run_at="2026-11-01T01:30:00-05:00"))
    assert one["run_at"] == "2026-11-01T05:30:00+00:00"
    assert two["run_at"] == "2026-11-01T06:30:00+00:00"


async def test_enabled_job_limit_and_pagination_preserve_all_members(jobs):
    service, scheduler, actor, agent, now = jobs
    created = [
        await service.create(actor, agent, str(uuid4()), body(name=f"Task {index}"))
        for index in range(50)
    ]
    with pytest.raises(RepositoryError) as full:
        await service.create(actor, agent, str(uuid4()), body())
    assert full.value.status_code == 429
    ids, cursor = [], None
    while True:
        page = service.list(actor, agent, cursor=cursor, limit=7)
        ids.extend(item["job_id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(ids) == len(set(ids)) == 50
    assert set(ids) == {item["job_id"] for item in created}
    await service.cancel(actor, created[0]["job_id"], str(uuid4()))
    await service.create(actor, agent, str(uuid4()), body(name="New capacity"))


def test_mcp_shutdown_failure_still_closes_runner_and_releases_runtime_lock(settings, monkeypatch):
    from contextlib import asynccontextmanager

    from fastapi.testclient import TestClient
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    from backend.app.coordination import RuntimeProcessLock
    from backend.app.main import create_app
    from backend.app.runner import Runner

    closed = []

    @asynccontextmanager
    async def failing_exit(self):
        yield
        raise RuntimeError("simulated MCP shutdown failure")

    async def runner_close(self):
        closed.append(True)

    monkeypatch.setattr(StreamableHTTPSessionManager, "run", failing_exit)
    monkeypatch.setattr(Runner, "close", runner_close)
    with pytest.raises(RuntimeError, match="simulated MCP shutdown failure"):
        with TestClient(create_app(settings, test_mode=True)) as client:
            assert client.get("/health").status_code == 200
    assert closed == [True]
    lock = RuntimeProcessLock(settings.platform_state_root / "backend.lock")
    lock.acquire()
    lock.release()


def test_job_http_api_requires_csrf_and_rejects_malformed_page_cursor(settings):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    with TestClient(
        create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    ) as client:
        headers = {"Origin": settings.allowed_origin, "X-Request-ID": str(uuid4())}
        registered = client.post(
            "/api/auth/register",
            headers=headers,
            json={
                "email": "job-api@example.test",
                "password": "correct horse battery",
            },
        )
        headers["X-CSRF-Token"] = registered.json()["csrf_token"]
        agent = client.post(
            "/api/agents", headers=headers, json={"name": "Tasks", "model": "gpt-6-luna"}
        ).json()["id"]
        from datetime import timedelta

        payload = body(
            run_at=(datetime.now(UTC) + timedelta(minutes=2)).isoformat(timespec="seconds")
        )
        base = f"/api/agents/{agent}/jobs"
        assert client.post(base, json=payload).status_code == 403
        job = client.post(base, json=payload, headers=headers).json()
        for cursor in ("WzAsIDBd", "WzAsIHt9XQ=="):
            for path in (base, f"/api/jobs/{job['job_id']}/runs"):
                invalid = client.get(path, params={"cursor": cursor})
                assert invalid.status_code == 422 and invalid.json()["code"] == "invalid_cursor"
        assert client.post(f"/api/jobs/{job['job_id']}/cancel").status_code == 403
