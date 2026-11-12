from __future__ import annotations

import json
import time
import uuid
from dataclasses import replace

import pytest

from backend.app.recovery import REPORT_MAX_AGE_MS, VerifiedModelReports
from backend.app.repository import Repository


def _toolkit_settings(settings, api_key: str = "local-test-key"):
    return replace(settings, auth_mode="toolkit", toolkit_api_key=api_key)


def test_model_report_is_bound_to_runtime_and_credentials(settings) -> None:
    report_store = VerifiedModelReports(_toolkit_settings(settings))
    candidates = {"gpt-6.1-sol": "passed", "gpt-6-luna": "failed"}

    report_store.write_report(candidates)
    result = report_store.refresh()
    changed_credentials = VerifiedModelReports(
        _toolkit_settings(settings, "rotated-test-key")
    ).refresh()

    assert result.models == frozenset({"gpt-6.1-sol"})
    assert result.reason is None
    assert changed_credentials.models == frozenset()
    assert changed_credentials.reason == "configuration_mismatch"
    key_info = settings.model_report_key_path.stat()
    assert key_info.st_mode & 0o777 == 0o600


def test_model_report_binds_auth_file_contents_without_publishing_them(
    settings,
    tmp_path,
) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text('{"access_token":"sensitive-local-test-value"}', encoding="utf-8")
    auth_settings = replace(
        settings,
        auth_mode="auth_file",
        auth_file_validation_path=auth_file,
    )
    report_store = VerifiedModelReports(auth_settings)
    report_store.write_report({"gpt-6.1-sol": "passed", "gpt-6-luna": "failed"})
    assert report_store.refresh().models == frozenset({"gpt-6.1-sol"})
    report_content = settings.model_report_path.read_text(encoding="utf-8")
    assert "sensitive-local-test-value" not in report_content

    auth_file.write_text('{"access_token":"rotated-local-test-value"}', encoding="utf-8")
    changed = report_store.refresh()
    assert changed.models == frozenset()
    assert changed.reason == "configuration_mismatch"


def test_model_report_refresh_observes_atomic_replacements(settings) -> None:
    report_store = VerifiedModelReports(_toolkit_settings(settings))
    report_store.write_report({"gpt-6.1-sol": "failed", "gpt-6-luna": "passed"})
    assert report_store.refresh().models == frozenset({"gpt-6-luna"})

    report_store.write_report({"gpt-6.1-sol": "passed", "gpt-6-luna": "passed"})
    assert report_store.refresh().models == frozenset(
        {"gpt-6.1-sol", "gpt-6-luna"}
    )


def test_model_report_rejects_stale_future_and_malformed_files(settings) -> None:
    report_store = VerifiedModelReports(_toolkit_settings(settings))
    now_ms = time.time_ns() // 1_000_000
    report_store.write_report(
        {"gpt-6.1-sol": "passed", "gpt-6-luna": "passed"},
        generated_at=now_ms - REPORT_MAX_AGE_MS - 1,
    )
    assert report_store.refresh().reason == "expired_report"

    report_store.write_report(
        {"gpt-6.1-sol": "passed", "gpt-6-luna": "passed"},
        generated_at=now_ms + 6 * 60 * 1000,
    )
    assert report_store.refresh().reason == "future_report"

    settings.model_report_path.write_text("{", encoding="utf-8")
    assert report_store.refresh().reason == "report_invalid"


def test_model_report_requires_regular_report_and_safe_key(settings, tmp_path) -> None:
    report_store = VerifiedModelReports(_toolkit_settings(settings))
    report_store.write_report({"gpt-6.1-sol": "passed", "gpt-6-luna": "passed"})
    settings.model_report_path.unlink()
    settings.model_report_path.symlink_to(tmp_path / "outside.json")
    assert report_store.refresh().reason == "report_invalid"

    key_path = settings.model_report_key_path
    key_path.chmod(0o644)
    assert report_store.refresh().reason == "report_invalid"


def test_model_availability_blocks_only_the_unverified_model(database) -> None:
    user_id = str(uuid.uuid4())
    verified_agent = str(uuid.uuid4())
    unverified_agent = str(uuid.uuid4())
    now_ms = time.time_ns() // 1_000_000
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO users(id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, f"{user_id}@example.test", "test-hash", now_ms),
        )
        for agent_id, model, blockers in (
            (verified_agent, "gpt-6.1-sol", '["other_blocker"]'),
            (unverified_agent, "gpt-6-luna", "[]"),
        ):
            connection.execute(
                """
                INSERT INTO agents(
                    id, owner_id, name, model, status, blockers_json, created_at
                ) VALUES (?, ?, 'Test assistant', ?, 'ready', ?, ?)
                """,
                (agent_id, user_id, model, blockers, now_ms),
            )

    repository = Repository(database)
    repository.update_model_availability(frozenset({"gpt-6.1-sol"}))
    with database.read() as connection:
        rows = {
            row["id"]: row
            for row in connection.execute(
                "SELECT id, paused, blockers_json FROM agents"
            ).fetchall()
        }

    assert rows[verified_agent]["paused"] == 0
    assert json.loads(rows[verified_agent]["blockers_json"]) == ["other_blocker"]
    assert rows[unverified_agent]["paused"] == 1
    assert json.loads(rows[unverified_agent]["blockers_json"]) == ["model_setup_required"]

    repository.update_model_availability(frozenset({"gpt-6.1-sol", "gpt-6-luna"}))
    with database.read() as connection:
        resumed = connection.execute(
            "SELECT paused, blockers_json FROM agents WHERE id = ?",
            (unverified_agent,),
        ).fetchone()
    assert resumed["paused"] == 1
    assert json.loads(resumed["blockers_json"]) == []


def test_model_report_rejects_boolean_version(settings) -> None:
    store = VerifiedModelReports(_toolkit_settings(settings))
    store.write_report({"gpt-6.1-sol": "passed", "gpt-6-luna": "passed"})
    report = json.loads(settings.model_report_path.read_bytes())
    report["version"] = True
    settings.model_report_path.write_text(json.dumps(report), encoding="utf-8")
    assert store.refresh().models == frozenset()
    assert store.refresh().reason == "report_invalid"


def test_model_report_preserves_zero_timestamp_as_expired(settings) -> None:
    store = VerifiedModelReports(_toolkit_settings(settings))
    store.write_report(
        {"gpt-6.1-sol": "passed", "gpt-6-luna": "passed"}, generated_at=0,
    )
    assert store.refresh().reason == "expired_report"


class RecoveryContainers:
    def __init__(self, running=(), fail=()):
        self.running = list(running)
        self.fail = set(fail)
        self.stopped = []
        self.removed = []
        self.before_stop = None

    async def running_agent_ids(self):
        return self.running

    async def stop_agent(self, agent_id):
        if self.before_stop is not None:
            self.before_stop(agent_id)
        self.stopped.append(agent_id)
        if agent_id in self.fail:
            raise RuntimeError("simulated cleanup failure")

    def remove_run_token(self, agent_id):
        self.removed.append(agent_id)


class RecoveryNetwork:
    def __init__(self):
        self.revoked = []

    def revoke_run(self, run_id):
        self.revoked.append(run_id)

    async def wait_run_closed(self, run_id):
        return None


def _recovery_fixture(database, status="running", intended=None):
    owner, agent, active, queued = [str(uuid.uuid4()) for _ in range(4)]
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO users VALUES (?, ?, 'hash', 1)", (owner, f"{owner}@example.test"),
        )
        connection.execute(
            """INSERT INTO agents(id, owner_id, name, model, status, chat_thread_id,
               blockers_json, generation, created_at)
               VALUES (?, ?, 'Assistant', 'gpt-6-luna', 'ready', 'saved-thread',
                       '["agent_storage_full"]', 7, 1)""", (agent, owner),
        )
        for run_id, state in ((active, status), (queued, "queued")):
            connection.execute(
                """INSERT INTO runs(id, agent_id, source, prompt, status, model,
                   thread_id, sdk_turn_id, run_token_hash, created_at)
                   VALUES (?, ?, 'chat', 'Original prompt', ?, 'gpt-6-luna',
                           'saved-thread', 'saved-turn', 'old-token-hash', 1)""",
                (run_id, agent, state),
            )
        if intended:
            connection.execute(
                """UPDATE runs SET intended_terminal_status = ?,
                   intended_termination_reason = 'user_stop', cleanup_state = 'failed'
                   WHERE id = ?""", (intended, active),
            )
    return agent, active, queued


async def test_startup_recovery_stops_before_finalizing_and_preserves_queue(database):
    from backend.app.event_store import EventStore
    from backend.app.recovery import StartupRecovery

    agent, active, queued = _recovery_fixture(database)
    containers = RecoveryContainers([agent])
    network = RecoveryNetwork()

    def assert_not_finalized(_agent_id):
        with database.read() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (active,)).fetchone()
        assert row["status"] == "stopping"
        assert row["finished_at"] is None
        assert row["run_token_hash"] is None
        assert active in network.revoked

    containers.before_stop = assert_not_finalized
    store = EventStore(database)
    await StartupRecovery(Repository(database), store, containers, network).recover()
    with database.read() as connection:
        runs = {row["id"]: row for row in connection.execute("SELECT * FROM runs")}
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
    assert runs[active]["status"] == "interrupted"
    assert runs[active]["cleanup_state"] == "confirmed"
    assert runs[active]["termination_reason"] == "backend_restart"
    assert runs[queued]["status"] == "queued"
    assert runs[active]["thread_id"] == "saved-thread"
    assert runs[active]["sdk_turn_id"] == "saved-turn"
    assert saved["chat_thread_id"] == "saved-thread"
    assert saved["paused"] == 1
    assert saved["generation"] == 8
    assert set(json.loads(saved["blockers_json"])) == {"agent_storage_full", "history_error"}
    assert containers.stopped == [agent]
    assert containers.removed == [agent]
    assert store.list_after(active)[-1].payload["history_incomplete"] is True


async def test_startup_cleanup_failure_is_repairable_without_repeating_work(database):
    from backend.app.event_store import EventStore
    from backend.app.recovery import StartupRecovery

    agent, active, queued = _recovery_fixture(database, "stopping", "interrupted")
    containers = RecoveryContainers([agent], fail=[agent])
    recovery = StartupRecovery(
        Repository(database), EventStore(database), containers, RecoveryNetwork(),
    )
    await recovery.recover()
    with database.read() as connection:
        run = connection.execute("SELECT * FROM runs WHERE id = ?", (active,)).fetchone()
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
    assert run["status"] == "stopping"
    assert run["cleanup_state"] == "failed"
    assert run["intended_termination_reason"] == "user_stop"
    assert "recovery_error" in json.loads(saved["blockers_json"])
    containers.fail.clear()
    await recovery.recover()
    with database.read() as connection:
        run = connection.execute("SELECT * FROM runs WHERE id = ?", (active,)).fetchone()
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
        queued_state = connection.execute(
            "SELECT status FROM runs WHERE id = ?", (queued,),
        ).fetchone()
    assert run["status"] == "interrupted"
    assert run["termination_reason"] == "user_stop"
    assert saved["paused"] == 1
    assert "recovery_error" not in json.loads(saved["blockers_json"])
    assert "agent_storage_full" in json.loads(saved["blockers_json"])
    assert queued_state["status"] == "queued"


async def test_startup_recovery_cleans_idle_containers_and_continues_after_agent_failure(database):
    from backend.app.event_store import EventStore
    from backend.app.recovery import StartupRecovery

    failed_agent, active, _ = _recovery_fixture(database)
    idle_agent, idle_run, idle_queued = _recovery_fixture(database, "completed")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE agents SET blockers_json = '[]', chat_thread_id = NULL WHERE id = ?",
            (idle_agent,),
        )
    containers = RecoveryContainers([failed_agent, idle_agent], fail=[failed_agent])
    await StartupRecovery(
        Repository(database), EventStore(database), containers, RecoveryNetwork(),
    ).recover()
    assert set(containers.stopped) == {failed_agent, idle_agent}
    with database.read() as connection:
        states = {
            row["id"]: row["status"]
            for row in connection.execute("SELECT id, status FROM runs")
        }
    assert states[active] == "stopping"
    assert states[idle_run] == "completed"
    runnable = Repository(database).claim_next_run(
        idle_agent, verified_models=frozenset({"gpt-6-luna"}),
    )
    assert runnable is not None
    assert runnable.run.id == idle_queued


class DockerScenario:
    def __init__(self, manager, agent_id, *, stop_fails=False, kill_fails=False):
        self.manager = manager
        self.name = manager.container_name(agent_id)
        self.running = True
        self.stop_fails = stop_fails
        self.kill_fails = kill_fails
        self.commands = []
        self.labels = {
            "qpl.platform": "personal-assistant", "qpl.instance": manager.instance_id,
            "qpl.agent": agent_id, "qpl.purpose": "agent",
        }
        self.mounts = []
        self.configured_mounts = []
        for volume, subpath, destination, writable in (
            (manager.settings.agent_volume, f"{agent_id}/workspace", "/workspace", True),
            (manager.settings.agent_volume, f"{agent_id}/codex", "/codex", True),
            (manager.settings.agent_volume, f"{agent_id}/credentials", "/run/credentials", False),
            (manager.settings.platform_volume, "gateway", "/run/platform-gateway", False),
        ):
            self.mounts.append({
                "Type": "volume", "Name": volume,
                "Source": f"/var/lib/docker/volumes/{volume}/_data",
                "Destination": destination, "RW": writable,
            })
            self.configured_mounts.append({
                "Type": "volume", "Source": volume, "Target": destination,
                "ReadOnly": not writable, "VolumeOptions": {"Subpath": subpath, "NoCopy": True},
            })

    async def docker(self, *args, **kwargs):
        from backend.app.agents import AgentPreparationError

        self.commands.append(args)
        if args[:2] == ("container", "ls"):
            return self.name
        if args[:2] == ("container", "inspect"):
            values = {
                "{{json .Config.Labels}}": self.labels,
                "{{json .Mounts}}": self.mounts,
                "{{json .Config}}": {
                    "User": "1000:1000", "Image": self.manager.settings.agent_image,
                },
                "{{json .HostConfig}}": {
                    "NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
                    "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges:true"],
                    "Mounts": self.configured_mounts,
                },
                "{{json .State}}": {
                    "Running": self.running, "Restarting": False,
                    "Status": "running" if self.running else "exited",
                },
            }
            return json.dumps(values[args[3]])
        if args[:2] in (("container", "stop"), ("container", "kill")):
            if (args[1] == "stop" and self.stop_fails) or (
                args[1] == "kill" and self.kill_fails
            ):
                raise AgentPreparationError("docker_operation_failed", "Simulated Docker failure.")
            self.running = False
            return self.name
        raise AssertionError(f"Unexpected Docker operation: {args}")


def _docker_scenario(settings, monkeypatch, **kwargs):
    from backend.app.containers import ContainerManager

    manager = ContainerManager(_toolkit_settings(settings), str(uuid.uuid4()))
    scenario = DockerScenario(manager, str(uuid.uuid4()), **kwargs)
    monkeypatch.setattr(manager, "_docker", scenario.docker)
    return manager, scenario


async def test_container_stop_falls_back_to_kill_and_confirms_exit(settings, monkeypatch):
    manager, scenario = _docker_scenario(settings, monkeypatch, stop_fails=True)
    await manager.stop_agent(scenario.labels["qpl.agent"])
    assert scenario.running is False
    assert any(command[:2] == ("container", "kill") for command in scenario.commands)
    assert scenario.commands[-1][3] == "{{json .State}}"


async def test_container_stop_rejects_wrong_instance_before_mutation(settings, monkeypatch):
    import pytest

    from backend.app.agents import AgentPreparationError

    manager, scenario = _docker_scenario(settings, monkeypatch)
    scenario.labels["qpl.instance"] = str(uuid.uuid4())
    with pytest.raises(AgentPreparationError):
        await manager.stop_agent(scenario.labels["qpl.agent"])
    assert scenario.running is True
    assert all(command[1] not in {"stop", "kill", "create"} for command in scenario.commands)


async def test_container_stop_fails_if_exit_cannot_be_confirmed(settings, monkeypatch):
    import pytest

    from backend.app.agents import AgentPreparationError

    manager, scenario = _docker_scenario(settings, monkeypatch, stop_fails=True, kill_fails=True)
    with pytest.raises(AgentPreparationError, match="confirmed stopped"):
        await manager.stop_agent(scenario.labels["qpl.agent"])
    assert scenario.running is True


async def test_running_inventory_uses_exact_instance_and_purpose_filters(settings, monkeypatch):
    manager, scenario = _docker_scenario(settings, monkeypatch)
    assert await manager.running_agent_ids() == [scenario.labels["qpl.agent"]]
    command = scenario.commands[0]
    assert f"label=qpl.instance={manager.instance_id}" in command
    assert "label=qpl.platform=personal-assistant" in command
    assert "label=qpl.purpose=agent" in command


def test_production_lifespan_recovers_before_starting_dispatch(database, settings, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app.containers import ContainerManager
    from backend.app.main import create_app
    from backend.app.network_gateway import NetworkGateway
    from backend.app.recovery import HistoryInspector
    from backend.app.runner import Runner

    agent, active, _ = _recovery_fixture(database)
    order = []

    async def noop(self):
        return None

    async def image(self):
        return "sha256:" + "a" * 64

    async def inventory(self):
        return [agent]

    async def stop(self, agent_id):
        order.append("stop")

    async def inspect(self, state, runs):
        order.append("inspect")

    def dispatch(self):
        with database.read() as connection:
            row = connection.execute("SELECT status FROM runs WHERE id = ?", (active,)).fetchone()
        assert row["status"] == "interrupted"
        assert order == ["stop", "stop", "inspect"]
        order.append("dispatch")

    async def healthy_storage(self):
        return {"quota": {"used_bytes": 0, "hard_bytes": 2 * 1024**3},
                "free_bytes": 3 * 1024**3}

    monkeypatch.setattr(ContainerManager, "platform_storage_usage", healthy_storage)
    monkeypatch.setattr(ContainerManager, "verify_runtime", noop)
    monkeypatch.setattr(ContainerManager, "resolve_agent_image", image)
    monkeypatch.setattr(ContainerManager, "running_agent_ids", inventory)
    monkeypatch.setattr(ContainerManager, "stop_agent", stop)
    monkeypatch.setattr(ContainerManager, "remove_run_token", lambda self, agent_id: None)
    monkeypatch.setattr(NetworkGateway, "start", noop)
    monkeypatch.setattr(NetworkGateway, "close", noop)
    from backend.app.workspace import WorkspaceService

    original_initialize = WorkspaceService.initialize

    def fake_staging(self):
        self.test_mode = True
        self.staging_root = settings.runtime_root / "upload-staging"
        original_initialize(self)

    monkeypatch.setattr(WorkspaceService, "initialize", fake_staging)
    monkeypatch.setattr(Runner, "start", dispatch)
    monkeypatch.setattr(HistoryInspector, "inspect", inspect)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
    assert order == ["stop", "stop", "inspect", "dispatch", "stop"]


async def test_missing_container_is_not_created_during_cleanup(settings, monkeypatch):
    manager, scenario = _docker_scenario(settings, monkeypatch)

    async def absent(*args, **kwargs):
        assert args[:2] == ("container", "ls")
        return ""

    monkeypatch.setattr(manager, "_docker", absent)
    await manager.stop_agent(scenario.labels["qpl.agent"])


async def test_container_cleanup_rejects_another_agents_volume_subpath(settings, monkeypatch):
    import pytest

    from backend.app.agents import AgentPreparationError

    manager, scenario = _docker_scenario(settings, monkeypatch)
    scenario.mounts[0]["Source"] = (
        f"/var/lib/docker/volumes/{settings.agent_volume}/_data/{uuid.uuid4()}/workspace"
    )
    with pytest.raises(AgentPreparationError):
        await manager.stop_agent(scenario.labels["qpl.agent"])
    assert scenario.running is True
    assert all(command[1] not in {"stop", "kill", "create"} for command in scenario.commands)


async def test_inventory_failure_keeps_active_work_nonterminal_and_paused(database):
    import pytest

    from backend.app.event_store import EventStore
    from backend.app.recovery import StartupRecovery

    agent, active, _ = _recovery_fixture(database)
    containers = RecoveryContainers()

    async def unavailable():
        raise RuntimeError("Docker inventory unavailable")

    containers.running_agent_ids = unavailable
    with pytest.raises(RuntimeError, match="inventory unavailable"):
        await StartupRecovery(
            Repository(database), EventStore(database), containers, RecoveryNetwork(),
        ).recover()
    with database.read() as connection:
        run = connection.execute("SELECT * FROM runs WHERE id = ?", (active,)).fetchone()
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
    assert run["status"] == "stopping"
    assert run["finished_at"] is None
    assert saved["paused"] == 1
    assert "recovery_error" in json.loads(saved["blockers_json"])
    assert containers.stopped == []


async def test_recovery_preserves_saved_outcome_jobs_and_workspace(database, settings):
    from backend.app.event_store import EventStore
    from backend.app.recovery import StartupRecovery

    agent, active, _ = _recovery_fixture(database, "stopping", "completed_with_errors")
    job_id = str(uuid.uuid4())
    workspace = settings.agent_data_root / agent / "workspace"
    workspace.mkdir(parents=True)
    artifact = workspace / "notes.txt"
    artifact.write_bytes(b"saved work\n")
    with database.transaction() as connection:
        connection.execute(
            """UPDATE runs SET intended_termination_reason = 'turn_complete',
               had_tool_errors = 1 WHERE id = ?""", (active,),
        )
        connection.execute(
            """INSERT INTO jobs(id, agent_id, name, prompt, created_by, run_at,
               next_run_at, created_at) VALUES (?, ?, 'Saved job', 'Saved prompt',
               'user', 1000, 1000, 1)""", (job_id, agent),
        )
        original_job = dict(connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,),
        ).fetchone())
    store = EventStore(database)
    recovery = StartupRecovery(
        Repository(database), store, RecoveryContainers([agent]), RecoveryNetwork(),
    )
    await recovery.recover()
    events_before = store.list_after(active)
    await recovery.recover()
    assert store.list_after(active) == events_before
    assert artifact.read_bytes() == b"saved work\n"
    with database.read() as connection:
        run = connection.execute("SELECT * FROM runs WHERE id = ?", (active,)).fetchone()
        saved_job = dict(connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,),
        ).fetchone())
    assert run["status"] == "completed_with_errors"
    assert run["termination_reason"] == "turn_complete"
    assert run["had_tool_errors"] == 1
    assert saved_job == original_job


async def test_recovery_still_stops_container_when_token_file_removal_fails(database):
    from backend.app.event_store import EventStore
    from backend.app.recovery import StartupRecovery

    agent, active, _ = _recovery_fixture(database)
    containers = RecoveryContainers([agent])

    def cannot_remove(_agent_id):
        raise OSError("Simulated token file removal failure")

    containers.remove_run_token = cannot_remove
    await StartupRecovery(
        Repository(database), EventStore(database), containers, RecoveryNetwork(),
    ).recover()
    assert containers.stopped == [agent]
    with database.read() as connection:
        run = connection.execute("SELECT * FROM runs WHERE id = ?", (active,)).fetchone()
    assert run["status"] == "stopping"
    assert run["cleanup_state"] == "failed"


class HistoryContainers(RecoveryContainers):
    def __init__(self, *, fail_inspection_stop=False):
        super().__init__()
        self.started = []
        self.fail_inspection_stop = fail_inspection_stop

    async def start_agent(self, agent_id):
        self.started.append(agent_id)

    async def stop_agent(self, agent_id):
        if self.started and self.fail_inspection_stop:
            raise RuntimeError("Inspection container could not stop")
        await super().stop_agent(agent_id)


class HistoryCodex:
    def __init__(self, histories):
        self.histories = histories
        self.closed = False
        self.reads = []

    async def connect(self, agent_id, *, history_only=False):
        assert history_only is True
        return self

    async def read_history(self, model, thread_id):
        self.reads.append(thread_id)
        value = self.histories[thread_id]
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        self.closed = True


async def test_history_inspection_imports_only_exact_saved_turn_after_cleanup(database):
    from backend.app.event_store import EventStore
    from backend.app.recovery import HistoryInspector, StartupRecovery

    agent, active, _ = _recovery_fixture(database)
    containers = HistoryContainers()
    sdk = HistoryCodex({"saved-thread": {"id": "saved-thread", "turns": [
        {"id": "other-turn", "items": [
            {"id": "wrong", "type": "agentMessage", "text": "Do not import"},
        ]},
        {"id": "saved-turn", "items": [
            {"id": "reply", "type": "agentMessage", "text": "Recovered reply"},
        ]},
    ]}})
    store = EventStore(database)
    inspector = HistoryInspector(Repository(database), store, containers, sdk)
    await StartupRecovery(
        Repository(database), store, containers, RecoveryNetwork(), inspector=inspector,
    ).recover()
    assert containers.stopped == [agent, agent]
    assert containers.started == [agent]
    assert sdk.closed is True
    snapshots = [event for event in store.list_after(active) if event.kind == "message_snapshot"]
    assert [event.payload for event in snapshots] == [
        {"item_id": "reply", "full_text": "Recovered reply"},
    ]
    with database.read() as connection:
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
    assert saved["paused"] == 1
    assert json.loads(saved["blockers_json"]) == ["agent_storage_full"]
    assert store.list_after(active)[-1].payload["history_incomplete"] is False


async def test_unreadable_thread_keeps_saved_events_and_history_blocker(database):
    from backend.app.event_store import EventStore
    from backend.app.recovery import HistoryInspector, StartupRecovery

    agent, active, _ = _recovery_fixture(database)
    store = EventStore(database)
    store.append(active, "message_snapshot", {"item_id": "saved", "full_text": "Saved reply"})
    sdk = HistoryCodex({"saved-thread": RuntimeError("Thread unavailable")})
    containers = HistoryContainers()
    await StartupRecovery(
        Repository(database), store, containers, RecoveryNetwork(),
        inspector=HistoryInspector(Repository(database), store, containers, sdk),
    ).recover()
    assert store.list_after(active)[0].payload["full_text"] == "Saved reply"
    assert containers.stopped == [agent, agent]
    assert sdk.closed is True
    with database.read() as connection:
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
    assert "history_error" in json.loads(saved["blockers_json"])
    assert "recovery_error" not in json.loads(saved["blockers_json"])


async def test_history_inspection_cleanup_failure_blocks_dispatch(database):
    from backend.app.event_store import EventStore
    from backend.app.recovery import HistoryInspector, StartupRecovery

    agent, _, _ = _recovery_fixture(database)
    containers = HistoryContainers(fail_inspection_stop=True)
    sdk = HistoryCodex({"saved-thread": {"id": "saved-thread", "turns": []}})
    store = EventStore(database)
    await StartupRecovery(
        Repository(database), store, containers, RecoveryNetwork(),
        inspector=HistoryInspector(Repository(database), store, containers, sdk),
    ).recover()
    with database.read() as connection:
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
    assert saved["paused"] == 1
    assert "recovery_error" in json.loads(saved["blockers_json"])


def test_new_chat_is_idempotent_and_preserves_archived_history_and_jobs(database):
    agent, active, queued = _recovery_fixture(database, "interrupted")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE agents SET paused = 1, blockers_json = ? "
            "WHERE id = ?", ('["history_error","model_setup_required"]', agent),
        )
        owner = connection.execute(
            "SELECT owner_id FROM agents WHERE id = ?", (agent,),
        ).fetchone()[0]
        job_id = str(uuid.uuid4())
        job_run = str(uuid.uuid4())
        connection.execute(
            """INSERT INTO jobs(id, agent_id, name, prompt, created_by, run_at, created_at)
               VALUES (?, ?, 'Job', 'Scheduled prompt', 'agent', 1, 1)""", (job_id, agent),
        )
        connection.execute(
            """INSERT INTO runs(id, agent_id, source, job_id, prompt, status, model, created_at)
               VALUES (?, ?, 'job', ?, 'Scheduled prompt', 'queued', 'gpt-6-luna', 1)""",
            (job_run, agent, job_id),
        )
    repository = Repository(database)
    request_id = str(uuid.uuid4())
    first, replayed = repository.new_chat(owner, agent, request_id)
    second, replayed_again = repository.new_chat(owner, agent, request_id)
    assert replayed is False
    assert replayed_again is True
    assert first.chat_epoch == second.chat_epoch == 1
    assert second.paused is True
    assert second.blockers == ("model_setup_required",)
    with database.read() as connection:
        rows = {row["id"]: row for row in connection.execute("SELECT * FROM runs")}
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
    assert rows[active]["status"] == "interrupted"
    assert rows[active]["thread_id"] == "saved-thread"
    assert rows[queued]["status"] == "cancelled"
    assert rows[queued]["termination_reason"] == "new_chat"
    assert rows[job_run]["status"] == "queued"
    assert saved["chat_thread_id"] is None
    import pytest

    from backend.app.repository import RepositoryError
    with pytest.raises(RepositoryError, match="older conversation"):
        repository.retry_chat_run(owner, active, str(uuid.uuid4()))


def test_new_chat_rejects_unpaused_or_unclean_agents_without_cancelling_queue(database):
    import pytest

    from backend.app.repository import RepositoryError

    agent, active, queued = _recovery_fixture(database)
    repository = Repository(database)
    with database.read() as connection:
        owner = connection.execute(
            "SELECT owner_id FROM agents WHERE id = ?", (agent,),
        ).fetchone()[0]
    with pytest.raises(RepositoryError):
        repository.new_chat(owner, agent, str(uuid.uuid4()))
    with database.transaction() as connection:
        connection.execute("UPDATE agents SET paused = 1 WHERE id = ?", (agent,))
    with pytest.raises(RepositoryError):
        repository.new_chat(owner, agent, str(uuid.uuid4()))
    with database.read() as connection:
        saved = connection.execute(
            "SELECT chat_epoch FROM agents WHERE id = ?", (agent,),
        ).fetchone()[0]
        status = connection.execute("SELECT status FROM runs WHERE id = ?", (queued,)).fetchone()[0]
    assert saved == 0
    assert status == "queued"


async def test_history_inspection_timeout_still_closes_session_and_container(database):
    import asyncio

    from backend.app.event_store import EventStore
    from backend.app.recovery import HistoryInspector, StartupRecovery

    agent, _, _ = _recovery_fixture(database)
    sdk = HistoryCodex({})

    async def slow_read(model, thread_id):
        await asyncio.Event().wait()

    sdk.read_history = slow_read
    containers = HistoryContainers()
    store = EventStore(database)
    await StartupRecovery(
        Repository(database), store, containers, RecoveryNetwork(),
        inspector=HistoryInspector(Repository(database), store, containers, sdk, deadline=0.01),
    ).recover()
    assert sdk.closed is True
    assert containers.stopped == [agent, agent]
    with database.read() as connection:
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
    assert "history_error" in json.loads(saved["blockers_json"])
    assert "recovery_error" not in json.loads(saved["blockers_json"])


async def test_missing_turn_id_never_imports_another_turn(database):
    from backend.app.event_store import EventStore
    from backend.app.recovery import HistoryInspector, StartupRecovery

    agent, active, _ = _recovery_fixture(database)
    with database.transaction() as connection:
        connection.execute("UPDATE runs SET sdk_turn_id = NULL WHERE id = ?", (active,))
    sdk = HistoryCodex({"saved-thread": {"id": "saved-thread", "turns": [
        {"id": "other-turn", "items": [
            {"id": "wrong", "type": "agentMessage", "text": "Do not guess"},
        ]},
    ]}})
    containers = HistoryContainers()
    store = EventStore(database)
    await StartupRecovery(
        Repository(database), store, containers, RecoveryNetwork(),
        inspector=HistoryInspector(Repository(database), store, containers, sdk),
    ).recover()
    assert not any(event.kind == "message_snapshot" for event in store.list_after(active))
    assert store.list_after(active)[-1].payload["history_incomplete"] is True


async def test_unused_new_chat_does_not_start_history_container_or_gain_blocker(database):
    from backend.app.event_store import EventStore
    from backend.app.recovery import HistoryInspector, StartupRecovery

    agent, active, queued = _recovery_fixture(database, "interrupted")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE agents SET chat_epoch = 1, chat_thread_id = NULL, "
            "blockers_json = '[]' WHERE id = ?",
            (agent,),
        )
    containers = HistoryContainers()
    store = EventStore(database)
    await StartupRecovery(
        Repository(database), store, containers, RecoveryNetwork(),
        inspector=HistoryInspector(Repository(database), store, containers, HistoryCodex({})),
    ).recover()
    assert containers.started == []
    with database.read() as connection:
        saved = connection.execute("SELECT * FROM agents WHERE id = ?", (agent,)).fetchone()
    assert saved["paused"] == 0
    assert saved["blockers_json"] == "[]"


def test_model_reports_use_instance_scoped_share_and_immutable_image(settings):
    instance = str(uuid.uuid4())
    image = "sha256:" + "a" * 64
    store = VerifiedModelReports.for_instance(_toolkit_settings(settings), instance, image)
    assert store.refresh().reason == "report_missing"
    assert store.settings.model_report_path.parent.is_dir()
    store.write_report({"gpt-6.1-sol": "passed", "gpt-6-luna": "failed"})
    share = settings.platform_state_root / "probe-share" / instance
    assert store.settings.model_report_key_path == share / "key" / "hmac.key"
    assert store.settings.model_report_path == share / "report" / "verified-models.json"
    assert store.refresh().models == frozenset({"gpt-6.1-sol"})
    changed = VerifiedModelReports.for_instance(
        _toolkit_settings(settings), instance, "sha256:" + "b" * 64,
    )
    assert changed.refresh().reason == "configuration_mismatch"
    other_instance = VerifiedModelReports.for_instance(
        _toolkit_settings(settings), str(uuid.uuid4()), image,
    )
    assert other_instance.refresh().models == frozenset()


def test_storage_guard_hysteresis_and_database_failure_latch(database):
    import sqlite3

    from backend.app.db import StorageWriteBlocked

    guard = database.storage
    guard.enabled = True
    guard.observe(
        {"quota": {"used_bytes": 0, "hard_bytes": 2 * 1024**3}, "free_bytes": 3 * 1024**3}
    )
    guard.observe(
        {
            "quota": {"used_bytes": (2 * 1024**3 * 80 + 99) // 100, "hard_bytes": 2 * 1024**3},
            "free_bytes": 3 * 1024**3,
        }
    )
    assert guard.latched
    with pytest.raises(StorageWriteBlocked):
        with database.transaction():
            pytest.fail("Protected storage must reject before opening a write transaction.")
    assert database.instance_id()
    guard.observe(
        {
            "quota": {"used_bytes": 2 * 1024**3 * 75 // 100, "hard_bytes": 2 * 1024**3},
            "free_bytes": 3 * 1024**3,
        }
    )
    assert not guard.can_recover
    guard.observe(
        {
            "quota": {"used_bytes": 2 * 1024**3 * 69 // 100, "hard_bytes": 2 * 1024**3},
            "free_bytes": 3 * 1024**3,
        }
    )
    assert guard.can_recover and guard.latched
    with guard.repair():
        with database.transaction() as connection:
            connection.execute("UPDATE runtime_metadata SET value=value WHERE key='instance_uuid'")
    guard.clear_after_repair()
    with pytest.raises(sqlite3.OperationalError):
        with database.transaction() as connection:
            connection.execute("INSERT INTO missing_storage_table VALUES (1)")
    assert guard.latched and guard.reason == "database_write_failed"


@pytest.mark.parametrize(
    "snapshot", [None, {}, {"quota": {"used_bytes": 0, "hard_bytes": 1}, "free_bytes": 3 * 1024**3}]
)
def test_storage_measurement_must_be_verified_and_fails_closed(database, snapshot):
    guard = database.storage
    guard.enabled = True
    guard.observe(snapshot)
    assert guard.latched and not guard.can_recover


async def test_storage_monitor_clears_only_after_cleanup_probe_and_second_measurement(database):
    from backend.app.main import PlatformStorageMonitor

    database.storage.enabled = True
    sample = {"quota": {"used_bytes": 0, "hard_bytes": 2 * 1024**3}, "free_bytes": 3 * 1024**3}
    calls = []

    async def measure():
        return sample

    async def stop():
        calls.append("stop")
        return True

    async def repair():
        calls.append("repair")
        with database.transaction() as connection:
            connection.execute("UPDATE runtime_metadata SET value=value WHERE key='instance_uuid'")

    monitor = PlatformStorageMonitor(database, measure, stop, repair)
    await monitor.check()
    assert not database.storage.latched
    sample["free_bytes"] = 1024**3 - 1
    await monitor.check()
    assert database.storage.latched and calls == ["stop"]
    sample["free_bytes"] = 2 * 1024**3
    await monitor.check()
    assert database.storage.latched and calls == ["stop", "stop"]
    sample["free_bytes"] += 1
    await monitor.check()
    assert not database.storage.latched and calls[-2:] == ["stop", "repair"]


async def test_storage_monitor_never_unfreezes_after_failed_cleanup(database):
    from backend.app.main import PlatformStorageMonitor

    database.storage.enabled = True
    database.storage.trip("simulated_failure")

    async def measure():
        return {"quota": {"used_bytes": 0, "hard_bytes": 2 * 1024**3}, "free_bytes": 3 * 1024**3}

    async def stop():
        return False

    async def repair():
        pytest.fail("Recovery cannot proceed before confirmed process cleanup.")

    await PlatformStorageMonitor(database, measure, stop, repair).check()
    assert database.storage.latched


def test_storage_pressure_keeps_owned_reads_and_rejects_new_writes(settings):
    from datetime import UTC, datetime, timedelta

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
                "email": "pressure@example.test",
                "password": "correct horse battery",
            },
        ).json()
        headers["X-CSRF-Token"] = user["csrf_token"]
        agent = client.post(
            "/api/agents", headers=headers, json={"name": "Pressure", "model": "gpt-6-luna"}
        ).json()["id"]
        database = client.app.state.database
        database.storage.enabled = True
        database.storage.observe(None)
        details = client.get(f"/api/agents/{agent}").json()
        assert details["storage_protection"]["active"]
        assert client.get(f"/api/agents/{agent}/files").status_code == 200
        response = client.post(
            f"/api/agents/{agent}/jobs",
            headers=headers,
            json={
                "name": "Blocked",
                "prompt": "Read notes",
                "run_at": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(timespec="seconds"),
            },
        )
        assert response.status_code == 507 and response.json()["code"] == "storage_protected"
        assert (
            client.post(
                "/api/agents", headers=headers, json={"name": "Blocked", "model": "gpt-6-luna"}
            ).status_code
            == 507
        )
        with database.read() as connection:
            assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM agents").fetchone()[0] == 1


def test_failed_database_bootstrap_still_stops_known_owned_containers(
    database, settings, monkeypatch
):
    import sqlite3

    from fastapi.testclient import TestClient

    from backend.app.containers import ContainerManager
    from backend.app.db import Database
    from backend.app.main import create_app

    stopped = []
    instance = database.instance_id()
    agent = str(uuid.uuid4())

    def fail_initialize(self):
        raise sqlite3.OperationalError("simulated database full")

    async def inventory(self):
        assert self.instance_id == instance
        return [agent]

    async def stop(self, agent_id):
        stopped.append(agent_id)

    monkeypatch.setattr(Database, "initialize", fail_initialize)
    monkeypatch.setattr(ContainerManager, "running_agent_ids", inventory)
    monkeypatch.setattr(ContainerManager, "stop_agent", stop)
    with pytest.raises(sqlite3.OperationalError):
        with TestClient(create_app(settings)):
            pass
    assert stopped == [agent]


def test_production_restart_at_75_percent_cannot_bypass_recovery_watermark(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app.containers import ContainerManager
    from backend.app.main import create_app
    from backend.app.network_gateway import NetworkGateway
    from backend.app.workspace import WorkspaceService

    async def noop(self):
        return None

    async def image(self):
        return "sha256:" + "a" * 64

    async def inventory(self):
        return []

    async def medium_storage(self):
        return {
            "quota": {"used_bytes": 2 * 1024**3 * 75 // 100, "hard_bytes": 2 * 1024**3},
            "free_bytes": 3 * 1024**3,
        }

    original = WorkspaceService.initialize

    def staging(self):
        self.test_mode = True
        self.staging_root = settings.runtime_root / "upload-staging"
        original(self)

    monkeypatch.setattr(ContainerManager, "verify_runtime", noop)
    monkeypatch.setattr(ContainerManager, "resolve_agent_image", image)
    monkeypatch.setattr(ContainerManager, "running_agent_ids", inventory)
    monkeypatch.setattr(ContainerManager, "platform_storage_usage", medium_storage)
    monkeypatch.setattr(NetworkGateway, "start", noop)
    monkeypatch.setattr(NetworkGateway, "close", noop)
    monkeypatch.setattr(WorkspaceService, "initialize", staging)
    for _ in range(2):
        with TestClient(create_app(settings)) as client:
            assert client.get("/health").json()["storage_protection"]["active"]
            assert client.app.state.database.storage.can_recover is False


def test_cached_identity_allows_emergency_stop_when_sqlite_cannot_be_read(
    database, settings, monkeypatch
):
    import sqlite3

    from fastapi.testclient import TestClient

    from backend.app.containers import ContainerManager
    from backend.app.db import Database
    from backend.app.main import create_app

    instance = database.instance_id()
    settings.platform_state_root.mkdir(parents=True, exist_ok=True)
    Database.preserve_instance_id(settings.platform_state_root / "instance-id", instance)
    stopped = []
    agent = str(uuid.uuid4())

    def unavailable(self):
        raise sqlite3.OperationalError("simulated unreadable database")

    async def inventory(self):
        assert self.instance_id == instance
        return [agent]

    async def stop(self, agent_id):
        stopped.append(agent_id)

    monkeypatch.setattr(Database, "instance_id", unavailable)
    monkeypatch.setattr(Database, "initialize", unavailable)
    monkeypatch.setattr(ContainerManager, "running_agent_ids", inventory)
    monkeypatch.setattr(ContainerManager, "stop_agent", stop)
    with pytest.raises(sqlite3.OperationalError):
        with TestClient(create_app(settings)):
            pass
    assert stopped == [agent]


def test_identity_cache_publishes_atomically_after_partial_writes(tmp_path, monkeypatch):
    import backend.app.db as module

    identity = str(uuid.uuid4())
    path = tmp_path / "instance-id"
    original = module.os.write

    def short_write(descriptor, data):
        return original(descriptor, data[: max(1, len(data) // 2)])

    monkeypatch.setattr(module.os, "write", short_write)
    module.Database.preserve_instance_id(path, identity)
    assert module.Database.cached_instance_id(path) == identity


def test_failed_identity_cache_sync_never_publishes_partial_identity(tmp_path, monkeypatch):
    import backend.app.db as module

    path = tmp_path / "instance-id"

    def fail_sync(descriptor):
        raise OSError("simulated cache sync failure")

    monkeypatch.setattr(module.os, "fsync", fail_sync)
    with pytest.raises(OSError):
        module.Database.preserve_instance_id(path, str(uuid.uuid4()))
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.fixture
def lifecycle(database, settings):
    from backend.app.agents import AgentService
    from backend.app.auth import AuthService
    from backend.app.coordination import Coordination
    from backend.app.event_store import EventStore
    from backend.app.runner import Runner
    from backend.app.workspace import WorkspaceService
    from backend.tests.test_runs import FakeCodex, FakeContainers, FakeNetwork

    user, _ = AuthService(database, settings).register(
        "controls@example.test", "correct horse battery", "127.0.0.1"
    )
    repository = Repository(database)
    agent, _ = repository.create_agent(user.id, "Controls", "gpt-6-luna", str(uuid.uuid4()))
    repository.set_agent_status(agent.id, "ready")
    root = settings.agent_data_root / agent.id
    for name in ("workspace", "codex", "credentials"):
        (root / name).mkdir(parents=True)
    coordination = Coordination()
    runtime = FakeContainers()
    runtime.running = set()

    async def inventory():
        return sorted(runtime.running)

    async def start(agent_id):
        runtime.running.add(agent_id)
        runtime.started.append(agent_id)

    async def stop(agent_id):
        runtime.running.discard(agent_id)
        runtime.stopped.append(agent_id)

    async def prepare(agent):
        return None

    from backend.app.containers import ContainerManager

    manager = ContainerManager(settings, database.instance_id())
    runtime.running_agent_ids = inventory
    runtime.start_agent = start
    runtime.stop_agent = stop
    runtime.prepare_agent = prepare
    runtime.reset_model_context = manager.reset_model_context
    runner = Runner(
        repository,
        EventStore(database),
        coordination,
        runtime,
        FakeNetwork(),
        FakeCodex(),
        enabled=False,
    )
    workspace = WorkspaceService(settings, repository, coordination, test_mode=True)
    workspace.initialize()
    clock = [0.0]
    service = AgentService(
        repository,
        runtime,
        runner=runner,
        coordination=coordination,
        workspace=workspace,
        clock=lambda: clock[0],
    )
    yield service, repository, user.id, agent.id, root, runtime, clock
    workspace.close()


async def test_wake_sleep_restart_preserve_pause_and_durable_replay(lifecycle):
    service, repo, owner, agent, root, runtime, clock = lifecycle
    with repo.database.transaction() as connection:
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    request_id = str(uuid.uuid4())
    await service.control(owner, agent, "wake", request_id)
    assert agent in runtime.running and repo.get_owned_agent(owner, agent).paused
    await service.control(owner, agent, "wake", request_id)
    assert runtime.started == [agent]
    await service.control(owner, agent, "sleep", str(uuid.uuid4()))
    assert agent not in runtime.running and repo.get_owned_agent(owner, agent).paused
    await service.control(owner, agent, "restart", str(uuid.uuid4()))
    assert agent in runtime.running and repo.get_owned_agent(owner, agent).paused


async def test_context_reset_preserves_files_credentials_jobs_and_saved_records(lifecycle):
    service, repo, owner, agent, root, runtime, clock = lifecycle
    (root / "workspace" / "notes.txt").write_bytes(b"keep workspace")
    (root / "credentials" / "login").write_bytes(b"keep credential")
    (root / "codex" / "history").write_bytes(b"old model history")
    with repo.database.transaction() as connection:
        connection.execute(
            "UPDATE agents SET paused=1, chat_thread_id='old-thread', "
            "blockers_json='[\"history_error\"]' WHERE id=?",
            (agent,),
        )
    old, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Old queued message")
    request_id = str(uuid.uuid4())
    await service.control(owner, agent, "reset-model-context", request_id, confirmed=True)
    assert not (root / "codex" / "history").exists()
    assert (root / "codex" / "config.toml").is_file()
    assert (root / "workspace" / "notes.txt").read_bytes() == b"keep workspace"
    assert (root / "credentials" / "login").read_bytes() == b"keep credential"
    current = repo.get_owned_agent(owner, agent)
    assert current.chat_epoch == 1 and current.paused and "history_error" not in current.blockers
    assert repo.get_owned_run(owner, old.id)[0].status == "cancelled"
    (root / "codex" / "new-history").write_bytes(b"new model data")
    await service.control(owner, agent, "reset-model-context", request_id, confirmed=True)
    assert (root / "codex" / "new-history").read_bytes() == b"new model data"
    assert repo.get_owned_agent(owner, agent).chat_epoch == 1


async def test_manual_wake_evicts_oldest_idle_and_idle_sleep_uses_monotonic_clock(lifecycle):
    service, repo, owner, first, root, runtime, clock = lifecycle
    others = []
    for name in ("Second", "Third"):
        agent, _ = repo.create_agent(owner, name, "gpt-6-luna", str(uuid.uuid4()))
        repo.set_agent_status(agent.id, "ready")
        others.append(agent.id)
    await service.control(owner, first, "wake", str(uuid.uuid4()))
    clock[0] = 1
    await service.control(owner, others[0], "wake", str(uuid.uuid4()))
    clock[0] = 2
    await service.control(owner, others[1], "wake", str(uuid.uuid4()))
    assert runtime.running == set(others)
    assert first in runtime.stopped
    clock[0] = 901
    await service.sleep_idle()
    assert runtime.running == {others[1]}
    assert not repo.get_owned_agent(owner, others[0]).paused


async def test_starting_runs_reserve_container_capacity_before_docker_starts(lifecycle):
    service, repo, owner, first, root, runtime, clock = lifecycle
    for name in ("Busy one", "Busy two"):
        agent, _ = repo.create_agent(owner, name, "gpt-6-luna", str(uuid.uuid4()))
        repo.set_agent_status(agent.id, "ready")
        repo.create_chat_run(owner, agent.id, str(uuid.uuid4()), "Busy")
        assert repo.claim_next_run(agent.id, verified_models=frozenset({"gpt-6-luna"}))
    from backend.app.repository import RepositoryError

    with pytest.raises(RepositoryError) as full:
        await service.control(owner, first, "wake", str(uuid.uuid4()))
    assert full.value.code == "container_capacity"
    assert not runtime.started


async def test_context_reset_requires_stopped_container_before_archiving(lifecycle):
    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    await service.control(owner, agent, "wake", str(uuid.uuid4()))
    with repo.database.transaction() as connection:
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    with pytest.raises(RepositoryError) as awake:
        await service.control(
            owner, agent, "reset-model-context", str(uuid.uuid4()), confirmed=True
        )
    assert awake.value.code == "sleep_required"
    assert repo.get_owned_agent(owner, agent).chat_epoch == 0


async def test_failed_restart_does_not_start_new_container_and_requires_new_inspected_request(
    lifecycle,
):
    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle

    async def fail_stop(agent_id):
        raise RuntimeError("simulated stop failure")

    runtime.stop_agent = fail_stop
    request_id = str(uuid.uuid4())
    with pytest.raises(RepositoryError):
        await service.control(owner, agent, "restart", request_id)
    assert runtime.started == []
    assert "recovery_error" in repo.get_owned_agent(owner, agent).blockers
    with pytest.raises(RepositoryError) as uncertain:
        await service.control(owner, agent, "restart", request_id)
    assert uncertain.value.code == "control_uncertain"
    with pytest.raises(RepositoryError) as not_repair:
        await service.control(owner, agent, "wake", str(uuid.uuid4()))
    assert not_repair.value.code == "restart_required"


async def test_control_reservation_blocks_claim_continue_upload_and_is_tombstoned_on_recovery(
    lifecycle,
):
    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Queued")
    request_id = str(uuid.uuid4())
    repo.reserve_agent_control(owner, agent, "wake", request_id, "hash")
    assert repo.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"})) is None
    with pytest.raises(RepositoryError):
        repo.check_workspace_mutation(owner, agent, upload=True)
    with pytest.raises(RepositoryError):
        repo.continue_agent(owner, agent, verified_models=frozenset({"gpt-6-luna"}))
    repo.interrupt_agent_controls()
    with pytest.raises(RepositoryError) as unknown:
        repo.agent_control_receipt(owner, agent, "wake", request_id, "hash")
    assert unknown.value.code == "control_uncertain"


async def test_history_reset_does_not_follow_codex_symlink(lifecycle, tmp_path):
    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    outside = tmp_path / "private-model"
    outside.mkdir()
    (outside / "keep").write_bytes(b"outside")
    (root / "codex").rmdir()
    (root / "codex").symlink_to(outside, target_is_directory=True)
    with repo.database.transaction() as connection:
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    with pytest.raises(RepositoryError):
        await service.control(
            owner, agent, "reset-model-context", str(uuid.uuid4()), confirmed=True
        )
    assert (outside / "keep").read_bytes() == b"outside"
    assert "context_reset_error" in repo.get_owned_agent(owner, agent).blockers


async def test_manual_wake_reports_capacity_without_waiting_for_busy_turns(lifecycle):
    import asyncio

    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    async with service.coordination.execution_slot(), service.coordination.execution_slot():
        with pytest.raises(RepositoryError) as busy:
            await asyncio.wait_for(service.control(owner, agent, "wake", str(uuid.uuid4())), 0.2)
    assert busy.value.code == "container_capacity"
    assert not runtime.started


async def test_toolkit_context_reset_preserves_exact_credential_bytes_and_inode(lifecycle):
    import threading
    import time

    service, repo, owner, agent, root, runtime, clock = lifecycle
    manager = runtime.reset_model_context.__self__
    manager.settings = replace(
        manager.settings, auth_mode="toolkit", toolkit_api_key="configured-new-test-key"
    )
    credential = root / "credentials" / "UMGPT_TOOLKIT_API_KEY"
    credential.write_bytes(b"preserve-existing-test-key")
    inode = credential.stat().st_ino
    manager.reset_model_context(
        repo.get_owned_agent(owner, agent),
        time.monotonic() + 30,
        threading.Event(),
        repo.database.storage.require_writable,
    )
    assert credential.read_bytes() == b"preserve-existing-test-key"
    assert credential.stat().st_ino == inode


async def test_late_continue_replay_cannot_undo_a_new_pause(lifecycle):
    service, repo, owner, agent, root, runtime, clock = lifecycle
    with repo.database.transaction() as connection:
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    request_id = str(uuid.uuid4())
    repo.continue_agent(
        owner, agent, verified_models=frozenset({"gpt-6-luna"}), request_id=request_id
    )
    with repo.database.transaction() as connection:
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    assert repo.continue_agent(
        owner, agent, verified_models=frozenset({"gpt-6-luna"}), request_id=request_id
    ).paused


async def test_sleep_interrupts_active_turn_with_manual_reason_and_keeps_queue_paused(lifecycle):
    import asyncio

    from backend.tests.test_runs import BlockingFakeTurn

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Long task")
    turn = BlockingFakeTurn()
    service.runner.codex.session.turn = turn
    service.runner.enabled = True
    service.runner.verified_models = frozenset({"gpt-6-luna"})
    service.runner.enqueue(agent)
    try:
        await asyncio.wait_for(turn.started.wait(), 2)
        await service.control(owner, agent, "sleep", str(uuid.uuid4()))
        ended = repo.get_owned_run(owner, run.id)[0]
        assert ended.status == "interrupted" and ended.termination_reason == "manual_sleep"
        assert ended.cleanup_state == "confirmed" and repo.get_owned_agent(owner, agent).paused
    finally:
        await service.runner.close()


async def test_restart_repairs_failed_cleanup_preserving_saved_outcome_and_other_blockers(
    lifecycle,
):
    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Old task")
    with repo.database.transaction() as connection:
        connection.execute(
            'UPDATE agents SET paused=1,'
            'blockers_json=\'["recovery_error","model_setup_required"]\' WHERE id=?',
            (agent,),
        )
        connection.execute(
            "UPDATE runs SET status='stopping',cleanup_state='failed',"
            "intended_terminal_status='completed_with_errors',"
            "intended_termination_reason='turn_complete' WHERE id=?",
            (run.id,),
        )
    await service.control(owner, agent, "restart", str(uuid.uuid4()))
    ended = repo.get_owned_run(owner, run.id)[0]
    assert ended.status == "completed_with_errors" and ended.cleanup_state == "confirmed"
    current = repo.get_owned_agent(owner, agent)
    assert current.paused and current.blockers == ("model_setup_required",)
    assert agent in runtime.running


async def test_concurrent_wake_eviction_preserves_two_container_cap_without_deadlock(lifecycle):
    import asyncio

    service, repo, owner, first, root, runtime, clock = lifecycle
    others = []
    for name in ("Concurrent second", "Concurrent third"):
        agent, _ = repo.create_agent(owner, name, "gpt-6-luna", str(uuid.uuid4()))
        repo.set_agent_status(agent.id, "ready")
        others.append(agent.id)
    await service.control(owner, first, "wake", str(uuid.uuid4()))
    # Advance monotonic time so the first container is strictly oldest.
    clock[0] = 1
    entered, release = asyncio.Event(), asyncio.Event()
    original = runtime.start_agent

    async def delayed_start(agent_id):
        if agent_id == others[0]:
            entered.set()
            await release.wait()
        await original(agent_id)
        assert len(runtime.running) <= 2

    runtime.start_agent = delayed_start
    one = asyncio.create_task(service.control(owner, others[0], "wake", str(uuid.uuid4())))
    await asyncio.wait_for(entered.wait(), 1)
    two = asyncio.create_task(service.control(owner, others[1], "wake", str(uuid.uuid4())))
    release.set()
    async with asyncio.timeout(2):
        await asyncio.gather(one, two)
    assert runtime.running == set(others)


def test_lifecycle_http_ownership_confirmation_and_csrf(settings):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    with TestClient(
        create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    ) as client:
        headers = {"Origin": settings.allowed_origin, "X-Request-ID": str(uuid.uuid4())}
        user = client.post(
            "/api/auth/register",
            headers=headers,
            json={"email": "control-owner@example.test", "password": "correct horse battery"},
        ).json()
        headers["X-CSRF-Token"] = user["csrf_token"]
        agent = client.post(
            "/api/agents", headers=headers, json={"name": "Owned", "model": "gpt-6-luna"}
        ).json()["id"]
        base = f"/api/agents/{agent}"
        assert client.post(base + "/wake").status_code == 403
        assert (
            client.post(
                base + "/reset-model-context", headers=headers, json={"confirmed": False}
            ).status_code
            == 422
        )
        assert client.get(base).json()["chat_epoch"] == 0
        assert client.post(base + "/wake", headers=headers).status_code == 200
        client.post("/api/auth/logout", headers=headers)
        other = client.post(
            "/api/auth/register",
            headers=headers,
            json={"email": "control-other@example.test", "password": "correct horse battery"},
        ).json()
        headers["X-CSRF-Token"] = other["csrf_token"]
        for action in ("wake", "sleep", "restart", "reset-model-context"):
            assert (
                client.post(
                    base + "/" + action, headers=headers, json={"confirmed": True}
                ).status_code
                == 404
            )
        assert agent in client.app.state.agent_service.preparer.running


def test_archive_export_canonical_tree_and_protected_reads(lifecycle):
    import hashlib
    import io

    service, repo, owner, agent, root, runtime, clock = lifecycle
    first, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), 'Saved "text"')
    with repo.database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET status='failed',cleanup_state='confirmed',"
            "run_token_hash='must-not-export' WHERE id=?",
            (first.id,),
        )
        connection.execute(
            "INSERT INTO events VALUES (?,2,'message_snapshot',NULL,?,1)",
            (first.id, json.dumps({"text": "Saved output"})),
        )
    retry, _, _ = repo.retry_chat_run(owner, first.id, str(uuid.uuid4()))
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (retry.id,))
    repo.database.storage.enabled = True
    repo.database.storage.trip("platform_quota_high")
    one, two = io.BytesIO(), io.BytesIO()
    digest, size = repo.export_run_archive(owner, [first.id], one)
    again, _ = repo.export_run_archive(owner, [first.id], two)
    assert one.getvalue() == two.getvalue() and digest == again
    assert digest == hashlib.sha256(one.getvalue()).hexdigest()
    assert size == len(one.getvalue())
    data = json.loads(one.getvalue())
    assert data["root_ids"] == [first.id]
    assert {item["run"]["id"] for item in data["runs"]} == {first.id, retry.id}
    saved = next(item for item in data["runs"] if item["run"]["id"] == first.id)
    assert saved["events"][1]["payload"] == {"text": "Saved output"}
    assert b"must-not-export" not in one.getvalue()
    assert repo.get_owned_run(owner, first.id)[0].status == "failed"


def test_archive_export_rejects_live_tree_ownership_and_oversized_artifact(lifecycle):
    import io

    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Saved")
    with pytest.raises(RepositoryError, match="terminal"):
        repo.export_run_archive(owner, [run.id], io.BytesIO())
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='failed' WHERE id=?", (run.id,))
    with pytest.raises(RepositoryError, match="not found"):
        repo.export_run_archive(str(uuid.uuid4()), [run.id], io.BytesIO())
    with pytest.raises(RepositoryError, match="smaller"):
        repo.export_run_archive(owner, [run.id], io.BytesIO(), max_bytes=10)
    retry, _, _ = repo.retry_chat_run(owner, run.id, str(uuid.uuid4()))
    with pytest.raises(RepositoryError, match="terminal"):
        repo.export_run_archive(owner, [run.id], io.BytesIO())
    with repo.database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET status='cancelled',cleanup_state='failed' WHERE id=?", (retry.id,)
        )
    with pytest.raises(RepositoryError, match="cleanup"):
        repo.export_run_archive(owner, [run.id], io.BytesIO())


def test_archive_export_http_download_csrf_isolation_and_storage_freeze(settings):
    import hashlib

    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    with TestClient(
        create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    ) as client:
        account = client.post(
            "/api/auth/register",
            headers={
                "origin": settings.allowed_origin,
                "x-request-id": str(uuid.uuid4()),
            },
            json={
                "email": "archive@example.test",
                "password": "correct horse battery",
            },
        ).json()
        headers = {
            "origin": settings.allowed_origin,
            "x-csrf-token": account["csrf_token"],
            "x-request-id": str(uuid.uuid4()),
        }
        agent = client.post(
            "/api/agents", headers=headers, json={"name": "Archive", "model": "gpt-6-luna"}
        ).json()["id"]
        repo = client.app.state.repository
        owner = repo.get_owned_agent(account["user"]["id"], agent).owner_id
        run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Archive me")
        with repo.database.transaction() as connection:
            connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
        listing = f"/api/agents/{agent}/runs"
        assert isinstance(client.get(listing).json(), list)
        page = client.get(listing + "?archive_roots=true&limit=1").json()
        assert [item['id'] for item in page['items']] == [run.id]
        assert page['next_cursor'] is None
        assert client.get(listing + "?archive_roots=true&cursor=invalid").status_code == 422
        path = "/api/runs/archive/export"
        assert client.post(path, json={"run_ids": [run.id]}).status_code == 403
        repo.database.storage.enabled = True
        repo.database.storage.trip("platform_quota_high")
        response = client.post(path, headers=headers, json={"run_ids": [run.id]})
        assert response.status_code == 200, response.text
        assert response.headers["x-archive-sha256"] == hashlib.sha256(response.content).hexdigest()
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["root_ids"] == [run.id]
        check_path = "/api/runs/archive/check"
        check_body = {"run_ids": [run.id], "sha256": response.headers["x-archive-sha256"]}
        assert client.post(check_path, json=check_body).status_code == 403
        checked = client.post(check_path, headers=headers, json=check_body)
        assert checked.status_code == 200, checked.text
        assert checked.json()["deletion_available"] is False
        assert "storage_repair_required" in checked.json()["blockers"]
        assert repo.get_owned_run(owner, run.id)[0].status == "cancelled"

        assert not list(client.app.state.workspace.staging_root.iterdir())
        assert (
            client.post(path, headers=headers, json={"run_ids": [str(uuid.uuid4())]}).status_code
            == 404
        )
        slot = client.app.state.archive_slot
        client.portal.call(slot.acquire)
        try:
            assert client.post(path, headers=headers, json={'run_ids': [run.id]}).status_code == 429
        finally:
            client.portal.call(slot.release)
        repo.database.storage.enabled = False


async def test_archive_download_releases_file_and_slot_when_client_disconnects():
    import asyncio
    import io

    from starlette.requests import ClientDisconnect

    from backend.app.api.runs import ArchiveDownload

    artifact = io.BytesIO(b'{"saved":true}\n')
    slot = asyncio.Semaphore(1)
    await slot.acquire()
    response = ArchiveDownload(artifact, slot, "digest", 15)
    started = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body":
            started.set()
            raise OSError("client disconnected")

    async def receive():
        await asyncio.Future()

    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert started.is_set() and artifact.closed and not slot.locked()


def test_archive_export_uses_one_snapshot_during_concurrent_event_write(lifecycle):
    import io

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Snapshot")
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))

    class ConcurrentOutput(io.BytesIO):
        changed = False

        def write(self, data):
            if not self.changed:
                self.changed = True
                with repo.database.transaction() as connection:
                    connection.execute(
                        "INSERT INTO events VALUES (?,2,'state',NULL,'{}',1)",
                        (run.id,),
                    )
            return super().write(data)

    output = ConcurrentOutput()
    repo.export_run_archive(owner, [run.id], output)
    assert len(json.loads(output.getvalue())["runs"][0]["events"]) == 1
    with repo.database.read() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (run.id,)).fetchone()[
                0
            ]
            == 2
        )


async def test_archive_download_timeout_releases_stalled_client(monkeypatch):
    import asyncio
    import io

    from backend.app.api.runs import ArchiveDownload

    original_timeout = asyncio.timeout
    monkeypatch.setattr(asyncio, 'timeout', lambda seconds: original_timeout(0.01))
    artifact = io.BytesIO(b'archive')
    slot = asyncio.Semaphore(1)
    await slot.acquire()

    async def stalled_send(message):
        await asyncio.Future()

    async def receive():
        await asyncio.Future()

    with pytest.raises(TimeoutError):
        await ArchiveDownload(artifact, slot, 'digest', 7)(
            {'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, stalled_send,
        )
    assert artifact.closed and not slot.locked()


def test_archive_preparation_honors_cancellation(lifecycle):
    import io
    import threading

    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), 'Cancel export')
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
    cancelled = threading.Event()

    class CancelOutput(io.BytesIO):
        def write(self, data):
            cancelled.set()
            return super().write(data)

    with pytest.raises(RepositoryError, match='interrupted'):
        repo.export_run_archive(owner, [run.id], CancelOutput(), cancelled=cancelled)
    assert repo.get_owned_run(owner, run.id)[0].status == 'cancelled'


def test_archive_root_pages_include_jobs_and_stable_tied_timestamp_order(lifecycle):
    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    ids = []
    with repo.database.transaction() as connection:
        job = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO jobs(id,agent_id,name,prompt,created_by,run_at,created_at) "
            "VALUES (?,?,'Archive job','Job output','user',1,1)", (job, agent),
        )
    for index in range(5):
        run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), f'Page {index}')
        ids.append(run.id)
        with repo.database.transaction() as connection:
            connection.execute("UPDATE runs SET status='cancelled',created_at=100 WHERE id=?",
                               (run.id,))
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET source='job',job_id=? WHERE id=?", (job, ids[0]))
    repo.database.storage.enabled = True
    repo.database.storage.trip('platform_quota_high')
    loaded, cursor = [], None
    while True:
        rows, cursor = repo.list_archive_roots(owner, agent, cursor=cursor, limit=2)
        loaded.extend(rows)
        if cursor is None:
            break
    assert [run.id for run in loaded] == sorted(ids, reverse=True)
    assert {run.source for run in loaded} == {'chat', 'job'}
    with pytest.raises(RepositoryError, match='not found'):
        repo.list_archive_roots(str(uuid.uuid4()), agent)
    with pytest.raises(RepositoryError):
        repo.list_archive_roots(owner, agent, cursor='not-a-uuid')


def test_archive_purge_check_digest_dependencies_and_protected_reads(lifecycle):
    import io

    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Check cleanup")
    with repo.database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET status='cancelled',thread_id='saved-thread' WHERE id=?", (run.id,)
        )
        connection.execute(
            "UPDATE agents SET paused=1,chat_thread_id='saved-thread' WHERE id=?", (agent,)
        )
    digest, size = repo.export_run_archive(owner, [run.id], io.BytesIO())
    plan = repo.check_archive_purge(owner, [run.id], digest)
    assert not plan["eligible"] and "current_model_context" in plan["blockers"]
    with repo.database.transaction() as connection:
        connection.execute(
            "UPDATE agents SET chat_thread_id=NULL,chat_epoch=1 WHERE id=?", (agent,)
        )
    plan = repo.check_archive_purge(owner, [run.id], digest)
    assert plan["eligible"] and plan["run_count"] == 1 and plan["event_count"] == 1
    assert plan["export_bytes"] == size and plan["sha256"] == digest
    with pytest.raises(RepositoryError, match="changed"):
        repo.check_archive_purge(owner, [run.id], "0" * 64)
    with pytest.raises(RepositoryError, match="not found"):
        repo.check_archive_purge(str(uuid.uuid4()), [run.id], digest)
    repo.database.storage.enabled = True
    repo.database.storage.trip("platform_quota_high")
    plan = repo.check_archive_purge(owner, [run.id], digest)
    assert "storage_repair_required" in plan["blockers"] and not plan["eligible"]
    assert repo.get_owned_run(owner, run.id)[0].status == "cancelled"


def test_archive_purge_check_blocks_external_thread_and_active_work(lifecycle):
    import io

    service, repo, owner, agent, root, runtime, clock = lifecycle
    saved, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Saved root")
    other, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Unselected dependency")
    with repo.database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET status='cancelled',thread_id='shared' WHERE id IN (?,?)",
            (saved.id, other.id),
        )
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    digest, _ = repo.export_run_archive(owner, [saved.id], io.BytesIO())
    assert (
        "unselected_thread_reference"
        in repo.check_archive_purge(
            owner,
            [saved.id],
            digest,
        )["blockers"]
    )
    with repo.database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET status='running',cleanup_state='pending' WHERE id=?", (other.id,)
        )
        connection.execute("UPDATE agents SET paused=0 WHERE id=?", (agent,))
    plan = repo.check_archive_purge(owner, [saved.id], digest)
    assert {"pause_required", "run_cleanup_pending"} <= set(plan["blockers"])


def test_archive_check_preserves_current_context_with_missing_run_thread_id(lifecycle):
    import io

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Missing saved thread")
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
        connection.execute(
            "UPDATE agents SET paused=1,chat_thread_id='current-sdk-thread' WHERE id=?", (agent,)
        )
    digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
    assert "current_model_context" in repo.check_archive_purge(owner, [run.id], digest)["blockers"]


@pytest.mark.parametrize("termination", ["disconnect", "timeout", "cancel"])
async def test_archive_check_disconnect_collects_inventory_and_releases_slot(
    lifecycle, monkeypatch, termination,
):
    import asyncio
    import io
    from types import SimpleNamespace

    from fastapi import Response

    from backend.app.api import runs as api
    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), 'Disconnect check')
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
    digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
    entered, disconnected, collected = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def inventory():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            collected.set()

    async def is_disconnected():
        return disconnected.is_set()

    runtime.running_agent_ids = inventory
    if termination == "timeout":
        monkeypatch.setattr(api, 'ARCHIVE_CHECK_TIMEOUT_SECONDS', 0.15)
    monkeypatch.setattr(api, '_current_user', lambda request: SimpleNamespace(id=owner))
    state = SimpleNamespace(
        repository=repo, workspace=service.workspace, agent_service=service,
        auth=SimpleNamespace(require_origin=lambda value: None, require_csrf=lambda *args: None),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state), headers={},
                              is_disconnected=is_disconnected)
    task = asyncio.create_task(api.check_archive_purge(
        request, api.ArchiveCheckRequest(run_ids=[run.id], sha256=digest), Response(),
    ))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if termination == "disconnect":
            disconnected.set()
        elif termination == "cancel":
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done, 'Disconnected cleanup check retained the global archive slot.'
        if termination == "cancel":
            with pytest.raises(asyncio.CancelledError):
                task.result()
        else:
            with pytest.raises(RepositoryError, match='interrupted'):
                task.result()
        assert collected.is_set() and not state.archive_slot.locked()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_archive_purge_engine_deletes_tree_and_preserves_tombstones_and_files(lifecycle):
    import io

    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    message_id, retry_id, purge_id = (str(uuid.uuid4()) for _ in range(3))
    first, _ = repo.create_chat_run(owner, agent, message_id, "Purge saved tree")
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='failed' WHERE id=?", (first.id,))
    retry, _, _ = repo.retry_chat_run(owner, first.id, retry_id)
    other, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Keep another root")
    with repo.database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET status='cancelled' WHERE id IN (?,?)", (retry.id, other.id)
        )
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
        connection.execute("INSERT INTO model_run_starts(started_at) VALUES (1)")
    (root / "workspace" / "keep.txt").write_bytes(b"keep workspace")
    (root / "codex" / "keep.txt").write_bytes(b"keep model files")
    digest, _ = repo.export_run_archive(owner, [first.id], io.BytesIO())
    result = repo.purge_archive_records(owner, [first.id], digest, purge_id, confirmed=True)
    assert result["purged"] and result["run_count"] == 2 and not result["replayed"]
    assert result["reclamation_pending"] and result["reclaimed_bytes"] == 0
    replay = repo.purge_archive_records(owner, [first.id], digest, purge_id, confirmed=True)
    assert replay["replayed"] and replay["run_count"] == 2
    with repo.database.read() as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM events WHERE run_id IN (?,?)", (first.id, retry.id)
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT COUNT(*) FROM model_run_starts").fetchone()[0] == 1
        assert {
            row[0]
            for row in connection.execute(
                "SELECT operation_state FROM idempotency WHERE entity_id IN (?,?)",
                (first.id, retry.id),
            )
        } == {"purged"}
    for action in (
        lambda: repo.create_chat_run(owner, agent, message_id, "Purge saved tree"),
        lambda: repo.retry_chat_run(owner, first.id, retry_id),
    ):
        with pytest.raises(RepositoryError) as raised:
            action()
        assert raised.value.status_code == 410
    assert repo.get_owned_run(owner, other.id)[0].prompt == "Keep another root"
    assert (root / "workspace" / "keep.txt").read_bytes() == b"keep workspace"
    assert (root / "codex" / "keep.txt").read_bytes() == b"keep model files"


def test_archive_purge_engine_requires_confirmation_digest_and_healthy_storage(lifecycle):
    import io

    from backend.app.db import StorageWriteBlocked
    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Protect archive")
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
    for confirmed, supplied in ((False, digest), (True, "0" * 64)):
        with pytest.raises(RepositoryError):
            repo.purge_archive_records(
                owner, [run.id], supplied, str(uuid.uuid4()), confirmed=confirmed
            )
    repo.database.storage.enabled = True
    repo.database.storage.trip("platform_quota_high")
    with pytest.raises(StorageWriteBlocked):
        repo.purge_archive_records(owner, [run.id], digest, str(uuid.uuid4()), confirmed=True)
    assert repo.get_owned_run(owner, run.id)[0].status == "cancelled"


@pytest.mark.parametrize("change", ["event", "retry", "context"])
def test_archive_purge_rechecks_changes_after_preflight(lifecycle, change):
    import io

    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Old export")
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='failed' WHERE id=?", (run.id,))
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
    assert repo.check_archive_purge(owner, [run.id], digest)["eligible"]
    if change == "retry":
        repo.retry_chat_run(owner, run.id, str(uuid.uuid4()))
    else:
        with repo.database.transaction() as connection:
            if change == "event":
                connection.execute("INSERT INTO events VALUES (?,2,'state',NULL,'{}',1)", (run.id,))
            else:
                connection.execute(
                    "UPDATE agents SET chat_thread_id='new-context' WHERE id=?", (agent,)
                )
    with pytest.raises(RepositoryError):
        repo.purge_archive_records(owner, [run.id], digest, str(uuid.uuid4()), confirmed=True)
    assert repo.get_owned_run(owner, run.id)[0].status == "failed"


def test_archive_purge_rolls_back_deletion_tombstones_and_receipt(lifecycle):
    import io
    import sqlite3

    service, repo, owner, agent, root, runtime, clock = lifecycle
    message_id, purge_id = str(uuid.uuid4()), str(uuid.uuid4())
    run, _ = repo.create_chat_run(owner, agent, message_id, "Rollback history")
    other, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Rollback second root")
    roots = [run.id, other.id]
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE agent_id=?", (agent,))
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
        connection.execute(
            "CREATE TRIGGER reject_purge BEFORE DELETE ON runs "
            f"WHEN OLD.id='{max(roots)}' BEGIN SELECT RAISE(ABORT,'injected purge failure'); END"
        )
    digest, _ = repo.export_run_archive(owner, roots, io.BytesIO())
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        repo.purge_archive_records(owner, roots, digest, purge_id, confirmed=True)
    with repo.database.read() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (run.id,)).fetchone()[
                0
            ]
            == 1
        )
        assert (
            connection.execute(
                "SELECT operation_state FROM idempotency WHERE request_id=?", (message_id,)
            ).fetchone()[0]
            == "accepted"
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM idempotency WHERE operation='archive.purge'",
            ).fetchone()[0]
            == 0
        )
    assert repo.get_owned_run(owner, run.id)[0].status == "cancelled"
    assert repo.get_owned_run(owner, other.id)[0].status == "cancelled"


def test_archive_purge_retains_job_and_replays_after_restart_and_storage_trip(lifecycle):
    import io

    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Saved job run")
    job, purge = str(uuid.uuid4()), str(uuid.uuid4())
    with repo.database.transaction() as connection:
        connection.execute(
            "INSERT INTO jobs(id,agent_id,name,prompt,created_by,run_at,created_at) "
            "VALUES (?,?,'Keep job','Future work','user',9999999999999,1)",
            (job, agent),
        )
        connection.execute(
            "UPDATE runs SET source='job',job_id=?,status='cancelled' WHERE id=?", (job, run.id)
        )
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
    repo.purge_archive_records(owner, [run.id], digest, purge, confirmed=True)
    restarted = Repository(repo.database)
    repo.database.storage.enabled = True
    repo.database.storage.trip("platform_quota_high")
    assert restarted.purge_archive_records(owner, [run.id], digest, purge, confirmed=True)[
        "replayed"
    ]
    with pytest.raises(RepositoryError) as error:
        restarted.purge_archive_records(owner, [run.id], "0" * 64, purge, confirmed=True)
    assert error.value.status_code == 409
    with repo.database.read() as connection:
        row = connection.execute("SELECT enabled,prompt FROM jobs WHERE id=?", (job,)).fetchone()
        assert row["enabled"] == 1 and row["prompt"] == "Future work"


@pytest.mark.parametrize("failure", ["cancel", "storage_trip"])
def test_archive_purge_late_abort_rolls_back_rows_and_receipts(lifecycle, monkeypatch, failure):
    import io
    import threading

    from backend.app.db import StorageWriteBlocked
    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Abort before commit")
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
        connection.execute(
            "CREATE TRIGGER interrupt_purge AFTER DELETE ON runs "
            "BEGIN SELECT interrupt_archive(); END"
        )
    digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
    cancelled = threading.Event()
    original_connect = repo.database.connect
    repo.database.storage.enabled = True
    repo.database.storage.checked = True

    def interrupt():
        if failure == "cancel":
            cancelled.set()
        else:
            repo.database.storage.trip("platform_quota_high")
        return 0

    def connect():
        connection = original_connect()
        connection.create_function("interrupt_archive", 0, interrupt)
        return connection

    monkeypatch.setattr(repo.database, "connect", connect)
    expected = RepositoryError if failure == "cancel" else StorageWriteBlocked
    with pytest.raises(expected):
        repo.purge_archive_records(
            owner, [run.id], digest, str(uuid.uuid4()), confirmed=True, cancelled=cancelled
        )
    with repo.database.read() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM runs WHERE id=?", (run.id,)).fetchone()[0] == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (run.id,)).fetchone()[
                0
            ]
            == 1
        )
        assert (
            connection.execute(
                "SELECT operation_state FROM idempotency WHERE entity_id=?", (run.id,)
            ).fetchone()[0]
            == "accepted"
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM idempotency WHERE operation='archive.purge'"
            ).fetchone()[0]
            == 0
        )


def test_archive_purge_sqlite_cancellation_does_not_trip_storage(lifecycle):
    import io
    import threading

    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Cancel large validation")
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
        connection.executemany(
            "INSERT INTO events VALUES (?,?,'state',NULL,'{}',1)",
            ((run.id, sequence) for sequence in range(2, 5002)),
        )
    digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
    repo.database.storage.enabled = True
    repo.database.storage.checked = True
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RepositoryError) as error:
        repo.purge_archive_records(
            owner, [run.id], digest, str(uuid.uuid4()), confirmed=True, cancelled=cancelled
        )
    assert error.value.status_code == 408
    assert not repo.database.storage.status()["active"]
    assert repo.get_owned_run(owner, run.id)[0].status == "cancelled"


@pytest.mark.parametrize("same_request", [True, False])
def test_concurrent_archive_purge_has_one_committed_deletion(lifecycle, same_request):
    import io
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from backend.app.repository import RepositoryError

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Concurrent purge")
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
    digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
    barrier = threading.Barrier(2)
    request = str(uuid.uuid4())
    ids = [request, request if same_request else str(uuid.uuid4())]

    def execute(request_id):
        barrier.wait(timeout=2)
        try:
            return repo.purge_archive_records(owner, [run.id], digest, request_id, confirmed=True)
        except RepositoryError as error:
            return {"error": error.status_code}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, ids))
    assert sum(result.get("purged", False) and not result["replayed"] for result in results) == 1
    if same_request:
        assert sum(result.get("replayed", False) for result in results) == 1
    else:
        assert sum(result.get("error") == 404 for result in results) == 1
    with repo.database.read() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM idempotency WHERE operation='archive.purge'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM runs WHERE id=?", (run.id,)).fetchone()[0] == 0
        )


def test_internal_reclamation_compacts_real_sqlite_and_preserves_saved_data(database):
    from backend.app.db import StorageMaintenanceError

    with database.transaction() as connection:
        connection.execute("CREATE TABLE maintenance_fixture(id INTEGER PRIMARY KEY, value BLOB)")
        connection.executemany(
            "INSERT INTO maintenance_fixture VALUES (?,?)",
            ((index, b"x" * 8192) for index in range(256)),
        )
        connection.execute("DELETE FROM maintenance_fixture WHERE id > 0")
    sample = {
        "quota": {"used_bytes": 16 * 1024**2, "hard_bytes": 2 * 1024**3},
        "free_bytes": 4 * 1024**3,
    }
    result = database.reclaim_space(sample)
    assert result["reclaimed_bytes"] > 0
    assert result["after_allocated_bytes"] < result["before_allocated_bytes"]
    with database.read() as connection:
        assert (
            connection.execute("SELECT value FROM maintenance_fixture").fetchone()[0] == b"x" * 8192
        )
        assert connection.execute("PRAGMA freelist_count").fetchone()[0] == 0
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not database.storage.status()["maintenance_active"]
    with pytest.raises(StorageMaintenanceError) as error:
        database.reclaim_space(
            {
                "quota": {"used_bytes": 2 * 1024**3 - 1, "hard_bytes": 2 * 1024**3},
                "free_bytes": 4 * 1024**3,
            }
        )
    assert error.value.code == "maintenance_space_required"


def test_maintenance_excludes_unrelated_writes_but_allows_saved_reads(database):
    from concurrent.futures import ThreadPoolExecutor

    from backend.app.db import StorageWriteBlocked

    with database.storage.maintenance():
        with database.transaction() as connection:
            connection.execute("INSERT INTO runtime_metadata VALUES ('maintenance-test','keep')")

        def ordinary_write():
            with pytest.raises(StorageWriteBlocked):
                with database.transaction():
                    pass
            with database.read() as connection:
                return connection.execute(
                    "SELECT value FROM runtime_metadata WHERE key='maintenance-test'"
                ).fetchone()[0]

        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(ordinary_write).result(timeout=2) == "keep"
    with database.transaction() as connection:
        connection.execute("DELETE FROM runtime_metadata WHERE key='maintenance-test'")


@pytest.mark.parametrize(
    "sample",
    [
        None,
        {},
        {"quota": {"used_bytes": True, "hard_bytes": 2 * 1024**3}, "free_bytes": 4 * 1024**3},
        {"quota": {"used_bytes": 0, "hard_bytes": 123}, "free_bytes": 4 * 1024**3},
    ],
)
def test_reclamation_rejects_unverified_measurements(database, sample):
    from backend.app.db import StorageMaintenanceError

    with pytest.raises(StorageMaintenanceError) as error:
        database.reclaim_space(sample)
    assert error.value.code == "maintenance_measurement"
    assert not database.storage.status()["maintenance_active"]


def test_reclamation_reader_conflict_is_retryable_and_preserves_data(database):
    from backend.app.db import StorageMaintenanceError

    with database.transaction() as connection:
        connection.execute("INSERT INTO runtime_metadata VALUES ('reader-test','original')")
    with database.read() as reader:
        reader.execute("BEGIN")
        assert (
            reader.execute("SELECT value FROM runtime_metadata WHERE key='reader-test'").fetchone()[
                0
            ]
            == "original"
        )
        with database.transaction() as connection:
            connection.execute("UPDATE runtime_metadata SET value='saved' WHERE key='reader-test'")
        sample = {
            "quota": {"used_bytes": 1024**2, "hard_bytes": 2 * 1024**3},
            "free_bytes": 4 * 1024**3,
        }
        with pytest.raises(StorageMaintenanceError) as error:
            database.reclaim_space(sample)
        assert error.value.code == "maintenance_busy"
        assert not database.storage.status()["maintenance_active"]
        assert not database.storage.status()["active"]
    with database.read() as connection:
        assert (
            connection.execute(
                "SELECT value FROM runtime_metadata WHERE key='reader-test'"
            ).fetchone()[0]
            == "saved"
        )


def test_reclamation_protection_and_cancellation_never_grant_write_authority(database):
    import threading

    from backend.app.db import StorageMaintenanceError, StorageWriteBlocked

    sample = {
        "quota": {"used_bytes": 1024**2, "hard_bytes": 2 * 1024**3},
        "free_bytes": 4 * 1024**3,
    }
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(StorageMaintenanceError) as error:
        database.reclaim_space(sample, cancelled=cancelled)
    assert error.value.code == "maintenance_cancelled"
    assert not database.storage.status()["maintenance_active"]
    database.storage.enabled = True
    database.storage.trip("platform_quota_high")
    with pytest.raises(StorageWriteBlocked):
        database.reclaim_space(sample)
    database.storage.can_recover = True
    with database.storage.repair():
        with pytest.raises(StorageWriteBlocked):
            database.reclaim_space(sample)
    assert database.storage.status()["active"]
    assert not database.storage.status()["maintenance_active"]


def test_reclamation_corrupt_database_latches_storage_before_releasing_maintenance(database):
    import sqlite3

    sample = {
        "quota": {"used_bytes": 1024**2, "hard_bytes": 2 * 1024**3},
        "free_bytes": 4 * 1024**3,
    }
    database.path.write_bytes(b"not a SQLite database" * 512)
    database.storage.enabled = True
    database.storage.observe(sample)
    with pytest.raises(sqlite3.DatabaseError):
        database.reclaim_space(sample)
    assert database.storage.status()["active"]
    assert database.storage.status()["reason"] == "database_maintenance_failed"
    assert not database.storage.status()["maintenance_active"]


def test_maintenance_rejects_commit_of_previously_admitted_writer(database):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from backend.app.db import StorageWriteBlocked

    started, release = threading.Event(), threading.Event()

    def writer():
        try:
            with database.transaction() as connection:
                connection.execute("INSERT INTO runtime_metadata VALUES ('late-writer','rollback')")
                started.set()
                assert release.wait(timeout=2)
        except StorageWriteBlocked:
            return "blocked"
        return "committed"

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(writer)
        try:
            assert started.wait(timeout=2)
            with database.storage.maintenance():
                release.set()
                assert pending.result(timeout=2) == "blocked"
        finally:
            release.set()
    with database.read() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM runtime_metadata WHERE key='late-writer'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("interrupt", ["cancel", "storage_trip"])
def test_vacuum_interruption_preserves_records_and_releases_maintenance(
    database, monkeypatch, interrupt
):
    import threading

    from backend.app.db import StorageMaintenanceError, StorageWriteBlocked

    with database.transaction() as connection:
        connection.execute("CREATE TABLE vacuum_fixture(id INTEGER PRIMARY KEY, value BLOB)")
        connection.executemany(
            "INSERT INTO vacuum_fixture VALUES (?,?)",
            ((index, b"x" * 8192) for index in range(256)),
        )
        connection.execute("DELETE FROM vacuum_fixture WHERE id > 0")
    sample = {
        "quota": {"used_bytes": 16 * 1024**2, "hard_bytes": 2 * 1024**3},
        "free_bytes": 4 * 1024**3,
    }
    database.storage.enabled = True
    database.storage.observe(sample)
    cancelled, entered = threading.Event(), threading.Event()
    original_connect = database.connect

    def trace(statement):
        if statement == "VACUUM":
            entered.set()
            if interrupt == "cancel":
                cancelled.set()
            else:
                database.storage.trip("platform_quota_high")

    def connect():
        connection = original_connect()
        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(database, "connect", connect)
    expected = StorageMaintenanceError if interrupt == "cancel" else StorageWriteBlocked
    with pytest.raises(expected):
        database.reclaim_space(sample, cancelled=cancelled)
    assert entered.is_set() and not database.storage.status()["maintenance_active"]
    assert database.storage.status()["active"] == (interrupt == "storage_trip")
    with database.read() as connection:
        assert connection.execute("SELECT value FROM vacuum_fixture").fetchone()[0] == b"x" * 8192
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_maintenance_scope_authority_cannot_outlive_its_generation(database):
    from contextvars import copy_context

    from backend.app.db import StorageWriteBlocked

    with database.storage.maintenance():
        old_context = copy_context()
    with database.storage.maintenance():

        def stale_writer():
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO runtime_metadata VALUES ('stale-scope','forbidden')"
                )

        with pytest.raises(StorageWriteBlocked):
            old_context.run(stale_writer)


@pytest.fixture
def purge_maintenance(lifecycle):
    import io

    service, repo, owner, agent, root, runtime, clock = lifecycle
    run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Recoverable maintenance")
    with repo.database.transaction() as connection:
        connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
        connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
        connection.executemany(
            "INSERT INTO events VALUES (?,?,'state',NULL,?,1)",
            ((run.id, sequence, json.dumps({"saved": "x" * 8192})) for sequence in range(2, 258)),
        )
    digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
    measurements = []

    async def measure():
        sample = {
            "quota": {"used_bytes": repo.database._allocated_bytes(), "hard_bytes": 2 * 1024**3},
            "free_bytes": 4 * 1024**3,
        }
        measurements.append(sample)
        return sample

    runtime.platform_storage_usage = measure
    return service, repo, owner, agent, runtime, run.id, digest, measurements


async def test_purge_coordination_persists_reclamation_and_replays_without_new_work(
    purge_maintenance,
):
    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    request_id = str(uuid.uuid4())
    result = await service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    assert result["purged"] and not result["reclamation_pending"]
    assert result["reclaimed_bytes"] > 0 and result["reclamation_state"] == "completed"
    assert len(measurements) == 2
    replay = await service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    assert replay["replayed"] and replay["reclaimed_bytes"] == result["reclaimed_bytes"]
    assert len(measurements) == 2
    assert not repo.database.storage.status()["maintenance_active"]


async def test_purge_coordination_keeps_deletion_and_retries_reclamation(
    purge_maintenance, monkeypatch
):
    from backend.app.db import StorageMaintenanceError

    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    original = repo.database.reclaim_space

    def busy(sample, **kwargs):
        if kwargs.get("_check_only"):
            return original(sample, **kwargs)
        raise StorageMaintenanceError("maintenance_busy", "Reader still active.")

    monkeypatch.setattr(repo.database, "reclaim_space", busy)
    request_id = str(uuid.uuid4())
    result = await service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    assert result["purged"] and result["reclamation_pending"]
    assert result["maintenance_error"] == "maintenance_busy"
    with repo.database.read() as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs WHERE id=?", (run,)).fetchone()[0] == 0
    monkeypatch.setattr(repo.database, "reclaim_space", original)
    replay = await service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    assert replay["replayed"] and not replay["reclamation_pending"]


@pytest.mark.parametrize("blocked", ["budget", "container"])
async def test_purge_coordination_checks_admission_before_deletion(purge_maintenance, blocked):
    from backend.app.db import StorageMaintenanceError
    from backend.app.repository import RepositoryError

    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    if blocked == "container":
        runtime.running.add(agent)
    else:

        async def insufficient():
            return {
                "quota": {"used_bytes": 2 * 1024**3 - 1, "hard_bytes": 2 * 1024**3},
                "free_bytes": 4 * 1024**3,
            }

        runtime.platform_storage_usage = insufficient
    with pytest.raises((StorageMaintenanceError, RepositoryError)):
        await service.purge_archive(owner, [run], digest, str(uuid.uuid4()), confirmed=True)
    assert repo.get_owned_run(owner, run)[0].status == "cancelled"
    assert not repo.database.storage.status()["maintenance_active"]


async def test_purge_coordination_cancel_after_delete_keeps_recoverable_receipt(
    purge_maintenance, monkeypatch
):
    import asyncio
    import threading
    import time

    from backend.app.db import StorageMaintenanceError

    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    entered = threading.Event()
    original = repo.database.reclaim_space

    def blocked(sample, **kwargs):
        if kwargs.get("_check_only"):
            return original(sample, **kwargs)
        entered.set()
        deadline = time.monotonic() + 2
        while not kwargs["cancelled"].wait(0.01):
            assert time.monotonic() < deadline
        raise StorageMaintenanceError("maintenance_cancelled", "Cancelled compaction.")

    monkeypatch.setattr(repo.database, "reclaim_space", blocked)
    request_id = str(uuid.uuid4())
    task = asyncio.create_task(
        service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    )
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    saved = repo.archive_purge_result(owner, [run], digest, request_id)
    assert saved["purged"] and saved["reclamation_pending"]
    assert saved["maintenance_error"] == "maintenance_cancelled"
    assert not repo.database.storage.status()["maintenance_active"]
    monkeypatch.setattr(repo.database, "reclaim_space", original)
    replay = await service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    assert replay["replayed"] and not replay["reclamation_pending"]


async def test_purge_coordination_restart_does_not_invent_previous_reclaimed_bytes(
    purge_maintenance,
):
    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    request_id = str(uuid.uuid4())
    repo.purge_archive_records(owner, [run], digest, request_id, confirmed=True)
    repo.record_archive_reclamation(owner, request_id, "running")
    # Simulate a crash after compaction but before its outcome was persisted.
    repo.database.reclaim_space(await runtime.platform_storage_usage())
    service.repository = Repository(repo.database)
    result = await service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    assert result["previous_reclamation_unconfirmed"]
    assert result["reclamation_scope"] == "database_files_compaction_interval"
    assert result["reclaimed_bytes"] == result["reclamation_measurement"]["reclaimed_bytes"]
    assert not result["reclamation_pending"]


async def test_concurrent_purge_coordination_compacts_only_once(purge_maintenance, monkeypatch):
    import asyncio

    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    original = repo.database.reclaim_space
    calls = []

    def counted(sample, **kwargs):
        if not kwargs.get("_check_only"):
            calls.append("compaction")
        return original(sample, **kwargs)

    monkeypatch.setattr(repo.database, "reclaim_space", counted)
    request_id = str(uuid.uuid4())
    results = await asyncio.gather(
        *(service.purge_archive(owner, [run], digest, request_id, confirmed=True) for _ in range(2))
    )
    assert len(calls) == 1 and sum(result["replayed"] for result in results) == 1


async def test_purge_coordination_storage_trip_after_delete_keeps_unconfirmed_receipt(
    purge_maintenance,
):
    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    original = runtime.platform_storage_usage
    calls = []
    repo.database.storage.enabled = True
    repo.database.storage.checked = True

    async def lost_measurement():
        calls.append(True)
        return await original() if len(calls) == 1 else None

    runtime.platform_storage_usage = lost_measurement
    request_id = str(uuid.uuid4())
    result = await service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    assert result["purged"] and result["reclamation_pending"]
    assert result["maintenance_error"] == "storage_protection"
    assert result["reclamation_state"] == "unconfirmed"
    assert repo.database.storage.status()["active"]
    assert not repo.database.storage.status()["maintenance_active"]
    saved = repo.archive_purge_result(owner, [run], digest, request_id)
    assert saved["purged"] and saved["reclamation_state"] == "running"
    replay = await service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    assert replay["purged"] and replay["maintenance_error"] == "storage_protection"


async def test_purge_coordination_recovers_committed_result_hidden_by_cancelled_io(
    purge_maintenance, monkeypatch
):
    import asyncio
    import threading

    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    committed = threading.Event()
    original = repo.purge_archive_records

    def delay_result(*args, **kwargs):
        result = original(*args, **kwargs)
        committed.set()
        assert kwargs["cancelled"].wait(timeout=2)
        return result

    monkeypatch.setattr(repo, "purge_archive_records", delay_result)
    request_id = str(uuid.uuid4())
    task = asyncio.create_task(
        service.purge_archive(owner, [run], digest, request_id, confirmed=True)
    )
    assert await asyncio.to_thread(committed.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    saved = repo.archive_purge_result(owner, [run], digest, request_id)
    assert saved["purged"] and saved["reclamation_pending"]
    assert saved["maintenance_error"] == "maintenance_cancelled"
    assert not repo.database.storage.status()["maintenance_active"]


async def test_purge_coordination_replays_if_other_request_deletes_before_plan(
    purge_maintenance, monkeypatch
):
    import asyncio
    import threading
    from contextvars import ContextVar

    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    role = ContextVar("purge-test-role", default="first")
    blocked, release = threading.Event(), threading.Event()
    original = repo.check_archive_purge

    def delayed(*args, **kwargs):
        if role.get() == "second" and kwargs.get("_connection") is None:
            blocked.set()
            assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(repo, "check_archive_purge", delayed)
    request_id = str(uuid.uuid4())

    async def second():
        role.set("second")
        return await service.purge_archive(owner, [run], digest, request_id, confirmed=True)

    pending = asyncio.create_task(second())
    try:
        assert await asyncio.to_thread(blocked.wait, 2)
        first = await service.purge_archive(owner, [run], digest, request_id, confirmed=True)
        release.set()
        replay = await pending
        assert first["purged"] and replay["replayed"]
        assert replay["reclaimed_bytes"] == first["reclaimed_bytes"]
        assert len(measurements) == 2
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)


async def test_purge_coordination_missing_quota_trips_before_deletion(purge_maintenance):
    from backend.app.agents import AgentPreparationError

    service, repo, owner, agent, runtime, run, digest, measurements = purge_maintenance
    repo.database.storage.enabled = True
    repo.database.storage.checked = True

    async def unavailable():
        raise AgentPreparationError('quota_unavailable', 'Quota helper is unavailable.')

    runtime.platform_storage_usage = unavailable
    with pytest.raises(AgentPreparationError):
        await service.purge_archive(owner, [run], digest, str(uuid.uuid4()), confirmed=True)
    assert repo.database.storage.status()['active']
    assert repo.database.storage.status()['reason'] == 'quota_measurement_unavailable'
    assert repo.get_owned_run(owner, run)[0].status == 'cancelled'
    assert not repo.database.storage.status()['maintenance_active']


def test_confirmed_archive_purge_api_requires_auth_digest_and_replays(settings, monkeypatch):
    import io

    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    with TestClient(
        create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    ) as client:
        headers = {"Origin": settings.allowed_origin, "X-Request-ID": str(uuid.uuid4())}
        session = client.post(
            "/api/auth/register",
            headers=headers,
            json={"email": "purge-api@example.test", "password": "correct horse battery"},
        ).json()
        headers["X-CSRF-Token"] = session["csrf_token"]
        agent = client.post(
            "/api/agents", headers=headers, json={"name": "Purge API", "model": "gpt-6-luna"}
        ).json()["id"]
        repo = client.app.state.repository
        owner = session["user"]["id"]
        run, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "API saved data")
        with repo.database.transaction() as connection:
            connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (run.id,))
            connection.execute("UPDATE agents SET paused=1 WHERE id=?", (agent,))
        digest, _ = repo.export_run_archive(owner, [run.id], io.BytesIO())
        path = "/api/runs/archive/purge"
        body = {"run_ids": [run.id], "sha256": digest, "confirmed": True}
        assert client.post(path, json=body).status_code == 403
        assert (
            client.post(path, headers=headers, json={**body, "confirmed": False}).status_code == 422
        )
        assert (
            client.post(path, headers=headers, json={**body, "confirmed": "true"}).status_code
            == 422
        )
        assert (
            client.post(path, headers=headers, json={**body, "sha256": "0" * 64}).status_code == 409
        )
        assert (
            client.post(
                path, headers=headers, json={**body, "run_ids": [str(uuid.uuid4())]}
            ).status_code
            == 404
        )
        provider = client.app.state.agent_service.preparer
        original_measure = provider.platform_storage_usage

        async def insufficient():
            return {
                "quota": {"used_bytes": 2 * 1024**3 - 1, "hard_bytes": 2 * 1024**3},
                "free_bytes": 4 * 1024**3,
            }

        provider.platform_storage_usage = insufficient
        checked = client.post(
            "/api/runs/archive/check", headers=headers, json={"run_ids": [run.id], "sha256": digest}
        )
        assert checked.status_code == 200 and not checked.json()["deletion_available"]
        assert "maintenance_space_required" in checked.json()["blockers"]
        provider.platform_storage_usage = original_measure
        response = client.post(path, headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert response.json()["purged"] and not response.json()["reclamation_pending"]
        assert response.headers["cache-control"] == "no-store"
        replay = client.post(path, headers=headers, json=body)
        assert replay.status_code == 200 and replay.json()["replayed"]
        assert client.get(f"/api/runs/{run.id}").status_code == 404
        assert client.get(f"/api/agents/{agent}").status_code == 200
        from backend.app.db import StorageMaintenanceError

        second, _ = repo.create_chat_run(owner, agent, str(uuid.uuid4()), "Recover pending space")
        with repo.database.transaction() as connection:
            connection.execute("UPDATE runs SET status='cancelled' WHERE id=?", (second.id,))
        second_digest, _ = repo.export_run_archive(owner, [second.id], io.BytesIO())
        second_body = {"run_ids": [second.id], "sha256": second_digest, "confirmed": True}
        second_headers = {**headers, "X-Request-ID": str(uuid.uuid4())}
        original = repo.database.reclaim_space

        def busy(sample, **kwargs):
            if kwargs.get("_check_only"):
                return original(sample, **kwargs)
            raise StorageMaintenanceError("maintenance_busy", "Reader busy.")

        monkeypatch.setattr(repo.database, "reclaim_space", busy)
        partial = client.post(path, headers=second_headers, json=second_body)
        assert partial.status_code == 200 and partial.json()["purged"]
        assert partial.json()["reclamation_pending"]
        monkeypatch.setattr(repo.database, "reclaim_space", original)
        assert not client.post(path, headers=second_headers, json=second_body).json()[
            "reclamation_pending"
        ]
        repo.database.storage._maintenance_owner = object()
        repo.database.storage._maintenance_active = True
        try:
            blocked = client.post(
                f"/api/agents/{agent}/messages",
                headers=headers,
                json={"prompt": "Blocked during maintenance"},
            )
            assert blocked.status_code == 503 and blocked.json()["code"] == "maintenance_busy"
        finally:
            repo.database.storage._maintenance_active = False
            repo.database.storage._maintenance_owner = None
        repo.database.storage.enabled = True
        repo.database.storage.trip("platform_quota_high")
        assert client.post(path, headers=headers, json=body).json()["replayed"]
        repo.database.storage.enabled = False
        client.post("/api/auth/logout", headers=headers)
        headers["X-Request-ID"] = str(uuid.uuid4())
        other = client.post(
            "/api/auth/register",
            headers=headers,
            json={"email": "purge-other@example.test", "password": "correct horse battery"},
        ).json()
        headers["X-CSRF-Token"] = other["csrf_token"]
        assert client.post(path, headers=headers, json=body).status_code == 404


@pytest.mark.parametrize(
    "mutation",
    ["other_agent", "missing", "root", "duplicate", "writable", "other_volume", "copy_up"],
)
async def test_container_cleanup_rejects_wrong_configured_subpaths(settings, monkeypatch, mutation):
    from backend.app.agents import AgentPreparationError

    manager, scenario = _docker_scenario(settings, monkeypatch)
    mount = scenario.configured_mounts[0]
    if mutation == "other_agent":
        mount["VolumeOptions"]["Subpath"] = f"{uuid.uuid4()}/workspace"
    elif mutation == "missing":
        mount.pop("VolumeOptions")
    elif mutation == "root":
        mount["VolumeOptions"]["Subpath"] = ""
    elif mutation == "duplicate":
        scenario.configured_mounts.append(dict(mount))
    elif mutation == "copy_up":
        mount["VolumeOptions"].pop("NoCopy")
    elif mutation == "writable":
        scenario.configured_mounts[2]["ReadOnly"] = False
    else:
        mount["Source"] = "other-volume"
    with pytest.raises(AgentPreparationError):
        await manager.stop_agent(scenario.labels["qpl.agent"])
    assert scenario.running
    assert all(command[1] not in {"stop", "kill", "create"} for command in scenario.commands)
