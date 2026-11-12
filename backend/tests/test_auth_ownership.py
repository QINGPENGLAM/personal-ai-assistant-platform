from __future__ import annotations

import time
import uuid
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.agents import AgentPreparationError
from backend.app.auth import AuthError, AuthService
from backend.app.config import Settings
from backend.app.containers import ContainerManager
from backend.app.db import Database
from backend.app.main import create_app
from backend.app.repository import AgentRecord, Repository, RepositoryError


def _create_user(database: Database) -> str:
    user_id = str(uuid.uuid4())
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO users(id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, f"{user_id}@example.test", "test-hash", time.time_ns() // 1_000_000),
        )
    return user_id


def test_registration_normalizes_email_and_stores_only_hashes(
    auth_service: AuthService,
    database: Database,
) -> None:
    user, raw_token = auth_service.register(
        "  Ana@Example.com ", "correct horse battery", "127.0.0.1"
    )

    assert user.email == "ana@example.com"
    assert raw_token
    with database.read() as connection:
        stored_user = connection.execute(
            "SELECT email, password_hash FROM users WHERE id = ?", (user.id,)
        ).fetchone()
        stored_session = connection.execute(
            "SELECT token_hash, csrf_token FROM sessions WHERE user_id = ?", (user.id,)
        ).fetchone()

    assert stored_user["email"] == "ana@example.com"
    assert stored_user["password_hash"] != "correct horse battery"
    assert raw_token not in stored_session["token_hash"]
    assert raw_token not in stored_session["csrf_token"]
    assert auth_service.authenticate(raw_token) == user


def test_registration_rejects_duplicate_normalized_email(auth_service: AuthService) -> None:
    auth_service.register("ana@example.com", "correct horse battery", "127.0.0.1")

    with pytest.raises(AuthError) as raised:
        auth_service.register(" Ana@Example.com ", "another long password", "127.0.0.1")

    assert raised.value.status_code == 409
    assert raised.value.code == "email_taken"


def test_login_uses_uniform_error_and_logout_revokes_session(auth_service: AuthService) -> None:
    user, token = auth_service.register("ana@example.com", "correct horse battery", "127.0.0.1")
    logged_in_user, login_token = auth_service.login(
        " ANA@example.com ", "correct horse battery", "127.0.0.1"
    )
    assert logged_in_user.email == user.email
    assert auth_service.authenticate(login_token) == logged_in_user

    with pytest.raises(AuthError) as raised:
        auth_service.login("missing@example.com", "wrong password", "127.0.0.1")
    assert raised.value.status_code == 401
    assert raised.value.code == "invalid_credentials"

    auth_service.logout(token)
    with pytest.raises(AuthError) as expired:
        auth_service.authenticate(token)
    assert expired.value.code == "login_required"


def test_csrf_and_origin_are_checked(auth_service: AuthService) -> None:
    user, _ = auth_service.register("ana@example.com", "correct horse battery", "127.0.0.1")
    auth_service.require_csrf(user, user.csrf_token)
    auth_service.require_origin("http://127.0.0.1:5173")

    with pytest.raises(AuthError) as csrf_error:
        auth_service.require_csrf(user, "wrong")
    assert csrf_error.value.status_code == 403

    with pytest.raises(AuthError) as origin_error:
        auth_service.require_origin("https://attacker.example")
    assert origin_error.value.status_code == 403


def test_agent_api_creates_idempotently_and_hides_other_users_assistants(
    settings: Settings,
) -> None:
    app = create_app(
        settings,
        test_mode=True,
        verified_models=("gpt-6-luna",),
    )
    with TestClient(app) as client:
        headers = {
            "Origin": settings.allowed_origin,
            "X-Request-ID": "4b6a51a6-c0b1-42b4-9a7f-e3d41facab18",
        }
        registered = client.post(
            "/api/auth/register",
            headers=headers,
            json={"email": "ana@example.com", "password": "correct horse battery"},
        )
        assert registered.status_code == 201
        csrf_token = registered.json()["csrf_token"]
        assert "httponly" in registered.headers["set-cookie"].lower()

        session = client.get("/api/auth/session")
        assert session.status_code == 200
        assert session.json()["csrf_token"] == csrf_token

        agent_headers = {
            **headers,
            "X-CSRF-Token": csrf_token,
            "X-Request-ID": "aab1bcb8-4500-4b54-a45b-d933bc213e39",
        }
        created = client.post(
            "/api/agents",
            headers=agent_headers,
            json={"name": "Ana's assistant", "model": "gpt-6-luna"},
        )
        assert created.status_code == 201
        assert created.json()["status"] == "ready"
        agent_id = created.json()["id"]

        message_headers = {
            **agent_headers,
            "X-Request-ID": "838d93a3-2ca4-46da-8102-622317bc6097",
        }
        accepted = client.post(
            f"/api/agents/{agent_id}/messages",
            headers=message_headers,
            json={"prompt": "Please summarize my workspace."},
        )
        assert accepted.status_code == 202
        assert accepted.json()["status"] == "queued"
        assert accepted.json()["queued"] is True
        run_id = accepted.json()["run_id"]

        message_replay = client.post(
            f"/api/agents/{agent_id}/messages",
            headers=message_headers,
            json={"prompt": "Please summarize my workspace."},
        )
        assert message_replay.status_code == 200
        assert message_replay.json()["run_id"] == run_id
        assert message_replay.json()["replayed"] is True

        conflict = client.post(
            f"/api/agents/{agent_id}/messages",
            headers=message_headers,
            json={"prompt": "A different message."},
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "idempotency_conflict"

        chat_history = client.get(f"/api/agents/{agent_id}/chat")
        assert chat_history.status_code == 200
        assert len(chat_history.json()) == 1
        assert chat_history.json()[0]["prompt"] == "Please summarize my workspace."
        run_detail = client.get(f"/api/runs/{run_id}")
        assert run_detail.status_code == 200
        assert run_detail.json()["last_event_sequence"] == 1
        assert run_detail.json()["cleanup_state"] == "not_required"
        assert client.get(f"/api/runs/{run_id}/events?after=2").status_code == 409
        assert client.get(f"/api/runs/{run_id}/events?after=-1").status_code == 409
        continued = client.post(
            f"/api/agents/{agent_id}/continue",
            headers={**agent_headers, "X-Request-ID": "b38a95f1-b2a7-45e8-a3de-660597a9ce33"},
        )
        assert continued.status_code == 200
        assert continued.json()["paused"] is False
        with client.app.state.database.transaction() as connection:
            connection.execute(
                "UPDATE runs SET status = 'failed' WHERE id = ?",
                (run_id,),
            )
        retry_headers = {
            **agent_headers,
            "X-Request-ID": "7ff65512-879e-4d24-9fb1-0ee1f08d0245",
        }
        retried = client.post(f"/api/runs/{run_id}/retry", headers=retry_headers)
        assert retried.status_code == 202
        assert retried.json()["status"] == "queued"
        assert retried.json()["context_warning"] is True
        retry_replay = client.post(f"/api/runs/{run_id}/retry", headers=retry_headers)
        assert retry_replay.status_code == 200
        assert retry_replay.json()["run_id"] == retried.json()["run_id"]
        assert retry_replay.json()["replayed"] is True

        replay = client.post(
            "/api/agents",
            headers=agent_headers,
            json={"name": "Ana's assistant", "model": "gpt-6-luna"},
        )
        assert replay.status_code == 200
        assert replay.json()["id"] == agent_id

        client.post(
            "/api/auth/logout",
            headers={
                **headers,
                "X-CSRF-Token": csrf_token,
                "X-Request-ID": "31a5ee53-8303-4387-8cb9-409870675ab1",
            },
        )
        second_user = client.post(
            "/api/auth/register",
            headers={
                **headers,
                "X-Request-ID": "f8c83316-4af4-4f85-87bd-c5e00ccf0de6",
            },
            json={"email": "ben@example.com", "password": "another correct horse"},
        )
        assert second_user.status_code == 201
        response = client.get(f"/api/agents/{agent_id}")
        assert response.status_code == 404
        assert response.json()["message"] == "Assistant not found."
        assert client.get(f"/api/runs/{run_id}").status_code == 404
        assert client.post(
            f"/api/runs/{run_id}/stop",
            headers={
                "Origin": settings.allowed_origin,
                "X-CSRF-Token": second_user.json()["csrf_token"],
            },
        ).status_code == 404


def test_container_create_arguments_are_scoped_and_network_is_disabled(
    settings: Settings,
) -> None:
    agent_id = "77b06fad-6ad9-45b4-92f3-38a7b86f0862"
    runtime_settings = replace(
        settings,
        auth_mode="toolkit",
        toolkit_api_key="test-secret",
    )
    manager = ContainerManager(runtime_settings, "9e17951f-254d-47f4-9716-165434aad530")

    args = manager.create_arguments(agent_id)

    instance_id = "9e17951f-254d-47f4-9716-165434aad530"
    assert args[:4] == [
        "container",
        "create",
        "--name",
        f"qpl-agent-{instance_id}-{agent_id}",
    ]
    assert "qpl.platform=personal-assistant" in args
    assert f"qpl.instance={instance_id}" in args
    assert f"qpl.agent={agent_id}" in args
    assert "qpl.purpose=agent" in args
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("--cpus") + 1] == "1"
    assert args[args.index("--memory") + 1] == "1g"
    assert args[args.index("--pids-limit") + 1] == "128"
    mounts = [
        args[index + 1]
        for index, value in enumerate(args[:-1])
        if value == "--mount"
    ]
    assert len(mounts) == 4
    assert all("volume-nocopy" in mount.split(",") for mount in mounts)
    assert all(f"volume-subpath={agent_id}/" in mount for mount in mounts[:3])
    assert any("target=/run/credentials" in mount and "readonly" in mount for mount in mounts)
    assert all("qpl-agent-test" in mount or "qpl-platform-test" in mount for mount in mounts)
    assert "Ana's assistant" not in args
    assert "test-secret" not in args

    invalid_gateway = replace(
        runtime_settings,
        gateway_socket=settings.runtime_root / "outside.sock",
    )
    with pytest.raises(AgentPreparationError, match="outside the platform-state volume"):
        ContainerManager(
            invalid_gateway,
            "9e17951f-254d-47f4-9716-165434aad530",
        ).create_arguments(agent_id)


def test_toolkit_configuration_keeps_key_out_of_codex_config(settings: Settings) -> None:
    agent_id = "77b06fad-6ad9-45b4-92f3-38a7b86f0862"
    runtime_settings = replace(
        settings,
        auth_mode="toolkit",
        toolkit_api_key="test-key-value",
    )
    manager = ContainerManager(runtime_settings, "9e17951f-254d-47f4-9716-165434aad530")
    runtime_settings.agent_data_root.mkdir(parents=True)
    manager._ensure_agent_directories(agent_id)
    manager._write_agent_configuration(
        AgentRecord(
            id=agent_id,
            owner_id="f873f29f-7e8e-429d-8858-e799f9cb227a",
            name="test assistant",
            model="gpt-6-luna",
            status="provisioning",
            paused=False,
            blockers=(),
            container_state="unknown",
            created_at=1,
        )
    )

    codex_config = (
        runtime_settings.agent_data_root / agent_id / "codex" / "config.toml"
    ).read_text(encoding="utf-8")
    credential = (
        runtime_settings.agent_data_root
        / agent_id
        / "credentials"
        / "UMGPT_TOOLKIT_API_KEY"
    )
    assert 'model_provider = "toolkit"' in codex_config
    assert 'env_key = "UMGPT_TOOLKIT_API_KEY"' in codex_config
    assert 'base_url = "https://api.toolkit.umgpt.umich.edu/v1"' in codex_config
    assert 'url = "http://127.0.0.1:8765/mcp"' in codex_config
    assert "test-key-value" not in codex_config
    assert credential.read_text(encoding="utf-8") == "test-key-value"
    assert credential.stat().st_mode & 0o777 == 0o600


def _load_quota_helper():
    helper_path = Path(__file__).parents[2] / "scripts" / "quota-helper.py"
    spec = spec_from_file_location("qpl_quota_helper", helper_path)
    assert spec is not None and spec.loader is not None
    helper = module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def test_quota_helper_rejects_noncanonical_agent_ids_and_unknown_fields() -> None:
    helper = _load_quota_helper()
    with pytest.raises(helper.HelperError, match="canonical UUID"):
        helper._dispatch({"operation": "ensure_agent", "agent_id": "NOT-A-UUID"})
    with pytest.raises(helper.HelperError, match="unsupported operation"):
        helper._dispatch({"operation": "verify", "path": "/tmp"})
    assert (
        helper._reported_hard_limit(
            "Project ID Used Soft Hard Warn/Grace\n10000 0 0 2097152 0 [--------]\n",
            10_000,
        )
        == 2_097_152
    )
    with pytest.raises(helper.HelperError, match="did not report"):
        helper._reported_hard_limit(
            "Project ID Used Soft Hard Warn/Grace\n10000 0 0 0 0 [--------]\n",
            10_001,
        )


@pytest.mark.parametrize("project_token", ["10000", "#10000"])
def test_quota_helper_reads_numeric_xfs_report_ids(project_token: str) -> None:
    helper = _load_quota_helper()
    # xfs_quota -n prints numeric IDs with a leading # on Ubuntu 24.04.
    report = (
        "Project quota on /var/lib/docker (/dev/nvme1n1)\n"
        "Project ID       Used       Soft       Hard    Warn/Grace\n"
        "---------- --------------------------------------------------\n"
        "#0                 164          0          0     00 [--------]\n"
        f"{project_token} 12 0 2097152 00 [--------]\n"
    )
    assert helper._reported_hard_limit(report, 10000) == 2097152
    assert helper._reported_value(report, 10000, "Used") == 12
    with pytest.raises(helper.HelperError, match="did not report"):
        helper._reported_hard_limit(report, 10001)


def test_quota_helper_rejects_project_mapping_paths_outside_fixed_volumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_quota_helper()
    volume_root = tmp_path / "docker-volumes"
    platform_root = volume_root / helper.PLATFORM_VOLUME / "_data"
    agent_root = volume_root / helper.AGENT_VOLUME / "_data"
    platform_root.mkdir(parents=True)
    agent_root.mkdir(parents=True)
    state_path = tmp_path / "quota" / "project-map.json"
    monkeypatch.setattr(helper, "VOLUME_ROOT", volume_root)
    monkeypatch.setattr(helper, "STATE_PATH", state_path)
    helper._save_mapping(
        {
            "version": 1,
            "next_id": helper.FIRST_AGENT_PROJECT_ID,
            "projects": {
                "platform": {
                    "id": helper.PLATFORM_PROJECT_ID,
                    "name": "qpl_platform_state",
                    "path": "/etc",
                }
            },
        }
    )

    with pytest.raises(helper.HelperError, match="platform project mapping is invalid"):
        helper._mapping()


def test_auth_api_logout_requires_csrf_and_revokes_session(settings: Settings) -> None:
    app = create_app(settings, test_mode=True)
    with TestClient(app) as client:
        headers = {
            "Origin": settings.allowed_origin,
            "X-Request-ID": "4b6a51a6-c0b1-42b4-9a7f-e3d41facab18",
        }
        registered = client.post(
            "/api/auth/register",
            headers=headers,
            json={"email": "ana@example.com", "password": "correct horse battery"},
        )
        assert registered.status_code == 201
        csrf_token = registered.json()["csrf_token"]
        assert "httponly" in registered.headers["set-cookie"].lower()

        no_csrf = client.post(
            "/api/auth/logout",
            headers={
                "Origin": settings.allowed_origin,
                "X-Request-ID": "47db31a3-4ba2-4c7a-8432-d2feaa03f5bc",
            },
        )
        assert no_csrf.status_code == 403
        assert client.get("/api/auth/session").status_code == 200

        logout = client.post(
            "/api/auth/logout",
            headers={
                "Origin": settings.allowed_origin,
                "X-CSRF-Token": csrf_token,
                "X-Request-ID": "aab1bcb8-4500-4b54-a45b-d933bc213e39",
            },
        )
        assert logout.status_code == 200
        assert logout.json() == {"logged_out": True}
        assert client.get("/api/auth/session").status_code == 401


def test_auth_api_rejects_missing_request_id_and_cross_origin(
    settings: Settings,
) -> None:
    app = create_app(settings, test_mode=True)
    with TestClient(app) as client:
        missing_id = client.post(
            "/api/auth/register",
            headers={"Origin": settings.allowed_origin},
            json={"email": "ana@example.com", "password": "correct horse battery"},
        )
        assert missing_id.status_code == 422
        assert missing_id.json()["code"] == "invalid_request_id"

        cross_origin = client.post(
            "/api/auth/register",
            headers={
                "Origin": "https://attacker.example",
                "X-Request-ID": "4b6a51a6-c0b1-42b4-9a7f-e3d41facab18",
            },
            json={"email": "ana@example.com", "password": "correct horse battery"},
        )
        assert cross_origin.status_code == 403
        assert cross_origin.json()["code"] == "origin_rejected"


def test_agent_creation_is_idempotent_and_owner_scoped(database: Database) -> None:
    repository = Repository(database)
    owner_id = _create_user(database)
    other_user_id = _create_user(database)
    request_id = str(uuid.uuid4())

    created, replayed = repository.create_agent(
        owner_id, "Ana's assistant", "gpt-6-luna", request_id
    )
    replay, replayed_again = repository.create_agent(
        owner_id, "Ana's assistant", "gpt-6-luna", request_id
    )

    assert replayed is False
    assert replayed_again is True
    assert replay.id == created.id
    assert created.status == "provisioning"
    assert repository.get_owned_agent(owner_id, created.id).id == created.id
    assert repository.list_agents(owner_id) == [created]

    with pytest.raises(RepositoryError) as raised:
        repository.get_owned_agent(other_user_id, created.id)
    assert raised.value.status_code == 404
    assert raised.value.message == "Assistant not found."


def test_agent_creation_request_id_cannot_be_reused_with_different_body(
    database: Database,
) -> None:
    repository = Repository(database)
    owner_id = _create_user(database)
    request_id = str(uuid.uuid4())
    repository.create_agent(owner_id, "Assistant A", "gpt-6-luna", request_id)

    with pytest.raises(RepositoryError) as raised:
        repository.create_agent(owner_id, "Assistant B", "gpt-6-luna", request_id)
    assert raised.value.status_code == 409
    assert raised.value.code == "idempotency_conflict"


def test_new_chat_api_requires_confirmation_csrf_ownership_and_replays_once(settings):
    app = create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    with TestClient(app) as client:
        headers = {"Origin": settings.allowed_origin, "X-Request-ID": str(uuid.uuid4())}
        registered = client.post("/api/auth/register", headers=headers, json={
            "email": "new-chat@example.test", "password": "correct horse battery",
        })
        headers["X-CSRF-Token"] = registered.json()["csrf_token"]
        created = client.post("/api/agents", headers=headers, json={
            "name": "Assistant", "model": "gpt-6-luna",
        })
        agent_id = created.json()["id"]
        with app.state.database.transaction() as connection:
            connection.execute(
                "UPDATE agents SET paused = 1, blockers_json = '[\"history_error\"]' WHERE id = ?",
                (agent_id,),
            )
        url = f"/api/agents/{agent_id}/new-chat"
        assert client.post(url, headers=headers, json={"confirmed": False}).status_code == 422
        assert client.post(url, headers=headers, json={"confirmed": 1}).status_code == 422
        assert client.post(url, headers={**headers, "X-CSRF-Token": "wrong"}, json={
            "confirmed": True,
        }).status_code == 403
        first = client.post(url, headers=headers, json={"confirmed": True})
        again = client.post(url, headers=headers, json={"confirmed": True})
        assert first.status_code == again.status_code == 200
        assert first.json()["operation_id"] == headers["X-Request-ID"]
        assert first.json()["replayed"] is False
        assert again.json()["replayed"] is True
        assert again.json()["agent"]["chat_epoch"] == 1
        assert again.json()["agent"]["paused"] is True
        other = client.post("/api/auth/register", headers=headers, json={
            "email": "other-new-chat@example.test", "password": "correct horse battery",
        })
        headers["X-CSRF-Token"] = other.json()["csrf_token"]
        assert client.post(url, headers=headers, json={"confirmed": True}).status_code == 404


def test_built_frontend_serves_spa_routes_without_masking_api_errors(settings, tmp_path):
    bundle = tmp_path / "frontend"
    (bundle / "assets").mkdir(parents=True)
    (bundle / "index.html").write_text('<html><div id="root"></div></html>')
    (bundle / "assets" / "app.js").write_text('console.log("bundle");')
    app = create_app(settings, test_mode=True, frontend_dist=bundle)
    with TestClient(app) as client:
        assert 'id="root"' in client.get("/agents/any-agent").text
        assert client.get("/assets/app.js").headers["content-type"].startswith("text/javascript")
        assert client.get("/api/unknown").status_code == 404
        assert client.get("/api/unknown").headers["content-type"] == "application/json"
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/unknown.txt").status_code == 404


def test_quota_commands_use_service_owned_mapping_files(tmp_path, monkeypatch):
    from types import SimpleNamespace

    helper = _load_quota_helper()
    projects = tmp_path / "projects"
    projid = tmp_path / "projid"
    monkeypatch.setattr(helper, "PROJECTS_PATH", projects)
    monkeypatch.setattr(helper, "PROJID_PATH", projid)
    commands = []

    def run(arguments, **kwargs):
        commands.append(arguments)
        hard = 100_000 if "report -p -n -i" in arguments else 2_097_152
        return SimpleNamespace(stdout=f"Project ID Used Soft Hard Warn/Grace\n10000 0 0 {hard} 0\n")

    monkeypatch.setattr(helper.subprocess, "run", run)
    monkeypatch.setattr(helper, "_project_attributes", lambda path: (10000, helper.PROJECT_INHERIT))
    project = {"name": "qpl_platform_state", "id": 10000, "path": str(tmp_path)}
    helper._run_xfs(tmp_path, project)
    helper._verify_project(project, tmp_path)
    assert len(commands) == 3
    for command in commands:
        assert command[command.index("-D") + 1] == str(projects)
        assert command[command.index("-P") + 1] == str(projid)


def test_private_quota_maps_reject_existing_system_project_id_conflicts(tmp_path, monkeypatch):
    helper = _load_quota_helper()
    system_projects = tmp_path / "system-projects"
    system_projid = tmp_path / "system-projid"
    system_projects.write_text("10000:/another-project\n")
    system_projid.write_text("another_project:10000\n")
    monkeypatch.setattr(helper, "PROJECTS_PATH", tmp_path / "private/projects")
    monkeypatch.setattr(helper, "PROJID_PATH", tmp_path / "private/projid")
    monkeypatch.setattr(helper, "SYSTEM_PROJECTS_PATH", system_projects, raising=False)
    monkeypatch.setattr(helper, "SYSTEM_PROJID_PATH", system_projid, raising=False)
    mapping = {"projects": {"platform": {
        "id": 10000, "name": "qpl_platform_state", "path": "/own",
    }}}
    with pytest.raises(helper.HelperError, match="already assigned"):
        helper._write_project_config(mapping)
    assert system_projects.read_text() == "10000:/another-project\n"


def test_quota_report_returns_usage_in_kib_and_rejects_incorrect_limits(tmp_path, monkeypatch):
    from types import SimpleNamespace

    helper = _load_quota_helper()
    project = {"id": 10000, "path": str(tmp_path)}
    monkeypatch.setattr(helper, "_project_attributes", lambda path: (10000, helper.PROJECT_INHERIT))
    block_hard = 2097152

    def report(command, **kwargs):
        is_blocks = "report -p -n -b" in command
        hard, used = (block_hard, 7) if is_blocks else (100000, 3)
        return SimpleNamespace(
            stdout=f"Project ID Used Soft Hard Warn/Grace\n10000 {used} 0 {hard} 0\n",
        )

    monkeypatch.setattr(helper.subprocess, "run", report)
    assert helper._verify_project(project, tmp_path) == {
        "used_bytes": 7 * 1024, "hard_bytes": 2 * 1024**3,
        "used_inodes": 3, "hard_inodes": 100000,
    }
    block_hard = 4194304
    with pytest.raises(helper.HelperError):
        helper._verify_project(project, tmp_path)


def test_verify_agent_quota_only_reads_existing_fixed_project(tmp_path, monkeypatch):
    helper = _load_quota_helper()
    agent_id = str(uuid.uuid4())
    path = tmp_path / agent_id
    entry = {"id": 10001, "name": "test", "path": str(path)}
    monkeypatch.setattr(
        helper, "_agent_path", lambda value, *, prepare: path if not prepare else forbidden(),
    )
    monkeypatch.setattr(helper, "_mapping", lambda: {"projects": {agent_id: entry}})
    monkeypatch.setattr(helper, "_mount_info", lambda value: (tmp_path, {"prjquota"}))
    monkeypatch.setattr(helper, "_verify_project", lambda project, mount: {"used_bytes": 0})

    def forbidden(*args):
        pytest.fail("Usage verification must not reapply project attributes.")

    monkeypatch.setattr(helper, "_run_xfs", forbidden)
    monkeypatch.setattr(helper, "_save_mapping", forbidden)
    assert helper._dispatch({"operation": "verify_agent", "agent_id": agent_id}) == {
        "ok": True, "quota": {"used_bytes": 0},
    }


def test_read_only_agent_path_does_not_create_or_change_permissions(tmp_path, monkeypatch):
    helper = _load_quota_helper()
    agent_id = str(uuid.uuid4())
    path = tmp_path / agent_id
    path.mkdir(mode=0o700)
    monkeypatch.setattr(helper, "_volume_root", lambda volume: tmp_path)

    def forbidden(*args):
        pytest.fail("Read-only quota verification must not alter directory permissions.")

    monkeypatch.setattr(helper.os, "chown", forbidden)
    monkeypatch.setattr(helper.os, "chmod", forbidden)
    assert helper._agent_path(agent_id, prepare=False) == path
    missing = str(uuid.uuid4())
    with pytest.raises(helper.HelperError):
        helper._agent_path(missing, prepare=False)
    assert not (tmp_path / missing).exists()


def test_platform_storage_query_is_read_only_and_reports_allocated_project_usage(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    helper = _load_quota_helper()
    project = {"id": 10000, "path": str(tmp_path)}
    monkeypatch.setattr(helper, "_volume_root", lambda volume: tmp_path)
    monkeypatch.setattr(helper, "_mapping", lambda: {"projects": {"platform": project}})
    monkeypatch.setattr(helper, "_mount_info", lambda path: (tmp_path, {"prjquota"}))
    monkeypatch.setattr(
        helper,
        "_verify_project",
        lambda entry, mount: {"used_bytes": 123, "hard_bytes": 2 * 1024**3},
    )
    monkeypatch.setattr(
        helper.os, "statvfs", lambda path: SimpleNamespace(f_bavail=9, f_frsize=4096)
    )

    def forbidden(*args):
        pytest.fail("A platform usage read must not provision or change paths.")

    monkeypatch.setattr(helper, "_ensure_project", forbidden)
    monkeypatch.setattr(helper.os, "chown", forbidden)
    assert helper._dispatch({"operation": "verify_platform"}) == {
        "ok": True,
        "quota": {"used_bytes": 123, "hard_bytes": 2 * 1024**3},
        "free_bytes": 9 * 4096,
    }
