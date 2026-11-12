import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.network_gateway import NetworkGateway


def test_stateless_mcp_initialization_and_scoped_job_tools(settings):
    with TestClient(
        create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    ) as client:
        headers = {"Origin": settings.allowed_origin, "X-Request-ID": str(uuid4())}
        registered = client.post(
            "/api/auth/register",
            headers=headers,
            json={
                "email": "mcp@example.test",
                "password": "correct horse battery",
            },
        )
        headers["X-CSRF-Token"] = registered.json()["csrf_token"]
        agent = client.post(
            "/api/agents", headers=headers, json={"name": "Jobs", "model": "gpt-6-luna"}
        ).json()["id"]
        app = client.app
        repository = app.state.repository
        owner = registered.json()["user"]["id"]
        run, _ = repository.create_chat_run(owner, agent, str(uuid4()), "Schedule notes")
        context = repository.claim_next_run(agent, verified_models=frozenset({"gpt-6-luna"}))
        network = NetworkGateway(settings, app.state.instance_id)
        app.state.network_gateway = network
        token = network.issue_run_token(run.id, agent, context.generation)
        transport = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, text/event-stream",
        }

        def rpc(method, params=None, identity=1):
            return client.post(
                "/mcp",
                headers=transport,
                json={
                    "jsonrpc": "2.0",
                    "id": identity,
                    "method": method,
                    "params": params or {},
                },
            )

        initialized = rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        )
        assert initialized.status_code == 200, initialized.text
        listed = rpc("tools/list").json()["result"]["tools"]
        assert {item["name"] for item in listed} == {"create_job", "list_jobs", "cancel_job"}
        repository.update_run_running(
            run.id,
            thread_id="chat-thread",
            generation=context.generation,
            run_token_hash=network.token_hash(token),
        )
        request_id = str(uuid4())
        arguments = {
            "request_id": request_id,
            "name": "Agent task",
            "prompt": "Read notes",
            "run_at": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(timespec="seconds"),
        }
        result = rpc("tools/call", {"name": "create_job", "arguments": arguments})
        assert result.status_code == 200, result.text
        content = result.json()["result"]
        assert not content.get("isError"), content
        job = json.loads(content["content"][0]["text"])
        assert job["created_by"] == "agent"
        assert (
            json.loads(
                rpc("tools/call", {"name": "create_job", "arguments": arguments}).json()["result"][
                    "content"
                ][0]["text"]
            )["job_id"]
            == job["job_id"]
        )
        assert client.get(f"/api/agents/{agent}/jobs").json()["items"][0]["job_id"] == job["job_id"]
        agent_list = rpc("tools/call", {"name": "list_jobs", "arguments": {}}).json()["result"]
        assert json.loads(agent_list["content"][0]["text"])["items"][0]["job_id"] == job["job_id"]
        cancelled = rpc(
            "tools/call",
            {
                "name": "cancel_job",
                "arguments": {
                    "request_id": str(uuid4()),
                    "job_id": job["job_id"],
                },
            },
        ).json()["result"]
        assert json.loads(cancelled["content"][0]["text"])["enabled"] is False
        other, _ = repository.create_agent(owner, "Other", "gpt-6-luna", str(uuid4()))
        from backend.app.jobs import Actor

        foreign = client.portal.call(
            app.state.jobs.create,
            Actor("user", owner),
            other.id,
            str(uuid4()),
            {key: value for key, value in arguments.items() if key != "request_id"},
        )
        denied = rpc(
            "tools/call",
            {
                "name": "cancel_job",
                "arguments": {
                    "request_id": str(uuid4()),
                    "job_id": foreign["job_id"],
                },
            },
        ).json()["result"]
        assert denied["isError"]
        assert json.loads(denied["content"][0]["text"])["code"] == "not_found"
        bad = rpc(
            "tools/call",
            {"name": "create_job", "arguments": {**arguments, "agent_id": str(uuid4())}},
        ).json()["result"]
        assert bad["isError"] is True
        assert json.loads(bad["content"][0]["text"])["code"] == "invalid_arguments"
        network.revoke_run(run.id)
        assert rpc("tools/list").status_code == 403
        app.state.network_gateway = None


async def test_python_mcp_client_and_old_call_waiting_on_lock_is_revoked(settings):
    import asyncio

    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    app = create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    async with app.router.lifespan_context(app):
        user, _ = app.state.auth.register(
            "sdk-mcp@example.test", "correct horse battery", "127.0.0.1"
        )
        repository = app.state.repository
        agent, _ = repository.create_agent(user.id, "SDK tools", "gpt-6-luna", str(uuid4()))
        repository.set_agent_status(agent.id, "ready")
        run, _ = repository.create_chat_run(user.id, agent.id, str(uuid4()), "Create a schedule")
        context = repository.claim_next_run(agent.id, verified_models=frozenset({"gpt-6-luna"}))
        network = NetworkGateway(settings, app.state.instance_id)
        app.state.network_gateway = network
        raw = network.issue_run_token(run.id, agent.id, context.generation)
        repository.update_run_running(
            run.id,
            thread_id="sdk-thread",
            generation=context.generation,
            run_token_hash=network.token_hash(raw),
        )
        entered = asyncio.Event()
        original_create = app.state.jobs.create

        async def observed_create(*args, **kwargs):
            entered.set()
            return await original_create(*args, **kwargs)

        app.state.jobs.create = observed_create
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), headers={"Authorization": f"Bearer {raw}"}
            ) as http:
                async with streamable_http_client("http://testserver/mcp", http_client=http) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        assert len((await session.list_tools()).tools) == 3
                        arguments = {
                            "request_id": str(uuid4()),
                            "name": "Too late",
                            "prompt": "Read notes",
                            "run_at": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(
                                timespec="seconds"
                            ),
                        }
                        async with app.state.coordination.agent_locks([agent.id]):
                            task = asyncio.create_task(session.call_tool("create_job", arguments))
                            await asyncio.wait_for(entered.wait(), 2)
                            repository.request_run_stop(user.id, run.id)
                            network.revoke_run(run.id)
                        result = await asyncio.wait_for(task, 2)
                        assert result.isError
                        assert json.loads(result.content[0].text)["code"] == "tool_unauthorized"
                        with repository.database.read() as connection:
                            assert (
                                connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
                            )
        finally:
            app.state.network_gateway = None
