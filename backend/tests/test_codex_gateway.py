from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

import backend.app.codex_gateway as codex_gateway


class FakeTurn:
    id = "turn-123"


class FakeThread:
    id = "thread-123"

    def __init__(self) -> None:
        self.turn_args: tuple[tuple[Any, ...], dict[str, Any]] | None = None

    async def turn(self, *args: Any, **kwargs: Any) -> FakeTurn:
        self.turn_args = args, kwargs
        return FakeTurn()


class FakeCodex:
    def __init__(self, config: codex_gateway.CodexConfig) -> None:
        self.config = config
        self.thread = FakeThread()
        self.prepared: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.closed = False

    async def __aenter__(self) -> FakeCodex:
        return self

    async def thread_start(self, **kwargs: Any) -> FakeThread:
        self.prepared.append(("start", (), kwargs))
        return self.thread

    async def thread_resume(self, thread_id: str, **kwargs: Any) -> FakeThread:
        self.prepared.append(("resume", (thread_id,), kwargs))
        return self.thread

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_gateway_uses_container_app_server_and_persists_thread_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_clients: list[FakeCodex] = []

    def fake_codex(config: codex_gateway.CodexConfig) -> FakeCodex:
        client = FakeCodex(config)
        codex_clients.append(client)
        return client

    monkeypatch.setattr(codex_gateway, "AsyncCodex", fake_codex)
    agent_id = str(uuid4())
    instance_id = "9e17951f-254d-47f4-9716-165434aad530"

    session = await codex_gateway.CodexGateway(instance_id).connect(agent_id)
    client = codex_clients[0]
    assert client.config.launch_args_override == (
        "docker",
        "exec",
        "-i",
        f"qpl-agent-{instance_id}-{agent_id}",
        "/usr/local/bin/launch-codex",
    )

    first_thread = await session.prepare_thread("gpt-6-luna")
    resumed_thread = await session.prepare_thread("gpt-6-luna", first_thread.id)
    turn = await session.start_turn(resumed_thread, "gpt-6-luna", "Summarize the workspace")

    assert turn.id == "turn-123"
    assert [entry[0] for entry in client.prepared] == ["start", "resume"]
    assert client.prepared[1][1] == ("thread-123",)
    assert all(
        entry[2]
        == {
            "approval_mode": codex_gateway.ApprovalMode.deny_all,
            "cwd": "/workspace",
            "model": "gpt-6-luna",
            "sandbox": codex_gateway.Sandbox.full_access,
        }
        for entry in client.prepared
    )
    assert client.thread.turn_args is not None
    assert client.thread.turn_args[0] == ("Summarize the workspace",)
    assert client.thread.turn_args[1] == client.prepared[0][2]

    await session.close()
    assert client.closed


@pytest.mark.asyncio
async def test_gateway_rejects_noncanonical_agent_id_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        codex_gateway,
        "AsyncCodex",
        lambda _config: pytest.fail("invalid agent id must not launch Codex"),
    )

    with pytest.raises(ValueError, match="canonical UUID"):
        await codex_gateway.CodexGateway(
            "9e17951f-254d-47f4-9716-165434aad530"
        ).connect(str(uuid4()).upper())


@pytest.mark.asyncio
async def test_history_session_reads_without_submitting_a_turn(monkeypatch) -> None:
    from types import SimpleNamespace

    clients = []

    def factory(config):
        client = FakeCodex(config)

        async def read(*, include_turns=False):
            assert include_turns is True
            return SimpleNamespace(thread=SimpleNamespace(
                id="thread-123", model_dump=lambda **kwargs: {"id": "thread-123", "turns": []},
            ))

        client.thread.read = read
        clients.append(client)
        return client

    monkeypatch.setattr(codex_gateway, "AsyncCodex", factory)
    gateway = codex_gateway.CodexGateway(str(uuid4()))
    session = await gateway.connect(str(uuid4()), history_only=True)
    history = await session.read_history("gpt-6-luna", "thread-123")
    assert history == {"id": "thread-123", "turns": []}
    assert clients[0].config.launch_args_override[-1] == "--history-only"
    assert clients[0].thread.turn_args is None
    with pytest.raises(ValueError, match="history-only"):
        await session.start_turn(clients[0].thread, "gpt-6-luna", "Must not run")
    assert clients[0].thread.turn_args is None
    await session.close()


def test_history_launcher_disables_platform_mcp_and_clears_run_environment(tmp_path) -> None:
    import os
    import subprocess
    from pathlib import Path

    executable = tmp_path / "codex"
    executable.write_text(
        '#!/bin/sh\n'
        '[ -z "${PLATFORM_TOKEN-}" ] || exit 90\n'
        '[ -z "${UMGPT_TOOLKIT_API_KEY-}" ] || exit 91\n'
        '[ -z "${HTTPS_PROXY-}" ] || exit 92\n'
        'printf "%s\\n" "$@"\n', encoding="utf-8",
    )
    executable.chmod(0o700)
    result = subprocess.run(
        ["/bin/sh", str(Path(__file__).parents[2] / "agent/launch-codex.sh"), "--history-only"],
        env={**os.environ, "PATH": str(tmp_path), "PLATFORM_TOKEN": "local-test-token",
             "UMGPT_TOOLKIT_API_KEY": "local-test-key", "HTTPS_PROXY": "http://invalid.test"},
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == [
        "-c", "mcp_servers.platform.enabled=false", "app-server", "--listen", "stdio://",
    ]


async def test_probe_gateway_uses_scoped_probe_launcher(monkeypatch):
    clients = []

    def factory(config):
        client = FakeCodex(config)
        clients.append(client)
        return client

    monkeypatch.setattr(codex_gateway, "AsyncCodex", factory)
    gateway = codex_gateway.CodexGateway(str(uuid4()))
    session = await gateway.connect(str(uuid4()), probe_only=True)
    assert clients[0].config.launch_args_override[-1] == "--probe"
    with pytest.raises(ValueError):
        await gateway.connect(str(uuid4()), history_only=True, probe_only=True)
    await session.close()
