from __future__ import annotations

import asyncio
import json
import socket
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import httpx
import pytest

from backend.app import network_gateway as gateway_module
from backend.app.config import Settings
from backend.app.network_gateway import GatewayError, NetworkGateway


class TrackingWriter:
    def __init__(self) -> None:
        self.closed = False
        self.closed_event = asyncio.Event()

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        await self.closed_event.wait()


def _load_bridge() -> ModuleType:
    bridge_path = Path(__file__).parents[2] / "agent" / "network_bridge.py"
    spec = spec_from_file_location("qpl_agent_network_bridge", bridge_path)
    assert spec is not None and spec.loader is not None
    bridge = module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge


def _short_socket_path() -> Path:
    return Path("/tmp") / f"qpl-{uuid4().hex[:12]}.sock"


@pytest.mark.asyncio
async def test_run_revocation_waits_for_existing_gateway_channels(
    settings: Settings,
) -> None:
    gateway = NetworkGateway(settings, str(uuid4()))
    run_id = str(uuid4())
    agent_id = str(uuid4())
    token = gateway.issue_run_token(run_id, agent_id, generation=0)
    writer = TrackingWriter()
    gateway.authority.add_connection(gateway.authority.token_hash(token), writer)

    gateway.revoke_run(run_id)
    waiting = asyncio.create_task(gateway.wait_run_closed(run_id))
    await asyncio.sleep(0)
    assert writer.closed
    assert not waiting.done()

    writer.closed_event.set()
    await asyncio.wait_for(waiting, timeout=1)


async def _start_bridge(bridge: ModuleType) -> tuple[asyncio.AbstractServer, int]:
    server = await asyncio.start_server(
        bridge._handle_client,
        host="127.0.0.1",
        port=0,
        limit=bridge.MAX_HTTP_HEADER_BYTES + 1,
    )
    socket = server.sockets[0]
    return server, socket.getsockname()[1]


@pytest.mark.asyncio
async def test_bridge_forwards_only_mcp_headers_and_preserves_202(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _load_bridge()
    gateway_path = _short_socket_path()
    token_path = tmp_path / "run-token"
    token_path.write_text("test-run-token\n", encoding="utf-8")
    token_path.chmod(0o600)
    monkeypatch.setattr(bridge, "GATEWAY_SOCKET", str(gateway_path))
    monkeypatch.setattr(bridge, "TOKEN_FILE", str(token_path))
    received: list[tuple[dict[str, object], bytes]] = []

    async def gateway_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request = json.loads(await reader.readline())
        body = await reader.readexactly(request["body_length"])
        received.append((request, body))
        writer.write(
            b'{"status":202,"headers":{"content-type":"application/json"},'
            b'"body_length":0}\n'
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    gateway = await asyncio.start_unix_server(gateway_handler, path=gateway_path)
    bridge_server, port = await _start_bridge(bridge)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"POST /mcp HTTP/1.1\r\n"
            b"Host: 127.0.0.1:8765\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 2\r\n"
            b"Authorization: Bearer forged\r\n\r\n{}"
        )
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
    finally:
        bridge_server.close()
        gateway.close()
        await bridge_server.wait_closed()
        await gateway.wait_closed()
        gateway_path.unlink(missing_ok=True)

    assert response.startswith(b"HTTP/1.1 202")
    assert len(received) == 1
    request, body = received[0]
    assert request["operation"] == "mcp"
    assert request["token"] == "test-run-token"
    assert request["headers"] == {"content-type": "application/json"}
    assert body == b"{}"


@pytest.mark.asyncio
async def test_bridge_connect_tunnels_only_after_gateway_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _load_bridge()
    gateway_path = _short_socket_path()
    token_path = tmp_path / "run-token"
    token_path.write_text("test-run-token\n", encoding="utf-8")
    token_path.chmod(0o600)
    monkeypatch.setattr(bridge, "GATEWAY_SOCKET", str(gateway_path))
    monkeypatch.setattr(bridge, "TOKEN_FILE", str(token_path))
    received: list[dict[str, object]] = []

    async def gateway_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        received.append(json.loads(await reader.readline()))
        writer.write(b'{"status":200}\n')
        await writer.drain()
        while data := await reader.read(1024):
            writer.write(data)
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    gateway = await asyncio.start_unix_server(gateway_handler, path=gateway_path)
    bridge_server, port = await _start_bridge(bridge)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"CONNECT api.openai.com:443 HTTP/1.1\r\n"
            b"Host: api.openai.com:443\r\n\r\n"
        )
        await writer.drain()
        response = await reader.readuntil(b"\r\n\r\n")
        writer.write(b"hello")
        await writer.drain()
        echoed = await reader.readexactly(5)
        writer.close()
        await writer.wait_closed()
    finally:
        bridge_server.close()
        gateway.close()
        await bridge_server.wait_closed()
        await gateway.wait_closed()
        gateway_path.unlink(missing_ok=True)

    assert response.startswith(b"HTTP/1.1 200")
    assert echoed == b"hello"
    assert received == [
        {
            "version": 1,
            "operation": "connect",
            "token": "test-run-token",
            "host": "api.openai.com",
            "port": 443,
        }
    ]


@pytest.mark.asyncio
async def test_bridge_rejects_private_connect_target_without_gateway_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _load_bridge()
    gateway_path = _short_socket_path()
    token_path = tmp_path / "run-token"
    token_path.write_text("test-run-token\n", encoding="utf-8")
    token_path.chmod(0o600)
    monkeypatch.setattr(bridge, "GATEWAY_SOCKET", str(gateway_path))
    monkeypatch.setattr(bridge, "TOKEN_FILE", str(token_path))
    gateway_calls = 0

    async def gateway_handler(
        _reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        nonlocal gateway_calls
        gateway_calls += 1
        writer.close()
        await writer.wait_closed()

    gateway = await asyncio.start_unix_server(gateway_handler, path=gateway_path)
    bridge_server, port = await _start_bridge(bridge)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"CONNECT 127.0.0.1:443 HTTP/1.1\r\n"
            b"Host: 127.0.0.1:443\r\n\r\n"
        )
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
    finally:
        bridge_server.close()
        gateway.close()
        await bridge_server.wait_closed()
        await gateway.wait_closed()
        gateway_path.unlink(missing_ok=True)

    assert response.startswith(b"HTTP/1.1 403")
    assert gateway_calls == 0


@pytest.mark.asyncio
async def test_bridge_rejects_oversized_mcp_body_and_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _load_bridge()
    gateway_path = _short_socket_path()
    token_path = tmp_path / "run-token"
    token_path.write_text("test-run-token\n", encoding="utf-8")
    token_path.chmod(0o600)
    monkeypatch.setattr(bridge, "GATEWAY_SOCKET", str(gateway_path))
    monkeypatch.setattr(bridge, "TOKEN_FILE", str(token_path))
    bridge_server, port = await _start_bridge(bridge)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"POST /mcp HTTP/1.1\r\n"
            b"Host: 127.0.0.1:8765\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {bridge.MAX_BODY_BYTES + 1}\r\n\r\n".encode()
        )
        await writer.drain()
        body_response = await reader.read()
        writer.close()
        await writer.wait_closed()

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"POST /mcp HTTP/1.1\r\n"
            b"Host: 127.0.0.1:8765\r\n"
            b"X-Fill: "
            + b"a" * bridge.MAX_HTTP_HEADER_BYTES
            + b"\r\n\r\n"
        )
        await writer.drain()
        header_response = await reader.read()
        writer.close()
        await writer.wait_closed()
    finally:
        bridge_server.close()
        await bridge_server.wait_closed()

    assert body_response.startswith(b"HTTP/1.1 413")
    assert header_response.startswith(b"HTTP/1.1 431")


def test_gateway_run_and_probe_tokens_have_separate_authority(settings: Settings) -> None:
    gateway = NetworkGateway(settings, "9e17951f-254d-47f4-9716-165434aad530")
    run_token = gateway.authority.issue_run_token(
        "7422c320-c257-4e8d-a629-e797fe28bdfe",
        "77b06fad-6ad9-45b4-92f3-38a7b86f0862",
        3,
    )
    run_hash, context = gateway.authority.resolve(run_token, "mcp")
    assert len(run_hash) == 64
    assert context.run_id == "7422c320-c257-4e8d-a629-e797fe28bdfe"
    assert context.agent_id == "77b06fad-6ad9-45b4-92f3-38a7b86f0862"
    assert context.generation == 3

    class TrackedWriter:
        closed = False

        def close(self) -> None:
            self.closed = True

    active_writer = TrackedWriter()
    gateway.authority.add_connection(run_hash, active_writer)
    probe_token = gateway.authority.issue_probe_token(gateway.instance_id)
    gateway.authority.resolve(probe_token, "connect")
    with pytest.raises(GatewayError, match="cannot access MCP"):
        gateway.authority.resolve(probe_token, "mcp")
    gateway.authority.revoke_token(run_token)
    assert active_writer.closed
    with pytest.raises(GatewayError, match="invalid or expired"):
        gateway.authority.resolve(run_token, "connect")


@pytest.mark.asyncio
async def test_gateway_rejects_dns_rebinding_to_private_or_metadata_addresses(
    settings: Settings,
) -> None:
    gateway = NetworkGateway(settings, "9e17951f-254d-47f4-9716-165434aad530")

    async def rebinding_lookup(host: str, port: int):
        assert host == "api.openai.com"
        assert port == 443
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443)),
        ]

    gateway._lookup_addresses = rebinding_lookup
    with pytest.raises(GatewayError, match="disallowed address"):
        await gateway._resolve_public_addresses("api.openai.com", 443)

    async def mapped_metadata_lookup(host: str, port: int):
        return [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("::ffff:169.254.169.254", port, 0, 0),
            )
        ]

    gateway._lookup_addresses = mapped_metadata_lookup
    with pytest.raises(GatewayError, match="disallowed address"):
        await gateway._resolve_public_addresses("api.openai.com", 443)

    async def multicast_lookup(host: str, port: int):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("224.0.0.1", port)),
        ]

    gateway._lookup_addresses = multicast_lookup
    with pytest.raises(GatewayError, match="disallowed address"):
        await gateway._resolve_public_addresses("api.openai.com", 443)

    async def public_lookup(host: str, port: int):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
        ]

    gateway._lookup_addresses = public_lookup
    assert await gateway._resolve_public_addresses("api.openai.com", 443) == [
        (socket.AF_INET, "93.184.216.34")
    ]


@pytest.mark.asyncio
async def test_gateway_mcp_uses_verified_token_and_preserves_notification_status(
    settings: Settings,
) -> None:
    gateway = NetworkGateway(settings, "9e17951f-254d-47f4-9716-165434aad530")
    token = gateway.authority.issue_run_token(
        "7422c320-c257-4e8d-a629-e797fe28bdfe",
        "77b06fad-6ad9-45b4-92f3-38a7b86f0862",
        3,
    )
    captured: list[httpx.Request] = []

    async def backend_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            202,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "session-1",
                "x-internal": "not-forwarded",
            },
            content=b"",
        )

    gateway._http = httpx.AsyncClient(
        transport=httpx.MockTransport(backend_handler),
        follow_redirects=False,
        trust_env=False,
    )
    socket_path = _short_socket_path()
    server = await asyncio.start_unix_server(
        gateway._handle_connection,
        path=socket_path,
        limit=gateway_module.MAX_PROTOCOL_HEADER_BYTES + 1,
    )
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
        body = b'{"jsonrpc":"2.0","method":"notifications/initialized"}'
        writer.write(
            json.dumps(
                {
                    "version": 1,
                    "operation": "mcp",
                    "token": token,
                    "method": "POST",
                    "body_length": len(body),
                    "headers": {"content-type": "application/json"},
                },
                separators=(",", ":"),
            ).encode()
            + b"\n"
            + body
        )
        await writer.drain()
        response = json.loads(await reader.readline())
        response_body = await reader.readexactly(response["body_length"])
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()
        socket_path.unlink(missing_ok=True)
        await gateway._http.aclose()

    assert response == {
        "status": 202,
        "headers": {
            "content-type": "application/json",
            "mcp-session-id": "session-1",
        },
        "body_length": 0,
    }
    assert response_body == b""
    assert len(captured) == 1
    assert captured[0].url == httpx.URL("http://127.0.0.1:8000/mcp")
    assert captured[0].headers["authorization"] == f"Bearer {token}"
    assert captured[0].content == body
