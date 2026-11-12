from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import re
import secrets
import socket
import stat
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import UUID

import httpx

from .config import Settings

PROTOCOL_VERSION = 1
MAX_PROTOCOL_HEADER_BYTES = 4096
MAX_MCP_BYTES = 1024 * 1024
MAX_TUNNELS_PER_RUN = 8
MAX_TUNNELS_PER_INSTANCE = 16
CONNECT_TIMEOUT_SECONDS = 5
TUNNEL_IDLE_TIMEOUT_SECONDS = 120
RUN_DEADLINE_SECONDS = 600
BACKEND_MCP_URL = "http://127.0.0.1:8000/mcp"
MCP_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "content-type",
        "mcp-protocol-version",
        "mcp-session-id",
        "last-event-id",
    }
)
MCP_RESPONSE_HEADERS = frozenset(
    {"content-type", "mcp-protocol-version", "mcp-session-id", "last-event-id"}
)
HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class GatewayError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class CloseableWriter(Protocol):
    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TokenContext:
    scope: Literal["run", "probe"]
    instance_id: str
    run_id: str | None
    agent_id: str | None
    generation: int | None
    allowed_hosts: frozenset[str]
    expires_at: float


class TokenAuthority:
    def __init__(self, instance_id: str, allowed_hosts: frozenset[str]) -> None:
        self.instance_id = self._canonical_uuid(instance_id, "instance_id")
        self.allowed_hosts = frozenset(host.lower().rstrip(".") for host in allowed_hosts)
        self._tokens: dict[str, TokenContext] = {}
        self._connections: dict[str, set[CloseableWriter]] = {}

    def issue_run_token(self, run_id: str, agent_id: str, generation: int) -> str:
        run_id = self._canonical_uuid(run_id, "run_id")
        agent_id = self._canonical_uuid(agent_id, "agent_id")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        return self._issue(
            secrets.token_urlsafe(32),
            TokenContext(
                scope="run",
                instance_id=self.instance_id,
                run_id=run_id,
                agent_id=agent_id,
                generation=generation,
                allowed_hosts=self.allowed_hosts,
                expires_at=time.monotonic() + RUN_DEADLINE_SECONDS,
            ),
        )

    def issue_probe_token(self, instance_id: str, lifetime_seconds: int = 600) -> str:
        if self._canonical_uuid(instance_id, "instance_id") != self.instance_id:
            raise ValueError("probe token instance must match this gateway")
        if (
            isinstance(lifetime_seconds, bool)
            or not isinstance(lifetime_seconds, int)
            or not 1 <= lifetime_seconds <= RUN_DEADLINE_SECONDS
        ):
            raise ValueError("probe token lifetime is outside the permitted range")
        return self._issue(
            secrets.token_urlsafe(32),
            TokenContext(
                scope="probe",
                instance_id=self.instance_id,
                run_id=None,
                agent_id=None,
                generation=None,
                allowed_hosts=self.allowed_hosts,
                expires_at=time.monotonic() + lifetime_seconds,
            ),
        )

    def _issue(self, raw_token: str, context: TokenContext) -> str:
        self._tokens[self.token_hash(raw_token)] = context
        return raw_token

    @staticmethod
    def _canonical_uuid(value: str, name: str) -> str:
        parsed = str(UUID(value))
        if parsed != value:
            raise ValueError(f"{name} must be a canonical UUID")
        return parsed

    @staticmethod
    def token_hash(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    def resolve(self, raw_token: Any, operation: str) -> tuple[str, TokenContext]:
        if not isinstance(raw_token, str) or not raw_token or len(raw_token) > 256:
            raise GatewayError(403, "The gateway token is invalid.")
        token_hash = self.token_hash(raw_token)
        context = self._tokens.get(token_hash)
        if context is None or context.expires_at <= time.monotonic():
            self.revoke_hash(token_hash)
            raise GatewayError(403, "The gateway token is invalid or expired.")
        if context.instance_id != self.instance_id:
            raise GatewayError(403, "The gateway token belongs to another instance.")
        if context.scope == "probe" and operation != "connect":
            raise GatewayError(403, "Probe tokens cannot access MCP.")
        if operation not in {"connect", "mcp"}:
            raise GatewayError(403, "The gateway operation is not authorized.")
        return token_hash, context

    def add_connection(self, token_hash: str, writer: CloseableWriter) -> None:
        self._connections.setdefault(token_hash, set()).add(writer)

    def remove_connection(self, token_hash: str, writer: CloseableWriter) -> None:
        connections = self._connections.get(token_hash)
        if connections is None:
            return
        connections.discard(writer)
        if not connections:
            self._connections.pop(token_hash, None)

    def revoke_token(self, raw_token: str) -> None:
        self.revoke_hash(self.token_hash(raw_token))

    def revoke_run(self, run_id: str) -> set[CloseableWriter]:
        canonical_run_id = str(UUID(run_id))
        connections: set[CloseableWriter] = set()
        for token_hash, context in tuple(self._tokens.items()):
            if context.scope == "run" and context.run_id == canonical_run_id:
                connections.update(self.revoke_hash(token_hash))
        return connections

    def revoke_all(self) -> None:
        for token_hash in tuple(self._tokens):
            self.revoke_hash(token_hash)

    def revoke_hash(self, token_hash: str) -> set[CloseableWriter]:
        self._tokens.pop(token_hash, None)
        connections = self._connections.pop(token_hash, set())
        for writer in connections:
            writer.close()
        return connections


class NetworkGateway:
    def __init__(self, settings: Settings, instance_id: str) -> None:
        self.settings = settings
        self.instance_id = str(UUID(instance_id))
        self.authority = TokenAuthority(self.instance_id, settings.allowed_model_hosts)
        self._server: asyncio.AbstractServer | None = None
        self._http: httpx.AsyncClient | None = None
        self._active_clients: set[asyncio.StreamWriter] = set()
        self._run_tunnel_counts: dict[str, int] = {}
        self._instance_tunnel_count = 0
        self._revoked_run_writers: dict[str, set[CloseableWriter]] = {}

    def issue_run_token(self, run_id: str, agent_id: str, generation: int) -> str:
        return self.authority.issue_run_token(run_id, agent_id, generation)

    def revoke_run(self, run_id: str) -> None:
        canonical_run_id = str(UUID(run_id))
        writers = self.authority.revoke_run(canonical_run_id)
        if writers:
            self._revoked_run_writers.setdefault(canonical_run_id, set()).update(writers)

    async def wait_run_closed(self, run_id: str) -> None:
        canonical_run_id = str(UUID(run_id))
        writers = self._revoked_run_writers.pop(canonical_run_id, set())
        if writers:
            await asyncio.gather(
                *(writer.wait_closed() for writer in writers),
                return_exceptions=True,
            )

    @staticmethod
    def token_hash(raw_token: str) -> str:
        return TokenAuthority.token_hash(raw_token)

    async def start(self) -> None:
        if os.geteuid() != 1000 or os.getegid() != 1000:
            raise RuntimeError("The gateway must run as UID/GID 1000")
        try:
            socket_subpath = self.settings.gateway_socket.parent.relative_to(
                self.settings.platform_state_root
            )
        except ValueError as error:
            raise RuntimeError("The gateway socket must be inside platform state") from error
        if not socket_subpath.parts or ".." in socket_subpath.parts:
            raise RuntimeError("The gateway socket must be inside platform state")
        parent = self.settings.gateway_socket.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_info = parent.lstat()
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or stat.S_ISLNK(parent_info.st_mode)
            or parent_info.st_uid != 1000
            or parent_info.st_gid != 1000
            or parent_info.st_mode & 0o077
        ):
            raise RuntimeError("The gateway socket directory has unsafe ownership or mode")

        try:
            socket_info = self.settings.gateway_socket.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(socket_info.st_mode):
                raise RuntimeError("The gateway socket path exists and is not a socket")
            self.settings.gateway_socket.unlink()

        self._http = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(10, connect=CONNECT_TIMEOUT_SECONDS),
            trust_env=False,
        )
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_connection,
                path=self.settings.gateway_socket,
                limit=MAX_PROTOCOL_HEADER_BYTES + 1,
            )
            os.chown(self.settings.gateway_socket, 1000, 1000)
            os.chmod(self.settings.gateway_socket, 0o600)
        except OSError:
            await self.close()
            raise

    async def close(self) -> None:
        self.authority.revoke_all()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for writer in tuple(self._active_clients):
            writer.close()
        await asyncio.gather(
            *(writer.wait_closed() for writer in tuple(self._active_clients)),
            return_exceptions=True,
        )
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        try:
            if stat.S_ISSOCK(self.settings.gateway_socket.lstat().st_mode):
                self.settings.gateway_socket.unlink()
        except FileNotFoundError:
            pass

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._active_clients.add(writer)
        operation = ""
        try:
            request = await self._read_protocol_request(reader)
            operation = request.get("operation", "")
            if operation == "connect":
                await self._handle_connect(reader, writer, request)
            elif operation == "mcp":
                await self._handle_mcp(reader, writer, request)
            else:
                raise GatewayError(400, "The gateway operation is invalid.")
        except GatewayError as error:
            if operation == "mcp":
                await self._write_mcp_response(writer, error.status, {}, b"")
            else:
                await self._write_json_line(writer, {"status": error.status})
        except (ConnectionError, OSError):
            pass
        finally:
            self._active_clients.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _read_protocol_request(self, reader: asyncio.StreamReader) -> dict[str, Any]:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=CONNECT_TIMEOUT_SECONDS)
        except (TimeoutError, asyncio.LimitOverrunError, ValueError) as error:
            raise GatewayError(400, "The gateway protocol header is invalid.") from error
        if (
            not line
            or len(line) > MAX_PROTOCOL_HEADER_BYTES
            or not line.endswith(b"\n")
        ):
            raise GatewayError(400, "The gateway protocol header is invalid.")
        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        try:
            request = json.loads(line, object_pairs_hook=reject_duplicate_keys)
        except ValueError as error:
            raise GatewayError(400, "The gateway protocol header is invalid.") from error
        if (
            not isinstance(request, dict)
            or request.get("version") != PROTOCOL_VERSION
            or isinstance(request.get("version"), bool)
        ):
            raise GatewayError(400, "The gateway protocol version is invalid.")
        return request

    async def _write_json_line(
        self,
        writer: asyncio.StreamWriter,
        response: dict[str, Any],
    ) -> None:
        line = json.dumps(response, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
        if len(line) > MAX_PROTOCOL_HEADER_BYTES:
            line = b'{"status":502}\n'
        writer.write(line)
        await writer.drain()

    async def _handle_connect(
        self,
        reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        request: dict[str, Any],
    ) -> None:
        if set(request) != {"version", "operation", "token", "host", "port"}:
            raise GatewayError(400, "The CONNECT request fields are invalid.")
        token_hash, context = self.authority.resolve(request["token"], "connect")
        host = self._normalize_host(request["host"], context.allowed_hosts)
        port = request["port"]
        if isinstance(port, bool) or not isinstance(port, int) or port != 443:
            raise GatewayError(403, "Only HTTPS connections on port 443 are allowed.")
        run_key = context.run_id or f"probe:{token_hash}"
        if (
            self._instance_tunnel_count >= MAX_TUNNELS_PER_INSTANCE
            or self._run_tunnel_counts.get(run_key, 0) >= MAX_TUNNELS_PER_RUN
        ):
            raise GatewayError(429, "The gateway tunnel limit has been reached.")
        self._instance_tunnel_count += 1
        self._run_tunnel_counts[run_key] = self._run_tunnel_counts.get(run_key, 0) + 1
        upstream_writer: asyncio.StreamWriter | None = None
        try:
            candidates = await self._resolve_public_addresses(host, port)
            self.authority.resolve(request["token"], "connect")
            upstream_reader, upstream_writer = await self._connect_candidates(candidates, port)
            self.authority.resolve(request["token"], "connect")
            self.authority.add_connection(token_hash, upstream_writer)
            await self._write_json_line(client_writer, {"status": 200})
            await self._tunnel(reader, client_writer, upstream_reader, upstream_writer)
        except GatewayError:
            raise
        except (OSError, TimeoutError):
            raise GatewayError(
                502,
                "The requested public HTTPS endpoint is unavailable.",
            ) from None
        finally:
            if upstream_writer is not None:
                self.authority.remove_connection(token_hash, upstream_writer)
                upstream_writer.close()
                try:
                    await upstream_writer.wait_closed()
                except (ConnectionError, OSError):
                    pass
            self._instance_tunnel_count -= 1
            remaining = self._run_tunnel_counts.get(run_key, 1) - 1
            if remaining:
                self._run_tunnel_counts[run_key] = remaining
            else:
                self._run_tunnel_counts.pop(run_key, None)

    @staticmethod
    def _normalize_host(host: Any, allowed_hosts: frozenset[str]) -> str:
        if not isinstance(host, str) or len(host) > 253 or not host.isascii():
            raise GatewayError(403, "The requested host is not allowed.")
        normalized = host.lower().rstrip(".")
        if not normalized or any(
            not HOST_LABEL.fullmatch(label) for label in normalized.split(".")
        ):
            raise GatewayError(403, "The requested host is not allowed.")
        try:
            ipaddress.ip_address(normalized)
        except ValueError:
            pass
        else:
            raise GatewayError(403, "IP-literal destinations are not allowed.")
        if normalized not in allowed_hosts:
            raise GatewayError(403, "The requested host is not allowed.")
        return normalized

    async def _lookup_addresses(
        self,
        host: str,
        port: int,
    ) -> list[tuple[int, int, int, str, tuple[Any, ...]]]:
        loop = asyncio.get_running_loop()
        return await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)

    async def _resolve_public_addresses(
        self,
        host: str,
        port: int,
    ) -> list[tuple[int, str]]:
        try:
            records = await asyncio.wait_for(
                self._lookup_addresses(host, port),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except (OSError, TimeoutError) as error:
            raise GatewayError(502, "The requested host could not be resolved.") from error
        if not records:
            raise GatewayError(502, "The requested host has no public address.")
        candidates: list[tuple[int, str]] = []
        seen: set[tuple[int, str]] = set()
        for family, _socktype, _protocol, _canonname, sockaddr in records:
            address_text = sockaddr[0]
            if not isinstance(address_text, str) or "%" in address_text:
                raise GatewayError(403, "The requested host resolved to a disallowed address.")
            try:
                address = ipaddress.ip_address(address_text)
            except ValueError as error:
                raise GatewayError(
                    403,
                    "The requested host resolved to a disallowed address.",
                ) from error
            mapped_address = (
                address.ipv4_mapped
                if isinstance(address, ipaddress.IPv6Address)
                else None
            )
            effective_address = mapped_address or address
            if (
                not effective_address.is_global
                or effective_address.is_multicast
                or effective_address.is_reserved
                or effective_address.is_unspecified
                or effective_address.is_loopback
                or effective_address.is_link_local
                or effective_address.is_private
            ):
                raise GatewayError(403, "The requested host resolved to a disallowed address.")
            if family not in {socket.AF_INET, socket.AF_INET6}:
                raise GatewayError(403, "The requested host resolved to an invalid address family.")
            candidate = (family, str(address))
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
        return candidates

    async def _connect_candidates(
        self,
        candidates: list[tuple[int, str]],
        port: int,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        async def connect() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
            last_error: OSError | None = None
            for family, address in candidates:
                try:
                    return await asyncio.open_connection(
                        host=address,
                        port=port,
                        family=family,
                    )
                except OSError as error:
                    last_error = error
            if last_error is not None:
                raise last_error
            raise OSError("no validated destination addresses")

        return await asyncio.wait_for(connect(), timeout=CONNECT_TIMEOUT_SECONDS)

    async def _tunnel(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        upstream_reader: asyncio.StreamReader,
        upstream_writer: asyncio.StreamWriter,
    ) -> None:
        activity = asyncio.Event()
        tasks = {
            asyncio.create_task(self._copy_stream(client_reader, upstream_writer, activity)),
            asyncio.create_task(self._copy_stream(upstream_reader, client_writer, activity)),
        }
        started_at = time.monotonic()
        last_activity = started_at
        activity_task: asyncio.Task[bool] | None = None
        try:
            while tasks:
                activity.clear()
                idle_remaining = TUNNEL_IDLE_TIMEOUT_SECONDS - (time.monotonic() - last_activity)
                deadline_remaining = RUN_DEADLINE_SECONDS - (time.monotonic() - started_at)
                timeout = min(idle_remaining, deadline_remaining)
                if timeout <= 0:
                    return
                activity_task = asyncio.create_task(activity.wait())
                done, _pending = await asyncio.wait(
                    {*tasks, activity_task},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    return
                if activity_task in done:
                    last_activity = time.monotonic()
                else:
                    activity_task.cancel()
                    await asyncio.gather(activity_task, return_exceptions=True)
                activity_task = None
                for task in tuple(tasks):
                    if task in done:
                        tasks.remove(task)
                        if task.cancelled() or task.exception() is not None:
                            return
                        return
        finally:
            if activity_task is not None:
                activity_task.cancel()
                await asyncio.gather(activity_task, return_exceptions=True)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _copy_stream(
        source: asyncio.StreamReader,
        destination: asyncio.StreamWriter,
        activity: asyncio.Event,
    ) -> None:
        while True:
            data = await source.read(64 * 1024)
            if not data:
                try:
                    destination.write_eof()
                    await destination.drain()
                except (OSError, RuntimeError):
                    pass
                return
            destination.write(data)
            await destination.drain()
            activity.set()

    async def _handle_mcp(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        request: dict[str, Any],
    ) -> None:
        if set(request) != {
            "version",
            "operation",
            "token",
            "method",
            "body_length",
            "headers",
        }:
            raise GatewayError(400, "The MCP request fields are invalid.")
        token_hash, _context = self.authority.resolve(request["token"], "mcp")
        if request["method"] != "POST":
            raise GatewayError(403, "Only MCP POST requests are supported.")
        body_length = request["body_length"]
        if (
            isinstance(body_length, bool)
            or not isinstance(body_length, int)
            or not 0 <= body_length <= MAX_MCP_BYTES
        ):
            raise GatewayError(413, "The MCP request body is too large.")
        headers = self._normalize_mcp_headers(request["headers"])
        try:
            body = await asyncio.wait_for(reader.readexactly(body_length), timeout=10)
        except (asyncio.IncompleteReadError, TimeoutError) as error:
            raise GatewayError(400, "The MCP request body is incomplete.") from error
        self.authority.resolve(request["token"], "mcp")
        if self._http is None:
            raise GatewayError(503, "The MCP gateway is unavailable.")
        headers["authorization"] = f"Bearer {request['token']}"
        response_headers: dict[str, str] = {}
        response_body = bytearray()
        try:
            async with self._http.stream(
                "POST",
                BACKEND_MCP_URL,
                headers=headers,
                content=body,
            ) as response:
                status_code = response.status_code
                for name, value in response.headers.items():
                    if (
                        name.lower() in MCP_RESPONSE_HEADERS
                        and "\r" not in value
                        and "\n" not in value
                    ):
                        response_headers[name.lower()] = value
                async for chunk in response.aiter_bytes():
                    if len(response_body) + len(chunk) > MAX_MCP_BYTES:
                        raise GatewayError(502, "The MCP response body is too large.")
                    response_body.extend(chunk)
        except httpx.RequestError as error:
            raise GatewayError(502, "The backend MCP endpoint is unavailable.") from error
        await self._write_mcp_response(
            writer,
            status_code,
            response_headers,
            bytes(response_body),
        )

    @staticmethod
    def _normalize_mcp_headers(raw_headers: Any) -> dict[str, str]:
        if not isinstance(raw_headers, dict):
            raise GatewayError(400, "The MCP headers are invalid.")
        normalized: dict[str, str] = {}
        for name, value in raw_headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise GatewayError(400, "The MCP headers are invalid.")
            name = name.lower()
            if (
                name not in MCP_REQUEST_HEADERS
                or "\r" in value
                or "\n" in value
                or len(value) > 1024
            ):
                raise GatewayError(400, "The MCP headers are invalid.")
            normalized[name] = value
        return normalized

    async def _write_mcp_response(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        if not 200 <= status <= 599 or len(body) > MAX_MCP_BYTES:
            status, headers, body = 502, {}, b""
        safe_headers = {
            name.lower(): value
            for name, value in headers.items()
            if name.lower() in MCP_RESPONSE_HEADERS
            and isinstance(value, str)
            and "\r" not in value
            and "\n" not in value
            and len(value) <= 1024
        }
        response = {
            "status": status,
            "headers": safe_headers,
            "body_length": len(body),
        }
        line = json.dumps(response, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
        if len(line) > MAX_PROTOCOL_HEADER_BYTES:
            line = b'{"status":502,"headers":{},"body_length":0}\n'
            body = b""
        writer.write(line + body)
        await writer.drain()
