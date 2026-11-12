from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import signal
import stat
import time
from http import HTTPStatus
from typing import Any

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8765
GATEWAY_SOCKET = "/run/platform-gateway/gateway.sock"
TOKEN_FILE = "/codex/.platform-run-token"
MAX_HTTP_HEADER_BYTES = 16 * 1024
MAX_PROTOCOL_HEADER_BYTES = 4096
MAX_BODY_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
CONNECT_TIMEOUT_SECONDS = 5
TUNNEL_IDLE_TIMEOUT_SECONDS = 120
TURN_DEADLINE_SECONDS = 600
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
logger = logging.getLogger("qpl.agent.network-bridge")
_active_writers: set[asyncio.StreamWriter] = set()


class BridgeError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _read_token() -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(TOKEN_FILE, flags)
    except OSError as error:
        raise BridgeError(401, "A run token is unavailable.") from error
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_gid != os.getegid()
            or info.st_mode & 0o077
            or info.st_size > MAX_PROTOCOL_HEADER_BYTES
        ):
            raise BridgeError(401, "The run token file is invalid.")
        try:
            token_bytes = os.read(descriptor, MAX_PROTOCOL_HEADER_BYTES + 1)
        except OSError as error:
            raise BridgeError(401, "The run token file is invalid.") from error
    finally:
        os.close(descriptor)
    if len(token_bytes) > MAX_PROTOCOL_HEADER_BYTES:
        raise BridgeError(401, "The run token file is invalid.")
    try:
        token = token_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise BridgeError(401, "The run token file is invalid.") from error
    if token.endswith("\n"):
        token = token[:-1]
    if not token or "\n" in token or "\r" in token:
        raise BridgeError(401, "The run token file is invalid.")
    return token


async def _read_header(reader: asyncio.StreamReader) -> bytes:
    try:
        header = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"),
            timeout=10,
        )
    except asyncio.LimitOverrunError as error:
        raise BridgeError(431, "Request headers are too large.") from error
    except asyncio.IncompleteReadError as error:
        raise BridgeError(400, "The HTTP request is incomplete.") from error
    except TimeoutError as error:
        raise BridgeError(408, "The HTTP request timed out.") from error
    if len(header) > MAX_HTTP_HEADER_BYTES:
        raise BridgeError(431, "Request headers are too large.")
    return header


def _parse_headers(header: bytes) -> tuple[str, str, dict[str, str]]:
    try:
        text = header.decode("ascii")
    except UnicodeDecodeError as error:
        raise BridgeError(400, "HTTP headers must be ASCII.") from error
    lines = text[:-4].split("\r\n")
    request_parts = lines[0].split(" ")
    if len(request_parts) != 3 or request_parts[2] not in {"HTTP/1.0", "HTTP/1.1"}:
        raise BridgeError(400, "The HTTP request line is invalid.")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or line[0] in " \t" or ":" not in line:
            raise BridgeError(400, "The HTTP headers are invalid.")
        name, value = line.split(":", 1)
        normalized = name.strip().lower()
        if (
            not normalized
            or not normalized.isascii()
            or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9a-z-]+", normalized)
            or normalized in headers
        ):
            raise BridgeError(400, "The HTTP headers are invalid.")
        headers[normalized] = value.strip()
    return request_parts[0], request_parts[1], headers


def _parse_connect_authority(authority: str, headers: dict[str, str]) -> tuple[str, int]:
    if authority.count(":") != 1 or any(character in authority for character in "/?#@[]"):
        raise BridgeError(400, "Only DNS host CONNECT targets are supported.")
    hostname, raw_port = authority.rsplit(":", 1)
    host = hostname.lower().rstrip(".")
    if not host.isascii() or not host or len(host) > 253:
        raise BridgeError(400, "The CONNECT host is invalid.")
    try:
        ipaddress.ip_address(host)
        raise BridgeError(403, "IP-literal CONNECT targets are not allowed.")
    except ValueError:
        pass
    labels = host.split(".")
    if any(not HOST_LABEL.fullmatch(label) for label in labels):
        raise BridgeError(400, "The CONNECT host is invalid.")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise BridgeError(400, "The CONNECT port is invalid.") from error
    if str(port) != raw_port or port != 443:
        raise BridgeError(403, "Only HTTPS CONNECT on port 443 is allowed.")
    host_header = headers.get("host")
    if host_header is not None and host_header.lower().rstrip(".") not in {
        host,
        f"{host}:443",
    }:
        raise BridgeError(400, "The Host header does not match the CONNECT target.")
    return host, port


def _protocol_line(payload: dict[str, Any]) -> bytes:
    line = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    if len(line) > MAX_PROTOCOL_HEADER_BYTES:
        raise BridgeError(431, "The gateway request header is too large.")
    return line


async def _open_gateway(
    payload: dict[str, Any],
    body: bytes | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, dict[str, Any]]:
    writer: asyncio.StreamWriter | None = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(GATEWAY_SOCKET, limit=MAX_PROTOCOL_HEADER_BYTES + 1),
            timeout=CONNECT_TIMEOUT_SECONDS,
        )
        writer.write(_protocol_line(payload))
        await asyncio.wait_for(writer.drain(), timeout=CONNECT_TIMEOUT_SECONDS)
        if body is not None:
            writer.write(body)
            await asyncio.wait_for(writer.drain(), timeout=CONNECT_TIMEOUT_SECONDS)
        response_line = await asyncio.wait_for(
            reader.readline(),
            timeout=CONNECT_TIMEOUT_SECONDS,
        )
    except (OSError, TimeoutError, BridgeError) as error:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        raise BridgeError(502, "The platform gateway is unavailable.") from error
    if (
        not response_line
        or len(response_line) > MAX_PROTOCOL_HEADER_BYTES
        or not response_line.endswith(b"\n")
    ):
        writer.close()
        await writer.wait_closed()
        raise BridgeError(502, "The platform gateway returned an invalid response.")
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError as error:
        writer.close()
        await writer.wait_closed()
        raise BridgeError(502, "The platform gateway returned an invalid response.") from error
    if not isinstance(response, dict) or not isinstance(response.get("status"), int):
        writer.close()
        await writer.wait_closed()
        raise BridgeError(502, "The platform gateway returned an invalid response.")
    return reader, writer, response


async def _send_http_error(
    writer: asyncio.StreamWriter,
    status: int,
    message: str,
) -> None:
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        status, phrase = 500, "Internal Server Error"
    body = (message + "\n").encode()
    writer.write(
        f"HTTP/1.1 {status} {phrase}\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n".encode()
        + body
    )
    await writer.drain()


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


async def _tunnel(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    gateway_reader: asyncio.StreamReader,
    gateway_writer: asyncio.StreamWriter,
) -> None:
    activity = asyncio.Event()
    tasks = {
        asyncio.create_task(_copy_stream(client_reader, gateway_writer, activity)),
        asyncio.create_task(_copy_stream(gateway_reader, client_writer, activity)),
    }
    started_at = time.monotonic()
    last_activity = started_at
    activity_task: asyncio.Task[bool] | None = None
    try:
        while tasks:
            activity.clear()
            idle_remaining = TUNNEL_IDLE_TIMEOUT_SECONDS - (time.monotonic() - last_activity)
            turn_remaining = TURN_DEADLINE_SECONDS - (time.monotonic() - started_at)
            timeout = min(idle_remaining, turn_remaining)
            if timeout <= 0:
                return
            activity_task = asyncio.create_task(activity.wait())
            done, _pending = await asyncio.wait(
                {*tasks, activity_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                activity_task.cancel()
                await asyncio.gather(activity_task, return_exceptions=True)
                activity_task = None
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
                    if task.cancelled():
                        return
                    if task.exception() is not None:
                        return
                    if task.result() is None:
                        return
    finally:
        if activity_task is not None:
            activity_task.cancel()
            await asyncio.gather(activity_task, return_exceptions=True)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _handle_connect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    authority: str,
    headers: dict[str, str],
) -> None:
    if "transfer-encoding" in headers or "content-length" in headers:
        raise BridgeError(400, "CONNECT requests cannot include a body.")
    host, port = _parse_connect_authority(authority, headers)
    gateway_reader, gateway_writer, response = await _open_gateway(
        {
            "version": 1,
            "operation": "connect",
            "token": _read_token(),
            "host": host,
            "port": port,
        }
    )
    if response["status"] != 200:
        gateway_writer.close()
        await gateway_writer.wait_closed()
        raise BridgeError(502, "The platform gateway rejected the connection.")
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    try:
        await _tunnel(reader, writer, gateway_reader, gateway_writer)
    finally:
        gateway_writer.close()
        await gateway_writer.wait_closed()


async def _read_mcp_body(
    reader: asyncio.StreamReader,
    headers: dict[str, str],
) -> bytes:
    if "transfer-encoding" in headers:
        raise BridgeError(400, "Chunked transfer encoding is not supported.")
    if headers.get("content-encoding", "identity").lower() != "identity":
        raise BridgeError(415, "Compressed MCP requests are not supported.")
    if headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise BridgeError(415, "MCP requests must use application/json.")
    content_length = headers.get("content-length")
    if content_length is None or not content_length.isascii() or not content_length.isdecimal():
        raise BridgeError(411, "MCP requests require a valid Content-Length.")
    if len(content_length) > 10:
        raise BridgeError(413, "The MCP request body is too large.")
    body_length = int(content_length)
    if body_length > MAX_BODY_BYTES:
        raise BridgeError(413, "The MCP request body is too large.")
    try:
        body = await asyncio.wait_for(reader.readexactly(body_length), timeout=10)
        parsed = json.loads(body)
    except (asyncio.IncompleteReadError, TimeoutError) as error:
        raise BridgeError(400, "The MCP request body is incomplete.") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BridgeError(400, "The MCP request body is not valid JSON.") from error
    if not isinstance(parsed, dict):
        raise BridgeError(400, "The MCP request body must be a JSON object.")
    return body


async def _handle_mcp(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    method: str,
    target: str,
    headers: dict[str, str],
) -> None:
    if method != "POST" or target != "/mcp":
        raise BridgeError(404, "Only POST /mcp is available.")
    if headers.get("host", "").lower() not in {
        "127.0.0.1:8765",
        "127.0.0.1",
    }:
        raise BridgeError(400, "The MCP Host header is invalid.")
    if headers.get("expect"):
        raise BridgeError(417, "Expect headers are not supported.")
    body = await _read_mcp_body(reader, headers)
    forwarded_headers = {
        name: value
        for name, value in headers.items()
        if name in MCP_REQUEST_HEADERS
    }
    gateway_reader, gateway_writer, response = await _open_gateway(
        {
            "version": 1,
            "operation": "mcp",
            "token": _read_token(),
            "method": "POST",
            "body_length": len(body),
            "headers": forwarded_headers,
        },
        body,
    )
    try:
        body_length = response.get("body_length")
        response_headers = response.get("headers")
        status = response["status"]
        if (
            not 200 <= status <= 599
            or not isinstance(body_length, int)
            or not 0 <= body_length <= MAX_RESPONSE_BYTES
            or not isinstance(response_headers, dict)
        ):
            raise BridgeError(502, "The platform gateway returned an invalid response.")
        response_body = await asyncio.wait_for(
            gateway_reader.readexactly(body_length),
            timeout=10,
        )
    except (asyncio.IncompleteReadError, TimeoutError) as error:
        raise BridgeError(502, "The platform gateway response was incomplete.") from error
    finally:
        gateway_writer.close()
        await gateway_writer.wait_closed()

    safe_headers: list[tuple[str, str]] = []
    for name, value in response_headers.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or name.lower() not in MCP_RESPONSE_HEADERS
            or "\r" in value
            or "\n" in value
        ):
            continue
        safe_headers.append((name.lower(), value))
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        phrase = "Response"
    writer.write(
        f"HTTP/1.1 {status} {phrase}\r\n".encode()
        + b"".join(f"{name}: {value}\r\n".encode() for name, value in safe_headers)
        + f"Content-Length: {len(response_body)}\r\nConnection: close\r\n\r\n".encode()
        + response_body
    )
    await writer.drain()


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    _active_writers.add(writer)
    try:
        method, target, headers = _parse_headers(await _read_header(reader))
        if method == "CONNECT":
            await _handle_connect(reader, writer, target, headers)
        else:
            await _handle_mcp(reader, writer, method, target, headers)
    except BridgeError as error:
        try:
            await _send_http_error(writer, error.status, str(error))
        except (ConnectionError, OSError):
            pass
    except (ConnectionError, OSError):
        logger.info("Agent network bridge client disconnected")
    finally:
        _active_writers.discard(writer)
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def serve() -> None:
    server = await asyncio.start_server(
        _handle_client,
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        limit=MAX_HTTP_HEADER_BYTES + 1,
    )
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, stopped.set)
    async with server:
        await stopped.wait()
        server.close()
        await server.wait_closed()
    for writer in tuple(_active_writers):
        writer.close()
    await asyncio.gather(
        *(writer.wait_closed() for writer in tuple(_active_writers)),
        return_exceptions=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
