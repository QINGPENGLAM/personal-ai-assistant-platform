#!/usr/bin/env python3
"""Loopback-only browser test service; fake mode never launches Docker or a model."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402


class TestContainers:
    async def start_agent(self, agent_id):
        return None

    async def stop_agent(self, agent_id):
        return None

    def write_run_token(self, agent_id, token):
        return None

    def remove_run_token(self, agent_id):
        return None


class TestNetwork:
    def issue_run_token(self, run_id, agent_id, generation):
        return f"test-{run_id}-{generation}"

    def revoke_run(self, run_id):
        return None

    async def wait_run_closed(self, run_id):
        return None

    @staticmethod
    def token_hash(token):
        return hashlib.sha256(token.encode()).hexdigest()


class TestTurn:
    id = "test-turn"

    def __init__(self, prompt):
        self.prompt = prompt

    async def stream(self):
        yield SimpleNamespace(method="item/started", payload={
            "item": {
                "id": "test-activity", "type": "commandExecution",
                "command": "Test harness activity",
            },
        })
        if self.prompt == "Wait for stop":
            await asyncio.Event().wait()
        yield SimpleNamespace(method="item/agentMessage/delta", payload={
            "itemId": "test-reply", "delta": f"Test reply: {self.prompt}",
        })
        yield SimpleNamespace(method="item/completed", payload={
            "item": {"id": "test-activity", "type": "commandExecution", "status": "completed"},
        })
        yield SimpleNamespace(method="turn/completed", payload={"turn": {"status": "completed"}})

    async def interrupt(self):
        return None


class TestCodex:
    async def connect(self, agent_id):
        return TestCodex()

    async def prepare_thread(self, model, thread_id=None):
        return SimpleNamespace(id=thread_id or "test-thread")

    async def start_turn(self, thread, model, prompt):
        return TestTurn(prompt)

    async def close(self):
        return None


def browser_settings():
    # Importing backend configuration evaluates its defaults; delay it until fake mode is chosen.
    from backend.app.config import Settings

    return Settings.from_env()


def test_app(settings):
    from backend.app import auth, repository
    from backend.app.main import create_app
    from backend.app.runner import Runner

    # Desktop/mobile suites share one loopback IP; production keeps its ten-attempt limit.
    auth.AUTH_ATTEMPTS_PER_MINUTE = 100
    # A whole desktop/mobile suite creates more assistants than a production instance.
    repository.MAX_AGENTS_PER_INSTANCE = 200
    app = create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with original(app):
            # Browser suites share a runtime; dedicated backend tests enforce the real 5/20 caps.
            app.state.repository.minute_start_limit = 1000
            app.state.repository.day_start_limit = 1000
            runner = Runner(
                app.state.repository, app.state.event_store, app.state.coordination,
                TestContainers(), TestNetwork(), TestCodex(),
                verified_models=frozenset({"gpt-6-luna"}),
            )
            app.state.runner = runner
            app.state.agent_service.runner = runner
            runner.capacity_admission = app.state.agent_service.make_room
            runner.start()
            try:
                yield
            finally:
                await runner.close()

    app.router.lifespan_context = lifespan
    return app


TEST_HOST_MARKER = Path("/etc/qpl-acceptance-host.json")
TEST_VOLUMES = ("qpl-platform-state", "qpl-agent-data")


def independent_test_session():
    # Only root provisioned dedicated hosts may opt in; never follow a marker symlink.
    try:
        descriptor = os.open(TEST_HOST_MARKER, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != 0
                or metadata.st_mode & 0o022
                or not 0 < metadata.st_size <= 4096
            ):
                raise ValueError("invalid marker metadata")
            content = json.loads(os.read(descriptor, 4097))
        finally:
            os.close(descriptor)
        if (
            not isinstance(content, dict)
            or set(content) != {"project", "purpose", "session_id"}
            or content["project"] != "hackathon-2-qpl"
            or content["purpose"] != "isolated-acceptance"
        ):
            raise ValueError("invalid marker identity")
        session = content["session_id"]
        if not isinstance(session, str) or str(UUID(session)) != session:
            raise ValueError("invalid session identity")
        return session
    except (OSError, ValueError, TypeError):
        raise RuntimeError("A valid root-owned independent test host marker is required.") from None


def test_docker(*arguments, timeout=30):
    # Pin the local socket; inherited remote contexts must not redirect test controls.
    environment = dict(os.environ)
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(name, None)
    try:
        return subprocess.run(
            ["docker", "--host", "unix:///var/run/docker.sock", *arguments],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("Independent test Docker verification is unavailable.") from None


def verify_test_volumes(session):
    try:
        volumes = json.loads(test_docker("volume", "inspect", *TEST_VOLUMES))
        if not isinstance(volumes, list) or len(volumes) != len(TEST_VOLUMES):
            raise ValueError("unexpected volume inventory")
        names = set()
        # Labels alone are insufficient: reject alternate storage drivers and path aliases.
        for volume in volumes:
            if not isinstance(volume, dict):
                raise ValueError("invalid volume metadata")
            name = volume.get("Name")
            labels = volume.get("Labels")
            if (
                name not in TEST_VOLUMES
                or name in names
                or volume.get("Driver") != "local"
                or volume.get("Scope") != "local"
                or volume.get("Options") not in (None, {})
                or volume.get("Mountpoint") != f"/var/lib/docker/volumes/{name}/_data"
                or not isinstance(labels, dict)
                or labels.get("qpl.purpose") != "isolated-acceptance"
                or labels.get("qpl.test-session") != session
            ):
                raise ValueError("unexpected volume identity")
            names.add(name)
    except (ValueError, TypeError):
        raise RuntimeError("Session-owned independent test volumes are required.") from None


def test_quota_request(operation):
    deadline = time.monotonic() + 10

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("quota request deadline exceeded")
        return seconds

    try:
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(remaining())
            connection.connect("/run/qpl-quota/helper.sock")
            connection.settimeout(remaining())
            connection.sendall(json.dumps({"operation": operation}).encode() + b"\n")
            response = bytearray()
            # Bound both blocking time and the helper response; reject truncated replies.
            while not response.endswith(b"\n") and len(response) <= 4096:
                connection.settimeout(remaining())
                chunk = connection.recv(4097 - len(response))
                if not chunk:
                    break
                response.extend(chunk)
        if len(response) > 4096 or not response.endswith(b"\n"):
            raise ValueError("invalid helper response")
        remaining()
        result = json.loads(response)
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise ValueError("quota verification refused")
        return result
    except (OSError, ValueError, TypeError):
        raise RuntimeError("Independent test quota verification is unavailable.") from None


def verify_test_quota():
    test_quota_request("verify")
    result = test_quota_request("verify_platform")
    quota = result.get("quota")
    fields = ("used_bytes", "hard_bytes", "used_inodes", "hard_inodes")
    if (
        result.get("ok") is not True
        or not isinstance(quota, dict)
        or any(type(quota.get(field)) is not int or quota[field] < 0 for field in fields)
        or quota["hard_bytes"] != 2 * 1024**3
        or quota["hard_inodes"] != 100000
        or quota["used_bytes"] >= quota["hard_bytes"] * 4 // 5
        or quota["used_inodes"] >= quota["hard_inodes"]
        or type(result.get("free_bytes")) is not int
        or result["free_bytes"] <= 2 * 1024**3
    ):
        raise RuntimeError("Healthy measured independent test quota headroom is required.")


def real_test_preflight():
    session = independent_test_session()
    try:
        info = json.loads(test_docker("info", "--format", "{{json .}}"))
        if (
            not isinstance(info, dict)
            or info.get("OSType") != "linux"
            or info.get("Driver") != "overlay2"
            or info.get("DockerRootDir") != "/var/lib/docker"
            or not re.match(r"^(?:2[6-9]|[3-9][0-9])\.", str(info.get("ServerVersion", "")))
            or not isinstance(info.get("SecurityOptions"), list)
            or any(
                "rootless" in str(option) or "userns" in str(option)
                for option in info["SecurityOptions"]
            )
        ):
            raise ValueError("unsupported Docker runtime")
    except (ValueError, TypeError):
        raise RuntimeError(
            "Independent tests require supported rootful Linux Docker/overlay2."
        ) from None
    verify_test_volumes(session)
    verify_test_quota()
    return session


TEST_AGENT_CLEANUP_CODE = r"""
import asyncio
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import UUID

# The trusted child gets no host configuration/credentials and only the local socket.
for key in ("DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "DOCKER_CONFIG"):
    os.environ.pop(key, None)
os.environ["DOCKER_HOST"] = "unix:///var/run/docker.sock"
from backend.app.config import Settings
from backend.app.containers import ContainerManager

settings = replace(
    Settings.from_env(), agent_image=sys.argv[1], auth_mode="auth_file",
    auth_file_source=Path(sys.argv[2]), platform_volume="qpl-platform-state",
    agent_volume="qpl-agent-data", platform_state_root=Path("/platform-state"),
    gateway_socket=Path("/platform-state/gateway/gateway.sock"),
)
names = subprocess.check_output([
    "docker", "--host", "unix:///var/run/docker.sock", "container", "ls", "--all",
    "--filter", "volume=qpl-platform-state", "--filter", "label=qpl.purpose=agent",
    "--filter", "label=qpl.platform=personal-assistant", "--format", "{{.Names}}",
], text=True, timeout=30).splitlines()

async def collect():
    for name in names:
        match = re.fullmatch(r"qpl-agent-([0-9a-f-]{36})-([0-9a-f-]{36})", name)
        if match is None:
            raise RuntimeError("Test agent name is invalid")
        instance, agent = match.groups()
        if (str(UUID(instance)) != instance or str(UUID(agent)) != agent
                or (sys.argv[3] != "-" and instance != sys.argv[3])):
            raise RuntimeError("Test agent instance is invalid")
        manager = ContainerManager(settings, instance)
        if manager.container_name(agent) != name:
            raise RuntimeError("Test agent identity is invalid")
        await manager.stop_agent(agent)
        await manager._docker("container", "rm", name)

asyncio.run(collect())
"""


class RealTestBackend:
    def __init__(
        self, session, platform_image, agent_image, auth_source, socket_gid, port, frontend_port
    ):
        if str(UUID(session)) != session:
            raise ValueError("Use a canonical test session UUID.")
        if not all(
            re.fullmatch(r"sha256:[0-9a-f]{64}", image) for image in (platform_image, agent_image)
        ):
            raise ValueError("Pin both test images to immutable IDs.")
        self.session = session
        self.name = f"qpl-real-backend-{session}"
        self.platform_image = platform_image
        self.agent_image = agent_image
        self.auth_source = Path(auth_source)
        if not self.auth_source.is_absolute() or "," in str(self.auth_source):
            raise ValueError("Use an absolute auth source without commas.")
        self.socket_gid = socket_gid
        self.port = port
        self.frontend_port = frontend_port
        self.instance_id = None

    def arguments(self):
        # Only this trusted backend gets control sockets; agent isolation is unchanged.
        arguments = [
            "container",
            "create",
            "--name",
            self.name,
            "--label",
            "qpl.platform=personal-assistant",
            "--label",
            "qpl.purpose=real-test-backend",
            "--label",
            f"qpl.test-session={self.session}",
            "--user",
            "1000:1000",
            "--group-add",
            str(self.socket_gid),
            "--network",
            "bridge",
            "--publish",
            f"127.0.0.1:{self.port}:8000",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            "2g",
            "--pids-limit",
            "256",
            "--mount",
            "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock",
            "--mount",
            "type=bind,source=/run/qpl-quota/helper.sock,target=/run/qpl-quota/helper.sock",
            "--mount",
            "type=volume,source=qpl-platform-state,target=/platform-state,volume-nocopy",
            "--mount",
            "type=volume,source=qpl-agent-data,target=/runtime,volume-nocopy",
            "--mount",
            f"type=bind,source={self.auth_source},target=/run/codex-auth.json,readonly",
            "--tmpfs",
            "/upload-staging:rw,noexec,nosuid,nodev,size=64m,mode=0700,uid=1000,gid=1000",
            "--env",
            "QPL_AUTH_MODE=auth_file",
            "--env",
            f"QPL_AUTH_FILE_SOURCE={self.auth_source}",
            "--env",
            "QPL_AUTH_FILE_VALIDATION=/run/codex-auth.json",
            "--env",
            f"QPL_AGENT_IMAGE_TAG={self.agent_image}",
            "--env",
            f"QPL_IMAGE_TAG={self.platform_image}",
            "--env",
            f"QPL_ALLOWED_ORIGIN=http://127.0.0.1:{self.frontend_port}",
        ]
        return [*arguments, self.platform_image]

    def exists(self):
        existing = test_docker(
            "container",
            "ls",
            "--all",
            "--filter",
            f"name=^{self.name}$",
            "--format",
            "{{.Names}}",
        ).strip()
        if existing not in {"", self.name}:
            raise RuntimeError("Independent backend identity conflict.")
        return bool(existing)

    def owned_metadata(self):
        # Do not inspect environment values: labels, immutable image and mounts are sufficient.
        template = (
            '{"labels":{{json .Config.Labels}},"image":{{json .Image}},'
            '"user":{{json .Config.User}},"running":{{json .State.Running}},'
            '"host":{{json .HostConfig}},"mounts":{{json .Mounts}}}'
        )
        try:
            metadata = json.loads(
                test_docker("container", "inspect", "--format", template, self.name)
            )
            labels = metadata.get("labels") if isinstance(metadata, dict) else None
            host = metadata.get("host") if isinstance(metadata, dict) else None
            if (
                not isinstance(labels, dict)
                or not isinstance(host, dict)
                or labels.get("qpl.platform") != "personal-assistant"
                or labels.get("qpl.purpose") != "real-test-backend"
                or labels.get("qpl.test-session") != self.session
                or metadata.get("image") != self.platform_image
                or metadata.get("user") != "1000:1000"
                or type(metadata.get("running")) is not bool
                or host.get("Privileged") is not False
                or host.get("NetworkMode") != "bridge"
                or host.get("CapAdd") not in (None, [])
                or host.get("CapDrop") != ["ALL"]
                or host.get("SecurityOpt") != ["no-new-privileges:true"]
                or host.get("Tmpfs") != {
                    "/upload-staging": (
                        "rw,noexec,nosuid,nodev,size=64m,mode=0700,uid=1000,gid=1000"
                    ),
                }
                or host.get("PortBindings")
                != {
                    "8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(self.port)}],
                }
                or not self.expected_mounts(metadata.get("mounts"))
            ):
                raise ValueError("unexpected backend metadata")
            return metadata
        except (ValueError, TypeError):
            raise RuntimeError("Independent backend identity conflict.") from None

    def expected_mounts(self, mounts):
        expected = {
            ("volume", "qpl-platform-state", "/platform-state", True),
            ("volume", "qpl-agent-data", "/runtime", True),
            ("bind", "/var/run/docker.sock", "/var/run/docker.sock", True),
            ("bind", "/run/qpl-quota/helper.sock", "/run/qpl-quota/helper.sock", True),
            ("bind", str(self.auth_source), "/run/codex-auth.json", False),
        }
        if not isinstance(mounts, list) or len(mounts) != len(expected):
            return False
        actual = set()
        # Effective volumes must be the fixed local roots, with no extra host exposure.
        for mount in mounts:
            if not isinstance(mount, dict) or type(mount.get("RW")) is not bool:
                return False
            kind = mount.get("Type")
            source = mount.get("Source")
            if kind == "volume":
                source = mount.get("Name")
                if mount.get("Source") != f"/var/lib/docker/volumes/{source}/_data":
                    return False
            if not all(
                isinstance(value, str) for value in (kind, source, mount.get("Destination"))
            ):
                return False
            actual.add((kind, source, mount["Destination"], mount["RW"]))
        return actual == expected

    def start(self):
        self.collect_cleanup_worker()
        if not self.exists():
            test_docker(*self.arguments())
        self.owned_metadata()
        test_docker("container", "start", self.name)

    def wait_ready(self, seconds=90):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not self.owned_metadata()["running"]:
                raise RuntimeError("Independent backend stopped before readiness.")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=1
                ) as response:
                    health = json.load(response)
                identity = health["instance_id"]
                storage = health["storage_protection"]
                if (
                    str(UUID(identity)) != identity
                    or health.get("agent_image") != self.agent_image
                    or not isinstance(storage, dict)
                    or storage.get("active") is not False
                    or storage.get("measurement_verified") is not True
                    or (self.instance_id is not None and self.instance_id != identity)
                ):
                    raise ValueError("unexpected runtime identity or storage state")
                self.instance_id = identity
                return health
            except (OSError, ValueError, KeyError, TypeError):
                time.sleep(0.25)
        raise RuntimeError("Independent backend readiness was not confirmed.")

    def stop(self):
        if not self.exists():
            self.stop_agents()
            return
        self.owned_metadata()
        try:
            test_docker("container", "stop", "--time", "40", self.name, timeout=45)
        except RuntimeError:
            if self.owned_metadata()["running"]:
                test_docker("container", "kill", self.name)
        if self.owned_metadata()["running"]:
            raise RuntimeError("Independent backend cleanup remains unconfirmed.")
        self.stop_agents()
        test_docker("container", "rm", self.name)

    def cleanup_arguments(self):
        # Reuse production validation in a clean trusted image, never host product settings.
        return [
            "run", "--name", f"qpl-real-cleanup-{self.session}",
            "--label", "qpl.purpose=real-test-cleanup",
            "--label", f"qpl.test-session={self.session}",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true", "--user", "1000:1000",
            "--group-add", str(self.socket_gid), "--memory", "256m", "--pids-limit", "64",
            "--mount", "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock",
            self.platform_image, "python", "-c", TEST_AGENT_CLEANUP_CODE,
            self.agent_image, str(self.auth_source), self.instance_id or "-",
        ]

    def stop_agents(self):
        # A failed readiness check still collects agents sharing the verified test volume.
        try:
            test_docker(*self.cleanup_arguments(), timeout=90)
        finally:
            self.collect_cleanup_worker()

    def collect_cleanup_worker(self):
        name = f"qpl-real-cleanup-{self.session}"
        existing = test_docker(
            "container", "ls", "--all", "--filter", f"name=^{name}$", "--format", "{{.Names}}",
        ).strip()
        if not existing:
            return
        labels = json.loads(test_docker(
            "container", "inspect", "--format", "{{json .Config.Labels}}", name,
        ))
        image = test_docker("container", "inspect", "--format", "{{.Image}}", name).strip()
        if (existing != name or not isinstance(labels, dict)
                or labels.get("qpl.purpose") != "real-test-cleanup"
                or labels.get("qpl.test-session") != self.session or image != self.platform_image):
            raise RuntimeError("Independent cleanup worker identity conflict.")
        test_docker("container", "stop", "--time", "5", name)
        running = test_docker(
            "container", "inspect", "--format", "{{json .State.Running}}", name,
        )
        if json.loads(running) is not False:
            raise RuntimeError("Independent cleanup worker termination is unconfirmed.")
        test_docker("container", "rm", name)


def configured_real_backend(session, port, frontend_port):
    if os.getenv("QPL_AUTH_MODE", "auth_file") != "auth_file":
        raise RuntimeError("Independent test startup currently requires auth-file mode.")
    source = Path(os.getenv("QPL_AUTH_FILE_SOURCE", ""))
    try:
        metadata = source.lstat()
        if (
            not source.is_absolute()
            or "," in str(source)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 1000
            or metadata.st_mode & 0o077
            or not 0 < metadata.st_size <= 1048576
            or not os.access(source, os.R_OK)
        ):
            raise ValueError("invalid auth source metadata")
    except (OSError, ValueError):
        raise RuntimeError(
            "Set an approved private UID1000 absolute QPL_AUTH_FILE_SOURCE."
        ) from None
    images = [
        test_docker(
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            os.getenv(name, fallback),
        ).strip()
        for name, fallback in (
            ("QPL_IMAGE_TAG", "qpl-platform:dev"),
            ("QPL_AGENT_IMAGE_TAG", "qpl-agent:dev"),
        )
    ]
    return RealTestBackend(
        session,
        *images,
        source,
        os.stat("/var/run/docker.sock").st_gid,
        port,
        frontend_port,
    )


def serve_real_backend(session, port, frontend_port):
    backend = configured_real_backend(session, port, frontend_port)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_args: stop.set())
    # One controller owns this dedicated runtime; externally requested restarts keep its volume.
    lock = Path(f"/tmp/qpl-real-{session}.lock")
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise RuntimeError("Independent controller lock identity conflict.")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            backend.start()
            backend.wait_ready()
            print(
                f"Real test backend: name={backend.name} port={port} pid={os.getpid()}", flush=True
            )
            while not stop.wait(1):
                pass
        finally:
            backend.stop()
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fake", "real"), required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=5173)
    args = parser.parse_args()
    if not all(1024 <= value <= 65535 for value in (args.port, args.frontend_port)):
        parser.error("Use a port between 1024 and 65535.")
    if args.mode == "real":
        if args.runtime is not None:
            parser.error("Real mode requires independent test volumes, not a product directory.")
        try:
            session = real_test_preflight()
        except RuntimeError as error:
            parser.error(str(error))
        if args.check_only:
            print("Independent test host preflight passed; no product or model was started.")
            return
        serve_real_backend(session, args.port, args.frontend_port)
        return
    if args.check_only:
        parser.error("--check-only is supported only for independent real test preflight.")
    settings = browser_settings()
    if args.mode == "fake":
        root = args.runtime or Path(tempfile.mkdtemp(prefix="qpl-browser-test-"))
        settings = replace(
            settings,
            runtime_root=root,
            database_path=root / "platform.sqlite",
            platform_state_root=root / "platform-state",
            agent_data_root=root / "agent-data",
            gateway_socket=root / "platform-state/gateway/gateway.sock",
            model_report_path=root / "platform-state/model-report/report/verified-models.json",
            allowed_origin=f"http://127.0.0.1:{args.frontend_port}",
            cookie_secure=False,
        )
        app = test_app(settings)
    print(f"Browser test service: mode={args.mode} port={args.port} pid={os.getpid()}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
