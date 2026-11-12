#!/usr/bin/env python3
"""Explicit real-model probe, isolated from the target service and its agents."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.codex_gateway import CodexGateway  # noqa: E402
from backend.app.config import Settings  # noqa: E402
from backend.app.containers import ContainerManager  # noqa: E402
from backend.app.coordination import RuntimeProcessLock  # noqa: E402
from backend.app.network_gateway import NetworkGateway, TokenAuthority  # noqa: E402
from backend.app.recovery import VerifiedModelReports  # noqa: E402
from backend.app.repository import AgentRecord  # noqa: E402
from backend.app.runner import Runner  # noqa: E402

DAY_MS = 86_400_000
PROMPT = "Reply with exactly READY and do not call tools."


class CleanupError(RuntimeError):
    pass


def canonical(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("Use a canonical UUID.")
    return value


def safe_read(path: Path, maximum: int = 65536) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_gid != os.getegid()
            or info.st_mode & 0o077
            or info.st_size > maximum
        ):
            raise ValueError("Probe metadata has unsafe ownership, mode or size.")
        data = os.read(descriptor, maximum + 1)
        if len(data) > maximum:
            raise ValueError("Probe metadata is too large.")
        return data
    finally:
        os.close(descriptor)


def atomic_write(path: Path, content: bytes) -> None:
    if len(content) > 65536:
        raise ValueError("Probe metadata is too large.")
    info = path.parent.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_gid != os.getegid()
        or info.st_mode & 0o077
    ):
        raise ValueError("Probe output directory is unsafe.")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class AttemptLedger:
    """Persist attempts per configuration; crashes and config round trips consume a window."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(safe_read(path))
        except FileNotFoundError:
            self.data = {"version": 2, "bindings": {}}
        if (
            not isinstance(self.data, dict)
            or self.data.get("version") != 2
            or not isinstance(self.data.get("bindings"), dict)
            or len(self.data["bindings"]) > 16
        ):
            raise ValueError("Probe attempt ledger is invalid.")
        for binding, attempts in self.data["bindings"].items():
            if (
                not isinstance(binding, str)
                or len(binding) > 128
                or not isinstance(attempts, dict)
                or len(attempts) > 2
            ):
                raise ValueError("Probe configuration ledger is invalid.")
            for entry in attempts.values():
                if (
                    not isinstance(entry, dict)
                    or type(entry.get("at")) is not int
                    or entry.get("status") not in {"started", "passed", "failed"}
                    or entry["at"] > self.now() + 300_000
                ):
                    raise ValueError("Probe attempt ledger entry is invalid.")

    @staticmethod
    def now() -> int:
        return time.time_ns() // 1_000_000

    def previous(self, binding: str, model: str) -> str | None:
        entry = self.data["bindings"].get(binding, {}).get(model)
        if entry is None or self.now() - entry["at"] > DAY_MS:
            return None
        return "passed" if entry["status"] == "passed" else "failed"

    def begin(self, binding: str, model: str) -> None:
        if self.previous(binding, model) is not None:
            raise ValueError("This configuration/model already consumed its attempt window.")
        for key, attempts in tuple(self.data["bindings"].items()):
            remaining = {
                name: entry
                for name, entry in attempts.items()
                if self.now() - entry["at"] <= DAY_MS
            }
            if remaining:
                self.data["bindings"][key] = remaining
            else:
                del self.data["bindings"][key]
        if binding not in self.data["bindings"]:
            if len(self.data["bindings"]) >= 16:
                raise ValueError("Probe configuration attempt capacity is exhausted.")
            self.data["bindings"][binding] = {}
        self.data["bindings"][binding][model] = {"at": self.now(), "status": "started"}
        self.save()

    def finish(self, binding: str, model: str, passed: bool) -> None:
        self.data["bindings"][binding][model]["status"] = "passed" if passed else "failed"
        self.save()

    def evidence_time(self, binding: str, model: str) -> int:
        return self.data["bindings"][binding][model]["at"]

    def save(self) -> None:
        atomic_write(self.path, json.dumps(self.data, sort_keys=True).encode())


async def ready_turn(session, model: str, *, deadline: float = 120) -> bool:
    async def attempt():
        thread = await session.prepare_thread(model)
        turn = await session.start_turn(thread, model, PROMPT)
        final = None
        completed = False
        async for event in turn.stream():
            payload = Runner._notification_payload(event)
            if event.method == "error":
                return False
            if event.method in {"item/started", "item/completed"}:
                item = payload.get("item", {})
                if not isinstance(item, dict) or item.get("type") not in {
                    "agentMessage",
                    "userMessage",
                    "reasoning",
                }:
                    return False
                if event.method == "item/completed" and item.get("type") == "agentMessage":
                    final = item.get("text")
            if event.method == "turn/completed":
                completed = payload.get("turn", {}).get("status") == "completed"
        return completed and isinstance(final, str) and final.strip() == "READY"

    try:
        return bool(await asyncio.wait_for(attempt(), timeout=deadline))
    except Exception:
        return False


class ProbeRuntime:
    def __init__(self, settings, instance_id, containers=None, gateway=None, codex=None):
        self.settings = settings
        self.instance_id = instance_id
        self.containers = containers or ContainerManager(settings, instance_id)
        self.gateway = gateway or NetworkGateway(settings, instance_id)
        self.codex = codex or CodexGateway(instance_id)

    async def probe(self, model: str) -> bool:
        agent_id = str(uuid4())
        agent = AgentRecord(
            agent_id, str(uuid4()), "Runtime probe", model, "ready", False, (), "unknown", 0
        )
        session = None
        token = self.gateway.authority.issue_probe_token(self.instance_id)
        passed = False
        try:

            async def execute():
                nonlocal session
                if len(await self.containers.running_agent_ids()) >= 2:
                    raise RuntimeError("Probe container capacity is exhausted.")
                # The smoke agent volume has its own bounded tmpfs; no target helper/data access.
                self.containers._ensure_agent_directories(agent_id)
                self.containers._write_agent_configuration(agent)
                self.containers.write_run_token(agent_id, token)
                await self.containers.start_agent(agent_id)
                version = await self.containers._docker(
                    "exec",
                    self.containers.container_name(agent_id),
                    "codex",
                    "--version",
                )
                if version.strip() != "codex-cli 0.166.0":
                    return False
                session = await self.codex.connect(agent_id, probe_only=True)
                return await ready_turn(session, model)

            passed = await asyncio.wait_for(execute(), timeout=120)
        except Exception:
            passed = False
        finally:
            writers = self.gateway.authority.revoke_hash(TokenAuthority.token_hash(token))
            operations = [self.containers.stop_agent(agent_id)]
            if session is not None:
                operations.append(session.close())
            operations.extend(writer.wait_closed() for writer in writers)
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*operations, return_exceptions=True),
                    timeout=35,
                )
                clean = not any(isinstance(result, BaseException) for result in results)
                self.containers.remove_run_token(agent_id)
            except Exception:
                clean = False
            if not clean:
                raise CleanupError("Probe cleanup was not confirmed.")
        return bool(passed)


async def publish_probe(runtime, store, key: bytes, ledger: AttemptLedger, destination: Path):
    binding = store._configuration_binding(key)
    candidates = {}
    for model in store.settings.models:
        if store._configuration_binding(key) != binding:
            raise ValueError("Probe configuration changed; no report published.")
        previous = ledger.previous(binding, model)
        if previous is None:
            ledger.begin(binding, model)
            try:
                passed = await runtime.probe(model)
            except BaseException:
                ledger.finish(binding, model, False)
                raise
            ledger.finish(binding, model, passed)
            previous = "passed" if passed else "failed"
        candidates[model] = previous
        if store._configuration_binding(key) != binding:
            raise ValueError("Probe configuration changed; no report published.")
    # Cached results retain their original evidence time; publication cannot extend validity.
    generated_at = min(ledger.evidence_time(binding, model) for model in candidates)
    atomic_write(destination, store.build_report(candidates, generated_at, key=key))
    return candidates


def docker(*arguments, timeout=30, environment=None):
    result = subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, timeout=timeout, env=environment
    )
    if result.returncode:
        raise RuntimeError("Docker operation failed; check the daemon and scoped runtime setup.")
    return result.stdout.strip()


def probe_arguments(settings: Settings, target: str, instance: str, image_id: str, socket_gid: int):
    canonical(target)
    canonical(instance)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("An immutable agent image ID is required.")
    state_volume = f"qpl-probe-state-{target}"
    agent_volume = f"qpl-probe-agent-{instance}"
    args = [
        "run",
        "--rm",
        "--name",
        f"qpl-probe-{instance}",
        "--label",
        "qpl.platform=personal-assistant",
        "--label",
        "qpl.purpose=probe",
        "--label",
        f"qpl.instance={instance}",
        "--user",
        "1000:1000",
        "--memory",
        "512m",
        "--cpus",
        "2",
        "--pids-limit",
        "256",
        "--no-healthcheck",
        "--group-add",
        str(socket_gid),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--mount",
        "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock",
        "--mount",
        f"type=volume,source={state_volume},target=/probe-state",
        "--mount",
        f"type=volume,source={agent_volume},target=/probe-agent-data",
        "--mount",
        f"type=volume,source={settings.platform_volume},target=/target-key,"
        f"volume-subpath=probe-share/{target}/key,readonly",
        "--mount",
        f"type=volume,source={settings.platform_volume},target=/target-report,"
        f"volume-subpath=probe-share/{target}/report",
        "--env",
        f"QPL_AUTH_MODE={settings.auth_mode}",
        "--env",
        "QPL_TOOLKIT_API_KEY",
        "--env",
        f"QPL_AGENT_IMAGE_TAG={image_id}",
        "--env",
        f"QPL_PLATFORM_VOLUME={state_volume}",
        "--env",
        f"QPL_AGENT_VOLUME={agent_volume}",
        "--env",
        "QPL_PLATFORM_STATE_ROOT=/probe-state",
        "--env",
        "QPL_AGENT_DATA_ROOT=/probe-agent-data",
        "--env",
        "QPL_GATEWAY_SOCKET=/probe-state/gateway/gateway.sock",
        "--env",
        f"QPL_PROBE_INSTANCE={instance}",
    ]
    if settings.auth_mode == "auth_file":
        source = settings.auth_file_source
        if source is None or not source.is_absolute() or "," in str(source):
            raise ValueError("Configure an absolute Linux host auth-file source.")
        args += [
            "--mount",
            f"type=bind,source={source},target=/run/codex-auth.json,readonly",
            "--env",
            f"QPL_AUTH_FILE_SOURCE={source}",
            "--env",
            "QPL_AUTH_FILE_VALIDATION=/run/codex-auth.json",
        ]
    return args + [
        settings.platform_image,
        "python",
        "scripts/check-runtime.py",
        "--inside-probe",
        "--target-instance",
        target,
    ]


async def inside(target: str):
    if os.geteuid() != 1000 or os.getegid() != 1000:
        raise RuntimeError("Probe must run in its trusted UID/GID 1000 container.")
    instance = canonical(os.environ["QPL_PROBE_INSTANCE"])
    settings = Settings.from_env()
    if (
        settings.platform_volume != f"qpl-probe-state-{target}"
        or settings.agent_volume != f"qpl-probe-agent-{instance}"
        or settings.platform_state_root != Path("/probe-state")
        or settings.agent_data_root != Path("/probe-agent-data")
    ):
        raise ValueError("Probe smoke runtime configuration is invalid.")
    if importlib.metadata.version("openai-codex") != "0.163.0":
        raise ValueError("The probe requires the pinned Python SDK.")
    store = VerifiedModelReports(settings)
    key = VerifiedModelReports._read_key(Path("/target-key/hmac.key"))
    runtime = ProbeRuntime(settings, instance)
    # Lock only the independent smoke metadata, never the target backend lock/database.
    lock = RuntimeProcessLock(Path("/probe-state/probe.lock"))
    lock.acquire()
    started = False
    try:
        await runtime.gateway.start()
        started = True
        task = asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        results = await asyncio.wait_for(
            publish_probe(
                runtime,
                store,
                key,
                AttemptLedger(Path("/probe-state/attempts.json")),
                Path("/target-report/verified-models.json"),
            ),
            timeout=600,
        )
        for model, status in results.items():
            print(f"{model}: {status}", flush=True)
    finally:
        try:
            if started:
                await runtime.gateway.close()
        finally:
            lock.release()


def host(settings: Settings, target: str):
    version = docker("version", "--format", "{{.Server.Version}}")
    if int(version.split(".")[0]) < 26:
        raise RuntimeError("Docker Engine 26 or newer is required.")
    info = json.loads(docker("info", "--format", "{{json .}}"))
    if (
        info.get("OSType") != "linux"
        or info.get("Driver") != "overlay2"
        or any(
            "rootless" in str(item) or "userns" in str(item)
            for item in info.get("SecurityOptions", [])
        )
    ):
        raise RuntimeError("Use the supported Linux Docker runtime.")
    image_id = docker("image", "inspect", "--format", "{{.Id}}", settings.agent_image)
    docker("image", "inspect", settings.platform_image)
    instance = str(uuid4())
    agent_volume = f"qpl-probe-agent-{instance}"
    state_volume = f"qpl-probe-state-{target}"
    # A persistent, bounded-content ledger survives probe/container/host restarts.
    docker(
        "volume",
        "create",
        "--label",
        "qpl.purpose=probe-state",
        "--label",
        f"qpl.target={target}",
        state_volume,
    )
    state_info = json.loads(docker("volume", "inspect", state_volume))[0]
    if (
        state_info.get("Driver") != "local"
        or state_info.get("Options") not in (None, {})
        or (state_info.get("Labels") or {}).get("qpl.purpose") != "probe-state"
        or (state_info.get("Labels") or {}).get("qpl.target") != target
    ):
        raise ValueError("Probe state volume identity conflict.")
    stale = docker(
        "container",
        "ls",
        "--filter",
        f"volume={state_volume}",
        "--filter",
        "label=qpl.purpose=agent",
        "--format",
        "{{.Names}}",
    )
    if stale:
        raise CleanupError("An earlier probe still has running agents; repair it before probing.")
    docker(
        "volume",
        "create",
        "--label",
        f"qpl.instance={instance}",
        "--driver",
        "local",
        "--opt",
        "type=tmpfs",
        "--opt",
        "device=tmpfs",
        "--opt",
        "o=size=2g,nr_inodes=100000,uid=1000,gid=1000,mode=0700",
        agent_volume,
    )
    agent_info = json.loads(docker("volume", "inspect", agent_volume))[0]
    if (
        agent_info.get("Driver") != "local"
        or (agent_info.get("Labels") or {}).get("qpl.instance") != instance
        or agent_info.get("Options")
        != {
            "type": "tmpfs",
            "device": "tmpfs",
            "o": "size=2g,nr_inodes=100000,uid=1000,gid=1000,mode=0700",
        }
    ):
        raise ValueError("Probe agent volume identity or bounds are invalid.")
    try:
        docker(
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0:0",
            "--mount",
            f"type=volume,source={state_volume},target=/probe-state",
            settings.platform_image,
            "python",
            "-c",
            "import os; os.chown('/probe-state',1000,1000); os.chmod('/probe-state',0o700)",
        )
        environment = dict(os.environ)
        if settings.auth_mode == "toolkit" and not environment.get("QPL_TOOLKIT_API_KEY"):
            environment["QPL_TOOLKIT_API_KEY"] = environment.get("UMGPT_TOOLKIT_API_KEY", "")
        args = probe_arguments(
            settings, target, instance, image_id, os.stat("/var/run/docker.sock").st_gid
        )
        output = docker(*args, timeout=660, environment=environment)
        for line in output.splitlines():
            if line in {
                f"{model}: {status}" for model in settings.models for status in ("passed", "failed")
            }:
                print(line)
    finally:
        # Collect only this smoke instance; exact labels and expected mount paths are validated.
        name = f"qpl-probe-{instance}"
        existing = docker(
            "container", "ls", "--all", "--filter", f"name=^{name}$", "--format", "{{.Names}}"
        )
        if existing:
            labels = json.loads(
                docker("container", "inspect", "--format", "{{json .Config.Labels}}", name)
            )
            if (
                existing != name
                or labels.get("qpl.instance") != instance
                or labels.get("qpl.purpose") != "probe"
                or labels.get("qpl.platform") != "personal-assistant"
            ):
                raise CleanupError("Probe container identity conflict.")
            docker("container", "stop", "--time", "40", name, timeout=45)
            remaining = docker(
                "container", "ls", "--all", "--filter", f"name=^{name}$", "--format", "{{.Names}}"
            )
            if remaining:
                state = json.loads(
                    docker("container", "inspect", "--format", "{{json .State}}", name)
                )
                if state.get("Running") is not False:
                    raise CleanupError("The probe container is not confirmed stopped.")
        cleanup_settings = Settings.from_env()
        from dataclasses import replace

        cleanup_settings = replace(
            cleanup_settings,
            agent_image=image_id,
            platform_volume=state_volume,
            agent_volume=agent_volume,
            platform_state_root=Path("/probe-state"),
            gateway_socket=Path("/probe-state/gateway/gateway.sock"),
        )
        manager = ContainerManager(cleanup_settings, instance)

        async def cleanup():
            names = await manager._docker(
                "container",
                "ls",
                "--all",
                "--filter",
                f"label=qpl.instance={instance}",
                "--filter",
                "label=qpl.platform=personal-assistant",
                "--filter",
                "label=qpl.purpose=agent",
                "--format",
                "{{.Names}}",
            )
            prefix = f"qpl-agent-{instance}-"
            for name in names.splitlines():
                agent_id = canonical(name.removeprefix(prefix))
                if manager.container_name(agent_id) != name:
                    raise CleanupError("Unexpected probe container identity.")
                await manager.stop_agent(agent_id)
                await manager._docker("container", "rm", name)

        asyncio.run(cleanup())
        docker("volume", "rm", agent_volume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-instance", required=True, type=canonical)
    parser.add_argument("--inside-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside_probe:
        asyncio.run(inside(args.target_instance))
    else:
        host(Settings.from_env(), args.target_instance)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt, asyncio.CancelledError):
        print(
            "Probe failed; no new availability is confirmed. "
            "Check Docker, scoped mounts, authentication and cleanup.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
