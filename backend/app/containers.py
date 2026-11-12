from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .agents import AgentPreparationError
from .config import Settings
from .repository import AgentRecord

SUPPORTED_AGENT_VOLUME = "qpl-agent-data"
SUPPORTED_PLATFORM_VOLUME = "qpl-platform-state"


class ContainerManager:
    def __init__(self, settings: Settings, instance_id: str) -> None:
        self.settings = settings
        self.instance_id = str(UUID(instance_id))

    async def resolve_agent_image(self) -> str:
        image_id = (
            await self._docker(
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                self.settings.agent_image,
            )
        ).strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise AgentPreparationError(
                "agent_image_unavailable", "The agent image is unavailable."
            )
        return image_id

    async def verify_runtime(self) -> None:
        if (
            self.settings.agent_volume != SUPPORTED_AGENT_VOLUME
            or self.settings.platform_volume != SUPPORTED_PLATFORM_VOLUME
        ):
            raise AgentPreparationError(
                "unsupported_runtime_volumes",
                "The quota helper requires the configured platform volume names.",
            )
        version = (await self._docker("version", "--format", "{{.Server.Version}}")).strip()
        match = re.match(r"^(\d+)\.", version)
        if match is None or int(match.group(1)) < 26:
            raise AgentPreparationError(
                "unsupported_docker_engine",
                "Docker Engine 26 or newer is required.",
            )
        info_json = await self._docker("info", "--format", "{{json .}}")
        try:
            info = json.loads(info_json)
        except json.JSONDecodeError as error:
            raise AgentPreparationError(
                "docker_runtime_invalid",
                "Docker returned invalid runtime information.",
            ) from error
        if (
            not isinstance(info, dict)
            or info.get("Driver") != "overlay2"
            or info.get("OSType") != "linux"
        ):
            raise AgentPreparationError(
                "unsupported_docker_storage",
                "The supported runtime requires Linux Docker with the overlay2 storage driver.",
            )
        security_options = info.get("SecurityOptions", [])
        if not isinstance(security_options, list) or any(
            "rootless" in str(option).lower() or "userns" in str(option).lower()
            for option in security_options
        ):
            raise AgentPreparationError(
                "unsupported_docker_security",
                "The supported runtime requires rootful Docker without user namespaces.",
            )
        await self._request_quota_helper({"operation": "verify"})
        await self._request_quota_helper({"operation": "ensure_platform"})

    def create_arguments(self, agent_id: str) -> list[str]:
        agent_id = str(UUID(agent_id))
        name = self.container_name(agent_id)
        try:
            gateway_subpath = self.settings.gateway_socket.parent.relative_to(
                self.settings.platform_state_root
            )
        except ValueError as error:
            raise AgentPreparationError(
                "runtime_configuration_invalid",
                "The platform gateway path is outside the platform-state volume.",
            ) from error
        if not gateway_subpath.parts or ".." in gateway_subpath.parts:
            raise AgentPreparationError(
                "runtime_configuration_invalid",
                "The platform gateway path is outside the platform-state volume.",
            )

        args = [
            "container",
            "create",
            "--name",
            name,
            "--label",
            "qpl.platform=personal-assistant",
            "--label",
            f"qpl.instance={self.instance_id}",
            "--label",
            f"qpl.agent={agent_id}",
            "--label",
            "qpl.purpose=agent",
            "--network",
            "none",
            "--read-only",
            "--user",
            "1000:1000",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--cpus",
            "1",
            "--memory",
            "1g",
            "--pids-limit",
            "128",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            (
                f"type=volume,source={self.settings.agent_volume},target=/workspace,"
                f"volume-subpath={agent_id}/workspace,volume-nocopy"
            ),
            "--mount",
            (
                f"type=volume,source={self.settings.agent_volume},target=/codex,"
                f"volume-subpath={agent_id}/codex,volume-nocopy"
            ),
            "--mount",
            (
                f"type=volume,source={self.settings.agent_volume},target=/run/credentials,"
                f"volume-subpath={agent_id}/credentials,volume-nocopy,readonly"
            ),
            "--mount",
            (
                f"type=volume,source={self.settings.platform_volume},"
                f"target=/run/platform-gateway,volume-subpath={gateway_subpath},volume-nocopy,readonly"
            ),
        ]
        if self.settings.auth_mode == "auth_file":
            source = self.settings.auth_file_source
            validation = self.settings.auth_file_validation_path
            if source is None or validation is None:
                raise AgentPreparationError(
                    "authentication_unavailable",
                    "Configure both the host auth file and its backend validation mount.",
                )
            if not source.is_absolute() or "," in str(source):
                raise AgentPreparationError(
                    "authentication_configuration_invalid",
                    "The host auth file path must be absolute and cannot contain commas.",
                )
            self._validate_auth_file(validation)
            args.extend(
                [
                    "--mount",
                    f"type=bind,source={source},target=/codex/auth.json,readonly",
                ]
            )
        args.append(self.settings.agent_image)
        return args

    async def prepare_agent(self, agent: AgentRecord) -> None:
        agent_id = str(UUID(agent.id))
        await self._request_quota_helper({"operation": "ensure_agent", "agent_id": agent_id})
        self._ensure_agent_directories(agent_id)
        self._write_agent_configuration(agent)
        await self._ensure_container(agent_id)

    def container_name(self, agent_id: str) -> str:
        canonical_agent_id = str(UUID(agent_id))
        if canonical_agent_id != agent_id:
            raise ValueError("agent_id must be a canonical UUID")
        return f"qpl-agent-{self.instance_id}-{canonical_agent_id}"

    def write_run_token(self, agent_id: str, token: str) -> None:
        if not token or "\n" in token or "\r" in token:
            raise ValueError("run token must be a non-empty single-line value")
        self._ensure_agent_directories(agent_id)
        path = self.settings.agent_data_root / str(UUID(agent_id)) / "codex" / ".platform-run-token"
        self._atomic_write(path, token.encode("ascii"))

    def remove_run_token(self, agent_id: str) -> None:
        path = self.settings.agent_data_root / str(UUID(agent_id)) / "codex" / ".platform-run-token"
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise AgentPreparationError(
                "agent_storage_unavailable",
                "The assistant run-token path is not a regular file.",
            )
        path.unlink()

    async def start_agent(self, agent_id: str) -> None:
        await self._ensure_container(agent_id)
        await self._docker("container", "start", self.container_name(agent_id))

    async def stop_agent(self, agent_id: str) -> None:
        name = self.container_name(agent_id)
        if not await self._ensure_container(agent_id, create_if_missing=False):
            return
        try:
            await self._docker("container", "stop", "--time", "5", name, timeout_seconds=5)
        except AgentPreparationError:
            pass
        # A successful CLI exit does not prove that the container stopped.
        if await self._confirmed_stopped(name):
            return
        try:
            await self._docker("container", "kill", name, timeout_seconds=10)
        except AgentPreparationError:
            pass
        if not await self._confirmed_stopped(name):
            raise AgentPreparationError(
                "container_stop_unconfirmed",
                "The assistant container could not be confirmed stopped.",
            )

    async def _confirmed_stopped(self, name: str) -> bool:
        try:
            state = json.loads(
                await self._docker(
                    "container",
                    "inspect",
                    "--format",
                    "{{json .State}}",
                    name,
                    timeout_seconds=5,
                )
            )
        except (AgentPreparationError, json.JSONDecodeError):
            return False
        return (
            isinstance(state, dict)
            and state.get("Running") is False
            and state.get("Restarting") is False
            and state.get("Status") in {"created", "exited", "dead"}
        )

    async def running_agent_ids(self) -> list[str]:
        output = await self._docker(
            "container",
            "ls",
            "--filter",
            "label=qpl.platform=personal-assistant",
            "--filter",
            f"label=qpl.instance={self.instance_id}",
            "--filter",
            "label=qpl.purpose=agent",
            "--format",
            "{{.Names}}",
        )
        agent_ids = []
        prefix = f"qpl-agent-{self.instance_id}-"
        for name in output.splitlines():
            agent_id = name.removeprefix(prefix)
            try:
                valid = name.startswith(prefix) and self.container_name(agent_id) == name
            except ValueError:
                valid = False
            if not valid:
                raise AgentPreparationError(
                    "container_identity_conflict", "A runtime container has an invalid identity."
                )
            agent_ids.append(agent_id)
        return sorted(set(agent_ids))

    def _ensure_agent_directories(self, agent_id: str) -> None:
        root = self.settings.agent_data_root
        try:
            root_info = root.lstat()
        except FileNotFoundError as error:
            raise AgentPreparationError(
                "agent_storage_unavailable", "The agent-data volume is not mounted."
            ) from error
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise AgentPreparationError(
                "agent_storage_unavailable", "The agent-data volume is not a safe directory."
            )
        if root_info.st_uid != os.geteuid() or root_info.st_gid != os.getegid():
            raise AgentPreparationError(
                "agent_storage_unavailable",
                "The agent-data volume is not owned by the backend runtime identity.",
            )
        agent_root = root / agent_id
        try:
            agent_root.mkdir(mode=0o700, exist_ok=True)
            info = agent_root.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_gid != os.getegid()
                or info.st_mode & 0o077
            ):
                raise AgentPreparationError(
                    "agent_storage_unavailable",
                    "The assistant storage path is not a safe directory.",
                )
            for name in ("workspace", "codex", "credentials"):
                child = agent_root / name
                child.mkdir(mode=0o700, exist_ok=True)
                child_info = child.lstat()
                if (
                    not stat.S_ISDIR(child_info.st_mode)
                    or stat.S_ISLNK(child_info.st_mode)
                    or child_info.st_uid != os.geteuid()
                    or child_info.st_gid != os.getegid()
                    or child_info.st_mode & 0o077
                ):
                    raise AgentPreparationError(
                        "agent_storage_unavailable",
                        "An assistant storage path is not a safe directory.",
                    )
        except OSError as error:
            raise AgentPreparationError(
                "agent_storage_unavailable",
                "The assistant storage directories could not be prepared.",
            ) from error

    def _write_agent_configuration(self, agent: AgentRecord, *, preserve_credentials=False) -> None:
        if agent.model not in self.settings.models:
            raise AgentPreparationError(
                "model_unavailable", "The assistant model is not supported by this runtime."
            )
        agent_root = self.settings.agent_data_root / str(UUID(agent.id))
        lines = [f'model = "{agent.model}"']
        if self.settings.auth_mode == "toolkit":
            if not preserve_credentials:
                key = self.settings.toolkit_api_key
                if not key:
                    raise AgentPreparationError(
                        "authentication_unavailable",
                        "The Toolkit API key is not configured.",
                    )
                self._atomic_write(
                    agent_root / "credentials" / "UMGPT_TOOLKIT_API_KEY",
                    key.encode("utf-8"),
                )
            lines.extend(
                [
                    'model_provider = "toolkit"',
                    "",
                    "[model_providers.toolkit]",
                    'name = "UM GPT Toolkit"',
                    'base_url = "https://api.toolkit.umgpt.umich.edu/v1"',
                    'env_key = "UMGPT_TOOLKIT_API_KEY"',
                ]
            )
        lines.extend(
            [
                "",
                "[mcp_servers.platform]",
                'url = "http://127.0.0.1:8765/mcp"',
                'bearer_token_env_var = "PLATFORM_TOKEN"',
                "",
            ]
        )
        self._atomic_write(
            agent_root / "codex" / "config.toml",
            "\n".join(lines).encode("utf-8"),
        )

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise AgentPreparationError(
                "agent_storage_unavailable",
                "Assistant configuration could not be written safely.",
            ) from error

    @staticmethod
    def _validate_auth_file(path: Path) -> None:
        try:
            info = path.lstat()
            readable = os.access(path, os.R_OK)
        except OSError as error:
            raise AgentPreparationError(
                "authentication_unavailable",
                "The configured Codex auth file is unavailable to the backend.",
            ) from error
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_size > 1_048_576
            or not readable
        ):
            raise AgentPreparationError(
                "authentication_unavailable",
                "The configured Codex auth file must be a readable regular file under 1 MiB.",
            )

    def reset_model_context(self, agent, deadline, cancelled, permission):
        root = None
        codex = None
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        permission()

        def clear(directory, depth=0):
            if depth > 128:
                raise AgentPreparationError(
                    "context_reset_failed", "Model history is nested too deeply."
                )
            with os.scandir(directory) as entries:
                for entry in entries:
                    permission()
                    if cancelled.is_set() or time.monotonic() >= deadline:
                        raise AgentPreparationError(
                            "context_reset_timeout", "Model history reset was interrupted."
                        )
                    if entry.is_dir(follow_symlinks=False):
                        child = os.open(entry.name, flags, dir_fd=directory)
                        try:
                            clear(child, depth + 1)
                        finally:
                            os.close(child)
                        os.rmdir(entry.name, dir_fd=directory)
                    else:
                        os.unlink(entry.name, dir_fd=directory)
            os.fsync(directory)

        try:
            root = os.open(self.settings.agent_data_root, flags)
            child = os.open(str(UUID(agent.id)), flags, dir_fd=root)
            os.close(root)
            root = child
            codex = os.open("codex", flags, dir_fd=root)
            clear(codex)
            permission()
            self._write_agent_configuration(agent, preserve_credentials=True)
            os.fsync(codex)
        finally:
            if codex is not None:
                os.close(codex)
            if root is not None:
                os.close(root)

    async def platform_storage_usage(self) -> dict[str, Any]:
        return await self._request_quota_helper({"operation": "verify_platform"})

    async def _request_quota_helper(self, payload: dict[str, str]) -> dict[str, Any]:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.settings.quota_helper_socket, limit=4097),
                timeout=5,
            )
        except (OSError, TimeoutError) as error:
            raise AgentPreparationError(
                "quota_helper_unavailable",
                "The required filesystem quota helper is unavailable.",
            ) from error

        try:
            writer.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
            await asyncio.wait_for(writer.drain(), timeout=5)
            response = await asyncio.wait_for(reader.readline(), timeout=5)
            if not response or len(response) > 4096 or not response.endswith(b"\n"):
                raise AgentPreparationError(
                    "quota_verification_failed",
                    "The filesystem quota helper returned an invalid response.",
                )
            result: Any = json.loads(response)
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise AgentPreparationError(
                    "quota_verification_failed",
                    "The filesystem quota helper could not verify the assistant quota.",
                )
            return result
        except (TimeoutError, OSError, json.JSONDecodeError) as error:
            raise AgentPreparationError(
                "quota_verification_failed",
                "The filesystem quota helper request did not complete successfully.",
            ) from error
        finally:
            writer.close()
            await writer.wait_closed()

    async def _ensure_container(self, agent_id: str, *, create_if_missing: bool = True) -> bool:
        name = self.container_name(agent_id)
        output = await self._docker(
            "container",
            "ls",
            "--all",
            "--filter",
            f"name=^{name}$",
            "--format",
            "{{.Names}}",
        )
        if output.strip():
            if output.strip() != name:
                raise AgentPreparationError(
                    "container_identity_conflict",
                    "A container name conflicts with this assistant.",
                )
            labels_json = await self._docker(
                "container",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                name,
            )
            try:
                labels = json.loads(labels_json)
            except json.JSONDecodeError as error:
                raise AgentPreparationError(
                    "container_identity_conflict",
                    "The existing assistant container has invalid metadata.",
                ) from error
            expected = {
                "qpl.platform": "personal-assistant",
                "qpl.instance": self.instance_id,
                "qpl.purpose": "agent",
                "qpl.agent": agent_id,
            }
            if not isinstance(labels, dict) or any(
                labels.get(key) != value for key, value in expected.items()
            ):
                raise AgentPreparationError(
                    "container_identity_conflict",
                    "The existing assistant container does not match this runtime.",
                )
            mounts_json = await self._docker(
                "container",
                "inspect",
                "--format",
                "{{json .Mounts}}",
                name,
            )
            host_config_json = await self._docker(
                "container",
                "inspect",
                "--format",
                "{{json .HostConfig}}",
                name,
            )
            config_json = await self._docker(
                "container",
                "inspect",
                "--format",
                "{{json .Config}}",
                name,
            )
            try:
                mounts = json.loads(mounts_json)
                host_config = json.loads(host_config_json)
                container_config = json.loads(config_json)
            except json.JSONDecodeError as error:
                raise AgentPreparationError(
                    "container_identity_conflict",
                    "The existing assistant container has invalid runtime metadata.",
                ) from error
            if (
                not isinstance(host_config, dict)
                or not self._has_expected_mounts(agent_id, mounts, host_config.get("Mounts"))
                or host_config.get("NetworkMode") != "none"
                or host_config.get("ReadonlyRootfs") is not True
                or host_config.get("Privileged") is not False
                or host_config.get("PortBindings") not in (None, {})
                or host_config.get("CapAdd") not in (None, [])
                or host_config.get("CapDrop") != ["ALL"]
                or host_config.get("SecurityOpt") != ["no-new-privileges:true"]
                or not isinstance(container_config, dict)
                or container_config.get("User") != "1000:1000"
                or container_config.get("Image") != self.settings.agent_image
            ):
                raise AgentPreparationError(
                    "container_identity_conflict",
                    "The existing assistant container violates its isolation settings.",
                )
            return True
        if not create_if_missing:
            return False
        container_id = await self._docker(*self.create_arguments(agent_id))
        if not container_id.strip():
            raise AgentPreparationError(
                "docker_operation_failed",
                "Docker did not return an assistant container identity.",
            )
        return True

    async def _docker(self, *arguments: str, timeout_seconds: int = 30) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise AgentPreparationError(
                "docker_unavailable", "The Docker CLI is unavailable to the backend."
            ) from error

        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise AgentPreparationError(
                "docker_timeout", "The Docker operation exceeded its time limit."
            ) from error
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        if process.returncode != 0:
            raise AgentPreparationError(
                "docker_operation_failed",
                "The Docker operation could not be completed.",
            )
        return stdout.decode("utf-8", errors="replace")

    def _has_expected_mounts(self, agent_id: str, mounts: Any, configured: Any) -> bool:
        if not isinstance(mounts, list) or not isinstance(configured, list):
            return False
        expected = {
            (self.settings.agent_volume, f"{agent_id}/workspace", "/workspace", True),
            (self.settings.agent_volume, f"{agent_id}/codex", "/codex", True),
            (
                self.settings.agent_volume,
                f"{agent_id}/credentials",
                "/run/credentials",
                False,
            ),
            (
                self.settings.platform_volume,
                self.settings.gateway_socket.parent.relative_to(
                    self.settings.platform_state_root
                ).as_posix(),
                "/run/platform-gateway",
                False,
            ),
        }
        if self.settings.auth_mode == "auth_file":
            if self.settings.auth_file_source is None:
                return False
            expected.add(
                (
                    str(self.settings.auth_file_source),
                    "/codex/auth.json",
                    "/codex/auth.json",
                    False,
                )
            )
        if len(mounts) != len(expected) or len(configured) != len(expected):
            return False
        # Docker reports volume roots in .Mounts; requested subpaths live in HostConfig.
        requested = set()
        by_target = {}
        for mount in configured:
            if not isinstance(mount, dict):
                return False
            kind = mount.get("Type")
            source = mount.get("Source")
            target = mount.get("Target")
            readonly = mount.get("ReadOnly", False)
            if (not isinstance(source, str) or not isinstance(target, str)
                    or not isinstance(readonly, bool) or target in by_target):
                return False
            if kind == "volume":
                options = mount.get("VolumeOptions")
                subpath = options.get("Subpath") if isinstance(options, dict) else None
                if not isinstance(subpath, str) or options.get("NoCopy") is not True:
                    return False
            elif kind == "bind":
                subpath = "/codex/auth.json"
            else:
                return False
            requested.add((source, subpath, target, not readonly))
            by_target[target] = (kind, source, subpath, not readonly)
        if requested != expected:
            return False
        # Cross-check effective mounts as well; neither inspection view suffices alone.
        actual: set[tuple[str, str, str, bool]] = set()
        for mount in mounts:
            if not isinstance(mount, dict):
                return False
            mount_type = mount.get("Type")
            name_or_source = mount.get("Source")
            destination = mount.get("Destination")
            read_write = mount.get("RW")
            if (
                not isinstance(mount_type, str)
                or not isinstance(name_or_source, str)
                or not isinstance(destination, str)
                or not isinstance(read_write, bool)
            ):
                return False
            subpath = self._mount_subpath(name_or_source, mount_type)
            if subpath is None:
                return False
            request = by_target.get(destination)
            if request is None or request[0] != mount_type:
                return False
            if mount_type == "volume":
                if mount.get("Name") != subpath[0] or subpath[1] not in {".", request[2]}:
                    return False
            elif mount_type != "bind" or subpath[1] != request[2]:
                return False
            actual.add((subpath[0], request[2], destination, read_write))
        return actual == expected

    def _mount_subpath(self, name_or_source: str, mount_type: str) -> tuple[str, str] | None:
        if mount_type == "bind":
            return name_or_source, "/codex/auth.json"
        parts = Path(name_or_source).parts
        try:
            volume_marker = parts.index("volumes")
            volume_name = parts[volume_marker + 1]
            if parts[volume_marker + 2] != "_data":
                return None
            subpath = Path(*parts[volume_marker + 3 :]).as_posix()
        except (ValueError, IndexError):
            return None
        if volume_name not in {self.settings.agent_volume, self.settings.platform_volume}:
            return None
        return volume_name, subpath
