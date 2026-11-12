#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import stat
import struct
import subprocess
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger("qpl.quota-helper")
SOCKET_PATH = Path("/run/qpl-quota/helper.sock")
VOLUME_ROOT = Path("/var/lib/docker/volumes")
AGENT_VOLUME = "qpl-agent-data"
PLATFORM_VOLUME = "qpl-platform-state"
STATE_PATH = Path("/var/lib/qpl-quota/project-map.json")
PROJECTS_PATH = Path("/var/lib/qpl-quota/projects")
PROJID_PATH = Path("/var/lib/qpl-quota/projid")
SYSTEM_PROJECTS_PATH = Path("/etc/projects")
SYSTEM_PROJID_PATH = Path("/etc/projid")
XFS_QUOTA = "/usr/sbin/xfs_quota"
AGENT_UID = 1000
AGENT_GID = 1000
PLATFORM_PROJECT_ID = 10_000
FIRST_AGENT_PROJECT_ID = 10_001
PROJECT_INHERIT = 0x00000200
FS_IOC_FSGETXATTR = 0x801C581F
QUOTA_BYTES = 2 * 1024**3
QUOTA_INODES = 100_000
MAX_REQUEST_BYTES = 4096
_operation_lock = asyncio.Lock()


class HelperError(Exception):
    pass


def _volume_root(volume_name: str) -> Path:
    if volume_name not in {AGENT_VOLUME, PLATFORM_VOLUME}:
        raise HelperError("unsupported named volume")
    path = VOLUME_ROOT / volume_name / "_data"
    try:
        info = path.lstat()
    except OSError as error:
        raise HelperError("required Docker volume is unavailable") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise HelperError("Docker volume root is not a real directory")
    return path


def _agent_path(agent_id: str, *, prepare: bool = True) -> Path:
    try:
        parsed = UUID(agent_id)
    except (ValueError, TypeError, AttributeError) as error:
        raise HelperError("agent_id must be a canonical UUID") from error
    if str(parsed) != agent_id:
        raise HelperError("agent_id must be a canonical UUID")
    root = _volume_root(AGENT_VOLUME)
    path = root / agent_id
    try:
        if prepare:
            path.mkdir(mode=0o700, exist_ok=True)
        info = path.lstat()
    except OSError as error:
        raise HelperError("assistant storage directory could not be prepared") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise HelperError("assistant storage path is not a real directory")
    try:
        if prepare:
            os.chown(path, AGENT_UID, AGENT_GID)
            os.chmod(path, 0o700)
    except OSError as error:
        raise HelperError("assistant storage permissions could not be set") from error
    return path


def _mapping() -> dict[str, Any]:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "version": 1,
            "next_id": FIRST_AGENT_PROJECT_ID,
            "projects": {},
        }
    except (OSError, json.JSONDecodeError) as error:
        raise HelperError("project mapping is unreadable or corrupt") from error
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or not isinstance(data.get("next_id"), int)
        or data["next_id"] < FIRST_AGENT_PROJECT_ID
        or not isinstance(data.get("projects"), dict)
    ):
        raise HelperError("project mapping is corrupt")
    projects = data["projects"]
    ids: set[int] = set()
    names: set[str] = set()
    paths: set[str] = set()
    for key, entry in projects.items():
        if (
            not isinstance(key, str)
            or not isinstance(entry, dict)
            or not isinstance(entry.get("id"), int)
            or entry["id"] < PLATFORM_PROJECT_ID
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("path"), str)
        ):
            raise HelperError("project mapping is corrupt")
        if entry["id"] in ids or entry["name"] in names or entry["path"] in paths:
            raise HelperError("project mapping reuses an ID, name, or path")
        ids.add(entry["id"])
        names.add(entry["name"])
        paths.add(entry["path"])
        if key == "platform":
            if (
                entry["id"] != PLATFORM_PROJECT_ID
                or entry["name"] != "qpl_platform_state"
                or entry["path"] != str(_volume_root(PLATFORM_VOLUME))
            ):
                raise HelperError("platform project mapping is invalid")
        else:
            try:
                agent_id = str(UUID(key))
            except ValueError as error:
                raise HelperError("agent project mapping has an invalid key") from error
            expected_name = f"qpl_agent_{agent_id.replace('-', '_')}"
            expected_path = str(_volume_root(AGENT_VOLUME) / agent_id)
            if (
                agent_id != key
                or entry["id"] < FIRST_AGENT_PROJECT_ID
                or entry["name"] != expected_name
                or entry["path"] != expected_path
            ):
                raise HelperError("agent project mapping is invalid")
    if ids and data["next_id"] <= max(ids):
        raise HelperError("project ID counter would reuse an assigned ID")
    return data


def _atomic_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, mode)
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise HelperError("quota configuration could not be written atomically") from error


def _save_mapping(mapping: dict[str, Any]) -> None:
    _atomic_write(
        STATE_PATH,
        (json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )


def _read_config(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except OSError as error:
        raise HelperError("XFS project configuration is unreadable") from error


def _write_project_config(mapping: dict[str, Any]) -> None:
    projects = mapping["projects"]
    existing_projects = list(
        dict.fromkeys(_read_config(PROJECTS_PATH) + _read_config(SYSTEM_PROJECTS_PATH))
    )
    existing_projid = list(
        dict.fromkeys(_read_config(PROJID_PATH) + _read_config(SYSTEM_PROJID_PATH))
    )
    project_paths: dict[str, str] = {}
    paths_by_id: dict[str, str] = {}
    project_ids: dict[str, str] = {}
    names_by_id: dict[str, str] = {}
    for line in existing_projects:
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split(":", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise HelperError("XFS projects file contains an invalid entry")
        if parts[0] in project_paths and project_paths[parts[0]] != parts[1]:
            raise HelperError("XFS projects file contains conflicting project IDs")
        if parts[1] in paths_by_id and paths_by_id[parts[1]] != parts[0]:
            raise HelperError("XFS project path is already assigned to another ID")
        project_paths[parts[0]] = parts[1]
        paths_by_id[parts[1]] = parts[0]
    for line in existing_projid:
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split(":", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise HelperError("XFS projid file contains an invalid entry")
        if parts[0] in project_ids and project_ids[parts[0]] != parts[1]:
            raise HelperError("XFS projid file contains conflicting names")
        project_ids[parts[0]] = parts[1]
        if parts[1] in names_by_id and names_by_id[parts[1]] != parts[0]:
            raise HelperError("XFS projid file contains duplicate numeric IDs")
        names_by_id[parts[1]] = parts[0]

    generated_projects = {entry["id"]: entry["path"] for entry in projects.values()}
    generated_projid = {entry["name"]: entry["id"] for entry in projects.values()}
    for entry in projects.values():
        project_id = str(entry["id"])
        name = entry["name"]
        path = entry["path"]
        if project_id in project_paths and project_paths[project_id] != path:
            raise HelperError("project ID is already assigned to another path")
        if name in project_ids and project_ids[name] != project_id:
            raise HelperError("project name is already assigned to another ID")
        if project_id in names_by_id and names_by_id[project_id] != name:
            raise HelperError("project ID is already assigned to another name")
    if len(generated_projects) != len(projects) or len(generated_projid) != len(projects):
        raise HelperError("project mapping reuses an ID or name")

    own_ids = set(generated_projects)
    own_names = set(generated_projid)
    own_paths = set(generated_projects.values())
    kept_projects = [
        line
        for line in existing_projects
        if not line
        or line.lstrip().startswith("#")
        or (
            line.split(":", 1)[0] not in {str(value) for value in own_ids}
            and line.split(":", 1)[-1] not in own_paths
        )
    ]
    kept_projid = [
        line
        for line in existing_projid
        if not line
        or line.lstrip().startswith("#")
        or (
            line.split(":", 1)[0] not in own_names
            and line.split(":", 1)[-1] not in {str(value) for value in own_ids}
        )
    ]
    kept_projects.extend(f"{entry['id']}:{entry['path']}" for entry in projects.values())
    kept_projid.extend(f"{entry['name']}:{entry['id']}" for entry in projects.values())
    for path, lines in ((PROJECTS_PATH, kept_projects), (PROJID_PATH, kept_projid)):
        _atomic_write(path, ("\n".join(lines).rstrip() + "\n").encode(), 0o644)


def _mount_info(path: Path) -> tuple[Path, str]:
    try:
        completed = subprocess.run(
            [
                "/usr/bin/findmnt",
                "--noheadings",
                "--raw",
                "--output",
                "TARGET,FSTYPE,OPTIONS",
                "--target",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HelperError("could not inspect the Docker volume filesystem") from error
    fields = completed.stdout.strip().split()
    if len(fields) != 3 or fields[1] != "xfs":
        raise HelperError("Docker volumes must reside on an XFS filesystem")
    options = set(fields[2].split(","))
    if not options.intersection({"prjquota", "pquota"}):
        raise HelperError("XFS project quotas are not enabled")
    return Path(fields[0]), fields[2]


def _run_xfs(mountpoint: Path, project: dict[str, Any]) -> None:
    try:
        subprocess.run(
            [
                XFS_QUOTA,
                "-D",
                str(PROJECTS_PATH),
                "-P",
                str(PROJID_PATH),
                "-x",
                "-c",
                f"project -s {project['name']}",
                "-c",
                f"limit -p bhard=2g ihard={QUOTA_INODES} {project['name']}",
                str(mountpoint),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HelperError("XFS project quota could not be applied") from error


def _project_attributes(path: Path) -> tuple[int, int]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            buffer = bytearray(28)
            fcntl.ioctl(descriptor, FS_IOC_FSGETXATTR, buffer, True)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise HelperError("XFS project inheritance could not be verified") from error
    flags, _extsize, _nextents, project_id, _cowextsize = struct.unpack_from("=5I", buffer)
    return project_id, flags


def _reported_value(output: str, project_id: int, column: str) -> int:
    headers: list[str] | None = None
    for line in output.splitlines():
        if "Project ID" in line and "Hard" in line:
            headers = re.findall(r"Used|Soft|Hard|Warn/Grace", line)
            continue
        if headers is None:
            continue
        fields = line.split()
        # Numeric xfs_quota reports may prefix project IDs with a single #.
        numeric_id = re.fullmatch(r"#?([0-9]+)", fields[0]) if fields else None
        if numeric_id is not None and int(numeric_id.group(1)) == project_id:
            values = fields[1:]
            hard_index = headers.index(column) if column in headers else -1
            if hard_index < 0 or len(values) <= hard_index:
                break
            try:
                return int(values[hard_index].rstrip("*"))
            except ValueError:
                break
    raise HelperError("XFS did not report the expected project hard limit")


def _reported_hard_limit(output: str, project_id: int) -> int:
    return _reported_value(output, project_id, "Hard")


def _verify_project(project: dict[str, Any], mountpoint: Path) -> dict[str, int]:
    path = Path(project["path"])
    project_id, flags = _project_attributes(path)
    if project_id != project["id"] or not flags & PROJECT_INHERIT:
        raise HelperError("XFS project ID or inheritance flag does not match")
    try:
        block_report = subprocess.run(
            [
                XFS_QUOTA,
                "-D",
                str(PROJECTS_PATH),
                "-P",
                str(PROJID_PATH),
                "-x",
                "-c",
                "report -p -n -b",
                str(mountpoint),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        inode_report = subprocess.run(
            [
                XFS_QUOTA,
                "-D",
                str(PROJECTS_PATH),
                "-P",
                str(PROJID_PATH),
                "-x",
                "-c",
                "report -p -n -i",
                str(mountpoint),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HelperError("XFS project limits could not be verified") from error
    block_hard = _reported_hard_limit(block_report.stdout, project["id"])
    inode_hard = _reported_hard_limit(inode_report.stdout, project["id"])
    if block_hard != QUOTA_BYTES // 1024 or inode_hard != QUOTA_INODES:
        raise HelperError("XFS project hard limits do not match the required quota")
    return {
        "used_bytes": _reported_value(block_report.stdout, project["id"], "Used") * 1024,
        "hard_bytes": QUOTA_BYTES,
        "used_inodes": _reported_value(inode_report.stdout, project["id"], "Used"),
        "hard_inodes": QUOTA_INODES,
    }


def _verify_filesystems() -> tuple[Path, Path]:
    platform_root = _volume_root(PLATFORM_VOLUME)
    agent_root = _volume_root(AGENT_VOLUME)
    platform_mount, _platform_options = _mount_info(platform_root)
    agent_mount, _agent_options = _mount_info(agent_root)
    if platform_mount != agent_mount:
        raise HelperError("platform and agent volumes must share the verified XFS filesystem")
    return platform_mount, platform_root


def _ensure_project(key: str, name: str, path: Path) -> dict[str, Any]:
    mapping = _mapping()
    projects = mapping["projects"]
    entry = projects.get(key)
    expected_path = str(path)
    if entry is None:
        if key == "platform":
            project_id = PLATFORM_PROJECT_ID
        else:
            project_id = mapping["next_id"]
            if project_id >= 2**31 - 1:
                raise HelperError("XFS project ID space is exhausted")
            mapping["next_id"] = project_id + 1
        entry = {"id": project_id, "name": name, "path": expected_path}
        projects[key] = entry
    elif entry["name"] != name or entry["path"] != expected_path:
        raise HelperError("saved project mapping does not match the fixed volume path")
    _save_mapping(mapping)
    _write_project_config(mapping)
    mountpoint, _options = _mount_info(path)
    _run_xfs(mountpoint, entry)
    return _verify_project(entry, mountpoint)


def _verify_all() -> None:
    mountpoint, _platform_root = _verify_filesystems()
    mapping = _mapping()
    for project in mapping["projects"].values():
        _verify_project(project, mountpoint)


def _dispatch(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise HelperError("request must be an object")
    quota = None
    operation = request.get("operation")
    if operation == "verify" and set(request) == {"operation"}:
        _verify_all()
    elif operation == "ensure_platform" and set(request) == {"operation"}:
        platform_root = _volume_root(PLATFORM_VOLUME)
        os.chown(platform_root, AGENT_UID, AGENT_GID)
        os.chmod(platform_root, 0o700)
        agent_root = _volume_root(AGENT_VOLUME)
        os.chown(agent_root, AGENT_UID, AGENT_GID)
        os.chmod(agent_root, 0o700)
        quota = _ensure_project("platform", "qpl_platform_state", platform_root)
    elif operation == "ensure_agent" and set(request) == {"operation", "agent_id"}:
        agent_id = request.get("agent_id")
        if not isinstance(agent_id, str):
            raise HelperError("agent_id must be a canonical UUID")
        quota = _ensure_project(
            agent_id,
            f"qpl_agent_{agent_id.replace('-', '_')}",
            _agent_path(agent_id),
        )
    elif operation == "verify_platform" and set(request) == {"operation"}:
        path = _volume_root(PLATFORM_VOLUME)
        entry = _mapping()["projects"].get("platform")
        if entry is None or entry["path"] != str(path):
            raise HelperError("platform quota has not been provisioned")
        mountpoint, _options = _mount_info(path)
        quota = _verify_project(entry, mountpoint)
        filesystem = os.statvfs(mountpoint)
        return {"ok": True, "quota": quota,
                "free_bytes": filesystem.f_bavail * filesystem.f_frsize}
    elif operation == "verify_agent" and set(request) == {"operation", "agent_id"}:
        agent_id = request.get("agent_id")
        if not isinstance(agent_id, str):
            raise HelperError("agent_id must be a canonical UUID")
        path = _agent_path(agent_id, prepare=False)
        entry = _mapping()["projects"].get(agent_id)
        if entry is None or entry["path"] != str(path):
            raise HelperError("agent quota has not been provisioned")
        mountpoint, _options = _mount_info(path)
        quota = _verify_project(entry, mountpoint)
    else:
        raise HelperError("unsupported operation or request fields")
    return {"ok": True, **({"quota": quota} if quota is not None else {})}


async def _serve_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        line = await asyncio.wait_for(reader.readline(), timeout=5)
        if not line or len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            raise HelperError("request must be one bounded JSON line")
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            raise HelperError("request is not valid JSON") from error
        async with _operation_lock:
            operation = asyncio.create_task(asyncio.to_thread(_dispatch, request))
            try:
                result = await asyncio.wait_for(asyncio.shield(operation), timeout=30)
            except TimeoutError:
                await operation
                raise
        response = json.dumps(result, separators=(",", ":")).encode() + b"\n"
    except HelperError as error:
        logger.warning("Quota helper rejected a request: %s", error)
        response = b'{"ok":false}\n'
    except TimeoutError:
        logger.error("Quota helper operation exceeded its response deadline")
        response = b'{"ok":false}\n'
    writer.write(response)
    try:
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def serve() -> None:
    SOCKET_PATH.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    parent_info = SOCKET_PATH.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise RuntimeError("Quota helper socket directory is not a real directory")
    os.chown(SOCKET_PATH.parent, AGENT_UID, AGENT_GID)
    os.chmod(SOCKET_PATH.parent, 0o750)
    try:
        info = SOCKET_PATH.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISSOCK(info.st_mode):
            raise RuntimeError("Quota helper socket path exists and is not a socket")
        SOCKET_PATH.unlink()
    server = await asyncio.start_unix_server(
        _serve_connection,
        path=SOCKET_PATH,
        limit=MAX_REQUEST_BYTES,
    )
    os.chown(SOCKET_PATH, AGENT_UID, AGENT_GID)
    os.chmod(SOCKET_PATH, 0o600)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve())
