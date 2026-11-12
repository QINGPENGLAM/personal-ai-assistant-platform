from __future__ import annotations

import asyncio
import codecs
import errno
import hashlib
import json
import os
import re
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from python_multipart.exceptions import MultipartParseError

from .config import Settings
from .coordination import Coordination
from .repository import Repository, RepositoryError

UPLOAD_LIMIT = 10 * 1024 * 1024
REQUEST_LIMIT = UPLOAD_LIMIT + 64 * 1024
PREVIEW_LIMIT = 256 * 1024
DOWNLOAD_LIMIT = 20 * 1024 * 1024
UPLOAD_DEADLINE = 30
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
INTERNAL_TEMP = re.compile(r"\.qpl-upload-[0-9a-f-]{36}\.tmp$")


def fail(status: int, code: str, message: str):
    return RepositoryError(status, code, message)


def normalize(path: str, *, file: bool = False) -> str:
    try:
        encoded = path.encode("utf-8")
    except UnicodeError:
        raise fail(422, "invalid_path", "Use a valid workspace-relative path.") from None
    if path.startswith("/") or "\x00" in path or ".." in path.split("/") or len(encoded) > 4096:
        raise fail(422, "invalid_path", "Absolute and parent paths are not allowed.")
    parts = PurePosixPath(path).parts
    if len(parts) > 32 or any(
        len(part.encode()) > 255 or INTERNAL_TEMP.fullmatch(part) for part in parts
    ):
        raise fail(422, "invalid_path", "The workspace path is unsupported.")
    result = "/".join(parts)
    if file and not result:
        raise fail(422, "invalid_path", "Choose one workspace file.")
    return result


def filesystem_error(error: OSError):
    if error.errno == errno.EEXIST:
        return fail(409, "file_changed", "The file changed. Inspect it again.")
    if error.errno == errno.ENOENT:
        return fail(404, "file_not_found", "Workspace entry not found.")
    if error.errno in {errno.ELOOP, errno.ENOTDIR, errno.EINVAL, errno.ENXIO}:
        return fail(422, "unsafe_path", "Symbolic links and special files are not accessible.")
    if error.errno == errno.EDQUOT:
        return fail(507, "agent_quota_exceeded", "Assistant quota is full. Remove files first.")
    if error.errno == errno.ENOSPC:
        return fail(507, "storage_full", "Storage is full. Remove files before uploading again.")
    return fail(503, "workspace_unavailable", "Workspace storage is unavailable.")


class Download:
    def __init__(self, descriptor: int, name: str, size: int):
        self.descriptor = descriptor
        self.name = name
        self.size = size
        self._lock = threading.Lock()

    @property
    def closed(self):
        return self.descriptor is None

    def close(self):
        with self._lock:
            if self.descriptor is not None:
                os.close(self.descriptor)
                self.descriptor = None

    def chunks(self):
        remaining = self.size
        try:
            while remaining:
                with self._lock:
                    if self.descriptor is None:
                        return
                    chunk = os.read(self.descriptor, min(65536, remaining))
                if not chunk:
                    return
                remaining -= len(chunk)
                yield chunk
        finally:
            self.close()


class WorkspaceService:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        coordination: Coordination,
        *,
        test_mode=False,
        quota_check=None,
    ):
        self.settings = settings
        self.repository = repository
        self.coordination = coordination
        self.test_mode = test_mode
        self.quota_check = quota_check
        self.staging_root = (
            settings.runtime_root / "upload-staging" if test_mode else Path("/upload-staging")
        )
        self._staging_fd = None
        self._uploads: set[str] = set()
        # File hashes do not compete with Codex SDK threads in the default executor.
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="workspace")

    def initialize(self):
        if self.test_mode:
            self.staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.staging_root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_gid != os.getegid()
            or info.st_mode & 0o077
        ):
            raise RuntimeError("Upload staging requires a private runtime-owned directory.")
        if not self.test_mode:
            self._verify_staging_mount()
        self._staging_fd = os.open(self.staging_root, DIR_FLAGS)

    def _verify_staging_mount(self):
        mount = None
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            if len(fields) > 6 and fields[4] == "/upload-staging":
                mount = fields[fields.index("-") + 1]
        usage = os.statvfs(self.staging_root)
        if mount != "tmpfs" or usage.f_blocks * usage.f_frsize > 64 * 1024 * 1024:
            raise RuntimeError("Upload staging must be a tmpfs bounded to 64 MiB.")

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._staging_fd is not None:
            os.close(self._staging_fd)
            self._staging_fd = None

    async def io(self, function, *args, on_cancel=None, dispose_cancelled_result=None):
        task = asyncio.get_running_loop().run_in_executor(self._executor, function, *args)
        cancelled = False
        # Collect in-flight file IO before releasing locks or closing its descriptors.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                if on_cancel is not None:
                    on_cancel()
            except BaseException:
                break
        if cancelled:
            if not task.cancelled():
                error = task.exception()
                if error is None and dispose_cancelled_result is not None:
                    dispose_cancelled_result(task.result())
            raise asyncio.CancelledError
        return task.result()

    @contextmanager
    def directory(self, agent_id: str, path: str = ""):
        descriptor = None
        try:
            if str(UUID(agent_id)) != agent_id:
                raise ValueError("Invalid assistant identity")
            descriptor = os.open(self.settings.agent_data_root, DIR_FLAGS)
            for part in (agent_id, "workspace", *PurePosixPath(normalize(path)).parts):
                next_descriptor = os.open(part, DIR_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            yield descriptor
        except OSError as error:
            raise filesystem_error(error) from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def open_file(self, parent: int, name: str):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise fail(422, "unsafe_file", "Only regular workspace files are supported.")
        return descriptor

    @contextmanager
    def file(self, agent_id: str, path: str):
        path = normalize(path, file=True)
        parent, name = path.rpartition("/")[::2]
        with self.directory(agent_id, parent) as directory:
            descriptor = None
            try:
                descriptor = self.open_file(directory, name)
                yield descriptor
            except OSError as error:
                raise filesystem_error(error) from None
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    @staticmethod
    def digest(descriptor: int, deadline: float, cancelled: threading.Event | None = None):
        original = os.fstat(descriptor)
        digest = hashlib.sha256()
        while True:
            if time.monotonic() > deadline or (cancelled is not None and cancelled.is_set()):
                raise fail(408, "file_inspection_timeout", "File inspection timed out.")
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            digest.update(chunk)
        current = os.fstat(descriptor)
        if (original.st_size, original.st_mtime_ns, original.st_ctime_ns) != (
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ):
            raise fail(409, "file_changed", "The file changed. Inspect it again before continuing.")
        return digest.hexdigest()

    async def list_files(self, owner: str, agent: str, path: str):
        self.repository.get_owned_agent(owner, agent)
        return await self.io(self._list, agent, normalize(path))

    def _list(self, agent, path):
        entries = []
        truncated = False
        with self.directory(agent, path) as directory, os.scandir(directory) as scan:
            for scanned, entry in enumerate(scan):
                if scanned == 200:
                    truncated = True
                    break
                if INTERNAL_TEMP.fullmatch(entry.name):
                    continue
                if len(entries) == 200:
                    truncated = True
                    break
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                kind = (
                    "directory"
                    if stat.S_ISDIR(info.st_mode)
                    else "file"
                    if stat.S_ISREG(info.st_mode)
                    else "unsupported"
                )
                entries.append(
                    {
                        "name": entry.name,
                        "path": f"{path}/{entry.name}".lstrip("/"),
                        "kind": kind,
                        "size": info.st_size if kind == "file" else None,
                    }
                )
        return {
            "path": path,
            "entries": sorted(entries, key=lambda entry: entry["name"]),
            "truncated": truncated,
            "limit": 200,
        }

    async def metadata(self, owner, agent, path):
        self.repository.get_owned_agent(owner, agent)
        cancelled = threading.Event()
        return await self.io(
            self._metadata,
            agent,
            normalize(path, file=True),
            time.monotonic() + 30,
            cancelled,
            on_cancel=cancelled.set,
        )

    def _metadata(self, agent, path, deadline, cancelled):
        with self.file(agent, path) as descriptor:
            info = os.fstat(descriptor)
            return {
                "path": path,
                "size": info.st_size,
                "sha256": self.digest(descriptor, deadline, cancelled),
                "modified_at": datetime.fromtimestamp(info.st_mtime, UTC).isoformat(),
            }

    async def preview(self, owner, agent, path):
        self.repository.get_owned_agent(owner, agent)
        return await self.io(self._preview, agent, normalize(path, file=True))

    def _preview(self, agent, path):
        with self.file(agent, path) as descriptor:
            size = os.fstat(descriptor).st_size
            data = os.read(descriptor, PREVIEW_LIMIT + 1)
            truncated = len(data) > PREVIEW_LIMIT or size > PREVIEW_LIMIT
            try:
                if any(byte < 32 and byte not in {9, 10, 13} for byte in data):
                    raise UnicodeError
                decoder = codecs.getincrementaldecoder("utf-8")("strict")
                text = decoder.decode(data[:PREVIEW_LIMIT], final=not truncated)
            except UnicodeError:
                text = None
            return {
                "path": path,
                "size": size,
                "text": text,
                "binary": text is None,
                "truncated": truncated,
            }

    async def download(self, owner, agent, path):
        self.repository.get_owned_agent(owner, agent)
        return await self.io(
            self._download,
            agent,
            normalize(path, file=True),
            dispose_cancelled_result=lambda opened: opened.close(),
        )

    def _download(self, agent, path):
        parent, name = path.rpartition("/")[::2]
        with self.directory(agent, parent) as directory:
            try:
                descriptor = self.open_file(directory, name)
            except OSError as error:
                raise filesystem_error(error) from None
        size = os.fstat(descriptor).st_size
        if size > DOWNLOAD_LIMIT:
            os.close(descriptor)
            raise fail(413, "download_too_large", "Downloads are limited to 20 MiB.")
        return Download(descriptor, name, size)

    async def recover_operations(self):
        # Only known backend-owned staging files are removed; never replay a workspace mutation.
        with os.scandir(self._staging_fd) as scan:
            for entry in scan:
                if re.fullmatch(r"\.qpl-stage-[0-9a-f]{32}\.tmp", entry.name) and entry.is_file(
                    follow_symlinks=False
                ):
                    os.unlink(entry.name, dir_fd=self._staging_fd)
        for operation in self.repository.interrupted_workspace_operations():
            try:
                agent = self.repository.get_owned_agent(
                    operation["actor_id"], operation["resource_id"]
                )
                if "recovery_error" in agent.blockers:
                    continue
                payload = json.loads(operation["response_json"])
                path = normalize(payload["path"], file=True)
                parent = path.rpartition("/")[0]
                temporary = f".qpl-upload-{operation['request_id']}.tmp"
                with self.directory(agent.id, parent) as directory:
                    self._remove_temporary(directory, temporary)
            except (RepositoryError, OSError, ValueError, KeyError, TypeError):
                # The uncertain tombstone remains; unsafe/missing paths are not followed.
                continue

    @staticmethod
    def _remove_temporary(directory, name):
        try:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                os.unlink(name, dir_fd=directory)
        except FileNotFoundError:
            pass

    async def upload(
        self, owner, agent, request_id, path, request, *, overwrite=False, expected_sha256=None
    ):
        self.repository.get_owned_agent(owner, agent)
        self.repository.database.storage.require_writable()
        path = normalize(path, file=True)
        request_id = str(UUID(request_id))
        self._validate_confirmation(overwrite, expected_sha256)
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > REQUEST_LIMIT):
            raise fail(413, "upload_too_large", "Uploads are limited to 10 MiB.")
        if owner in self._uploads or len(self._uploads) >= 4:
            raise fail(
                429, "upload_limit", "Another upload is active. Try again after it finishes."
            )
        self._uploads.add(owner)
        stage_name = f".qpl-stage-{uuid4().hex}.tmp"
        descriptor = None
        try:
            async with asyncio.timeout(UPLOAD_DEADLINE):
                descriptor = os.open(
                    stage_name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=self._staging_fd,
                )
                receiver = UploadReceiver(descriptor, request.headers.get("content-type", ""))
                total = 0
                async for chunk in request.stream():
                    self.repository.database.storage.require_writable()
                    total += len(chunk)
                    if total > REQUEST_LIMIT:
                        raise fail(413, "upload_too_large", "Uploads are limited to 10 MiB.")
                    await self.io(receiver.feed, chunk)
                receiver.finish()
                body = {
                    "path": path,
                    "sha256": receiver.digest.hexdigest(),
                    "size": receiver.size,
                    "overwrite": overwrite,
                    "expected_sha256": expected_sha256,
                }
                body_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
                async with self.coordination.agent_locks([agent]):
                    previous = self.repository.workspace_receipt(
                        owner, agent, request_id, "workspace.upload", body_hash
                    )
                    if previous is not None:
                        return previous
                    self.repository.check_workspace_mutation(owner, agent, upload=True)
                    await self._check_quota(agent, receiver.size)
                    cancelled = threading.Event()
                    await self.io(
                        self._precondition,
                        agent,
                        path,
                        overwrite,
                        expected_sha256,
                        time.monotonic() + UPLOAD_DEADLINE,
                        cancelled,
                        on_cancel=cancelled.set,
                    )
                    self.repository.reserve_workspace_operation(
                        owner, agent, request_id, "workspace.upload", body_hash, {"path": path}
                    )
                    try:
                        await self.io(
                            self._install,
                            agent,
                            path,
                            descriptor,
                            request_id,
                            overwrite,
                            expected_sha256,
                            cancelled,
                            on_cancel=cancelled.set,
                        )
                        response = {
                            "path": path,
                            "size": receiver.size,
                            "sha256": body["sha256"],
                            "operation_id": request_id,
                            "state": "completed",
                            "replayed": False,
                        }
                        self.repository.finish_workspace_operation(
                            owner, agent, request_id, "workspace.upload", response
                        )
                        return response
                    except BaseException:
                        self.repository.finish_workspace_operation(
                            owner, agent, request_id, "workspace.upload"
                        )
                        raise
        except TimeoutError:
            raise fail(
                408, "upload_timeout", "Upload timed out. Inspect the target before trying again."
            ) from None
        except RepositoryError as error:
            if error.code == "agent_quota_exceeded":
                self.repository.set_agent_storage_blocker(agent, blocked=True)
            raise
        except OSError as error:
            if error.errno == errno.EDQUOT:
                self.repository.set_agent_storage_blocker(agent, blocked=True)
            raise filesystem_error(error) from None
        finally:
            try:
                if descriptor is not None:
                    os.close(descriptor)
                with suppress(FileNotFoundError):
                    os.unlink(stage_name, dir_fd=self._staging_fd)
            finally:
                self._uploads.discard(owner)

    @staticmethod
    def _validate_confirmation(overwrite, expected):
        if overwrite and (
            not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise fail(
                422, "confirmation_required", "Confirm the current file hash before replacing it."
            )
        if not overwrite and expected is not None:
            raise fail(
                422, "invalid_confirmation", "A new file must not include an overwrite hash."
            )

    async def _check_quota(self, agent, size=0):
        if self.quota_check is None:
            if self.test_mode:
                return None
            raise fail(503, "quota_unavailable", "File quota verification is unavailable.")
        result = await self.quota_check(agent)
        usage = result.get("quota") if isinstance(result, dict) else None
        fields = ("used_bytes", "hard_bytes", "used_inodes", "hard_inodes")
        if (
            not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in fields)
            or usage["hard_bytes"] != 2 * 1024**3
            or usage["hard_inodes"] != 100_000
        ):
            raise fail(503, "quota_unavailable", "File quota verification is unavailable.")
        if usage and (
            usage["used_bytes"] + size > usage["hard_bytes"]
            or usage["used_inodes"] >= usage["hard_inodes"]
        ):
            self.repository.set_agent_storage_blocker(agent, blocked=True)
            raise fail(
                507, "storage_full", "Assistant storage is full. Remove files before uploading."
            )
        return usage

    def _precondition(self, agent, path, overwrite, expected, deadline, cancelled):
        parent, name = path.rpartition("/")[::2]
        with self.directory(agent, parent) as directory:
            return self._target_hash(directory, name, overwrite, expected, deadline, cancelled)

    def _target_hash(self, directory, name, overwrite, expected, deadline, cancelled):
        try:
            descriptor = self.open_file(directory, name)
        except FileNotFoundError:
            if overwrite:
                raise fail(409, "file_changed", "The confirmed file no longer exists.") from None
            return
        try:
            if not overwrite:
                raise fail(409, "overwrite_required", "Confirm replacement of the existing file.")
            if self.digest(descriptor, deadline, cancelled) != expected:
                raise fail(409, "file_changed", "The file changed. Inspect and confirm it again.")
        finally:
            os.close(descriptor)

    def _install(self, agent, path, source, request_id, overwrite, expected, cancelled):
        self.repository.database.storage.require_writable()
        parent, name = path.rpartition("/")[::2]
        temporary = f".qpl-upload-{request_id}.tmp"
        deadline = time.monotonic() + UPLOAD_DEADLINE
        with self.directory(agent, parent) as directory:
            target = None
            try:
                self._target_hash(directory, name, overwrite, expected, deadline, cancelled)
                target = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
                os.fchown(
                    target,
                    os.geteuid() if self.test_mode else 1000,
                    os.getegid() if self.test_mode else 1000,
                )
                os.fchmod(target, 0o600)
                os.lseek(source, 0, os.SEEK_SET)
                while True:
                    if cancelled.is_set() or time.monotonic() > deadline:
                        raise fail(408, "upload_timeout", "Upload timed out before replacement.")
                    chunk = os.read(source, 65536)
                    if not chunk:
                        break
                    with self.repository.database.storage._lock:
                        self.repository.database.storage.require_writable()
                        write_all(target, chunk)
                os.fsync(target)
                if cancelled.is_set():
                    raise fail(408, "upload_timeout", "Upload timed out before replacement.")
                with self.repository.database.storage._lock:
                    self.repository.database.storage.require_writable()
                    if overwrite:
                        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                    else:
                        # Do not overwrite a target created after preflight.
                        os.link(
                            temporary,
                            name,
                            src_dir_fd=directory,
                            dst_dir_fd=directory,
                            follow_symlinks=False,
                        )
                        os.unlink(temporary, dir_fd=directory)
                os.fsync(directory)
            finally:
                if target is not None:
                    os.close(target)
                self._remove_temporary(directory, temporary)

    async def delete(self, owner, agent, request_id, path, expected_sha256):
        self.repository.get_owned_agent(owner, agent)
        path = normalize(path, file=True)
        self._validate_confirmation(True, expected_sha256)
        request_id = str(UUID(request_id))
        body_hash = hashlib.sha256(
            json.dumps({"path": path, "sha256": expected_sha256}, sort_keys=True).encode()
        ).hexdigest()
        async with self.coordination.agent_locks([agent]):
            previous = self.repository.workspace_receipt(
                owner, agent, request_id, "workspace.delete", body_hash
            )
            if previous is not None:
                return previous
            self.repository.check_workspace_mutation(owner, agent, upload=False)
            cancelled = threading.Event()
            await self.io(
                self._precondition,
                agent,
                path,
                True,
                expected_sha256,
                time.monotonic() + 30,
                cancelled,
                on_cancel=cancelled.set,
            )
            self.repository.reserve_workspace_operation(
                owner, agent, request_id, "workspace.delete", body_hash, {"path": path}
            )
            try:
                await self.io(
                    self._delete, agent, path, expected_sha256, cancelled, on_cancel=cancelled.set
                )
                result = {
                    "path": path,
                    "operation_id": request_id,
                    "state": "completed",
                    "replayed": False,
                }
                self.repository.finish_workspace_operation(
                    owner, agent, request_id, "workspace.delete", result
                )
            except BaseException:
                self.repository.finish_workspace_operation(
                    owner, agent, request_id, "workspace.delete"
                )
                raise
            try:
                usage = await self._check_quota(agent)
                if usage or self.test_mode:
                    self.repository.set_agent_storage_blocker(agent, blocked=False)
            except (RepositoryError, OSError):
                pass
            return result

    def _delete(self, agent, path, expected, cancelled):
        parent, name = path.rpartition("/")[::2]
        with self.directory(agent, parent) as directory:
            self._target_hash(directory, name, True, expected, time.monotonic() + 30, cancelled)
            if cancelled.is_set():
                raise fail(408, "file_operation_timeout", "The deletion was interrupted.")
            with self.repository.database.storage._lock:
                self.repository.database.storage.require_writable()
                os.unlink(name, dir_fd=directory)
            os.fsync(directory)


def write_all(descriptor, data):
    remaining = memoryview(data)
    while remaining:
        count = os.write(descriptor, remaining)
        if count <= 0:
            raise OSError(errno.EIO, "File write did not progress")
        remaining = remaining[count:]


class UploadReceiver:
    def __init__(self, descriptor, content_type):
        from python_multipart.multipart import MultipartParser, parse_options_header

        kind, options = parse_options_header(content_type)
        boundary = options.get(b"boundary", b"")
        if kind != b"multipart/form-data" or not 1 <= len(boundary) <= 200:
            raise fail(422, "invalid_upload", "Send one multipart file named file.")
        self.descriptor = descriptor
        self.size = 0
        self.digest = hashlib.sha256()
        self.headers = {}
        self.field = bytearray()
        self.value = bytearray()
        self.seen = False
        self.ended = False
        self.completed = False
        self.parser = MultipartParser(
            boundary,
            {
                "on_part_begin": self.begin,
                "on_header_field": self.header_field,
                "on_header_value": self.header_value,
                "on_header_end": self.header_end,
                "on_headers_finished": self.headers_finished,
                "on_part_data": self.data,
                "on_part_end": self.part_end,
                "on_end": self.end,
            },
        )

    def begin(self):
        if self.seen:
            raise fail(422, "invalid_upload", "Upload exactly one file at a time.")
        self.seen = True

    def header_field(self, data, start, end):
        self.field.extend(data[start:end])
        if len(self.field) > 4096:
            raise fail(422, "invalid_upload", "Upload headers are too large.")

    def header_value(self, data, start, end):
        self.value.extend(data[start:end])
        if len(self.value) > 8192:
            raise fail(422, "invalid_upload", "Upload headers are too large.")

    def header_end(self):
        key = bytes(self.field).lower()
        if key in self.headers or len(self.headers) >= 8:
            raise fail(422, "invalid_upload", "Upload headers are unsupported.")
        self.headers[key] = bytes(self.value)
        self.field.clear()
        self.value.clear()

    def headers_finished(self):
        from python_multipart.multipart import parse_options_header

        kind, options = parse_options_header(self.headers.get(b"content-disposition", b""))
        if kind != b"form-data" or options.get(b"name") != b"file" or b"filename" not in options:
            raise fail(422, "invalid_upload", "Upload one file using the file field.")

    def data(self, data, start, end):
        chunk = memoryview(data)[start:end]
        if self.size + len(chunk) > UPLOAD_LIMIT:
            raise fail(413, "upload_too_large", "Uploads are limited to 10 MiB.")
        write_all(self.descriptor, chunk)
        self.digest.update(chunk)
        self.size += len(chunk)

    def part_end(self):
        self.ended = True

    def end(self):
        self.completed = True

    def feed(self, chunk):
        try:
            self.parser.write(chunk)
        except (ValueError, MultipartParseError):
            raise fail(422, "invalid_upload", "The multipart upload is malformed.") from None

    def finish(self):
        self.parser.finalize()
        if not (self.seen and self.ended and self.completed):
            raise fail(422, "invalid_upload", "The upload was incomplete.")
