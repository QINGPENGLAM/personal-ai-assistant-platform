from __future__ import annotations

import os
import sqlite3
import stat
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path


class StorageWriteBlocked(RuntimeError):
    pass


class StorageMaintenanceError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


MAINTENANCE_TIMEOUT_SECONDS = 30


class StorageProtection:
    """Conservative emergency mode grants zero terminal allocations until verified repair."""

    def __init__(self):
        self.enabled = False
        self.checked = False
        self.latched = False
        self.can_recover = False
        self.reason = None
        self.snapshot = None
        self.on_trip = None
        self._lock = threading.RLock()
        self._maintenance_active = False
        self._maintenance_owner = None
        self._maintenance_authority = ContextVar(f"storage_maintenance_{id(self)}", default=None)
        self._repair = ContextVar(f"storage_repair_{id(self)}", default=False)

    def trip(self, reason):
        with self._lock:
            self.latched = True
            self.can_recover = False
            self.reason = reason
            callback = self.on_trip
        if callback is not None:
            callback()

    def observe(self, snapshot):
        try:
            quota = snapshot["quota"]
            used, hard, free = quota["used_bytes"], quota["hard_bytes"], snapshot["free_bytes"]
            if (
                any(type(value) is not int or value < 0 for value in (used, hard, free))
                or hard != 2 * 1024**3
            ):
                raise ValueError
        except (TypeError, KeyError, ValueError):
            self.checked = False
            self.trip("quota_measurement_unavailable")
            return
        with self._lock:
            self.checked = True
            self.snapshot = {"used_bytes": used, "hard_bytes": hard, "free_bytes": free}
        if used * 100 >= hard * 80:
            self.trip("platform_quota_high")
        elif free < 1024**3:
            self.trip("filesystem_space_low")
        else:
            self.can_recover = self.latched and used * 100 < hard * 70 and free > 2 * 1024**3

    def require_writable(self):
        if self._maintenance_active and not self.owns_maintenance():
            raise StorageWriteBlocked("Database maintenance is active. Retry this write shortly.")
        if self.enabled and (self.latched or not self.checked):
            if not (self._repair.get() and self.can_recover):
                raise StorageWriteBlocked(
                    "Platform storage protection is active. Free space and retry recovery."
                )

    @contextmanager
    def maintenance(self):
        with self._lock:
            if self._maintenance_active:
                raise StorageMaintenanceError(
                    "maintenance_busy", "Database maintenance is already active."
                )
            if self.status()["active"]:
                raise StorageWriteBlocked(
                    "Database maintenance requires healthy storage authority."
                )
            self.require_writable()
            self._maintenance_owner = object()
            self._maintenance_active = True
        token = self._maintenance_authority.set(self._maintenance_owner)
        try:
            yield
        finally:
            self._maintenance_authority.reset(token)
            with self._lock:
                self._maintenance_active = False
                self._maintenance_owner = None

    def owns_maintenance(self):
        return (
            self._maintenance_active
            and self._maintenance_owner is not None
            and self._maintenance_authority.get() is self._maintenance_owner
        )

    @contextmanager
    def repair(self):
        if not self.can_recover:
            raise StorageWriteBlocked("Storage recovery needs verified low-water measurements.")
        token = self._repair.set(True)
        try:
            yield
        finally:
            self._repair.reset(token)

    def clear_after_repair(self):
        with self._lock:
            if not self.can_recover:
                raise StorageWriteBlocked("Storage recovery has not been verified.")
            self.latched = False
            self.can_recover = False
            self.reason = None

    def status(self):
        with self._lock:
            return {
                "maintenance_active": self._maintenance_active,
                "active": self.enabled and (self.latched or not self.checked),
                "reason": self.reason,
                "measurement_verified": self.checked,
                **(self.snapshot or {}),
            }


class Database:
    def __init__(self, path: Path, busy_timeout_ms: int = 5_000) -> None:
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms
        self.storage = StorageProtection()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        self.storage.require_writable()
        connection = None
        try:
            connection = self.connect()
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            self.storage.require_writable()
            yield connection
            with self.storage._lock:
                self.storage.require_writable()
                connection.commit()
        except sqlite3.Error as error:
            if self.storage.enabled and not isinstance(error, sqlite3.IntegrityError):
                self.storage.trip("database_write_failed")
            raise
        finally:
            if connection is not None:
                try:
                    if connection.in_transaction:
                        connection.rollback()
                finally:
                    connection.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            connection.executescript(schema)
            version = connection.execute(
                "SELECT value FROM runtime_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if version is None or version["value"] != "1":
                raise RuntimeError("Unsupported database schema version")
            connection.execute(
                "INSERT OR IGNORE INTO runtime_metadata(key, value) VALUES (?, ?)",
                ("instance_uuid", str(uuid.uuid4())),
            )

    def instance_id(self) -> str:
        with self.read() as connection:
            row = connection.execute(
                "SELECT value FROM runtime_metadata WHERE key = 'instance_uuid'"
            ).fetchone()
        if row is None:
            raise RuntimeError("Runtime instance identity is missing")
        try:
            return str(uuid.UUID(row["value"]))
        except (ValueError, TypeError) as error:
            raise RuntimeError("Runtime instance identity is invalid") from error

    @staticmethod
    def cached_instance_id(path: Path) -> str:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o077
                or info.st_size > 37
            ):
                raise RuntimeError("Runtime identity cache is unsafe.")
            value = os.read(descriptor, 38).decode("ascii").strip()
            if str(uuid.UUID(value)) != value:
                raise RuntimeError("Runtime identity cache is invalid.")
            return value
        finally:
            os.close(descriptor)

    @staticmethod
    def preserve_instance_id(path: Path, instance_id: str) -> None:
        try:
            if Database.cached_instance_id(path) != instance_id:
                raise RuntimeError("Runtime identity cache does not match the database.")
            return
        except FileNotFoundError:
            pass
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            data = memoryview((instance_id + "\n").encode("ascii"))
            while data:
                count = os.write(descriptor, data)
                if count <= 0:
                    raise OSError("Runtime identity cache write did not progress.")
                data = data[count:]
            os.fsync(descriptor)
            # Publish only fully synced bytes, without overwriting a concurrent identity.
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                if Database.cached_instance_id(path) != instance_id:
                    raise RuntimeError(
                        "Runtime identity cache changed during publication."
                    ) from None
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.close(descriptor)
            os.unlink(temporary)

    def _allocated_bytes(self):
        total = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                info = Path(str(self.path) + suffix).lstat()
            except FileNotFoundError:
                if not suffix:
                    raise StorageMaintenanceError(
                        "maintenance_path", "Database file is missing."
                    ) from None
                continue
            if not stat.S_ISREG(info.st_mode):
                raise StorageMaintenanceError(
                    "maintenance_path", "Database files must be regular files."
                )
            total += info.st_blocks * 512
        return total

    def reclaim_space(
        self, sample, *, cancelled=None, _within_maintenance=False, _check_only=False
    ):
        """Internal, healthy-storage-only reclamation; caller supplies fresh quota measurements."""
        try:
            used = sample["quota"]["used_bytes"]
            hard = sample["quota"]["hard_bytes"]
            free = sample["free_bytes"]
            if (
                any(type(value) is not int or value < 0 for value in (used, hard, free))
                or hard != 2 * 1024**3
                or used > hard
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise StorageMaintenanceError(
                "maintenance_measurement", "Verified storage measurements are required."
            ) from None
        deadline = time.monotonic() + MAINTENANCE_TIMEOUT_SECONDS

        def interrupted():
            return (
                time.monotonic() >= deadline
                or (cancelled is not None and cancelled.is_set())
                or self.storage.status()["active"]
            )

        def require_authority():
            self.storage.require_writable()
            if interrupted():
                raise StorageMaintenanceError(
                    "maintenance_cancelled", "Database maintenance was interrupted."
                )

        if _within_maintenance and not self.storage.owns_maintenance():
            raise StorageWriteBlocked("Current maintenance authority is required.")
        scope = (
            nullcontext() if (_within_maintenance or _check_only) else self.storage.maintenance()
        )
        with scope:
            try:
                require_authority()
                before = self._allocated_bytes()
                with closing(self.connect()) as connection:
                    connection.execute("PRAGMA busy_timeout=1000")
                    # Refuse builds that cannot keep VACUUM's transient database in memory.
                    options = {row[0] for row in connection.execute("PRAGMA compile_options")}
                    if "TEMP_STORE=0" in options:
                        raise StorageMaintenanceError(
                            "maintenance_temp_store", "SQLite memory scratch is unavailable."
                        )
                    connection.execute("PRAGMA temp_store=MEMORY")
                    pages = connection.execute("PRAGMA page_count").fetchone()[0]
                    page_size = connection.execute("PRAGMA page_size").fetchone()[0]
                    logical_bytes = pages * page_size
                    # Allow the documented 2x extra space plus WAL-frame slack.
                    required = 2 * logical_bytes + 2 * pages * 24 + 64 * 1024
                    host = os.statvfs(self.path.parent)
                    free = min(free, host.f_bavail * host.f_frsize)
                    budget = min(336 * 1024**2, hard - used, max(0, free - 1024**3))
                    if required > budget:
                        raise StorageMaintenanceError(
                            "maintenance_space_required",
                            "Insufficient reserved space for safe database compaction.",
                        )
                    if _check_only:
                        require_authority()
                        return {"additional_budget_bytes": budget, "required_extra_bytes": required}
                    connection.set_progress_handler(lambda: int(interrupted()), 1000)
                    try:
                        for command in (
                            "PRAGMA wal_checkpoint(TRUNCATE)",
                            "VACUUM",
                            "PRAGMA wal_checkpoint(TRUNCATE)",
                        ):
                            require_authority()
                            cursor = connection.execute(command)
                            if command.startswith("PRAGMA") and cursor.fetchone()[0] != 0:
                                raise StorageMaintenanceError(
                                    "maintenance_busy",
                                    "Close active database readers and retry maintenance.",
                                )
                            cursor.close()
                        require_authority()
                        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                            self.storage.trip("database_maintenance_failed")
                            raise StorageMaintenanceError(
                                "maintenance_integrity", "Database integrity verification failed."
                            )
                    finally:
                        connection.set_progress_handler(None, 0)
                require_authority()
                after = self._allocated_bytes()
                return {
                    "before_allocated_bytes": before,
                    "after_allocated_bytes": after,
                    "reclaimed_bytes": max(0, before - after),
                    "additional_budget_bytes": budget,
                    "required_extra_bytes": required,
                }
            except sqlite3.Error as error:
                code = getattr(error, "sqlite_errorcode", 0) & 0xFF
                if code == sqlite3.SQLITE_INTERRUPT:
                    self.storage.require_writable()
                    raise StorageMaintenanceError(
                        "maintenance_cancelled", "Database maintenance was interrupted."
                    ) from None
                if code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                    raise StorageMaintenanceError(
                        "maintenance_busy",
                        "Database maintenance is waiting for another connection.",
                    ) from None
                if self.storage.enabled:
                    self.storage.trip("database_maintenance_failed")
                raise
