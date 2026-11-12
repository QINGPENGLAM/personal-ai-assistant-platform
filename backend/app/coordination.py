from __future__ import annotations

import asyncio
import fcntl
import os
import stat
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID


class CoordinationError(RuntimeError):
    pass


class RuntimeProcessLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            raise CoordinationError("The runtime process lock is already held.")
        parent = self.path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_info = parent.lstat()
        if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
            raise CoordinationError("The runtime lock directory is not a safe directory.")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as error:
            raise CoordinationError("The runtime process lock could not be opened.") from error
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_gid != os.getegid()
                or info.st_mode & 0o077
            ):
                raise CoordinationError("The runtime process lock has unsafe ownership or mode.")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(fd)
            raise CoordinationError("Another backend owns this runtime.") from error
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class Coordination:
    def __init__(self, execution_capacity: int = 2) -> None:
        if (
            isinstance(execution_capacity, bool)
            or not isinstance(execution_capacity, int)
            or execution_capacity < 1
        ):
            raise ValueError("execution_capacity must be a positive integer")
        self.execution_slots = asyncio.Semaphore(execution_capacity)
        self.lifecycle_lock = asyncio.Lock()
        self._agent_locks: dict[str, asyncio.Lock] = {}
        self._agent_locks_guard = asyncio.Lock()

    @asynccontextmanager
    async def execution_slot(self) -> AsyncIterator[None]:
        await self.execution_slots.acquire()
        try:
            yield
        finally:
            self.execution_slots.release()

    @asynccontextmanager
    async def agent_locks(self, agent_ids: Iterable[str]) -> AsyncIterator[None]:
        canonical_ids = sorted({self._canonical_uuid(value) for value in agent_ids})
        locks = await self._get_agent_locks(canonical_ids)
        acquired: list[asyncio.Lock] = []
        try:
            for lock in locks:
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()

    @asynccontextmanager
    async def admission(self, agent_ids: Iterable[str]) -> AsyncIterator[None]:
        async with self.execution_slot():
            async with self.lifecycle_lock:
                async with self.agent_locks(agent_ids):
                    yield

    async def _get_agent_locks(self, agent_ids: list[str]) -> list[asyncio.Lock]:
        async with self._agent_locks_guard:
            return [
                self._agent_locks.setdefault(agent_id, asyncio.Lock())
                for agent_id in agent_ids
            ]

    @staticmethod
    def _canonical_uuid(value: str) -> str:
        canonical = str(UUID(value))
        if canonical != value:
            raise ValueError("agent IDs must be canonical UUIDs")
        return canonical
