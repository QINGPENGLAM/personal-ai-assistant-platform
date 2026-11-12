from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib.metadata
import json
import logging
import os
import stat
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from .config import Settings
from .event_store import EventStore
from .repository import Repository, RunRecord

logger = logging.getLogger(__name__)

REPORT_VERSION = 1
REPORT_MAX_BYTES = 64 * 1024
AUTH_FILE_MAX_BYTES = 1024 * 1024
REPORT_MAX_AGE_MS = 24 * 60 * 60 * 1000
REPORT_FUTURE_SKEW_MS = 5 * 60 * 1000
CODEX_CLI_VERSION = "0.166.0"
TOOLKIT_BASE_URL = "https://api.toolkit.umgpt.umich.edu/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True, slots=True)
class ModelReportResult:
    models: frozenset[str]
    reason: str | None


class VerifiedModelReports:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @classmethod
    def for_instance(cls, settings: Settings, instance_id: str, image_id: str):
        if str(UUID(instance_id)) != instance_id:
            raise ValueError("The report instance must be a canonical UUID.")
        if len(image_id) != 71 or not image_id.startswith("sha256:"):
            raise ValueError("The report requires an immutable image identity.")
        int(image_id[7:], 16)
        return cls(replace(
            settings, agent_image=image_id,
            model_report_path=settings.platform_state_root / "probe-share" / instance_id
            / "report" / "verified-models.json",
        ))

    def write_report(self, candidates: dict[str, str], generated_at: int | None = None) -> None:
        content = self.build_report(candidates, generated_at)
        path = self.settings.model_report_path
        self._ensure_report_directory()
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
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
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def refresh(self) -> ModelReportResult:
        try:
            key = self._load_or_create_key()
            report = self._read_report()
            expected_binding = self._configuration_binding(key)
            if not hmac.compare_digest(report["binding"], expected_binding):
                return self._invalid("configuration_mismatch")
            now_ms = time.time_ns() // 1_000_000
            generated_at = report["generated_at"]
            if generated_at > now_ms + REPORT_FUTURE_SKEW_MS:
                return self._invalid("future_report")
            if now_ms - generated_at > REPORT_MAX_AGE_MS:
                return self._invalid("expired_report")
            candidates = report["candidates"]
            if set(candidates) != set(self.settings.models):
                return self._invalid("candidate_mismatch")
            return ModelReportResult(
                models=frozenset(
                    model for model, result in candidates.items() if result == "passed"
                ),
                reason=None,
            )
        except FileNotFoundError:
            return self._invalid("report_missing")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            logger.warning("Verified-model report is unavailable: %s", error)
            return self._invalid("report_invalid")

    def build_report(
        self, candidates: dict[str, str], generated_at: int | None = None,
        *, key: bytes | None = None,
    ) -> bytes:
        if set(candidates) != set(self.settings.models) or any(
            result not in {"passed", "failed"} for result in candidates.values()
        ):
            raise ValueError("A model report must include every configured candidate once.")
        key = self._load_or_create_key() if key is None else key
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("The model-report HMAC key is invalid.")
        report = {
            "version": REPORT_VERSION,
            "generated_at": (
                time.time_ns() // 1_000_000 if generated_at is None else generated_at
            ),
            "binding": self._configuration_binding(key),
            "candidates": dict(sorted(candidates.items())),
        }
        return (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()

    def _configuration_binding(self, key: bytes) -> str:
        credential_digest = hashlib.sha256(self._credential_content()).hexdigest()
        if self.settings.auth_mode == "toolkit":
            provider = {"id": "toolkit", "base_url": TOOLKIT_BASE_URL}
        else:
            provider = {"id": "openai", "base_url": OPENAI_BASE_URL}
        try:
            sdk_version = importlib.metadata.version("openai-codex")
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError("The pinned Python Codex SDK is not installed.") from error
        configuration = {
            "version": REPORT_VERSION,
            "codex_cli_version": CODEX_CLI_VERSION,
            "python_sdk_version": sdk_version,
            "agent_image": self.settings.agent_image,
            "models": sorted(self.settings.models),
            "auth_mode": self.settings.auth_mode,
            "provider": provider,
            "gateway_hosts": sorted(self.settings.allowed_model_hosts),
            "credential_digest": credential_digest,
        }
        canonical = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hmac.new(key, canonical, hashlib.sha256).hexdigest()

    def _credential_content(self) -> bytes:
        if self.settings.auth_mode == "toolkit":
            if not self.settings.toolkit_api_key:
                raise ValueError("The configured Toolkit credential is missing.")
            return self.settings.toolkit_api_key.encode()
        path = self.settings.auth_file_validation_path
        if path is None:
            raise ValueError("The configured Codex authentication file is unavailable.")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > AUTH_FILE_MAX_BYTES:
                raise ValueError("The configured Codex authentication file is invalid.")
            chunks: list[bytes] = []
            total = 0
            while total <= AUTH_FILE_MAX_BYTES:
                chunk = os.read(descriptor, min(64 * 1024, AUTH_FILE_MAX_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            if total > AUTH_FILE_MAX_BYTES:
                raise ValueError("The configured Codex authentication file is too large.")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def _load_or_create_key(self) -> bytes:
        path = self.settings.model_report_key_path
        self._ensure_key_directory()
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            return self._read_key(path)
        try:
            key = os.urandom(32)
            with os.fdopen(descriptor, "wb", closefd=False) as file:
                file.write(key)
                file.flush()
                os.fsync(file.fileno())
        except BaseException:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            raise
        os.close(descriptor)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return key

    @staticmethod
    def _read_key(path: Path) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_gid != os.getegid()
                or info.st_mode & 0o077
                or info.st_size != 32
            ):
                raise ValueError("The model-report HMAC key is invalid.")
            key = os.read(descriptor, 33)
            if len(key) != 32:
                raise ValueError("The model-report HMAC key is invalid.")
            return key
        finally:
            os.close(descriptor)

    def _read_report(self) -> dict[str, Any]:
        self._ensure_report_directory(create=False)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(self.settings.model_report_path, flags)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_gid != os.getegid()
                or info.st_mode & 0o077
                or info.st_size > REPORT_MAX_BYTES
            ):
                raise ValueError("The verified-model report is not a bounded regular file.")
            data = os.read(descriptor, REPORT_MAX_BYTES + 1)
            if len(data) > REPORT_MAX_BYTES:
                raise ValueError("The verified-model report exceeds its size limit.")
        finally:
            os.close(descriptor)
        report = json.loads(data)
        if (
            not isinstance(report, dict)
            or set(report) != {"version", "generated_at", "binding", "candidates"}
            or type(report["version"]) is not int
            or report["version"] != REPORT_VERSION
            or isinstance(report["generated_at"], bool)
            or not isinstance(report["generated_at"], int)
            or not isinstance(report["binding"], str)
            or len(report["binding"]) != 64
            or not isinstance(report["candidates"], dict)
            or any(
                not isinstance(model, str)
                or result not in {"passed", "failed"}
                for model, result in report["candidates"].items()
            )
        ):
            raise ValueError("The verified-model report has an invalid schema.")
        return report

    @staticmethod
    def _invalid(reason: str) -> ModelReportResult:
        return ModelReportResult(models=frozenset(), reason=reason)

    def _ensure_key_directory(self) -> None:
        root = self._ensure_report_root(create=True)
        self._ensure_owned_directory(root / "key", create=True)
        self._ensure_owned_directory(root / "report", create=True)

    def _ensure_report_directory(self, *, create: bool = True) -> None:
        root = self._ensure_report_root(create=create)
        self._ensure_owned_directory(root / "report", create=create)

    def _ensure_report_root(self, *, create: bool) -> Path:
        base = self.settings.platform_state_root
        path = self.settings.model_report_path
        parts = path.relative_to(base).parts
        if parts == ("model-report", "report", "verified-models.json"):
            directories = [base, base / "model-report"]
        elif (
            len(parts) == 4 and parts[0] == "probe-share"
            and parts[2:] == ("report", "verified-models.json")
            and str(UUID(parts[1])) == parts[1]
        ):
            directories = [base, base / "probe-share", base / "probe-share" / parts[1]]
        else:
            raise ValueError("The model report must use the dedicated platform-state subpath.")
        for directory in directories:
            self._ensure_owned_directory(directory, create=create)
        return directories[-1]

    @staticmethod
    def _ensure_owned_directory(path: Path, *, create: bool) -> None:
        if create:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_gid != os.getegid()
            or info.st_mode & 0o077
        ):
            raise ValueError("The model-report directory has unsafe permissions.")


class RecoveryRuntime(Protocol):
    async def running_agent_ids(self) -> list[str]: ...

    async def stop_agent(self, agent_id: str) -> None: ...

    def remove_run_token(self, agent_id: str) -> None: ...


class RecoveryAuthority(Protocol):
    def revoke_run(self, run_id: str) -> None: ...

    async def wait_run_closed(self, run_id: str) -> None: ...


class StartupRecovery:
    """Recover durable run state only after old container authority is terminated."""

    def __init__(
        self, repository: Repository, event_store: EventStore,
        containers: RecoveryRuntime, network: RecoveryAuthority,
        *, inspector: HistoryInspector | None = None,
    ) -> None:
        self.repository = repository
        self.event_store = event_store
        self.containers = containers
        self.network = network
        self.inspector = inspector

    async def recover(self, *, inspect_history: bool = True) -> None:
        runs = self.repository.prepare_startup_recovery()
        by_agent: dict[str, list[RunRecord]] = {}
        for run in runs:
            by_agent.setdefault(run.agent_id, []).append(run)
            self.network.revoke_run(run.id)
        # Inventory failures block startup: unknown running containers consume capacity.
        running = await self.containers.running_agent_ids()
        for agent_id in sorted(set(by_agent) | set(running)):
            affected = by_agent.get(agent_id, [])
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        self.containers.stop_agent(agent_id),
                        *(self.network.wait_run_closed(run.id) for run in affected),
                        return_exceptions=True,
                    ), timeout=35,
                )
                # File cleanup must never prevent termination of the old container.
                self.containers.remove_run_token(agent_id)
                if any(isinstance(result, BaseException) for result in results):
                    raise RuntimeError("Container or transport cleanup was not confirmed.")
                for run in affected:
                    intended, reason = self.repository.run_cleanup_intent(run.id)
                    self.event_store.finalize(run.id, {
                        "status": intended, "reason": reason, "history_incomplete": True,
                    })
                    self.repository.complete_recovered_run(run.id)
                self.repository.set_recovery_blocker(agent_id, blocked=False)
            except Exception:
                logger.exception("Startup cleanup failed for assistant %s", agent_id)
                for run in affected:
                    self.repository.fail_run_cleanup(run.id)
                known = self.repository.set_recovery_blocker(agent_id, blocked=True)
                if not known:
                    raise RuntimeError(
                        "An untracked assistant container could not be stopped."
                    ) from None
        if self.inspector is not None and inspect_history:
            for state in self.repository.history_states():
                # Cleanup all old containers before starting any metadata-only lease.
                agent_runs = by_agent.get(str(state["id"]), [])
                recovered_runs = [
                    replace(
                        run, status=self.repository.run_cleanup_intent(run.id)[0] or "interrupted",
                    )
                    for run in agent_runs
                ]
                await self.inspector.inspect(state, recovered_runs)


class HistoryRuntime(RecoveryRuntime, Protocol):
    async def start_agent(self, agent_id: str) -> None: ...


class HistorySession(Protocol):
    async def read_history(self, model: str, thread_id: str) -> dict[str, object]: ...

    async def close(self) -> None: ...


class HistoryGateway(Protocol):
    async def connect(self, agent_id: str, *, history_only: bool = False) -> HistorySession: ...


class HistoryInspector:
    """Serial startup-only leases read SDK metadata without granting run authority."""

    def __init__(
        self, repository: Repository, event_store: EventStore,
        containers: HistoryRuntime, codex: HistoryGateway, *, deadline: float = 30,
    ) -> None:
        self.repository = repository
        self.event_store = event_store
        self.containers = containers
        self.codex = codex
        self.deadline = deadline

    async def inspect(self, state: dict[str, Any], runs: list[RunRecord]) -> None:
        agent_id = state["id"]
        current_thread = state["chat_thread_id"]
        if "recovery_error" in json.loads(state["blockers_json"]):
            return
        if current_thread is None and not runs:
            self.repository.set_history_blocker(agent_id, blocked=bool(state["used"]))
            return
        session: HistorySession | None = None
        readable: set[str] = set()
        restored: set[str] = set()

        async def read_metadata() -> None:
            nonlocal session
            if len(await self.containers.running_agent_ids()) >= 2:
                raise RuntimeError("No container capacity for history inspection.")
            await self.containers.start_agent(agent_id)
            session = await self.codex.connect(agent_id, history_only=True)
            threads = {run.thread_id for run in runs if run.thread_id is not None}
            if current_thread is not None:
                threads.add(current_thread)
            for thread_id in sorted(threads):
                try:
                    history = await session.read_history(state["model"], thread_id)
                    if history.get("id") != thread_id:
                        raise ValueError("Stored thread identity mismatch.")
                    turns = history.get("turns")
                    if not isinstance(turns, list):
                        raise ValueError("Stored thread has no turn metadata.")
                    readable.add(thread_id)
                    for run in runs:
                        if run.thread_id == thread_id and await self._restore_turn(run, turns):
                            restored.add(run.id)
                except Exception:
                    logger.warning("History metadata unavailable for assistant %s", agent_id)

        try:
            await asyncio.wait_for(read_metadata(), timeout=self.deadline)
        except Exception:
            logger.warning("History inspection did not complete for assistant %s", agent_id)
        finally:
            # Stop independently of SDK closure, including failed or timed-out launch attempts.
            operations = [self.containers.stop_agent(agent_id)]
            if session is not None:
                operations.append(session.close())
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*operations, return_exceptions=True), timeout=35,
                )
                clean = not any(isinstance(result, BaseException) for result in results)
            except TimeoutError:
                clean = False
            if not clean:
                self.repository.set_recovery_blocker(agent_id, blocked=True)
            self.repository.set_history_blocker(
                agent_id, blocked=bool(state["used"]) and current_thread not in readable,
            )
        if not clean:
            return
        for run in runs:
            if run.id in restored:
                self.event_store.finalize(run.id, {
                    "status": run.status, "history_incomplete": False,
                    "activities_closed": True,
                })

    async def _restore_turn(self, run: RunRecord, turns: list[Any]) -> bool:
        if run.sdk_turn_id is None:
            return False
        matches = [
            turn for turn in turns
            if isinstance(turn, dict) and turn.get("id") == run.sdk_turn_id
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("items"), list):
            return False
        complete = True
        for index, item in enumerate(matches[0]["items"]):
            if index >= 20_000:
                return False
            if index % 64 == 0:
                await asyncio.sleep(0)
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                complete = False
                continue
            item_id = item["id"]
            if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                text = item["text"][:64 * 1024].encode("utf-8")
                complete = complete and len(item["text"]) <= 64 * 1024 and len(text) <= 64 * 1024
                self.event_store.append(run.id, "message_snapshot", {
                    "item_id": item_id,
                    "full_text": text[:64 * 1024].decode("utf-8", errors="ignore"),
                }, sdk_item_id=item_id)
            elif item.get("type") in {"commandExecution", "fileChange", "mcpToolCall"}:
                self.event_store.append(run.id, "activity_completed", {
                    "item_id": item_id, "name": item["type"],
                    "status": item.get("status", "unknown"),
                    "summary": "Recovered activity metadata",
                }, sdk_item_id=item_id)
            elif item.get("type") not in {"userMessage", "reasoning"}:
                complete = False
        return complete
