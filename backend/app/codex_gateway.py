from __future__ import annotations

from uuid import UUID

from openai_codex import (
    ApprovalMode,
    AsyncCodex,
    AsyncThread,
    AsyncTurnHandle,
    CodexConfig,
    Sandbox,
)

WORKSPACE = "/workspace"


class CodexSession:
    def __init__(self, codex: AsyncCodex, *, history_only: bool = False) -> None:
        self._codex = codex
        self.history_only = history_only

    async def prepare_thread(self, model: str, thread_id: str | None = None) -> AsyncThread:
        if self.history_only and thread_id is None:
            raise ValueError("A history-only session requires an existing thread.")
        options = {
            "approval_mode": ApprovalMode.deny_all,
            "cwd": WORKSPACE,
            "model": model,
            "sandbox": Sandbox.full_access,
        }
        if thread_id is None:
            return await self._codex.thread_start(**options)
        return await self._codex.thread_resume(thread_id, **options)

    async def start_turn(
        self,
        thread: AsyncThread,
        model: str,
        prompt: str,
    ) -> AsyncTurnHandle:
        if self.history_only:
            raise ValueError("A history-only session cannot submit a turn.")
        return await thread.turn(
            prompt,
            approval_mode=ApprovalMode.deny_all,
            cwd=WORKSPACE,
            model=model,
            sandbox=Sandbox.full_access,
        )

    async def read_history(self, model: str, thread_id: str) -> dict[str, object]:
        if not self.history_only:
            raise ValueError("History reads require a history-only session.")
        thread = await self.prepare_thread(model, thread_id)
        response = await thread.read(include_turns=True)
        if response.thread.id != thread_id:
            raise ValueError("Stored thread identity does not match the requested thread.")
        return response.thread.model_dump(mode="json", by_alias=True, exclude_none=True)

    async def close(self) -> None:
        await self._codex.close()


class CodexGateway:
    def __init__(self, instance_id: str, docker_command: str = "docker") -> None:
        canonical_instance_id = str(UUID(instance_id))
        if canonical_instance_id != instance_id:
            raise ValueError("instance_id must be a canonical UUID")
        self.instance_id = canonical_instance_id
        self.docker_command = docker_command

    async def connect(
        self, agent_id: str, *, history_only: bool = False, probe_only: bool = False,
    ) -> CodexSession:
        if history_only and probe_only:
            raise ValueError("Choose one Codex launcher mode.")
        canonical_agent_id = str(UUID(agent_id))
        if canonical_agent_id != agent_id:
            raise ValueError("agent_id must be a canonical UUID")

        codex = AsyncCodex(
            CodexConfig(
                launch_args_override=(
                    self.docker_command,
                    "exec",
                    "-i",
                    f"qpl-agent-{self.instance_id}-{canonical_agent_id}",
                    "/usr/local/bin/launch-codex",
                    *(("--history-only",) if history_only else ("--probe",) if probe_only else ()),
                )
            )
        )
        await codex.__aenter__()
        return CodexSession(codex, history_only=history_only)
