from __future__ import annotations

from .repository import AgentRecord, Repository


def require_owned_agent(repository: Repository, user_id: str, agent_id: str) -> AgentRecord:
    return repository.get_owned_agent(user_id, agent_id)
