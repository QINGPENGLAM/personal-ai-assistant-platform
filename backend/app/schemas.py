from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import SUPPORTED_MODELS

Email = str
ModelName = Literal["gpt-6.1-sol", "gpt-6-luna"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class RegisterRequest(StrictModel):
    email: Email = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[^@\s]{1,64}@[^@\s.]+(?:\.[^@\s.]+)+", value):
            raise ValueError("Enter a valid email address")
        return value


class LoginRequest(StrictModel):
    email: Email = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[^@\s]{1,64}@[^@\s.]+(?:\.[^@\s.]+)+", value):
            raise ValueError("Enter a valid email address")
        return value


class UserResponse(StrictModel):
    id: UUID
    email: str


class AuthResponse(StrictModel):
    user: UserResponse
    csrf_token: str


class AgentCreateRequest(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    model: ModelName

    @field_validator("name")
    @classmethod
    def reject_blank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name must not be blank")
        return value.strip()


class RunAdmissionResponse(StrictModel):
    limited: bool
    retry_at: str | None
    minute_count: int
    day_count: int
    minute_limit: int
    day_limit: int


class AgentResponse(StrictModel):
    storage_protection: dict[str, object] | None = None
    run_admission: RunAdmissionResponse | None = None
    id: UUID
    name: str
    model: str
    status: Literal["provisioning", "provisioning_error", "ready"]
    paused: bool
    chat_epoch: int = 0
    blockers: list[str]
    container_state: Literal["stopped", "starting", "running", "stopping", "unknown"]
    created_at: str


class NewChatRequest(StrictModel):
    confirmed: bool = Field(strict=True)

    @field_validator("confirmed")
    @classmethod
    def require_confirmation(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Confirm that old context is archived and queued chat is cancelled.")
        return value


class NewChatAccepted(StrictModel):
    operation_id: UUID
    replayed: bool
    agent: AgentResponse


RunStatus = Literal[
    "queued",
    "starting",
    "running",
    "stopping",
    "completed",
    "completed_with_errors",
    "failed",
    "interrupted",
    "cancelled",
]
EventKind = Literal[
    "message_delta",
    "message_snapshot",
    "activity_started",
    "activity_completed",
    "error",
    "state",
]


class ChatMessageRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=16 * 1024)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > 16 * 1024:
            raise ValueError("Message must contain 1–16 KiB of text.")
        return value


class ChatRunResponse(StrictModel):
    source: Literal["chat", "job"]
    job_id: UUID | None
    scheduled_for: str | None
    coalesced_count: int
    id: UUID
    agent_id: UUID
    retry_of: UUID | None
    prompt: str
    status: RunStatus
    model: str
    chat_epoch: int
    created_at: str
    started_at: str | None
    finished_at: str | None
    termination_reason: str | None
    cleanup_state: Literal["not_required", "pending", "confirmed", "failed"]
    error_code: str | None
    error_message: str | None
    paused: bool


class ChatMessageAccepted(StrictModel):
    run_id: UUID
    status: RunStatus
    queued: bool
    paused: bool
    replayed: bool


class RetryAccepted(StrictModel):
    run_id: UUID
    status: RunStatus
    queued: bool
    paused: bool
    replayed: bool
    context_warning: bool


class RunEventResponse(StrictModel):
    sequence: int
    kind: EventKind
    sdk_item_id: str | None
    payload: dict[str, object]
    created_at: str


class RunDetailsResponse(ChatRunResponse):
    last_event_sequence: int


class ErrorResponse(StrictModel):
    code: str
    message: str
    request_id: str


def is_supported_model(model: str) -> bool:
    return model in SUPPORTED_MODELS
