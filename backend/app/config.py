from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_MODEL_HOSTS = frozenset(
    {
        "api.openai.com",
        "chatgpt.com",
        "auth.openai.com",
        "api.toolkit.umgpt.umich.edu",
    }
)
SUPPORTED_MODELS = ("gpt-6.1-sol", "gpt-6-luna")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be 'true' or 'false'")
    return normalized == "true"


@dataclass(frozen=True, slots=True)
class Settings:
    runtime_root: Path
    database_path: Path
    allowed_origin: str
    cookie_secure: bool
    platform_image: str
    agent_image: str
    auth_mode: str
    auth_file_source: Path | None
    toolkit_api_key: str | None = field(repr=False)
    model_report_path: Path
    gateway_socket: Path
    quota_helper_socket: Path
    agent_data_root: Path
    platform_state_root: Path
    auth_file_validation_path: Path | None
    platform_volume: str
    agent_volume: str
    allowed_model_hosts: frozenset[str] = ALLOWED_MODEL_HOSTS
    models: tuple[str, ...] = SUPPORTED_MODELS

    @property
    def model_report_key_path(self) -> Path:
        return self.model_report_path.parent.parent / "key" / "hmac.key"

    @classmethod
    def from_env(cls) -> Settings:
        runtime_root = Path(os.getenv("QPL_RUNTIME_ROOT", "./runtime")).expanduser()
        platform_state_root = Path(
            os.getenv("QPL_PLATFORM_STATE_ROOT", "/platform-state")
        ).expanduser()
        auth_mode = os.getenv("QPL_AUTH_MODE", "auth_file")
        if auth_mode not in {"auth_file", "toolkit"}:
            raise ValueError("QPL_AUTH_MODE must be 'auth_file' or 'toolkit'")

        auth_file_value = os.getenv("QPL_AUTH_FILE_SOURCE", "").strip()
        auth_file_validation_value = os.getenv("QPL_AUTH_FILE_VALIDATION", "").strip()
        toolkit_api_key = os.getenv("QPL_TOOLKIT_API_KEY") or None
        if auth_mode == "auth_file" and not auth_file_value:
            auth_file_source = None
        else:
            auth_file_source = Path(auth_file_value).expanduser() if auth_file_value else None
        if auth_mode == "auth_file" and not auth_file_validation_value:
            auth_file_validation_path = None
        else:
            auth_file_validation_path = (
                Path(auth_file_validation_value).expanduser()
                if auth_file_validation_value
                else None
            )
        if auth_mode == "toolkit" and not toolkit_api_key:
            toolkit_api_key = None

        allowed_origin = os.getenv("QPL_ALLOWED_ORIGIN", "http://127.0.0.1:5173").rstrip("/")
        if not allowed_origin.startswith(("http://", "https://")):
            raise ValueError("QPL_ALLOWED_ORIGIN must be an HTTP(S) origin")

        return cls(
            runtime_root=runtime_root,
            database_path=Path(
                os.getenv("QPL_DATABASE_PATH", str(runtime_root / "platform.sqlite"))
            ).expanduser(),
            allowed_origin=allowed_origin,
            cookie_secure=_bool_env("QPL_COOKIE_SECURE", False),
            platform_image=os.getenv("QPL_IMAGE_TAG", "qpl-platform:dev"),
            agent_image=os.getenv("QPL_AGENT_IMAGE_TAG", "qpl-agent:dev"),
            auth_mode=auth_mode,
            auth_file_source=auth_file_source,
            toolkit_api_key=toolkit_api_key,
            model_report_path=Path(
                os.getenv(
                    "QPL_MODEL_REPORT_PATH",
                    str(platform_state_root / "model-report" / "report" / "verified-models.json"),
                )
            ).expanduser(),
            gateway_socket=Path(
                os.getenv("QPL_GATEWAY_SOCKET", "/platform-state/gateway/gateway.sock")
            ).expanduser(),
            quota_helper_socket=Path(
                os.getenv("QPL_QUOTA_HELPER_SOCKET", "/run/qpl-quota/helper.sock")
            ).expanduser(),
            agent_data_root=Path(os.getenv("QPL_AGENT_DATA_ROOT", "/runtime")).expanduser(),
            platform_state_root=platform_state_root,
            auth_file_validation_path=auth_file_validation_path,
            platform_volume=os.getenv("QPL_PLATFORM_VOLUME", "qpl-platform-state"),
            agent_volume=os.getenv("QPL_AGENT_VOLUME", "qpl-agent-data"),
        )


settings = Settings.from_env()
