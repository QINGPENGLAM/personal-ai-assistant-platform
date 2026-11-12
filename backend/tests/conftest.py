from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.auth import AuthService
from backend.app.config import Settings
from backend.app.db import Database


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    runtime_root = tmp_path / "runtime"
    return Settings(
        runtime_root=runtime_root,
        database_path=runtime_root / "platform.sqlite",
        allowed_origin="http://127.0.0.1:5173",
        cookie_secure=False,
        platform_image="qpl-platform:test",
        agent_image="qpl-agent:test",
        auth_mode="auth_file",
        auth_file_source=None,
        toolkit_api_key=None,
        model_report_path=(
            runtime_root / "platform-state" / "model-report" / "report" / "verified-models.json"
        ),
        gateway_socket=runtime_root / "platform-state" / "gateway" / "gateway.sock",
        quota_helper_socket=runtime_root / "quota-helper.sock",
        agent_data_root=runtime_root / "agent-data",
        platform_state_root=runtime_root / "platform-state",
        auth_file_validation_path=None,
        platform_volume="qpl-platform-test",
        agent_volume="qpl-agent-test",
    )


@pytest.fixture
def database(settings: Settings) -> Database:
    database = Database(settings.database_path)
    database.initialize()
    return database


@pytest.fixture
def auth_service(database: Database, settings: Settings) -> AuthService:
    return AuthService(database, settings)
