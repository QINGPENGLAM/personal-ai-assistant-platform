from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response, status

from ..auth import AuthError, AuthService
from ..config import Settings
from ..schemas import AuthResponse, LoginRequest, RegisterRequest, UserResponse

router = APIRouter(prefix="/api/auth", tags=["authentication"])
COOKIE_NAME = "qpl_session"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def _request_id(request: Request) -> str:
    value = request.headers.get("x-request-id")
    if value is None:
        raise AuthError(422, "invalid_request_id", "X-Request-ID is required.")
    try:
        return str(UUID(value))
    except ValueError as error:
        raise AuthError(
            422, "invalid_request_id", "X-Request-ID must be a UUID."
        ) from error


def _client_ip(request: Request) -> str:
    if request.client is None:
        return "unknown"
    return request.client.host


def _set_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
        secure=settings.cookie_secure,
        httponly=True,
        samesite="lax",
    )


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(request: Request, response: Response, body: RegisterRequest) -> AuthResponse:
    _request_id(request)
    auth: AuthService = request.app.state.auth
    settings: Settings = request.app.state.settings
    auth.require_origin(request.headers.get("origin"))
    user, token = auth.register(body.email, body.password, _client_ip(request))
    _set_cookie(response, token, settings)
    return AuthResponse(
        user=UserResponse(id=user.id, email=user.email),
        csrf_token=user.csrf_token,
    )


@router.post("/login", response_model=AuthResponse)
def login(request: Request, response: Response, body: LoginRequest) -> AuthResponse:
    _request_id(request)
    auth: AuthService = request.app.state.auth
    settings: Settings = request.app.state.settings
    auth.require_origin(request.headers.get("origin"))
    user, token = auth.login(body.email, body.password, _client_ip(request))
    _set_cookie(response, token, settings)
    return AuthResponse(
        user=UserResponse(id=user.id, email=user.email),
        csrf_token=user.csrf_token,
    )


@router.get("/session", response_model=AuthResponse)
def session(request: Request) -> AuthResponse:
    auth: AuthService = request.app.state.auth
    user = auth.authenticate(request.cookies.get(COOKIE_NAME))
    return AuthResponse(
        user=UserResponse(id=user.id, email=user.email),
        csrf_token=user.csrf_token,
    )


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    _request_id(request)
    auth: AuthService = request.app.state.auth
    user = auth.authenticate(request.cookies.get(COOKIE_NAME))
    auth.require_origin(request.headers.get("origin"))
    auth.require_csrf(user, request.headers.get("x-csrf-token"))
    auth.logout(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        secure=auth.settings.cookie_secure,
        httponly=True,
        samesite="lax",
    )
    return {"logged_out": True}
