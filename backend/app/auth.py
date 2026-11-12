from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from .config import Settings
from .db import Database

SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1_000
AUTH_ATTEMPTS_PER_MINUTE = 10
AUTH_WINDOW_SECONDS = 60.0


class AuthError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    id: str
    email: str
    csrf_token: str


class AttemptLimiter:
    def __init__(self) -> None:
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check_and_record(self, key: str) -> None:
        now = time.monotonic()
        cutoff = now - AUTH_WINDOW_SECONDS
        with self._lock:
            attempts = self._attempts[key]
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            if len(attempts) >= AUTH_ATTEMPTS_PER_MINUTE:
                raise AuthError(429, "rate_limited", "Too many authentication attempts.")
            attempts.append(now)


class AuthService:
    def __init__(self, database: Database, settings: Settings) -> None:
        self.database = database
        self.settings = settings
        self.password_hasher = PasswordHasher()
        self.attempt_limiter = AttemptLimiter()

    def register(self, email: str, password: str, client_ip: str) -> tuple[AuthenticatedUser, str]:
        self.attempt_limiter.check_and_record(client_ip)
        normalized_email = self._normalize_email(email)
        user_id = str(uuid.uuid4())
        raw_token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        now_ms = self._now_ms()
        password_hash = self.password_hasher.hash(password)

        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO users(id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, normalized_email, password_hash, now_ms),
                )
                connection.execute(
                    """
                    INSERT INTO sessions(token_hash, user_id, csrf_token, expires_at, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self._token_hash(raw_token),
                        user_id,
                        csrf_token,
                        now_ms + SESSION_TTL_MS,
                        now_ms,
                    ),
                )
        except sqlite3.IntegrityError as error:
            if "UNIQUE constraint failed" in str(error):
                raise AuthError(
                    409, "email_taken", "An account with that email already exists."
                ) from error
            raise

        return AuthenticatedUser(user_id, normalized_email, csrf_token), raw_token

    def login(self, email: str, password: str, client_ip: str) -> tuple[AuthenticatedUser, str]:
        self.attempt_limiter.check_and_record(client_ip)
        normalized_email = self._normalize_email(email)
        with self.database.read() as connection:
            user = connection.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ?",
                (normalized_email,),
            ).fetchone()
        if user is None or not self._verify_password(user["password_hash"], password):
            raise AuthError(401, "invalid_credentials", "Email or password is incorrect.")

        raw_token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        now_ms = self._now_ms()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sessions(token_hash, user_id, csrf_token, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self._token_hash(raw_token),
                    user["id"],
                    csrf_token,
                    now_ms + SESSION_TTL_MS,
                    now_ms,
                ),
            )
        return AuthenticatedUser(user["id"], user["email"], csrf_token), raw_token

    def authenticate(self, raw_token: str | None) -> AuthenticatedUser:
        if not raw_token:
            raise AuthError(401, "login_required", "Please log in.")
        now_ms = self._now_ms()
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT users.id, users.email, sessions.csrf_token, sessions.expires_at
                FROM sessions JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ?
                """,
                (self._token_hash(raw_token),),
            ).fetchone()
        if row is None or row["expires_at"] <= now_ms:
            if row is not None:
                with self.database.transaction() as connection:
                    connection.execute(
                        "DELETE FROM sessions WHERE token_hash = ?",
                        (self._token_hash(raw_token),),
                    )
            raise AuthError(401, "login_required", "Please log in.")
        return AuthenticatedUser(row["id"], row["email"], row["csrf_token"])

    def require_csrf(self, user: AuthenticatedUser, supplied: str | None) -> None:
        if not supplied or not secrets.compare_digest(user.csrf_token, supplied):
            raise AuthError(403, "csrf_failed", "The request could not be verified.")

    def require_origin(self, origin: str | None) -> None:
        if origin != self.settings.allowed_origin:
            raise AuthError(403, "origin_rejected", "The request origin is not allowed.")

    def logout(self, raw_token: str | None) -> None:
        if not raw_token:
            return
        with self.database.transaction() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?",
                (self._token_hash(raw_token),),
            )

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().lower()

    @staticmethod
    def _token_hash(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    @staticmethod
    def _now_ms() -> int:
        return time.time_ns() // 1_000_000

    def _verify_password(self, password_hash: str, password: str) -> bool:
        try:
            return self.password_hasher.verify(password_hash, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
