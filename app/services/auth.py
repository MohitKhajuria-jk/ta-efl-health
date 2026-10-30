"""Local desktop login. No built-in or hard-coded credentials.

On first launch the user creates a local account. Passwords are stored only as
salted PBKDF2-HMAC-SHA256 hashes. This gates casual access to a research
workstation; it is not an enterprise identity system.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from experiments.store import atomic_write_text

ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


class AuthError(ValueError):
    pass


class LocalAuth:
    def __init__(self, storage_dir: Path) -> None:
        self.path = Path(storage_dir) / "users.json"

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            raise AuthError("User store is unreadable; use 'Forgot password?' to reset it")

    @staticmethod
    def _hash(password: str, salt: bytes) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS).hex()

    def has_users(self) -> bool:
        try:
            return bool(self._load())
        except AuthError:
            return True

    def create_user(self, username: str, password: str) -> None:
        if not USERNAME_RE.match(username or ""):
            raise AuthError("Username must be 3-32 characters: letters, digits, '.', '_' or '-'")
        if len(password or "") < 8:
            raise AuthError("Password must be at least 8 characters")
        users = self._load()
        if username in users:
            raise AuthError("That username already exists")
        salt = secrets.token_bytes(16)
        users[username] = {"salt": salt.hex(), "hash": self._hash(password, salt),
                           "iterations": str(ITERATIONS)}
        atomic_write_text(self.path, json.dumps(users, indent=2))

    def verify(self, username: str, password: str) -> bool:
        users = self._load()
        entry = users.get(username)
        if entry is None:
            self._hash(password or "", b"0" * 16)  # similar timing for unknown users
            return False
        expected = entry["hash"]
        actual = self._hash(password or "", bytes.fromhex(entry["salt"]))
        return hmac.compare_digest(expected, actual)

    def reset(self) -> None:
        """Delete all local accounts (experiments and models are not touched)."""
        if self.path.is_file():
            self.path.unlink()

    def reset_password(self, username: str, password: str) -> Path:
        """Local administrator recovery for one account; never expose over HTTP.

        Preserve the previous account store before atomically changing the hash.
        The caller must restart the web server to invalidate existing sessions.
        """
        if not 8 <= len(password or "") <= 128:
            raise AuthError("Password must be 8-128 characters")
        users = self._load()
        if username not in users:
            raise AuthError("That local account does not exist")
        if not isinstance(users[username], dict):
            raise AuthError("Account entry is unreadable; no changes were made")
        original = self.path.read_text(encoding="utf-8")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = self.path.parent / "account-backups" / f"users-{stamp}-{secrets.token_hex(4)}.json"
        atomic_write_text(backup, original)
        salt = secrets.token_bytes(16)
        users[username] = {**users[username], "salt": salt.hex(),
                           "hash": self._hash(password, salt), "iterations": str(ITERATIONS)}
        atomic_write_text(self.path, json.dumps(users, indent=2))
        return backup
