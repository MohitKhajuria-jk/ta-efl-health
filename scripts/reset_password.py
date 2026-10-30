"""Reset one local account using hidden prompts, preserving other accounts and research data."""
import argparse
import getpass
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.configuration.settings import SettingsManager
from app.services.auth import AuthError, LocalAuth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", help="Existing local username")
    parser.add_argument("--settings", type=Path, default=Path(
        os.environ.get("TA_EFL_SETTINGS", ROOT / "config/settings.json")))
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Run this command in an interactive terminal so the password stays hidden.")
    manager = SettingsManager(args.settings)
    settings = manager.load()
    if manager.warnings:
        parser.error("The selected configuration is invalid. Correct it before resetting an account.")
    try:
        password = getpass.getpass("New password (8-128 characters): ")
        repeated = getpass.getpass("Confirm new password: ")
        if password != repeated:
            raise AuthError("Passwords do not match; no changes were made")
        backup = LocalAuth(settings.storage_path).reset_password(args.username, password)
    except (AuthError, OSError) as exc:
        print(f"Password reset failed: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled; no changes were made.", file=sys.stderr)
        return 1
    print(f"Password updated for {args.username}. Account backup: {backup}")
    print("Restart the Python server to sign out existing sessions, then sign in with the new password.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
