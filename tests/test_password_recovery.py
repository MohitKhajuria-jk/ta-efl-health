import json
import pytest
from app.services.auth import AuthError, LocalAuth


def test_recovery_updates_only_selected_account_and_keeps_backup(tmp_path):
    auth = LocalAuth(tmp_path)
    auth.create_user("alice", "old-test-password")
    auth.create_user("bob", "other-test-password")
    original = auth.path.read_text(encoding="utf-8")
    other = json.loads(original)["bob"]
    research = tmp_path / "experiments" / "preserved.json"
    research.parent.mkdir()
    research.write_text('{"unchanged":true}')
    backup = auth.reset_password("alice", "new-test-password")
    assert backup.read_text(encoding="utf-8") == original
    assert auth.verify("alice", "new-test-password")
    assert not auth.verify("alice", "old-test-password")
    assert auth.verify("bob", "other-test-password")
    assert json.loads(auth.path.read_text())["bob"] == other
    assert research.read_text() == '{"unchanged":true}'
    assert "new-test-password" not in auth.path.read_text()


@pytest.mark.parametrize("username,password", [
    ("missing", "valid-test-password"), ("alice", "short"), ("alice", "x" * 129),
])
def test_invalid_recovery_leaves_store_untouched(tmp_path, username, password):
    auth = LocalAuth(tmp_path)
    auth.create_user("alice", "old-test-password")
    original = auth.path.read_bytes()
    with pytest.raises(AuthError):
        auth.reset_password(username, password)
    assert auth.path.read_bytes() == original
    assert not (tmp_path / "account-backups").exists()
