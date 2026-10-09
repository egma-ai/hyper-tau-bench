"""The Daytona launcher's ChatGPT token handling (tools/daytona)."""

import base64
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "daytona"))

import chatgpt_login  # noqa: E402
import keeper_helper  # noqa: E402
import launch  # noqa: E402


def fake_jwt(**claims) -> str:
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"e30.{body}.sig"


def test_access_is_cached_until_too_close_to_expiry(monkeypatch):
    fetches = []

    def fetch():
        fetches.append(1)
        return f"token-{len(fetches)}".encode(), time.time() + 240 * 3600

    monkeypatch.setattr(chatgpt_login, "fetch_access", fetch)
    access = launch.ChatGPTAccess(None)
    assert access.get() == b"token-1"
    assert access.get() == b"token-1"
    assert len(fetches) == 1

    # A new sandbox needs a token that outlives it: re-fetch before that.
    access.expires_at = time.time() + (launch.MIN_TOKEN_HOURS - 1) * 3600
    assert access.get() == b"token-2"


def test_missing_stored_login_fails_the_task_not_the_thread(monkeypatch):
    def fetch():
        raise SystemExit("No ChatGPT login is stored on Daytona yet.")

    monkeypatch.setattr(chatgpt_login, "fetch_access", fetch)
    with pytest.raises(RuntimeError, match="No ChatGPT login"):
        launch.ChatGPTAccess(None).get()


def test_local_login_near_expiry_is_refused(tmp_path):
    login = tmp_path / "auth.json"
    login.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": fake_jwt(exp=time.time() + 5 * 3600),
                    "refresh_token": "never-uploaded",
                    "account_id": "acct",
                }
            }
        )
    )
    with pytest.raises(RuntimeError, match="expires in"):
        launch.ChatGPTAccess(login).get()


def test_local_login_uploads_only_access_token_and_account(tmp_path):
    login = tmp_path / "auth.json"
    access_token = fake_jwt(exp=time.time() + 100 * 3600)
    login.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "never-uploaded",
                    "id_token": fake_jwt(),
                    "account_id": "acct",
                }
            }
        )
    )
    uploaded = json.loads(launch.ChatGPTAccess(login).get())
    assert uploaded == {"tokens": {"access_token": access_token, "account_id": "acct"}}


def test_keeper_describes_a_login_without_its_tokens():
    expires_at = int(time.time()) + 10 * 3600
    auth = {
        "last_refresh": "2026-10-09T04:46:43Z",
        "tokens": {
            "access_token": fake_jwt(exp=expires_at),
            "refresh_token": "secret-refresh",
            "id_token": fake_jwt(
                **{"https://api.openai.com/auth": {"chatgpt_plan_type": "pro"}}
            ),
            "account_id": "acct",
        },
    }
    status = keeper_helper.describe(auth)
    assert status["refreshable"] and status["plan"] == "pro"
    assert status["expires_at"] == expires_at and 9.9 <= status["hours_left"] <= 10
    assert "secret-refresh" not in json.dumps(status)
    assert keeper_helper.describe(None) == {"logged_in": False}
