"""Report a ChatGPT plan's Codex usage through the Hyper-tau model gateway.

Sends one tiny low-effort turn (Codex app-server -> the repo's model gateway in
chatgpt mode -> ChatGPT) and prints the plan's rate-limit window as Codex sees
it, plus the login's access-token expiry. Run it before and after a batch to
see how much of the allowance the batch used. Never prints tokens.

    uv run python tools/daytona/usage_probe.py \\
        --chatgpt-auth ~/.hypertau-codex/auth.json --codex "$(which codex)"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from tau2.hyper.sandbox.model_gateway import MODEL_GATEWAY_PORT


def utc(timestamp: float | None) -> str:
    if not timestamp:
        return "?"
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M")


def rpc(process: subprocess.Popen, message: dict) -> None:
    assert process.stdin is not None
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chatgpt-auth", type=Path, required=True)
    parser.add_argument("--codex", default="codex", help="codex binary (>= 0.159)")
    parser.add_argument("--model", default="gpt-6.1-sol")
    args = parser.parse_args()

    auth_file = args.chatgpt_auth.expanduser().resolve()
    access_token = json.loads(auth_file.read_text())["tokens"]["access_token"]
    claims_b64 = access_token.split(".")[1]
    claims = json.loads(
        base64.urlsafe_b64decode(claims_b64 + "=" * (-len(claims_b64) % 4))
    )
    print(f"access token expires: {utc(claims.get('exp'))} UTC")

    token = secrets.token_urlsafe(24)
    gateway = subprocess.Popen(
        [sys.executable, "-m", "tau2.hyper.sandbox.model_gateway"],
        env={
            **os.environ,
            "TAU2_MODEL_GATEWAY_PROVIDER": "chatgpt",
            "TAU2_MODEL_GATEWAY_MODEL": args.model,
            "TAU2_MODEL_GATEWAY_TOKEN": token,
            "TAU2_MODEL_GATEWAY_UPSTREAM_KEY": "",
            "TAU2_MODEL_GATEWAY_EXPIRES_AT": str(time.time() + 600),
            "TAU2_MODEL_GATEWAY_WIRE": "openai",
            "TAU2_MODEL_GATEWAY_AUTH_FILE": str(auth_file),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as home:
        codex_home = Path(home) / ".codex"
        codex_home.mkdir()
        (codex_home / "config.toml").write_text(
            f'model = "{args.model}"\n'
            'model_provider = "gw"\n'
            'model_reasoning_effort = "low"\n'
            'approval_policy = "never"\n'
            'sandbox_mode = "read-only"\n'
            "[model_providers.gw]\n"
            'name = "usage probe"\n'
            f'base_url = "http://127.0.0.1:{MODEL_GATEWAY_PORT}/chatgpt/v1"\n'
            'env_key = "PROBE_GATEWAY_TOKEN"\n'
            'wire_api = "responses"\n'
            "[skills.bundled]\n"
            "enabled = false\n"
        )
        time.sleep(2)
        process = subprocess.Popen(
            [args.codex, "app-server", "--listen", "stdio://"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ, "HOME": home, "PROBE_GATEWAY_TOKEN": token},
        )
        limits, status = None, None
        try:
            rpc(
                process,
                {
                    "method": "initialize",
                    "id": 0,
                    "params": {"clientInfo": {"name": "usage_probe", "version": "1"}},
                },
            )
            rpc(process, {"method": "initialized", "params": {}})
            rpc(
                process,
                {
                    "method": "thread/start",
                    "id": 1,
                    "params": {
                        "model": args.model,
                        "cwd": home,
                        "approvalPolicy": "never",
                    },
                },
            )
            deadline = time.monotonic() + 180
            assert process.stdout is not None
            while time.monotonic() < deadline:
                line = process.stdout.readline()
                if not line:
                    break
                message = json.loads(line)
                if message.get("id") == 1 and "result" in message:
                    thread_id = message["result"]["thread"]["id"]
                    rpc(
                        process,
                        {
                            "method": "turn/start",
                            "id": 2,
                            "params": {
                                "threadId": thread_id,
                                "input": [{"type": "text", "text": "Reply with OK."}],
                            },
                        },
                    )
                elif "id" in message and "method" in message:
                    rpc(
                        process,
                        {"id": message["id"], "result": {"decision": "decline"}},
                    )
                elif message.get("method") == "account/rateLimits/updated":
                    limits = message["params"]["rateLimits"]
                elif message.get("method") == "turn/completed":
                    status = message["params"]["turn"].get("status")
                    break
        finally:
            process.terminate()
            gateway.terminate()
            # Codex writes session files while it shuts down; let it finish
            # before the temporary home is removed.
            process.wait(timeout=15)

    print(f"probe turn: {status}")
    if not limits:
        sys.exit("no rate-limit report received")
    for name in ("primary", "secondary"):
        window = limits.get(name)
        if window:
            print(
                f"{name} window: {window.get('usedPercent')}% used of "
                f"{window.get('windowDurationMins', 0) / 60 / 24:.1f} days, "
                f"resets {utc(window.get('resetsAt'))} UTC"
            )
    credits = limits.get("credits") or {}
    print(
        f"credits: has={credits.get('hasCredits')} "
        f"balance={credits.get('balance')} unlimited={credits.get('unlimited')}"
    )


if __name__ == "__main__":
    main()
