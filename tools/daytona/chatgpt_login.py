"""Sign in to ChatGPT once for every Hyper-tau Daytona run.

The Codex login used with `launch.py --developer-auth chatgpt` lives in one
small Daytona sandbox, the login keeper. Launchers on any machine fetch an
access-token file from it, and only the keeper ever refreshes the login (with
Codex's own refresh flow), so parallel runs and later runs never invalidate
each other. Refresh tokens are single-use: a login that two places refresh
logs both of them out.

    # one time: sign in inside the keeper (prints a device code to approve)
    uv run --with "daytona>=0.210.0" python tools/daytona/chatgpt_login.py login
    # or adopt an existing dedicated login (its local copy becomes access-only)
    uv run --with "daytona>=0.210.0" python tools/daytona/chatgpt_login.py \\
        store --auth ~/.hypertau-codex/auth.json
    # expiry and plan usage, without spending a model turn
    uv run --with "daytona>=0.210.0" python tools/daytona/chatgpt_login.py status --usage

Tokens are never printed. The run sandboxes only ever get the access token.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = HERE / "keeper_helper.py"
DOCKERFILE = HERE.parents[1] / "docker" / "hyper-construction" / "Dockerfile"
REMOTE_ROOT = "/root/keeper"
REMOTE_HELPER = f"{REMOTE_ROOT}/keeper_helper.py"
KEEPER_NAME = "hypertau-chatgpt-login"
KEEPER_LABELS = {"hypertau-role": "chatgpt-login"}
KEEPER_IMAGE = "python:3.12-slim-bookworm"
# The keeper refreshes the stored login once it has less than this left, so
# every fetched token stays valid for a full task sandbox lifetime.
REFRESH_BELOW_HOURS = 48.0


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def utc(timestamp: float | None) -> str:
    if not timestamp:
        return "?"
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M")


def codex_version() -> str:
    """The Codex version the construction image ships."""
    match = re.search(r"^ARG CODEX_VERSION=(\S+)", DOCKERFILE.read_text(), re.M)
    if not match:
        raise SystemExit(f"no CODEX_VERSION in {DOCKERFILE}")
    return match.group(1)


# --- Keeper sandbox -----------------------------------------------------------


def find_keeper(daytona):
    from daytona import ListSandboxesQuery

    return next(iter(daytona.list(ListSandboxesQuery(labels=KEEPER_LABELS))), None)


def ensure_started(sandbox) -> None:
    """Start a stopped or archived keeper; wait out stopping/archiving."""
    for attempt in range(1, 7):
        sandbox.refresh_data()
        state = str(getattr(sandbox.state, "value", sandbox.state)).lower()
        if state == "started":
            return
        try:
            log(f"starting the login keeper ({state})")
            sandbox.start(timeout=600)
            return
        except Exception:  # noqa: BLE001 - e.g. still stopping; retry
            if attempt == 6:
                raise
            time.sleep(20)


def open_keeper(daytona, create: bool = False):
    """The keeper sandbox, started, with the current helper and Codex."""
    from daytona import CreateSandboxFromImageParams, Image, Resources

    sandbox = find_keeper(daytona)
    if sandbox is None:
        if not create:
            raise SystemExit(
                "No ChatGPT login is stored on Daytona yet. Sign in once with:\n"
                '  uv run --with "daytona>=0.210.0" python '
                "tools/daytona/chatgpt_login.py login"
            )
        log("creating the login keeper sandbox")
        sandbox = daytona.create(
            CreateSandboxFromImageParams(
                name=KEEPER_NAME,
                image=Image.base(KEEPER_IMAGE),
                resources=Resources(cpu=1, memory=1, disk=3),
                labels=KEEPER_LABELS,
                auto_stop_interval=15,
                auto_delete_interval=-1,
            ),
            timeout=900,
        )
    ensure_started(sandbox)
    sandbox.process.exec(f"mkdir -p {REMOTE_ROOT} && chmod 700 {REMOTE_ROOT}")
    sandbox.fs.upload_file(HELPER.read_bytes(), REMOTE_HELPER)
    installed = helper(
        sandbox, "install", "--codex-version", codex_version(), timeout=900
    )
    if installed.get("installed"):
        log(f"installed Codex {installed['codex']} in the login keeper")
    return sandbox


def helper(sandbox, *args: str, timeout: int = 180) -> dict:
    """Run keeper_helper.py in the keeper; its JSON reply (errors raise)."""
    response = sandbox.process.exec(
        f"python3 {REMOTE_HELPER} {' '.join(args)}", timeout=timeout
    )
    lines = (response.result or "").strip().splitlines()
    try:
        reply = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        reply = {"error": "unexpected keeper output", "detail": "\n".join(lines[-5:])}
    if response.exit_code != 0 or "error" in reply:
        detail = reply.get("detail") or reply.get("log_tail") or ""
        raise RuntimeError(
            f"login keeper: {reply.get('error') or f'exit {response.exit_code}'}"
            + (f" ({detail})" if detail else "")
        )
    return reply


def describe(status: dict) -> str:
    if not status.get("logged_in"):
        return "no login stored"
    text = (
        f"plan {status.get('plan')}, access token valid until "
        f"{utc(status.get('expires_at'))} UTC ({status.get('hours_left')} h), "
        f"last refreshed {status.get('last_refresh')}"
    )
    for name, window in (status.get("usage") or {}).items():
        days = (window.get("windowDurationMins") or 0) / 60 / 24
        text += (
            f"\n  {name} usage window: {window.get('usedPercent')}% used of "
            f"{days:.1f} days, resets {utc(window.get('resetsAt'))} UTC"
        )
    return text


# --- What launch.py uses ------------------------------------------------------


def fetch_access(refresh_below_hours: float = REFRESH_BELOW_HOURS):
    """(access-file bytes, expiry) from the stored login, refreshed when due.

    The bytes are the {"tokens": {"access_token", "account_id"}} file the
    sealed runner's model gateway reads; the refresh token never leaves the
    keeper.
    """
    from daytona import Daytona

    sandbox = open_keeper(Daytona())
    remote = f"{REMOTE_ROOT}/access-{secrets.token_hex(8)}.json"
    try:
        status = helper(
            sandbox,
            "access",
            "--refresh-below",
            str(refresh_below_hours),
            "--out",
            remote,
            timeout=300,
        )
        payload = sandbox.fs.download_file(remote)
    finally:
        sandbox.process.exec(f"rm -f {remote}")
    if status.get("refreshed"):
        log("the login keeper refreshed the stored ChatGPT login")
    return payload, float(status["expires_at"])


# --- Commands -----------------------------------------------------------------


def cmd_login(daytona, args) -> None:
    sandbox = open_keeper(daytona, create=True)
    current = helper(sandbox, "status")
    if current.get("refreshable") and not args.replace:
        raise SystemExit(
            f"A login is already stored ({describe(current)}).\n"
            "Pass --replace to sign in again."
        )
    started = helper(sandbox, "login-start", timeout=120)
    if not started.get("code"):
        raise SystemExit("Codex did not print a device code; try again")
    print(
        f"\nSIGN IN: open {started['url']} and enter {started['code']}"
        f" (expires {utc(started['expires_at'])} UTC)\n",
        flush=True,
    )
    try:
        while time.time() < started["expires_at"] + 60:
            result = helper(sandbox, "login-wait", "--timeout", "50", timeout=120)
            if not result.get("waiting"):
                log(f"signed in: {describe(result)}")
                return
        raise SystemExit("The device code expired before it was approved")
    except BaseException:
        helper(sandbox, "login-cancel")
        raise


def cmd_store(daytona, args) -> None:
    source = args.auth.expanduser()
    auth = json.loads(source.read_text())
    if not (auth.get("tokens") or {}).get("refresh_token"):
        raise SystemExit(f"{source} has no refresh token to store")
    sandbox = open_keeper(daytona, create=True)
    sandbox.fs.upload_file(source.read_bytes(), f"{REMOTE_ROOT}/incoming.json")
    helper(sandbox, "store", *(["--replace"] if args.replace else []))
    # Prove the stored login works against ChatGPT before dropping the local copy.
    status = helper(sandbox, "status", "--usage")
    if not status.get("usage"):
        raise SystemExit("stored, but the usage check got no reply from ChatGPT")
    if not args.keep_local:
        # From now on only the keeper may refresh this login.
        auth["tokens"].pop("refresh_token", None)
        tmp = source.with_suffix(".tmp")
        tmp.write_text(json.dumps(auth, indent=2))
        tmp.chmod(0o600)
        tmp.replace(source)
        log(f"{source} is now access-only; its refresh token lives in the keeper")
    log(f"stored: {describe(status)}")


def cmd_status(daytona, args) -> None:
    sandbox = open_keeper(daytona)
    status = helper(sandbox, "status", *(["--usage"] if args.usage else []))
    log(f"login keeper {sandbox.id}: {describe(status)}")


def cmd_refresh(daytona, args) -> None:
    sandbox = open_keeper(daytona)
    status = helper(sandbox, "refresh", timeout=300)
    log(f"refreshed: {describe(status)}")


def cmd_forget(daytona, args) -> None:
    sandbox = find_keeper(daytona)
    if sandbox is None:
        log("no login keeper on this Daytona org")
        return
    if not args.yes:
        raise SystemExit("This deletes the stored ChatGPT login; pass --yes")
    daytona.delete(sandbox)
    log(
        "login keeper deleted. To also revoke the login itself, sign out of "
        "all sessions in ChatGPT's security settings."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login", help="device-code sign-in in the keeper")
    login.add_argument("--replace", action="store_true")
    store = commands.add_parser("store", help="move an existing login in")
    store.add_argument("--auth", type=Path, required=True, help="Codex auth.json")
    store.add_argument("--replace", action="store_true")
    store.add_argument(
        "--keep-local",
        action="store_true",
        help="keep the local refresh token (only if nothing will refresh it)",
    )
    status = commands.add_parser("status", help="expiry (and --usage)")
    status.add_argument("--usage", action="store_true")
    commands.add_parser("refresh", help="refresh the stored login now")
    forget = commands.add_parser("forget", help="delete the keeper and its login")
    forget.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    from daytona import Daytona

    handler = {
        "login": cmd_login,
        "store": cmd_store,
        "status": cmd_status,
        "refresh": cmd_refresh,
        "forget": cmd_forget,
    }[args.command]
    handler(Daytona(), args)


if __name__ == "__main__":
    main()
