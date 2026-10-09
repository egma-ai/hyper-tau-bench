"""Runs inside the ChatGPT login keeper sandbox (see chatgpt_login.py).

The keeper holds the one refresh-capable Codex login that every Hyper-tau
Daytona run uses. Refresh tokens are single-use, so this sandbox is the only
place that ever refreshes it, through Codex's own refresh flow (app-server
`account/read` with `refreshToken`), under a file lock. Each command prints one
JSON object on stdout; tokens are never printed. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import json
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path("/root/keeper")
HOME = ROOT / "home"  # CODEX_HOME of the stored login
STAGING = ROOT / "staging"  # CODEX_HOME of a device login awaiting approval
CODEX = ROOT / "bin" / "codex"
INCOMING = ROOT / "incoming.json"
LOCK = ROOT / "lock"
LOGIN_LOG = ROOT / "login.log"
LOGIN_PID = ROOT / "login.pid"
CODEX_CONFIG = 'cli_auth_credentials_store = "file"\n'
NPM_TARBALL = (
    "https://registry.npmjs.org/@openai/codex/-/codex-{version}-linux-{arch}.tgz"
)
DEVICE_CODE = re.compile(r"\b[A-Z0-9]{4}-[A-Z0-9]{4,6}\b")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def emit(payload: dict, code: int = 0) -> None:
    print(json.dumps(payload))
    sys.exit(code)


def fail(message: str, **extra) -> None:
    emit({"error": message, **extra}, code=1)


def claims(jwt: str) -> dict:
    part = jwt.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def load(home: Path) -> dict | None:
    path = home / "auth.json"
    return json.loads(path.read_text()) if path.exists() else None


def describe(auth: dict | None) -> dict:
    tokens = (auth or {}).get("tokens") or {}
    if not tokens.get("access_token"):
        return {"logged_in": False}
    expires_at = claims(tokens["access_token"]).get("exp", 0)
    plan = None
    if tokens.get("id_token"):
        profile = claims(tokens["id_token"]).get("https://api.openai.com/auth", {})
        plan = profile.get("chatgpt_plan_type")
    return {
        "logged_in": True,
        "refreshable": bool(tokens.get("refresh_token")),
        "expires_at": expires_at,
        "hours_left": round((expires_at - time.time()) / 3600, 1),
        "last_refresh": auth.get("last_refresh"),
        "plan": plan,
    }


def write_private(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.chmod(0o600)
    tmp.replace(path)


def prepare_home(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    home.chmod(0o700)
    (home / "config.toml").write_text(CODEX_CONFIG)


@contextlib.contextmanager
def locked():
    """Serialize everything that may refresh the login, across launchers."""
    with open(LOCK, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def app_server(home: Path, requests: list[dict], timeout: float = 120) -> dict:
    """Send requests to `codex app-server` on the given login; {id: reply}."""
    process = subprocess.Popen(
        [str(CODEX), "app-server", "--listen", "stdio://"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env={**os.environ, "CODEX_HOME": str(home)},
    )
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    def send(message: dict) -> None:
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    replies: dict = {}
    try:
        client = {"name": "hypertau_login_keeper", "version": "1"}
        send({"method": "initialize", "id": 0, "params": {"clientInfo": client}})
        send({"method": "initialized", "params": {}})
        for request in requests:
            send(request)
        wanted = {request["id"] for request in requests}
        deadline = time.monotonic() + timeout
        while wanted - replies.keys():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                break
            if line is None:
                break
            message = json.loads(line)
            if message.get("id") in wanted:
                replies[message["id"]] = message
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
    return replies


def refresh(home: Path) -> None:
    """Have Codex refresh the login now; it rotates and saves the tokens."""
    before = describe(load(home)).get("expires_at", 0)
    request = {"method": "account/read", "id": 1, "params": {"refreshToken": True}}
    reply = app_server(home, [request]).get(1) or {}
    after = describe(load(home)).get("expires_at", 0)
    if "result" not in reply or after <= before:
        fail(
            "login refresh failed; sign in again with chatgpt_login.py login",
            detail=reply.get("error") or "no reply from codex app-server",
        )


# --- Commands -----------------------------------------------------------------


def cmd_install(args) -> None:
    """Directories, Codex config and the Codex binary (once per version)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    ROOT.chmod(0o700)
    prepare_home(HOME)
    version_file = CODEX.with_suffix(".version")
    if CODEX.exists() and version_file.exists():
        if version_file.read_text().strip() == args.codex_version:
            emit({"codex": args.codex_version, "installed": False})
    arch = {"x86_64": "x64", "aarch64": "arm64"}[platform.machine()]
    triple = {"x64": "x86_64", "arm64": "aarch64"}[arch] + "-unknown-linux-musl"
    member = f"package/vendor/{triple}/bin/codex"
    CODEX.parent.mkdir(parents=True, exist_ok=True)
    url = NPM_TARBALL.format(version=args.codex_version, arch=arch)
    with urllib.request.urlopen(url, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for entry in archive:
                if entry.name == member:
                    with open(CODEX.with_suffix(".tmp"), "wb") as out:
                        shutil.copyfileobj(archive.extractfile(entry), out)
                    break
            else:
                fail(f"{member} not found in {url}")
    CODEX.with_suffix(".tmp").chmod(0o755)
    CODEX.with_suffix(".tmp").replace(CODEX)
    version_file.write_text(args.codex_version)
    emit({"codex": args.codex_version, "installed": True})


def cmd_status(args) -> None:
    status = describe(load(HOME))
    if args.usage and status["logged_in"]:
        request = {
            "method": "account/rateLimits/read",
            "id": 1,
            "params": {"excludeResetCreditDetails": True},
        }
        with locked():  # Codex may refresh an old login when it starts
            reply = app_server(HOME, [request]).get(1) or {}
        limits = (reply.get("result") or {}).get("rateLimits") or {}
        status["usage"] = {
            name: limits[name] for name in ("primary", "secondary") if limits.get(name)
        }
        status = {**describe(load(HOME)), "usage": status["usage"]}
    emit(status)


def cmd_store(args) -> None:
    """Adopt an uploaded Codex login (INCOMING) as the stored login."""
    try:
        incoming = INCOMING.read_text()
    finally:
        INCOMING.unlink(missing_ok=True)
    status = describe(json.loads(incoming))
    if not status.get("refreshable"):
        fail("that file is not a refreshable Codex ChatGPT login")
    with locked():
        if describe(load(HOME)).get("refreshable") and not args.replace:
            fail("a login is already stored; pass --replace to overwrite it")
        prepare_home(HOME)
        write_private(HOME / "auth.json", incoming)
    emit(describe(load(HOME)))


def stored_login() -> dict:
    status = describe(load(HOME))
    if not status.get("refreshable"):
        fail("no stored login; run chatgpt_login.py login (or store)")
    return status


def cmd_access(args) -> None:
    """Write --out (access token + account id), refreshing first if due."""
    with locked():
        refreshed = stored_login()["hours_left"] < args.refresh_below
        if refreshed:
            refresh(HOME)
        auth = load(HOME)
        tokens = auth["tokens"]
        access = {
            "tokens": {
                "access_token": tokens["access_token"],
                "account_id": tokens.get("account_id"),
            }
        }
        write_private(Path(args.out), json.dumps(access))
    emit({**describe(auth), "refreshed": refreshed})


def cmd_refresh(args) -> None:
    with locked():
        stored_login()
        refresh(HOME)
    emit(describe(load(HOME)))


def login_running() -> bool:
    try:
        os.kill(int(LOGIN_PID.read_text()), 0)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False
    return True


def login_code() -> dict:
    text = ANSI.sub("", LOGIN_LOG.read_text()) if LOGIN_LOG.exists() else ""
    url = next((w for w in text.split() if w.startswith("https://")), None)
    code = DEVICE_CODE.search(text)
    return {"url": url, "code": code.group(0) if code else None}


def cmd_login_start(args) -> None:
    """Start a device login into STAGING; login-wait adopts it once approved."""
    if not login_running():
        shutil.rmtree(STAGING, ignore_errors=True)
        prepare_home(STAGING)
        with open(LOGIN_LOG, "w") as log:
            process = subprocess.Popen(
                [str(CODEX), "login", "--device-auth"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CODEX_HOME": str(STAGING)},
                start_new_session=True,
            )
        LOGIN_PID.write_text(str(process.pid))
        LOGIN_PID.with_suffix(".started").write_text(str(time.time()))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not login_code()["code"]:
        time.sleep(1)
    started = float(LOGIN_PID.with_suffix(".started").read_text())
    emit({**login_code(), "expires_at": started + 15 * 60})


def cmd_login_wait(args) -> None:
    deadline = time.monotonic() + args.timeout
    while login_running() and time.monotonic() < deadline:
        time.sleep(2)
    if login_running():
        emit({"waiting": True, **login_code()})
    staged = describe(load(STAGING))
    if not staged.get("refreshable"):
        tail = ANSI.sub("", LOGIN_LOG.read_text())[-400:] if LOGIN_LOG.exists() else ""
        fail("device login did not complete", log_tail=tail)
    with locked():
        prepare_home(HOME)
        write_private(HOME / "auth.json", (STAGING / "auth.json").read_text())
    shutil.rmtree(STAGING, ignore_errors=True)
    emit(describe(load(HOME)))


def cmd_login_cancel(args) -> None:
    if login_running():
        os.killpg(int(LOGIN_PID.read_text()), signal.SIGTERM)
    shutil.rmtree(STAGING, ignore_errors=True)
    emit({"cancelled": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install")
    install.add_argument("--codex-version", required=True)
    status = commands.add_parser("status")
    status.add_argument("--usage", action="store_true")
    store = commands.add_parser("store")
    store.add_argument("--replace", action="store_true")
    access = commands.add_parser("access")
    access.add_argument("--refresh-below", type=float, default=48.0, help="hours")
    access.add_argument("--out", required=True)
    commands.add_parser("refresh")
    commands.add_parser("login-start")
    wait = commands.add_parser("login-wait")
    wait.add_argument("--timeout", type=float, default=50.0)
    commands.add_parser("login-cancel")
    args = parser.parse_args()
    {
        "install": cmd_install,
        "status": cmd_status,
        "store": cmd_store,
        "access": cmd_access,
        "refresh": cmd_refresh,
        "login-start": cmd_login_start,
        "login-wait": cmd_login_wait,
        "login-cancel": cmd_login_cancel,
    }[args.command](args)


if __name__ == "__main__":
    main()
