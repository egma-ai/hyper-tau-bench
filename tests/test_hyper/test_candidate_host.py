"""Tests for serving sealed candidate sessions over TCP."""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

from tau2.hyper.sandbox.candidate_host import (
    CANDIDATE_BOOTSTRAP,
    CandidateHost,
    connect_candidate_session,
    parse_endpoint,
)
from tau2.hyper.sandbox.sealed_runner import (
    CandidateProcess,
    SealedRunnerConfig,
)

# Speaks the candidate_server line protocol without importing a kit:
# "configure" and "echo" answer directly, "broker" round-trips a model
# request through the trusted side first.
FAKE_CANDIDATE = textwrap.dedent(
    """
    import json, sys
    RPC = "__TAU2_CANDIDATE_RPC__"
    for raw in sys.stdin:
        request = json.loads(raw)
        method = request["method"]
        if method == "broker":
            print("__TAU2_MODEL_REQUEST__" + json.dumps({"id": "m1", "model": "x"}),
                  flush=True)
            brokered = json.loads(sys.stdin.readline())
            result = {"brokered": brokered}
        elif method == "crash":
            print("fatal candidate error", file=sys.stderr, flush=True)
            sys.exit(7)
        else:
            result = {"method": method, "payload": request["payload"]}
        print("noise that is not an RPC frame", flush=True)
        print(RPC + json.dumps({"id": request["id"], "ok": True, "result": result}),
              flush=True)
    """
)


@pytest.fixture
def serve_host(tmp_path):
    servers = []

    def start(command, **kwargs) -> str:
        kwargs.setdefault("isolate", False)
        kwargs.setdefault("session_root", tmp_path / "sessions")
        kwargs.setdefault("workspace", tmp_path)
        host = CandidateHost(command=command, **kwargs)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        thread = threading.Thread(
            target=host.serve_forever, args=("127.0.0.1", port), daemon=True
        )
        thread.start()
        servers.append(thread)
        # Wait until it listens; a bare connect-close starts no session.
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        return f"127.0.0.1:{port}"

    return start


def _read_line(session, timeout=10.0) -> str:
    result = {}

    def target():
        result["line"] = next(iter(session.stdout), None)

    reader = threading.Thread(target=target, daemon=True)
    reader.start()
    reader.join(timeout)
    assert "line" in result, "timed out waiting for candidate output"
    return result["line"]


def test_parse_endpoint_accepts_host_port_forms():
    assert parse_endpoint("candidate:7000") == ("candidate", 7000)
    assert parse_endpoint("tcp://candidate:7000/") == ("candidate", 7000)
    assert parse_endpoint("[::1]:7000") == ("::1", 7000)
    with pytest.raises(ValueError):
        parse_endpoint("candidate")


def test_bootstrap_matches_docker_run_path(tmp_path):
    process = CandidateProcess.__new__(CandidateProcess)
    process.config = SealedRunnerConfig(kit_path=tmp_path, domain="demo")
    assert process._build_command()[-1] == CANDIDATE_BOOTSTRAP


def test_session_relays_both_directions_and_exit_status(serve_host, tmp_path):
    script = (
        "import sys\n"
        "for line in sys.stdin:\n"
        "    print('echo:' + line.strip(), flush=True)\n"
        "print('bye', file=sys.stderr, flush=True)\n"
        "sys.exit(3)\n"
    )
    endpoint = serve_host([sys.executable, "-c", script])
    session = connect_candidate_session(endpoint)
    session.stdin.write("hello\n")
    session.stdin.flush()
    assert _read_line(session) == "echo:hello\n"
    session.stdin.close()
    assert session.wait(timeout=10) == 3
    assert list(session.stdout) == []
    assert "bye\n" in list(session.stderr)


def test_connect_and_close_starts_no_session(serve_host, tmp_path):
    marker = tmp_path / "started"
    endpoint = serve_host(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    )
    host, port = parse_endpoint(endpoint)
    for _ in range(3):
        socket.create_connection((host, port), timeout=5).close()
    time.sleep(0.5)
    assert not marker.exists()


def test_client_hangup_kills_a_candidate_that_ignores_stdin(serve_host, tmp_path):
    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen(['sleep', '600'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "print('ready', flush=True)\n"
        "time.sleep(600)\n"
    )
    endpoint = serve_host([sys.executable, "-c", script], exit_grace_seconds=0.5)
    session = connect_candidate_session(endpoint)
    session.stdin.write("go\n")
    assert _read_line(session) == "ready\n"
    child_pid = int(pid_file.read_text())

    session.terminate()
    assert session.wait(timeout=10) == -1
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _alive(child_pid):
        time.sleep(0.1)
    assert not _alive(child_pid)


def test_sealed_runner_drives_a_remote_candidate(serve_host, tmp_path):
    endpoint = serve_host([sys.executable, "-c", FAKE_CANDIDATE])
    runner = CandidateProcess(
        SealedRunnerConfig(
            kit_path=tmp_path, domain="demo", candidate_endpoint=endpoint
        )
    )
    try:
        assert runner.request("echo", {"value": 1}) == {
            "method": "echo",
            "payload": {"value": 1},
        }
        # Model requests travel back over the same connection; with no
        # gateway configured the trusted side answers with an error.
        brokered = runner.request("broker")["brokered"]
        assert brokered["id"] == "m1"
        assert brokered["ok"] is False
        assert "gateway is not initialized" in brokered["error"]["message"]

        with pytest.raises(RuntimeError, match="closed its response stream"):
            runner.request("crash")
        assert any("fatal candidate error" in line for line in runner._stderr_tail)
        with pytest.raises(RuntimeError, match="exited with 7"):
            runner.request("echo")
    finally:
        runner.close()


def test_sealed_runner_reads_endpoint_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU2_SEALED_CANDIDATE_ENDPOINT", "candidate:7000")
    config = SealedRunnerConfig(kit_path=tmp_path, domain="demo")
    assert config.candidate_endpoint == "candidate:7000"
    monkeypatch.delenv("TAU2_SEALED_CANDIDATE_ENDPOINT")
    assert (
        SealedRunnerConfig(kit_path=tmp_path, domain="demo").candidate_endpoint is None
    )


def test_unreachable_endpoint_fails_with_a_clear_error(tmp_path):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    with pytest.raises(RuntimeError, match="Could not reach the candidate host"):
        connect_candidate_session(f"127.0.0.1:{port}", connect_timeout=0.5)


def _alive(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return False
    return "\nState:\tZ" not in status


def _isolation_available(tmp_path) -> bool:
    if os.geteuid() != 0:
        return False
    host = CandidateHost(
        command=["true"], session_root=tmp_path / "probe-sessions", workspace=tmp_path
    )
    return host.isolate


@pytest.fixture
def shared_dir():
    """A directory other uids can traverse (pytest's tmp root is 0700)."""
    path = Path(tempfile.mkdtemp(prefix="candidate-host-"))
    path.chmod(0o755)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_isolated_sessions_get_private_uids_and_are_cleaned_up(serve_host, shared_dir):
    if not _isolation_available(shared_dir):
        pytest.skip("per-session uids need root with CHOWN/SETUID/SETGID")
    report = (
        "import json, os, subprocess, sys\n"
        "sys.stdin.readline()\n"
        "home = os.environ['HOME']\n"
        "open(os.path.join(home, 'private'), 'w').write('secret')\n"
        "child = subprocess.Popen(['sleep', '600'], start_new_session=True)\n"
        "print(json.dumps({'uid': os.getuid(), 'home': home,\n"
        "                  'tmp': os.environ['TMPDIR'], 'child': child.pid}),\n"
        "      flush=True)\n"
        "sys.stdin.readline()\n"
    )
    endpoint = serve_host(
        [sys.executable, "-I", "-c", report],
        isolate=None,
        exit_grace_seconds=0.5,
        workspace=shared_dir,
        session_root=shared_dir / "sessions",
    )
    first = connect_candidate_session(endpoint)
    second = connect_candidate_session(endpoint)
    first.stdin.write("start\n")
    second.stdin.write("start\n")
    one = json.loads(_read_line(first))
    two = json.loads(_read_line(second))

    assert one["uid"] != two["uid"]
    assert 0 not in (one["uid"], two["uid"])
    assert one["home"] == one["tmp"] != two["home"]
    assert (Path(one["home"]).stat().st_mode & 0o777) == 0o700

    # A detached grandchild does not survive its session, and the session's
    # files are gone once it ends.
    first.stdin.write("done\n")
    first.stdin.close()
    assert first.wait(timeout=20) == 0
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _alive(one["child"]):
        time.sleep(0.1)
    assert not _alive(one["child"])
    assert not Path(one["home"]).exists()
    assert _alive(two["child"])

    second.terminate()
    second.wait(timeout=20)
