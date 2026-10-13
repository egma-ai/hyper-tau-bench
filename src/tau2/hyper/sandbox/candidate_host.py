"""Serve sealed candidate sessions over TCP from one long-lived container.

``CandidateProcess`` normally starts a fresh ``docker run`` container for
every candidate session. Runtimes that cannot reach a Docker daemon instead
run one locked-down candidate service (read-only root filesystem, read-only
workspace, internal-only network, no credentials) with this host as its
entrypoint, and point the trusted side at it through
``TAU2_SEALED_CANDIDATE_ENDPOINT``. Each TCP connection becomes one
candidate session:

- the first byte from the client starts ``candidate_server`` with the same
  bootstrap as the ``docker run`` path (``python -I``, trusted runtime
  source ahead of the workspace on ``sys.path``), in the workspace;
- when the host runs as root, every session gets its own uid, a private
  home and temp directory, and a process limit, so concurrent and later
  sessions cannot see each other's processes or files;
- bytes from the client are the session's stdin, and the session's output
  comes back as tagged lines: ``o`` + a stdout line, ``e`` + a stderr line,
  and finally one ``x`` + ``{"returncode": N}`` line;
- a client disconnect ends the session, and an ended session's processes
  are killed and its files deleted.

The host never imports tau2 or workspace code: it runs as a plain script
(``python -I candidate_host.py``). The client half of the protocol is
``connect_candidate_session``, which returns a ``subprocess.Popen``-shaped
handle that ``CandidateProcess`` drives exactly like a local container.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import random
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Sequence

CANDIDATE_BOOTSTRAP = (
    "import sys;"
    "sys.path.insert(0, '/opt/tau2/src');"
    "sys.path.append('/workspace');"
    "from tau2.hyper.sandbox.candidate_server import main;"
    "main()"
)

STDOUT_TAG = "o"
STDERR_TAG = "e"
EXIT_TAG = "x"

DEFAULT_PORT = 7000
DEFAULT_WORKSPACE = "/workspace"
DEFAULT_SESSION_ROOT = "/tmp/tau2-candidate-sessions"
DEFAULT_MAX_SESSIONS = 64
DEFAULT_SESSION_NPROC = 256

# Host environment a session may inherit; everything else (credentials
# included) stays out of candidate processes.
_ENV_PASSTHROUGH = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TAU2_DATA_DIR",
    "TAU2_SOURCE_REVISION",
)

# Session uids stay inside the 16-bit range that user-namespace remapping
# and rootless Docker map by default.
_UID_LOW = 20000
_UID_HIGH = 65000

# Runs as the session uid once a session ends: kill everything the session
# left behind, then delete its files. kill(-1) never signals its caller, and
# an unprivileged caller can only signal processes of its own uid.
_CLEANUP_SCRIPT = """
import os, shutil, signal, sys
for _ in range(5):
    try:
        os.kill(-1, signal.SIGKILL)
    except OSError:
        break
shutil.rmtree(sys.argv[1], ignore_errors=True)
uid = os.getuid()
try:
    entries = list(os.scandir('/dev/shm'))
except OSError:
    entries = []
for entry in entries:
    try:
        if entry.stat(follow_symlinks=False).st_uid != uid:
            continue
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            os.unlink(entry.path)
    except OSError:
        pass
"""


def _log(message: str) -> None:
    print(f"candidate-host: {message}", file=sys.stderr, flush=True)


def _limits_prelude(nproc: Optional[int]) -> str:
    statements = [
        "import resource",
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))",
    ]
    if nproc is not None:
        statements.append(
            f"resource.setrlimit(resource.RLIMIT_NPROC, ({nproc}, {nproc}))"
        )
    return ";".join(statements) + ";"


class _Session:
    """One client connection and the candidate process serving it."""

    def __init__(self, host: "CandidateHost", conn: socket.socket, first: bytes):
        self.host = host
        self.conn = conn
        self.first = first
        self.uid: Optional[int] = None
        self.directory: Optional[Path] = None
        self.process: Optional[subprocess.Popen] = None
        self._send_lock = threading.Lock()
        self._client_open = True
        self._client_closed = threading.Event()

    # Output -----------------------------------------------------------------

    def _send(self, tag: str, payload: bytes) -> None:
        if not payload.endswith(b"\n"):
            payload += b"\n"
        with self._send_lock:
            if not self._client_open:
                return
            try:
                self.conn.sendall(tag.encode("ascii") + payload)
            except OSError:
                self._client_open = False

    def _pump(self, stream, tag: str) -> None:
        try:
            for line in iter(stream.readline, b""):
                self._send(tag, line)
        except (OSError, ValueError):
            pass

    def _pump_stdin(self) -> None:
        process = self.process
        assert process is not None and process.stdin is not None
        try:
            data = self.first
            while data:
                process.stdin.write(data)
                process.stdin.flush()
                data = self.conn.recv(65536)
        except OSError:
            pass
        finally:
            self._client_closed.set()
            try:
                process.stdin.close()
            except OSError:
                pass

    # Lifecycle ----------------------------------------------------------------

    def _prepare(self) -> dict[str, str]:
        host = self.host
        self.uid = host.next_uid() if host.isolate else None
        name = f"s{next(host.session_counter)}-{os.getpid()}"
        self.directory = host.session_root / name
        self.directory.mkdir(mode=0o700)
        if self.uid is not None:
            os.chown(self.directory, self.uid, self.uid)
        env = {key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ}
        env.update(
            {
                "HOME": str(self.directory),
                "TMPDIR": str(self.directory),
                "USER": "candidate",
                "LOGNAME": "candidate",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        return env

    def _spawn(self, env: dict[str, str]) -> subprocess.Popen:
        host = self.host
        identity: dict = {}
        if self.uid is not None:
            identity = {"user": self.uid, "group": self.uid, "extra_groups": [self.uid]}
        return subprocess.Popen(
            host.command(nproc=host.session_nproc if host.isolate else None),
            cwd=host.workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            umask=0o077,
            close_fds=True,
            **identity,
        )

    def _terminate(self) -> None:
        """Kill every process of this session and delete its files."""
        process = self.process
        if self.uid is not None and self.directory is not None:
            try:
                subprocess.run(
                    [sys.executable, "-I", "-c", _CLEANUP_SCRIPT, str(self.directory)],
                    user=self.uid,
                    group=self.uid,
                    extra_groups=[self.uid],
                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    timeout=60,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                _log(f"session cleanup failed for uid {self.uid}: {error}")
        elif process is not None:
            try:
                os.killpg(process.pid, 9)
            except OSError:
                pass
        if self.directory is not None:
            if self.uid is None:
                shutil.rmtree(self.directory, ignore_errors=True)
            else:
                try:
                    self.directory.rmdir()
                except OSError:
                    pass

    def serve(self) -> None:
        try:
            self.process = self._spawn(self._prepare())
        except OSError as error:
            self._send(STDERR_TAG, f"candidate host could not start: {error}".encode())
            self._send(EXIT_TAG, json.dumps({"returncode": 127}).encode())
            self._terminate()
            return
        process = self.process
        pumps = [
            threading.Thread(target=self._pump, args=(process.stdout, STDOUT_TAG)),
            threading.Thread(target=self._pump, args=(process.stderr, STDERR_TAG)),
        ]
        stdin_pump = threading.Thread(target=self._pump_stdin)
        for thread in (*pumps, stdin_pump):
            thread.daemon = True
            thread.start()

        # A client that hangs up closes the candidate's stdin, which ends a
        # well-behaved candidate_server; one that ignores it is killed.
        closed_at: Optional[float] = None
        killed = False
        while True:
            try:
                returncode = process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            if self._client_closed.is_set() and not killed:
                closed_at = closed_at or time.monotonic()
                if time.monotonic() - closed_at > self.host.exit_grace_seconds:
                    self._terminate()
                    killed = True

        # The candidate exited: kill anything it left running (which also
        # closes the output pipes it shared), deliver the remaining output,
        # then the exit status.
        self._terminate()
        for thread in pumps:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in pumps):
            self._shutdown_socket()
        self._send(EXIT_TAG, json.dumps({"returncode": returncode}).encode())
        self._shutdown_socket()
        stdin_pump.join(timeout=10)
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass

    def _shutdown_socket(self) -> None:
        with self._send_lock:
            self._client_open = False
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


class CandidateHost:
    """Accept TCP connections and serve one candidate session per connection."""

    def __init__(
        self,
        *,
        workspace: str | os.PathLike = DEFAULT_WORKSPACE,
        session_root: str | os.PathLike = DEFAULT_SESSION_ROOT,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        session_nproc: int = DEFAULT_SESSION_NPROC,
        isolate: Optional[bool] = None,
        command: Optional[Sequence[str]] = None,
        exit_grace_seconds: float = 10.0,
        slot_timeout_seconds: float = 900.0,
    ):
        self.workspace = str(workspace)
        self.session_root = Path(session_root)
        self.session_nproc = session_nproc
        self.isolate = os.geteuid() == 0 if isolate is None else isolate
        self._command = list(command) if command is not None else None
        self.exit_grace_seconds = exit_grace_seconds
        self.slot_timeout_seconds = slot_timeout_seconds
        self.session_counter = itertools.count(1)
        self._slots = threading.BoundedSemaphore(max_sessions)
        self._uid_lock = threading.Lock()
        self._next_uid = random.randrange(_UID_LOW, _UID_HIGH)
        self.session_root.mkdir(parents=True, exist_ok=True)
        if self.isolate:
            # Sessions may traverse to their own directory but not list
            # or create entries next to it.
            os.chmod(self.session_root, 0o711)
            self.isolate = self._can_isolate()

    def _can_isolate(self) -> bool:
        """Check that this container grants what per-session uids need.

        Root needs CHOWN, SETUID and SETGID (the rest of its capabilities
        can be dropped). Without them every session shares the host's uid.
        """
        uid = self.next_uid()
        probe = self.session_root / f"probe-{os.getpid()}"
        try:
            probe.mkdir(mode=0o700, exist_ok=True)
            os.chown(probe, uid, uid)
            subprocess.run(
                [sys.executable, "-I", "-c", "pass"],
                user=uid,
                group=uid,
                extra_groups=[uid],
                env={},
                stdin=subprocess.DEVNULL,
                check=True,
                timeout=30,
            )
            return True
        except (OSError, subprocess.SubprocessError) as error:
            _log(f"per-session uids unavailable, sessions share a uid: {error}")
            return False
        finally:
            try:
                probe.rmdir()
            except OSError:
                pass

    def command(self, *, nproc: Optional[int]) -> list[str]:
        if self._command is not None:
            return list(self._command)
        return [
            sys.executable,
            "-I",
            "-c",
            _limits_prelude(nproc) + CANDIDATE_BOOTSTRAP,
        ]

    def next_uid(self) -> int:
        with self._uid_lock:
            uid = self._next_uid
            self._next_uid = uid + 1 if uid + 1 < _UID_HIGH else _UID_LOW
            return uid

    def handle(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            first = conn.recv(65536)
        except OSError:
            return
        if not first:
            # A bare connect-and-close (health probe) starts nothing.
            return
        session = _Session(self, conn, first)
        if not self._slots.acquire(timeout=self.slot_timeout_seconds):
            session._send(STDERR_TAG, b"candidate host is at capacity")
            session._send(EXIT_TAG, json.dumps({"returncode": 75}).encode())
            return
        try:
            session.serve()
        except Exception as error:  # noqa: BLE001 - keep serving other sessions
            _log(f"session failed: {type(error).__name__}: {error}")
            session._terminate()
        finally:
            self._slots.release()

    def serve_forever(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
        candidate_host = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                candidate_host.handle(self.request)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
            request_queue_size = 128

        if os.getpid() == 1:
            _log(
                "running as PID 1: orphaned candidate processes are not reaped; "
                "run the container with an init process (compose `init: true`)"
            )
        with Server((host, port), Handler) as server:
            mode = "per-session uids" if self.isolate else "shared uid"
            _log(f"serving {self.workspace} on {host}:{port} ({mode})")
            server.serve_forever()


# Client -----------------------------------------------------------------------


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """Split ``host:port`` (optionally ``tcp://host:port``) into its parts."""
    value = endpoint.strip().removeprefix("tcp://").rstrip("/")
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(
            f"Candidate endpoint must look like host:port, got {endpoint!r}"
        )
    return host.strip("[]"), int(port)


class _SocketWriter:
    """Text writer that sends to the session; close() half-closes the socket."""

    def __init__(self, sock: socket.socket):
        self._sock = sock

    def write(self, text: str) -> int:
        self._sock.sendall(text.encode("utf-8"))
        return len(text)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _queue_lines(lines: "queue.Queue[Optional[str]]") -> Iterator[str]:
    while True:
        line = lines.get()
        if line is None:
            return
        yield line


class CandidateSession:
    """``subprocess.Popen``-shaped handle on a remote candidate session."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self.stdin = _SocketWriter(sock)
        # Bounded like an OS pipe: a candidate that floods stdout blocks
        # instead of growing trusted-side memory.
        self._stdout: queue.Queue[Optional[str]] = queue.Queue(maxsize=1024)
        self._stderr: queue.Queue[Optional[str]] = queue.Queue()
        self.stdout = _queue_lines(self._stdout)
        self.stderr = _queue_lines(self._stderr)
        self.returncode: Optional[int] = None
        self._exited = threading.Event()
        self._closing = threading.Event()
        self._reader = threading.Thread(
            target=self._read, name="candidate-session-reader", daemon=True
        )
        self._reader.start()

    def _put_stdout(self, item: Optional[str]) -> None:
        while True:
            try:
                self._stdout.put(item, timeout=0.5)
                return
            except queue.Full:
                if not self._closing.is_set():
                    continue
            # Nobody will read the backlog of a session being torn down.
            while True:
                try:
                    self._stdout.get_nowait()
                except queue.Empty:
                    break

    def _read(self) -> None:
        returncode: Optional[int] = None
        try:
            with self._sock.makefile(
                "r", encoding="utf-8", errors="replace", newline="\n"
            ) as frames:
                for frame in frames:
                    tag, payload = frame[:1], frame[1:]
                    if tag == STDOUT_TAG:
                        self._put_stdout(payload)
                    elif tag == STDERR_TAG:
                        self._stderr.put(payload)
                    elif tag == EXIT_TAG:
                        try:
                            returncode = int(json.loads(payload)["returncode"])
                        except (ValueError, KeyError, TypeError):
                            returncode = -1
                        break
        except OSError:
            pass
        finally:
            if returncode is None:
                self._stderr.put(
                    "candidate host closed the session without an exit status\n"
                )
                returncode = -1
            self.returncode = returncode
            self._put_stdout(None)
            self._stderr.put(None)
            self._exited.set()
            try:
                self._sock.close()
            except OSError:
                pass

    def poll(self) -> Optional[int]:
        return self.returncode if self._exited.is_set() else None

    def wait(self, timeout: Optional[float] = None) -> int:
        if not self._exited.wait(timeout):
            raise subprocess.TimeoutExpired("candidate-session", timeout or 0)
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        # Dropping the connection makes the host kill the session.
        self._closing.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    kill = terminate


def connect_candidate_session(
    endpoint: str, *, connect_timeout: float = 60.0
) -> CandidateSession:
    """Open a candidate session, retrying while the host is still starting."""
    host, port = parse_endpoint(endpoint)
    deadline = time.monotonic() + connect_timeout
    delay = 0.25
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=10)
            break
        except OSError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Could not reach the candidate host at {endpoint}: {error}"
                ) from error
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return CandidateSession(sock)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--session-root", default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--max-sessions", type=int, default=DEFAULT_MAX_SESSIONS)
    parser.add_argument("--session-nproc", type=int, default=DEFAULT_SESSION_NPROC)
    args = parser.parse_args(argv)
    CandidateHost(
        workspace=args.workspace,
        session_root=args.session_root,
        max_sessions=args.max_sessions,
        session_nproc=args.session_nproc,
    ).serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()
