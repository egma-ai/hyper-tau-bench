"""Run Hyper-tau release tasks in parallel, one Daytona sandbox per task.

Each sandbox is a Docker-in-Docker VM (the same base Harbor uses on Daytona)
that runs the unmodified sealed runner (`tau2 hyper-tau`) for one task:
tools/daytona/runner.sh installs the host, applies this checkout's diff on
top of the upstream base commit, builds the construction image and runs the
task. This script uploads the inputs, polls, collects logs and the run
recording, and deletes the sandbox.

Usage (from the repo root, with DAYTONA_API_KEY, OPENAI_API_KEY and
OPENROUTER_API_KEY set or in .env):

    uv run --with "daytona>=0.210.0" python tools/daytona/launch.py \\
        --run-name pilot --tasks 001,016,021 --concurrency 3 \\
        --developer-llm gpt-6.1-sol --developer-effort max \\
        --developer-auth chatgpt --chatgpt-auth ~/.hypertau-codex/auth.json

Results land in data/simulations/hyper_tau_daytona/<run-name>/ (state.json,
one directory per task, summary.json and summary.md). Re-running the same
command resumes: finished tasks are skipped and sandboxes that are still
running are re-attached.

With --developer-auth chatgpt, only the login's access token and account id
are uploaded (never the refresh token), and the sealed runner keeps them in
the model-gateway sidecar, out of the Developer's container.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO_ROOT / "data" / "tau2" / "hyper" / "tasks"
RESULTS_ROOT = REPO_ROOT / "data" / "simulations" / "hyper_tau_daytona"
RUNNER = Path(__file__).with_name("runner.sh")
UPSTREAM_REPO = "https://github.com/sierra-research/hyper-tau-bench"

DIND_IMAGE = "docker:28.3.3-dind"
SANDBOX_HOME = "/root/hyper"
SANDBOX_RECORDINGS = "/root/htb/data/simulations/hyper_tau"
# Hard wall-clock cap per sandbox: 8 h build budget + setup + scoring.
SANDBOX_TTL_MINUTES = 12 * 60
USAGE_LIMIT_MARKERS = ("usage limit", "usage_limit", "rate_limit_exceeded")

_print_lock = threading.Lock()
_state_lock = threading.Lock()


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{stamp}] {message}", flush=True)


# --- Inputs -----------------------------------------------------------------


def resolve_tasks(spec: str) -> list[str]:
    """Task ids from 'all' or a comma list of ids or 3-digit release slots."""
    available = sorted(p.stem for p in TASKS_DIR.glob("*.json"))
    if spec == "all":
        return available
    chosen = []
    for item in (part.strip() for part in spec.split(",") if part.strip()):
        matches = [t for t in available if t == item or t.startswith(f"{item}_")]
        if len(matches) != 1:
            raise SystemExit(f"Task {item!r} matches {len(matches)} release tasks")
        chosen.append(matches[0])
    return chosen


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def branch_patch(base_commit: str) -> bytes:
    """This checkout's changes relative to the upstream base commit."""
    untracked = git("ls-files", "--others", "--exclude-standard", "src", "docker")
    if untracked.strip():
        log(f"warning: untracked files are NOT shipped:\n{untracked}")
    return subprocess.run(
        ["git", "diff", "--binary", base_commit],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def load_dotenv_keys() -> dict[str, str]:
    """Provider keys for the sandbox .env, from the environment or ./.env."""
    values = {}
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                values[key.strip()] = value.strip().strip("'\"")
    keys = {}
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        value = os.environ.get(name) or values.get(name)
        if value:
            keys[name] = value
    if "OPENAI_API_KEY" not in keys:
        raise SystemExit("OPENAI_API_KEY is required (simulators, judges, agents)")
    if "OPENROUTER_API_KEY" not in keys:
        log("warning: OPENROUTER_API_KEY missing; non-OpenAI menu models will fail")
    return keys


def chatgpt_access_file(path: Path) -> bytes:
    """Access token + account id from a Codex login; refuses near-expiry."""
    tokens = json.loads(path.expanduser().read_text())["tokens"]
    access_token, account_id = tokens["access_token"], tokens.get("account_id")
    claims_b64 = access_token.split(".")[1]
    claims = json.loads(
        base64.urlsafe_b64decode(claims_b64 + "=" * (-len(claims_b64) % 4))
    )
    hours_left = (claims.get("exp", 0) - time.time()) / 3600
    if hours_left < 12:
        raise SystemExit(
            f"ChatGPT access token expires in {hours_left:.1f} h; refresh the "
            "login (run any codex command with it) and retry"
        )
    log(f"ChatGPT access token valid for {hours_left:.0f} more hours")
    return json.dumps(
        {"tokens": {"access_token": access_token, "account_id": account_id}}
    ).encode()


# --- State --------------------------------------------------------------------


class RunState:
    """state.json: one entry per task, rewritten atomically on every change."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "state.json"
        self.tasks: dict[str, dict] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )

    def update(self, task_id: str, **fields) -> None:
        with _state_lock:
            self.tasks.setdefault(task_id, {}).update(fields)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.tasks, indent=2, sort_keys=True))
            tmp.replace(self.path)

    def get(self, task_id: str) -> dict:
        with _state_lock:
            return dict(self.tasks.get(task_id, {}))


# --- One task ---------------------------------------------------------------


def sandbox_exec(sandbox, command: str, timeout: int = 120) -> tuple[int, str]:
    response = sandbox.process.exec(command, timeout=timeout)
    return response.exit_code, response.result or ""


def with_retries(action, attempts: int = 4, delay: float = 10.0):
    """Run a Daytona API call, retrying transient failures."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception:  # noqa: BLE001 - re-raised after the last attempt
            if attempt == attempts:
                raise
            time.sleep(delay * attempt)


def create_sandbox(daytona, run_name: str, task_id: str):
    from daytona import CreateSandboxFromImageParams, Image, Resources

    params = CreateSandboxFromImageParams(
        image=Image.base(DIND_IMAGE),
        resources=Resources(cpu=4, memory=8, disk=10),
        labels={"hypertau-run": run_name, "hypertau-slot": task_id[:3]},
        auto_stop_interval=0,
        ttl_minutes=SANDBOX_TTL_MINUTES,
        network_block_all=False,
    )
    return daytona.create(params, timeout=300)


def start_task(sandbox, job: dict, inputs: dict[str, bytes]) -> None:
    sandbox_exec(sandbox, f"mkdir -p {SANDBOX_HOME} && chmod 700 {SANDBOX_HOME}")
    job_env = "".join(f"{key}={json.dumps(value)}\n" for key, value in job.items())
    files = {"job.env": job_env.encode(), "runner.sh": RUNNER.read_bytes(), **inputs}
    for name, content in files.items():
        sandbox.fs.upload_file(content, f"{SANDBOX_HOME}/{name}")
    sandbox_exec(sandbox, f"chmod 600 {SANDBOX_HOME}/*")
    sandbox_exec(
        sandbox,
        f"nohup sh {SANDBOX_HOME}/runner.sh > {SANDBOX_HOME}/nohup.out 2>&1 &",
    )


def collect(sandbox, task_dir: Path) -> list[Path]:
    """Download logs and finished recordings; return the recording paths."""
    task_dir.mkdir(parents=True, exist_ok=True)
    for name in ("runner.log", "build.log", "run.log", "exit_code", "state"):
        try:
            content = sandbox.fs.download_file(f"{SANDBOX_HOME}/{name}")
        except Exception:  # noqa: BLE001 - a step may not have produced it
            continue
        if content is not None:
            (task_dir / name).write_bytes(content)
    _, listing = with_retries(
        lambda: sandbox_exec(
            sandbox, f"ls -1 {SANDBOX_RECORDINGS}/*.json 2>/dev/null || true"
        )
    )
    recordings = []
    for remote in listing.split():
        if remote.endswith(".in_progress.json"):
            continue
        content = with_retries(lambda remote=remote: sandbox.fs.download_file(remote))
        if content is not None:
            local = task_dir / Path(remote).name
            local.write_bytes(content)
            recordings.append(local)
    return recordings


def score(recordings: list[Path]) -> dict:
    """Reward and domain from the newest complete recording."""
    for path in sorted(recordings, reverse=True):
        data = json.loads(path.read_text())
        result = data.get("result") or {}
        if data.get("status") == "complete" and "final_test_reward" in result:
            return {
                "reward": result["final_test_reward"],
                "quality_reward": result.get("final_quality_reward"),
                "performance_penalty": result.get("performance_penalty"),
                "domain": (data.get("task_metadata") or {}).get("source_domain"),
                "recording": path.name,
            }
    return {}


def discover_sandboxes(daytona, run_name: str) -> dict[str, str]:
    """This run's sandboxes still on Daytona, keyed by release slot.

    Lets a restarted launcher (or one on a fresh machine without state.json)
    re-attach to tasks that kept running instead of starting duplicates.
    """
    from daytona import ListSandboxesQuery

    found = {}
    for sandbox in daytona.list(ListSandboxesQuery(labels={"hypertau-run": run_name})):
        slot = (sandbox.labels or {}).get("hypertau-slot")
        if slot:
            found[slot] = sandbox.id
    return found


def run_one(daytona, args, state: RunState, task_id: str, inputs, job_base) -> None:
    task_dir = args.run_dir / task_id
    entry = state.get(task_id)
    if args.abort.is_set():
        state.update(task_id, status="skipped", error="stopped after a failure")
        log(f"{task_id[:3]} skipped (--stop-on-failure)")
        return
    sandbox = None
    try:
        sandbox_id = (
            entry.get("sandbox_id") if entry.get("status") == "running" else None
        )
        sandbox_id = sandbox_id or args.discovered.get(task_id[:3])
        if sandbox_id and entry.get("status") != "failed":
            try:
                sandbox = daytona.get(sandbox_id)
                log(f"{task_id[:3]} re-attached to {sandbox.id}")
            except Exception:  # noqa: BLE001 - sandbox is gone; start over
                sandbox = None
        elif sandbox_id:
            # A retry of a failed task starts from a clean sandbox.
            try:
                daytona.delete(daytona.get(sandbox_id))
            except Exception:  # noqa: BLE001 - already gone
                pass
        if sandbox is None:
            sandbox = create_sandbox(daytona, args.run_name, task_id)
            state.update(
                task_id,
                status="running",
                sandbox_id=sandbox.id,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            log(f"{task_id[:3]} sandbox {sandbox.id} created")
            start_task(sandbox, {**job_base, "TASK_ID": task_id}, inputs)

        last_state = None
        poll_errors = 0
        deadline = time.monotonic() + SANDBOX_TTL_MINUTES * 60
        while time.monotonic() < deadline:
            try:
                _, current = sandbox_exec(
                    sandbox, f"cat {SANDBOX_HOME}/state 2>/dev/null"
                )
                poll_errors = 0
            except Exception as exc:  # noqa: BLE001 - transient API errors
                poll_errors += 1
                if poll_errors >= 15:
                    raise
                log(f"{task_id[:3]} poll error {poll_errors}/15: {exc}")
                time.sleep(args.poll_seconds)
                continue
            current = current.strip() or "booting"
            if current != last_state:
                log(f"{task_id[:3]} {current}")
                state.update(task_id, phase=current)
                last_state = current
            if current == "done" or current.startswith("failed:"):
                break
            time.sleep(args.poll_seconds)

        recordings = collect(sandbox, task_dir)
        result = score(recordings)
        run_log = (task_dir / "run.log").read_text(errors="replace").lower()
        usage_limited = any(marker in run_log for marker in USAGE_LIMIT_MARKERS)
        status = "done" if result else "failed"
        state.update(
            task_id,
            status=status,
            finished_at=datetime.now(timezone.utc).isoformat(),
            usage_limited=usage_limited,
            **result,
        )
        log(
            f"{task_id[:3]} {status}"
            + (f" reward={result['reward']:.3f}" if result else f" ({last_state})")
            + (" [usage limit hit]" if usage_limited else "")
        )
    except Exception as exc:  # noqa: BLE001 - record and move on
        state.update(task_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        log(f"{task_id[:3]} error: {type(exc).__name__}: {exc}")
    finally:
        if args.stop_on_failure and state.get(task_id).get("status") == "failed":
            args.abort.set()
        if sandbox is not None and not args.keep_sandboxes:
            final = state.get(task_id)
            # Keep a sandbox when the launcher itself errored: the task may
            # still be running there, and a rerun re-attaches by label.
            finished_in_sandbox = final.get("status") == "done" or str(
                final.get("phase", "")
            ).startswith(("failed:", "done"))
            if finished_in_sandbox:
                try:
                    daytona.delete(sandbox)
                    log(f"{task_id[:3]} sandbox deleted")
                except Exception as exc:  # noqa: BLE001
                    log(f"{task_id[:3]} delete failed: {exc}")


# --- Summary ------------------------------------------------------------------


def summarize(args, state: RunState, tasks: list[str]) -> dict:
    rows = [(t, state.get(t)) for t in tasks]
    done = [(t, e) for t, e in rows if e.get("status") == "done"]
    rewards = [e["reward"] for _, e in done]
    by_domain: dict[str, list[float]] = {}
    for _, entry in done:
        by_domain.setdefault(entry.get("domain") or "unknown", []).append(
            entry["reward"]
        )
    summary = {
        "run_name": args.run_name,
        "developer": {
            "harness": "codex",
            "model": args.developer_llm,
            "reasoning_effort": args.developer_effort,
            "auth": args.developer_auth,
        },
        "tasks_total": len(tasks),
        "tasks_done": len(done),
        "tasks_failed": sum(1 for _, e in rows if e.get("status") == "failed"),
        # Failed builds count as 0, the way a submission is scored.
        "overall_pct": round(100 * sum(rewards) / len(tasks), 1) if tasks else None,
        "overall_completed_pct": (
            round(100 * sum(rewards) / len(rewards), 1) if rewards else None
        ),
        "domains_pct": {
            domain: round(100 * sum(values) / len(values), 1)
            for domain, values in sorted(by_domain.items())
        },
        "usage_limited_tasks": [t for t, e in rows if e.get("usage_limited")],
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        f"# {args.run_name}: {args.developer_llm} ({args.developer_effort})",
        "",
        f"Overall {summary['overall_pct']}% over {len(tasks)} tasks "
        f"({len(done)} done, {summary['tasks_failed']} failed).",
        "",
        "| Task | Status | Reward | Domain |",
        "|---|---|---|---|",
    ]
    for task_id, entry in rows:
        reward = entry.get("reward")
        lines.append(
            f"| {task_id} | {entry.get('status', 'pending')} | "
            f"{'' if reward is None else f'{reward:.3f}'} | {entry.get('domain', '')} |"
        )
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


# --- Main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tasks", default="all", help="'all' or comma ids/slots")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--developer-llm", default="gpt-6.1-sol")
    parser.add_argument("--developer-effort", default="max")
    parser.add_argument(
        "--developer-auth", choices=("api-key", "chatgpt"), default="chatgpt"
    )
    parser.add_argument("--chatgpt-auth", type=Path, help="Codex login auth.json")
    parser.add_argument("--base-commit", help="upstream commit (default: merge-base)")
    parser.add_argument("--repo-url", default=UPSTREAM_REPO)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--stagger-seconds",
        type=int,
        default=0,
        help="delay between sandbox launches (e.g. let a first task prove setup)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="start no further tasks once one fails",
    )
    parser.add_argument("--keep-sandboxes", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = resolve_tasks(args.tasks)
    args.run_dir = RESULTS_ROOT / args.run_name
    args.run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(args.run_dir)
    if args.summarize_only:
        print(json.dumps(summarize(args, state, tasks), indent=2))
        return

    base = args.base_commit or git("merge-base", "HEAD", "origin/main").strip()
    patch = branch_patch(base)
    keys = load_dotenv_keys()
    inputs = {"branch.patch": patch}
    dotenv = dict(keys)
    if args.developer_auth == "chatgpt":
        if not args.chatgpt_auth:
            raise SystemExit("--chatgpt-auth is required with --developer-auth chatgpt")
        inputs["chatgpt-auth.json"] = chatgpt_access_file(args.chatgpt_auth)
        dotenv["TAU2_CHATGPT_AUTH_FILE"] = f"{SANDBOX_HOME}/chatgpt-auth.json"
    inputs["dotenv"] = "".join(f"{k}={v}\n" for k, v in dotenv.items()).encode()
    job_base = {
        "BASE_COMMIT": base,
        "REPO_URL": args.repo_url,
        "DEV_LLM": args.developer_llm,
        "DEV_EFFORT": args.developer_effort,
        "DEV_AUTH": args.developer_auth,
    }
    pending = [t for t in tasks if state.get(t).get("status") != "done"]
    log(
        f"run {args.run_name}: {len(tasks)} tasks, {len(pending)} to run, "
        f"concurrency {args.concurrency}, base {base[:10]}, patch {len(patch)} bytes"
    )
    if args.dry_run:
        for task_id in pending:
            log(f"would run {task_id}")
        return

    from daytona import Daytona

    daytona = Daytona()
    args.abort = threading.Event()
    args.discovered = discover_sandboxes(daytona, args.run_name)
    if args.discovered:
        log(f"found {len(args.discovered)} existing sandboxes for this run")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for index, task_id in enumerate(pending):
            if index and args.stagger_seconds and not args.abort.is_set():
                time.sleep(args.stagger_seconds)
            pool.submit(run_one, daytona, args, state, task_id, inputs, job_base)
    summary = summarize(args, state, tasks)
    log(f"summary: {json.dumps(summary)}")


if __name__ == "__main__":
    sys.exit(main())
