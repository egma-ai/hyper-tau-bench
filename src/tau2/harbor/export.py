"""Export Hyper-τ construction tasks as a Harbor dataset.

The Harbor agent is the Developer: it gets the task's brief as its
instruction and the developer kit in ``/workspace``. Each exported task runs
four containers:

- ``main`` — where the Harbor agent runs, built on the sealed construction
  runtime image (the toolchain a native Developer gets). Besides the kit it
  holds only a stdio MCP server (:mod:`tau2.harbor.mcp_stub`, served as
  ``hyper_tau``) that relays the Developer tools to ``hyper-runtime``, and a
  ``hyper-tau-tool`` shell wrapper for agents without MCP support.
- ``hyper-runtime`` — the trusted sidecar (:mod:`tau2.harbor.runtime`):
  it builds the kit, holds the Client simulator, the hidden task partitions
  and the model credentials, and serves ``run_local_test``,
  ``talk_to_client``, ``run_live_experiment``, ``run_sample_scenarios`` and
  ``submit``. At ``submit``, or when the agent phase ends, it seals the
  Client API deployment and snapshots the workspace.
- ``candidate`` — the sealed runtime for the Developer's code
  (:mod:`tau2.hyper.sandbox.candidate_host`): read-only workspace, read-only
  root filesystem, no capabilities beyond per-session uids, and only an
  internal network, so candidate code reaches models solely through the
  trusted side, as in ``tau2 hyper-tau``.
- the verifier (``tests/``) — the runtime image with a fresh ``candidate``
  service; it unpacks the snapshot and scores it on the hidden test
  partition (:mod:`tau2.harbor.verify`).

The Client's opening message is part of the brief, so it is generated once
per task at export time (one Client model call) and fixed in the task.

Harbor launches the agent with its own flags. The agent's image adds the
settings the native Codex and Claude Code harnesses use (Codex reads them
from ``/etc/codex/config.toml``, under the config Harbor writes; Claude Code
from its managed settings and environment), so web search, sub-agents and
the other features native runs switch off stay off. That includes the
native per-call tool limit (the construction budget), so long local tests
and sample-scenario runs simply block until they finish.
"""

from __future__ import annotations

import json
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import toml
from loguru import logger

from tau2.harbor import mcp_stub
from tau2.harbor.common import (
    DEFAULT_AGENT_ALLOWED_HOSTS,
    DEFAULT_RUNTIME_IMAGE,
    compose_literal,
    dump_compose,
    dump_task_toml,
    new_token,
    provider_env,
    task_slug,
    write_task,
)
from tau2.harbor.runtime import (
    BRIDGE_DIR,
    CONFIG_ENV_VAR,
    READY_FILE,
    STATE_DIR,
    TOKEN_ENV_VAR,
    WORKSPACE,
    ConstructionConfig,
    make_orchestrator,
)
from tau2.hyper.data_model import HyperTask
from tau2.hyper.harnesses.claude import (
    CLAUDE_CODE_ENVIRONMENT,
    ClaudeCodeSandboxBuilder,
)
from tau2.hyper.harnesses.codex import CodexSandboxBuilder
from tau2.hyper.runtime_contract import DEFAULT_CONSTRUCTION_RUNTIME_IMAGE
from tau2.hyper.sandbox.builder import BuildBudget
from tau2.hyper.sandbox.candidate_host import DEFAULT_PORT
from tau2.hyper.sandbox.native_builder import NativeSandboxBuilder
from tau2.hyper.task_loader import (
    load_active_hyper_tau_task,
    load_active_hyper_tau_tasks,
)

RUNTIME_SERVICE = "hyper-runtime"
CANDIDATE_SERVICE = "candidate"
SANDBOX_NETWORK = "sandbox"
WORKSPACE_VOLUME = "workspace"
BRIDGE_VOLUME = "hyper-tau-bridge"
MCP_SERVER_NAME = "hyper_tau"
STUB_PATH = "/opt/hyper-tau-harbor/mcp_stub.py"
STUB_PYTHON = "/usr/local/bin/python3"
TOOL_COMMAND = "/usr/local/bin/hyper-tau-tool"
CANDIDATE_HOST_SCRIPT = "/opt/tau2/src/tau2/hyper/sandbox/candidate_host.py"
CANDIDATE_ENDPOINT = f"{CANDIDATE_SERVICE}:{DEFAULT_PORT}"
CODEX_SYSTEM_CONFIG = "/etc/codex/config.toml"
CLAUDE_MANAGED_SETTINGS = "/etc/claude-code/managed-settings.json"
# Built-in tools the pinned Claude Code (CLAUDE_CODE_HARNESS_VERSION) offers
# beyond the native harness's tool list. Native runs leave them out with
# --tools; Harbor launches Claude Code with its own flags, so the agent's
# image denies them instead.
CLAUDE_CODE_EXTRA_TOOLS = (
    "Agent",
    "CronCreate",
    "CronDelete",
    "CronList",
    "EnterWorktree",
    "ExitWorktree",
    "ReportFindings",
    "ScheduleWakeup",
    "SendMessage",
    "Skill",
    "Task",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "Workflow",
)

HARBOR_NOTES = """\
## Working in this environment

- The developer kit is the Git repository at `{workspace}`, your working \
directory. Whatever `{workspace}` holds when you call `submit`, or when your \
time runs out, is what gets evaluated.
- If you cannot call MCP tools directly, use the same tools from a shell: \
`hyper-tau-tool list` shows them and `hyper-tau-tool <tool> '<JSON \
arguments>'` calls one, for example `hyper-tau-tool run_local_test \
'{{"task_path": "workspace/my_customer_scenario.json"}}'`.
"""

WAIT_NOTE = """\
- A tool call that runs longer than {seconds:g} seconds returns a job id \
instead of its result. Call `wait_for_result` with that job id to keep \
waiting. Never repeat the original call: it would run again and use its \
quota again.
"""


@dataclass
class ExportOptions:
    """What to export and how the exported tasks run."""

    output_dir: Path
    task_ids: Optional[list[str]] = None
    source_domain: Optional[str] = None
    runtime_image: str = DEFAULT_RUNTIME_IMAGE
    # Default: the task's own pinned construction runtime image.
    construction_image: Optional[str] = None
    client_llm: Optional[str] = None
    user_llm: Optional[str] = None
    agent_llm: Optional[str] = None
    primary_model_only: bool = False
    agent_network: str = "allowlist"
    allowed_hosts: list[str] = field(
        default_factory=lambda: list(DEFAULT_AGENT_ALLOWED_HOSTS)
    )
    # Default: the task's wall-clock construction budget (eight hours).
    agent_timeout_sec: Optional[float] = None
    verifier_timeout_sec: float = 14400.0
    # Off by default: tool calls block until they finish, as in native runs.
    # For an MCP client that caps a call below a tool's running time, a call
    # still running after this many seconds returns a job id instead, for
    # the extra wait_for_result tool.
    mcp_call_timeout_sec: float = 0.0
    reuse_client_openings: bool = True
    allow_legacy_domain: bool = False


def load_construction_tasks(options: ExportOptions) -> list[HyperTask]:
    if options.task_ids:
        tasks = [
            load_active_hyper_tau_task(
                task_id, allow_legacy=options.allow_legacy_domain
            )
            for task_id in options.task_ids
        ]
        if options.source_domain:
            tasks = [
                task for task in tasks if task.source_domain == options.source_domain
            ]
        return tasks
    return load_active_hyper_tau_tasks(
        options.source_domain, allow_legacy=options.allow_legacy_domain
    )


def previous_client_opening(task_dir: Path, task_id: str, client_llm: str) -> str:
    """The opening an earlier export of this task generated, if reusable."""
    path = Path(task_dir) / "task.toml"
    if not path.is_file():
        return ""
    try:
        metadata = tomllib.loads(path.read_text()).get("metadata", {})
    except tomllib.TOMLDecodeError:
        return ""
    if metadata.get("task_id") != task_id or metadata.get("client_llm") != client_llm:
        return ""
    return str(metadata.get("client_opening") or "")


def export_tasks(options: ExportOptions) -> list[Path]:
    """Write one Harbor task per Hyper-τ construction task."""
    if options.agent_network not in ("allowlist", "public"):
        raise ValueError(f"Unknown agent network mode {options.agent_network!r}")
    tasks = load_construction_tasks(options)
    if not tasks:
        raise ValueError("No Hyper-τ construction tasks selected")
    output_dir = Path(options.output_dir)
    written = []
    for task in tasks:
        task_dir = output_dir / task_slug(task.id)
        write_construction_task(task_dir, task, options)
        written.append(task_dir)
    logger.info(f"Wrote {len(written)} Harbor tasks to {output_dir}")
    return written


def write_construction_task(task_dir: Path, task: HyperTask, options: ExportOptions):
    sandbox_config = dict(task.hyper.sandbox_config or {})
    budget = BuildBudget(
        max_steps=sandbox_config.get("max_steps", 0),
        max_time_seconds=int(
            options.agent_timeout_sec or sandbox_config.get("max_time_seconds") or 0
        ),
    )
    config = ConstructionConfig(
        task_id=task.id,
        client_llm=options.client_llm,
        user_llm=options.user_llm,
        agent_llm=options.agent_llm,
        primary_model_only=options.primary_model_only,
        allow_legacy_domain=options.allow_legacy_domain,
        max_time_seconds=budget.max_time_seconds,
    )
    orchestrator = make_orchestrator(task, config)
    client_enabled = orchestrator._client_enabled()
    opening = ""
    if client_enabled:
        if options.reuse_client_openings:
            opening = previous_client_opening(
                task_dir, task.id, orchestrator.client_llm
            )
        if not opening:
            logger.info(f"Generating the Client's opening for {task.id}")
            opening = orchestrator.generate_client_opening()
            if not opening.strip():
                raise RuntimeError(f"The Client's opening for {task.id} is empty")
        # The rest of the Client's conversation continues the opening on
        # the same model.
        config.client_opening = opening
        config.client_llm = orchestrator.client_llm

    brief = orchestrator._build_brief(opening)
    instruction = NativeSandboxBuilder._developer_prompt(brief, budget).rstrip()
    notes = HARBOR_NOTES.format(workspace=WORKSPACE)
    if options.mcp_call_timeout_sec > 0:
        notes += WAIT_NOTE.format(seconds=options.mcp_call_timeout_sec)
    instruction = f"{instruction}\n\n{notes}"

    construction_image = (
        options.construction_image
        or sandbox_config.get("docker_image")
        or DEFAULT_CONSTRUCTION_RUNTIME_IMAGE
    )
    bridge_token = new_token()
    name = task_slug(task.id)

    stub_args = [
        "-I",
        STUB_PATH,
        "--dir",
        str(BRIDGE_DIR),
        "--token",
        bridge_token,
        "--server-name",
        MCP_SERVER_NAME,
        "--timeout",
        f"{budget.max_time_seconds}",
    ]
    if options.mcp_call_timeout_sec > 0:
        stub_args += ["--call-timeout", f"{options.mcp_call_timeout_sec:g}"]
    codex_config = codex_system_config(stub_args, budget.max_time_seconds)

    agent: dict = {"timeout_sec": float(budget.max_time_seconds)}
    if options.agent_network == "allowlist":
        agent["network_mode"] = "allowlist"
        agent["allowed_hosts"] = list(options.allowed_hosts)

    inner_env = {
        **provider_env(),
        # Concurrency of scored inner-loop conversations (default 32).
        "TAU2_HYPER_INNER_MAX_WORKERS": "${TAU2_HYPER_INNER_MAX_WORKERS:-}",
    }
    finalize_command = shlex.join(
        [
            "python",
            "-m",
            "tau2.harbor.runtime",
            "finalize",
            "--state-dir",
            str(STATE_DIR),
            "--timeout",
            "600",
        ]
    )
    task_config = {
        "schema_version": "1.4",
        # The sidecar's workspace snapshot and session record are what the
        # verifier scores.
        "artifacts": [{"source": str(STATE_DIR), "service": RUNTIME_SERVICE}],
        "task": {
            "name": f"hyper-tau/{name}",
            "description": (
                f"Hyper-τ construction task {task.id}: build a customer-service "
                f"agent for {task.source_domain} from the evidence in a kit"
            ),
            "keywords": [
                "hyper-tau",
                "tau2",
                "agent-construction",
                "customer-service",
                task.source_domain,
            ],
        },
        "metadata": {
            "benchmark": "hyper-tau",
            "task_id": task.id,
            "source_domain": task.source_domain,
            "client_enabled": client_enabled,
            "client_llm": orchestrator.client_llm,
            "user_llm": orchestrator.user_llm,
            "agent_llm": orchestrator.agent_llm,
            "primary_model_only": options.primary_model_only,
            "test_tasks": len(task.test_task_ids),
            "live_experiment_tasks": len(task.live_experiment_task_ids),
            "sample_scenarios": len(task.training_task_ids),
            "construction_image": construction_image,
            # Reused by a later export so the brief stays fixed.
            "client_opening": opening,
        },
        "agent": agent,
        "verifier": {
            "timeout_sec": float(options.verifier_timeout_sec),
            "environment_mode": "separate",
            # Scored conversations run concurrently in the verifier.
            "environment": {
                "build_timeout_sec": 3600.0,
                "cpus": 4,
                "memory_mb": 8192,
            },
            "env": inner_env,
            "collect": [
                {
                    # Seal and snapshot if the Developer never called submit.
                    "service": RUNTIME_SERVICE,
                    "command": finalize_command,
                    "timeout_sec": 900.0,
                }
            ],
        },
        "environment": {
            "build_timeout_sec": 3600.0,
            "cpus": int(sandbox_config.get("docker_cpus") or 2),
            "memory_mb": _memory_mb(sandbox_config.get("docker_memory"), 4096),
            "workdir": str(WORKSPACE),
            "mcp_servers": [
                {
                    "name": MCP_SERVER_NAME,
                    "transport": "stdio",
                    "command": STUB_PYTHON,
                    "args": stub_args,
                }
            ],
        },
    }

    candidate = candidate_service(construction_image)
    runtime_command = [
        "python",
        "-m",
        "tau2.harbor.runtime",
        "serve",
        "--state-dir",
        str(STATE_DIR),
    ]
    compose = {
        "services": {
            "main": {
                "depends_on": {RUNTIME_SERVICE: {"condition": "service_healthy"}},
                "volumes": [
                    f"{WORKSPACE_VOLUME}:{WORKSPACE}",
                    f"{BRIDGE_VOLUME}:{BRIDGE_DIR}",
                ],
            },
            RUNTIME_SERVICE: {
                "image": options.runtime_image,
                "command": runtime_command,
                "depends_on": {CANDIDATE_SERVICE: {"condition": "service_healthy"}},
                "environment": {
                    CONFIG_ENV_VAR: compose_literal(config.to_json()),
                    TOKEN_ENV_VAR: bridge_token,
                    "TAU2_SEALED_CANDIDATE_ENDPOINT": CANDIDATE_ENDPOINT,
                    **inner_env,
                },
                "volumes": [
                    f"{WORKSPACE_VOLUME}:{WORKSPACE}",
                    f"{BRIDGE_VOLUME}:{BRIDGE_DIR}",
                ],
                # Explicit networking keeps the trusted sidecar (Client,
                # user simulator and brokered candidate model calls) outside
                # the agent-phase allowlist.
                "networks": ["default", SANDBOX_NETWORK],
                "healthcheck": {
                    "test": ["CMD", "test", "-f", f"{STATE_DIR}/{READY_FILE}"],
                    "interval": "2s",
                    "timeout": "5s",
                    "retries": 900,
                },
            },
            CANDIDATE_SERVICE: candidate,
        },
        "networks": {SANDBOX_NETWORK: {"internal": True}},
        "volumes": {WORKSPACE_VOLUME: {}, BRIDGE_VOLUME: {}},
    }
    verifier_compose = {
        "services": {
            "main": {
                "depends_on": {CANDIDATE_SERVICE: {"condition": "service_healthy"}},
                "volumes": [f"{WORKSPACE_VOLUME}:{WORKSPACE}"],
                "networks": ["default", SANDBOX_NETWORK],
                "environment": {"TAU2_SEALED_CANDIDATE_ENDPOINT": CANDIDATE_ENDPOINT},
            },
            CANDIDATE_SERVICE: candidate,
        },
        "networks": {SANDBOX_NETWORK: {"internal": True}},
        "volumes": {WORKSPACE_VOLUME: {}},
    }

    tool_wrapper = (
        "#!/bin/sh\n"
        "# Call the Developer tools from a shell:\n"
        "#   hyper-tau-tool list\n"
        '#   hyper-tau-tool TOOL \'{"argument": "value"}\'\n'
        f"STUB='{STUB_PYTHON} -I {STUB_PATH} --dir {BRIDGE_DIR} "
        f"--token {bridge_token} --timeout {budget.max_time_seconds}'\n"
        'if [ "$1" = "list" ]; then exec $STUB list; fi\n'
        'exec $STUB call "$@"\n'
    )
    claude_environment = " \\\n    ".join(
        f"{name}={value}"
        for name, value in {
            **CLAUDE_CODE_ENVIRONMENT,
            "MCP_TIMEOUT": "60000",
            "MCP_TOOL_TIMEOUT": str(budget.max_time_seconds * 1000),
        }.items()
    )
    main_dockerfile = (
        f"FROM {construction_image}\n"
        "# The Developer's container: the construction toolchain and the kit\n"
        f"# (a volume the {RUNTIME_SERVICE} sidecar builds). Every Developer\n"
        "# tool goes through the stub to that sidecar.\n"
        f"COPY mcp_stub.py {STUB_PATH}\n"
        f"COPY hyper-tau-tool {TOOL_COMMAND}\n"
        f"RUN chmod 0755 {TOOL_COMMAND} && mkdir -p {BRIDGE_DIR} {WORKSPACE}\n"
        "# The settings native runs give Codex and Claude Code: web search,\n"
        "# sub-agents, plugins and memory off.\n"
        "# Local tests, live experiments and sample scenarios can run for\n"
        "# minutes, so tool calls also get the native per-call limit: Claude\n"
        "# Code reads MCP_TOOL_TIMEOUT, Codex its system config.\n"
        f"COPY codex-config.toml {CODEX_SYSTEM_CONFIG}\n"
        f"COPY claude-managed-settings.json {CLAUDE_MANAGED_SETTINGS}\n"
        f"ENV {claude_environment}\n"
        f"WORKDIR {WORKSPACE}\n"
    )
    test_script = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "mkdir -p /logs/verifier\n"
        "exec "
        + shlex.join(
            [
                "python",
                "-m",
                "tau2.harbor.verify",
                "--state-dir",
                str(STATE_DIR),
                "--workspace",
                str(WORKSPACE),
                "--log-dir",
                "/logs/verifier",
            ]
        )
        + "\n"
    )
    verifier_dockerfile = (
        f"FROM {options.runtime_image}\n"
        "COPY test.sh /tests/test.sh\n"
        "RUN chmod 0755 /tests/test.sh\n"
    )
    solve_script = (
        "#!/bin/sh\n"
        "# Hyper-τ has no reference build: this baseline submits the starting\n"
        "# kit unchanged, which exercises the whole construction and scoring\n"
        "# pipeline and scores the untouched scaffold.\n"
        "set -e\n"
        f"exec {TOOL_COMMAND} submit\n"
    )
    write_task(
        task_dir,
        task_toml=dump_task_toml(task_config),
        instruction=instruction + ("" if instruction.endswith("\n") else "\n"),
        files={
            "environment/Dockerfile": main_dockerfile,
            "environment/docker-compose.yaml": dump_compose(compose),
            "environment/mcp_stub.py": Path(mcp_stub.__file__).read_text(),
            "environment/hyper-tau-tool": tool_wrapper,
            "environment/codex-config.toml": codex_config,
            "environment/claude-managed-settings.json": (
                claude_code_managed_settings()
            ),
            "tests/Dockerfile": verifier_dockerfile,
            "tests/docker-compose.yaml": dump_compose(verifier_compose),
            "tests/test.sh": test_script,
            "solution/solve.sh": solve_script,
        },
        executables=(
            "environment/hyper-tau-tool",
            "tests/test.sh",
            "solution/solve.sh",
        ),
    )


def codex_system_config(stub_args: list[str], tool_timeout_sec: int) -> str:
    """Codex's native harness settings, as the agent image's system config.

    Codex merges the config Harbor writes (the task's MCP server, the run's
    options) over this one, so a Harbor run keeps the native settings: web
    search, apps, sub-agents and memories off, and the native per-call tool
    limit, without which the pinned Codex gives up on a call after five
    minutes. Only the native model gateway is left out: Harbor connects
    Codex to the provider itself.
    """
    config = tomllib.loads(
        CodexSandboxBuilder("").render_runtime_config(include_client_tool=False)
    )
    for key in ("model", "model_provider", "model_reasoning_effort"):
        config.pop(key, None)
    config.pop("model_providers")
    server = config["mcp_servers"][MCP_SERVER_NAME]
    # The stub serves exactly the task's tools and needs no environment.
    del server["enabled_tools"], server["env_vars"]
    server.update(
        command=STUB_PYTHON, args=stub_args, tool_timeout_sec=tool_timeout_sec
    )
    return toml.dumps(config)


def claude_code_managed_settings() -> str:
    """Claude Code's native harness settings, as managed settings.

    Managed settings take precedence over user and project settings, so a
    Harbor run keeps the native settings, and the tools the native harness
    leaves out (web search and fetch, sub-agents, scheduling, worktrees)
    stay out of the model's tool list. Permission mode and allowed tools are
    left to Harbor's launch.
    """
    settings = json.loads(
        ClaudeCodeSandboxBuilder("").render_runtime_config(include_client_tool=False)
    )
    denied = settings.pop("permissions")["deny"]
    settings["permissions"] = {"deny": [*denied, *CLAUDE_CODE_EXTRA_TOOLS]}
    return json.dumps(settings, indent=2, sort_keys=True) + "\n"


def candidate_service(image: str) -> dict:
    """The locked-down service that runs the Developer's code."""
    probe = (
        "import socket; "
        f"socket.create_connection(('127.0.0.1', {DEFAULT_PORT}), 5).close()"
    )
    return {
        "image": image,
        "command": [
            "python",
            "-I",
            CANDIDATE_HOST_SCRIPT,
            "--port",
            str(DEFAULT_PORT),
            "--workspace",
            str(WORKSPACE),
        ],
        "volumes": [f"{WORKSPACE_VOLUME}:{WORKSPACE}:ro"],
        # Internal only: candidate code has no route to the internet or the
        # model providers; the trusted side brokers its model calls.
        "networks": [SANDBOX_NETWORK],
        "read_only": True,
        "tmpfs": ["/tmp", "/dev/shm"],
        # Per-session uids need CHOWN, SETUID and SETGID; nothing else.
        "cap_drop": ["ALL"],
        "cap_add": ["CHOWN", "SETUID", "SETGID"],
        "security_opt": ["no-new-privileges:true"],
        "init": True,
        "pids_limit": 4096,
        "healthcheck": {
            "test": ["CMD", "python", "-I", "-c", probe],
            "interval": "2s",
            "timeout": "5s",
            "retries": 60,
        },
    }


def _memory_mb(value, default: int) -> int:
    """Docker-style memory (``4g``, ``512m``, or bytes) in megabytes."""
    if value is None:
        return default
    text = str(value).strip().lower()
    units = {"k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    if text.endswith("b"):
        text = text[:-1]
    return int(int(text) / (1024 * 1024))


__all__ = ["ExportOptions", "export_tasks"]
