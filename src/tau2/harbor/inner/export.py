"""Export τ³ tasks as a Harbor dataset: one customer-service conversation each.

Each exported task has three containers:

- ``main`` — where the Harbor agent runs. It holds nothing but a small
  stdio MCP server (:mod:`tau2.harbor.mcp_stub`) that relays tool calls to
  the sidecar over files in a shared volume, plus a ``tau2-tool`` shell
  wrapper for agents without MCP support.
- ``tau-runtime`` — the trusted sidecar (:mod:`tau2.harbor.inner.runtime`)
  with the domain environment, the user simulator, and the tau2
  orchestrator. It is the only place the task's user scenario lives.
- the verifier (``tests/``) — scores the recorded conversation with the
  tau2 evaluator (:mod:`tau2.harbor.inner.verify`).

Both trusted containers run the ``tau2-harbor-runtime`` image built from
this repository (``make harbor-images``).
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.config import DEFAULT_LLM_ARGS_USER
from tau2.data_model.tasks import Task
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
from tau2.harbor.inner.runtime import (
    END_TOOL,
    ORACLE_TOOL,
    SEND_TOOL,
    START_TOOL,
    STATE_FILE,
    build_run_config,
)
from tau2.hyper.run_defaults import DEFAULT_USER_LLM, supports_reasoning_effort
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.runner.build import _build_env_kwargs, build_environment
from tau2.runner.helpers import get_tasks

RUNTIME_SERVICE = "tau-runtime"
BRIDGE_DIR = "/var/run/tau2-bridge"
STATE_DIR = "/var/lib/tau2-harbor"
STUB_PATH = "/opt/tau2-harbor/mcp_stub.py"
TOOL_COMMAND = "/usr/local/bin/tau2-tool"
MCP_SERVER_NAME = "tau2"
MAIN_BASE_IMAGE = "python:3.12-slim"
STUB_PYTHON = "/usr/local/bin/python3"

# Hyper-τ maintains the plus corpora; the canonical airline and retail
# domains are frozen legacy baselines (see AGENTS.md).
LEGACY_DOMAINS = {"airline": "airline_plus", "retail": "retail_plus"}
DEFAULT_RETRIEVAL_CONFIGS = {"banking_knowledge": "bm25"}

INSTRUCTION = """\
You are a customer service agent. Help the customer according to the \
<policy> below.

You talk to the customer, and act on their behalf, only through the tools of \
the `{server}` MCP server:

- Call `{start}` first. The conversation has already been opened for you \
with "{greeting}", and the tool returns the customer's first message.
- Call `{send}` to say something to the customer. It returns the \
customer's reply.
- Call the other tools on the server to look up or change records, as the \
policy allows. Each returns its result.
- When the customer's request is fully handled, call `{end}`. If the \
customer ends the conversation, you are done.

In each step either send one message to the customer or make one tool call, \
never both. The customer sees only what you send with `{send}`. Nothing \
outside the `{server}` tools affects the outcome: there are no files to read \
or edit.

If you cannot call MCP tools directly, run the same tools from a shell: \
`tau2-tool list` shows them, and `tau2-tool <tool> '<JSON arguments>'` calls \
one, for example `tau2-tool {send} '{{"message": "Hello"}}'`.

<policy>
{policy}
</policy>
"""


@dataclass
class InnerExportOptions:
    """What to export and how the exported tasks run."""

    domain: str
    output_dir: Path
    split: Optional[str] = "base"
    task_ids: Optional[list[str]] = None
    num_tasks: Optional[int] = None
    runtime_image: str = DEFAULT_RUNTIME_IMAGE
    user_llm: str = DEFAULT_USER_LLM
    user_llm_args: Optional[dict] = None
    retrieval_config: Optional[str] = None
    evaluation_type: str = "all"
    agent_network: str = "allowlist"
    allowed_hosts: list[str] = field(
        default_factory=lambda: list(DEFAULT_AGENT_ALLOWED_HOSTS)
    )
    agent_timeout_sec: float = 3600.0
    verifier_timeout_sec: float = 1800.0
    mcp_call_timeout_sec: float = 0.0
    allow_legacy_domain: bool = False


def resolve_user_llm_args(model: str, user_llm_args: Optional[dict]) -> dict:
    """The user simulator's LLM arguments: explicit, else the seat default.

    Reasoning models get Hyper-τ's maintained inner-loop setting (no
    reasoning); others get tau2's deterministic default.
    """
    if user_llm_args is not None:
        return dict(user_llm_args)
    if supports_reasoning_effort(model):
        return {"reasoning_effort": "none"}
    return dict(DEFAULT_LLM_ARGS_USER)


def check_domain(domain: str, allow_legacy_domain: bool) -> None:
    if domain in LEGACY_DOMAINS and not allow_legacy_domain:
        raise ValueError(
            f"{domain!r} is a frozen legacy domain; export "
            f"{LEGACY_DOMAINS[domain]!r} instead (or pass --allow-legacy-domain "
            "for an explicit ablation)."
        )
    if domain.startswith("hyper_"):
        raise ValueError(
            "Hyper-τ construction tasks are exported with `tau2 harbor "
            "export-outer`, not as inner-loop conversations."
        )


def load_export_tasks(options: InnerExportOptions) -> list[Task]:
    split = options.split
    try:
        return get_tasks(
            options.domain,
            task_split_name=split,
            task_ids=options.task_ids,
            num_tasks=options.num_tasks,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        if split is None or "split" not in str(error).lower():
            raise
        logger.warning(f"{options.domain} has no {split!r} split; using all tasks")
        return get_tasks(
            options.domain, task_ids=options.task_ids, num_tasks=options.num_tasks
        )


def export_inner_tasks(options: InnerExportOptions) -> list[Path]:
    """Write one Harbor task per τ³ task and return their directories."""
    check_domain(options.domain, options.allow_legacy_domain)
    retrieval_config = options.retrieval_config or DEFAULT_RETRIEVAL_CONFIGS.get(
        options.domain
    )
    user_llm_args = resolve_user_llm_args(options.user_llm, options.user_llm_args)
    tasks = load_export_tasks(options)
    if not tasks:
        raise ValueError(f"No tasks selected in domain {options.domain!r}")
    config = build_run_config(options.domain, retrieval_config=retrieval_config)
    output_dir = Path(options.output_dir)
    written = []
    for task in tasks:
        environment = build_environment(
            options.domain, env_kwargs=_build_env_kwargs(config, task)
        )
        task_dir = output_dir / task_slug(options.domain, task.id)
        write_inner_task(
            task_dir,
            task,
            options,
            policy=environment.get_policy(),
            retrieval_config=retrieval_config,
            user_llm_args=user_llm_args,
        )
        written.append(task_dir)
    logger.info(f"Wrote {len(written)} Harbor tasks to {output_dir}")
    return written


def write_inner_task(
    task_dir: Path,
    task: Task,
    options: InnerExportOptions,
    *,
    policy: str,
    retrieval_config: Optional[str],
    user_llm_args: dict,
) -> None:
    bridge_token = new_token()
    oracle_token = new_token()
    domain = options.domain
    name = task_slug(domain, task.id)

    stub_args = ["--dir", BRIDGE_DIR, "--token", bridge_token]
    if options.mcp_call_timeout_sec > 0:
        stub_args += ["--call-timeout", f"{options.mcp_call_timeout_sec:g}"]

    agent: dict = {"timeout_sec": float(options.agent_timeout_sec)}
    if options.agent_network == "allowlist":
        agent["network_mode"] = "allowlist"
        agent["allowed_hosts"] = list(options.allowed_hosts)
    elif options.agent_network != "public":
        raise ValueError(f"Unknown agent network mode {options.agent_network!r}")

    task_config = {
        "schema_version": "1.4",
        # The sidecar's recorded conversation is what the verifier scores.
        "artifacts": [{"source": STATE_DIR, "service": RUNTIME_SERVICE}],
        "task": {
            "name": f"tau2/{name}",
            "description": (
                f"τ³ customer-service conversation: {domain} task {task.id}"
            ),
            "keywords": ["tau2", "tau3-bench", "hyper-tau", domain, "customer-service"],
        },
        "metadata": {
            "benchmark": "tau2",
            "loop": "inner",
            "domain": domain,
            "task_id": task.id,
            "split": options.split or "",
            "user_llm": options.user_llm,
            "user_llm_args": json.dumps(user_llm_args, sort_keys=True),
            "evaluation_type": options.evaluation_type,
            "retrieval_config": retrieval_config or "",
        },
        "agent": agent,
        "verifier": {
            "timeout_sec": float(options.verifier_timeout_sec),
            "environment_mode": "separate",
            "env": provider_env(),
        },
        "environment": {
            "build_timeout_sec": 1800.0,
            "cpus": 2,
            "memory_mb": 4096,
            "mcp_servers": [
                {
                    "name": MCP_SERVER_NAME,
                    "transport": "stdio",
                    "command": STUB_PYTHON,
                    # -I: the stub needs nothing from the environment.
                    "args": ["-I", STUB_PATH, *stub_args],
                }
            ],
        },
    }

    runtime_command = [
        "python",
        "-m",
        "tau2.harbor.inner.runtime",
        "--domain",
        domain,
        "--task-id",
        task.id,
        "--bridge-dir",
        BRIDGE_DIR,
        "--state-dir",
        STATE_DIR,
        "--user-llm",
        options.user_llm,
        "--user-llm-args",
        json.dumps(user_llm_args, sort_keys=True),
    ]
    if retrieval_config:
        runtime_command += ["--retrieval-config", retrieval_config]
    runtime_environment = {
        "TAU2_HARBOR_BRIDGE_TOKEN": bridge_token,
        "TAU2_HARBOR_ORACLE_TOKEN": oracle_token,
    }
    compose = {
        "services": {
            "main": {
                "depends_on": {RUNTIME_SERVICE: {"condition": "service_healthy"}},
                "volumes": [f"tau2-bridge:{BRIDGE_DIR}"],
            },
            RUNTIME_SERVICE: {
                "image": options.runtime_image,
                "command": [compose_literal(part) for part in runtime_command],
                "environment": {
                    **runtime_environment,
                    # A run can swap the user simulator without re-exporting.
                    "TAU2_HARBOR_USER_LLM": "${TAU2_HARBOR_USER_LLM:-}",
                    "TAU2_HARBOR_USER_LLM_ARGS": "${TAU2_HARBOR_USER_LLM_ARGS:-}",
                    **provider_env(),
                },
                "volumes": [f"tau2-bridge:{BRIDGE_DIR}"],
                # Explicit networking keeps the sidecar (and its model calls
                # for the user simulator) outside the agent-phase allowlist.
                "networks": ["default"],
                "healthcheck": {
                    "test": ["CMD", "test", "-f", f"{STATE_DIR}/ready"],
                    "interval": "2s",
                    "timeout": "5s",
                    "retries": 150,
                },
            },
        },
        "volumes": {"tau2-bridge": {}},
    }

    tool_wrapper = (
        "#!/bin/sh\n"
        "# Call this task's conversation tools from a shell:\n"
        "#   tau2-tool list\n"
        '#   tau2-tool TOOL \'{"argument": "value"}\'\n'
        f"STUB='{STUB_PYTHON} -I {STUB_PATH} --dir {BRIDGE_DIR} "
        f"--token {bridge_token}'\n"
        'if [ "$1" = "list" ]; then exec $STUB list; fi\n'
        'exec $STUB call "$@"\n'
    )
    main_dockerfile = (
        f"FROM {MAIN_BASE_IMAGE}\n"
        "# The agent's container holds no task state: every tool call goes\n"
        f"# through the stub to the {RUNTIME_SERVICE} sidecar.\n"
        f"COPY mcp_stub.py {STUB_PATH}\n"
        f"COPY tau2-tool {TOOL_COMMAND}\n"
        f"RUN chmod 0755 {TOOL_COMMAND} && mkdir -p {BRIDGE_DIR}\n"
        "# Customer replies wait on the user simulator's model.\n"
        "ENV MCP_TIMEOUT=60000 MCP_TOOL_TIMEOUT=7200000\n"
        "WORKDIR /app\n"
    )
    verify_command = [
        "python",
        "-m",
        "tau2.harbor.inner.verify",
        "--domain",
        domain,
        "--task-id",
        task.id,
        "--state",
        f"{STATE_DIR}/{STATE_FILE}",
        "--log-dir",
        "/logs/verifier",
        "--evaluation-type",
        options.evaluation_type,
    ]
    if retrieval_config:
        verify_command += ["--retrieval-config", retrieval_config]
    test_script = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "mkdir -p /logs/verifier\n"
        f"exec {shlex.join(verify_command)}\n"
    )
    verifier_dockerfile = (
        f"FROM {options.runtime_image}\n"
        "COPY test.sh /tests/test.sh\n"
        "RUN chmod 0755 /tests/test.sh\n"
    )
    oracle_arguments = json.dumps({"oracle_token": oracle_token})
    solve_script = (
        "#!/bin/sh\n"
        "# Replays the task's reference actions as the conversation.\n"
        "set -e\n"
        f"exec {TOOL_COMMAND} {ORACLE_TOOL} {shlex.quote(oracle_arguments)}\n"
    )
    instruction = INSTRUCTION.format(
        server=MCP_SERVER_NAME,
        start=START_TOOL,
        send=SEND_TOOL,
        end=END_TOOL,
        greeting=DEFAULT_FIRST_AGENT_MESSAGE.content,
        policy=policy.strip(),
    )
    write_task(
        task_dir,
        task_toml=dump_task_toml(task_config),
        instruction=instruction,
        files={
            "environment/Dockerfile": main_dockerfile,
            "environment/docker-compose.yaml": dump_compose(compose),
            "environment/mcp_stub.py": Path(mcp_stub.__file__).read_text(),
            "environment/tau2-tool": tool_wrapper,
            "tests/Dockerfile": verifier_dockerfile,
            "tests/test.sh": test_script,
            "solution/solve.sh": solve_script,
        },
        executables=(
            "environment/tau2-tool",
            "tests/test.sh",
            "solution/solve.sh",
        ),
    )


__all__ = ["InnerExportOptions", "export_inner_tasks", "resolve_user_llm_args"]
