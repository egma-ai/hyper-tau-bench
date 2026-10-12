"""``tau2 harbor``: export the Hyper-τ construction tasks as a Harbor dataset.

``tau2 harbor export`` writes one Harbor task per Hyper-τ construction task:
the Harbor agent is the Developer who builds a customer-service agent, and
the task's verifier scores that agent on the hidden test conversations, as
``tau2 hyper-tau`` does. Run the result with
``harbor run -p <dataset dir> -a <agent> -m <model>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tau2.harbor.common import DEFAULT_AGENT_ALLOWED_HOSTS, DEFAULT_RUNTIME_IMAGE


def add_harbor_parser(subparsers) -> None:
    """Register ``tau2 harbor`` and its subcommands."""
    harbor_parser = subparsers.add_parser(
        "harbor",
        help="Export the Hyper-τ construction tasks as Harbor tasks",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = harbor_parser.add_subparsers(dest="harbor_command", required=True)

    export = commands.add_parser(
        "export",
        help="One Harbor task per Hyper-τ construction task (the agent builds one)",
    )
    export.add_argument(
        "--output-dir", type=Path, required=True, help="Dataset directory to write"
    )
    export.add_argument("--task-ids", nargs="+", default=None)
    export.add_argument(
        "--source-domain",
        default=None,
        help="Only tasks for this source domain (e.g. airline_plus)",
    )
    export.add_argument(
        "--runtime-image",
        default=DEFAULT_RUNTIME_IMAGE,
        help=f"Trusted runtime image (default: {DEFAULT_RUNTIME_IMAGE})",
    )
    export.add_argument(
        "--construction-image",
        default=None,
        help=(
            "Image for the Developer's container and the sealed candidate "
            "(default: the task's pinned construction runtime image)"
        ),
    )
    export.add_argument("--client-llm", default=None)
    export.add_argument("--user-llm", default=None)
    export.add_argument("--agent-llm", default=None)
    export.add_argument(
        "--primary-model-only",
        action="store_true",
        help="Offer the constructed agent only its tier's primary model",
    )
    export.add_argument(
        "--regenerate-client-openings",
        action="store_true",
        help="Ask the Client for a new opening even if an earlier export has one",
    )
    export.add_argument(
        "--agent-network",
        choices=["allowlist", "public"],
        default="allowlist",
        help="Agent-phase network policy (default: model providers only)",
    )
    export.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help=(
            "Extra host the agent may reach under the allowlist (repeatable; "
            f"always allowed: {', '.join(DEFAULT_AGENT_ALLOWED_HOSTS)})"
        ),
    )
    export.add_argument(
        "--agent-timeout-sec",
        type=float,
        default=None,
        help="Agent-phase time limit (default: the task's construction budget, "
        "eight hours)",
    )
    export.add_argument("--verifier-timeout-sec", type=float, default=None)
    export.add_argument(
        "--mcp-call-timeout",
        type=float,
        default=0.0,
        help=(
            "Seconds after which a still-running tool call returns a job id "
            "for wait_for_result, for MCP clients that cap a call below a "
            "tool's running time (default: 0, calls block until they finish "
            "as in native runs)"
        ),
    )
    export.add_argument(
        "--allow-legacy-domain",
        action="store_true",
        help="Allow the frozen legacy airline/retail domains (ablations only)",
    )
    export.set_defaults(func=run_export)


def _allowed_hosts(args) -> list[str]:
    hosts = list(DEFAULT_AGENT_ALLOWED_HOSTS)
    hosts.extend(host for host in args.allowed_host if host not in hosts)
    return hosts


def run_export(args) -> None:
    from tau2.harbor.export import ExportOptions, export_tasks

    options = ExportOptions(
        output_dir=args.output_dir,
        task_ids=args.task_ids,
        source_domain=args.source_domain,
        runtime_image=args.runtime_image,
        construction_image=args.construction_image,
        client_llm=args.client_llm,
        user_llm=args.user_llm,
        agent_llm=args.agent_llm,
        primary_model_only=args.primary_model_only,
        agent_network=args.agent_network,
        allowed_hosts=_allowed_hosts(args),
        agent_timeout_sec=args.agent_timeout_sec,
        mcp_call_timeout_sec=args.mcp_call_timeout,
        reuse_client_openings=not args.regenerate_client_openings,
        allow_legacy_domain=args.allow_legacy_domain,
    )
    if args.verifier_timeout_sec is not None:
        options.verifier_timeout_sec = args.verifier_timeout_sec
    written = export_tasks(options)
    print(f"Wrote {len(written)} Harbor tasks to {args.output_dir}")
