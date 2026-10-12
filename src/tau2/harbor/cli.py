"""``tau2 harbor``: export the benchmark as Harbor datasets.

``tau2 harbor export-inner`` writes one Harbor task per τ³ customer-service
conversation (the Harbor agent is the customer-service agent);
``tau2 harbor export-outer`` writes one Harbor task per Hyper-τ construction
task (the Harbor agent is the Developer who builds one). Run the results with
``harbor run -p <dataset dir> -a <agent> -m <model>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2.harbor.common import DEFAULT_AGENT_ALLOWED_HOSTS, DEFAULT_RUNTIME_IMAGE


def add_harbor_parser(subparsers) -> None:
    """Register ``tau2 harbor`` and its subcommands."""
    harbor_parser = subparsers.add_parser(
        "harbor",
        help="Export τ³ conversations or Hyper-τ construction tasks as Harbor tasks",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = harbor_parser.add_subparsers(dest="harbor_command", required=True)

    inner = commands.add_parser(
        "export-inner",
        help="One Harbor task per τ³ conversation (the agent serves a customer)",
    )
    inner.add_argument("--domain", required=True, help="τ³ domain, e.g. retail_plus")
    inner.add_argument(
        "--output-dir", type=Path, required=True, help="Dataset directory to write"
    )
    inner.add_argument(
        "--split",
        default="base",
        help="Task split to export (default: base; 'all' for every task)",
    )
    inner.add_argument("--task-ids", nargs="+", default=None)
    inner.add_argument("--num-tasks", type=int, default=None)
    inner.add_argument(
        "--user-llm",
        default=None,
        help="User simulator model (default: Hyper-τ's inner-loop user model)",
    )
    inner.add_argument(
        "--user-llm-args",
        type=json.loads,
        default=None,
        help="User simulator LLM arguments as JSON",
    )
    inner.add_argument(
        "--retrieval-config",
        default=None,
        help="banking_knowledge retrieval configuration (default: bm25)",
    )
    inner.add_argument(
        "--evaluation-type",
        default="all",
        help="tau2 evaluation type used by the verifier (default: all)",
    )
    _add_common_arguments(inner, default_agent_timeout=3600.0, mcp_call_timeout=0.0)
    inner.set_defaults(func=run_export_inner)

    outer = commands.add_parser(
        "export-outer",
        help="One Harbor task per Hyper-τ construction task (the agent builds one)",
    )
    outer.add_argument(
        "--output-dir", type=Path, required=True, help="Dataset directory to write"
    )
    outer.add_argument("--task-ids", nargs="+", default=None)
    outer.add_argument(
        "--source-domain",
        default=None,
        help="Only tasks for this source domain (e.g. airline_plus)",
    )
    outer.add_argument(
        "--construction-image",
        default=None,
        help=(
            "Image for the Developer's container and the sealed candidate "
            "(default: the task's pinned construction runtime image)"
        ),
    )
    outer.add_argument("--client-llm", default=None)
    outer.add_argument("--user-llm", default=None)
    outer.add_argument("--agent-llm", default=None)
    outer.add_argument(
        "--primary-model-only",
        action="store_true",
        help="Offer the constructed agent only its tier's primary model",
    )
    outer.add_argument(
        "--regenerate-client-openings",
        action="store_true",
        help="Ask the Client for a new opening even if an earlier export has one",
    )
    _add_common_arguments(outer, default_agent_timeout=None, mcp_call_timeout=50.0)
    outer.set_defaults(func=run_export_outer)


def _add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_agent_timeout: float | None,
    mcp_call_timeout: float,
) -> None:
    parser.add_argument(
        "--runtime-image",
        default=DEFAULT_RUNTIME_IMAGE,
        help=f"Trusted runtime image (default: {DEFAULT_RUNTIME_IMAGE})",
    )
    parser.add_argument(
        "--agent-network",
        choices=["allowlist", "public"],
        default="allowlist",
        help="Agent-phase network policy (default: model providers only)",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help=(
            "Extra host the agent may reach under the allowlist (repeatable; "
            f"always allowed: {', '.join(DEFAULT_AGENT_ALLOWED_HOSTS)})"
        ),
    )
    parser.add_argument(
        "--agent-timeout-sec",
        type=float,
        default=default_agent_timeout,
        help="Agent-phase time limit"
        + (
            f" (default: {default_agent_timeout:g})"
            if default_agent_timeout
            else " (default: the task's construction budget, eight hours)"
        ),
    )
    parser.add_argument("--verifier-timeout-sec", type=float, default=None)
    parser.add_argument(
        "--mcp-call-timeout",
        type=float,
        default=mcp_call_timeout,
        help=(
            "Seconds after which a still-running tool call returns a job id "
            "for wait_for_result, for MCP clients with a short per-call limit "
            "such as Codex (60 s); 0 disables "
            f"(default: {mcp_call_timeout:g})"
        ),
    )
    parser.add_argument(
        "--allow-legacy-domain",
        action="store_true",
        help="Allow the frozen legacy airline/retail domains (ablations only)",
    )


def _allowed_hosts(args) -> list[str]:
    hosts = list(DEFAULT_AGENT_ALLOWED_HOSTS)
    hosts.extend(host for host in args.allowed_host if host not in hosts)
    return hosts


def run_export_inner(args) -> None:
    from tau2.harbor.inner.export import InnerExportOptions, export_inner_tasks
    from tau2.hyper.run_defaults import DEFAULT_USER_LLM

    options = InnerExportOptions(
        domain=args.domain,
        output_dir=args.output_dir,
        split=None if args.split == "all" else args.split,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        runtime_image=args.runtime_image,
        user_llm=args.user_llm or DEFAULT_USER_LLM,
        user_llm_args=args.user_llm_args,
        retrieval_config=args.retrieval_config,
        evaluation_type=args.evaluation_type,
        agent_network=args.agent_network,
        allowed_hosts=_allowed_hosts(args),
        agent_timeout_sec=args.agent_timeout_sec,
        mcp_call_timeout_sec=args.mcp_call_timeout,
        allow_legacy_domain=args.allow_legacy_domain,
    )
    if args.verifier_timeout_sec is not None:
        options.verifier_timeout_sec = args.verifier_timeout_sec
    written = export_inner_tasks(options)
    print(f"Wrote {len(written)} Harbor tasks to {args.output_dir}")


def run_export_outer(args) -> None:
    from tau2.harbor.outer.export import OuterExportOptions, export_outer_tasks

    options = OuterExportOptions(
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
    written = export_outer_tasks(options)
    print(f"Wrote {len(written)} Harbor tasks to {args.output_dir}")
