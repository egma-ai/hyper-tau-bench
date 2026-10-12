"""Small helpers shared by the Harbor runtimes, verifiers, and exporters."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any

import toml
import yaml
from loguru import logger

# Hosts the agent may reach during the agent phase of an exported task:
# the model providers Harbor's installed agents call. A run can add more
# with Harbor's --allowed-host / extra_allowed_hosts.
DEFAULT_AGENT_ALLOWED_HOSTS = (
    "api.openai.com",
    "api.anthropic.com",
    "openrouter.ai",
    "generativelanguage.googleapis.com",
)

# Provider keys forwarded to trusted sidecars and verifiers. Which ones a
# run needs is decided by model_routing.toml.
PROVIDER_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
)

DEFAULT_RUNTIME_IMAGE = "tau2-harbor-runtime:local"


def warm_knowledge_base(domain: str, retrieval_config: str | None = None) -> None:
    """Embed the banking knowledge base once, before any environment is built.

    Each banking_knowledge environment builds its retrieval pipelines from the
    embeddings cache, and a fresh container starts with an empty one, so
    conversations run in parallel would each embed the whole knowledge base.
    A failure here is left for the environments themselves to report.
    """
    if domain != "banking_knowledge":
        return
    from tau2.domains.banking_knowledge.retrieval import DEFAULT_RETRIEVAL_VARIANT
    from tau2.knowledge.embeddings_cache import (
        get_unique_embedder_configs_for_retrieval_configs,
        warm_kb_cache,
    )

    variant = retrieval_config or DEFAULT_RETRIEVAL_VARIANT
    try:
        warm_kb_cache(get_unique_embedder_configs_for_retrieval_configs([variant]))
    except Exception as error:
        logger.warning(f"Could not embed the knowledge base ahead of time: {error}")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON so a concurrent reader never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2, default=str)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_reward(log_dir: Path, rewards: dict[str, float]) -> None:
    """Write Harbor's reward file (``reward`` is the headline metric)."""
    if "reward" not in rewards:
        raise ValueError("Harbor rewards need a 'reward' entry")
    write_json_atomic(
        Path(log_dir) / "reward.json",
        {name: float(value) for name, value in rewards.items()},
    )


def new_token() -> str:
    return secrets.token_hex(24)


def toml_string(value: str) -> str:
    """Quote a string for TOML (JSON string syntax is valid TOML)."""
    return json.dumps(value)


def provider_env(template: str = "${{{name}:-}}") -> dict[str, str]:
    """Map each provider key variable to a Harbor/compose interpolation."""
    return {name: template.format(name=name) for name in PROVIDER_KEY_ENV_VARS}


def compose_literal(value: str) -> str:
    """Escape a value so Docker Compose does not interpolate ``$`` in it."""
    return value.replace("$", "$$")


def dump_compose(document: dict) -> str:
    """Render a compose document (all literal strings already escaped)."""
    # No line folding: long values (such as JSON settings) stay on one line.
    return yaml.safe_dump(document, sort_keys=False, width=1 << 20, allow_unicode=True)


def dump_task_toml(config: dict) -> str:
    """Render task.toml with top-level keys before any table, as TOML requires."""
    scalars = {key: value for key, value in config.items() if not _is_table(value)}
    tables = {key: value for key, value in config.items() if _is_table(value)}
    return toml.dumps(scalars) + "\n" + toml.dumps(tables)


def _is_table(value: Any) -> bool:
    return isinstance(value, dict)


def task_slug(*parts: str) -> str:
    """A Harbor-safe directory and package name built from ``parts``."""
    raw = "__".join(parts)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-.")
    return slug or "task"


def write_task(
    task_dir: Path,
    *,
    task_toml: str,
    instruction: str,
    files: dict[str, str],
    executables: tuple[str, ...] = (),
) -> None:
    """Write one Harbor task directory (replacing an existing one)."""
    task_dir = Path(task_dir)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(task_toml)
    (task_dir / "instruction.md").write_text(instruction)
    for relative, content in files.items():
        path = task_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if relative in executables:
            path.chmod(0o755)
