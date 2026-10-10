"""Select a coding-agent harness without coupling the CLI to adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tau2.hyper.sandbox.builder import SandboxBuilder

DEFAULT_DEVELOPER_HARNESS = "codex"
DEVELOPER_HARNESSES = ("codex", "claude-code", "opencode", "prime-agent")
# How the Developer seat's model calls are billed: an API key (default), or a
# ChatGPT plan through a Codex login (codex harness only).
DEVELOPER_AUTH_MODES = ("api-key", "chatgpt")
# Codex builds this checkout can run: the current pin (default) and the
# release's pin, which the paper's Codex rows used. The construction image
# must be built with the same CODEX_VERSION.
CODEX_VERSIONS = ("0.162.0", "0.144.6")
# Experiment skills (codex harness only): the search_corpus run's skill, and
# the telecom experiment's method, method + search and search-only versions.
DEVELOPER_SKILLS = ("search-corpus", "method", "method-search", "search")


def create_developer_builder(
    developer_harness: str,
    developer_llm: str,
    developer_llm_args: dict | None,
    developer_reasoning_effort: str | None,
    developer_auth: str = "api-key",
    developer_corpus_search: bool = False,
    developer_skill: str | None = None,
    developer_codex_version: str | None = None,
) -> SandboxBuilder:
    """Build the selected coding-agent integration for a construction run."""
    native_llm_args = dict(developer_llm_args or {})
    if developer_reasoning_effort and developer_reasoning_effort != "none":
        native_llm_args["reasoning_effort"] = developer_reasoning_effort
    if developer_auth != "api-key" and developer_harness != "codex":
        raise ValueError(
            f"Developer auth {developer_auth!r} is only supported by the codex harness"
        )
    if developer_corpus_search and developer_harness != "codex":
        raise ValueError("search_corpus is only wired for the codex harness")
    if developer_skill and developer_harness != "codex":
        raise ValueError("Developer skills are only wired for the codex harness")
    if developer_codex_version and developer_harness != "codex":
        raise ValueError("--developer-codex-version applies to the codex harness")

    if developer_harness == "codex":
        from tau2.hyper.harnesses.codex import (
            CODEX_HARNESS_VERSION,
            CodexSandboxBuilder,
        )

        return CodexSandboxBuilder(
            llm=developer_llm,
            llm_args=native_llm_args,
            developer_auth=developer_auth,
            corpus_search=developer_corpus_search,
            developer_skill=developer_skill,
            codex_version=developer_codex_version or CODEX_HARNESS_VERSION,
        )
    if developer_harness == "claude-code":
        from tau2.hyper.harnesses.claude import ClaudeCodeSandboxBuilder

        return ClaudeCodeSandboxBuilder(llm=developer_llm, llm_args=native_llm_args)
    if developer_harness == "opencode":
        from tau2.hyper.harnesses.opencode import OpenCodeSandboxBuilder

        return OpenCodeSandboxBuilder(llm=developer_llm, llm_args=native_llm_args)
    if developer_harness == "prime-agent":
        from tau2.hyper.harnesses.prime import PrimeAgentSandboxBuilder

        return PrimeAgentSandboxBuilder(llm=developer_llm, llm_args=native_llm_args)
    raise NotImplementedError(
        f"The {developer_harness!r} harness is selected but its native "
        "runtime adapter is not installed"
    )
