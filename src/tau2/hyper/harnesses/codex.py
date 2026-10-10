"""Codex app-server integration for the Hyper-τ construction sandbox."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tau2.hyper.harnesses.factory import CODEX_VERSIONS, DEVELOPER_AUTH_MODES
from tau2.hyper.sandbox.builder import BuildStep
from tau2.hyper.sandbox.native_builder import NativeSandboxBuilder
from tau2.hyper.sandbox.native_runtime import NativeProcessEvent

CODEX_HARNESS_VERSION = "0.162.0"
# The release's pin, used by the paper's Codex rows. It keeps Codex's bundled
# skills, as the release did; the newer pin turns them off.
CODEX_RELEASE_VERSION = "0.144.6"

_SKILLS_SOURCE = Path(__file__).with_name("skills")
_SKILLS_ROOT = "/runtime-home/.codex/skills"


@dataclass(frozen=True)
class DeveloperSkill:
    """An experiment skill, installed where Codex discovers user skills."""

    name: str
    source: Path
    needs_corpus_search: bool

    @property
    def install_path(self) -> str:
        return f"{_SKILLS_ROOT}/{self.name}/SKILL.md"


# --developer-skill choices. Two versions share the name corpus-discovery: the
# method alone, and the method plus the search_corpus paragraphs.
SKILLS = {
    "search-corpus": DeveloperSkill(
        "search-corpus", _SKILLS_SOURCE / "search-corpus" / "SKILL.md", True
    ),
    "method": DeveloperSkill(
        "corpus-discovery",
        _SKILLS_SOURCE / "corpus-discovery-method" / "SKILL.md",
        False,
    ),
    "method-search": DeveloperSkill(
        "corpus-discovery",
        _SKILLS_SOURCE / "corpus-discovery-search" / "SKILL.md",
        True,
    ),
    "search": DeveloperSkill(
        "corpus-search", _SKILLS_SOURCE / "corpus-search" / "SKILL.md", True
    ),
}
# The search_corpus run's skill, installed by --developer-corpus-search alone.
CORPUS_SEARCH_SKILL = SKILLS["search-corpus"].source
CORPUS_SEARCH_SKILL_PATH = SKILLS["search-corpus"].install_path


class CodexSandboxBuilder(NativeSandboxBuilder):
    """Run the pinned real Codex app-server inside the outer Docker boundary."""

    harness_name = "codex"
    harness_version = CODEX_HARNESS_VERSION
    model_gateway_provider = "openai"
    runtime_config_path = "/runtime-home/.codex/config.toml"

    def __init__(
        self,
        llm: str,
        *,
        developer_auth: str = "api-key",
        corpus_search: bool = False,
        developer_skill: str | None = None,
        codex_version: str = CODEX_HARNESS_VERSION,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.corpus_search_enabled = corpus_search
        if developer_auth not in DEVELOPER_AUTH_MODES:
            raise ValueError(
                f"Unsupported developer auth {developer_auth!r}; "
                f"choose from {list(DEVELOPER_AUTH_MODES)}"
            )
        self.developer_auth = developer_auth
        if developer_auth == "chatgpt":
            # Codex still talks only to the gateway; the gateway's chatgpt
            # provider bills the calls to a ChatGPT plan.
            self.model_gateway_provider = "chatgpt"
        if codex_version not in CODEX_VERSIONS:
            raise ValueError(
                f"Unsupported Codex version {codex_version!r}; "
                f"choose from {list(CODEX_VERSIONS)}"
            )
        self.harness_version = codex_version
        if developer_skill is None and corpus_search:
            developer_skill = "search-corpus"
        if developer_skill is not None and developer_skill not in SKILLS:
            raise ValueError(
                f"Unsupported developer skill {developer_skill!r}; "
                f"choose from {list(SKILLS)}"
            )
        if developer_skill and SKILLS[developer_skill].needs_corpus_search:
            if not corpus_search:
                raise ValueError(
                    f"The {developer_skill!r} skill describes search_corpus; "
                    "enable the tool with --developer-corpus-search"
                )
        self.developer_skill = developer_skill

    @property
    def bundled_skills(self) -> bool:
        """Whether Codex offers its bundled skills (the release pin did)."""
        return self.harness_version == CODEX_RELEASE_VERSION

    def harness_config_metadata(self) -> dict:
        metadata = {
            "interface": "app-server-stdio",
            "approval_policy": "never",
            "inner_sandbox": "danger-full-access",
            "web_search": "disabled",
            "apps": False,
            "multi_agent": False,
            "memory": False,
            "bundled_skills": self.bundled_skills,
            "history_persistence": "none",
            "model_gateway": "provider-only/per-run/model-scoped",
            "gateway_token_inherited_by_shell": False,
            "mcp_servers": ["hyper_tau"],
        }
        if self.llm.startswith("gpt-6"):
            # Codex's built-in profile for GPT-6 models fixes these regardless
            # of the [features] table: every tool is called from a JavaScript
            # `exec` tool, and sub-agent tools are offered, but Codex tells the
            # model not to spawn agents unless asked (any effort below ultra).
            metadata["model_profile"] = {
                "code_mode": True,
                "subagent_tools": "offered/explicit-request-only",
            }
        if self.developer_auth == "chatgpt":
            metadata["developer_auth"] = "chatgpt-subscription"
        if self.corpus_search_enabled:
            metadata["extra_tools"] = ["search_corpus"]
        if self.developer_skill:
            metadata["extra_skills"] = [SKILLS[self.developer_skill].name]
            metadata["developer_skill"] = self.developer_skill
        return metadata

    def runtime_environment(self, broker) -> dict[str, str]:
        environment = super().runtime_environment(broker)
        environment["TAU2_DEVELOPER_MODEL"] = self.llm
        reasoning_effort = self.llm_args.get("reasoning_effort")
        if reasoning_effort and reasoning_effort != "none":
            environment["TAU2_DEVELOPER_REASONING_EFFORT"] = str(reasoning_effort)
        return environment

    def render_runtime_config(
        self,
        *,
        include_client_tool: bool,
        include_live_experiment_tool: bool = False,
        include_sample_scenarios_tool: bool = False,
        include_corpus_search_tool: bool = False,
    ) -> str:
        enabled_tools = ["run_local_test", "submit"]
        if include_corpus_search_tool:
            enabled_tools.insert(1, "search_corpus")
        if include_sample_scenarios_tool:
            enabled_tools.insert(1, "run_sample_scenarios")
        if include_live_experiment_tool:
            enabled_tools.insert(1, "run_live_experiment")
        if include_client_tool:
            enabled_tools.insert(1, "talk_to_client")
        tools = ", ".join(json.dumps(tool) for tool in enabled_tools)
        reasoning_effort = self.llm_args.get("reasoning_effort")
        reasoning_line = (
            f"model_reasoning_effort = {json.dumps(str(reasoning_effort))}\n"
            if reasoning_effort and reasoning_effort != "none"
            else ""
        )
        return (
            f"model = {json.dumps(self.llm)}\n"
            'model_provider = "tau2_gateway"\n'
            f"{reasoning_line}"
            'approval_policy = "never"\n'
            'sandbox_mode = "danger-full-access"\n'
            "allow_login_shell = false\n"
            'web_search = "disabled"\n'
            'file_opener = "none"\n'
            "\n[history]\n"
            'persistence = "none"\n'
            "\n[memories]\n"
            "generate_memories = false\n"
            "use_memories = false\n"
            "\n[shell_environment_policy]\n"
            'inherit = "all"\n'
            "ignore_default_excludes = false\n"
            'exclude = ["TAU2_MODEL_GATEWAY_TOKEN"]\n'
            "\n[model_providers.tau2_gateway]\n"
            'name = "Hyper-tau model gateway"\n'
            f'base_url = "http://tau2-model-gateway:8143/'
            f'{self.model_gateway_provider}/v1"\n'
            'env_key = "TAU2_MODEL_GATEWAY_TOKEN"\n'
            'wire_api = "responses"\n'
            "supports_websockets = false\n"
            "supports_standalone_web_search = false\n"
            "\n[features]\n"
            "apps = false\n"
            "remote_plugin = false\n"
            "multi_agent = false\n"
            "goals = false\n"
            "hooks = false\n"
            "memories = false\n"
            "network_proxy = false\n"
            "skill_mcp_dependency_install = false\n"
            "shell_tool = true\n"
            "\n[feedback]\n"
            "enabled = false\n"
            + ("" if self.bundled_skills else "\n[skills.bundled]\nenabled = false\n")
            + "\n[mcp_servers.hyper_tau]\n"
            'command = "/opt/tau2/.venv/bin/python"\n'
            'args = ["-m", "tau2.hyper.sandbox.callback_mcp"]\n'
            "required = true\n"
            "startup_timeout_sec = 30\n"
            "tool_timeout_sec = 28800\n"
            'default_tools_approval_mode = "auto"\n'
            f"enabled_tools = [{tools}]\n"
            'env_vars = ["TAU2_CALLBACK_DIR", "TAU2_CALLBACK_TOKEN", '
            '"TAU2_CALLBACK_TIMEOUT_SECONDS", "TAU2_CLIENT_TOOL_ENABLED", '
            '"TAU2_LIVE_EXPERIMENT_TOOL_ENABLED", '
            '"TAU2_SAMPLE_SCENARIOS_TOOL_ENABLED", '
            + (
                '"TAU2_CORPUS_SEARCH_TOOL_ENABLED", '
                if include_corpus_search_tool
                else ""
            )
            + '"PATH", "PYTHONPATH", "TAU2_DATA_DIR"]\n'
        )

    def runtime_files(
        self,
        *,
        include_corpus_search_tool: bool = False,
        **tool_flags,
    ) -> dict[str, str]:
        files = super().runtime_files(
            include_corpus_search_tool=include_corpus_search_tool, **tool_flags
        )
        if self.developer_skill:
            skill = SKILLS[self.developer_skill]
            files[skill.install_path] = skill.source.read_text()
        return files

    def harness_command(self) -> list[str]:
        return ["python", "-m", "tau2.hyper.harnesses.codex_driver"]

    def normalize_event(self, event: NativeProcessEvent) -> list[BuildStep]:
        if event.channel != "stdout":
            return []
        try:
            message = json.loads(event.text)
        except json.JSONDecodeError:
            return []
        if message.get("method") != "item/completed":
            return []
        item = message.get("params", {}).get("item", {})
        item_type = item.get("type")
        if item_type == "agentMessage":
            return [BuildStep(step_idx=0, role="assistant", content=item.get("text"))]
        if item_type == "reasoning":
            summary = item.get("summary") or []
            if isinstance(summary, str):
                content = summary
            else:
                content = "\n".join(
                    part if isinstance(part, str) else str(part.get("text", ""))
                    for part in summary
                )
            return [BuildStep(step_idx=0, role="assistant", content=content)]
        if item_type == "commandExecution":
            return [
                BuildStep(
                    step_idx=0,
                    role="tool",
                    tool_calls=[
                        {
                            "id": item.get("id", ""),
                            "name": "shell",
                            "arguments": {
                                "command": item.get("command"),
                                "cwd": item.get("cwd"),
                            },
                        }
                    ],
                    tool_results=[
                        {
                            "id": item.get("id", ""),
                            "name": "shell",
                            "result": item.get("aggregatedOutput", ""),
                            "exit_code": item.get("exitCode"),
                            "status": item.get("status"),
                        }
                    ],
                )
            ]
        if item_type == "fileChange":
            return [
                BuildStep(
                    step_idx=0,
                    role="tool",
                    tool_calls=[
                        {
                            "id": item.get("id", ""),
                            "name": "file_change",
                            "arguments": {"changes": item.get("changes", [])},
                        }
                    ],
                    tool_results=[
                        {
                            "id": item.get("id", ""),
                            "name": "file_change",
                            "result": item.get("status", ""),
                        }
                    ],
                )
            ]
        if item_type == "mcpToolCall":
            tool_name = f"mcp:{item.get('server', '')}/{item.get('tool', '')}"
            return [
                BuildStep(
                    step_idx=0,
                    role="tool",
                    tool_calls=[
                        {
                            "id": item.get("id", ""),
                            "name": tool_name,
                            "arguments": item.get("arguments", {}),
                        }
                    ],
                    tool_results=[
                        {
                            "id": item.get("id", ""),
                            "name": tool_name,
                            "result": item.get("result") or item.get("error") or "",
                            "status": item.get("status"),
                        }
                    ],
                )
            ]
        return []
