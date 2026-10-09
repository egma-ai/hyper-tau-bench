"""Construction runs select supported native coding harnesses."""

from tau2.hyper.harnesses.factory import (
    DEFAULT_DEVELOPER_HARNESS,
    create_developer_builder,
)


def test_codex_is_the_default_developer_harness():
    assert DEFAULT_DEVELOPER_HARNESS == "codex"


def test_codex_selection_builds_the_native_adapter():
    from tau2.hyper.harnesses.codex import CodexSandboxBuilder

    builder = create_developer_builder(
        "codex", "gpt-5.6-sol", {"reasoning_effort": "xhigh"}, "xhigh"
    )
    assert isinstance(builder, CodexSandboxBuilder)
    assert builder.llm == "gpt-5.6-sol"


def test_codex_receives_the_effort_flag_without_llm_args():
    """The --developer-reasoning-effort flag alone must reach Codex.

    Regression guard: Codex used to get only the CLI-built llm_args, which
    carry no effort for model families the CLI does not recognise, so a run
    silently fell back to the model's default effort.
    """
    builder = create_developer_builder("codex", "gpt-6.1-sol", None, "max")

    assert builder.llm_args == {"reasoning_effort": "max"}
    config = builder.render_runtime_config(include_client_tool=False)
    assert 'model_reasoning_effort = "max"' in config


def test_claude_code_selection_builds_the_native_adapter():
    from tau2.hyper.harnesses.claude import ClaudeCodeSandboxBuilder

    builder = create_developer_builder("claude-code", "claude-opus-4-6", {}, "high")
    assert isinstance(builder, ClaudeCodeSandboxBuilder)
    assert builder.llm_args.get("reasoning_effort") == "high"


def test_workbench_launch_seams_use_the_default_harness_factory():
    """The web app shares the CLI's default harness selection."""
    from pathlib import Path

    import tau2.hyper.web.app as app_module

    source = Path(app_module.__file__).read_text()
    assert "create_developer_builder" in source
    assert "DEFAULT_DEVELOPER_HARNESS" in source


def test_chatgpt_developer_auth_is_codex_only():
    import pytest

    with pytest.raises(ValueError, match="only supported by the codex harness"):
        create_developer_builder(
            "claude-code", "claude-opus-4-6", {}, "high", developer_auth="chatgpt"
        )
