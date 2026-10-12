"""Tests for exporting τ³ tasks as Harbor tasks."""

from __future__ import annotations

import json
import tomllib

import pytest
import yaml

from tau2.harbor.inner.export import (
    InnerExportOptions,
    export_inner_tasks,
    resolve_user_llm_args,
)


def export(tmp_path, **kwargs):
    options = InnerExportOptions(domain="mock", output_dir=tmp_path / "dataset")
    for key, value in kwargs.items():
        setattr(options, key, value)
    return export_inner_tasks(options)


def test_exports_one_complete_task_per_conversation(tmp_path):
    written = export(tmp_path, task_ids=["create_task_1"])
    assert [path.name for path in written] == ["mock__create_task_1"]
    task_dir = written[0]
    files = sorted(
        str(path.relative_to(task_dir))
        for path in task_dir.rglob("*")
        if path.is_file()
    )
    assert files == [
        "environment/Dockerfile",
        "environment/docker-compose.yaml",
        "environment/mcp_stub.py",
        "environment/tau2-tool",
        "instruction.md",
        "solution/solve.sh",
        "task.toml",
        "tests/Dockerfile",
        "tests/test.sh",
    ]

    config = tomllib.loads((task_dir / "task.toml").read_text())
    assert config["task"]["name"] == "tau2/mock__create_task_1"
    assert config["artifacts"] == [
        {"source": "/var/lib/tau2-harbor", "service": "tau-runtime"}
    ]
    assert config["agent"]["network_mode"] == "allowlist"
    assert "api.openai.com" in config["agent"]["allowed_hosts"]
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["verifier"]["env"]["OPENAI_API_KEY"] == "${OPENAI_API_KEY:-}"
    (server,) = config["environment"]["mcp_servers"]
    assert server["transport"] == "stdio"
    token = server["args"][server["args"].index("--token") + 1]

    compose = yaml.safe_load(
        (task_dir / "environment" / "docker-compose.yaml").read_text()
    )
    runtime = compose["services"]["tau-runtime"]
    assert runtime["environment"]["TAU2_HARBOR_BRIDGE_TOKEN"] == token
    assert runtime["networks"] == ["default"]
    assert "networks" not in compose["services"]["main"]
    assert runtime["command"][runtime["command"].index("--task-id") + 1] == (
        "create_task_1"
    )
    user_args = runtime["command"][runtime["command"].index("--user-llm-args") + 1]
    assert json.loads(user_args) == {"reasoning_effort": "none"}

    oracle_token = runtime["environment"]["TAU2_HARBOR_ORACLE_TOKEN"]
    assert oracle_token != token
    solve = (task_dir / "solution" / "solve.sh").read_text()
    assert oracle_token in solve and "oracle_replay" in solve
    # Only the solution and the sidecar know the oracle token.
    for visible in (
        "instruction.md",
        "environment/Dockerfile",
        "environment/tau2-tool",
    ):
        assert oracle_token not in (task_dir / visible).read_text()
    assert token in (task_dir / "environment" / "tau2-tool").read_text()

    instruction = (task_dir / "instruction.md").read_text()
    assert "start_conversation" in instruction and "<policy>" in instruction
    assert "Important Meeting" not in instruction  # no user scenario leaks
    test_sh = (task_dir / "tests" / "test.sh").read_text()
    assert "tau2.harbor.inner.verify" in test_sh and "create_task_1" in test_sh


def test_dollar_signs_reach_the_sidecar_literally(tmp_path):
    (task_dir,) = export(
        tmp_path, task_ids=["create_task_1"], user_llm_args={"stop": "$5"}
    )
    raw = (task_dir / "environment" / "docker-compose.yaml").read_text()
    assert "$$5" in raw


def test_legacy_domains_are_refused(tmp_path):
    with pytest.raises(ValueError, match="airline_plus"):
        export_inner_tasks(InnerExportOptions(domain="airline", output_dir=tmp_path))


def test_public_network_and_call_timeout_options(tmp_path):
    (task_dir,) = export(
        tmp_path,
        task_ids=["create_task_1"],
        agent_network="public",
        mcp_call_timeout_sec=50,
    )
    config = tomllib.loads((task_dir / "task.toml").read_text())
    assert "network_mode" not in config["agent"]
    args = config["environment"]["mcp_servers"][0]["args"]
    assert args[args.index("--call-timeout") + 1] == "50"


def test_user_llm_args_default_by_model_family():
    assert resolve_user_llm_args("gpt-5.5", None) == {"reasoning_effort": "none"}
    assert resolve_user_llm_args("gpt-4.1-2025-04-14", None) == {"temperature": 0.0}
    assert resolve_user_llm_args("gpt-5.5", {"temperature": 1}) == {"temperature": 1}


def test_split_selection_and_limits(tmp_path):
    written = export(tmp_path, num_tasks=3)
    assert len(written) == 3
