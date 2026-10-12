"""Tests for exporting Hyper-τ construction tasks as Harbor tasks."""

from __future__ import annotations

import json
import tomllib

import pytest
import yaml

from tau2.harbor.outer.export import (
    OuterExportOptions,
    _memory_mb,
    export_outer_tasks,
)
from tau2.harbor.outer.runtime import CONFIG_ENV_VAR, TOKEN_ENV_VAR, ConstructionConfig
from tau2.hyper.sandbox.orchestrator import SandboxOrchestrator

NO_CLIENT_TASK = "008_retail_plus_construction_core_evidence_performance_hard"
CLIENT_TASK = (
    "004_airline_plus_construction_core_evidence_hard_client_performance_medium"
)
OPENING = "Hi, I run airline support. Rebooking fees are $25 (or ${FEE})."


def export(tmp_path, *task_ids, **kwargs):
    options = OuterExportOptions(
        output_dir=tmp_path / "dataset", task_ids=list(task_ids)
    )
    for key, value in kwargs.items():
        setattr(options, key, value)
    return export_outer_tasks(options)


def compose_value(value: str) -> str:
    """What Docker Compose passes on for an escaped literal."""
    return value.replace("$$", "$")


def load(task_dir):
    config = tomllib.loads((task_dir / "task.toml").read_text())
    compose = yaml.safe_load(
        (task_dir / "environment" / "docker-compose.yaml").read_text()
    )
    verifier = yaml.safe_load((task_dir / "tests" / "docker-compose.yaml").read_text())
    return config, compose, verifier


@pytest.fixture
def client_openings(monkeypatch):
    calls = []

    def generate_client_opening(self):
        calls.append(self.task.id)
        return OPENING

    monkeypatch.setattr(
        SandboxOrchestrator, "generate_client_opening", generate_client_opening
    )
    return calls


def test_exports_a_complete_construction_task(tmp_path, client_openings):
    (task_dir,) = export(tmp_path, NO_CLIENT_TASK)
    assert task_dir.name == NO_CLIENT_TASK
    files = sorted(
        str(path.relative_to(task_dir))
        for path in task_dir.rglob("*")
        if path.is_file()
    )
    assert files == [
        "environment/Dockerfile",
        "environment/docker-compose.yaml",
        "environment/hyper-tau-tool",
        "environment/mcp_stub.py",
        "instruction.md",
        "solution/solve.sh",
        "task.toml",
        "tests/Dockerfile",
        "tests/docker-compose.yaml",
        "tests/test.sh",
    ]
    assert client_openings == []  # no Client, no model call

    config, compose, verifier = load(task_dir)
    assert config["task"]["name"] == f"hyper-tau/{NO_CLIENT_TASK}"
    assert config["metadata"]["client_enabled"] is False
    assert config["artifacts"] == [
        {"source": "/var/lib/hyper-tau", "service": "hyper-runtime"}
    ]
    assert config["agent"]["timeout_sec"] == 28800.0
    assert config["agent"]["network_mode"] == "allowlist"
    assert config["verifier"]["environment_mode"] == "separate"
    (hook,) = config["verifier"]["collect"]
    assert hook["service"] == "hyper-runtime"
    assert "tau2.harbor.outer.runtime finalize" in hook["command"]
    assert config["environment"]["workdir"] == "/workspace"
    (server,) = config["environment"]["mcp_servers"]
    assert server["name"] == "hyper_tau"
    assert server["args"][:2] == ["-I", "/opt/hyper-tau-harbor/mcp_stub.py"]
    assert server["args"][server["args"].index("--call-timeout") + 1] == "50"
    token = server["args"][server["args"].index("--token") + 1]

    services = compose["services"]
    assert set(services) == {"main", "hyper-runtime", "candidate"}
    # The agent's container keeps Harbor's egress control; the trusted
    # sidecar and the candidate declare their own networks.
    assert "networks" not in services["main"]
    assert services["main"]["depends_on"] == {
        "hyper-runtime": {"condition": "service_healthy"}
    }
    runtime = services["hyper-runtime"]
    assert runtime["networks"] == ["default", "sandbox"]
    assert runtime["environment"][TOKEN_ENV_VAR] == token
    assert runtime["environment"]["TAU2_SEALED_CANDIDATE_ENDPOINT"] == "candidate:7000"
    settings = ConstructionConfig.from_json(
        compose_value(runtime["environment"][CONFIG_ENV_VAR])
    )
    assert settings.task_id == NO_CLIENT_TASK
    assert settings.bridge_token == ""
    assert settings.max_time_seconds == 28800

    candidate = services["candidate"]
    assert candidate["networks"] == ["sandbox"]
    assert compose["networks"]["sandbox"] == {"internal": True}
    assert candidate["volumes"] == ["workspace:/workspace:ro"]
    assert candidate["read_only"] is True
    assert candidate["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in candidate["security_opt"]
    assert verifier["services"]["candidate"] == candidate
    assert verifier["services"]["main"]["environment"] == {
        "TAU2_SEALED_CANDIDATE_ENDPOINT": "candidate:7000"
    }

    instruction = (task_dir / "instruction.md").read_text()
    assert instruction.startswith("Work autonomously as the Developer")
    assert "28800 seconds" in instruction
    assert "talk_to_client(message)" not in instruction
    assert "`wait_for_result`" in instruction
    assert "hyper-tau-tool" in instruction
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
    assert dockerfile.startswith("FROM tau2-construction-runtime:contract-v7\n")
    assert token in (task_dir / "environment" / "hyper-tau-tool").read_text()
    assert "hyper-tau-tool submit" in (task_dir / "solution" / "solve.sh").read_text()


def test_client_opening_is_fixed_at_export_and_reused(tmp_path, client_openings):
    (task_dir,) = export(tmp_path, CLIENT_TASK, agent_timeout_sec=3600)
    assert client_openings == [CLIENT_TASK]
    config, compose, _ = load(task_dir)
    assert config["metadata"]["client_enabled"] is True
    assert config["metadata"]["client_opening"] == OPENING
    assert config["agent"]["timeout_sec"] == 3600.0

    instruction = (task_dir / "instruction.md").read_text()
    assert OPENING in instruction
    assert "talk_to_client(message)" in instruction
    assert "3600 seconds" in instruction

    runtime = compose["services"]["hyper-runtime"]
    raw = runtime["environment"][CONFIG_ENV_VAR]
    assert "$$25" in raw and "$${FEE}" in raw
    settings = ConstructionConfig.from_json(compose_value(raw))
    assert settings.client_opening == OPENING
    assert settings.client_llm == config["metadata"]["client_llm"]
    assert settings.max_time_seconds == 3600

    # A later export keeps the same brief instead of asking the Client again.
    export(tmp_path, CLIENT_TASK, agent_timeout_sec=3600)
    assert client_openings == [CLIENT_TASK]
    assert OPENING in (task_dir / "instruction.md").read_text()
    export(tmp_path, CLIENT_TASK, reuse_client_openings=False)
    assert client_openings == [CLIENT_TASK, CLIENT_TASK]


def test_soft_call_timeout_can_be_disabled(tmp_path, client_openings):
    (task_dir,) = export(tmp_path, NO_CLIENT_TASK, mcp_call_timeout_sec=0)
    config, _, _ = load(task_dir)
    assert "--call-timeout" not in config["environment"]["mcp_servers"][0]["args"]
    assert "wait_for_result" not in (task_dir / "instruction.md").read_text()


def test_unknown_tasks_and_network_modes_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        export(tmp_path, "no-such-task")
    with pytest.raises(ValueError, match="network"):
        export(tmp_path, NO_CLIENT_TASK, agent_network="none")


def test_memory_sizes_convert_to_megabytes():
    assert _memory_mb("4g", 1) == 4096
    assert _memory_mb("512m", 1) == 512
    assert _memory_mb(str(2 * 1024 * 1024 * 1024), 1) == 2048
    assert _memory_mb(None, 4096) == 4096
    assert json.dumps(_memory_mb("1.5g", 1)) == "1536"
