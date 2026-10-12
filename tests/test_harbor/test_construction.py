"""Tests for the Harbor construction runtime and verifier."""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from loguru import logger

from tau2.data_model.message import AssistantMessage
from tau2.harbor import mcp_stub
from tau2.harbor import runtime as harbor_runtime
from tau2.harbor import verify as harbor_verify
from tau2.harbor.common import warm_knowledge_base
from tau2.harbor.runtime import (
    FINALIZED_FILE,
    STATE_FILE,
    SUBMISSION_FILE,
    ConstructionConfig,
    ConstructionRuntime,
    request_finalize,
)
from tau2.hyper.data_model import OuterLoopResult
from tau2.hyper.sandbox.orchestrator import SandboxOrchestrator

NO_CLIENT_TASK = "008_retail_plus_construction_core_evidence_performance_hard"
CLIENT_TASK = (
    "004_airline_plus_construction_core_evidence_hard_client_performance_medium"
)
OPENING = "Hi, I run support operations. We need an agent for rebookings."


@pytest.fixture(autouse=True)
def tiny_kit(monkeypatch):
    """Build a two-file kit instead of the full evidence corpus."""

    def build_kit(task, kit_path, **kwargs):
        kit_path = Path(kit_path)
        (kit_path / "workspace").mkdir(parents=True, exist_ok=True)
        (kit_path / "README.md").write_text(f"# Kit for {task.id}\n")
        (kit_path / "workspace" / "agent.py").write_text("def create_agent(): ...\n")

    monkeypatch.setattr("tau2.hyper.sandbox.orchestrator.build_kit", build_kit)


def make_runtime(tmp_path: Path, task_id: str = NO_CLIENT_TASK, **config):
    runtime = ConstructionRuntime(
        ConstructionConfig(task_id=task_id, bridge_token="bridge-token", **config),
        workspace=tmp_path / "workspace",
        state_dir=tmp_path / "state",
        bridge_dir=tmp_path / "bridge",
    )
    runtime.start()
    return runtime


def read_state(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "state" / STATE_FILE).read_text())


def stub_call(tmp_path: Path, tool: str, arguments: dict | None = None) -> dict:
    bridge = mcp_stub.Bridge(tmp_path / "bridge", "bridge-token", timeout=30)
    return bridge.call(tool, arguments or {})


def test_runtime_serves_the_developer_tools_and_records_the_submission(tmp_path):
    runtime = make_runtime(tmp_path)
    try:
        assert (tmp_path / "state" / "ready").exists()
        state = read_state(tmp_path)
        assert state["finalized"] is False
        assert state["kit_baseline_commit"]
        assert "bridge_token" not in state["config"]

        tools = stub_call(tmp_path, mcp_stub.LIST_TOOLS)
        assert [tool["name"] for tool in tools] == ["run_local_test", "submit"]

        missing = stub_call(tmp_path, "run_local_test", {})
        assert missing["isError"] is True
        assert "task_path" in missing["content"][0]["text"]

        # Edits made before submit are what gets scored.
        agent = tmp_path / "workspace" / "workspace" / "agent.py"
        agent.write_text("def create_agent():\n    return 'built'\n")
        submitted = stub_call(tmp_path, "submit")
        assert submitted["isError"] is False
        agent.write_text("# edited after submit\n")

        after = stub_call(tmp_path, "run_local_test", {"task_path": "x.json"})
        assert after["isError"] is True
        assert "Construction has ended" in after["content"][0]["text"]
    finally:
        runtime.stop()

    state = read_state(tmp_path)
    assert state["finalized"] is True
    assert state["build_result"]["submitted"] is True
    assert state["build_result"]["done_reason"] == "submitted"
    assert [step["tool_calls"][0]["name"] for step in state["build_result"]["steps"]]
    assert state["build_result"]["total_tool_calls"] == 2
    assert state["deployment_snapshot"]["enabled_capability_ids"] == []
    assert (tmp_path / "state" / FINALIZED_FILE).read_text().strip() == "submitted"

    unpacked = tmp_path / "unpacked"
    harbor_verify.extract_submission(tmp_path / "state" / SUBMISSION_FILE, unpacked)
    assert "return 'built'" in (unpacked / "workspace" / "agent.py").read_text()
    assert (unpacked / ".git").is_dir()
    # Finalizing again (the collect hook after a submit) changes nothing.
    assert request_finalize(tmp_path / "state") == 0


def test_runtime_finalizes_when_the_agent_never_submits(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.finalize("agent_ended")
    runtime.stop()
    state = read_state(tmp_path)
    assert state["finalized"] is True
    assert state["build_result"]["submitted"] is False
    assert state["build_result"]["done_reason"] == "agent_ended"
    assert (tmp_path / "state" / SUBMISSION_FILE).is_file()


def test_runtime_resumes_the_exported_client_opening(tmp_path, monkeypatch):
    prompts = []

    def client_model(*, model, messages, **kwargs):
        prompts.append(messages)
        return AssistantMessage(role="assistant", content="Rebookings are free.")

    monkeypatch.setattr("tau2.hyper.client.generate", client_model)
    runtime = make_runtime(tmp_path, CLIENT_TASK, client_opening=OPENING)
    try:
        names = [tool["name"] for tool in stub_call(tmp_path, mcp_stub.LIST_TOOLS)]
        assert "talk_to_client" in names
        # No model call happened at start: the opening came from the export.
        assert prompts == []
        reply = stub_call(tmp_path, "talk_to_client", {"message": "Any fees?"})
        assert reply["content"][0]["text"] == "Rebookings are free."
        assert OPENING in json.dumps([m.model_dump() for m in prompts[0]])
    finally:
        runtime.stop()
    state = read_state(tmp_path)
    assert state["client_opening"] == OPENING
    assert state["client"]["turns_used"] == 1
    assert state["build_result"]["client_turns_used"] == 1


def test_runtime_main_needs_a_bridge_token(tmp_path, monkeypatch):
    monkeypatch.delenv(harbor_runtime.TOKEN_ENV_VAR, raising=False)
    monkeypatch.setenv(
        harbor_runtime.CONFIG_ENV_VAR,
        ConstructionConfig(task_id=NO_CLIENT_TASK).to_json(),
    )
    with pytest.raises(ValueError, match="bridge token"):
        harbor_runtime.main(["serve", "--state-dir", str(tmp_path)])


def _add(bundle: tarfile.TarFile, info: tarfile.TarInfo, data: bytes = b"") -> None:
    info.size = len(data)
    bundle.addfile(info, io.BytesIO(data) if data else None)


def test_extract_submission_skips_entries_that_escape_the_workspace(tmp_path):
    archive = tmp_path / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        _add(bundle, tarfile.TarInfo("./workspace/agent.py"), b"print('ok')\n")
        link = tarfile.TarInfo("./workspace/passwd")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        _add(bundle, link)
        inside = tarfile.TarInfo("./workspace/alias.py")
        inside.type = tarfile.SYMTYPE
        inside.linkname = "agent.py"
        _add(bundle, inside)
        _add(bundle, tarfile.TarInfo("../escaped.txt"), b"x")
        _add(bundle, tarfile.TarInfo("/abs.txt"), b"x")
        device = tarfile.TarInfo("./workspace/null")
        device.type = tarfile.CHRTYPE
        _add(bundle, device)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "stale.txt").write_text("left over")
    harbor_verify.extract_submission(archive, workspace)

    assert (workspace / "workspace" / "agent.py").read_text() == "print('ok')\n"
    assert os.readlink(workspace / "workspace" / "alias.py") == "agent.py"
    assert not (workspace / "workspace" / "passwd").exists()
    assert not (workspace / "workspace" / "passwd").is_symlink()
    assert not (workspace / "workspace" / "null").exists()
    assert not (tmp_path / "escaped.txt").exists()
    assert not (workspace / "stale.txt").exists()


def test_verifier_refuses_a_session_that_was_never_finalized(tmp_path):
    with pytest.raises(harbor_verify.SubmissionError, match="No construction record"):
        harbor_verify.load_state(tmp_path)
    runtime = make_runtime(tmp_path)
    runtime.stop()
    with pytest.raises(harbor_verify.SubmissionError, match="never finalized"):
        harbor_verify.load_state(tmp_path / "state")


def test_verifier_scores_the_snapshot_with_the_recorded_session(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    stub_call(tmp_path, "submit")
    runtime.stop()
    (tmp_path / "workspace" / "README.md").write_text("changed after the snapshot")

    seen = {}

    def score_submission(
        self, kit_path, build_result, *, kit_baseline_commit, run_metadata_overrides
    ):
        seen.update(
            kit_path=kit_path,
            build_result=build_result,
            kit_baseline_commit=kit_baseline_commit,
            overrides=run_metadata_overrides,
            snapshot=self._deployment_snapshot,
            readme=(Path(kit_path) / "README.md").read_text(),
        )
        return OuterLoopResult(
            domain=self.task.source_domain,
            final_test_reward=0.25,
            final_quality_reward=0.5,
            performance_reward=1.0,
            performance_penalty=0.25,
            total_outer_steps=build_result.total_steps,
            client_turns_used=build_result.client_turns_used,
        )

    monkeypatch.setattr(SandboxOrchestrator, "score_submission", score_submission)
    scored_workspace = tmp_path / "scored"
    log_dir = tmp_path / "verifier"
    harbor_verify.main(
        [
            "--state-dir",
            str(tmp_path / "state"),
            "--workspace",
            str(scored_workspace),
            "--log-dir",
            str(log_dir),
        ]
    )

    rewards = json.loads((log_dir / "reward.json").read_text())
    assert rewards == {
        "reward": 0.25,
        "quality_reward": 0.5,
        "performance_reward": 1.0,
        "performance_penalty": 0.25,
    }
    result = json.loads((log_dir / harbor_verify.RESULT_FILE).read_text())
    assert result["final_test_reward"] == 0.25

    state = read_state(tmp_path)
    assert seen["kit_path"] == scored_workspace
    assert seen["readme"].startswith("# Kit for")
    assert seen["kit_baseline_commit"] == state["kit_baseline_commit"]
    assert seen["snapshot"].sha256 == state["deployment_snapshot"]["sha256"]
    assert seen["build_result"].submitted is True
    assert seen["build_result"].total_tool_calls == 1
    assert seen["overrides"]["harbor"]["done_reason"] == "submitted"
    assert "client" not in seen["overrides"]


def test_verifier_points_out_conversations_that_errored():
    result = OuterLoopResult(
        domain="retail_plus",
        final_test_reward=0.0,
        total_outer_steps=1,
        client_turns_used=0,
        test_details=[
            {"task_id": "a", "reward": 0.0, "error": "task_error: boom"},
            {"task_id": "b", "reward": 0.0, "error": "infrastructure_error: 503"},
            {"task_id": "c", "reward": 1.0, "error": None},
        ],
    )
    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        harbor_verify.log_scoring_errors(result)
    finally:
        logger.remove(sink)
    assert len(messages) == 1
    assert "2 of 3 test conversations ended in an error" in messages[0]
    assert "'task_error': 1" in messages[0]
    assert "'infrastructure_error': 1" in messages[0]


def test_knowledge_base_is_embedded_once_for_banking_tasks(monkeypatch):
    calls = []
    monkeypatch.setattr("tau2.knowledge.embeddings_cache.warm_kb_cache", calls.append)
    warm_knowledge_base("retail_plus")
    assert calls == []
    warm_knowledge_base("banking_knowledge")
    assert calls == [[("openai", {"model": "text-embedding-3-large"})]]

    def unavailable(configs):
        raise RuntimeError("no embeddings key")

    monkeypatch.setattr("tau2.knowledge.embeddings_cache.warm_kb_cache", unavailable)
    warm_knowledge_base("banking_knowledge")  # left for the environments to report
