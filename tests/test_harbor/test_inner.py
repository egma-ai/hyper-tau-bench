"""Tests for the inner-loop Harbor runtime, bridge, and verifier."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from tau2.data_model.message import AssistantMessage
from tau2.harbor import mcp_stub
from tau2.harbor.bridge import ToolBridge
from tau2.harbor.inner import runtime as inner_runtime
from tau2.harbor.inner import verify as inner_verify
from tau2.harbor.inner.runtime import (
    END_TOOL,
    ORACLE_TOOL,
    SEND_TOOL,
    START_TOOL,
    InnerConversation,
    build_run_config,
    load_task,
)

ORACLE_TOKEN = "oracle-secret"


class ScriptedCustomer:
    """Stands in for the user simulator's model, one reply per call."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[list] = []

    def __call__(self, *, model, messages, tools=None, call_name=None, **kwargs):
        self.prompts.append(messages)
        if not self.replies:
            raise AssertionError("The customer was asked for more turns than scripted")
        return AssistantMessage(role="assistant", content=self.replies.pop(0))


@pytest.fixture
def customer(monkeypatch):
    def install(*replies: str) -> ScriptedCustomer:
        scripted = ScriptedCustomer(list(replies))
        monkeypatch.setattr("tau2.user.user_simulator.generate", scripted)
        return scripted

    return install


def make_conversation(tmp_path: Path, task_id: str = "create_task_1"):
    config = build_run_config("mock", user_llm="scripted-customer")
    return InnerConversation(
        load_task("mock", task_id),
        config,
        state_path=tmp_path / "conversation.json",
        oracle_token=ORACLE_TOKEN,
    )


def read_state(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "conversation.json").read_text())


def score(tmp_path: Path, task_id: str = "create_task_1") -> dict:
    log_dir = tmp_path / "verifier"
    inner_verify.main(
        [
            "--domain",
            "mock",
            "--task-id",
            task_id,
            "--state",
            str(tmp_path / "conversation.json"),
            "--log-dir",
            str(log_dir),
        ]
    )
    return json.loads((log_dir / "reward.json").read_text())


def test_tool_list_hides_the_oracle_and_keeps_domain_schemas(tmp_path, customer):
    customer()
    conversation = make_conversation(tmp_path)
    tools = {tool["name"]: tool for tool in conversation.list_tools()}
    assert {START_TOOL, SEND_TOOL, END_TOOL} <= set(tools)
    assert ORACLE_TOOL not in tools
    create = tools["create_task"]
    assert create["inputSchema"]["required"] == ["user_id", "title"]
    assert create["description"]


def test_conversation_runs_turn_by_turn_and_scores_like_tau2(tmp_path, customer):
    scripted = customer(
        "Hi, please create a task called 'Important Meeting' for user_1.",
        "Great, thanks! ###STOP###",
    )
    conversation = make_conversation(tmp_path)

    early = conversation.call(SEND_TOOL, {"message": "hello"})
    assert early.is_error and START_TOOL in early.text
    assert read_state(tmp_path)["started"] is False

    opening = conversation.call(START_TOOL, {})
    assert not opening.is_error
    assert "Important Meeting" in opening.text
    # The simulator saw tau2's standard greeting as the agent's first turn.
    assert "How can I help you today?" in json.dumps(
        [message.model_dump() for message in scripted.prompts[0]], default=str
    )
    # Calling start again re-reads the latest message instead of restarting.
    assert conversation.call(START_TOOL, {}).text == opening.text

    created = conversation.call(
        "create_task", {"user_id": "user_1", "title": "Important Meeting"}
    )
    assert not created.is_error
    assert "Important Meeting" in created.text

    bad = conversation.call("update_task_status", {"task_id": "nope", "status": "x"})
    assert bad.is_error

    empty = conversation.call(SEND_TOOL, {"message": "  "})
    assert empty.is_error

    reply = conversation.call(SEND_TOOL, {"message": "Done: the task is created."})
    assert "Great, thanks!" in reply.text
    assert "customer ended the conversation" in reply.text

    after = conversation.call(SEND_TOOL, {"message": "anything else?"})
    assert after.is_error and "No further actions" in after.text

    state = read_state(tmp_path)
    assert state["finished"] is True
    assert state["termination_reason"] == "user_stop"
    roles = [message["role"] for message in state["simulation"]["messages"]]
    assert roles == ["assistant", "user", "assistant", "tool", "assistant", "tool"] + [
        "assistant",
        "user",
    ]

    rewards = score(tmp_path)
    assert rewards["reward"] == 1.0
    details = json.loads((tmp_path / "verifier" / "details.json").read_text())
    assert details["termination_reason"] == "user_stop"


def test_agent_can_end_the_conversation(tmp_path, customer):
    customer("Please create a task called 'Important Meeting' for user_1.")
    conversation = make_conversation(tmp_path)
    conversation.call(START_TOOL, {})
    conversation.call(
        "create_task", {"user_id": "user_1", "title": "Important Meeting"}
    )
    ended = conversation.call(END_TOOL, {})
    assert "You ended the conversation" in ended.text
    assert read_state(tmp_path)["termination_reason"] == "agent_stop"
    assert score(tmp_path)["reward"] == 1.0


def test_wrong_actions_score_zero(tmp_path, customer):
    customer("Please create a task called 'Important Meeting' for user_1.")
    conversation = make_conversation(tmp_path)
    conversation.call(START_TOOL, {})
    conversation.call("create_task", {"user_id": "user_1", "title": "Wrong title"})
    conversation.call(END_TOOL, {})
    assert score(tmp_path)["reward"] == 0.0


def test_unfinished_conversation_scores_zero(tmp_path, customer):
    customer("Please create a task called 'Important Meeting' for user_1.")
    conversation = make_conversation(tmp_path)
    conversation.call(START_TOOL, {})
    conversation.call(
        "create_task", {"user_id": "user_1", "title": "Important Meeting"}
    )
    state = read_state(tmp_path)
    assert state["finished"] is False
    assert score(tmp_path)["reward"] == 0.0
    details = json.loads((tmp_path / "verifier" / "details.json").read_text())
    assert "not finished" in details["reward_info"]["info"]["note"]


def test_user_simulator_failure_is_an_infrastructure_error(tmp_path, customer):
    customer()  # no scripted replies: the first customer turn raises
    conversation = make_conversation(tmp_path)
    result = conversation.call(START_TOOL, {})
    assert result.is_error and "runtime failed" in result.text
    state = read_state(tmp_path)
    assert state["infrastructure_error"]
    assert state["finished"] is False
    with pytest.raises(inner_verify.ConversationStateError, match="runtime failed"):
        score(tmp_path)


def test_oracle_replay_needs_the_token_and_scores_full_marks(tmp_path, customer):
    customer()
    conversation = make_conversation(tmp_path)
    denied = conversation.call(ORACLE_TOOL, {"oracle_token": "guess"})
    assert denied.is_error and "Unknown tool" in denied.text
    assert not conversation.call(ORACLE_TOOL, {"oracle_token": ORACLE_TOKEN}).is_error
    state = read_state(tmp_path)
    assert state["oracle"] is True and state["finished"] is True
    assert score(tmp_path)["reward"] == 1.0
    again = conversation.call(START_TOOL, {})
    assert "You ended the conversation" in again.text


def test_task_history_is_shown_when_the_conversation_starts(tmp_path, customer):
    customer("Yes please, go ahead.")
    conversation = make_conversation(tmp_path, "update_task_with_message_history")
    opening = conversation.call(START_TOOL, {})
    assert opening.text.startswith("The conversation so far:")
    assert "Customer:" in opening.text


def test_stub_and_bridge_serve_the_conversation_over_files(tmp_path, customer):
    customer(
        "Please create a task called 'Important Meeting' for user_1.",
        "Thanks ###STOP###",
    )
    conversation = make_conversation(tmp_path)
    bridge_dir = tmp_path / "bridge"
    bridge = ToolBridge(
        bridge_dir,
        "bridge-token",
        list_tools=conversation.list_tools,
        call_tool=conversation.call,
    )
    bridge.start()
    try:
        server = mcp_stub.Server(
            mcp_stub.Bridge(bridge_dir, "bridge-token", timeout=30), name="tau2"
        )
        requests = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": START_TOOL, "arguments": {}},
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "create_task",
                    "arguments": {"user_id": "user_1", "title": "Important Meeting"},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": SEND_TOOL, "arguments": {"message": "Created!"}},
            },
            {"jsonrpc": "2.0", "id": 6, "method": "resources/list"},
        ]
        stdin = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
        stdout = io.StringIO()
        server.serve(stdin, stdout)
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    finally:
        bridge.stop()

    assert [response["id"] for response in responses] == [1, 2, 3, 4, 5, 6]
    assert responses[0]["result"]["protocolVersion"] == "2025-06-18"
    names = [tool["name"] for tool in responses[1]["result"]["tools"]]
    assert START_TOOL in names and ORACLE_TOOL not in names
    assert "Important Meeting" in responses[2]["result"]["content"][0]["text"]
    assert responses[3]["result"]["isError"] is False
    assert "Thanks" in responses[4]["result"]["content"][0]["text"]
    assert responses[5]["error"]["code"] == -32601
    assert read_state(tmp_path)["termination_reason"] == "user_stop"
    assert not list(bridge_dir.glob("*.json"))


def test_runtime_main_rejects_unknown_tasks(tmp_path):
    with pytest.raises(ValueError):
        inner_runtime.main(
            [
                "--domain",
                "mock",
                "--task-id",
                "does-not-exist",
                "--bridge-dir",
                str(tmp_path / "bridge"),
                "--state-dir",
                str(tmp_path / "state"),
            ]
        )
