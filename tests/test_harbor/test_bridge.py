"""Tests for the stdio MCP stub and the sidecar end of its file bridge."""

from __future__ import annotations

import io
import json
import threading
import time

import pytest

from tau2.harbor import mcp_stub
from tau2.harbor.bridge import ToolBridge, ToolResult


@pytest.fixture
def serve(tmp_path):
    bridges = []

    def start(call_tool, list_tools=lambda: [], token="token"):
        bridge = ToolBridge(
            tmp_path / "bridge", token, list_tools=list_tools, call_tool=call_tool
        )
        bridge.start()
        bridges.append(bridge)
        return bridge.directory

    yield start
    for bridge in bridges:
        bridge.stop()


def rpc(server, *requests):
    stdin = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    stdout = io.StringIO()
    server.serve(stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def call(message_id, name, arguments=None):
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }


def test_bridge_rejects_a_wrong_token(serve):
    bridge_dir = serve(lambda name, arguments: ToolResult("ok"), token="right")
    with pytest.raises(mcp_stub.BridgeError, match="Invalid bridge token"):
        mcp_stub.Bridge(bridge_dir, "wrong", timeout=10).call("anything", {})
    result = mcp_stub.Bridge(bridge_dir, "right", timeout=10).call("anything", {})
    assert result == {"content": [{"type": "text", "text": "ok"}], "isError": False}


def test_bridge_does_not_follow_planted_links(tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"token": "right", "tool": "x", "arguments": {}}))
    bridge_dir = tmp_path / "bridge"
    calls = []
    bridge = ToolBridge(
        bridge_dir,
        "right",
        list_tools=lambda: [],
        call_tool=lambda name, arguments: calls.append(name) or ToolResult("ok"),
    )
    bridge.start()
    try:
        bridge_dir.joinpath("request-abc123.json").symlink_to(secret)
        response = bridge_dir / "response-abc123.json"
        deadline = threading.Event()
        for _ in range(200):
            if response.exists():
                break
            deadline.wait(0.05)
        payload = json.loads(response.read_text())
    finally:
        bridge.stop()
    assert payload["ok"] is False
    assert calls == []
    assert json.loads(secret.read_text())["token"] == "right"


def test_stub_cli_call_prints_text_and_exit_status(serve, capsys):
    bridge_dir = serve(
        lambda name, arguments: ToolResult(
            f"{name}:{arguments.get('value')}", is_error=name == "bad"
        )
    )
    common = ["--dir", str(bridge_dir), "--token", "token", "--timeout", "10"]
    assert mcp_stub.main(common + ["call", "good", '{"value": 3}']) == 0
    assert capsys.readouterr().out == "good:3\n"
    assert mcp_stub.main(common + ["call", "bad"]) == 1


def test_slow_calls_return_a_job_to_wait_on_when_capped(serve):
    release = threading.Event()

    def slow(name, arguments):
        release.wait(10)
        return ToolResult(f"done:{arguments['n']}")

    bridge_dir = serve(slow, list_tools=lambda: [{"name": "slow"}])
    server = mcp_stub.Server(
        mcp_stub.Bridge(bridge_dir, "token", timeout=30), name="tau2", call_timeout=0.3
    )
    listed, first = rpc(
        server,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        call(2, "slow", {"n": 1}),
    )
    assert [tool["name"] for tool in listed["result"]["tools"]] == [
        "slow",
        mcp_stub.WAIT_TOOL,
    ]
    text = first["result"]["content"][0]["text"]
    assert "still running" in text and not first["result"]["isError"]
    job_id = text.split("job_id ")[1].split(")")[0]

    (pending,) = rpc(server, call(3, mcp_stub.WAIT_TOOL, {"job_id": job_id}))
    assert "still running" in pending["result"]["content"][0]["text"]
    release.set()
    deadline = time.monotonic() + 10
    while True:
        (waited,) = rpc(server, call(4, mcp_stub.WAIT_TOOL, {"job_id": job_id}))
        result_text = waited["result"]["content"][0]["text"]
        if "still running" not in result_text or time.monotonic() > deadline:
            break
    assert result_text == "done:1"
    (unknown,) = rpc(server, call(5, mcp_stub.WAIT_TOOL, {"job_id": job_id}))
    assert unknown["result"]["isError"] is True


def test_fast_calls_are_unchanged_when_capped(serve):
    bridge_dir = serve(lambda name, arguments: ToolResult("quick"))
    server = mcp_stub.Server(
        mcp_stub.Bridge(bridge_dir, "token", timeout=30), name="tau2", call_timeout=5
    )
    (response,) = rpc(server, call(1, "fast"))
    assert response["result"] == {
        "content": [{"type": "text", "text": "quick"}],
        "isError": False,
    }


def test_initialize_falls_back_to_a_supported_protocol_version():
    server = mcp_stub.Server(mcp_stub.Bridge(None, "t", timeout=1), name="tau2")
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        }
    )
    assert (
        response["result"]["protocolVersion"] == mcp_stub.SUPPORTED_PROTOCOL_VERSIONS[0]
    )
    assert (
        server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    )
    assert server.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"] == {}
