#!/usr/bin/env python3
"""Stdio MCP server that forwards every tool to a trusted sidecar.

Copied into a Harbor task's main image and launched by the agent's MCP
client. It answers the MCP handshake itself and relays ``tools/list`` and
``tools/call`` through request/response files in a directory shared with
the sidecar (a compose volume), using the same file protocol as the
construction harnesses' ``callback_mcp``. The sidecar holds all task
state; this script holds none and imports nothing outside the standard
library, so any Python 3.8+ in the main image can run it.

    mcp_stub.py --dir /var/run/tau2-bridge --token T            # MCP server
    mcp_stub.py --dir /var/run/tau2-bridge --token T call NAME '{...}'
    mcp_stub.py --dir /var/run/tau2-bridge --token T list

Some MCP clients cap how long one tool call may take (Codex defaults to
60 seconds and Harbor's generated Codex config cannot raise it). With
``--call-timeout N`` a call still running after N seconds returns a job id
instead, and the extra ``wait_for_result`` tool keeps waiting for it; the
call itself keeps running in the sidecar either way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

LIST_TOOLS = "__list_tools__"
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")


class BridgeError(RuntimeError):
    pass


PENDING = object()


class Bridge:
    def __init__(self, directory: Path, token: str, timeout: float):
        self.directory = directory
        self.token = token
        self.timeout = timeout

    def submit(self, tool: str, arguments: dict) -> str:
        request_id = uuid.uuid4().hex
        temporary = self.directory / f".request-{request_id}.tmp"
        temporary.write_text(
            json.dumps({"token": self.token, "tool": tool, "arguments": arguments})
        )
        temporary.replace(self.directory / f"request-{request_id}.json")
        return request_id

    def wait(self, request_id: str, seconds: float):
        """The call's result, or PENDING if it is still running after ``seconds``."""
        response = self.directory / f"response-{request_id}.json"
        deadline = time.monotonic() + seconds
        while True:
            if response.exists():
                payload = json.loads(response.read_text())
                try:
                    response.unlink()
                except OSError:
                    pass
                if not payload.get("ok"):
                    error = payload.get("error") or {}
                    raise BridgeError(str(error.get("message") or "tool call failed"))
                return payload.get("result")
            if time.monotonic() >= deadline:
                return PENDING
            time.sleep(0.05)

    def abandon(self, request_id: str) -> None:
        try:
            (self.directory / f"request-{request_id}.json").unlink()
        except OSError:
            pass

    def call(self, tool: str, arguments: dict):
        request_id = self.submit(tool, arguments)
        result = self.wait(request_id, self.timeout)
        if result is PENDING:
            self.abandon(request_id)
            raise BridgeError(
                f"No response from the task runtime after {self.timeout:g}s"
            )
        return result


def _result(message_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


WAIT_TOOL = "wait_for_result"


def _text(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class Server:
    def __init__(self, bridge: Bridge, name: str, call_timeout: float = 0.0):
        self.bridge = bridge
        self.name = name
        self.call_timeout = call_timeout
        self._tools = None
        self._jobs = {}  # job id -> hard deadline

    def _tool_list(self) -> list:
        tools = list(self.bridge.call(LIST_TOOLS, {}))
        if self.call_timeout > 0:
            tools.append(
                {
                    "name": WAIT_TOOL,
                    "description": (
                        "Keep waiting for a tool call that is still running. "
                        "Pass the job_id the tool returned; returns the call's "
                        "result, or the same job_id again if it is still running."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {"job_id": {"type": "string"}},
                        "required": ["job_id"],
                    },
                }
            )
        return tools

    def _await(self, job_id: str) -> dict:
        deadline = self._jobs[job_id]
        seconds = min(self.call_timeout, max(0.0, deadline - time.monotonic()))
        result = self.bridge.wait(job_id, seconds)
        if result is not PENDING:
            del self._jobs[job_id]
            return result
        if time.monotonic() >= deadline:
            del self._jobs[job_id]
            self.bridge.abandon(job_id)
            return _text("The tool call timed out in the task runtime.", True)
        return _text(
            f"The tool call is still running (job_id {job_id}). Call "
            f'{WAIT_TOOL} with {{"job_id": "{job_id}"}} to keep waiting for '
            "its result. Do not repeat the original call."
        )

    def _call(self, name: str, arguments: dict) -> dict:
        if self.call_timeout <= 0:
            return self.bridge.call(name, arguments)
        if name == WAIT_TOOL:
            job_id = str(arguments.get("job_id", ""))
            if job_id not in self._jobs:
                return _text(f"Unknown or finished job_id: {job_id!r}", True)
            return self._await(job_id)
        job_id = self.bridge.submit(name, arguments)
        self._jobs[job_id] = time.monotonic() + self.bridge.timeout
        return self._await(job_id)

    def handle(self, message: dict):
        method = message.get("method")
        message_id = message.get("id")
        if method == "initialize":
            requested = (message.get("params") or {}).get("protocolVersion")
            version = (
                requested
                if requested in SUPPORTED_PROTOCOL_VERSIONS
                else SUPPORTED_PROTOCOL_VERSIONS[0]
            )
            return _result(
                message_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": "1"},
                },
            )
        if message_id is None:
            return None  # notifications need no answer
        if method == "ping":
            return _result(message_id, {})
        if method == "tools/list":
            if self._tools is None:
                self._tools = self._tool_list()
            return _result(message_id, {"tools": self._tools})
        if method == "tools/call":
            params = message.get("params") or {}
            arguments = params.get("arguments") or {}
            try:
                result = self._call(str(params.get("name", "")), arguments)
            except Exception as error:  # noqa: BLE001 - rendered as a tool error
                result = _text(str(error), True)
            return _result(message_id, result)
        return _error(message_id, -32601, f"Method not found: {method}")

    def serve(self, stdin, stdout) -> None:
        for line in stdin:
            if not line.strip():
                continue
            try:
                response = self.handle(json.loads(line))
            except Exception as error:  # noqa: BLE001 - keep the server alive
                response = _error(None, -32603, str(error))
            if response is not None:
                stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
                stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--dir", required=True, help="directory shared with the sidecar"
    )
    parser.add_argument("--token", default=os.environ.get("TAU2_BRIDGE_TOKEN", ""))
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--server-name", default="tau2")
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=float(os.environ.get("TAU2_BRIDGE_CALL_TIMEOUT") or 0),
        help="return a job id for calls running longer than this (0: never)",
    )
    parser.add_argument("command", nargs="?", choices=["call", "list"])
    parser.add_argument("tool", nargs="?")
    parser.add_argument("arguments", nargs="?", default="{}")
    args = parser.parse_args(argv)

    bridge = Bridge(Path(args.dir), args.token, args.timeout)
    if args.command == "list":
        for tool in bridge.call(LIST_TOOLS, {}):
            print(f"{tool['name']}: {tool.get('description', '')}")
            print(f"  arguments: {json.dumps(tool.get('inputSchema') or {})}")
        return 0
    if args.command == "call":
        if not args.tool:
            parser.error("call needs a tool name")
        result = bridge.call(args.tool, json.loads(args.arguments))
        if isinstance(result, dict) and "content" in result:
            for block in result.get("content") or []:
                if block.get("type") == "text":
                    print(block.get("text", ""))
            return 1 if result.get("isError") else 0
        print(json.dumps(result, indent=2))
        return 0

    Server(bridge, args.server_name, args.call_timeout).serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
