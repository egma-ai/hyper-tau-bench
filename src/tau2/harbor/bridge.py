"""Trusted end of the file bridge between a Harbor main container and a sidecar.

The agent's MCP client runs :mod:`tau2.harbor.mcp_stub` inside ``main``.
The stub writes each request into a directory shared with the
``hyper-runtime`` sidecar; this module answers them there, through the same
file protocol as the construction harnesses' ``CallbackBroker``.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from tau2.harbor.mcp_stub import LIST_TOOLS
from tau2.hyper.sandbox.callback_broker import (
    CallbackAuthenticationError,
    CallbackBrokerError,
    serve_file_requests,
)


@dataclass
class ToolResult:
    """Text returned to the agent for one tool call."""

    text: str
    is_error: bool = False

    def to_mcp(self) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": self.text}],
            "isError": self.is_error,
        }


class ToolBridge:
    """Serve a tool table to the main container's MCP stub.

    Calls run one at a time, in arrival order: a construction session is a
    single sequential thread of actions.
    """

    def __init__(
        self,
        directory: Path,
        token: str,
        *,
        list_tools: Callable[[], list[dict[str, Any]]],
        call_tool: Callable[[str, dict[str, Any]], ToolResult],
        directory_mode: int = 0o777,
    ):
        if not token:
            raise ValueError("The bridge needs a non-empty token")
        self.directory = Path(directory)
        self.token = token
        self._list_tools = list_tools
        self._call_tool = call_tool
        self._directory_mode = directory_mode
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def handle(self, request: dict[str, Any]) -> Any:
        if not secrets.compare_digest(str(request.get("token", "")), self.token):
            raise CallbackAuthenticationError("Invalid bridge token")
        tool = str(request.get("tool", ""))
        arguments = request.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise CallbackBrokerError("Tool arguments must be an object")
        if tool == LIST_TOOLS:
            return self._list_tools()
        return self._call_tool(tool, arguments).to_mcp()

    def start(self) -> None:
        if self._thread is not None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        # The agent may run as any user in main; it must be able to create
        # requests and remove the responses it has read.
        self.directory.chmod(self._directory_mode)
        self._thread = threading.Thread(
            target=serve_file_requests,
            args=(self.directory, self.handle, self._stop),
            name="tau2-harbor-bridge",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
