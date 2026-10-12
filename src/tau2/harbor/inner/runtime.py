"""Trusted τ³ conversation runtime for one Harbor task (the ``tau-runtime`` sidecar).

The Harbor agent plays the customer-service agent. This process owns
everything else, exactly as in a tau2 text run: the domain environment and
its database, the LLM user simulator, and the standard half-duplex
:class:`~tau2.orchestrator.orchestrator.Orchestrator`. Each agent tool call
becomes one agent turn of that orchestrator:

- ``start_conversation`` starts the run and returns the user's first message;
- ``send_message_to_user`` is a text turn and returns the user's reply (after
  any user-side tool calls);
- each domain tool is a single tool-call turn and returns the tool result;
- ``end_conversation`` is the ``###STOP###`` turn.

After every turn the recorded conversation is written to the state
directory, which the task collects as an artifact for the verifier
(:mod:`tau2.harbor.inner.verify`). The agent never sees the task's user
scenario or evaluation criteria.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import (
    SimulationRun,
    TerminationReason,
    TextRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.harbor.bridge import ToolBridge, ToolResult
from tau2.harbor.common import write_json_atomic
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.runner.build import _build_env_kwargs, build_environment, build_user
from tau2.runner.helpers import get_tasks
from tau2.utils.llm_utils import get_cost
from tau2.utils.utils import get_now

STOP_TOKEN = "###STOP###"
START_TOOL = "start_conversation"
SEND_TOOL = "send_message_to_user"
END_TOOL = "end_conversation"
# Not listed to the agent: replays the task's reference actions for
# Harbor's oracle agent, gated by a token only solution/solve.sh carries.
ORACLE_TOOL = "oracle_replay"

STATE_FILE = "conversation.json"
READY_FILE = "ready"


class ToolUsageError(Exception):
    """A tool call the agent can correct (shown to it as a tool error)."""


class BridgeAgent(HalfDuplexAgent):
    """Agent whose turns are supplied by the external Harbor agent."""

    def __init__(self, tools, domain_policy: str):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.pending: Optional[AssistantMessage] = None

    def get_init_state(self, message_history: Optional[list[Message]] = None):
        return None

    def generate_next_message(self, message, state):
        if self.pending is None:
            raise RuntimeError("The orchestrator asked for an agent turn nobody sent")
        turn, self.pending = self.pending, None
        return turn, state

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        return bool(message.content) and STOP_TOKEN in message.content

    def set_seed(self, seed: int):
        return None


def build_run_config(
    domain: str,
    *,
    user_llm: Optional[str] = None,
    user_llm_args: Optional[dict] = None,
    retrieval_config: Optional[str] = None,
) -> TextRunConfig:
    """The tau2 text-run settings both the runtime and the verifier use."""
    kwargs: dict[str, Any] = {"domain": domain}
    if user_llm:
        kwargs["llm_user"] = user_llm
    if user_llm_args is not None:
        kwargs["llm_args_user"] = user_llm_args
    if retrieval_config:
        kwargs["retrieval_config"] = retrieval_config
    return TextRunConfig(**kwargs)


def load_task(domain: str, task_id: str) -> Task:
    tasks = get_tasks(domain, task_ids=[task_id])
    if len(tasks) != 1:
        raise ValueError(f"Task {task_id!r} not found in domain {domain!r}")
    return tasks[0]


class InnerConversation:
    """One τ³ conversation driven turn by turn by an external agent."""

    def __init__(
        self,
        task: Task,
        config: TextRunConfig,
        *,
        state_path: Path,
        oracle_token: Optional[str] = None,
    ):
        self.task = task
        self.config = config
        self.state_path = Path(state_path)
        self.oracle_token = oracle_token
        env_kwargs = _build_env_kwargs(config, task)
        self.environment = build_environment(config.domain, env_kwargs=env_kwargs)
        self.domain_tools = {tool.name: tool for tool in self.environment.get_tools()}
        clashes = {START_TOOL, SEND_TOOL, END_TOOL, ORACLE_TOOL} & set(
            self.domain_tools
        )
        if clashes:
            raise ValueError(f"Domain tools clash with conversation tools: {clashes}")
        self.agent = BridgeAgent(
            tools=list(self.domain_tools.values()),
            domain_policy=self.environment.get_policy(),
        )
        self.user = build_user(
            config.effective_user,
            self.environment,
            task,
            llm=config.llm_user,
            llm_args=config.llm_args_user,
        )
        self.orchestrator = Orchestrator(
            domain=config.domain,
            agent=self.agent,
            user=self.user,
            environment=self.environment,
            task=task,
            max_steps=config.effective_max_steps,
            max_errors=config.max_errors,
            seed=config.seed,
            simulation_id=str(uuid.uuid4()),
            validate_communication=config.enforce_communication_protocol,
            timeout=config.timeout,
        )
        self.started = False
        self.finished = False
        self.oracle = False
        self.result: Optional[SimulationRun] = None
        self.infrastructure_error: Optional[str] = None
        self._lock = threading.Lock()
        self.persist()

    # Tool surface -------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        empty = {"type": "object", "properties": {}}
        tools = [
            {
                "name": START_TOOL,
                "description": (
                    "Start the conversation with the customer (or resume it) and "
                    "return the customer's latest message. Call it once before "
                    "anything else."
                ),
                "inputSchema": empty,
            },
            {
                "name": SEND_TOOL,
                "description": (
                    "Send a message to the customer and return the customer's "
                    "reply. Messages to the customer go through this tool only."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The text to send to the customer.",
                        }
                    },
                    "required": ["message"],
                },
            },
            {
                "name": END_TOOL,
                "description": (
                    "End the conversation once the customer's request is fully "
                    "handled. No further actions are possible afterwards."
                ),
                "inputSchema": empty,
            },
        ]
        for tool in self.domain_tools.values():
            function = tool.openai_schema["function"]
            tools.append(
                {
                    "name": function["name"],
                    "description": function.get("description") or "",
                    "inputSchema": function.get("parameters") or empty,
                }
            )
        return tools

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        with self._lock:
            try:
                return self._call(name, arguments)
            except ToolUsageError as error:
                return ToolResult(str(error), is_error=True)
            except Exception as error:  # noqa: BLE001 - recorded, run is over
                logger.exception("Conversation runtime failed")
                self.infrastructure_error = f"{type(error).__name__}: {error}"
                self.finished = True
                self.persist()
                return ToolResult(
                    "The conversation runtime failed and the conversation is over: "
                    f"{self.infrastructure_error}",
                    is_error=True,
                )

    def _call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == ORACLE_TOOL:
            return self._oracle_replay(arguments)
        if name == START_TOOL:
            if not self.started and not self.finished:
                self._start()
            return self._observation(with_history=True)
        if self.finished:
            raise ToolUsageError(self._ended_text())
        if not self.started:
            raise ToolUsageError(f"Call {START_TOOL} first.")
        if name == SEND_TOOL:
            text = arguments.get("message")
            if not isinstance(text, str) or not text.strip():
                raise ToolUsageError("message must be a non-empty string")
            self._act(AssistantMessage(role="assistant", content=text))
        elif name == END_TOOL:
            self._act(AssistantMessage(role="assistant", content=STOP_TOKEN))
        elif name in self.domain_tools:
            self._act(
                AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:16]}",
                            name=name,
                            arguments=arguments,
                            requestor="assistant",
                        )
                    ],
                )
            )
        else:
            raise ToolUsageError(f"Unknown tool: {name}")
        return self._observation()

    # Orchestration ------------------------------------------------------------

    def _start(self) -> None:
        orchestrator = self.orchestrator
        orchestrator._run_start_time = get_now()
        orchestrator._run_start_perf = time.perf_counter()
        orchestrator.initialize()
        self.started = True
        self._advance()

    def _act(self, turn: AssistantMessage) -> None:
        orchestrator = self.orchestrator
        if orchestrator.done or orchestrator.to_role != Role.AGENT:
            raise RuntimeError("It is not the agent's turn")
        self.agent.pending = turn
        orchestrator.step()
        orchestrator._check_termination()
        self._advance()

    def _advance(self) -> None:
        """Run user and environment turns until the agent is due again."""
        orchestrator = self.orchestrator
        while not orchestrator.done and orchestrator.to_role != Role.AGENT:
            orchestrator.step()
            orchestrator._check_termination()
        if orchestrator.done and self.result is None:
            self.result = orchestrator._finalize()
            self.result.policy = self.environment.get_policy()
            self.finished = True
        self.persist()

    def _observation(self, *, with_history: bool = False) -> ToolResult:
        if self.infrastructure_error is not None:
            raise ToolUsageError(self._ended_text())
        if not self.started:
            raise ToolUsageError(f"Call {START_TOOL} first.")
        message = self.orchestrator.message
        text = ""
        is_error = False
        if isinstance(message, UserMessage):
            text = message.content or ""
        elif isinstance(message, ToolMessage):
            text = message.content or ""
            is_error = bool(message.error)
        elif isinstance(message, MultiToolMessage):
            text = "\n\n".join(item.content or "" for item in message.tool_messages)
            is_error = any(item.error for item in message.tool_messages)
        if with_history and (history := self._history_text()):
            text = f"{history}\n\nThe customer's latest message:\n{text}"
        if self.finished:
            text = "\n\n".join(part for part in (text, self._ended_text()) if part)
        return ToolResult(text, is_error=is_error)

    def _history_text(self) -> str:
        """The conversation a task starts from, as the agent would have seen it."""
        initial = self.task.initial_state
        history = initial.message_history if initial is not None else None
        lines = []
        for message in history or []:
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls or []:
                    arguments = json.dumps(call.arguments, sort_keys=True)
                    lines.append(f"You called {call.name}({arguments})")
                if message.content:
                    lines.append(f"You: {message.content}")
            elif isinstance(message, UserMessage) and not message.is_tool_call():
                lines.append(f"Customer: {message.content or ''}")
            elif isinstance(message, ToolMessage) and message.requestor == "assistant":
                lines.append(f"Tool result: {message.content or ''}")
        if not lines:
            return ""
        return "The conversation so far:\n" + "\n".join(lines)

    def _ended_text(self) -> str:
        if self.infrastructure_error is not None:
            return f"The conversation runtime failed: {self.infrastructure_error}"
        reason = self.result.termination_reason if self.result else None
        if reason == TerminationReason.USER_STOP:
            return "[The customer ended the conversation. No further actions are possible.]"
        if reason == TerminationReason.AGENT_STOP:
            return "[You ended the conversation. No further actions are possible.]"
        value = reason.value if reason is not None else "unknown"
        return f"[The conversation is over ({value}). No further actions are possible.]"

    # Oracle -------------------------------------------------------------------

    def _oracle_replay(self, arguments: dict[str, Any]) -> ToolResult:
        """Record the task's reference actions as the conversation."""
        supplied = str(arguments.get("oracle_token", ""))
        if not self.oracle_token or not secrets.compare_digest(
            supplied, self.oracle_token
        ):
            raise ToolUsageError(f"Unknown tool: {ORACLE_TOOL}")
        if self.started or self.finished:
            raise ToolUsageError("The conversation has already started")
        task = self.task
        initial_state = task.initial_state
        history = (
            [message.model_copy(deep=True) for message in initial_state.message_history]
            if initial_state is not None and initial_state.message_history
            else []
        )
        start_time = get_now()
        self.orchestrator._initialize_environment(
            initialization_data=(
                initial_state.initialization_data if initial_state else None
            ),
            initialization_actions=(
                initial_state.initialization_actions if initial_state else None
            ),
            message_history=history,
        )
        messages: list[Message] = list(history)
        criteria = task.evaluation_criteria
        for index, action in enumerate(criteria.actions or [] if criteria else []):
            call = ToolCall(
                id=f"oracle_{index}",
                name=action.name,
                arguments=action.arguments,
                requestor=action.requestor,
            )
            turn_type = (
                AssistantMessage if action.requestor == "assistant" else UserMessage
            )
            messages.append(
                turn_type(role=action.requestor, content=None, tool_calls=[call])
            )
            messages.append(self.environment.get_response(call))
        for info in (criteria.communicate_info or []) if criteria else []:
            messages.append(AssistantMessage(role="assistant", content=info))
        messages.append(AssistantMessage(role="assistant", content=STOP_TOKEN))
        for index, message in enumerate(messages):
            message.turn_idx = index
        agent_cost, user_cost = get_cost(messages)
        self.result = SimulationRun(
            id=self.orchestrator.simulation_id,
            task_id=task.id,
            start_time=start_time,
            end_time=get_now(),
            duration=0.0,
            termination_reason=TerminationReason.AGENT_STOP,
            agent_cost=agent_cost,
            user_cost=user_cost,
            messages=messages,
            seed=self.config.seed,
            mode=self.orchestrator.mode.value,
            policy=self.environment.get_policy(),
        )
        self.started = True
        self.finished = True
        self.oracle = True
        self.persist()
        return ToolResult(f"Recorded {len(messages)} reference messages.")

    # State ----------------------------------------------------------------------

    def _partial_run(self) -> SimulationRun:
        orchestrator = self.orchestrator
        messages = orchestrator.get_trajectory() if self.started else []
        agent_cost, user_cost = get_cost(messages) if messages else (None, None)
        start_perf = getattr(orchestrator, "_run_start_perf", None)
        return SimulationRun(
            id=orchestrator.simulation_id,
            task_id=self.task.id,
            start_time=getattr(orchestrator, "_run_start_time", None) or get_now(),
            end_time=get_now(),
            duration=time.perf_counter() - start_perf if start_perf else 0.0,
            termination_reason=(
                TerminationReason.INFRASTRUCTURE_ERROR
                if self.infrastructure_error is not None
                else TerminationReason.AGENT_ERROR
            ),
            agent_cost=agent_cost,
            user_cost=user_cost,
            messages=messages,
            seed=self.config.seed,
            mode=orchestrator.mode.value,
            policy=self.environment.get_policy(),
        )

    def snapshot(self) -> dict[str, Any]:
        simulation = self.result or self._partial_run()
        return {
            "domain": self.config.domain,
            "task_id": self.task.id,
            "started": self.started,
            "finished": self.finished and self.infrastructure_error is None,
            "oracle": self.oracle,
            "infrastructure_error": self.infrastructure_error,
            "termination_reason": (
                self.result.termination_reason.value if self.result else None
            ),
            "simulation": simulation.model_dump(mode="json"),
        }

    def persist(self) -> None:
        write_json_atomic(self.state_path, self.snapshot())


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--domain", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--bridge-dir", default="/var/run/tau2-bridge")
    parser.add_argument("--state-dir", default="/var/lib/tau2-harbor")
    parser.add_argument("--retrieval-config", default=None)
    parser.add_argument("--user-llm", default=None)
    parser.add_argument("--user-llm-args", default=None, help="JSON object")
    args = parser.parse_args(argv)

    token = os.environ.get("TAU2_HARBOR_BRIDGE_TOKEN", "")
    oracle_token = os.environ.get("TAU2_HARBOR_ORACLE_TOKEN") or None
    # The exported task fixes the user simulator; a run may override it.
    user_llm = os.environ.get("TAU2_HARBOR_USER_LLM") or args.user_llm
    raw_user_args = (
        os.environ.get("TAU2_HARBOR_USER_LLM_ARGS") or args.user_llm_args or ""
    )
    user_llm_args = json.loads(raw_user_args) if raw_user_args.strip() else None

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / READY_FILE).unlink(missing_ok=True)
    config = build_run_config(
        args.domain,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        retrieval_config=args.retrieval_config,
    )
    conversation = InnerConversation(
        load_task(args.domain, args.task_id),
        config,
        state_path=state_dir / STATE_FILE,
        oracle_token=oracle_token,
    )
    bridge = ToolBridge(
        Path(args.bridge_dir),
        token,
        list_tools=conversation.list_tools,
        call_tool=conversation.call,
    )
    bridge.start()
    (state_dir / READY_FILE).write_text("ready\n")
    logger.info(
        f"tau-runtime ready: domain={args.domain} task={args.task_id} "
        f"user_llm={config.llm_user}"
    )

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    stop.wait()
    bridge.stop()
    conversation.persist()


if __name__ == "__main__":
    main()
