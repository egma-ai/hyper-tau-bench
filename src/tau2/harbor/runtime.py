"""Trusted Hyper-τ construction runtime for one Harbor task (the ``hyper-runtime`` sidecar).

The Harbor agent plays the Developer. It edits the kit in ``/workspace``, a
volume this sidecar shares with the agent's container and with the
``candidate`` service that runs the Developer's code. This process owns
everything the Developer must not see, exactly as the host process does in
``tau2 hyper-tau``: it builds the kit, holds the Client simulator and the
hidden task partitions, and serves the Developer tools (``run_local_test``,
``talk_to_client``, ``run_live_experiment``, ``run_sample_scenarios``,
``submit``) through the file bridge.

Construction ends at ``submit`` or, failing that, when the agent phase
ends (Harbor runs ``finalize`` as a collect hook). Either way the runtime
seals the Client API deployment, snapshots the workspace, and records the
session for the verifier (:mod:`tau2.harbor.verify`), which scores
the snapshot with the sealed inner loop.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import tarfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tau2.harbor.bridge import ToolBridge, ToolResult
from tau2.harbor.common import warm_knowledge_base, write_json_atomic
from tau2.hyper.data_model import HyperTask
from tau2.hyper.result_serialization import bounded_tool_result
from tau2.hyper.sandbox.builder import BuildBudget, BuildResult, BuildStep
from tau2.hyper.sandbox.callback_broker import CallbackBroker, CallbackBrokerError
from tau2.hyper.sandbox.callback_mcp import _tool_definitions
from tau2.hyper.sandbox.orchestrator import SandboxOrchestrator
from tau2.hyper.task_loader import load_active_hyper_tau_task

WORKSPACE = Path("/workspace")
STATE_DIR = Path("/var/lib/hyper-tau")
BRIDGE_DIR = Path("/var/run/hyper-tau-bridge")
TASK_CONFIG_PATH = Path("/etc/hyper-tau/task.json")
CONFIG_ENV_VAR = "HYPER_TAU_TASK_CONFIG"
TOKEN_ENV_VAR = "HYPER_TAU_BRIDGE_TOKEN"
STATE_FILE = "state.json"
SUBMISSION_FILE = "submission.tar.gz"
READY_FILE = "ready"
FINALIZED_FILE = "finalized"
PID_FILE = "runtime.pid"
TOOL_RESULT_CHARS = 4000

SUBMIT_TOOL = "submit"


@dataclass
class ConstructionConfig:
    """The exported task's construction settings (shared with the verifier)."""

    task_id: str
    bridge_token: str = ""
    client_opening: Optional[str] = None
    client_llm: Optional[str] = None
    user_llm: Optional[str] = None
    agent_llm: Optional[str] = None
    primary_model_only: bool = False
    allow_legacy_domain: bool = False
    max_time_seconds: int = 0

    @classmethod
    def from_json(cls, text: str) -> "ConstructionConfig":
        return cls(**json.loads(text))

    def to_json(self) -> str:
        """The settings as JSON, without the bridge token."""
        return json.dumps(self.public(), sort_keys=True)

    def public(self) -> dict[str, Any]:
        """The settings without the bridge token."""
        data = asdict(self)
        data.pop("bridge_token")
        return data


class HarborToolBuilder:
    """Collects the Developer-tool contexts the orchestrator hands a builder.

    Harbor runs the Developer itself, so this builder never builds; it only
    receives what :meth:`SandboxOrchestrator.prepare_construction` wires
    into a builder, for this runtime to serve.
    """

    def __init__(self) -> None:
        self.response_phrasing_pack = None
        self.local_test_wiring = None
        self.client_context = None
        self.live_experiment_context = None
        self.sample_scenarios_context = None

    def set_client_context(self, ctx) -> None:
        self.client_context = ctx

    def set_live_experiment_context(self, ctx) -> None:
        self.live_experiment_context = ctx

    def set_sample_scenarios_context(self, ctx) -> None:
        self.sample_scenarios_context = ctx

    def build(self, *args, **kwargs) -> BuildResult:
        raise RuntimeError("In a Harbor task the Harbor agent is the Developer")


def load_construction_task(config: ConstructionConfig) -> HyperTask:
    return load_active_hyper_tau_task(
        config.task_id, allow_legacy=config.allow_legacy_domain
    )


def make_orchestrator(
    task: HyperTask, config: ConstructionConfig, builder=None
) -> SandboxOrchestrator:
    """Resolve the run's model seats exactly as ``tau2 hyper-tau`` does."""
    allowed_agent_models_override = None
    if config.primary_model_only:
        from tau2.hyper.performance_profiles import get_primary_model_config

        if not isinstance(task.performance_profile, str):
            raise ValueError(
                "primary_model_only requires one stock performance tier; "
                f"task {task.id!r} has a multi-tier or custom profile"
            )
        allowed_agent_models_override = [
            get_primary_model_config(task.performance_profile, task.source_domain)
        ]
    sandbox_config = task.hyper.sandbox_config or {}
    budget = BuildBudget(
        max_steps=sandbox_config.get("max_steps", 0),
        max_time_seconds=(
            config.max_time_seconds or sandbox_config.get("max_time_seconds") or 0
        ),
    )
    return SandboxOrchestrator.from_task(
        task,
        builder if builder is not None else HarborToolBuilder(),
        client_llm=config.client_llm,
        user_llm=config.user_llm,
        agent_llm=config.agent_llm,
        allowed_agent_models_override=allowed_agent_models_override,
        budget=budget,
    )


def snapshot_workspace(workspace: Path, archive: Path) -> None:
    """Archive the workspace as submitted (including its Git history)."""
    temporary = archive.with_name(archive.name + ".tmp")
    # Kits are mostly already-compressed documents: favour speed.
    with tarfile.open(temporary, "w:gz", compresslevel=1) as bundle:
        bundle.add(workspace, arcname=".")
    os.replace(temporary, archive)


@dataclass
class SessionRecord:
    """What the runtime observed of the construction session."""

    started_at: float = field(default_factory=time.time)
    steps: list[BuildStep] = field(default_factory=list)
    tool_calls: int = 0


class ConstructionRuntime:
    """Serve one Developer session and record it for the verifier."""

    def __init__(
        self,
        config: ConstructionConfig,
        *,
        workspace: Path = WORKSPACE,
        state_dir: Path = STATE_DIR,
        bridge_dir: Path = BRIDGE_DIR,
    ):
        self.config = config
        self.workspace = Path(workspace)
        self.state_dir = Path(state_dir)
        self.bridge_dir = Path(bridge_dir)
        self.task = load_construction_task(config)
        self.builder = HarborToolBuilder()
        self.orchestrator = make_orchestrator(self.task, config, self.builder)
        self.broker: Optional[CallbackBroker] = None
        self.bridge: Optional[ToolBridge] = None
        self.record = SessionRecord()
        self.kit_baseline_commit = ""
        self.client_opening = ""
        self.done_reason = ""
        self._finalize_lock = threading.Lock()
        self._finalized = threading.Event()

    # Lifecycle ----------------------------------------------------------------

    def start(self) -> None:
        """Build the kit, open the Client conversation, and serve the tools."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        start = self.orchestrator.prepare_construction(
            self.workspace, client_opening=self.config.client_opening
        )
        self.kit_baseline_commit = start.kit_baseline_commit
        self.client_opening = start.client_opening
        # The agent may run as any user in its container.
        for path in [self.workspace, *self.workspace.rglob("*")]:
            if not path.is_symlink():
                path.chmod(
                    path.stat().st_mode | 0o666 | (0o111 if path.is_dir() else 0)
                )
        self.broker = CallbackBroker(
            self.workspace,
            client_context=self.builder.client_context,
            live_experiment_context=self.builder.live_experiment_context,
            sample_scenarios_context=self.builder.sample_scenarios_context,
            response_phrasing_pack=self.builder.response_phrasing_pack,
            local_test_wiring=self.builder.local_test_wiring,
        )
        tools = _tool_definitions(
            include_client=self.broker.client_tool_enabled,
            include_live_experiment=self.broker.live_experiment_tool_enabled,
            include_sample_scenarios=self.broker.sample_scenarios_tool_enabled,
        )
        self.bridge = ToolBridge(
            self.bridge_dir,
            self.config.bridge_token,
            list_tools=lambda: tools,
            call_tool=self.call,
        )
        warm_knowledge_base(self.task.source_domain)
        self.bridge.start()
        self.persist()
        (self.state_dir / READY_FILE).write_text("ready\n")
        logger.info(
            f"hyper-runtime ready: task={self.task.id} "
            f"domain={self.task.source_domain} tools={[t['name'] for t in tools]}"
        )

    def stop(self) -> None:
        if self.bridge is not None:
            self.bridge.stop()

    # Tools ----------------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        assert self.broker is not None
        if self._finalized.is_set():
            return ToolResult(
                "Construction has ended and the submission was recorded; "
                "no further tool calls are possible.",
                is_error=True,
            )
        started = time.time()
        try:
            text = self.broker.dispatch(
                token=self.broker.token, tool=name, arguments=arguments
            )
            result = ToolResult(str(text))
        except CallbackBrokerError as error:
            result = ToolResult(str(error), is_error=True)
        except Exception as error:  # noqa: BLE001 - shown to the Developer
            logger.exception(f"Developer tool {name} failed")
            result = ToolResult(f"{type(error).__name__}: {error}", is_error=True)
        self._record(name, arguments, result, started)
        if name == SUBMIT_TOOL and self.broker.submitted.is_set():
            try:
                self.finalize("submitted")
            except Exception as error:  # noqa: BLE001 - shown to the Developer
                logger.exception("Recording the submission failed")
                return ToolResult(
                    f"Recording the submission failed ({error}); call submit again.",
                    is_error=True,
                )
        else:
            self.persist()
        return result

    def _record(
        self, name: str, arguments: dict, result: ToolResult, started: float
    ) -> None:
        self.record.tool_calls += 1
        call_id = f"call_{self.record.tool_calls}"
        self.record.steps.append(
            BuildStep(
                step_idx=len(self.record.steps) + 1,
                role="assistant",
                tool_calls=[{"id": call_id, "name": name, "arguments": arguments}],
                tool_results=[
                    {
                        "tool_call_id": call_id,
                        "name": name,
                        "content": bounded_tool_result(result.text, TOOL_RESULT_CHARS),
                        "is_error": result.is_error,
                    }
                ],
                timestamp=started - self.record.started_at,
            )
        )

    # Finalization ---------------------------------------------------------------

    def finalize(self, reason: str) -> None:
        """Seal the deployment, snapshot the workspace, and record the session."""
        with self._finalize_lock:
            if self._finalized.is_set():
                return
            self.done_reason = reason
            self.orchestrator.seal_deployment()
            snapshot_workspace(self.workspace, self.state_dir / SUBMISSION_FILE)
            self._finalized.set()
            self.persist()
            (self.state_dir / FINALIZED_FILE).write_text(reason + "\n")
            logger.info(f"Construction finalized: {reason}")

    def build_result(self) -> BuildResult:
        broker = self.broker
        client = self.builder.client_context
        return BuildResult(
            kit_path=self.workspace,
            steps=list(self.record.steps),
            total_steps=len(self.record.steps),
            total_tool_calls=self.record.tool_calls,
            client_turns_used=client.turns_used if client is not None else 0,
            submitted=bool(broker and broker.submitted.is_set()),
            done_reason=self.done_reason,
            elapsed_seconds=time.time() - self.record.started_at,
            builder_type="harbor",
            metadata={
                "harness": {"name": "harbor", "version": "", "config": {}},
                "callback": broker.metadata() if broker is not None else {},
                "trajectory_scope": "developer tool calls served by hyper-runtime",
            },
        )

    def snapshot(self) -> dict[str, Any]:
        orchestrator = self.orchestrator
        snapshot = orchestrator._deployment_snapshot
        client = self.builder.client_context
        sample = self.builder.sample_scenarios_context
        build = self.build_result()
        return {
            "task_id": self.task.id,
            "source_domain": self.task.source_domain,
            "config": self.config.public(),
            "finalized": self._finalized.is_set(),
            "kit_baseline_commit": self.kit_baseline_commit,
            "client_opening": self.client_opening,
            "deployment_snapshot": (
                snapshot.model_dump(mode="json") if snapshot is not None else None
            ),
            "inner_loop": {
                "agent_llm": orchestrator.agent_llm,
                "agent_llm_args": orchestrator.agent_llm_args or {},
                "allowed_agent_models": orchestrator.allowed_agent_models or [],
                "user_llm": orchestrator.user_llm,
                "user_llm_args": orchestrator.user_llm_args or {},
            },
            "client": client.result_metadata() if client is not None else None,
            "sample_scenarios": {
                "enabled": bool(self.task.training_task_ids),
                "scenario_count": len(self.task.training_task_ids),
                "runs_used": sample.runs_used if sample is not None else 0,
                "max_runs": sample.max_runs if sample is not None else 0,
            },
            "build_result": {
                "submitted": build.submitted,
                "done_reason": build.done_reason,
                "elapsed_seconds": build.elapsed_seconds,
                "total_steps": build.total_steps,
                "total_tool_calls": build.total_tool_calls,
                "client_turns_used": build.client_turns_used,
                "metadata": build.metadata,
                "steps": [asdict(step) for step in build.steps],
            },
        }

    def persist(self) -> None:
        write_json_atomic(self.state_dir / STATE_FILE, self.snapshot())


def request_finalize(state_dir: Path, *, timeout: float = 600.0) -> int:
    """Ask the running runtime to finalize (the agent phase is over)."""
    state_dir = Path(state_dir)
    marker = state_dir / FINALIZED_FILE
    if marker.exists():
        print(f"already finalized: {marker.read_text().strip()}")
        return 0
    pid = int((state_dir / PID_FILE).read_text())
    os.kill(pid, signal.SIGUSR1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            print(f"finalized: {marker.read_text().strip()}")
            return 0
        time.sleep(0.2)
    print("timed out waiting for the runtime to finalize")
    return 1


def serve(config: ConstructionConfig, state_dir: Path) -> None:
    runtime = ConstructionRuntime(config, state_dir=state_dir)
    finalize_requested = threading.Event()
    stop = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: finalize_requested.set())
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    (Path(state_dir) / PID_FILE).parent.mkdir(parents=True, exist_ok=True)
    (Path(state_dir) / PID_FILE).write_text(str(os.getpid()))
    runtime.start()
    while not stop.is_set():
        if finalize_requested.wait(0.5):
            finalize_requested.clear()
            # Off the signal path: a tool call may still be running.
            threading.Thread(
                target=runtime.finalize, args=("agent_ended",), daemon=True
            ).start()
    runtime.stop()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "command", nargs="?", default="serve", choices=["serve", "finalize"]
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=TASK_CONFIG_PATH,
        help=f"task settings (JSON); ${CONFIG_ENV_VAR} takes precedence",
    )
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.command == "finalize":
        return request_finalize(args.state_dir, timeout=args.timeout)
    text = os.environ.get(CONFIG_ENV_VAR) or args.config.read_text()
    config = ConstructionConfig.from_json(text)
    config.bridge_token = os.environ.get(TOKEN_ENV_VAR) or config.bridge_token
    if not config.bridge_token:
        raise ValueError(f"No bridge token: set ${TOKEN_ENV_VAR}")
    serve(config, args.state_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
