"""Verifier for an inner-loop Harbor task: score the recorded τ³ conversation.

Reads the conversation the ``tau-runtime`` sidecar recorded (collected into
the verifier as a task artifact) and scores it with the same tau2 evaluator
``tau2 run`` uses, so a Harbor trial and a tau2 simulation of the same
conversation get the same reward. Writes Harbor's ``reward.json`` plus the
full simulation and reward breakdown next to it.

A conversation the agent never finished (no ``end_conversation`` and no
customer stop) scores 0, like a tau2 run that hits its step limit. A
conversation the runtime itself could not finish (the user simulator's
model failed, say) is an infrastructure error: the verifier exits non-zero
so Harbor records an error instead of a zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.harbor.common import write_json_atomic, write_reward
from tau2.harbor.inner.runtime import STATE_FILE, build_run_config, load_task
from tau2.runner.build import _build_env_kwargs

DEFAULT_STATE_PATH = Path("/var/lib/tau2-harbor") / STATE_FILE


class ConversationStateError(RuntimeError):
    """The recorded conversation cannot be scored."""


def score_conversation(
    state: dict[str, Any],
    task: Task,
    *,
    retrieval_config: Optional[str] = None,
    evaluation_type: EvaluationType = EvaluationType.ALL,
) -> tuple[SimulationRun, RewardInfo]:
    """Score one recorded conversation and return it with its reward."""
    if state.get("infrastructure_error"):
        raise ConversationStateError(
            "The conversation runtime failed: " + str(state["infrastructure_error"])
        )
    domain = state["domain"]
    simulation = SimulationRun.model_validate(state["simulation"])
    if state.get("task_id") != task.id or simulation.task_id != task.id:
        raise ConversationStateError(
            f"Recorded conversation is for task {state.get('task_id')!r}, "
            f"not {task.id!r}"
        )
    if not state.get("finished"):
        reward_info = RewardInfo(
            reward=0.0,
            reward_basis=None,
            info={
                "note": (
                    "The conversation was not finished: the agent never called "
                    "end_conversation and the customer never ended it."
                    if state.get("started")
                    else "The agent never started the conversation."
                )
            },
        )
    else:
        config = build_run_config(domain, retrieval_config=retrieval_config)
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=evaluation_type,
            solo_mode=False,
            domain=domain,
            env_kwargs=_build_env_kwargs(config, task),
        )
    simulation.reward_info = reward_info
    return simulation, reward_info


def rewards_for(reward_info: RewardInfo) -> dict[str, float]:
    """Harbor metrics: the headline reward plus each scored component."""
    rewards = {"reward": float(reward_info.reward)}
    for component, value in (reward_info.reward_breakdown or {}).items():
        name = getattr(component, "value", component)
        rewards[str(name).lower()] = float(value)
    return rewards


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--domain", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--retrieval-config", default=None)
    parser.add_argument(
        "--evaluation-type",
        default=EvaluationType.ALL.value,
        choices=[item.value for item in EvaluationType],
    )
    args = parser.parse_args(argv)

    if not args.state.exists():
        raise ConversationStateError(
            f"No recorded conversation at {args.state}; the tau-runtime sidecar "
            "never started"
        )
    state = json.loads(args.state.read_text())
    if state.get("domain") != args.domain:
        raise ConversationStateError(
            f"Recorded conversation is for domain {state.get('domain')!r}, "
            f"not {args.domain!r}"
        )
    task = load_task(args.domain, args.task_id)
    simulation, reward_info = score_conversation(
        state,
        task,
        retrieval_config=args.retrieval_config,
        evaluation_type=EvaluationType(args.evaluation_type),
    )
    rewards = rewards_for(reward_info)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.log_dir / "simulation.json", simulation.model_dump(mode="json")
    )
    write_json_atomic(
        args.log_dir / "details.json",
        {
            "domain": args.domain,
            "task_id": task.id,
            "oracle": bool(state.get("oracle")),
            "finished": bool(state.get("finished")),
            "termination_reason": state.get("termination_reason"),
            "rewards": rewards,
            "reward_info": reward_info.model_dump(mode="json"),
        },
    )
    write_reward(args.log_dir, rewards)
    logger.info(f"Task {task.id}: reward={rewards['reward']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
