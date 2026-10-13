"""Harbor verifier for a Hyper-τ construction task.

Runs in Harbor's separate verifier environment, after the agent phase. The
``hyper-runtime`` sidecar's state directory arrives as an artifact: the
workspace snapshot taken when the Developer submitted (or when the agent
phase ended) and the session record. The snapshot is unpacked into the
verifier's ``/workspace``, which it shares with a fresh sealed ``candidate``
service, and scored exactly as ``tau2 hyper-tau`` scores a submission: the
hidden test partition through the sealed inner loop, against the Client API
deployment the Developer left in place, with the task's performance
requirements applied.

Writes ``reward.json`` (``reward`` is the final Hyper-τ reward, alongside
its quality, performance-multiplier and penalty components) and the full
``outer_loop_result.json`` to the verifier log directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tau2.harbor.common import warm_knowledge_base, write_json_atomic, write_reward
from tau2.harbor.runtime import (
    STATE_DIR,
    STATE_FILE,
    SUBMISSION_FILE,
    WORKSPACE,
    ConstructionConfig,
    load_construction_task,
    make_orchestrator,
)
from tau2.hyper.client_api.capabilities import DeploymentSnapshot
from tau2.hyper.data_model import OuterLoopResult
from tau2.hyper.sandbox.builder import BuildResult, BuildStep

DEFAULT_LOG_DIR = Path("/logs/verifier")
RESULT_FILE = "outer_loop_result.json"


class SubmissionError(RuntimeError):
    """The construction session left nothing to score (an infrastructure fault)."""


def load_state(state_dir: Path) -> dict[str, Any]:
    path = Path(state_dir) / STATE_FILE
    if not path.is_file():
        raise SubmissionError(f"No construction record at {path}")
    state = json.loads(path.read_text())
    if not state.get("finalized"):
        raise SubmissionError(
            "The construction session was never finalized, so no submission "
            "was recorded"
        )
    return state


def _skip_unsafe_member(
    member: tarfile.TarInfo, destination: str
) -> Optional[tarfile.TarInfo]:
    """Python's ``data`` filter, skipping (not failing on) refused entries.

    The snapshot is the Developer's workspace, so it may hold links that
    point outside it or special files; those are dropped, not followed.
    """
    try:
        return tarfile.data_filter(member, destination)
    except tarfile.FilterError as error:
        logger.warning(f"Skipping {member.name!r} from the submission: {error}")
        return None


def extract_submission(archive: Path, workspace: Path) -> None:
    """Unpack the submitted workspace into an empty ``workspace``."""
    archive = Path(archive)
    if not archive.is_file():
        raise SubmissionError(f"No workspace snapshot at {archive}")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    for child in workspace.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(workspace, filter=_skip_unsafe_member)
    # The sealed candidate reads the workspace under unprivileged uids.
    workspace.chmod(0o755)


def build_result_from_state(state: dict[str, Any], workspace: Path) -> BuildResult:
    record = state["build_result"]
    return BuildResult(
        kit_path=Path(workspace),
        steps=[BuildStep(**step) for step in record.get("steps", [])],
        total_steps=record.get("total_steps", 0),
        total_tool_calls=record.get("total_tool_calls", 0),
        client_turns_used=record.get("client_turns_used", 0),
        submitted=bool(record.get("submitted")),
        done_reason=record.get("done_reason", ""),
        elapsed_seconds=float(record.get("elapsed_seconds", 0.0)),
        builder_type="harbor",
        metadata=record.get("metadata", {}),
    )


def score_state(
    state: dict[str, Any], *, workspace: Path, state_dir: Path
) -> OuterLoopResult:
    """Score the recorded submission with the sealed inner loop."""
    config = ConstructionConfig(**state["config"])
    task = load_construction_task(config)
    if task.id != state.get("task_id"):
        raise SubmissionError(
            f"The construction record is for task {state.get('task_id')!r}, "
            f"not {task.id!r}"
        )
    extract_submission(Path(state_dir) / SUBMISSION_FILE, workspace)

    orchestrator = make_orchestrator(task, config)
    snapshot = state.get("deployment_snapshot")
    if snapshot is not None:
        orchestrator._deployment_snapshot = DeploymentSnapshot(**snapshot)
    overrides: dict[str, Any] = {
        "sample_scenarios": state["sample_scenarios"],
        "harbor": {
            "client_opening": state.get("client_opening", ""),
            "construction_config": state["config"],
            "done_reason": state["build_result"].get("done_reason", ""),
        },
    }
    if state.get("client") is not None:
        overrides["client"] = state["client"]
    warm_knowledge_base(task.source_domain)
    return orchestrator.score_submission(
        Path(workspace),
        build_result_from_state(state, workspace),
        kit_baseline_commit=state["kit_baseline_commit"],
        run_metadata_overrides=overrides,
    )


def rewards_for(result: OuterLoopResult) -> dict[str, float]:
    quality = result.final_quality_reward
    return {
        "reward": float(result.final_test_reward),
        "quality_reward": float(
            quality if quality is not None else result.final_test_reward
        ),
        "performance_reward": float(result.performance_reward),
        "performance_penalty": float(result.performance_penalty),
    }


def log_scoring_errors(result: OuterLoopResult) -> None:
    """Point out test conversations that scored zero because they errored."""
    details = result.test_details or []
    kinds = Counter(
        str(detail["error"]).split(":", 1)[0]
        for detail in details
        if detail.get("error")
    )
    if kinds:
        logger.warning(
            f"{sum(kinds.values())} of {len(details)} test "
            f"conversations ended in an error ({dict(kinds)}); "
            f"see {RESULT_FILE} for each one"
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args(argv)

    state = load_state(args.state_dir)
    result = score_state(state, workspace=args.workspace, state_dir=args.state_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.log_dir / RESULT_FILE, result.model_dump(mode="json"))
    rewards = rewards_for(result)
    write_reward(args.log_dir, rewards)
    log_scoring_errors(result)
    logger.info(f"Hyper-τ rewards: {rewards}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
