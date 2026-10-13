"""The construction image installs the harness versions the drivers record.

Each Developer harness driver records its version in the run's harness
identity; docker/hyper-construction/Dockerfile installs each harness at a
pinned version. The two must agree, or runs record a harness they did not
run.
"""

import re
from pathlib import Path

from tau2.hyper.harnesses.claude import CLAUDE_CODE_HARNESS_VERSION
from tau2.hyper.harnesses.codex import CODEX_HARNESS_VERSION
from tau2.hyper.harnesses.opencode import OPENCODE_HARNESS_VERSION
from tau2.hyper.harnesses.prime import PRIME_AGENT_HARNESS_VERSION

DOCKERFILE = (
    Path(__file__).resolve().parents[2] / "docker" / "hyper-construction" / "Dockerfile"
)


def test_construction_image_installs_the_recorded_harness_versions():
    args = dict(re.findall(r"^ARG (\w+)=(\S+)$", DOCKERFILE.read_text(), re.M))

    assert args["CODEX_VERSION"] == CODEX_HARNESS_VERSION
    assert args["CLAUDE_CODE_VERSION"] == CLAUDE_CODE_HARNESS_VERSION
    assert args["OPENCODE_VERSION"] == OPENCODE_HARNESS_VERSION
    assert args["PRIME_AGENT_VERSION"] == PRIME_AGENT_HARNESS_VERSION
