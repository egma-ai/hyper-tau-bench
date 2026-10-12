# Trusted runtime for the Harbor tasks `tau2 harbor export` writes.
#
# It carries the full tau2 source and data (domains, Hyper-τ tasks and
# evidence, model routing) and runs only on the trusted side of a task: the
# `hyper-runtime` sidecar and the verifier. Harbor agents and
# Developer-written candidate code never run in this image (the agent's
# container and the sealed `candidate` service use the construction runtime
# image instead).
FROM node:24-bookworm-slim AS node-runtime

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG TAU2_SOURCE_REVISION=unknown
ARG SANDBOX_RUNTIME_VERSION=0.0.23

LABEL org.opencontainers.image.revision="${TAU2_SOURCE_REVISION}"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV TAU2_SOURCE_REVISION="${TAU2_SOURCE_REVISION}"

# Git builds each kit's baseline repository and records the Developer's
# changes when a submission is scored. banking_knowledge environments use the
# default retrieval configuration (alltools), whose agentic-shell sandbox
# needs sandbox-runtime (srt) and the tools it shells out to: without them
# every banking conversation fails before it starts.
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap ca-certificates git ripgrep socat \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && npm install --global @anthropic-ai/sandbox-runtime@${SANDBOX_RUNTIME_VERSION} \
    && npm cache clean --force

WORKDIR /opt/tau2

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CONCURRENT_DOWNLOADS=1 uv sync --frozen --no-dev --extra knowledge \
        --no-install-project

COPY model_routing.toml ./
COPY src ./src
COPY data/tau2 ./data/tau2
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra knowledge \
    && .venv/bin/python -c \
        "import tau2.harbor.runtime, tau2.harbor.verify; \
from tau2.knowledge.sandbox_manager import _check_sandbox_dependencies; \
_check_sandbox_dependencies()"

ENV PATH="/opt/tau2/.venv/bin:${PATH}"
ENV TAU2_DATA_DIR="/opt/tau2/data"
