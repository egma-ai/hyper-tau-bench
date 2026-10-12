"""Harbor packaging for the Hyper-τ construction tasks.

`Harbor <https://github.com/harbor-framework/harbor>`_ runs an agent in a
task's ``main`` container and scores it with the task's verifier. This
package turns each Hyper-τ construction task into one Harbor task in which
the Harbor agent is the Developer working in the kit:

- :mod:`tau2.harbor.runtime` — the trusted ``hyper-runtime`` sidecar. It
  builds the kit and serves the Developer tools (local tests, the Client,
  live experiments, sample scenarios, submit).
- :mod:`tau2.harbor.verify` — the verifier. It scores the submission on the
  hidden test conversations with the sealed inner loop.
- :mod:`tau2.harbor.export` — writes a Harbor dataset of these tasks.

The agent reaches its tools through a stdio MCP server in ``main``
(:mod:`tau2.harbor.mcp_stub`) that forwards calls over files in a volume
shared with the sidecar (see :mod:`tau2.harbor.bridge`), so ``main`` needs
no route to the sidecar and holds no task state, hidden tasks, or
credentials beyond the agent's own.
"""
