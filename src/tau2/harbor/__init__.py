"""Harbor packaging for both Hyper-τ loops.

`Harbor <https://github.com/harbor-framework/harbor>`_ runs an agent in a
task's ``main`` container and scores it with the task's verifier. This
package turns each loop into Harbor tasks:

- :mod:`tau2.harbor.inner` — one τ³ customer-service conversation per task.
  The Harbor agent is the customer-service agent; a trusted ``tau-runtime``
  sidecar runs the real tau2 orchestrator, environment, and user simulator,
  and the verifier scores the recorded conversation with the tau2
  evaluator.
- :mod:`tau2.harbor.outer` — one Hyper-τ construction task per task. The
  Harbor agent is the Developer working in the kit; a trusted
  ``hyper-runtime`` sidecar builds the kit and serves the Developer tools
  (local tests, the Client, live experiments, sample scenarios, submit),
  and the verifier scores the submission with the sealed inner loop.

In both, the agent reaches its tools through a stdio MCP server in ``main``
that forwards calls over files in a volume shared with the sidecar (see
:mod:`tau2.harbor.bridge`), so ``main`` needs no route to any sidecar and
holds no task state, hidden tasks, or credentials beyond the agent's own.
"""
