"""Harbor tasks for the τ³ inner loop: one customer-service conversation each.

- :mod:`.runtime` — the trusted ``tau-runtime`` sidecar that runs the tau2
  orchestrator, environment, and user simulator behind the agent's tools.
- :mod:`.verify` — the verifier that scores the recorded conversation.
- :mod:`.export` — writes a Harbor dataset of these tasks.
"""
