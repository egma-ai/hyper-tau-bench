"""Shared fixtures for the Hyper-τ tests."""

import pytest


@pytest.fixture(autouse=True)
def _restore_real_generate(monkeypatch):
    """Undo the candidate server's process-wide model broker after each test.

    Constructing ``CandidateServer`` (or running ``candidate_server.main``)
    replaces ``generate`` in these modules with a broker that reads model
    responses from stdin. Without this, every later test that calls a model —
    e.g. tests/test_orchestrator.py and tests/test_run.py in a full run —
    fails with "reading from stdin while output is captured".
    """
    import tau2.agent.llm_agent as llm_agent
    import tau2.utils.llm_utils as llm_utils
    from tau2.hyper import agent_context

    for module in (llm_utils, agent_context, llm_agent):
        monkeypatch.setattr(module, "generate", module.generate)
