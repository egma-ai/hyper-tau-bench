"""Shared fixtures for the Hyper-τ tests."""

import pytest


@pytest.fixture(autouse=True)
def restore_model_generate(monkeypatch):
    """Undo the model broker a ``CandidateServer`` installs in this process.

    A ``CandidateServer`` routes ``generate`` through its stdio broker for the
    rest of the process, as a sealed candidate needs. Tests that build one
    in-process must not leave that behind: later tests would send their model
    calls to pytest's stdin.
    """
    import tau2.agent.llm_agent as llm_agent
    import tau2.utils.llm_utils as llm_utils
    from tau2.hyper import agent_context

    for module in (llm_utils, agent_context, llm_agent):
        monkeypatch.setattr(module, "generate", module.generate)
