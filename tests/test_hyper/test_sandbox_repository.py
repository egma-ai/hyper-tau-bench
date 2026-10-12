"""Tests for the developer-visible sandbox Git baseline."""

import subprocess

from tau2.hyper.sandbox.orchestrator import (
    _classify_workspace_provenance,
    initialize_kit_repository,
    sanitize_kit_git_metadata,
)


def _git(kit_path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=kit_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_initialize_kit_repository_creates_one_visible_baseline(tmp_path):
    (tmp_path / "README.md").write_text("Visible instructions\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "agent.py").write_text("# starter\n")

    commit = initialize_kit_repository(tmp_path)

    assert commit == _git(tmp_path, "rev-parse", "HEAD")
    assert _git(tmp_path, "rev-list", "--count", "HEAD") == "1"
    assert _git(tmp_path, "status", "--porcelain") == ""
    assert _git(tmp_path, "branch", "--show-current") == "main"
    assert _git(tmp_path, "log", "-1", "--pretty=%s") == "Initial import"


def test_initialize_kit_repository_replaces_no_existing_history(tmp_path):
    """An explicit kit with history is rejected instead of overwritten."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)

    try:
        initialize_kit_repository(tmp_path)
    except ValueError as exc:
        assert "already contains Git metadata" in str(exc)
    else:
        raise AssertionError("expected an existing repository to be rejected")


def test_scoring_ignores_developer_git_configuration(tmp_path):
    """Developer-written Git config cannot run commands in the scorer."""
    kit = tmp_path / "kit"
    (kit / "workspace").mkdir(parents=True)
    (kit / "workspace" / "agent.py").write_text("# starter\n")
    baseline = initialize_kit_repository(kit)

    marker = tmp_path / "ran"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"#!/bin/sh\necho >> {marker}\nexit 1\n")
    hook.chmod(0o755)
    _git(kit, "config", "core.fsmonitor", str(hook))
    _git(kit, "config", "filter.planted.clean", f"echo >> {marker}; cat")
    (kit / "workspace" / ".gitattributes").write_text("* filter=planted\n")
    (kit / ".git" / "hooks" / "post-checkout").write_text("#!/bin/sh\n")
    (kit / "workspace" / "agent.py").write_text("# built\n")

    # Unsanitized, the provenance scan runs the planted commands.
    _classify_workspace_provenance(kit, baseline)
    assert marker.exists()
    marker.unlink()

    sanitize_kit_git_metadata(kit)
    provenance, deleted = _classify_workspace_provenance(kit, baseline)

    assert not marker.exists()
    assert provenance == {
        "workspace/agent.py": "modified",
        "workspace/.gitattributes": "new",
    }
    assert deleted == []
    assert not (kit / ".git" / "hooks").exists()
    assert _git(kit, "config", "user.name") == "Workspace"


def test_sanitize_drops_git_metadata_that_is_not_a_directory(tmp_path):
    (tmp_path / ".git").write_text("gitdir: /somewhere/else\n")
    sanitize_kit_git_metadata(tmp_path)
    assert not (tmp_path / ".git").exists()
    sanitize_kit_git_metadata(tmp_path)  # nothing to do
