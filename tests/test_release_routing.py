"""Exercise publication guards and the locked release tool without remote effects."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomlkit
import yaml
from semantic_release.cli.config import RawConfig, RuntimeContext
from semantic_release.errors import NotAReleaseBranch
from tomlkit.items import Table

ROOT = Path(__file__).parents[1]


def _workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def _allows(expression, event, ref, message="feat: example"):
    # Interpret only the boolean/string subset used by these actual job guards.
    # A syntax change outside that subset fails instead of silently passing.
    expression = expression.replace("&&", " and ").replace("||", " or ").replace("!contains", "not contains")
    # GitHub string equality and contains() are case-insensitive. Canonical-ref
    # shell checks below must provide the additional case-sensitive boundary.
    github = SimpleNamespace(
        ref=ref.casefold(),
        event_name=event.casefold(),
        event=SimpleNamespace(head_commit=None if message is None else SimpleNamespace(message=message.casefold())),
    )
    return bool(
        eval(expression, {"__builtins__": {}, "contains": lambda value, part: part in value}, {"github": github})
    )


def _canonical_ref_allowed(steps, ref, tmp_path):
    guard = next(step for step in steps if step.get("name") == "Verify canonical publication branch")
    return (
        subprocess.run(
            ["bash", "-e", "-c", guard["run"]],
            cwd=tmp_path,
            env={**os.environ, "GITHUB_REF": ref},
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


@pytest.mark.parametrize(
    "event,ref,message,allowed",
    [
        ("push", "refs/heads/main", "fix: preserve legacy replay", True),
        ("workflow_dispatch", "refs/heads/main", None, True),
        ("push", "refs/heads/main", "chore(release): 5.0.0", False),
        ("push", "refs/heads/main", None, False),
        ("pull_request", "refs/heads/main", "feat: example", False),
        ("push", "refs/heads/release/v5", "feat!: example", False),
        ("workflow_dispatch", "refs/heads/release/v5", None, False),
        ("workflow_dispatch", "refs/heads/main-preview", None, False),
        ("workflow_dispatch", "refs/heads/Main", None, False),
        ("workflow_dispatch", "refs/tags/v5.0.0", None, False),
    ],
)
def test_only_main_can_publish_even_with_manual_dispatch(tmp_path, event, ref, message, allowed):
    workflow = _workflow("release.yml")
    assert workflow[True]["push"]["branches"] == ["main"]
    job = workflow["jobs"]["release"]
    assert job["steps"][0]["name"] == "Verify canonical publication branch"
    assert (_allows(job["if"], event, ref, message) and _canonical_ref_allowed(job["steps"], ref, tmp_path)) == allowed


@pytest.mark.parametrize("event", ["push", "workflow_dispatch", "pull_request"])
@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/Main", "refs/heads/release/v5", "refs/tags/v5.0.0"])
def test_integration_documentation_builds_without_deploying(tmp_path, event, ref):
    workflow = _workflow("docs.yml")
    assert "release/v5" in workflow[True]["push"]["branches"]
    upload = next(step for step in workflow["jobs"]["build-docs"]["steps"] if "upload-pages" in step.get("uses", ""))
    expected = ref == "refs/heads/main" and event in {"push", "workflow_dispatch"}
    for job, expression in (
        (workflow["jobs"]["build-docs"], upload["if"]),
        (workflow["jobs"]["deploy-docs"], workflow["jobs"]["deploy-docs"]["if"]),
    ):
        assert (_allows(expression, event, ref) and _canonical_ref_allowed(job["steps"], ref, tmp_path)) == expected
    steps = workflow["jobs"]["build-docs"]["steps"]
    guard = next(step for step in steps if step.get("name") == "Verify canonical publication branch")
    assert steps.index(guard) < steps.index(upload)
    assert guard["if"] == upload["if"]
    assert workflow["jobs"]["deploy-docs"]["steps"][0]["name"] == "Verify canonical publication branch"


def test_integration_pr_checks_are_not_restricted_to_main():
    for name in ("codecov.yml", "pylint.yml", "code-cost.yml", "docs.yml", "llms-txt.yml", "pr-title-validation.yml"):
        events = _workflow(name)[True]
        assert "pull_request" in events
        assert not (events["pull_request"] or {}).get("branches")
        assert not (events["pull_request"] or {}).get("branches-ignore")
    assert "release/v5" in _workflow("security-audit.yml")[True]["push"]["branches"]
    install = next(
        step["run"] for step in _workflow("codecov.yml")["jobs"]["test"]["steps"] if "uv sync" in step.get("run", "")
    )
    assert "--extra release" in install


def test_pr_tests_have_a_distinct_required_check_context():
    workflow = _workflow("codecov.yml")
    assert {"push", "pull_request"} <= workflow[True].keys()
    job = workflow["jobs"]["test"]
    assert "if" not in job
    contexts = {event: job["name"].replace("${{ github.event_name }}", event) for event in ("push", "pull_request")}
    assert contexts["push"] != contexts["pull_request"]
    assert "pull_request" in contexts["pull_request"]
    # The default PR checkout tests the merge result, rather than replacing it
    # with the head ref and making the two differently named checks equivalent.
    checkout = next(step for step in job["steps"] if step.get("name") == "Checkout")
    assert "ref" not in checkout.get("with", {})


def _project_metadata_with_version(project_text, version):
    """Update only the structured project version, retaining the release configuration."""
    metadata = tomlkit.parse(project_text)
    project = metadata["project"]
    assert isinstance(project, Table)
    project["version"] = version
    return tomlkit.dumps(metadata)


def _make_release_repo(tmp_path, project_text):
    """Freeze synthetic history independently of the checkout's current version."""
    (tmp_path / "pyproject.toml").write_text(_project_metadata_with_version(project_text, "4.20.0"))
    git = ["git", "-c", "core.hooksPath=/dev/null"]
    commands = [
        ["init", "--initial-branch=main"],
        ["config", "user.name", "COLEMAN release-routing fixture"],
        ["config", "user.email", "release-fixture@example.invalid"],
        ["remote", "add", "origin", "https://github.com/example/coleman-release-fixture.git"],
        ["add", "pyproject.toml"],
        ["commit", "-m", "chore(release): 4.20.0"],
        ["tag", "v4.20.0"],
    ]
    for command in commands:
        subprocess.run([*git, *command], cwd=tmp_path, check=True, capture_output=True)
    # A separate repo has no shared refs with the checkout. This hook is an
    # additional fail-closed defense; previews use --noop/--print/no-push too.
    hook = tmp_path / ".git/hooks/pre-push"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    return tmp_path


@pytest.fixture
def release_repo(tmp_path):
    """Use the real project release configuration with a fixed historical version."""
    return _make_release_repo(tmp_path, (ROOT / "pyproject.toml").read_text())


def _git(directory, *args):
    return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=directory, text=True).strip()


def _preview(directory, *args):
    """Run installed PSR without credentials, builds, commits, tags or publication."""
    before = (_git(directory, "show-ref"), _git(directory, "status", "--porcelain"))
    original = (directory / "pyproject.toml").read_bytes()
    environment = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "TMPDIR") if key in os.environ}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_release",
            "--noop",
            "version",
            "--print",
            "--no-push",
            "--no-vcs-release",
            "--skip-build",
            *args,
        ],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert before == (_git(directory, "show-ref"), _git(directory, "status", "--porcelain"))
    assert original == (directory / "pyproject.toml").read_bytes()
    return result.stdout.strip()


@pytest.mark.parametrize("checkout_version", ["4.20.0", "4.20.1", "5.0.0"])
def test_real_release_tool_keeps_v4_maintenance_separate_and_promotes_once(tmp_path, checkout_version):
    checkout_text = _project_metadata_with_version((ROOT / "pyproject.toml").read_text(), checkout_version)
    checkout = tomllib.loads(checkout_text)
    release_repo = _make_release_repo(tmp_path, checkout_text)
    baseline = tomllib.loads((release_repo / "pyproject.toml").read_text())
    assert baseline["project"]["version"] == "4.20.0"
    assert baseline["tool"]["semantic_release"] == checkout["tool"]["semantic_release"]
    _git(release_repo, "switch", "-c", "release/v5")
    _git(release_repo, "commit", "--allow-empty", "-m", "feat!: introduce generic decision contracts")
    assert _preview(release_repo) == "5.0.0-alpha.1"
    _git(release_repo, "switch", "main")
    _git(release_repo, "commit", "--allow-empty", "-m", "fix: preserve native CI checkpoint recovery")
    assert _preview(release_repo) == "4.20.1"
    metadata = release_repo / "pyproject.toml"
    metadata.write_text(_project_metadata_with_version(metadata.read_text(), "4.20.1"))
    _git(release_repo, "add", "pyproject.toml")
    _git(release_repo, "commit", "-m", "chore(release): 4.20.1")
    _git(release_repo, "tag", "v4.20.1")
    _git(release_repo, "switch", "release/v5")
    _git(release_repo, "merge", "--no-ff", "main", "-m", "chore: synchronize stable v4 maintenance")
    assert tomllib.loads(metadata.read_text())["project"]["version"] == "4.20.1"
    assert _preview(release_repo) == "5.0.0-alpha.1"
    _git(release_repo, "switch", "main")
    _git(release_repo, "merge", "--no-ff", "release/v5", "-m", "feat!: integrate qualified v5 contracts")
    assert _preview(release_repo) == "5.0.0"


@pytest.mark.parametrize("token", ["alpha", "beta", "rc"])
def test_real_prerelease_rehearsals_are_local_and_explicit(release_repo, token):
    _git(release_repo, "switch", "-c", "release/v5")
    _git(release_repo, "commit", "--allow-empty", "-m", "feat!: qualify generic contracts")
    assert _preview(release_repo, "--major", "--as-prerelease", "--prerelease-token", token) == f"5.0.0-{token}.1"


@pytest.mark.parametrize(
    "branch", ["Main", "main-preview", "master", "release/v50", "release/v5-fix", "feat/v5-355-contracts"]
)
def test_release_groups_reject_unrecognized_branches(branch):
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["semantic_release"]
    branches = RawConfig.model_validate(config).branches
    with pytest.raises(NotAReleaseBranch):
        RuntimeContext.select_branch_options(branches, branch)
