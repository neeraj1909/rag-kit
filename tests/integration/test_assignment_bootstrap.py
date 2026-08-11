"""Behavioral proof for the safe assignment-template copier."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pytest

from ragkit.infrastructure import assignment as assignment_module
from ragkit.infrastructure import load_config

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "bootstrap_assignment.py"
TEMPLATES = ROOT / "examples" / "assignment_profiles"


def run_bootstrap(
    template: str, destination: Path, *flags: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--template",
            template,
            "--destination",
            str(destination),
            *flags,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("template", "vector_store", "generator"),
    (
        ("local-offline", "memory", "extractive"),
        ("hosted-persistent", "chroma", "openai"),
    ),
)
def test_assignment_templates_are_complete_and_config_valid(
    template: str, vector_store: str, generator: str
) -> None:
    root = TEMPLATES / template

    assert tuple(sorted(path.name for path in root.iterdir())) == ("ASSIGNMENT.md", "ragkit.toml")
    profile = load_config(root / "ragkit.toml")
    assert profile.components.vector_store == vector_store
    assert profile.components.generator == generator
    assert profile.source == "data"
    instructions = (root / "ASSIGNMENT.md").read_text(encoding="utf-8")
    assert "ragkit ask --config ragkit.toml --query" not in instructions


@pytest.mark.integration
def test_dry_run_is_deterministic_fast_and_has_no_effect(tmp_path: Path) -> None:
    destination = tmp_path / "new-assignment"

    started = perf_counter()
    first = run_bootstrap("local-offline", destination, "--dry-run")
    elapsed = perf_counter() - started
    second = run_bootstrap("local-offline", destination, "--dry-run")

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert json.loads(first.stdout) == {
        "destination": str(destination.resolve()),
        "files": ["ASSIGNMENT.md", "ragkit.toml"],
        "status": "dry-run",
        "template": "local-offline",
    }
    assert not destination.exists()
    assert elapsed < 30 * 60


@pytest.mark.integration
def test_copy_is_idempotent_and_generated_profile_loads(tmp_path: Path) -> None:
    destination = tmp_path / "assignment"

    created = run_bootstrap("local-offline", destination)
    unchanged = run_bootstrap("local-offline", destination)

    assert json.loads(created.stdout)["status"] == "created"
    assert json.loads(unchanged.stdout)["status"] == "unchanged"
    assert load_config(destination / "ragkit.toml").name == "assignment-local-offline"
    assert "configs/" not in (destination / "ASSIGNMENT.md").read_text(encoding="utf-8")


@pytest.mark.integration
def test_fresh_local_assignment_reaches_cited_answer_within_budget(tmp_path: Path) -> None:
    destination = tmp_path / "assignment"
    started = perf_counter()

    assert run_bootstrap("local-offline", destination).returncode == 0
    shutil.copytree(ROOT / "tests" / "fixtures" / "corpus", destination / "data")
    inspected = subprocess.run(
        [
            sys.executable,
            "-m",
            "ragkit.cli.main",
            "inspect-config",
            "--config",
            "ragkit.toml",
        ],
        cwd=destination,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    answered = subprocess.run(
        [
            sys.executable,
            "-m",
            "ragkit.cli.main",
            "ask",
            "--config",
            "ragkit.toml",
            "What is the fixture answer?",
        ],
        cwd=destination,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    elapsed = perf_counter() - started

    assert inspected.returncode == 0, inspected.stderr
    assert answered.returncode == 0, answered.stderr
    payload = json.loads(answered.stdout)
    assert "cobalt observatory" in payload["answer"].casefold()
    assert payload["citations"][0]["rank"] == 1
    assert payload["citations"][0]["evidence"][0]["source_uri"].endswith("/data/answer.txt")
    assert payload["citations"][0]["evidence"][0]["locator"]["kind"] == "text_span"
    assert elapsed < 30 * 60


@pytest.mark.integration
def test_collision_refuses_every_write_until_explicit_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "assignment"
    assert run_bootstrap("local-offline", destination).returncode == 0
    user_file = destination / "notes.txt"
    user_file.write_text("keep me", encoding="utf-8")
    config = destination / "ragkit.toml"
    config.write_text("user-owned config", encoding="utf-8")
    assignment_before = (destination / "ASSIGNMENT.md").read_bytes()

    refused = run_bootstrap("local-offline", destination)

    assert refused.returncode == 2
    assert "ragkit.toml" in refused.stderr
    assert config.read_text(encoding="utf-8") == "user-owned config"
    assert (destination / "ASSIGNMENT.md").read_bytes() == assignment_before

    overwritten = run_bootstrap("local-offline", destination, "--overwrite")
    assert json.loads(overwritten.stdout)["status"] == "overwritten"
    assert load_config(config).name == "assignment-local-offline"
    assert user_file.read_text(encoding="utf-8") == "keep me"


@pytest.mark.integration
def test_symlinks_and_broad_destinations_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)

    symlinked = run_bootstrap("local-offline", linked)
    broad = run_bootstrap("local-offline", Path("/"), "--dry-run")

    assert symlinked.returncode == 2
    assert "symlink" in symlinked.stderr
    assert broad.returncode == 2
    assert "unsafe destination" in broad.stderr


@pytest.mark.integration
def test_hardlinked_managed_file_is_rejected_without_external_mutation(tmp_path: Path) -> None:
    destination = tmp_path / "assignment"
    assert run_bootstrap("local-offline", destination).returncode == 0
    outside = tmp_path / "outside.txt"
    outside.write_text("outside user data", encoding="utf-8")
    managed = destination / "ragkit.toml"
    managed.unlink()
    managed.hardlink_to(outside)

    refused = run_bootstrap("local-offline", destination, "--overwrite")

    assert refused.returncode == 2
    assert "hardlink" in refused.stderr
    assert outside.read_text(encoding="utf-8") == "outside user data"


@pytest.mark.integration
def test_staging_failure_leaves_every_managed_file_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "assignment"
    assert run_bootstrap("local-offline", destination).returncode == 0
    before = {name: f"user {name}" for name in ("ASSIGNMENT.md", "ragkit.toml")}
    for name, content in before.items():
        (destination / name).write_text(content, encoding="utf-8")
    original = assignment_module._copyfile
    calls = 0

    def fail_second_copy(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")
        original(source, target)

    monkeypatch.setattr(assignment_module, "_copyfile", fail_second_copy)

    with pytest.raises(OSError, match="injected staging failure"):
        assignment_module.bootstrap_assignment(
            "local-offline",
            destination,
            template_root=TEMPLATES,
            repository_root=ROOT,
            overwrite=True,
        )

    assert {name: (destination / name).read_text(encoding="utf-8") for name in before} == before


@pytest.mark.integration
def test_mid_commit_failure_rolls_back_every_managed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "assignment"
    assert run_bootstrap("local-offline", destination).returncode == 0
    before = {name: f"user {name}" for name in ("ASSIGNMENT.md", "ragkit.toml")}
    for name, content in before.items():
        (destination / name).write_text(content, encoding="utf-8")
    original = assignment_module._replace
    calls = 0

    def fail_fourth_replace(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected commit failure")
        original(source, target)

    monkeypatch.setattr(assignment_module, "_replace", fail_fourth_replace)

    with pytest.raises(OSError, match="injected commit failure"):
        assignment_module.bootstrap_assignment(
            "local-offline",
            destination,
            template_root=TEMPLATES,
            repository_root=ROOT,
            overwrite=True,
        )

    assert {name: (destination / name).read_text(encoding="utf-8") for name in before} == before


@pytest.mark.integration
def test_cli_emits_typed_failure_for_unknown_template(tmp_path: Path) -> None:
    failed = run_bootstrap("not-a-template", tmp_path / "assignment")

    assert failed.returncode == 2
    assert "unknown assignment template" in failed.stderr
    assert "Traceback" not in failed.stderr
