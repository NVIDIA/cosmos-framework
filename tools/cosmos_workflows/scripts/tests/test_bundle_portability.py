# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Exercise workflow helpers through the existing framework skill entrypoints."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT.parents[1]
NEW_SKILLS: set[str] = set()
EXTENDED = {"cosmos3-post-training", "cosmos3-inference", "cosmos3-setup", "cosmos3-env-troubleshoot"}


@pytest.fixture(scope="module")
def relocated(tmp_path_factory):
    checkout = tmp_path_factory.mktemp("relocated framework")
    destination = checkout / "tools" / "cosmos_workflows"
    shutil.copytree(
        ROOT, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache")
    )
    for discovery in (".agents", ".claude"):
        shutil.copytree(FRAMEWORK / discovery / "skills", checkout / discovery / "skills", symlinks=True)
    return destination


def run_helper(bundle: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    for name in ("COSMOS_WORKFLOWS_ROOT", "COSMOS_SKILLS_ROOT", "PYTHONPATH"):
        environment.pop(name, None)
    environment["COSMOS_SKILL_BANK_PATH"] = "/nonexistent/deprecated-skill-bank"
    return subprocess.run(
        [sys.executable, str(bundle / script), *args],
        cwd=bundle.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


@pytest.mark.parametrize(
    ("model", "backend", "expected"),
    [
        ("nvidia/Cosmos3-Nano", "cosmos-framework", "cosmos-framework"),
        ("nvidia/Cosmos3-Nano", "auto", "cosmos-framework"),
        ("nvidia/Cosmos3-Edge", "auto", "cosmos-framework"),
    ],
)
def test_relocated_resolver_routes_to_existing_skill(relocated, model, backend, expected):
    result = json.loads(
        run_helper(
            relocated,
            "scripts/resolve_cosmos_model.py",
            "--model",
            model,
            "--action",
            "train",
            "--backend",
            backend,
            "--format",
            "json",
        ).stdout
    )
    assert result["selected_backend"] == expected
    assert Path(result["skill_path"]) == relocated.parents[1] / ".agents/skills/cosmos3-post-training/SKILL.md"
    for key in ("skill_path", "guide_path", "metadata_path", "backend_contract_path"):
        assert Path(result[key]).is_file(), result
        assert Path(result[key]).is_relative_to(relocated.parents[1])


@pytest.mark.parametrize(
    ("action", "owner"),
    [
        ("train", "cosmos3-post-training"),
        ("evaluate", "cosmos3-post-training"),
        ("inference", "cosmos3-inference"),
        ("inference_microservice", "cosmos3-inference"),
    ],
)
def test_reasoner_action_ownership_preserves_framework_choice(relocated, action, owner):
    result = json.loads(
        run_helper(
            relocated,
            "scripts/resolve_cosmos_model.py",
            "--model",
            "nvidia/Cosmos3-Nano",
            "--action",
            action,
            "--backend",
            "cosmos-framework",
            "--format",
            "json",
        ).stdout
    )
    assert result["selected_backend"] == "cosmos-framework"
    assert Path(result["skill_path"]).parent.name == owner
    assert Path(result["skill_path"]).is_file()


@pytest.mark.parametrize(
    "helper",
    [
        "scripts/resolve_cosmos_image.py",
        "scripts/check_cosmos_launch_preflight.py",
        "models/cosmos3-reasoner/scripts/cosmos_workflow.py",
        "models/cosmos3-reasoner/scripts/evaluation_workflow.py",
        "models/cosmos3-reasoner/scripts/framework_checkpoint_action.py",
    ],
)
def test_helper_imports_without_runtime_or_old_skill_bank(relocated, helper):
    assert "usage:" in run_helper(relocated, helper, "--help").stdout.lower()


def test_env_setup_is_relocatable_and_preserves_working_directory(relocated):
    environment = os.environ.copy()
    environment.pop("COSMOS_WORKFLOWS_ROOT", None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1/env.sh"; printf "%s\\n%s\\n" "$COSMOS_WORKFLOWS_ROOT" "$PWD"',
            "bash",
            str(relocated),
        ],
        cwd=relocated.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [str(relocated), str(relocated.parent)]


def test_migration_adds_no_new_skill_entrypoints(relocated):
    skills = list(relocated.rglob("SKILL.md"))
    assert {p.parent.name for p in skills} == NEW_SKILLS
    for discovery in (".agents", ".claude"):
        links = (relocated.parents[1] / discovery / "skills").iterdir()
        assert {link.name for link in links if link.is_symlink()} == NEW_SKILLS
    for skill in skills:
        frontmatter = yaml.safe_load(skill.read_text().split("---", 2)[1])
        assert frontmatter["name"] == skill.parent.name
        for discovery in (".agents", ".claude"):
            link = relocated.parents[1] / discovery / "skills" / skill.parent.name
            assert link.is_symlink()
            assert not Path(os.readlink(link)).is_absolute()
            assert link.resolve() == skill.parent


def test_existing_skill_copies_share_the_integrated_references():
    for name in EXTENDED:
        first = FRAMEWORK / ".agents/skills" / name / "SKILL.md"
        second = FRAMEWORK / ".claude/skills" / name / "SKILL.md"
        assert first.is_file() and not first.is_symlink()
        first_parts = first.read_text().split("---", 2)
        second_parts = second.read_text().split("---", 2)
        assert yaml.safe_load(first_parts[1]) == yaml.safe_load(second_parts[1])
        assert second_parts[2].replace(".claude/skills/", ".agents/skills/") == first_parts[2]
        targets = [target for target in re.findall(r"\]\(([^)]+)\)", first.read_text()) if "cosmos_workflows" in target]
        assert targets, name
        for target in targets:
            assert (first.parent / target).is_file(), (name, target)


def test_provenance_records_packaged_files_and_existing_routes():
    manifest = json.loads((ROOT / "migration.json").read_text())
    assert len(manifest["source_commit"]) == 40
    destinations = [entry["destination"] for entry in manifest["files"]]
    assert len(destinations) == len(set(destinations))
    for entry in manifest["files"]:
        assert (ROOT / entry["destination"]).is_file(), entry
        assert len(entry["source_sha256"]) == 64
    for name in manifest["skill_routes"].values():
        assert (FRAMEWORK / ".agents/skills" / name / "SKILL.md").is_file(), name


def test_local_image_stamps_can_be_updated_without_registry_assumptions():
    import stamp_versions

    line = "container_image: cosmos-framework:local  # versions-key: images.containers.cosmos_framework"
    original_image = "cosmos-framework:local"  # unpinned: test fixture
    updated, ok = stamp_versions.replace_value(line, "registry.example.com/cosmos/framework:verified")
    assert ok
    assert updated == line.replace(original_image, "registry.example.com/cosmos/framework:verified")


def test_local_markdown_references_resolve_without_source_repository():
    for document in ROOT.rglob("*.md"):
        if any(part.startswith(".") for part in document.relative_to(ROOT).parts):
            continue
        for match in re.finditer(r"\]\(([^)]+)\)", document.read_text()):
            target = match.group(1).split("#", 1)[0]
            if not target or "://" in target or target.startswith(("/", "$", "<")):
                continue
            assert (document.parent / target).exists(), (document, target)
