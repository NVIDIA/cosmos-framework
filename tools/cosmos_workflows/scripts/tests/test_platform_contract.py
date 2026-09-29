# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-check platform guides against their scripts, templates, and metadata.

These static tests verify referenced paths, job tracking, and declared execution
contracts. They use no GPU, cluster, or credentials and do not prove live launch
compatibility.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLATFORM_DIR = REPO / "execution/platforms"

# Platforms that implement the four-verb execution contract. data staging (S3
# layer) and cosmos-setup-gpu-host (host preflight) launch nothing by design.
RUN_PLATFORMS = [
    "docker",
    "kubernetes",
    "slurm",
    "brev",
    "virtualenv",
]

VERB_PATTERNS = {
    "submit": re.compile(r"\bsubmit\b", re.I),
    "status": re.compile(r"\bstatus\b", re.I),
    "logs": re.compile(r"\blogs\b", re.I),
    "cancel": re.compile(r"\bcancel\b|\bteardown\b", re.I),
}


def _skill_text(name: str) -> str:
    return (PLATFORM_DIR / name / "guide.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("platform", RUN_PLATFORMS)
@pytest.mark.parametrize("verb", sorted(VERB_PATTERNS))
def test_platform_documents_every_verb(platform, verb):
    """Each run platform must document all four verbs of the execution contract."""
    assert VERB_PATTERNS[verb].search(_skill_text(platform)), (
        f"{platform}/guide.md never mentions the '{verb}' verb; the four-verb "
        f"contract requires submit/status/logs/cancel on every run platform"
    )


@pytest.mark.parametrize("platform", RUN_PLATFORMS)
def test_platform_wires_the_job_record(platform):
    """Every run platform must mint and update a job record, not just describe one.

    Brev regressed to prose ("open the record") with no command, which makes its
    jobs untrackable across invocations.
    """
    text = _skill_text(platform)
    assert "cosmos_job_record.py" in text, (
        f"{platform}/guide.md never invokes scripts/cosmos_job_record.py — jobs "
        f"launched by this platform cannot be tracked across invocations"
    )
    assert re.search(r"cosmos_job_record\.py[\"']?\s+open", text), (
        f"{platform}/guide.md never calls `cosmos_job_record.py open` to mint a job id"
    )
    assert re.search(r"cosmos_job_record\.py[\"']?\s+mark", text), (
        f"{platform}/guide.md never calls `cosmos_job_record.py mark` to update state"
    )


def test_brev_submit_contract_is_idempotent():
    """The normative Brev submit verb must guard against CLI command replay."""
    text = _skill_text("brev")
    execution = text.split("## Execution — the four verbs", 1)[1].split("### `brev exec` argument form", 1)[0]
    assert "docker inspect '$JOB_ID'" in execution
    assert "$JOB_ID already submitted" in execution


# Shell-assigned paths that point into the workflow bundle, e.g.
#   SETUP_SCRIPT="${COSMOS_WORKFLOWS_ROOT}/execution/platforms/.../setup.sh"
WORKFLOW_PATH_RE = re.compile(
    r"\$\{COSMOS_WORKFLOWS_ROOT(?::-[^}]*)?\}"
    r"(/[A-Za-z0-9_./-]+\.(?:sh|py|tmpl|yaml|json|md))"
)


@pytest.mark.parametrize(
    "skill_md",
    sorted(p for p in PLATFORM_DIR.rglob("*.md") if "__pycache__" not in p.parts),
    ids=lambda p: str(p.relative_to(PLATFORM_DIR)),
)
def test_workflow_relative_paths_resolve(skill_md):
    """A `$COSMOS_WORKFLOWS_ROOT/...`-anchored path in a platform skill must exist in the repo.

    Catches the dropped-path-segment class directly: the k8s preflight pointed at
    `${COSMOS_WORKFLOWS_ROOT}/platform/...` (missing `skills/`) and could never run.
    """
    missing = []
    for rel in WORKFLOW_PATH_RE.findall(skill_md.read_text(encoding="utf-8")):
        target = REPO / rel.lstrip("/")
        if not target.exists():
            missing.append(rel)
    assert not missing, (
        f"{skill_md.relative_to(REPO)} references workflow paths that do not exist: {sorted(set(missing))}"
    )


def test_slurm_enroot_conversion_uses_job_unique_node_local_temp():
    """Direct Enroot imports need the real Enroot variable, not only Pyxis' alias.

    A fixed ``/tmp/enroot-cosmos`` directory can be removed by cleanup from another
    allocation.  Enroot then fails during whiteout conversion with ``getcwd``
    and ``failed to resolve path`` errors after all image layers were fetched.
    """
    text = _skill_text("slurm")
    assert "ENROOT_TEMP_PATH=/tmp/enroot-cosmos-\\${SLURM_JOB_ID}" in text
    assert "SLURM_ENROOT_TEMP_PATH=\\${ENROOT_TEMP_PATH}" in text
    assert "--chdir=/tmp" in text


def test_slurm_sqsh_conversion_uses_explicit_site_partition_and_cpu_resources():
    """Conversion needs explicit site selection, separate from GPU resources."""
    text = _skill_text("slurm")
    info = (PLATFORM_DIR / "slurm" / "references" / "skill_info.yaml").read_text()
    assert "sqsh_conversion_partition: null" in info
    assert "polar" not in info
    assert "max_time_hours:" not in info
    assert "sqsh_conversion_timeout_minutes: 120" in info
    assert "sqsh_conversion_cpus_per_task: 4" in info
    assert "sqsh_conversion_memory_mb: 7200" in info
    assert "--mem=7200M" in text
    assert "QOSGrpMemLimit" in text


def test_slurm_consumes_model_action_lifecycle_without_private_renderers():
    text = _skill_text("slurm") + (PLATFORM_DIR / "slurm" / "references" / "slurm-container-execution.md").read_text(
        encoding="utf-8"
    )
    guardrails = (PLATFORM_DIR / "slurm" / "references" / "cosmos-slurm-guardrails.md").read_text(encoding="utf-8")
    for term in (
        "pre_commands",
        "post_commands",
        "supporting_files",
        "processes_per_node",
        "child_exit_code_path",
    ):
        assert term in text
    assert "model-specific retry launcher" in text
    assert "Cosmos-only SLURM" in guardrails
    assert "renderer" in guardrails


# Required flags of `cosmos_job_record.py open`, per its argparse definition. A
# documented invocation missing any of these fails at runtime with exit 2 —
# which is exactly how this test was born: a hand-written Brev example omitted
# --network-arch/--action/--storage-tier and looked entirely plausible.
JOB_RECORD_OPEN_REQUIRED = ("--platform", "--image", "--network-arch", "--action", "--storage-tier")

# `\\\n` must precede `[^\n]` in the alternation: otherwise `[^\n]` consumes the
# backslash and the continuation branch can never match, truncating the capture
# at the first line of a multi-line invocation.
OPEN_INVOCATION_RE = re.compile(r"cosmos_job_record\.py[\"']?\s+open\b((?:\\\n|[^\n])*)", re.M)


@pytest.mark.parametrize("platform", RUN_PLATFORMS)
def test_documented_job_record_open_has_required_flags(platform):
    """Every documented `cosmos_job_record.py open` must carry all required flags.

    Guards docs against the script's real argparse signature, so a plausible but
    incomplete example cannot ship.
    """
    text = _skill_text(platform)
    invocations = OPEN_INVOCATION_RE.findall(text)
    assert invocations, f"{platform}/guide.md documents no `cosmos_job_record.py open` call"
    for args in invocations:
        flat = args.replace("\\\n", " ")
        missing = [f for f in JOB_RECORD_OPEN_REQUIRED if f not in flat]
        assert not missing, (
            f"{platform}/guide.md: `cosmos_job_record.py open` example is missing "
            f"required flag(s) {missing} — it would exit 2 at runtime"
        )


def test_job_record_required_flags_match_the_script():
    """Keep the list above honest against cosmos_job_record.py itself."""
    src = (REPO / "scripts/cosmos_job_record.py").read_text(encoding="utf-8")
    open_block = src[src.index('"open"') :]
    for flag in JOB_RECORD_OPEN_REQUIRED:
        assert f'"{flag}"' in open_block, (
            f"{flag} is listed as required here but no longer appears in "
            f"cosmos_job_record.py's `open` parser — update JOB_RECORD_OPEN_REQUIRED"
        )


# Scripts the platform skills invoke DIRECTLY (as `"$COSMOS_WORKFLOWS_ROOT/scripts/x.py" ...`
# rather than `python3 .../x.py`) must carry the executable bit, or every
# documented submit sequence dies at step one with "permission denied".
DIRECT_EXEC_RE = re.compile(r'"?\$\{?COSMOS_WORKFLOWS_ROOT\}?/(scripts/[A-Za-z0-9_./-]+\.(?:py|sh))"?\s')


def test_directly_invoked_scripts_are_executable():
    """Any script a platform skill execs directly must be executable.

    This shipped broken: cosmos_job_record.py and redact_secrets.py were committed
    100644 while every other script in scripts/ was 100755, so the first line of
    `submit` failed with "permission denied" on all five platforms.

    Checked two ways, because neither alone is reliable everywhere:

    * The **filesystem** bit is authoritative for "would this actually run here",
      and a checkout materializes the committed mode, so it catches the bug in
      CI. This is the assertion that always runs.
    * The **git index** mode additionally catches a local `chmod +x` that was
      never committed — real, but only checkable where git works. CI containers
      often run as a different uid than the checkout owner, so git refuses with
      dubious-ownership, and a source export may have no `.git` at all. When git
      cannot answer we skip that half rather than failing: an earlier version
      asserted on it unconditionally and reported "absent from git" for a file
      that was present and correct.
    """
    referenced = set()
    for skill_md in PLATFORM_DIR.rglob("*.md"):
        if "__pycache__" in skill_md.parts:
            continue
        referenced.update(DIRECT_EXEC_RE.findall(skill_md.read_text(encoding="utf-8")))
    assert referenced, "no directly-invoked workflow scripts found — has the invocation style changed?"

    missing = sorted(r for r in referenced if not (REPO / r).is_file())
    assert not missing, f"platform skills reference scripts that do not exist: {missing}"

    not_executable = sorted(r for r in referenced if not os.access(REPO / r, os.X_OK))
    assert not not_executable, (
        f"referenced directly by a platform skill but not executable (chmod +x and commit the mode): {not_executable}"
    )

    # Second, weaker check — only where git can actually answer.
    proc = subprocess.run(
        ["git", "ls-files", "-s", "--", *sorted(referenced)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return  # no usable git index here; the filesystem assertions above stand
    recorded = {line.split("\t")[-1]: line.split()[0] for line in proc.stdout.splitlines() if "\t" in line}
    stale_mode = sorted(p for p, m in recorded.items() if m != "100755")
    assert not stale_mode, (
        f"executable on disk but not in the git index, so the bit will not "
        f"travel — run `git update-index --chmod=+x <path>`: {stale_mode}"
    )
