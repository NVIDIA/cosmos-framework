# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 CPU tests for `utils/provenance.py`, the per-attempt run provenance stamp.

Pinned behaviors: the tree hash is a function of (relative path, content) only -- stable across calls
and locations, sensitive to one byte or one rename, blind to non-source files; the record carries every
documented key and a broken or absent git yields None, never an exception; the stamp writes one JSON
per attempt without overwriting an earlier one and keeps exactly one top-level `provenance` mapping in
`config_resolved.yaml` (rewritten by an atomic rename, against the REAL `LazyConfig.save_yaml` output of a
real GRPO root, changing nothing but the new key); the runtime tree walk has an outer subprocess timeout;
`dump_resolved` stamps only when asked, and only on rank 0.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from cosmos_framework.posttrain.utils import provenance
from cosmos_framework.posttrain.utils.provenance import (
    collect_provenance,
    git_commit,
    git_dirty,
    hash_tree,
    hash_tree_with_timeout,
    stamp_provenance,
)

RECORD_KEYS = {
    "attempt_ts",
    "timestamp",
    "hostname",
    "slurm_job_id",
    "python_version",
    "torch_version",
    "argv",
    "cwd",
    "env",
    "repo_root",
    "posttrain_root",
    "git_commit",
    "git_dirty",
    "git_dirty_scope",
    "recipe_variant",
    "tree_hash",
    "tree_files",
}


def _scratch_repo(repo: Path) -> Any:
    """``git init`` at ``repo`` with a throwaway identity; returns a ``git(*args) -> stdout`` runner."""
    repo.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "posttrain-test",
        "GIT_AUTHOR_EMAIL": "posttrain-test@example.invalid",
        "GIT_COMMITTER_NAME": "posttrain-test",
        "GIT_COMMITTER_EMAIL": "posttrain-test@example.invalid",
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
        ).stdout

    git("init", "-q")
    return git


def _write_source_tree(root: Path) -> None:
    (root / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "mod.py").write_text("x = 1\n")
    (root / "top.py").write_text("print('hi')\n")
    (root / "notes.md").write_text("not source\n")
    (root / "pkg" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "skipped.py").write_text("y = 2\n")


@pytest.mark.L0
@pytest.mark.CPU
def test_hash_tree_depends_only_on_source_paths_and_contents(tmp_path: Path) -> None:
    source = tmp_path / "a"
    _write_source_tree(source)

    first = hash_tree(source)
    assert first == hash_tree(source)
    assert first.files == 3
    assert len(first.sha256) == 64 and int(first.sha256, 16) >= 0

    relocated = tmp_path / "elsewhere" / "b"
    shutil.copytree(source, relocated)
    assert hash_tree(relocated) == first

    (source / "notes.md").write_text("edited, but not source\n")
    assert hash_tree(source) == first

    (source / "pkg" / "mod.py").write_text("x = 2\n")
    edited = hash_tree(source)
    assert edited.sha256 != first.sha256 and edited.files == 3

    (source / "top.py").rename(source / "renamed.py")
    renamed = hash_tree(source)
    assert renamed.sha256 not in {first.sha256, edited.sha256} and renamed.files == 3

    with pytest.raises(TimeoutError):
        hash_tree(source, deadline=time.monotonic() - 1.0)
    with pytest.raises(FileNotFoundError):
        hash_tree(tmp_path / "missing")


@pytest.mark.L0
@pytest.mark.CPU
def test_hash_tree_with_timeout_isolates_the_walk_and_enforces_an_outer_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_source_tree(source)
    assert hash_tree_with_timeout(source, timeout_s=5.0) == hash_tree(source)

    def _timeout(*args: Any, **kwargs: Any) -> None:
        assert kwargs["timeout"] == 0.25
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(provenance.subprocess, "run", _timeout)
    with pytest.raises(TimeoutError, match="exceeded 0.25s"):
        hash_tree_with_timeout(source, timeout_s=0.25)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("failure", ["no_repo", "no_git_binary", "timeout"])
def test_collect_provenance_records_every_key_and_survives_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = tmp_path / "posttrain"
    _write_source_tree(source)
    repo_root = tmp_path / "not-a-repo"
    repo_root.mkdir()
    if failure == "no_git_binary":
        monkeypatch.setattr(provenance, "_git", lambda *_args, **_kwargs: None)
    if failure == "timeout":

        def _timed_out(_repo_root: Any, args: list[str], timeout_s: float) -> str:
            raise provenance.GitProbeError(f"git {args[0]} exceeded its {timeout_s:g}s time box")

        monkeypatch.setattr(provenance, "_git", _timed_out)
    monkeypatch.setenv("I4_ROOT", "/trees/live")
    monkeypatch.setenv("PYTHONPATH", "/trees/live:/site")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")

    record = collect_provenance(repo_root=repo_root, posttrain_root=source, attempt_ts="20260914201500")

    assert RECORD_KEYS <= set(record)
    if failure == "timeout":
        # Review R134-08: a probe that had an answer and ran out of time records the reason; the canaries' silent
        # ``git_dirty: null`` is gone.
        assert set(record["errors"]) == {"git_commit", "git_dirty"}
        assert record["errors"]["git_commit"].startswith("GitProbeError: git rev-parse exceeded its ")
        assert record["errors"]["git_dirty"].startswith("GitProbeError: git status exceeded its ")
    else:
        assert "errors" not in record, record
    assert record["git_dirty_scope"] == "."  # the posttrain root is not inside this repo_root
    assert json.loads(json.dumps(record)) == record
    assert record["git_commit"] is None and record["git_dirty"] is None
    assert record["tree_hash"] == hash_tree(source).sha256 and record["tree_files"] == 3
    assert record["attempt_ts"] == "20260914201500"
    assert record["env"] == {"I4_ROOT": "/trees/live", "PYTHONPATH": "/trees/live:/site"}
    assert record["slurm_job_id"] == "4242"
    assert record["repo_root"] == str(repo_root) and record["posttrain_root"] == str(source)
    assert isinstance(record["hostname"], str) and record["hostname"]
    assert isinstance(record["argv"], list) and record["cwd"] == os.getcwd()
    assert record["python_version"].count(".") >= 1
    assert record["timestamp"].startswith("20") and record["timestamp"][10] == "T"


@pytest.mark.L0
@pytest.mark.CPU
def test_collect_provenance_isolates_a_failing_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One probe raising must cost exactly its field, recorded under `errors`, not the record."""

    def _broken_hostname() -> str:
        raise OSError("no resolver")

    monkeypatch.setattr(provenance.socket, "gethostname", _broken_hostname)
    record = collect_provenance(repo_root=tmp_path, posttrain_root=tmp_path / "missing", attempt_ts="1")

    assert record["hostname"] is None
    assert record["tree_hash"] is None and record["tree_files"] is None
    assert record["errors"] == {
        "hostname": "OSError: no resolver",
        "tree_hash": f"FileNotFoundError: tree hash root is not a directory: {tmp_path / 'missing'}",
    }
    assert json.loads(json.dumps(record)) == record


@pytest.mark.L0
@pytest.mark.CPU
def test_repo_root_is_the_checkout_containing_the_posttrain_tree() -> None:
    assert provenance.REPO_ROOT / "cosmos_framework" / "posttrain" == provenance.POSTTRAIN_ROOT


@pytest.mark.L0
@pytest.mark.CPU
def test_git_probes_report_the_checkout_state(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        # Without a git binary the probes must degrade to None (the same contract the record relies on).
        assert git_commit(tmp_path) is None and git_dirty(tmp_path) is None
        return
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "posttrain-test",
        "GIT_AUTHOR_EMAIL": "posttrain-test@example.invalid",
        "GIT_COMMITTER_NAME": "posttrain-test",
        "GIT_COMMITTER_EMAIL": "posttrain-test@example.invalid",
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
        ).stdout

    git("init", "-q")
    assert git_commit(repo) is None, "an unborn HEAD is a git failure, not a commit"
    (repo / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("commit", "-q", "-m", "init")
    head = git("rev-parse", "HEAD").strip()

    assert len(head) >= 40 and git_commit(repo) == head
    assert git_dirty(repo) is False
    (repo / "a.py").write_text("x = 2\n")
    assert git_dirty(repo) is True
    assert git_commit(tmp_path / "not-a-repo-at-all") is None


@pytest.mark.L0
@pytest.mark.CPU
def test_git_dirty_is_scoped_to_the_posttrain_tree_and_a_timeout_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R134-08: both canaries of 22bfce78f40 stamped ``git_dirty: null`` without a reason -- an unscoped
    ``git status`` of a 20,873-file checkout on a networked filesystem missed its 5 s box and ``_git`` swallowed the
    ``TimeoutExpired``. Now the probe is limited to the posttrain pathspec (the tree ``tree_hash`` covers; the record
    names it as ``git_dirty_scope``): a modified file elsewhere in the checkout does not flip it, a modified posttrain file
    does, an unborn HEAD still reads ``None`` without an error, and a time-boxed probe records its reason."""
    if shutil.which("git") is None:
        pytest.skip("git binary required")
    repo = tmp_path / "repo"
    posttrain = repo / "cosmos_framework" / "posttrain"
    posttrain.mkdir(parents=True)
    (posttrain / "mod.py").write_text("x = 1\n")
    (repo / "other.py").write_text("y = 1\n")
    git = _scratch_repo(repo)
    assert provenance.dirty_scope(repo, posttrain) == "cosmos_framework/posttrain"
    assert provenance.dirty_scope(repo, tmp_path / "elsewhere") is None and provenance.dirty_scope(repo, repo) is None
    unborn = collect_provenance(repo_root=repo, posttrain_root=posttrain, attempt_ts="1")
    # An unborn HEAD: rev-parse has nothing to answer (None, no error); status answers "no tracked file differs" (False).
    assert unborn["git_commit"] is None and unborn["git_dirty"] is False and "errors" not in unborn
    assert git_commit(tmp_path / "does-not-exist") is None  # a missing directory is a silent None too (the parent test)
    git("add", ".")
    git("commit", "-q", "-m", "init")
    head = git("rev-parse", "HEAD").strip()
    (repo / "other.py").write_text("y = 2\n")
    assert git_dirty(repo) is True and git_dirty(repo, paths=("cosmos_framework/posttrain",)) is False
    record = collect_provenance(repo_root=repo, posttrain_root=posttrain, attempt_ts="2")
    assert record["git_commit"] == head and record["git_dirty"] is False
    assert record["git_dirty_scope"] == "cosmos_framework/posttrain" and "errors" not in record
    (posttrain / "mod.py").write_text("x = 2\n")
    assert collect_provenance(repo_root=repo, posttrain_root=posttrain, attempt_ts="3")["git_dirty"] is True

    def _slow_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(provenance.subprocess, "run", _slow_run)
    with pytest.raises(provenance.GitProbeError, match="git status exceeded its 0.5s time box"):
        git_dirty(repo, timeout_s=0.5)
    with pytest.raises(provenance.GitProbeError, match="git rev-parse exceeded its 0.5s time box"):
        git_commit(repo, timeout_s=0.5)


@pytest.mark.L0
@pytest.mark.CPU
def test_stamp_provenance_writes_one_file_per_attempt_and_embeds_once(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    resolved = out / "config_resolved.yaml"
    base = {"job": {"name": "n"}, "rl": {"kl_beta": 0.01, "run_ts": "20260901120000", "resume": "off"}}
    resolved.write_text(yaml.safe_dump(base, default_flow_style=False))
    first = {"attempt_ts": "20260901120000", "git_commit": None, "tree_hash": "ab" * 32, "argv": ["x", "a=1"]}

    stamped = stamp_provenance(str(out), resolved_config=str(resolved), record=first)

    assert stamped == first
    first_path = out / "provenance.20260901120000.json"
    assert json.loads(first_path.read_text()) == first
    assert yaml.safe_load(resolved.read_text()) == {**base, "provenance": first}

    # A second attempt stamped in the same second keeps the first record and replaces the YAML key.
    second = {**first, "git_commit": "0" * 40}
    stamp_provenance(str(out), resolved_config=str(resolved), record=second)

    assert json.loads(first_path.read_text()) == first
    assert json.loads((out / "provenance.20260901120000-1.json").read_text()) == second
    text = resolved.read_text()
    assert yaml.safe_load(text) == {**base, "provenance": second}
    assert text.count("provenance:") == 1
    assert sorted(path.name for path in out.iterdir()) == [
        "config_resolved.yaml",
        "provenance.20260901120000-1.json",
        "provenance.20260901120000.json",
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_stamp_provenance_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = {"attempt_ts": "20260901120000", "tree_hash": None}

    blocker = tmp_path / "file"
    blocker.write_text("")
    assert stamp_provenance(str(blocker / "run"), record=record) is None

    out = tmp_path / "run"
    out.mkdir()
    not_a_mapping = out / "config_resolved.yaml"
    not_a_mapping.write_text("- just\n- a list\n")
    assert stamp_provenance(str(out), resolved_config=str(not_a_mapping), record=record) == record
    assert json.loads((out / "provenance.20260901120000.json").read_text()) == record
    assert not_a_mapping.read_text() == "- just\n- a list\n"

    def _broken_collect(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("collector exploded")

    monkeypatch.setattr(provenance, "collect_provenance", _broken_collect)
    assert stamp_provenance(str(out)) is None
    assert sorted(path.name for path in out.iterdir()) == ["config_resolved.yaml", "provenance.20260901120000.json"]


@pytest.mark.L0
@pytest.mark.CPU
def test_dump_resolved_stamps_provenance_only_on_request_and_rank0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.utils.entry import dump_resolved
    from cosmos_framework.utils.lazy_config import LazyConfig

    def _fake_save_yaml(config: Any, filename: str) -> str:
        Path(filename).write_text(yaml.safe_dump({"job": {"name": config.job.name}}, default_flow_style=False))
        return filename

    canned = {"attempt_ts": "20260901120000", "git_commit": "f" * 40, "tree_hash": "0" * 64}
    monkeypatch.setattr(LazyConfig, "save_yaml", _fake_save_yaml)
    monkeypatch.setattr(provenance, "collect_provenance", lambda **_kwargs: dict(canned))
    monkeypatch.setenv("RANK", "0")
    config = SimpleNamespace(job=SimpleNamespace(name="n"))

    plain = tmp_path / "plain"
    dump_resolved(config, str(plain))
    assert sorted(path.name for path in plain.iterdir()) == ["config_resolved.yaml"]
    assert "provenance" not in yaml.safe_load((plain / "config_resolved.yaml").read_text())

    stamped = tmp_path / "stamped"
    dump_resolved(config, str(stamped), provenance=True)
    assert sorted(path.name for path in stamped.iterdir()) == ["config_resolved.yaml", "provenance.20260901120000.json"]
    assert json.loads((stamped / "provenance.20260901120000.json").read_text()) == canned
    assert yaml.safe_load((stamped / "config_resolved.yaml").read_text()) == {
        "job": {"name": "n"},
        "provenance": canned,
    }

    monkeypatch.setenv("RANK", "1")
    other_rank = tmp_path / "rank1"
    dump_resolved(config, str(other_rank), provenance=True)
    assert not other_rank.exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_embed_provenance_round_trips_the_real_resolved_config_dump(tmp_path: Path) -> None:
    """Against the real dumper, not a stand-in: `LazyConfig.save_yaml` on a real GRPO root, then the embed. The
    round trip changes nothing but the new top-level key (every other key and scalar survives the safe_load /
    safe_dump pass), leaves no staging file behind, still loads through OmegaConf, and a re-stamp replaces the key."""
    from omegaconf import OmegaConf

    from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
    from cosmos_framework.posttrain.config.rl import GRPOConfig, RefConfig
    from cosmos_framework.posttrain.utils.provenance import embed_provenance
    from cosmos_framework.utils.lazy_config import LazyConfig

    root = PosttrainGRPOConfig(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
        rl=GRPOConfig(kl_beta=0.01, kl_coeff_clamp=5.0, ref=RefConfig(build_ref=True)),
    )
    output = tmp_path / "config"
    output.mkdir()
    resolved = output / "config_resolved.yaml"
    LazyConfig.save_yaml(root, str(resolved))
    before = yaml.safe_load(resolved.read_text())
    assert isinstance(before, dict) and "provenance" not in before
    assert before["rl"]["kl_beta"] == 0.01 and before["rl"]["kl_coeff_clamp"] == 5.0
    assert before["rl"]["ref"] == {"build_ref": True, "update_every": 0, "model_dir": None}

    record = {
        "attempt_ts": "20260901120000",
        "git_commit": "f" * 40,
        "tree_hash": "0" * 64,
        "argv": ["x", "rl.kl_beta=0.01"],
    }
    embed_provenance(str(resolved), record)
    after_text = resolved.read_text()
    assert yaml.safe_load(after_text) == {**before, "provenance": record}
    assert sorted(path.name for path in output.iterdir()) == ["config_resolved.yaml"], "no .tmp left behind"
    loaded = OmegaConf.load(str(resolved))
    assert OmegaConf.to_container(loaded, resolve=True) == {**before, "provenance": record}

    embed_provenance(str(resolved), {**record, "git_commit": "0" * 40})
    text = resolved.read_text()
    assert text.count("provenance:") == 1 and yaml.safe_load(text)["provenance"]["git_commit"] == "0" * 40
    assert yaml.safe_load(text)["rl"] == before["rl"]


@pytest.mark.L0
@pytest.mark.CPU
def test_collect_provenance_reads_the_relaunched_slurm_job_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review R125-D-04: the torchrun launcher unsets SLURM_JOB_ID (the rendezvous id) before the entry runs, so both
    arms recorded ``slurm_job_id: null``; the launcher now re-exports it as POSTTRAIN_SLURM_JOB_ID first and the record
    falls back to it. The parent read SLURM_JOB_ID only."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("POSTTRAIN_SLURM_JOB_ID", "747502")
    record = collect_provenance(repo_root=tmp_path, posttrain_root=tmp_path, budget_s=5.0)
    assert record["slurm_job_id"] == "747502"
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    assert collect_provenance(repo_root=tmp_path, posttrain_root=tmp_path, budget_s=5.0)["slurm_job_id"] == "1"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("POSTTRAIN_SLURM_JOB_ID", "")
    assert collect_provenance(repo_root=tmp_path, posttrain_root=tmp_path, budget_s=5.0)["slurm_job_id"] is None
    launcher = (Path(provenance.__file__).resolve().parent / "launch_multinode_torchrun.sh").read_text()
    export_line = next(i for i, line in enumerate(launcher.splitlines()) if "export POSTTRAIN_SLURM_JOB_ID=" in line)
    unset_line = next(
        i for i, line in enumerate(launcher.splitlines()) if line.strip().startswith("unset SLURM_JOB_ID")
    )
    assert export_line < unset_line, "the launcher must export the id before it unsets it"


@pytest.mark.L0
@pytest.mark.CPU
def test_collect_provenance_records_the_resolved_recipe_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R142-04 (FIXTURE: 4821e1d9943 had no registry and no key). A recipe variant selected by ``SFT_RECIPE_VARIANT``
    left no trace in the record (``ENV_KEYS`` is ``I4_ROOT`` / ``PYTHONPATH``) and the resolved config carries the
    variant's VALUES (``rl.mask_think_opener`` tells ``prefilled_opener`` from ``default`` today) but not its NAME, so a
    readout could not name the variant that produced them. A recipe's ``build_config``
    registers the RESOLVED name with ``record_recipe_variant``; the record carries it as ``recipe_variant`` (``None`` when
    no recipe resolved one in this process, as in a GRPO run); an empty name is refused and leaves the registry as it was."""
    monkeypatch.setattr(provenance._RecipeVariant, "name", None)
    assert provenance.recipe_variant() is None
    assert collect_provenance(repo_root=tmp_path, posttrain_root=tmp_path, budget_s=5.0)["recipe_variant"] is None
    provenance.record_recipe_variant("prefilled_opener")
    record = collect_provenance(repo_root=tmp_path, posttrain_root=tmp_path, budget_s=5.0)
    assert record["recipe_variant"] == "prefilled_opener" and "errors" not in record, record
    assert json.loads(json.dumps(record))["recipe_variant"] == "prefilled_opener"
    for bad in ("", None, 3):
        with pytest.raises(ValueError, match="recipe variant name"):
            provenance.record_recipe_variant(bad)  # type: ignore[arg-type]
    assert provenance.recipe_variant() == "prefilled_opener"
