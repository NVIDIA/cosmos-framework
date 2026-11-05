# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real source bytes, source loading and complete PPO checkpoint reconstruction."""

from __future__ import annotations

import gc
import json
import shutil
import weakref
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
from cosmos_framework.posttrain.control.checkpointing import save_policy_checkpoint
from cosmos_framework.posttrain.model.gr00t.source import load_ppo_source, resolve_ppo_source
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.gr00t_source import SourceGaussianPolicy, load_backbone, write_source
from cosmos_framework.posttrain.tests.helpers.ppo_continuous import LocalWindowGateway, collect_scored, make_collector

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


def _change_weight(directory: Path) -> None:
    path = directory / (
        "model.safetensors" if (directory / "model.safetensors").exists() else "model-00001-of-00002.safetensors"
    )
    size = path.stat().st_size
    tensors = load_file(str(path))
    tensors["weight"].add_(1)
    save_file(tensors, str(path))
    assert path.stat().st_size == size


@pytest.mark.parametrize(
    "resource", ["statistics.json", "processor_config.json", "tokenizer.json", "embodiment_id.json"]
)
def test_source_identity_binds_processor_inputs(tmp_path: Path, resource: str) -> None:
    directory = write_source(tmp_path / "source")
    original = resolve_ppo_source(str(directory)).identity()
    base = directory.parent / "processor-source" if resource == "tokenizer.json" else directory
    path = base / resource
    value = json.loads(path.read_text())
    value["source_marker"] = "changed"
    path.write_text(json.dumps(value))
    assert resolve_ppo_source(str(directory)).identity() != original


def test_source_follows_hf_blob_symlinks(tmp_path: Path) -> None:
    directory = write_source(tmp_path / "snapshot" / "source")
    original = resolve_ppo_source(str(directory)).identity()
    path = directory / "model-00001-of-00002.safetensors"
    blob = tmp_path / "blob"
    path.rename(blob)
    path.symlink_to(blob)
    assert resolve_ppo_source(str(directory)).identity() == original
    _change_weight(directory)
    assert resolve_ppo_source(str(directory)).identity() != original


def test_single_file_precedes_an_unused_sharded_inventory(tmp_path: Path) -> None:
    directory = write_source(tmp_path / "source")
    backbone = load_backbone(resolve_ppo_source(str(directory)))
    save_file(backbone.state_dict(), str(directory / "model.safetensors"))
    original = resolve_ppo_source(str(directory)).identity()
    # The production HF loader selects the single file, so the stale index is unused.
    (directory / "model.safetensors.index.json").write_text("invalid unused index")
    assert resolve_ppo_source(str(directory)).identity() == original
    _change_weight(directory)
    assert resolve_ppo_source(str(directory)).identity() != original


def test_configuration_change_after_resolution_is_rejected(tmp_path: Path) -> None:
    directory = write_source(tmp_path / "source")
    source = resolve_ppo_source(str(directory))
    path = directory / "processor_config.json"
    config = json.loads(path.read_text())
    config["processor_kwargs"]["model_name"] = "different/source"
    path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="configuration changed"):
        source.identity()


@pytest.mark.parametrize("corruption", ["missing", "truncated", "wrong_keys", "escaping"])
def test_invalid_source_inventory_precedes_loading(tmp_path: Path, corruption: str) -> None:
    directory = write_source(tmp_path / "source")
    path = directory / "model-00002-of-00002.safetensors"
    if corruption == "missing":
        path.rename(path.with_suffix(".absent"))
    elif corruption == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif corruption == "wrong_keys":
        save_file({"other": torch.ones(2)}, str(path))
    else:
        index = directory / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": "../escape.safetensors"}}))
    loaded = []

    def load(source):
        loaded.append(source)
        return load_backbone(source)

    with pytest.raises((ValueError, FileNotFoundError)):
        load_ppo_source(make_model_config(str(directory)), load)
    assert not loaded


def test_source_change_during_loading_releases_model_with_retained_error(tmp_path: Path) -> None:
    directory = write_source(tmp_path / "source")
    references = []

    def load(source):
        model = load_backbone(source)
        references.append(weakref.ref(model))
        _change_weight(directory)
        return model

    with pytest.raises(RuntimeError, match="changed while it was being loaded") as retained:
        load_ppo_source(make_model_config(str(directory)), load)
    assert retained.value.__traceback__ is not None
    gc.collect()
    assert references and all(reference() is None for reference in references)


def test_partial_loader_failure_releases_resources_in_retained_exception_chain(tmp_path: Path) -> None:
    directory = write_source(tmp_path / "source")
    references = []

    def fail(model):
        references.append(weakref.ref(model))
        raise ValueError("invalid processor")

    def load(source):
        try:
            fail(load_backbone(source))
        except ValueError as error:
            raise RuntimeError("processor construction failed") from error

    with pytest.raises(RuntimeError, match="processor construction failed") as retained:
        load_ppo_source(make_model_config(str(directory)), load)
    assert isinstance(retained.value.__cause__, ValueError)
    assert retained.value.__cause__.__traceback__ is not None
    gc.collect()
    assert references and all(reference() is None for reference in references)


def test_public_checkpoint_load_rejects_changed_source_before_state_mutation(tmp_path: Path) -> None:
    directory = write_source(tmp_path / "source")
    recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
    model = SourceGaussianPolicy(directory)
    learner = JointPPOTrainer(model, recipe)
    collector = make_collector(model, LocalWindowGateway(), recipe, 0, 2)
    receipt = learner.prepare_ppo_targets(collect_scored(collector, 0))
    learner.train_batch(receipt["batch_id"])
    learner.clear_batch(receipt["batch_id"])
    learner.set_publication_version(1)
    checkpoint = str(tmp_path / "checkpoint")
    save_policy_checkpoint(learner, checkpoint, data=collector)
    unchanged = JointPPOTrainer(SourceGaussianPolicy(directory), recipe)
    unchanged.load(checkpoint)
    assert unchanged.progress() == learner.progress()
    _change_weight(directory)
    changed = JointPPOTrainer(SourceGaussianPolicy(directory), recipe)
    before = changed.evaluation_state()
    with pytest.raises(RuntimeError, match="identity differs"):
        changed.load(checkpoint)
    assert changed.evaluation_state() == before


def _rank_source_lifecycle(rank: int, root: str, scenarios: tuple[str, ...]) -> None:
    import torch.distributed as dist

    directory = Path(root)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory / 'rendezvous'}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    reports = []
    try:
        for scenario in scenarios:
            case = directory / scenario
            references = []
            loaded = retained = None
            publish_called = False

            def load(source):
                model = load_backbone(source)
                references.append(weakref.ref(model))
                if rank == 1 and scenario == "load":
                    raise ValueError("rank-local loading fault")
                if rank == 1 and scenario == "source":
                    _change_weight(source.checkpoint)
                return model

            def publish(model):
                nonlocal publish_called
                publish_called = True
                save_file(model.state_dict(), str(case / "published.safetensors"))

            try:
                loaded, _ = load_ppo_source(make_model_config(str(case / f"rank-{rank}")), load, publish=publish)
            except Exception as error:
                retained = error
            finally:
                loaded = None
            gc.collect()
            reports.append(
                {
                    "scenario": scenario,
                    "error": None if retained is None else str(retained),
                    "publish_called": publish_called,
                    "models_constructed": len(references),
                    "models_released": all(reference() is None for reference in references),
                }
            )
        (directory / f"report-{rank}.json").write_text(json.dumps(reports))
    finally:
        dist.destroy_process_group()


def test_source_failures_converge_release_models_and_recover(tmp_path: Path) -> None:
    # One process group exercises the full failure lifecycle and a successful subsequent load.
    scenarios = ("missing", "different", "load", "source", "publish", "success")
    for scenario in scenarios:
        case = tmp_path / scenario
        original = write_source(case / "rank-0")
        other = case / "rank-1"
        shutil.copytree(original, other)
        if scenario == "different":
            _change_weight(other)
        elif scenario == "missing":
            path = other / "model-00002-of-00002.safetensors"
            path.rename(path.with_suffix(".absent"))
        elif scenario == "publish":
            (case / "published.safetensors").mkdir()  # A genuine rank-zero publication IO failure.
    torch.multiprocessing.spawn(_rank_source_lifecycle, args=(str(tmp_path), scenarios), nprocs=2, join=True)
    reports = [json.loads((tmp_path / f"report-{rank}.json").read_text()) for rank in range(2)]
    assert all(tuple(row["scenario"] for row in report) == scenarios for report in reports)
    for index, scenario in enumerate(scenarios):
        rows = [report[index] for report in reports]
        expected_models = 0 if scenario in ("missing", "different") else 1
        assert all(row["models_constructed"] == expected_models and row["models_released"] for row in rows)
        assert [row["publish_called"] for row in rows] == [scenario in ("publish", "success"), False]
        destination = tmp_path / scenario / "published.safetensors"
        if scenario == "success":
            assert all(row["error"] is None for row in rows)
            torch.testing.assert_close(load_file(str(destination))["weight"], torch.eye(2))
        else:
            expected = {
                "missing": "failed on one or more ranks",
                "different": "differs across ranks",
                "load": "rank-local loading fault",
                "source": "differs across ranks",
                "publish": "configuration publication",
            }[scenario]
            assert all(expected in row["error"] for row in rows)
            assert not destination.is_file()
