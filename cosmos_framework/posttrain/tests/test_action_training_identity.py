# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Training-source refusal is distinct from model/artifact reload compatibility."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state
from cosmos_framework.posttrain.entrypoints.action_sft import _Source
from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.test_action_sft import Dataset, assert_state_equal
from cosmos_framework.posttrain.utils import action_training_identity as source

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def installed_sources(tmp_path, monkeypatch):
    root = tmp_path / "installed" / "posttrain"
    for relative in source.TRAINING_SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source._POSTTRAIN_ROOT / relative).read_bytes())
    monkeypatch.setattr(source, "_POSTTRAIN_ROOT", root)
    return root


def test_source_inventory_is_explicit_complete_and_installation_path_independent(installed_sources):
    original_rng = collect_rng_state()
    identity = source.action_training_source_identity()
    assert len(identity["files"]) == len(set(source.TRAINING_SOURCE_FILES))
    assert {"data/datasets/flow_sft.py", "data/flow_media.py"} <= identity["files"].keys()
    for relative, digest in identity["files"].items():
        assert not Path(relative).is_absolute()
        assert digest == hashlib.sha256((installed_sources / relative).read_bytes()).hexdigest()
    source.require_matching_training_source({"implementation": identity}, source.action_training_source_identity())
    assert_state_equal(collect_rng_state(), original_rng)
    # Unrelated docs and recipes do not invalidate the shared update implementation.
    (installed_sources / "not_training.md").write_text("changed documentation\n")
    assert source.action_training_source_identity() == identity


@pytest.mark.parametrize("relative", source.TRAINING_SOURCE_FILES)
def test_each_bound_source_change_refuses_resume_without_mutating_rng(installed_sources, relative):
    saved = {"implementation": source.action_training_source_identity()}
    with (installed_sources / relative).open("ab") as stream:
        stream.write(b"\n# simulated different installed implementation\n")
    before = collect_rng_state()
    current = source.action_training_source_identity()
    with pytest.raises(ValueError, match="original qualified source build") as failure:
        source.require_matching_training_source(saved, current)
    assert relative in str(failure.value)
    assert_state_equal(collect_rng_state(), before)


def test_missing_installed_dependency_cannot_start_a_new_training_contract(installed_sources):
    (installed_sources / "roles/trainer/finite.py").unlink()
    with pytest.raises(RuntimeError, match="roles/trainer/finite.py"):
        source.action_training_source_identity()


@pytest.mark.parametrize("saved", [None, {}, {"seed": 17}, {"implementation": {"format": "unknown"}}])
def test_legacy_resume_refuses_without_inventing_source_metadata(saved):
    with pytest.raises(ValueError, match="lacks the required training source contract"):
        source.require_matching_training_source(saved, source.action_training_source_identity())


@pytest.mark.parametrize("change", ["missing", "extra", "invalid_digest", "invalid_key", "extra_field"])
def test_incomplete_or_malformed_source_inventory_refuses(change):
    current = source.action_training_source_identity()
    saved = copy.deepcopy(current)
    if change == "missing":
        saved["files"].pop("roles/trainer/llm.py")
    elif change == "extra":
        saved["files"]["unrecognized.py"] = "a" * 64
    elif change == "invalid_digest":
        saved["files"]["roles/trainer/llm.py"] = "z" * 64
    elif change == "invalid_key":
        saved["files"][42] = "a" * 64
    else:
        saved["bypass"] = True
    with pytest.raises(ValueError, match="training (implementation differs|source contract is malformed)"):
        source.require_matching_training_source({"implementation": saved}, current)


def _trainer(contract):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(829)
        net = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.2), nn.Linear(3, 1))
    net._cosmos_adapter_parameters = dict(net.named_parameters())
    codec = AdapterCheckpointCodec(net, identity={"model": "tiny-native-action-fixture"})
    codec.bind_training_identity(contract)
    net._cosmos_adapter_checkpoint_codec = codec
    optimizer = torch.optim.AdamW(codec.parameters.values(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    def loss(model, batch):
        x = torch.tensor([[row.value, -row.value] for row in batch], dtype=torch.float32)
        return (model.net(x) - torch.rand(len(batch), 1)).square().mean(), {}

    trainer = LLMTrainer(
        SimpleNamespace(net=net),
        Objective(loss, reduction=Reduction.MEAN, identity={"task": "native-source-identity-fixture"}),
        optimizer,
        lr_scheduler=scheduler,
        checkpoint_manager=CheckpointManager(checkpoint_format="dcp", keep_optim_last=0),
        require_gradients=True,
        nonfinite_update="fail_stop",
    )
    data = _Source(Dataset(count=7))
    cursor = IndexedCursor(data, shuffle=True, seed=19, cursor_resume="strict")
    return trainer, codec, data, cursor


def _snapshot(trainer, cursor):
    return copy.deepcopy(
        {
            "model": trainer.model.net.state_dict(),
            "optimizer": trainer.opt.state_dict(),
            "scheduler": trainer.lr_scheduler.state_dict(),
            "trainer": trainer._trainer_state(),
            "cursor": cursor.state_dict(),
            "rng": collect_rng_state(),
        }
    )


def test_real_committed_checkpoint_refuses_each_changed_source_before_state_restore(tmp_path):
    """Real DCP save/load, AdamW and stochastic updates; no mocked resume or RNG."""
    from cosmos_framework.posttrain.utils.rank_state import collect_rank_state, state_for_current_rank

    original_rng = collect_rng_state()
    contract = {"format": "cosmos-native-action-sft-v1", "implementation": source.action_training_source_identity()}

    def update(trainer, data, cursor):
        assert trainer.train([data[cursor.next_index()]])["optim/update_outcome"] == "applied"

    try:
        trainer, codec, data, cursor = _trainer(contract)
        for _ in range(3):
            update(trainer, data, cursor)
        path = str(tmp_path / "iter_000003")
        trainer.save(path, data_state=collect_rank_state({"cursor": cursor.state_dict()}))
        trainer.finalize_checkpoints()
        metadata = read_adapter_metadata(path, require_resume=True, manager=trainer.checkpoint_manager)
        saved_metadata = copy.deepcopy(metadata)
        update(trainer, data, cursor)
        expected = _snapshot(trainer, cursor)

        for relative in source.TRAINING_SOURCE_FILES:
            changed = copy.deepcopy(contract)
            changed["implementation"]["files"][relative] = "0" * 64
            resumed, resumed_codec, _, resumed_cursor = _trainer(changed)
            before = _snapshot(resumed, resumed_cursor)
            # Early entrypoint admission reports the exact dependency. The codec is
            # a second guard even for callers using LLMTrainer.load directly.
            with pytest.raises(ValueError, match="training implementation differs"):
                source.require_matching_training_source(
                    metadata["identity"]["training_contract"], changed["implementation"]
                )
            with pytest.raises(ValueError, match="identity differs"):
                resumed.load(path)
            resumed_codec.validate_metadata(metadata, purpose="reload")
            assert_state_equal(_snapshot(resumed, resumed_cursor), before)
        assert metadata == saved_metadata

        # Current-build checkpoints genuinely restore the next stochastic update.
        resumed, resumed_codec, resumed_data, resumed_cursor = _trainer(contract)
        source.require_matching_training_source(metadata["identity"]["training_contract"], contract["implementation"])
        assert resumed.load(path) == 3
        restored = state_for_current_rank(resumed.checkpoint_manager.load_component(path, "data"))
        resumed_cursor.load_state_dict(restored["cursor"])
        update(resumed, resumed_data, resumed_cursor)
        assert_state_equal(_snapshot(resumed, resumed_cursor), expected)
    finally:
        restore_rng_state(original_rng)


def test_legacy_model_metadata_remains_reloadable_but_not_resumable():
    contract = {"implementation": source.action_training_source_identity()}
    trainer, codec, _, _ = _trainer(contract)
    metadata = codec.metadata(trainer.opt)
    metadata["identity"]["training_contract"] = {"seed": 17}
    codec.validate_metadata(metadata, purpose="reload")
    with pytest.raises(ValueError, match="identity differs"):
        codec.validate_metadata(metadata, trainer.opt, purpose="resume")


@pytest.mark.parametrize("relative", ("data/datasets/flow_sft.py", "data/flow_media.py"))
def test_changed_data_execution_with_same_content_refuses_before_native_preparation(
    tmp_path, installed_sources, monkeypatch, relative
):
    """Real content identities and committed metadata; only CUDA/launch plumbing is stubbed."""
    from PIL import Image

    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.data.datasets.flow_sft import FlowSFTDataset
    from cosmos_framework.posttrain.entrypoints import action_sft
    from cosmos_framework.posttrain.utils import determinism, entry, qlora_metrics
    from cosmos_framework.posttrain.utils.rank_state import collect_rank_state
    from cosmos_framework.utils.config import load_config

    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 0, 255))
    image_path = tmp_path / "camera.png"
    image.save(image_path)
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "id": "row0",
                "episode_id": "episode0",
                "observation": {},
                "actions": [[0.25]],
                "media": {"camera": "camera.png"},
            }
        )
        + "\n"
    )
    dataset = FlowSFTDataset(str(rows_path))
    contract = {
        "format": "cosmos-native-action-sft-v1",
        "implementation": source.action_training_source_identity(),
        "dataset_fingerprint": dataset.cursor_fingerprint(),
    }
    trainer, _, data, cursor = _trainer(contract)
    for _ in range(3):
        trainer.train([data[cursor.next_index()]])
    checkpoint = tmp_path / "iter_000003"
    trainer.save(str(checkpoint), data_state=collect_rank_state({"cursor": cursor.state_dict()}))
    trainer.finalize_checkpoints()
    checkpoint_bytes = {str(p.relative_to(checkpoint)): p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}

    path = installed_sources / relative
    original = path.read_text()
    if relative.endswith("flow_sft.py"):
        assert original.count("        return row\n") == 1
        modified = original.replace("        return row\n", "        row['actions'][0][0] += 1\n        return row\n")
    else:
        assert original.count("yield image") == 2
        modified = original.replace("yield image", "yield image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)")
    path.write_text(modified)
    spec = importlib.util.spec_from_file_location("changed_data_execution_fixture", path)
    changed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(changed)
    if relative.endswith("flow_sft.py"):
        altered = changed.FlowSFTDataset(str(rows_path))
        assert altered.cursor_fingerprint() == dataset.cursor_fingerprint()
        assert altered[0]["actions"] != dataset[0]["actions"]
    else:
        with changed.open_flow_image(str(image_path)) as altered:
            assert altered.getpixel((0, 0)) != image.getpixel((0, 0))
        assert FlowSFTDataset(str(rows_path)).cursor_fingerprint() == dataset.cursor_fingerprint()

    config_path = "cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py"
    config = load_config(config_path, ["--"])
    before = _snapshot(trainer, cursor)
    phases = []

    def forbidden(*_args, **_kwargs):
        pytest.fail("changed data source must refuse before preparation, model construction or state restoration")

    with monkeypatch.context() as patch:
        patch.setenv("LOCAL_RANK", "0")
        patch.setattr(torch.cuda, "is_available", lambda: True)
        patch.setattr(torch.cuda, "set_device", lambda _device: None)
        patch.setattr(torch.distributed, "init_process_group", lambda *_a, **_k: None)
        patch.setattr(determinism, "enable_determinism", lambda **_k: None)
        patch.setattr(entry, "out_dir", lambda _config: str(tmp_path / "resume"))
        patch.setattr(entry, "dump_resolved", lambda *_a, **_k: None)
        patch.setattr(
            entry,
            "pin_run_attempt",
            lambda *_a: SimpleNamespace(out_dir=str(tmp_path / "resume"), resume_checkpoint=str(checkpoint)),
        )
        patch.setattr(
            CheckpointLifecycle,
            "from_config",
            lambda *_a, **_k: SimpleNamespace(manager=trainer.checkpoint_manager, load_trainer_and_component=forbidden),
        )
        patch.setattr(action_sft, "prepare_sources", forbidden)
        patch.setattr(action_sft, "_factory", forbidden)
        patch.setattr(qlora_metrics, "record_qlora_phase", lambda name, **_k: phases.append(name))
        with pytest.raises(ValueError, match="training implementation differs") as failure:
            action_sft.run(config, config_path=config_path, opts=[])
    assert relative in str(failure.value)
    assert "original qualified source build" in str(failure.value)
    assert phases == []
    assert_state_equal(_snapshot(trainer, cursor), before)
    assert {
        str(p.relative_to(checkpoint)): p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()
    } == checkpoint_bytes


def test_prior_inventory_without_data_sources_requires_original_build_but_allows_reload():
    current = source.action_training_source_identity()
    saved = copy.deepcopy(current)
    for relative in ("data/datasets/flow_sft.py", "data/flow_media.py"):
        saved["files"].pop(relative)
    original = copy.deepcopy(saved)
    trainer, codec, _, _ = _trainer({"implementation": current})
    metadata = codec.metadata(trainer.opt)
    metadata["identity"]["training_contract"] = {"implementation": saved}
    with pytest.raises(ValueError, match="original qualified source build") as failure:
        source.require_matching_training_source(metadata["identity"]["training_contract"], current)
    assert all(relative in str(failure.value) for relative in ("data/datasets/flow_sft.py", "data/flow_media.py"))
    codec.validate_metadata(metadata, purpose="reload")
    assert metadata["identity"]["training_contract"]["implementation"] == original
