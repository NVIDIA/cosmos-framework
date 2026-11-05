# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contracts for the bidirectional teacher and explicit local QA data path."""

from types import SimpleNamespace

import pytest
from omegaconf.errors import InterpolationResolutionError

from cosmos_framework.data.generator.local_simulation.dataloader import (
    LocalSimulationDataLoader,
    LocalSimulationJointDataLoader,
)
from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset
from cosmos_framework.utils.config import load_config

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("transform_overrides", [{}, {"resolution": "720"}])
def test_bounded_stream_identity_detects_policy_changes_without_changing_legacy_identity(
    transform_overrides: dict[str, str],
) -> None:
    dataset = object.__new__(LocalSimulationDataset)
    dataset.manifest, dataset.records, dataset.seed = {}, [{}], 42
    dataset.transform_overrides, dataset.max_num_frames = transform_overrides, None
    loader = LocalSimulationDataLoader(dataset, rank=0, world_size=1)
    legacy = LocalSimulationJointDataLoader._child_identity(loader)
    assert "bounded_transfer_policy" not in legacy
    dataset.max_num_frames = 149
    bounded = LocalSimulationJointDataLoader._child_identity(loader)
    assert bounded["bounded_transfer_policy"] == {"max_num_frames": 149, "transform_overrides": transform_overrides}
    assert {k: v for k, v in bounded.items() if k != "bounded_transfer_policy"} == legacy

    dataset.max_num_frames = 150
    changed_frames = LocalSimulationJointDataLoader._child_identity(loader)
    assert changed_frames["bounded_transfer_policy"] == {
        "max_num_frames": 150,
        "transform_overrides": transform_overrides,
    }
    assert changed_frames != bounded

    dataset.transform_overrides = {"resolution": "480"}
    changed_transforms = LocalSimulationJointDataLoader._child_identity(loader)
    assert changed_transforms["transform_overrides"] == {"resolution": "480"}
    assert changed_transforms["bounded_transfer_policy"] == {
        "max_num_frames": 150,
        "transform_overrides": {"resolution": "480"},
    }
    assert changed_transforms != changed_frames


def _compose(monkeypatch: pytest.MonkeyPatch, local: bool = False):
    monkeypatch.setenv("COSMOS_SIM_TRANSFER_BASE_CHECKPOINT", "/checkpoints/base")
    monkeypatch.setenv("COSMOS_SIM_TRANSFER_LOCAL_BUNDLE", "/data/transfer")
    name = "sim_local_transfer_stage0_bidir" if local else "sim_transfer_stage0_bidir"
    root = "local.py" if local else "config.py"
    return load_config(f"cosmos_framework/configs/simulation/{root}", ["--", f"experiment={name}"])


def test_bidirectional_teacher_preserves_the_stage0_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _compose(monkeypatch)
    model = config.model.config
    assert model.video_temporal_causal is False
    assert model.causal_training_strategy == "none"
    assert model.joint_attn_implementation == "two_way"
    assert model.sound_gen and model.action_gen and model.vision_gen
    assert model.sound_dim == model.sound_tokenizer.io_channels == 64
    assert model.sound_latent_fps * model.sound_tokenizer.hop_size == model.sound_tokenizer.sample_rate
    assert model.max_num_tokens_after_packing == 74_000
    assert model.diffusion_expert_config.max_vae_latent_side_after_patchify == 52
    assert model.rectified_flow_training_config.loss_scale == 10
    assert dict(model.rectified_flow_training_config.shift) == {"256": 3, "480": 5, "720": 10}
    assert model.rectified_flow_training_config.train_time_video_distribution == "waver"
    assert model.parallelism.data_parallel_shard_degree == 8
    assert model.parallelism.context_parallel_shard_degree == 1
    assert model.parallelism.fsdp_reduce_dtype == "bfloat16"
    assert model.activation_checkpointing.mode == "selective"
    assert model.compile.enabled and model.compile.compile_dynamic
    assert not model.compile.use_cuda_graphs
    assert model.ema.enabled and model.ema.rate == 0.1 and model.ema.iteration_shift == 0
    assert config.checkpoint.load_path == "/checkpoints/base"
    assert not config.checkpoint.load_ema_to_reg and not config.checkpoint.load_training_state
    assert config.checkpoint.strict_resume and config.checkpoint.dcp_async_mode_enabled
    assert config.checkpoint.save_iter == 250
    assert config.optimizer.lr == 1e-4 and config.optimizer.weight_decay == 0.05
    assert list(config.optimizer.betas) == [0.9, 0.99]
    assert len(config.optimizer.keys_to_select) == 10
    assert dict(config.optimizer.lr_multipliers) == {"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0}
    assert list(config.scheduler.f_max) == [0.4] and list(config.scheduler.f_min) == [0.0]
    assert list(config.scheduler.f_start) == [1e-6]
    assert list(config.scheduler.cycle_lengths) == [100_000]
    assert config.trainer.max_iter == 150_000 and config.trainer.seed == 0
    assert config.trainer.grad_accum_iter == 1
    assert config.trainer.callbacks.wall_clock_checkpoint.keep_last == 3


def test_stage0_satisfies_native_tokenizer_initialization_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import cosmos_framework.model.generator.omni_mot_model as model_module

    config = _compose(monkeypatch).model.config
    tokenizer = SimpleNamespace(eos_token_id=1)
    tokenizers = iter(
        [
            SimpleNamespace(tokenizer=tokenizer),
            SimpleNamespace(latent_ch=48),
            SimpleNamespace(latent_ch=64, sample_rate=48_000, temporal_compression_factor=1920),
        ]
    )
    monkeypatch.setattr(model_module, "lazy_instantiate", lambda _: next(tokenizers))
    monkeypatch.setattr(model_module, "add_special_tokens", lambda value: (value, {}))
    holder = SimpleNamespace(config=config)
    model_module.OmniMoTModel.set_up_tokenizers(holder)
    assert holder.tokenizer_sound_gen.latent_ch == config.sound_dim


def test_base_checkpoint_is_required_not_an_internal_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COSMOS_SIM_TRANSFER_BASE_CHECKPOINT", raising=False)
    with pytest.raises(InterpolationResolutionError, match="COSMOS_SIM_TRANSFER_BASE_CHECKPOINT"):
        load_config("cosmos_framework/configs/simulation/config.py", ["--", "experiment=sim_transfer_stage0_bidir"])


def test_local_stage0_preserves_both_resolution_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _compose(monkeypatch, local=True)
    loader = config.dataloader_train
    assert loader.max_sequence_length == 74_000
    assert "local_simulation_resume" in config.trainer.callbacks
    for resolution, batch, ratio, frames in (("480", 2, 4, 201), ("720", 1, 2, 149)):
        stream = loader.dataloaders[f"video_transfer_4modality_data_{resolution}"]
        assert stream.ratio == ratio and stream.dataloader.batch_size == batch
        dataset = stream.dataloader.dataset
        assert dataset.bundle_path == "/data/transfer"
        assert dataset.stratum is None
        assert dataset.max_num_frames == frames
        assert dataset.transform_overrides.resolution == resolution
        assert dataset.transform_overrides.cfg_dropout_rate == 0.1
        assert dataset.transform_overrides.resize_on_read is False
        assert dataset.transform_overrides.min_num_frames == 5
