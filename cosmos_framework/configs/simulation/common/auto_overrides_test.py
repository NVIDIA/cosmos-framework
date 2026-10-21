# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
from cosmos_framework.configs.simulation.common.auto_overrides import framework_auto_overrides


def test_transform_values_and_checkpoint_identity_are_preserved():
    original = [
        "dataloader_train.dataloaders.cam_lidar_joint_480p_separate_view_text.dataloader.dataset.num_video_frames=24",
        'dataloader_train.dataloaders.cam_lidar_joint_480p_separate_view_text.dataloader.dataset.camera_sampler.rigs=[["camera_front_wide_120fov"]]',
        'model.config.student_load_from.load_path="/accepted/stage2/model"',
        'model.config.teacher_load_from.load_path="/original/base/model"',
        "model.config.ema.iteration_shift=0",
    ]
    translated = framework_auto_overrides(original)
    assert len(translated) == len(original)
    assert [item.partition("=")[2] for item in translated] == [item.partition("=")[2] for item in original]
    assert ".dataset.transform_overrides.num_video_frames=" in translated[0]
    assert ".dataset.transform_overrides.camera_sampler.rigs=" in translated[1]
    assert translated[2:] == original[2:]
    assert framework_auto_overrides(translated) == translated


def test_only_obsolete_cluster_and_io_cache_bindings_are_removed():
    original = [
        "job.cluster.object_store_credential_checkpoint=/credential",
        'dataloader_train.dataloaders.cam_lidar_joint_480p_separate_view_text.dataloader.cache_replay_name="old-io-name"',
        "checkpoint.load_from_object_store.credentials=/credential",
        "model.config.lidar_tokenizer.vae_path=/matched/lidar.safetensors",
    ]
    assert framework_auto_overrides(original) == original[2:]


def test_phase_callback_uses_native_target_and_keeps_output_identity():
    original = [
        'trainer.callbacks.dmd2_parity.output_path="/accepted/phase_metrics.jsonl"',
        "trainer.callbacks.dmd2_parity._target_=phase_metrics.PhaseMetrics",
    ]
    translated = framework_auto_overrides(original)
    assert translated[0] == original[0]
    assert (
        translated[1]
        == "trainer.callbacks.dmd2_parity._target_=cosmos_framework.callbacks.sim_auto_phase_metrics.PhaseMetrics"
    )
    assert framework_auto_overrides(translated) == translated
