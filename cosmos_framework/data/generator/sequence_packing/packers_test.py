# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regression coverage for sensor clocks and joint sensor clean replay."""

import pytest
import torch

from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.data.generator.sequence_packing.packers import (
    make_teacher_forcing_clean_pack,
    pack_input_sequence,
    validate_cross_view_fps,
)
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("view_id", [-1, 8])
@pytest.mark.parametrize("frame_idx", [0, 2])
def test_autoregressive_pack_preserves_physical_view_identity(view_id: int, frame_idx: int) -> None:
    from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive

    ids = torch.tensor([view_id])  # [V]
    packed = pack_input_sequence_autoregressive(
        vision_latent=torch.zeros(1, 4, 1, 2, 2),  # [B,C,T,H,W]
        action_latent=None,
        text_tokens=[1, 2] if frame_idx == 0 else None,
        timestep=0.0,
        fps_vision=[24.0],
        fps_action=None,
        special_tokens={"eos_token_id": 3, "start_of_generation": 4, "end_of_generation": 5},
        frame_idx=frame_idx,
        cached_text_offset=None if frame_idx == 0 else 4,
        vision_view_ids=ids,
    )

    assert packed.vision_view_ids is not None
    assert len(packed.vision_view_ids) == 1
    torch.testing.assert_close(packed.vision_view_ids[0], ids)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("fps", [None, 0.0, -1.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "num_items,views_per_item,separate_camera_items",
    [(None, [2], False), ([3], None, True), ([3], [1, 1, 1], True)],
)
def test_cross_view_fps_rejects_invalid_camera_clock(
    fps: float | None,
    num_items: list[int] | None,
    views_per_item: list[int] | None,
    separate_camera_items: bool,
) -> None:
    fps_values = torch.tensor([fps]) if fps is not None else None  # [1] | None
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        num_vision_items_per_sample=num_items,
        num_views_per_vision_item=views_per_item,
        fps_vision=fps_values,
    )

    with pytest.raises(ValueError, match="positive, finite fps_vision for sample 0"):
        validate_cross_view_fps(
            [SequencePlan(has_text=True, has_vision=True)], data, separate_camera_items=separate_camera_items
        )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "fps,require_single_view,raises_error", [(None, False, False), (None, True, True), (30.0, True, False)]
)
def test_same_view_seconds_window_requires_single_camera_clock(
    fps: float | None, require_single_view: bool, raises_error: bool
) -> None:
    fps_values = torch.tensor([fps]) if fps is not None else None  # [1] | None
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        num_views_per_vision_item=[1],
        fps_vision=fps_values,
    )
    plans = [SequencePlan(has_text=True, has_vision=True)]

    if raises_error:
        with pytest.raises(ValueError, match="positive, finite fps_vision for sample 0"):
            validate_cross_view_fps(plans, data, require_single_view=require_single_view)
    else:
        validate_cross_view_fps(plans, data, require_single_view=require_single_view)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("fps,require_single_view", [(None, False), (None, True), (0.0, True), (30.0, True)])
def test_same_view_seconds_window_requires_action_control_clock(fps: float | None, require_single_view: bool) -> None:
    fps_action = torch.tensor([fps]) if fps is not None else None  # [1] | None
    data = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        fps_vision=torch.tensor([30.0, 30.0]),  # [2]
        fps_action=fps_action,
    )
    # Only the second sample has an action item, so its clock occupies action index 0.
    plans = [
        SequencePlan(has_text=True, has_vision=True),
        SequencePlan(has_text=True, has_vision=True, has_action=True),
    ]

    if require_single_view and (fps is None or fps <= 0):
        with pytest.raises(ValueError, match="positive, finite fps_action for sample 1"):
            validate_cross_view_fps(plans, data, require_single_view=require_single_view)
    else:
        validate_cross_view_fps(plans, data, require_single_view=require_single_view)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("modality", ["vision", "lidar", "radar"])
@pytest.mark.parametrize("fps", [None, 0.0])
def test_cross_view_fps_requires_each_joint_sensor_clock(modality: str, fps: float | None) -> None:
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        fps_vision=torch.tensor([30.0]),  # [1]
        fps_lidar=torch.tensor([10.0]),  # [1]
        fps_radar=torch.tensor([20.0]),  # [1]
    )
    fps_values = torch.tensor([fps]) if fps is not None else None  # [1] | None
    setattr(data, f"fps_{modality}", fps_values)
    plans = [SequencePlan(has_text=True, has_vision=True, has_lidar=True, has_radar=True)]

    with pytest.raises(ValueError, match=f"positive, finite fps_{modality} for sample 0"):
        validate_cross_view_fps(plans, data)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("modality", ["vision", "lidar", "radar"])
def test_cross_view_fps_rejects_missing_sample_clock(modality: str) -> None:
    data = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        fps_vision=torch.tensor([30.0, 30.0]),  # [2]
        fps_lidar=torch.tensor([10.0, 10.0]),  # [2]
        fps_radar=torch.tensor([20.0, 20.0]),  # [2]
    )
    short_fps_values = torch.tensor([30.0])  # [1]
    setattr(data, f"fps_{modality}", short_fps_values)
    plans = [SequencePlan(has_text=True, has_vision=True, has_lidar=True, has_radar=True) for _ in range(2)]

    with pytest.raises(ValueError, match=f"positive, finite fps_{modality} for sample 1"):
        validate_cross_view_fps(plans, data)


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_view_fps_indexes_samples_independently_of_transfer_items() -> None:
    # Single-view transfer, LiDAR-only, then multiview transfer. Invalid camera clocks
    # in the first two samples are unused; the third sample owns vision items 2 and 3.
    fps_vision = torch.tensor([float("nan"), 0.0, 30.0])  # [3]
    data = GenerationDataClean(
        batch_size=3,
        is_image_batch=False,
        num_vision_items_per_sample=[2, 0, 2],
        num_views_per_vision_item=[1, 1, 3, 3],
        fps_vision=fps_vision,
    )
    plans = [
        SequencePlan(has_text=True, has_vision=True),
        SequencePlan(has_text=True, has_lidar=True),
        SequencePlan(has_text=True, has_vision=True),
    ]

    validate_cross_view_fps(plans, data)

    fps_vision[2] = 0.0  # [3]
    with pytest.raises(ValueError, match="positive, finite fps_vision for sample 2"):
        validate_cross_view_fps(plans, data)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("enable_fps_modulation", [False, True])
def test_cross_view_fps_preserves_joint_transfer_frame_periods(enable_fps_modulation: bool) -> None:
    camera_items = [torch.zeros(1, 4, 4, 1, 1) for _ in range(2)]  # each [1,C,T,H,W]
    lidar_items = [torch.zeros(1, 4, 3, 1, 1) for _ in range(2)]  # each [1,C,T,H,W]
    radar_items = [torch.zeros(1, 4, 5, 1, 1) for _ in range(2)]  # each [1,C,T,H,W]
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=camera_items,
        x0_tokens_lidar=lidar_items,
        x0_tokens_radar=radar_items,
        num_vision_items_per_sample=[2],
        num_views_per_vision_item=[2, 2],
        num_lidar_items_per_sample=[2],
        num_radar_items_per_sample=[2],
        fps_vision=torch.tensor([30.0]),  # [1]
        fps_lidar=torch.tensor([10.0]),  # [1]
        fps_radar=torch.tensor([20.0]),  # [1]
    )
    plans = [SequencePlan(has_text=True, has_vision=True, has_lidar=True, has_radar=True)]
    timesteps = torch.tensor([0.5])  # [1]

    validate_cross_view_fps(plans, data)
    packed = pack_input_sequence(
        sequence_plans=plans,
        input_text_indexes=[[1, 2]],
        gen_data_clean=data,
        input_timesteps=timesteps,
        special_tokens={"eos_token_id": 3, "start_of_generation": 4, "end_of_generation": 5},
        enable_fps_modulation=enable_fps_modulation,
        temporal_compression_factor=4,
        lidar_temporal_compression_factor=1,
        radar_temporal_compression_factor=1,
    )

    assert packed.vision is not None
    assert packed.lidar is not None
    assert packed.radar is not None
    assert packed.vision.seconds_per_frame == pytest.approx([4 / 30, 4 / 30])
    assert packed.lidar.seconds_per_frame == pytest.approx([1 / 10, 1 / 10])
    assert packed.radar.seconds_per_frame == pytest.approx([1 / 20, 1 / 20])


@pytest.mark.L0
@pytest.mark.CPU
def test_clean_replay_marks_lidar_clean_without_mutating_the_training_pack() -> None:
    lidar = ModalityData(
        tokens=[torch.randn(3, 2)],  # list of [T,C]
        condition_mask=[torch.tensor([1.0, 0.0, 0.0])],  # list of [T]
        mse_loss_indexes=torch.tensor([1, 2]),  # [N_noisy]
        timesteps=torch.tensor([0.4, 0.4]),  # [N_noisy]
        noisy_frame_indexes=[torch.tensor([1, 2])],  # list of [N_noisy]
    )
    packed = PackedSequence(text_ids=torch.tensor([1]), sample_lens=[3], lidar=lidar)  # [S_text]
    clean = make_teacher_forcing_clean_pack(packed)
    assert clean.lidar is not None
    assert clean.lidar.condition_mask[0].all()
    assert clean.lidar.mse_loss_indexes.numel() == 0
    assert clean.lidar.timesteps.numel() == 0
    assert clean.lidar.noisy_frame_indexes[0].numel() == 0
    torch.testing.assert_close(clean.lidar.tokens[0], lidar.tokens[0])
    assert clean.lidar.tokens[0].data_ptr() != lidar.tokens[0].data_ptr()
    assert lidar.mse_loss_indexes.tolist() == [1, 2]
    assert lidar.condition_mask[0].tolist() == [1, 0, 0]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("modality_name", ["vision", "lidar", "action", "sound"])
def test_clean_replay_preserves_each_modality_and_its_metadata_types(modality_name: str) -> None:
    modality = ModalityData(
        tokens=[torch.arange(6, dtype=torch.float64).reshape(3, 2)],  # list[[3,2]]
        condition_mask=[torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)],  # list[[3]]
        mse_loss_indexes=torch.tensor([1, 2], dtype=torch.int32),  # [2]
        timesteps=torch.tensor([0.4, 0.4], dtype=torch.float64),  # [2]
        noisy_frame_indexes=[torch.tensor([1, 2], dtype=torch.int32)],  # list[[2]]
    )
    packed = PackedSequence(text_ids=torch.tensor([1]), sample_lens=[3])  # [1]
    setattr(packed, modality_name, modality)
    clean = getattr(make_teacher_forcing_clean_pack(packed), modality_name)
    assert isinstance(clean, ModalityData)
    assert torch.equal(clean.tokens[0], modality.tokens[0])
    assert clean.tokens[0].data_ptr() != modality.tokens[0].data_ptr()
    assert clean.condition_mask[0].tolist() == [1.0, 1.0, 1.0]
    for field in ("mse_loss_indexes", "timesteps"):
        original = getattr(modality, field)
        emptied = getattr(clean, field)
        assert emptied.numel() == 0
        assert (emptied.dtype, emptied.device) == (original.dtype, original.device)
        assert original.numel() == 2
    assert clean.noisy_frame_indexes[0].numel() == 0
    assert clean.noisy_frame_indexes[0].dtype == torch.int32
    assert modality.noisy_frame_indexes[0].tolist() == [1, 2]
