# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

import pytest
import torch

from cosmos_framework.model.generator.algorithm.loss.flow_matching import (
    ActionSlotLossStats,
    compute_flow_matching_loss,
)
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    EGO_POSE,
    RIGHT_OPENNESS,
    RIGHT_WRIST_POSE,
    UNIFIED_ACTION_DIM,
)


class _UnitWeightFlow:
    def train_time_weight(self, timesteps, tensor_kwargs):
        return torch.ones_like(timesteps, **tensor_kwargs)


@pytest.mark.L0
def test_action_valid_mask_excludes_slots_and_denominator() -> None:
    pred = [torch.tensor([[1.0, 100.0, 3.0], [1.0, 100.0, 3.0]])]
    target = [torch.zeros(2, 3)]
    condition_mask = [torch.zeros(2, 1)]
    valid_mask = [torch.tensor([True, False, True])]

    weighted, per_instance = compute_flow_matching_loss(
        pred=pred,
        target=target,
        condition_mask=condition_mask,
        timesteps=torch.zeros(1, 1),
        has_valid_tokens=True,
        rectified_flow=_UnitWeightFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32},
        action_valid_mask=valid_mask,
    )

    # Only errors 1^2 and 3^2 are valid, for each of two frames: 20 / 4 = 5.
    torch.testing.assert_close(per_instance, torch.tensor([5.0]))
    torch.testing.assert_close(weighted, torch.tensor(5.0))


@pytest.mark.L0
def test_action_valid_mask_validates_width() -> None:
    with pytest.raises(ValueError, match="action_valid_mask width"):
        compute_flow_matching_loss(
            pred=[torch.zeros(2, 3)],
            target=[torch.zeros(2, 3)],
            condition_mask=[torch.zeros(2, 1)],
            timesteps=torch.zeros(1, 1),
            has_valid_tokens=True,
            rectified_flow=_UnitWeightFlow(),
            tensor_kwargs_fp32={"dtype": torch.float32},
            action_valid_mask=[torch.ones(2, dtype=torch.bool)],
        )


@pytest.mark.L0
def test_padded_action_valid_mask_is_cropped_with_raw_action_dim() -> None:
    weighted, _ = compute_flow_matching_loss(
        pred=[torch.tensor([[1.0, 100.0, 3.0, 1000.0]])],
        target=[torch.zeros(1, 4)],
        condition_mask=[torch.zeros(1, 1)],
        timesteps=torch.zeros(1, 1),
        has_valid_tokens=True,
        rectified_flow=_UnitWeightFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32},
        raw_action_dim=[torch.tensor(3)],
        action_valid_mask=[torch.tensor([True, False, True, False])],
    )
    torch.testing.assert_close(weighted, torch.tensor(5.0))


@pytest.mark.L0
def test_action_slot_losses_share_original_computation_with_padded_model_width() -> None:
    max_action_dim = UNIFIED_ACTION_DIM + 5
    pred = torch.full((1, max_action_dim), 100.0, requires_grad=True)
    with torch.no_grad():
        pred[:, EGO_POSE] = 2.0
        pred[:, RIGHT_WRIST_POSE] = 3.0
        pred[:, RIGHT_OPENNESS] = 4.0
    valid_mask = torch.zeros(max_action_dim, dtype=torch.bool)
    valid_mask[EGO_POSE] = True
    valid_mask[RIGHT_WRIST_POSE] = True
    valid_mask[RIGHT_OPENNESS] = True

    slot_stats = ActionSlotLossStats(sample_loss=torch.zeros(1, 7), sample_count=torch.zeros(1, 7))
    weighted, per_instance = compute_flow_matching_loss(
        pred=[pred],
        target=[torch.zeros_like(pred)],
        condition_mask=[torch.zeros(1, 1)],
        timesteps=torch.zeros(1, 1),
        has_valid_tokens=True,
        rectified_flow=_UnitWeightFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32},
        raw_action_dim=[torch.tensor(UNIFIED_ACTION_DIM)],
        action_valid_mask=[valid_mask],
        action_slot_stats=slot_stats,
    )

    torch.testing.assert_close(weighted, torch.tensor(7.0))
    torch.testing.assert_close(per_instance, torch.tensor([7.0]))
    torch.testing.assert_close(
        slot_stats.sample_loss,
        torch.tensor([[4.0, 9.0, 0.0, 16.0, 0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(slot_stats.sample_count, torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))
    assert weighted.requires_grad
    assert not slot_stats.sample_loss.requires_grad


@pytest.mark.L0
def test_conditioned_frames_remain_in_default_denominator_with_slot_mask() -> None:
    pred = [torch.tensor([[1.0, 100.0, 3.0], [1.0, 100.0, 3.0]])]
    target = [torch.zeros(2, 3)]
    condition_mask = [torch.tensor([[1.0], [0.0]])]
    valid_mask = [torch.tensor([True, False, True])]
    common_kwargs = dict(
        pred=pred,
        target=target,
        condition_mask=condition_mask,
        timesteps=torch.zeros(1, 1),
        has_valid_tokens=True,
        rectified_flow=_UnitWeightFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32},
        action_valid_mask=valid_mask,
    )

    default_weighted, default_per_instance = compute_flow_matching_loss(**common_kwargs)
    active_weighted, active_per_instance = compute_flow_matching_loss(
        **common_kwargs,
        normalize_by_active=True,
    )

    # One noisy frame contributes 1^2 + 3^2 = 10. Default parity with
    # ``.mean()`` divides by 2 timesteps * 2 valid channels; active mode divides
    # only by 1 noisy timestep * 2 valid channels.
    torch.testing.assert_close(default_per_instance, torch.tensor([2.5]))
    torch.testing.assert_close(default_weighted, torch.tensor(2.5))
    torch.testing.assert_close(active_per_instance, torch.tensor([5.0]))
    torch.testing.assert_close(active_weighted, torch.tensor(5.0))
