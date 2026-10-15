# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reference objectives for sparse modalities and uneven distributed batches."""

from unittest.mock import patch

import pytest
import torch

from cosmos_framework.model.generator.algorithm.loss.modality_reduction import (
    MODALITIES,
    ModalityLossItems,
    reduce_global_modality_means,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _items(losses: torch.Tensor, owners: list[int], valid: list[bool] | None = None) -> ModalityLossItems:
    return ModalityLossItems(
        weighted_losses=losses,
        sample_ids=torch.tensor(owners, dtype=torch.long),  # [N_items]
        valid=torch.tensor(valid if valid is not None else [True] * len(owners)),  # [N_items]
    )


def test_sample_means_exclude_invalid_items_and_preserve_fixed_weights() -> None:
    image = torch.tensor([2.0, 6.0, 1000.0, 10.0], requires_grad=True)  # [4]
    video = torch.tensor([7.0], requires_grad=True)  # [1]
    action = torch.tensor([3.0, 900.0], requires_grad=True)  # [2]
    sound_probe = torch.tensor([8.0], requires_grad=True)  # [1]
    loss, means = reduce_global_modality_means(
        {
            "image": _items(image, [0, 0, 0, 1], [True, True, False, True]),
            "video": _items(video, [2]),
            "action": _items(action, [2, 3], [True, False]),
            "sound": _items(sound_probe * 0, [0], [False]),
        },
        {"image": 2.0, "video": 3.0, "action": 10.0, "sound": 100.0},
    )
    # Image sample means are 4 and 10. Neither invalid items nor the absent audio
    # modality dilute the sample denominators or redistribute configured weights.
    assert loss.item() == pytest.approx(2 * 7 + 3 * 7 + 10 * 3)
    assert means["sound"].item() == 0
    loss.backward()
    torch.testing.assert_close(image.grad, torch.tensor([0.5, 0.5, 0.0, 1.0]))
    torch.testing.assert_close(video.grad, torch.tensor([3.0]))
    torch.testing.assert_close(action.grad, torch.tensor([10.0, 0.0]))
    torch.testing.assert_close(sound_probe.grad, torch.zeros(1))


def test_reordering_items_keeps_objective_and_gradients() -> None:
    values = torch.tensor([1.0, 5.0, 9.0, 11.0], requires_grad=True)  # [4]
    permutation = torch.tensor([3, 1, 0, 2])  # [4]
    baseline, _ = reduce_global_modality_means({"video": _items(values, [0, 1, 1, 2])}, {"video": 2.0})
    reordered, _ = reduce_global_modality_means({"video": _items(values[permutation], [2, 1, 0, 1])}, {"video": 2.0})
    baseline_grad = torch.autograd.grad(baseline, values, retain_graph=True)[0]  # [4]
    reordered_grad = torch.autograd.grad(reordered, values)[0]  # [4]
    torch.testing.assert_close(baseline, reordered)
    torch.testing.assert_close(baseline_grad, reordered_grad)


@pytest.mark.parametrize("cp_copies", [1, 2])
def test_uneven_rank_partitions_and_cp_copies_match_global_reference(cp_copies: int) -> None:
    # Two independent batches: the first rank has two images and one video, the
    # second only a video with action. CP replicas process the same owner batch.
    parameter = torch.tensor(2.0, requires_grad=True)  # []
    image_losses = parameter.square() * torch.tensor([1.0, 3.0])  # [2]
    video_losses = parameter.square() * torch.tensor([5.0, 7.0])  # [2]
    action_losses = parameter.square() * torch.tensor([11.0])  # [1]
    global_items = {
        "image": _items(image_losses, [0, 1]),
        "video": _items(video_losses, [2, 3]),
        "action": _items(action_losses, [3]),
    }
    weights = {"image": 2.0, "video": 3.0, "action": 0.5, "sound": 20.0}
    reference, _ = reduce_global_modality_means(global_items, weights)
    reference_gradient = torch.autograd.grad(reference, parameter, retain_graph=True)[0]  # []
    sums = torch.tensor([16.0, 48.0, 44.0, 0.0, 0.0, 0.0], dtype=torch.float64)  # [6]
    counts = torch.tensor([2.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)  # [6]
    global_stats = torch.stack((sums, counts)) * cp_copies  # [2,6]
    world_size = 2 * cp_copies
    local_items = [
        {"image": global_items["image"], "video": _items(video_losses[:1], [2])},
        {"video": _items(video_losses[1:], [3]), "action": global_items["action"]},
    ]

    def all_reduce(tensor: torch.Tensor, **kwargs: object) -> None:
        tensor.copy_(global_stats)  # [2,6]

    gradients: list[torch.Tensor] = []
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.all_reduce", side_effect=all_reduce) as collective,
    ):
        for rank in range(world_size):
            local_loss, means = reduce_global_modality_means(
                local_items[rank // cp_copies], weights, gradient_average_size=world_size
            )
            torch.testing.assert_close(local_loss, reference)
            assert set(means) == set(MODALITIES)
            gradients.append(torch.autograd.grad(local_loss, parameter, retain_graph=True)[0])  # []
        assert collective.call_count == world_size
    averaged_gradient = torch.stack(gradients).mean()  # []
    torch.testing.assert_close(averaged_gradient, reference_gradient)


def test_invalid_metadata_fails_before_reduction() -> None:
    values = torch.ones(2, requires_grad=True)  # [2]
    with pytest.raises(ValueError, match="aligned"):
        reduce_global_modality_means({"image": _items(values, [0])}, {"image": 1.0})
    with pytest.raises(ValueError, match="Unknown"):
        reduce_global_modality_means({"unknown": _items(values, [0, 1])}, {})


def test_radar_sample_means_retain_supervision_and_absent_head_gradients() -> None:
    radar = torch.tensor([2.0, 6.0, 10.0, 99.0], requires_grad=True)  # [4]
    probe = torch.tensor([7.0], requires_grad=True)  # [1]
    loss, means = reduce_global_modality_means(
        {"radar": _items(radar, [0, 0, 2, 2], [True, True, True, False])},
        {"radar": 3.0},
    )  # [], dict of []
    assert means["radar"].item() == pytest.approx(7.0)
    assert loss.item() == pytest.approx(21.0)
    loss.backward()
    torch.testing.assert_close(radar.grad, torch.tensor([0.75, 0.75, 1.5, 0.0]))
    absent_loss, absent_means = reduce_global_modality_means(
        {"radar": _items(probe * 0, [0], [False])}, {"radar": 3.0}
    )  # [], dict of []
    absent_loss.backward()
    assert absent_means["radar"].item() == 0
    torch.testing.assert_close(probe.grad, torch.zeros(1))
