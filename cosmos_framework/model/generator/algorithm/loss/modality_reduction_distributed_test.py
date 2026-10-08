# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real Gloo/DDP checks of global sample means with sharded CP intermediates."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.parallel import DistributedDataParallel

from cosmos_framework.model.generator.algorithm.loss.modality_reduction import ModalityLossItems, reduce_global_modality_means
from cosmos_framework.model.generator.mot.context_parallel_utils import all_gather_tensor

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
_WEIGHTS = {"image": 2.0, "video": 3.0, "action": 5.0, "sound": 7.0}


class _Network(torch.nn.Module):
    def __init__(self, cp_mesh: DeviceMesh | None) -> None:
        super().__init__()
        self.backbone: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(1.5))  # []
        self.heads: torch.nn.Parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))  # [4]
        self.cp_mesh: DeviceMesh | None = cp_mesh

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # [N] -> [4,4]
        hidden = self.backbone * features  # [N]
        if self.cp_mesh is not None:
            hidden = all_gather_tensor(hidden, 0, self.cp_mesh, correct_cp_gradients=True)  # [4]
        return hidden[:, None] * self.heads[None, :]  # [4,4]


def _items(predictions: torch.Tensor, owner: int) -> dict[str, ModalityLossItems]:
    # Group 0 has two controls/targets in one image sample, another image, and
    # one video. Group 1 has four videos, only the last of which has actions.
    weights = 1 + (torch.arange(4, dtype=predictions.dtype) + 1 + owner * 4) / 10  # [4]
    losses = predictions.square() * weights[:, None]  # [4,4]
    ids = torch.tensor([0, 0, 1, 2] if owner == 0 else [0, 1, 2, 3]) + owner * 10  # [4]
    masks = {
        "image": [True, True, True, False] if owner == 0 else [False] * 4,
        "video": [False, False, False, True] if owner == 0 else [True] * 4,
        "action": [False, False, False, owner == 1],
        "sound": [False] * 4,
    }
    return {
        name: ModalityLossItems(losses[:, index], ids, torch.tensor(masks[name])) for index, name in enumerate(_WEIGHTS)
    }


def _worker(rank: int, cp_size: int, port: int) -> None:
    torch.set_num_threads(1)
    world_size = 2 * cp_size
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size, timeout=timedelta(seconds=90)
    )
    mesh = init_device_mesh("cpu", (2, cp_size), mesh_dim_names=("dp", "cp"))
    owner = rank // cp_size
    network = _Network(mesh["cp"] if cp_size > 1 else None)
    ddp = DistributedDataParallel(network)
    for reverse_owners in (False, True):
        batch_owner = 1 - owner if reverse_owners else owner
        features = torch.arange(1 + batch_owner * 4, 5 + batch_owner * 4, dtype=torch.float32)  # [4]
        local_features = features.chunk(cp_size)[rank % cp_size]  # [4/CP]
        ddp.zero_grad(set_to_none=True)
        predictions = ddp(local_features)  # [4,4]
        loss, _ = reduce_global_modality_means(
            _items(predictions, batch_owner), _WEIGHTS, gradient_average_size=world_size
        )  # []
        loss.backward()

        reference = _Network(None)
        groups = [
            _items(reference(torch.arange(1 + index * 4, 5 + index * 4, dtype=torch.float32)), index)
            for index in range(2)
        ]
        combined = {
            name: ModalityLossItems(
                torch.cat([group[name].weighted_losses for group in groups]),  # [8]
                torch.cat([group[name].sample_ids for group in groups]),  # [8]
                torch.cat([group[name].valid for group in groups]),  # [8]
            )
            for name in _WEIGHTS
        }
        # The global reference is local: avoid entering an extra all-reduce here.
        reference_loss = sum(
            _WEIGHTS[name] * item.sample_sum_and_count()[0] / item.sample_sum_and_count()[1].clamp(min=1)
            for name, item in combined.items()
        )  # []
        reference_loss.backward()
        torch.testing.assert_close(loss, reference_loss)
        torch.testing.assert_close(network.backbone.grad, reference.backbone.grad)
        torch.testing.assert_close(network.heads.grad, reference.heads.grad)
        assert network.heads.grad[3].item() == 0
    dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [1, 2])
def test_ddp_global_means_and_cp_backbone_head_gradients(cp_size: int) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(cp_size, port), nprocs=2 * cp_size, join=True)
