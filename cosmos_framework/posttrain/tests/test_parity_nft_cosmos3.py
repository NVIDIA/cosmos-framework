# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L1/GPU nightly parity for the real four-GPU Cosmos3 DiffusionNFT stack.

The launcher uses fixed local prompts, optimizer-step seeds, deterministic rank-lane
rewards, and the real 8B OmniMoT/FSDP rollout, decoder, NFT objective, backward,
optimizer, and moving-old refresh. It never calls an external reward service. NATTEN
Blackwell FMHA has no deterministic backward contract, so the golden uses a narrow
tolerance even though two independent capture runs were bit-identical.

Capture or refresh the golden on one four-GPU GB200 node:

    PYTHONHASHSEED=42 PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
      -m cosmos_framework.posttrain.tests.nft_parity_cli \
      --model cosmos3 --steps 4
"""

from __future__ import annotations

import pytest

from cosmos_framework.utils.device import get_gpu_architecture
from cosmos_framework.utils.helper_test import RunIf, run_command
from cosmos_framework.posttrain.tests.helpers.parity import assert_loss_parity, parse_printed_losses
from cosmos_framework.posttrain.tests.helpers.qualification import GoldenProfile

_PROFILE = GoldenProfile(
    name="cosmos3-nft-8b-gb200",
    algorithm="diffusion-nft",
    accelerator="B200",
    torch_version="2.13.0a0+8145d630e8.nv26.06",
    world_size=4,
    steps=4,
    seed=0,
    rtol=5e-4,
    atol=1e-4,
    reference=(
        5.966474533081055,
        6.982649803161621,
        6.278761863708496,
        7.796826362609863,
    ),
)
_LAUNCH = (
    f"PYTHONHASHSEED=42 PYTHONPATH=. torchrun --standalone --nproc_per_node={_PROFILE.world_size} "
    f"-m cosmos_framework.posttrain.tests.nft_parity_cli --model cosmos3"
)


@RunIf(min_gpus=_PROFILE.world_size)
@pytest.mark.L1
@pytest.mark.GPU
def test_cosmos3_nft_loss_parity() -> None:
    """The real OmniMoT NFT loss trajectory matches its pinned N=4 golden."""
    import torch

    _PROFILE.require_runtime(
        torch_version=torch.__version__,
        world_size=_PROFILE.world_size,
        accelerator=get_gpu_architecture(),
    )
    result = run_command(
        f"{_LAUNCH} --steps {_PROFILE.steps} --seed {_PROFILE.seed}",
        max_retry_counter=1,
    )
    losses = parse_printed_losses((result.stdout or "") + (result.stderr or ""), _PROFILE.steps)
    assert_loss_parity(
        losses,
        _PROFILE.reference,
        rtol=_PROFILE.rtol,
        atol=_PROFILE.atol,
    )


@RunIf(min_gpus=_PROFILE.world_size)
@pytest.mark.L1
@pytest.mark.GPU
def test_cosmos3_nft_learning_canary() -> None:
    """All updates are finite, nonzero-gradient, and refresh old on the pinned cadence."""
    result = run_command(
        f"{_LAUNCH} --steps {_PROFILE.steps} --seed {_PROFILE.seed} --canary",
        is_raise=False,
        max_retry_counter=1,
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert "NFT PARITY OK" in output, f"Cosmos3 NFT canary did not pass:\n{output[-2000:]}"
