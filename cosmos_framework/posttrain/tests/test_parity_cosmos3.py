# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L1/GPU nightly 8B parity — two complementary tiers, both driven through the parity launcher under
`torchrun --nproc_per_node=4` (the way the golden was captured, so N — hence the FSDP reduction order —
matches):

- `test_cosmos3_loss_parity`: the real OmniMoT DPO recipe's 20-step loss matches a pinned golden.
  Deterministic (parity._enable_determinism sets FLASH_ATTENTION_DETERMINISTIC=1 + cuDNN/cuBLAS +
  use_deterministic_algorithms, compile off), so the loss is bit-identical across reruns on the same
  GB200 + torch + N — the golden is asserted at the harness's tight default tolerance (1e-5/1e-6, well
  under any real regression), re-captured on a torch/N change.
- `test_cosmos3_overfit`: reference-free canary — the loop memorizes a tiny fixed pair set (one per
  rank), so the loss must collapse well under the ln 2 chance init. Catches signal-destroying bugs a
  golden can hide, and needs no re-capture on a torch/N change.

Capture/refresh the golden on GB200 (one-off; update `_PROFILE.reference` and its runtime fields):
    PYTHONHASHSEED=42 torchrun --nproc_per_node=4 --standalone \\
        -m cosmos_framework.posttrain.tests.parity_cli --model cosmos3 --steps 20 --print-losses
"""

from __future__ import annotations

import pytest

from cosmos_framework.utils.device import get_gpu_architecture
from cosmos_framework.utils.helper_test import RunIf, run_command
from cosmos_framework.posttrain.tests.helpers.parity import assert_loss_parity, parse_printed_losses
from cosmos_framework.posttrain.tests.helpers.qualification import GoldenProfile

_PROFILE = GoldenProfile(
    name="cosmos3-dpo-8b-gb200",
    algorithm="dpo",
    accelerator="B200",
    torch_version="2.9.0a0+145a3a7bda.nv25.10",
    world_size=4,
    steps=20,
    seed=0,
    rtol=1e-5,
    atol=1e-6,
    reference=(
        0.6931471824645996,
        0.6907031536102295,
        0.14423547685146332,
        0.31637686491012573,
        0.912585437297821,
        0.4934064745903015,
        0.018458617851138115,
        0.09762674570083618,
        0.0013878007885068655,
        4.202785817142285e-07,
        0.9044762849807739,
        0.0006595104350708425,
        3.693242547342379e-07,
        0.03206409513950348,
        6.835276167294069e-08,
        0.011184033937752247,
        2.544129984016763e-06,
        0.0037608123384416103,
        2.6352758407592773,
        2.7996988137601875e-05,
    ),
)

# The parity launcher, under torchrun at the pinned N (PYTHONHASHSEED + determinism => bit-exact golden).
_LAUNCH = (
    f"PYTHONHASHSEED=42 PYTHONPATH=. torchrun --nproc_per_node={_PROFILE.world_size} --standalone "
    f"-m cosmos_framework.posttrain.tests.parity_cli --model cosmos3"
)


@RunIf(min_gpus=_PROFILE.world_size)
@pytest.mark.L1
@pytest.mark.GPU
def test_cosmos3_loss_parity():
    """The 8B loss trajectory matches the pinned N=4 golden at the harness's tight default tolerance
    (the run is bit-identical under the determinism stack)."""
    import torch

    _PROFILE.require_runtime(
        torch_version=torch.__version__,
        world_size=_PROFILE.world_size,
        accelerator=get_gpu_architecture(),
    )
    result = run_command(
        f"{_LAUNCH} --steps {_PROFILE.steps} --seed {_PROFILE.seed} --print-losses", max_retry_counter=1
    )
    losses = parse_printed_losses((result.stdout or "") + (result.stderr or ""), _PROFILE.steps)
    assert_loss_parity(losses, _PROFILE.reference, rtol=_PROFILE.rtol, atol=_PROFILE.atol)


@RunIf(min_gpus=_PROFILE.world_size)
@pytest.mark.L1
@pytest.mark.GPU
def test_cosmos3_overfit():
    """Reference-free canary: the 8B loop memorizes a tiny fixed pair set (one per rank), so the loss
    must collapse well under the ln 2 chance init. N- and torch-agnostic (no golden to re-capture)."""
    result = run_command(
        f"{_LAUNCH} --overfit --overfit-steps 30 --overfit-floor 0.2", is_raise=False, max_retry_counter=1
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert "PARITY OK" in output, f"8B overfit canary did not pass:\n{output[-2000:]}"
