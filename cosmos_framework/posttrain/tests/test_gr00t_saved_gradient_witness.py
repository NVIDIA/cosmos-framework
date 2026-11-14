# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed real GR00T derivative witnesses, without model or distributed execution.

The six coordinates were extracted from an already failed control. They expose
the FP32 reference rounding boundary; they are not a fresh qualification set.
The independent oracle uses math.fsum over all 32 native FP32 numerators, and
divides once by the fixed 32 * 10 * 7 valid-action count.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from cosmos_framework.posttrain.roles.trainer.gradient_window import OwnedGradientWindow, Owner

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("shards,replicas", [(4, 1), (2, 2)])
def test_saved_gr00t_gradients_keep_fp64_through_global_sum_and_window(shards, replicas):
    record = json.loads((Path(__file__).parent / "fixtures/gr00t_owned_gradient_witness.json").read_text())
    assert record["source_capture_sha256"] == "56f66e8c6a7213e2604586e26ee47eb93bab6fcd1ed864694e40d5eaee0aa527"
    assert record["original_update2_gate"] == "FAIL_PRESERVED"
    values = torch.tensor(record["local_numerator_fp32_bits"], dtype=torch.int32).view(torch.float32).reshape(8, 4, 6)
    expected = torch.tensor(
        [math.fsum(float(v) for v in values[:, :, i].reshape(-1)) / 2240 for i in range(6)], dtype=torch.float32
    )
    legacy = torch.zeros(6)
    for micro in values:
        legacy.add_((micro[0] + micro[1] + micro[2] + micro[3]) * 0.25)
    legacy.mul_(4 / 2240)
    assert legacy.view(torch.int32).tolist() == record["reference_fp32_mean_bits"]
    assert not torch.equal(legacy, expected)  # A reference-order tweak retains this rounding boundary.

    for rank in range(4):
        # Uneven shards include an empty last rank in FSDP and replicated owners
        # in HSDP. The logical gradient is never divided by the replica count.
        width = math.ceil(6 / shards)
        start = (rank % shards) * width
        p = torch.nn.Parameter(torch.zeros(max(0, min(width, 6 - start))))
        owner = Owner("witness", p, (6,), shards, replicas, ("witness", "shared_alias"))
        window = OwnedGradientWindow({"root": [owner]}, microbatches=8, world_size=4)
        window.begin_window()
        for micro in values.double():
            # Two distinct SUM groupings both retain FP64. This is a CPU
            # transport simulation, not an assertion about arbitrary NCCL trees.
            summed = (
                ((micro[2] + micro[0]) + (micro[3] + micro[1]))
                if shards == 4
                else ((micro[0] + micro[1]) + (micro[2] + micro[3]))
            )
            local = torch.zeros(width, dtype=torch.float64)
            local[: p.numel()] = summed[start : start + p.numel()]
            window.begin_micro(4 * 10 * 7)
            window.hook("root")(local)
            p.grad = torch.zeros_like(p)
            window.after_backward()
        assert p.grad is None
        window.finalize()
        assert torch.equal(p.grad, expected[start : start + p.numel()])
        assert window.count == record["global_valid_actions"] == 2240
        assert window.identity()["groups"]["root"][0]["replicas"] == replicas
