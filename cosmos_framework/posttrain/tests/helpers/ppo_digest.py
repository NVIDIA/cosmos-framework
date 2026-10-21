# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic wire values covered by pre-streaming SHA256 golden digests."""

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOCondition, Pi05PPOTrace
from cosmos_framework.posttrain.schema import ObsBundle, SampleSource, Trajectory, Transition


@dataclass(frozen=True)
class DigestEnvelope:
    payload: object
    revision: int = field(init=False, default=7)


def digest_values() -> dict[str, object]:
    """Use small typed production records without constructing a model."""
    shared = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4).to(torch.bfloat16)
    condition = Pi05PPOCondition(
        prefix_cache=((shared, shared.transpose(2, 3)),),
        prefix_mask=torch.tensor([[True, False, True], [False, True, True]]),
        value_features=torch.arange(8, dtype=torch.float32).reshape(2, 4),
    )
    trace = Pi05PPOTrace(
        latent_before=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        latent_after=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 2,
        denoise_index=torch.tensor([0, 2]),
    )
    samples = [
        Trajectory(
            [
                Transition(
                    obs=ObsBundle(representations={"condition": condition}, id=f"sample-{index}"),
                    source=SampleSource(f"sample-{index}", "episode", "fragment"),
                    policy_info={"trace": trace, "sample_index": index, "weight_version": 3},
                    action=np.arange(7, dtype=np.float64),
                )
            ],
            id=f"sample-{index}",
        )
        for index in range(2)
    ]
    return {
        "scalars": [None, False, True, 0, -193, 2**96, 0.0, -0.0, 1.25, "π/\x00", b"\x00\xff"],
        "mutable_bytes": [bytearray(b"\x00\x80\xff"), memoryview(bytearray(range(20)))[::2]],
        "multidimensional_memoryview": memoryview(np.arange(12, dtype=np.int16).reshape(3, 4)),
        "mixed_mapping_keys": {(7, "x"): [1, 2], "7": (1, 2), 7: {"z": False, "a": True}},
        "torch_strided": torch.arange(30, dtype=torch.int64).reshape(5, 6)[:, ::2],
        "torch_bfloat16": shared,
        "torch_float16": torch.tensor([1.5, -2.5], dtype=torch.float16),
        "torch_complex": torch.tensor([1 + 2j, -3j], dtype=torch.complex64),
        "torch_nonfinite": torch.tensor([float("inf"), float("-inf"), float("nan")]),
        "torch_scalar": torch.tensor(-17, dtype=torch.int32),
        "torch_empty": torch.empty((2, 0, 3), dtype=torch.float64),
        "numpy_strided": np.arange(24, dtype=np.int32).reshape(4, 6)[:, ::2],
        "numpy_fortran": np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4)),
        "numpy_big_endian": np.arange(12, dtype=">i2"),
        "numpy_datetime": np.array(["2026-09-20", "2026-09-21"], dtype="datetime64[D]"),
        "numpy_scalar": np.float32(1.25),
        "numpy_empty": np.empty((0, 3), dtype=np.uint8),
        "pil": Image.fromarray(np.arange(18, dtype=np.uint8).reshape(2, 3, 3)),
        "nested_records": DigestEnvelope({"trajectories": samples, "condition_alias": condition}),
        "large_tensor": {"prefix": torch.arange(262144, dtype=torch.float32).reshape(1, 1, 1024, 256)},
    }
