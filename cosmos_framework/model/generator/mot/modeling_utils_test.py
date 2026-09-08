# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math

import pytest
import torch

from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder

pytestmark = pytest.mark.L0


def test_timestep_frequencies_preserve_cpu_numerics_and_are_nonpersistent() -> None:
    embedder = TimestepEmbedder(hidden_size=8, frequency_embedding_size=6)
    timesteps = torch.tensor([0.25, 1.0], dtype=torch.float32)  # [N]
    expected_frequencies = torch.exp(  # [D/2]
        -math.log(10000) * torch.arange(start=0, end=3, dtype=torch.float32) / 3
    )
    expected = TimestepEmbedder.timestep_embedding(
        timesteps,
        dim=6,
        frequencies=expected_frequencies,
    )  # [N,D]

    actual = embedder.timestep_embedding(  # [N,D]
        timesteps,
        dim=6,
        frequencies=embedder._timestep_frequencies,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "_timestep_frequencies" not in embedder.state_dict()


def test_timestep_frequencies_are_initialized_on_buffer_device() -> None:
    embedder = TimestepEmbedder(hidden_size=8, frequency_embedding_size=6)
    device = torch.device("cpu")
    embedder._init_weights(buffer_device=device)
    expected_frequencies = torch.exp(  # [D/2]
        -math.log(10000) * torch.arange(start=0, end=3, dtype=torch.float32) / 3
    )

    assert embedder._timestep_frequencies.device == device
    assert dict(embedder.named_buffers())["_timestep_frequencies"] is embedder._timestep_frequencies
    assert "_timestep_frequencies" not in embedder.state_dict()
    torch.testing.assert_close(embedder._timestep_frequencies, expected_frequencies, rtol=0, atol=0)


@pytest.mark.GPU
def test_compiled_timestep_embedder_reuses_precomputed_frequencies() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the compiled CUDA Graph path")

    embedder = TimestepEmbedder(hidden_size=8, frequency_embedding_size=6).cuda()
    embedder._init_weights(buffer_device=torch.device("cuda"))
    frequency_data_ptr = embedder._timestep_frequencies.data_ptr()
    timesteps = torch.tensor([0.25, 1.0], device="cuda", dtype=torch.float32)  # [N]
    compiled_embedder = torch.compile(embedder, mode="reduce-overhead", fullgraph=True)

    compiled_outputs: list[torch.Tensor] = []
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        actual = compiled_embedder(timesteps).clone()  # [N,hidden_size]
        compiled_outputs.append(actual)
    expected = embedder(timesteps)  # [N,hidden_size]

    assert embedder._timestep_frequencies.data_ptr() == frequency_data_ptr
    torch.testing.assert_close(compiled_outputs[1], compiled_outputs[0], rtol=0, atol=0)
    torch.testing.assert_close(compiled_outputs[1], expected)
