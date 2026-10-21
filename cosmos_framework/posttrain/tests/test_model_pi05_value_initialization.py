# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real value-head initialization is independent of the learner/Omni constructor context."""

import pytest
import torch
from safetensors.torch import save_file

from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model, Pi05ValueHead
from cosmos_framework.posttrain.model.pi05.source import Pi05Source, load_model_weights

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("checkpoint_has_value", [False, True])
def test_value_head_matches_across_default_dtypes_and_preserves_source_and_rng(tmp_path, checkpoint_has_value):
    config = Pi05ModelConfig(str(tmp_path), str(tmp_path / "tokenizer.model"), str(tmp_path / "norm_stats.json"))
    original_dtype = torch.get_default_dtype()
    expected = {name: value.clone() for name, value in Pi05ValueHead().state_dict().items()}
    if checkpoint_has_value:
        # A supplied head is authoritative, even when it differs from the default seed.
        expected = {name: value + 0.125 for name, value in expected.items()}
    source_weights = {"actor": torch.arange(3, dtype=torch.float32)}
    if checkpoint_has_value:
        source_weights.update({f"value_head.{name}": value for name, value in expected.items()})
    save_file(source_weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "tokenizer.model").write_bytes(b"unused-by-weight-loader")
    (tmp_path / "norm_stats.json").write_text("{}")
    source = Pi05Source.resolve(config)
    try:
        with torch.random.fork_rng(devices=[]):
            for index, dtype in enumerate((torch.float32, torch.bfloat16)):
                torch.set_default_dtype(dtype)
                torch.random.default_generator.manual_seed(19 + index)
                before = torch.random.get_rng_state()
                # Only the large backbone/expert is meta. The production value head
                # is genuinely initialized on CPU, then the policy places the full model.
                model = Pi05Model(config, device="meta")
                assert torch.equal(torch.random.get_rng_state(), before)
                assert isinstance(model.value_head, Pi05ValueHead)
                container = torch.nn.Module()
                container.actor = torch.nn.Parameter(torch.empty(3, dtype=torch.float32))
                container.value_head = model.value_head
                load_model_weights(container, source)
                for name, parameter in container.value_head.named_parameters():
                    assert parameter.device.type == "cpu" and parameter.dtype == torch.float32
                    torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)
                assert torch.equal(torch.random.get_rng_state(), before)
    finally:
        torch.set_default_dtype(original_dtype)
