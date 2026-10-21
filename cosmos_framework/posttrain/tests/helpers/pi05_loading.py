# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small real parameter trees and processor sources for policy-construction tests."""

import io
import json
from pathlib import Path

import sentencepiece
import torch
from safetensors.torch import save_file

from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model
from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOCondition


class SmallPi05Model(torch.nn.Module):
    apply_reference_precision = Pi05Model.apply_reference_precision

    def __init__(self, config: Pi05ModelConfig, *, device="cpu"):
        super().__init__()
        self.config = config
        with torch.device(device):
            self.paligemma_with_expert = torch.nn.Module()
            self.paligemma_with_expert.paligemma = torch.nn.Linear(3, 3, dtype=torch.bfloat16).requires_grad_(False)
            self.paligemma_with_expert.paligemma.input_layernorm = torch.nn.LayerNorm(3)
            self.paligemma_with_expert.paligemma.requires_grad_(False)
            self.paligemma_with_expert.gemma_expert = torch.nn.Module()
            self.paligemma_with_expert.gemma_expert.model = torch.nn.Linear(3, 3, dtype=torch.bfloat16)
            self.action_in_proj = torch.nn.Linear(3, 3)
            self.action_out_proj = torch.nn.Linear(3, 3)
            self.time_mlp_in = torch.nn.Linear(3, 3)
            self.time_mlp_out = torch.nn.Linear(3, 3)
            self.value_head = torch.nn.Linear(3, 1)

    @torch.no_grad()
    def prepare_condition(self, inputs):
        features = self.paligemma_with_expert.paligemma(inputs.images[0].mean(dim=(2, 3)).to(torch.bfloat16)).float()
        mask = torch.cat((inputs.image_masks[:, :, None].expand(-1, -1, 256).flatten(1), inputs.token_mask), dim=1)
        projection = features.mean(dim=1).to(torch.bfloat16)[:, None, None, None].expand(-1, 1, 968, 256)
        return Pi05PPOCondition(
            tuple((projection, projection) for _ in range(18)), mask, features.repeat(1, 683)[:, :2048]
        )


def write_processor_source(directory: Path) -> Pi05ModelConfig:
    buffer = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(["move the red block to the left of the bowl"]),
        model_writer=buffer,
        model_type="char",
        vocab_size=32,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    tokenizer = directory / "tokenizer.model"
    tokenizer.write_bytes(buffer.getvalue())
    stats = directory / "norm_stats.json"
    stats.write_text(
        json.dumps({"norm_stats": {name: {"q01": [-2.0] * 32, "q99": [6.0] * 32} for name in ("state", "actions")}})
    )
    return Pi05ModelConfig(str(directory), str(tokenizer), str(stats))


def write_source(directory: Path) -> Pi05ModelConfig:
    config = write_processor_source(directory)
    with torch.random.fork_rng(devices=[]):
        model = SmallPi05Model(config)
    save_file(model.state_dict(), str(directory / "model.safetensors"))
    return config
