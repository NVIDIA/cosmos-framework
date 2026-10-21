# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Physical Intelligence
# SPDX-License-Identifier: OpenMDW-1.1
"""The fixed OpenPI π0.5 backbone/expert tree with reference checkpoint names.

Adapted from OpenPI models_pytorch at the RLinf revision in config.py. Frozen
prefix construction stays outside the learner's sharded trainable Pi05Heads.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import GemmaConfig, PaliGemmaConfig
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from cosmos_framework.posttrain.model.pi05.compat import Pi05GemmaModel, additive_attention_mask
from cosmos_framework.posttrain.model.pi05.config import PALIGEMMA_VOCAB_SIZE, Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.processing import Pi05Inputs

if TYPE_CHECKING:
    from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOCondition


class Pi05ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Raw SFT sources omit this head. Its identity declares an independent
        # CPU seed, unaffected by backbone construction or an Omni dtype context.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(42)
            layers: list[nn.Module] = []
            width = 2048
            for output in (1024, 512, 256):
                layers.extend((nn.Linear(width, output, dtype=torch.float32), nn.ReLU()))
                width = output
            layers.append(nn.Linear(width, 1, dtype=torch.float32))
            self.mlp = nn.Sequential(*layers)
            self.initialize_weights()

    def initialize_weights(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                if layer is self.mlp[-1]:
                    nn.init.normal_(layer.weight, std=0.02)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class Pi05Model(nn.Module):
    def __init__(self, config: Pi05ModelConfig, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.config = config
        vlm_config = PaliGemmaConfig()
        vlm_config.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config.text_config.vocab_size = PALIGEMMA_VOCAB_SIZE
        vlm_config.vision_config.intermediate_size = 4304
        vlm_config.vision_config.projection_dim = 2048
        vlm_config.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config.vision_config._attn_implementation = "sdpa"
        expert_config = GemmaConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=18,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=256,
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            hidden_activation="gelu_pytorch_tanh",
        )
        with torch.device(device):
            # These containers preserve the original checkpoint tree without
            # constructing unused AutoModel generation implementations.
            self.paligemma_with_expert = nn.Module()
            paligemma = nn.Module()
            paligemma.model = nn.Module()
            paligemma.model.vision_tower = SiglipVisionModel(vlm_config.vision_config)
            paligemma.model.multi_modal_projector = PaliGemmaMultiModalProjector(vlm_config)
            paligemma.model.language_model = Pi05GemmaModel(vlm_config.text_config, adaptive=False)
            paligemma.lm_head = nn.Linear(2048, PALIGEMMA_VOCAB_SIZE, bias=False)
            paligemma.lm_head.weight = paligemma.model.language_model.embed_tokens.weight
            self.paligemma_with_expert.paligemma = paligemma
            expert = nn.Module()
            expert.model = Pi05GemmaModel(expert_config, adaptive=True)
            expert.lm_head = nn.Linear(1024, PALIGEMMA_VOCAB_SIZE, bias=False)
            self.paligemma_with_expert.gemma_expert = expert
            self.action_in_proj = nn.Linear(32, 1024, dtype=torch.float32)
            self.action_out_proj = nn.Linear(1024, 32, dtype=torch.float32)
            self.time_mlp_in = nn.Linear(1024, 1024, dtype=torch.float32)
            self.time_mlp_out = nn.Linear(1024, 1024, dtype=torch.float32)
            self.value_head = Pi05ValueHead()
        self.apply_reference_precision()
        paligemma.requires_grad_(False).eval()
        expert.lm_head.requires_grad_(False)

    def apply_reference_precision(self) -> None:
        """Reproduce RLinf's post-load BF16 roundtrip before restoring selected FP32 storage.

        The FP32 norm and vision-embedding values are BF16-quantized by the
        reference loader. Call this after loading source weights, before sharding;
        action/time/value projections remain untouched.
        """
        self.paligemma_with_expert.to(dtype=torch.bfloat16)
        for name, parameter in self.paligemma_with_expert.named_parameters():
            if any(
                key in name
                for key in ("vision_model.embeddings", "input_layernorm", "post_attention_layernorm", "model.norm")
            ):
                parameter.data = parameter.data.to(torch.float32)

    @torch.no_grad()
    def prepare_condition(self, inputs: Pi05Inputs) -> Pi05PPOCondition:
        from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOCondition

        paligemma = self.paligemma_with_expert.paligemma.model
        vision = paligemma.vision_tower.vision_model
        embeddings = []
        masks = []
        for index, image in enumerate(inputs.images):
            hidden = vision.embeddings(image)
            hidden = hidden.to(vision.encoder.layers[0].self_attn.q_proj.weight.dtype)
            hidden = vision.encoder(inputs_embeds=hidden).last_hidden_state
            hidden = vision.post_layernorm(hidden)
            hidden = paligemma.multi_modal_projector(hidden)
            embeddings.append(hidden)
            masks.append(inputs.image_masks[:, index, None].expand(-1, hidden.shape[1]))
        language = paligemma.language_model.embed_tokens(inputs.tokens)
        embeddings.append(language * math.sqrt(language.shape[-1]))
        masks.append(inputs.token_mask)
        hidden = torch.cat(embeddings, dim=1)
        mask = torch.cat(masks, dim=1)
        attention = additive_attention_mask(mask[:, :, None] & mask[:, None, :])
        positions = torch.cumsum(mask, dim=1) - 1
        output, cache = paligemma.language_model(hidden, attention, positions)
        # Reference pools every language slot, including padding, but excludes
        # the missing third camera's 256 slots. A pad-mask mean is different.
        pooled = torch.cat((output[:, :512], output[:, 768:]), dim=1).mean(dim=1).float()
        return Pi05PPOCondition(cache, mask, pooled)
