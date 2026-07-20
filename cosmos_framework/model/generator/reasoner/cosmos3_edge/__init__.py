# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Framework-native Cosmos3-Edge VLM (renewed ``nvidia/Cosmos3-Edge``, no remote code).

Importing this package registers ``model_type="cosmos3_edge"`` with the transformers
Auto classes, so ``AutoConfig.from_pretrained``/``AutoModelForImageTextToText`` resolve
the renewed HF snapshot without ``trust_remote_code`` (HFModel imports this package).
"""

from transformers import AutoConfig, AutoModelForImageTextToText

from cosmos_framework.model.generator.reasoner.cosmos3_edge.configuration_cosmos3_edge import (
    Cosmos3EdgeConfig,
    Cosmos3EdgeProjectorConfig,
    Cosmos3EdgeTextConfig,
    Cosmos3EdgeVisionConfig,
)
from cosmos_framework.model.generator.reasoner.cosmos3_edge.modeling_cosmos3_edge import (
    Cosmos3EdgeForConditionalGeneration,
    Cosmos3EdgeModel,
    Cosmos3EdgePreTrainedModel,
    Cosmos3EdgeTextModel,
)

AutoConfig.register("cosmos3_edge", Cosmos3EdgeConfig, exist_ok=True)
AutoModelForImageTextToText.register(Cosmos3EdgeConfig, Cosmos3EdgeForConditionalGeneration, exist_ok=True)

__all__ = [
    "Cosmos3EdgeConfig",
    "Cosmos3EdgeForConditionalGeneration",
    "Cosmos3EdgeModel",
    "Cosmos3EdgePreTrainedModel",
    "Cosmos3EdgeProjectorConfig",
    "Cosmos3EdgeTextConfig",
    "Cosmos3EdgeTextModel",
    "Cosmos3EdgeVisionConfig",
]
