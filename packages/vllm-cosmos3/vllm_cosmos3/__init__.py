# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""vLLM plugin: register Cosmos3 models.

See https://docs.vllm.ai/en/latest/design/plugin_system
"""

import logging

logger = logging.getLogger(__name__)


def register():
    from vllm import ModelRegistry

    arch = "Cosmos3ForConditionalGeneration"
    if arch not in ModelRegistry.get_supported_archs():
        logger.info("Registering architecture %s", arch)
        ModelRegistry.register_model(
            arch,
            "vllm_cosmos3.model:Cosmos3ForConditionalGeneration",
        )
