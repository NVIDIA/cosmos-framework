# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""vLLM plugin: register Cosmos3 models.

See https://docs.vllm.ai/en/latest/design/plugin_system
"""


def register():
    # Import for the side effect of registering `cosmos3_omni` with
    # transformers.AutoConfig; must happen before vLLM calls
    # AutoConfig.from_pretrained on the checkpoint.
    import transformers_cosmos3 as transformers_cosmos3
    from vllm import ModelRegistry

    arch = "Cosmos3ReasonerForConditionalGeneration"
    if arch not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            arch,
            "vllm_cosmos3.model:Cosmos3ReasonerForConditionalGeneration",
        )
