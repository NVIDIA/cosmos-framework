# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import diffusers

from .pipeline import Cosmos3OmniDiffusersPipeline
from .transformer import Cosmos3OmniTransformer

# Spoof the eventual upstream module paths so save_pretrained writes
# "diffusers" as the library in model_index.json.
Cosmos3OmniTransformer.__module__ = "diffusers.models.transformers.transformer_cosmos3"
Cosmos3OmniDiffusersPipeline.__module__ = "diffusers.pipelines.cosmos.pipeline_cosmos3_omni"

# Loader does `getattr(importlib.import_module(library), class_name)`,
# so the class must be reachable as `diffusers.<ClassName>`.
diffusers.Cosmos3OmniTransformer = Cosmos3OmniTransformer
diffusers.Cosmos3OmniDiffusersPipeline = Cosmos3OmniDiffusersPipeline
