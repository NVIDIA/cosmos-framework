# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Caption-to-embedding key mappings shared by dataset configuration and workers.

Keep these mappings in a dependency-light leaf module rather than in
``data_sources.data_registration``. Spawned Lance DataLoader workers import the
image augmentor modules; importing a shared constant from ``data_registration``
would also pull WebDataset registration code and its transitive dependencies
into every worker. Registration and augmentor code can instead import this
module without coupling the Lance runtime import path to WebDataset setup.

The current Lance JSON-caption pipeline does not dereference this mapping, but
it still imports the image augmentor module that also serves embedding-backed
WebDataset pipelines.
"""

# embeddings are packed together. Need to clean data to reduce entropy.
_CAPTION_EMBEDDING_KEY_MAPPING_IMAGES: dict[str, str] = {
    "ai_v3p1": "ai_v3p1",
    "qwen2p5_7b_v4": "qwen2p5_7b_v4",
    "prompts": "qwen2p5_7b_v4",
}
