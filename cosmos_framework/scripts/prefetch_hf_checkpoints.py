# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pre-download all HF models used by the cosmos3 CI."""

import itertools

from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf


def prefetch_all() -> None:
    register_checkpoints()

    from cosmos_framework.inference.args import _CHECKPOINTS

    for cfg in _CHECKPOINTS.values():
        cfg.hf.download()

    from cosmos_framework.inference.common.checkpoints import CHECKPOINTS, DATASETS

    for cfg in itertools.chain(CHECKPOINTS.values(), DATASETS.values()):
        cfg.hf.download()

    from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS

    for cfg in _CHECKPOINTS.values():
        cfg.hf.download()

    for repo in [
        # 'cosmos_framework.auxiliary.guardrail.llamaGuard3.llamaGuard3',
        "meta-llama/Llama-Guard-3-8B",
        # 'cosmos_framework.auxiliary.guardrail.qwen3guard.qwen3guard',
        "Qwen/Qwen3Guard-Gen-0.6B",
        # 'cosmos_framework.auxiliary.guardrail.video_content_safety_filter.vision_encoder',
        "google/siglip-so400m-patch14-384",
    ]:
        CheckpointDirHf(repository=repo, revision="main").download()


def main():
    prefetch_all()


if __name__ == "__main__":
    main()
