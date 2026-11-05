# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.data.generator.action.agibot_grippers import (
    action_episode_gripper,
    episode_gripper,
)

_DATASET = "embodiment_c__EmbodimentCWorld_20260208_agibot-custom"
_SHARD = "20251113_330h/2025111101/gripper/task_2156"


def test_gripper_manifest_uses_portable_source_suffixes() -> None:
    assert action_episode_gripper(_DATASET, _SHARD, 0) == "omnipicker"
    assert (
        episode_gripper(f"open-dataset/EmbodimentCWorld_20260208/agibot-custom/{_SHARD}/episode_0/window_0")
        == "omnipicker"
    )
