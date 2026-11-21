# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""One reviewed, inference-only source transition for the selected native policies.

The native64/02 full-head artifacts predate optional GR00T head adapters and
evaluation-backbone reuse. Neither feature runs in the selected full-head path.
Only the exact old/current source closures below may compare as equivalent;
base, processor, task, precision, tensors, and trainable inventory stay strict.
This is not a training-resume or export migration. Unknown builds fail closed.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

INFERENCE_COMPATIBILITY = "cosmos.family-inference.native64-to-e64.full-head-r32.v1"

_OLD_SHARED = {
    "family": "803ca945270c55b207e596a8abe6e266d925be50c4ced4f45e7b2ba3d662fa03",
    "nf4": "0e72ae81d5477c6a96ee254aef9751a8bf406443e371afc809f2c04dc6f38491",
    "state": "61d9627b251c0b98d25683b2078f529c1a6118ea4adc98783a53f8986b2f4330",
    "adapter": "e58e8d51e25d2b26f2f1f1122d735c46a6870b3dd6b963d75459774324ea9ff7",
}
_CURRENT_SHARED = {
    **_OLD_SHARED,
    "family": "36708960f323bcebe4283e6f6760f467095c91353283ebbd9fd09f2469a10e11",
}
_OLD_GR00T = {
    "action_sft.py": "539e9b3ebb5c2133eed9819f4716958c6369c74fb97ba9c7135fdabc9e9b1353",
    "adaptation.py": "fd024cbc3b128923b31a37af3925dbbadd2831a6522bef4930fb2f439d44ac45",
    "packer.py": "efb528a6dc132c7fb809a24425a38f16c049d6f91c0b78e40a150f151cd2fead",
    "policy.py": "58aaa0a8acfa75046ea5e554073b7a1ff2945ea1c0441c5887a9ee5872db3515",
    "qlora_source.py": "264e0a9a22355b6ffed1fd1cb0ac0d736ea219c935a7a6e9d261504e10f222ce",
}
_CURRENT_GR00T = {
    **_OLD_GR00T,
    "action_sft.py": "0bfdac4de03308e198174dca409bbf949c500792a56d459fba615e4ef10b4e71",
    "adaptation.py": "2ce4e15613405c5abd71978e45a02ad5354ee4048ea328df0efb0f50d689e961",
    # Upstream main changed only one docstring line; the reviewed old hash stays in _OLD_GR00T.
    "packer.py": "bae0ac6a8940c1af1b27e63d81dcee7ce62871df9d78db427f6a1aa5b1333a2e",
    "policy.py": "c6c459a9f7962bf29e427d525774f1fe86f73dca63f7fe072e1b5ac9c122d1a3",
    "head_adaptation.py": "38242c02aad0af1a9ccc24feecaba502911aeb7d8c49a8a42551843840a7ff6d",
}
_ALPAMAYO_TASK = {
    "family": "alpamayo-r1-trajectory-only-v1",
    "loss": "native-flow-matching",
    "train_action_head": True,
    "activation_checkpointing": "full",
    "inference_steps": 10,
    "integrator": "euler",
    "reasoning_supervision": False,
}


def _selected(identity):
    """Limit this transition to the reviewed full-head recipes, not any old hash."""
    adapter = identity.get("adapter", {})
    if not isinstance(adapter, Mapping):
        return False
    mode = adapter.get("mode")
    seed = adapter.get("seed")
    if (
        identity.get("format") != "cosmos.family-backbone.v1"
        or mode not in ("nf4_lora", "bf16_lora")
        or type(seed) is not int
        or not 0 <= seed < 2**63
        or adapter != {"mode": mode, "rank": 32, "alpha": 64.0, "dropout": 0.0, "seed": seed}
        or "head_adapters" in identity
        or identity.get("torch_version") != "2.10.0+cu128"
        or identity.get("dependencies") != {"transformers": "4.57.6", "safetensors": "0.8.0"}
    ):
        return False
    task = identity.get("task_contract", {})
    if not isinstance(task, Mapping):
        return False
    if task.get("family") == "alpamayo-r1-trajectory-only-v1":
        return task == _ALPAMAYO_TASK
    if task.get("family") != "gr00t_n1d7":
        return False
    return task.get("config") == {
        "action_dim": 7,
        "backbone_adaptation": {
            **adapter,
            "profile": "gr00t_n1d7_reason2_2b_layer16_v1",
            "train_action_head": True,
        },
        "camera_map": {"image": "image", "wrist_image": "wrist_image"},
        "denoising_steps": 4,
        "embodiment_tag": "libero_sim",
        "execution_horizon": 8,
        "language_key": "annotation.language.language_instruction",
        "model_action_horizon": 40,
        "prompt_key": "prompt",
        "state_map": {key: key for key in ("gripper", "pitch", "roll", "x", "y", "yaw", "z")},
        "training_prefix": 10,
    }


def _reviewed_pair(saved, expected):
    if not isinstance(saved, Mapping) or not isinstance(expected, Mapping):
        return False
    if (
        not _selected(saved)
        or not _selected(expected)
        or saved.get("implementation") != _OLD_SHARED
        or expected.get("implementation") != _CURRENT_SHARED
        or saved["task_contract"]["family"] != expected["task_contract"]["family"]
    ):
        return False
    if saved["task_contract"]["family"] == "gr00t_n1d7":
        return (
            saved["task_contract"].get("implementation") == _OLD_GR00T
            and expected["task_contract"].get("implementation") == _CURRENT_GR00T
        )
    return True


def family_inference_identity_for_read(saved, expected):
    """Return a comparison copy; never modify the artifact or live model identity.

    The codec still compares every remaining field exactly. A failed guard leaves
    the original expectation intact, so the ordinary identity error is retained.
    """
    if not _reviewed_pair(saved, expected):
        return expected
    result = copy.deepcopy(expected)
    result["implementation"] = copy.deepcopy(saved["implementation"])
    if saved["task_contract"]["family"] == "gr00t_n1d7":
        result["task_contract"]["implementation"] = copy.deepcopy(saved["task_contract"]["implementation"])
    return result


def gr00t_unmerged_inference_identity_for_read(saved, expected, *, implementation):
    """Apply the same exact source check to GR00T's native unmerged subset contract.

    Alpamayo's unmerged loader already binds its unchanged model source directly.
    Merged dense loading and all exports deliberately do not call this function.
    """
    current = {**expected, "format": "cosmos.family-backbone.v1", "implementation": implementation}
    if not _reviewed_pair(saved, current) or saved["task_contract"]["family"] != "gr00t_n1d7":
        return expected
    result = copy.deepcopy(expected)
    result["task_contract"]["implementation"] = copy.deepcopy(saved["task_contract"]["implementation"])
    return result
