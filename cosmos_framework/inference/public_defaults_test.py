# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest

from cosmos_framework.inference.args import _CHECKPOINTS, OmniSampleOverrides

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_edge_defaults_follow_architecture_for_local_exports() -> None:
    vlm = SimpleNamespace(
        model_instance={"_target_": "cosmos_framework.model.generator.mot.unified_mot.Nemotron3DenseVLTextForCausalLM"},
        model_name="example/local-backbone",
    )
    key = OmniSampleOverrides._vlm_model_key(vlm)
    assert OmniSampleOverrides._VLM_MODEL_SIZE[key] == "2B"
    assert OmniSampleOverrides._NUM_FRAMES_DEFAULTS[key] == 121


def test_named_model_defaults_resolve_through_hub() -> None:
    assert _CHECKPOINTS
    for checkpoint in _CHECKPOINTS.values():
        assert not checkpoint.s3_uri
        assert checkpoint.hf.repository.startswith("nvidia/")
        assert checkpoint.hf.revision
