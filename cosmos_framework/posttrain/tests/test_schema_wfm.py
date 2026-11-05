# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contract tests for the typed online-WFM conditioning envelope."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cosmos_framework.posttrain.schema import WFM_CONDITIONING_KEY, ObsBundle, Transition, WFMConditioningInput


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_conditioning_input_is_a_frozen_named_envelope() -> None:
    model_batch = {"prompt": ["a robot opens a drawer"]}
    conditioning = WFMConditioningInput(
        id="source-0",
        template_identity="template-0",
        model_batch=model_batch,
        prompt="a robot opens a drawer",
    )
    transition = Transition(obs=ObsBundle(representations={WFM_CONDITIONING_KEY: conditioning}))

    assert WFM_CONDITIONING_KEY == "wfm_conditioning"
    assert conditioning.model_batch is model_batch
    assert transition.obs.representations[WFM_CONDITIONING_KEY] is conditioning
    with pytest.raises(FrozenInstanceError):
        setattr(conditioning, "id", "changed")

    with pytest.raises(ValueError, match="template_identity"):
        WFMConditioningInput(
            id="source-0",
            template_identity="",
            model_batch=model_batch,
            prompt="a robot opens a drawer",
        )
