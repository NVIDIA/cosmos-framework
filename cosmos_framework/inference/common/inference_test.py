# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np

from cosmos_framework.inference.common.args import GuardrailArgs
from cosmos_framework.inference.common.inference import GuardrailRunners
from cosmos_framework.auxiliary.guardrail.common import presets


def test_guardrail_runners():
    guardrail_args = GuardrailArgs(guardrails=True, offload_guardrail_models=False)
    runners = GuardrailRunners.create(guardrail_args)
    assert runners.text is not None
    assert runners.video is not None

    assert presets.run_text_guardrail("test", runners.text)
    assert not presets.run_text_guardrail("Tesla Cybertruck", runners.text)

    frames_thwc = np.random.randint(0, 255, (1, 16, 16, 3), dtype=np.uint8)
    clean_frames_thwc = presets.run_video_guardrail(frames_thwc, runners.video)
    assert clean_frames_thwc is not None
    np.testing.assert_allclose(frames_thwc, clean_frames_thwc)
