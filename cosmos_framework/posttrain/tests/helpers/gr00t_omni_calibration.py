# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Test-only worker extension; importing the normal extension registers the production pipeline."""

from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import Gr00tOmniWorkerExtension
from cosmos_framework.posttrain.tests.helpers.gr00t_diagnostics import Gr00tForwardCapture


class CalibrationWorkerExtension(Gr00tOmniWorkerExtension):
    def arm_ode_capture(self) -> dict:
        pipeline = self.model_runner.pipeline
        if not pipeline.ppo or getattr(self, "_capture_restore", None) is not None:
            raise ValueError("ODE capture requires an idle PPO-capable pipeline")
        original = pipeline._ppo_forward
        owned = "_ppo_forward" in vars(pipeline)
        self._last_ode_capture = None

        def restore() -> None:
            if owned:
                pipeline._ppo_forward = original
            else:
                del pipeline._ppo_forward
            self._capture_restore = None

        def capture(request):
            # Restore the method before entering it, including its exception path.
            restore()
            with Gr00tForwardCapture(
                pipeline.ppo_model.gr00t, pipeline.policy.processor, "_collator", pipeline.policy.processor
            ) as recorded:
                result = original(request)
            self._last_ode_capture = recorded.data
            return result

        self._capture_restore = restore
        pipeline._ppo_forward = capture
        return {"armed": True}

    def take_ode_capture(self) -> dict:
        if self._capture_restore is not None:
            self._capture_restore()
        result = self._last_ode_capture
        self._last_ode_capture = None
        return {"capture": result}
