# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native/Omni calibration checks; never imported by the production launcher."""


def validate_baseline(config) -> None:
    recipe = config.rl
    environment = config.environment.config
    expected = {
        "suite": "libero_10",
        "resume": "off",
        "max_steps": 0,
        "save_every": 0,
        "eval_profile": "aligned_200",
    }
    for key, value in expected.items():
        owner = environment if key in {"suite", "eval_profile"} else recipe
        if getattr(owner, key) != value:
            raise ValueError(f"baseline requires rl.{key}={value}")
    if recipe.validation.save_checkpoint:
        raise ValueError("baseline requires rl.validation.save_checkpoint=false")


from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody


class CalibrationGr00tBody(Gr00tPolicyBody):
    """Test-only observation of the ordinary Omni request through its worker extension seam."""

    def _engine_kwargs(self) -> dict:
        return {
            **super()._engine_kwargs(),
            "worker_extension_cls": "cosmos_framework.posttrain.tests.helpers.gr00t_omni_calibration.CalibrationWorkerExtension",
        }

    def capture_ode(self, observation: dict, *, seed: int, sampling_mode: str = "native_ode") -> dict:
        """Observe one ordinary inference request; never substitute a diagnostic model forward."""
        if sampling_mode not in ("ode", "native_ode"):
            raise ValueError("capture_ode requires deterministic ODE inference")
        self._worker_rpc("arm_ode_capture")
        try:
            action = self.infer_batch([observation], seed=seed, weight_version=0, sampling_mode=sampling_mode)[0]
        finally:
            captured = self._worker_rpc("take_ode_capture")["capture"]
        if captured is None:
            raise RuntimeError("the GR00T request did not produce its armed ODE capture")
        return {"action": action["action"], "capture": captured}
