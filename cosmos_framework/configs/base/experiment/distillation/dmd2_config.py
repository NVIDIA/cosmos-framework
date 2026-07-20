# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Literal

import attrs
from hydra.core.config_store import ConfigStore  # type: ignore[import]

from cosmos_framework.configs.base.defaults.model_config import FixedStepSamplerConfig, OmniMoTModelConfig
from cosmos_framework.configs.base.defaults.reasoner import PretrainedWeightsConfig, VLMConfig

# Cosmos3 way of registering/configuring optimizers
from cosmos_framework.utils.generator.optimizer import build_optimizer
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

__all__: tuple[str, ...] = (
    "DMD2OptimizerConfig",
    "DMD2RFConfig",
    "register_dmd2_optimizer",
)

_STANDARD_KEYS_TO_SELECT: list[str] = [
    "moe_gen",
    "q_norm",
    "k_norm",
    "time_embedder",
    "vae2llm",
    "llm2vae",
]


@attrs.define(slots=False)
class DMD2OptimizerConfig:
    """Unified optimizer config for DMD2 student (net) and critic (fake_score) networks.

    Both fields are LazyDicts wrapping ``build_optimizer`` — they are instantiated at training
    time via ``lazy_instantiate(config.net, model=self.net)``.  Experiment overrides go under
    ``optimizer=dict(net=dict(...), fake_score=dict(...))``.
    """

    net: LazyDict = L(build_optimizer)(
        model=None,
        optimizer_type="FusedAdam",
        lr=1e-6,
        weight_decay=0.01,
        betas=[0.9, 0.99],
        fused=True,
        eps=1e-8,
        keys_to_select=_STANDARD_KEYS_TO_SELECT,
        lr_multipliers={},
    )
    fake_score: LazyDict = L(build_optimizer)(
        model=None,
        optimizer_type="FusedAdam",
        lr=2e-7,
        weight_decay=0.01,
        betas=[0.0, 0.999],
        fused=True,
        eps=1e-8,
        keys_to_select=_STANDARD_KEYS_TO_SELECT,
        lr_multipliers={},
    )


def register_dmd2_optimizer() -> None:
    cs = ConfigStore.instance()
    cs.store(group="optimizer", package="optimizer", name="dmd2", node=DMD2OptimizerConfig())


@attrs.define(slots=False)
class DMD2RFConfig(OmniMoTModelConfig):
    """
    Config for DMD2RF model.

    Inherits all base fields from ``OmniMoTModelConfig`` and adds
    DMD2RF-specific knobs (teacher/fake-score, optimizers, sampler schedule, etc.).
    """

    # The student's vlm_config is inherited from OmniMoTModelConfig (model.config.vlm_config).
    # Teacher and fake-score use separate VLMConfig instances because they may differ in size
    # or checkpoint path from the student; all three default with pretrained_weights.enabled=False
    # since DMD2 initialises them from teacher_load_from rather than the standard pretrain path.
    vlm_config: VLMConfig = VLMConfig(pretrained_weights=PretrainedWeightsConfig(enabled=False))
    vlm_config_teacher: VLMConfig = VLMConfig(pretrained_weights=PretrainedWeightsConfig(enabled=False))
    vlm_config_fake_score: VLMConfig = VLMConfig(pretrained_weights=PretrainedWeightsConfig(enabled=False))

    # ---------------- Fixed-step sampler schedule ----------------
    # Controls the discrete sigma schedule used for student inference and for sampling
    # the training noise level in _sample_student_sigma.  See FixedStepSamplerConfig.
    fixed_step_sampler_config: FixedStepSamplerConfig = FixedStepSamplerConfig()

    # ---------------- Distillation / loss scheduling ----------------
    loss_scale_fake_score: float = 1.0
    loss_scale_sid: float = 1.0
    # "x0" uses the clean-data VSD direction.
    # "velocity" applies the raw RF velocity direction to the generated x0 tensor.
    vsd_gradient_space: Literal["x0", "velocity"] = "x0"
    # "mean" uses active-element mean loss. "sum" uses half-MSE sum.
    # "per_instance_sum" uses per-instance active-element sum without the half-MSE factor.
    vsd_loss_reduction: Literal["mean", "sum", "per_instance_sum"] = "mean"
    # "active_mean" uses fake-score FM loss summed over generated elements and
    # divided by active element count. "per_instance_sum" sums generated elements per instance.
    fake_score_loss_reduction: Literal["active_mean", "per_instance_sum"] = "active_mean"
    student_update_freq: int = 5
    teacher_guidance: float = 1.0
    # Prompt used for the teacher CFG unconditional / negative branch. The empty
    # default preserves the existing classifier-free guidance behavior.
    teacher_negative_prompt: str = ""
    # Preserve existing DMD2 clipping behavior by default; experiments can set
    # this False for no-clipping runs.
    grad_clip: bool = True
    warmup_student_steps: int = 0  # Number of iterations of student-only phase before alternating with critic updates
    warmup_critic_steps: int = 0  # Number of iterations of critic-only phase before alternating with student updates

    # ---------------- Student simulation mode ----------------
    # "forward": single pass from a randomly sampled sigma (current default).
    # "backward": multi-step rollout from a pure-noise start (t_list[0] must be 1.0)
    # over an iteration-cycled number of t_list steps, descending to sigma=0.
    simulation_mode: Literal["forward", "backward"] = "forward"
    # Number of trailing denoising steps in backward simulation that receive gradients.
    # 1  = only the last step (low memory, matches predict2_distill).
    # -1 = all steps (true BPTT, O(N) memory).
    backward_grad_steps: int = 1

    # ---------------- Model architecture / checkpointing ----------------
    # Note: In cosmos3 we use VLMConfig in DMD2RFConfig to instantiate the network.
    # The teacher/fakescore/student use same instantiation function to build the net 3 times.
    teacher_load_from: LazyDict | None = None
    student_load_from: LazyDict | None = None
    # Load teacher ckpt and copy weights into student/fake-score nets (train-time only)
    load_teacher_weights: bool = True

    # ---------------- Enforcing value of certain fields upon creation of the config ----------------
    def __attrs_post_init__(self) -> None:
        assert not (self.warmup_student_steps > 0 and self.warmup_critic_steps > 0), (
            "Only one of warmup_student_steps and warmup_critic_steps can be nonzero, "
            f"got warmup_student_steps={self.warmup_student_steps}, warmup_critic_steps={self.warmup_critic_steps}"
        )
