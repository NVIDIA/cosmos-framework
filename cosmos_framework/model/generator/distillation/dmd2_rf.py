# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DMD2 model using Rectified Flow parameterization for Cosmos3.

Supports three student simulation modes (``simulation_mode`` config field):

- ``"forward"`` (default): single-denoising simulation. Clean data examples are noised to a
  level sampled from ``fixed_step_sampler_config.t_list``, and the student predicts the clean
  example (via RF velocity) from this noised example in one denoising operation. The VSD /
  fake-score losses are computed on re-noised student outputs.

- ``"forward_random"``: single-denoising simulation from continuously sampled random noise.
  The VSD / fake-score losses are computed on re-noised student outputs.

- ``"backward"``: multi-step rollout. The student always denoises from a pure-noise start
  (``t_list[0]`` must be 1.0) toward 0, over an iteration-cycled number of ``t_list`` steps (schedule
  prefix). No clean data is blended into the start state. Gradients flow through the last
  ``backward_grad_steps`` student forward passes (1 = last step only, −1 = full BPTT through all steps).

Reference: https://arxiv.org/abs/2405.14867
"""

from __future__ import annotations

import collections
import json
import time
from typing import Any, Mapping, Optional, cast

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader  # noqa: F401 - used after OSS release transform
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader  # noqa: F401 - used after OSS release transform
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate
from cosmos_framework.utils import log, misc
from cosmos_framework.model.generator.distillation.common_loss import (
    variational_score_distillation_loss,
    variational_score_distillation_loss_from_gradient,
)
from cosmos_framework.configs.base.defaults.model_config import FixedStepSamplerConfig
from cosmos_framework.configs.base.experiment.distillation.dmd2_config import DMD2RFConfig
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.distillation.dmd_common import (
    DMDTrainingBase,
    get_data_resolutions,
    vision_token_count,
)
from cosmos_framework.model.generator.distillation.dmd_common import (
    generated_vision_item_indices as _generated_vision_item_indices,
)
from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_broadcast_tensor_list
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.model.generator.utils.data_and_condition import (
    GenerationDataClean,
    GenerationDataNoised,
    _expand_per_sample_to_per_vision_item,
)
from cosmos_framework.model.generator.utils.safetensors_loader import (
    load_language_model as load_language_model_safetensors,
)
from cosmos_framework.data.generator.sequence_packing import (
    PackedSequence,
    SequencePlan,
    build_sequence_plans_from_data_batch,
)
from cosmos_framework.data.generator.sequence_packing.modality import add_special_tokens

__all__: tuple[str, ...] = ("DMD2RFModel",)

_MAX_NUM_TEXT_TOKENS = 4096


def _select_items_by_indices(items: list[Any], indices: list[int]) -> list[Any]:
    return [items[idx] for idx in indices]


class DMD2RFModel(DMDTrainingBase, OmniMoTModel):
    """
    DMD2 distillation model using Rectified Flow parameterization.

    https://arxiv.org/abs/2405.14867
    """

    # ------------------------ Initialization & configuration ------------------------

    def __init__(self, config: DMD2RFConfig) -> None:
        """
        Args:
            config (DMD2RFConfig): The configuration for the DMD model
        """
        # Keep training-only networks out of the nn.Module registry. This matches
        # self-forcing DMD and keeps inference state dicts student-only.
        self._net_teacher_holder: list[torch.nn.Module] = []
        self._net_fake_score_holder: list[torch.nn.Module] = []
        self._validate_asymmetric_initialization_config(config)
        is_inference_mode = bool(getattr(config.parallelism, "enable_inference_mode", False))
        if not is_inference_mode and not config.load_student_pretrained_weights:
            config.vlm_config.pretrained_weights.enabled = False
        super().__init__(config)
        self.config: DMD2RFConfig = config

    @staticmethod
    def _validate_asymmetric_initialization_config(config: DMD2RFConfig) -> None:
        """Validate copy/load combinations after Hydra experiment overrides."""
        if config.copy_teacher_weights_to_student and config.load_student_pretrained_weights:
            raise ValueError(
                "copy_teacher_weights_to_student and load_student_pretrained_weights are mutually exclusive."
            )
        if config.copy_teacher_weights_to_ema and not config.copy_teacher_weights_to_student:
            raise ValueError("copy_teacher_weights_to_ema requires copy_teacher_weights_to_student.")
        if config.copy_teacher_weights_to_fake_score and config.load_fake_score_pretrained_weights:
            raise ValueError(
                "copy_teacher_weights_to_fake_score and load_fake_score_pretrained_weights are mutually exclusive."
            )
        if (
            config.vlm_config_teacher.model_name != config.vlm_config.model_name
            and not config.retokenize_text_for_teacher
        ):
            raise ValueError("retokenize_text_for_teacher must be True when teacher and student VLMs differ.")
        if (
            config.vlm_config_fake_score.model_name != config.vlm_config.model_name
            and not config.retokenize_text_for_fake_score
        ):
            raise ValueError("retokenize_text_for_fake_score must be True when fake-score and student VLMs differ.")

    @property
    def net_teacher(self) -> torch.nn.Module:
        """Frozen teacher kept out of the module registry / inference state dict."""
        if not self._net_teacher_holder:
            raise AttributeError("DMD2RF teacher has not been initialized.")
        return self._net_teacher_holder[0]

    @property
    def net_fake_score(self) -> torch.nn.Module:
        """Fake-score critic kept out of the registry; persisted by the distill checkpointer."""
        if not self._net_fake_score_holder:
            raise AttributeError("DMD2RF fake-score network has not been initialized.")
        return self._net_fake_score_holder[0]

    def _set_up_fixed_step_sampler(self) -> None:
        """Build the fixed-step sampler used during distilled student inference."""
        sampler_cfg = self.config.fixed_step_sampler_config
        if sampler_cfg is None:
            self.fixed_step_sampler = None
            return
        self.fixed_step_sampler = FixedStepSampler(
            t_list=list(sampler_cfg.t_list),
            sample_type=sampler_cfg.sample_type,
            num_train_timesteps=float(self.config.rectified_flow_inference_config.num_train_timesteps),
        )

    def _require_fixed_step_sampler_config(self) -> FixedStepSamplerConfig:
        sampler_cfg = self.config.fixed_step_sampler_config
        if sampler_cfg is None:
            raise ValueError("fixed_step_sampler_config is required for fixed-step DMD2 training simulation.")
        return sampler_cfg

    def _needs_fake_score(self) -> bool:
        """Whether this distillation variant builds and trains a fake-score net."""
        return True

    def load_pretrained_model_if_needed(self, *, has_resumable_checkpoint: bool, has_load_path: bool) -> None:
        """DMD2 manages student/fake-score pretraining inside set_up_model."""
        if self.config.load_student_pretrained_weights or self.config.load_fake_score_pretrained_weights:
            log.info("[DMD2RF] Skipping base pretrained callback; DMD2 initialization is handled in set_up_model.")
            return
        super().load_pretrained_model_if_needed(
            has_resumable_checkpoint=has_resumable_checkpoint,
            has_load_path=has_load_path,
        )

    def set_up_tokenizers(self) -> None:
        """Set up student tokenizer plus optional teacher/fake-score tokenizers."""
        super().set_up_tokenizers()
        self._vlm_tokenizers_by_net: dict[str, Any] = {"student": self.vlm_tokenizer}
        self._vlm_configs_by_net: dict[str, Any] = {"student": self.config.vlm_config}
        self._llm_special_tokens_by_net: dict[str, dict[str, int]] = {"student": self.llm_special_tokens}
        if self.config.retokenize_text_for_teacher:
            self._set_up_text_tokenizer_for_net("teacher", self.config.vlm_config_teacher)
        else:
            self._vlm_tokenizers_by_net["teacher"] = self.vlm_tokenizer
            self._vlm_configs_by_net["teacher"] = self.config.vlm_config
            self._llm_special_tokens_by_net["teacher"] = self.llm_special_tokens
        if self.config.retokenize_text_for_fake_score:
            self._set_up_text_tokenizer_for_net("fake_score", self.config.vlm_config_fake_score)
        else:
            self._vlm_tokenizers_by_net["fake_score"] = self.vlm_tokenizer
            self._vlm_configs_by_net["fake_score"] = self.config.vlm_config
            self._llm_special_tokens_by_net["fake_score"] = self.llm_special_tokens

    def _set_up_text_tokenizer_for_net(self, net_type: str, vlm_config: Any) -> None:
        """Instantiate the tokenizer used for a non-student denoiser."""
        processor = lazy_instantiate(vlm_config.tokenizer)
        tokenizer, special_tokens = add_special_tokens(processor.tokenizer)
        special_tokens["eos_token_id"] = tokenizer.eos_token_id
        self._vlm_tokenizers_by_net[net_type] = tokenizer
        self._vlm_configs_by_net[net_type] = vlm_config
        self._llm_special_tokens_by_net[net_type] = special_tokens

    def _load_pretrained_weights_for_net(self, net: torch.nn.Module, vlm_config: Any, target_name: str) -> None:
        """Load VLM pretrained weights into one DMD2 denoiser network."""
        pretrained_weights = vlm_config.pretrained_weights
        if not pretrained_weights.enabled:
            raise ValueError(f"{target_name} requested pretrained weights, but pretrained_weights.enabled is False.")
        log.info(f"Loading {target_name} reasoner pathway weights from {pretrained_weights.backbone_path}")
        load_language_model_safetensors(
            model=net.language_model,
            checkpoint_path=pretrained_weights.backbone_path,
            credential_path=pretrained_weights.credentials_path,
            parallel_dims=self.parallel_dims,
            checkpoint_format=pretrained_weights.checkpoint_format,
        )
        if self.config.diffusion_expert_config.load_weights_from_pretrained:
            log.info(f"Copying {target_name} understanding pathway weights to generation pathway.")
            net.language_model.init_moe()
        log.info(f"Successfully loaded {target_name} pretrained weights.")

    @misc.timer("DMD2RFModel: set_up_model")
    def set_up_model(self) -> None:
        t0 = time.time()
        is_inference_mode = bool(getattr(self.config.parallelism, "enable_inference_mode", False))
        # Build the student net (self.net) and shared components (tokenizer, etc.).
        super().set_up_model()
        if not is_inference_mode and self.config.load_student_pretrained_weights:
            self._load_pretrained_weights_for_net(self.net, self.config.vlm_config, "student")
            if self.config.ema.enabled:
                self._load_pretrained_weights_for_net(self.net_ema, self.config.vlm_config, "student EMA")

        if is_inference_mode:
            log.info("[DMD2RF] set_up_model: inference mode; skipping teacher and fake-score nets.")
            self.denoiser_nets = {"student": self.net}
            self._set_up_fixed_step_sampler()
            torch.cuda.empty_cache()
            log.info(f"[DMD2RF] set_up_model: inference setup done in {time.time() - t0:.1f}s")
            return

        load_teacher_weights = self.config.load_teacher_weights

        if load_teacher_weights:
            assert self.config.teacher_load_from.load_path, (
                "A pretrained teacher model checkpoint is required for distillation"
            )

        needs_fake_score = self._needs_fake_score()

        # Build teacher and optional fake-score nets by temporarily swapping self.vlm_config,
        # since build_net() reads architecture from that attribute.
        saved_vlm_config = self.vlm_config

        self.config.vlm_config_teacher.pretrained_weights.enabled = False
        self.vlm_config = self.config.vlm_config_teacher
        self._net_teacher_holder = [self.build_net(self.precision, lora_enabled=False)]

        if needs_fake_score:
            if not self.config.load_fake_score_pretrained_weights:
                self.config.vlm_config_fake_score.pretrained_weights.enabled = False
            self.vlm_config = self.config.vlm_config_fake_score
            self._net_fake_score_holder = [self.build_net(self.precision)]
        else:
            self._net_fake_score_holder = []

        self.vlm_config = saved_vlm_config  # restore
        if needs_fake_score:
            assert self.net_fake_score is not None, "DMD2 requires a fake_score network."

        log.info("==========Instantiating networks...==========")
        log.info(f"net: {self.net}")
        log.info(f"net_teacher: {self.net_teacher}")
        if needs_fake_score:
            log.info(f"net_fake_score: {self.net_fake_score}")

        # Load the teacher checkpoint, then copy its weights into the student and
        # optional fake-score nets as the warm-start for distillation.
        if load_teacher_weights:
            self._load_checkpoint_to_net(
                self.net_teacher,
                self.config.teacher_load_from.load_path,
                credential_path=self.config.teacher_load_from.credentials,
            )

            if self.config.copy_teacher_weights_to_student:
                self._copy_teacher_weights(target_net=self.net, target_name="student")
            if needs_fake_score and self.config.copy_teacher_weights_to_fake_score:
                self._copy_teacher_weights(target_net=self.net_fake_score, target_name="fake score")

            if self.config.ema.enabled and self.config.copy_teacher_weights_to_ema:
                self.net_ema_worker.copy_to(src_model=self.net_teacher, tgt_model=self.net_ema)

        if needs_fake_score and self.config.load_fake_score_pretrained_weights:
            self._load_pretrained_weights_for_net(self.net_fake_score, self.config.vlm_config_fake_score, "fake score")

        # Optionally override the student with a dedicated student checkpoint
        # (e.g. resuming a distillation run from a mid-training snapshot).
        if self.config.student_load_from and self.config.student_load_from.load_path:
            self._load_checkpoint_to_net(
                self.net,
                self.config.student_load_from.load_path,
                credential_path=self.config.student_load_from.credentials,
            )
            if self.config.ema.enabled:
                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

        # Freeze the teacher — it is used only for inference during training.
        # Set eval() so dropout / norm running stats stay in inference mode even
        # when the trainer recursively calls model.train() at the start of each step.
        log.info("[DMD2RF] set_up_model: freezing teacher net ...")
        self.net_teacher.eval().requires_grad_(False)

        # Register all denoiser nets in a dict for easy lookup by name in _pack_and_denoise.
        self.denoiser_nets = {
            "teacher": self.net_teacher,
            "student": self.net,
        }
        if needs_fake_score:
            self.denoiser_nets["fake_score"] = self.net_fake_score

        self._set_up_fixed_step_sampler()

        torch.cuda.empty_cache()
        log.info(f"[DMD2RF] set_up_model: all done in {time.time() - t0:.1f}s")

    def _copy_teacher_weights(self, target_net: torch.nn.Module, target_name: str) -> None:
        """Copy teacher weights into a target network, with logging on key mismatches.

        Args:
            target_net: The target network to copy weights to.
            target_name: The name of the target network.
        """
        log.info(f"==========Copying teacher weights to {target_name} net==========")
        to_load = {k: v for k, v in self.net_teacher.state_dict().items() if not k.endswith("_extra_state")}
        key_match_status = target_net.load_state_dict(to_load, strict=False)
        missing_all = [k for k in key_match_status.missing_keys if not k.endswith("_extra_state")]
        missing = [k for k in missing_all if ".lora_" not in k]
        missing_lora = [k for k in missing_all if ".lora_" in k]
        unexpected = [k for k in key_match_status.unexpected_keys if not k.endswith("_extra_state")]
        if missing or unexpected:
            log.warning(f"==========teacher -> {target_name}: Missing: {missing[:10]}, Unexpected: {unexpected}")
        elif missing_lora:
            log.info(
                f"==========teacher -> {target_name}: Base keys matched successfully; "
                f"left {len(missing_lora)} LoRA adapter keys initialized."
            )
        if not missing_all and not unexpected:
            log.info(f"==========teacher -> {target_name}: All keys matched successfully.")

    def _load_checkpoint_to_net(
        self,
        net: torch.nn.Module,
        ckpt_path: str,
        prefix: str = "net_ema",
        credential_path: str | None = None,
    ) -> None:
        """Load a DCP checkpoint into a single network."""

        # Open the checkpoint storage and snapshot the model's current state dict.
        # Infer the key prefix from the checkpoint path: ".dcp/model" layouts use "net",
        # everything else (the default) uses "net_ema".
        storage_reader = (
            S3StorageReader(credential_path=credential_path or "", path=ckpt_path)
            if ckpt_path.startswith("s3://")
            else FileSystemReader(ckpt_path)
        )
        if ckpt_path.endswith(".dcp/model"):
            prefix = "net"
        _state_dict = get_model_state_dict(net)

        # Compare checkpoint keys against the (prefixed) model keys to surface
        # missing / unexpected keys before attempting to load.
        metadata = storage_reader.read_metadata()
        checkpoint_keys = metadata.state_dict_metadata.keys()

        model_keys = set(_state_dict.keys())
        prefixed_model_keys = {f"{prefix}.{k}" for k in model_keys}

        missing_keys = prefixed_model_keys - checkpoint_keys
        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")

        # Filter out the complementary prefix ("net." when loading "net_ema." and vice versa)
        # and _extra_state keys, which are TE metadata not present in all checkpoints.
        unexpected_keys = checkpoint_keys - prefixed_model_keys
        assert prefix in ["net", "net_ema"], "prefix must be either net or net_ema"
        if prefix == "net_ema":
            unexpected_keys = [k for k in unexpected_keys if "net." not in k]
        else:
            unexpected_keys = [k for k in unexpected_keys if "net_ema." not in k]
        log.warning("Ignoring _extra_state keys..")
        unexpected_keys = [k for k in unexpected_keys if "_extra_state" not in k]
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        if not missing_keys and not unexpected_keys:
            log.info("All keys matched successfully.")

        # DCP requires keys to be stored under their prefixed names. Build a
        # prefixed view of the state dict, load into it, then strip the prefix back.
        _new_state_dict = collections.OrderedDict()
        for k in _state_dict.keys():
            _new_state_dict[f"{prefix}.{k}"] = _state_dict[k]
        dcp.load(_new_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True))
        for k in _state_dict.keys():
            _state_dict[k] = _new_state_dict[f"{prefix}.{k}"]

        # Apply the loaded weights to the network (non-strict to tolerate minor mismatches).
        log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=False)))
        del _state_dict, _new_state_dict

    # ------------------------ Critic iteration ------------------------

    def get_critic_iteration(self, iteration: int) -> int:
        """Effective critic (fake-score) update index after removing student updates."""
        return iteration - self.get_student_iteration(iteration) - 1

    # ------------------------ Helper methods and utils for callbacks ------------------------


    @staticmethod
    def _flow_matching_per_instance_sum_loss(
        pred: list[torch.Tensor],
        target: list[torch.Tensor],
        condition_mask: list[torch.Tensor],
        has_valid_tokens: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fake-score FM loss as per-instance active-element sum."""
        if not has_valid_tokens:
            dummy_loss = 0.0 * sum(p.sum() for p in pred)  # scalar
            return dummy_loss, dummy_loss.unsqueeze(0)  # scalar, [1]

        per_instance_losses = []
        for i in range(len(pred)):
            sqerr_i = (pred[i] - target[i]) ** 2  # [sample_shape]
            noisy_mask_i = 1.0 - condition_mask[i]  # [T_i,...]
            loss_i = (sqerr_i * noisy_mask_i).sum()  # scalar
            per_instance_losses.append(loss_i)

        per_instance_loss = torch.stack(per_instance_losses)  # [B]
        return per_instance_loss.mean(), per_instance_loss  # scalar, [B]

    def _get_vision_noise_sampling_metadata(
        self,
        data_batch: dict[str, Any],
        gen_data_clean: GenerationDataClean,
    ) -> tuple[list[str], list[int]]:
        """Return per-sample resolution and token count for RF noise sampling."""
        data_resolutions = get_data_resolutions(data_batch, gen_data_clean.batch_size, self.config.resolution)
        num_tokens_per_vision_item = [vision_token_count(x) for x in gen_data_clean.x0_tokens_vision]
        generated_vision_item_indices = _generated_vision_item_indices(
            gen_data_clean.num_vision_items_per_sample,
            len(num_tokens_per_vision_item),
        )
        num_tokens_per_sample = [num_tokens_per_vision_item[idx] for idx in generated_vision_item_indices]
        return data_resolutions, num_tokens_per_sample

    def _sample_critic_noise_level(
        self,
        gen_data_student: GenerationDataClean,
        resolutions: list[str],
        num_tokens_per_sample: list[int],
        generated_item_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:  # [B,T_vis], [B,T_vis]
        """Sample critic noise using rollout lengths and the caller's pre-rollout token counts."""
        num_frames_per_item = [x.shape[2] for x in gen_data_student.x0_tokens_vision]
        timesteps, sigmas = self._get_train_noise_level_vision(
            batch_size=gen_data_student.batch_size,
            is_image_batch=gen_data_student.is_image_batch,
            num_vision_latent_frames=[num_frames_per_item[idx] for idx in generated_item_indices],
            resolutions=resolutions,
            num_tokens=num_tokens_per_sample,
        )  # [B,T_vis], [B,T_vis]

        # Broadcast timesteps/sigmas across CP group to ensure consistency.
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            timesteps = timesteps.contiguous()  # [B,T_vis]
            sigmas = sigmas.contiguous()  # [B,T_vis]
            context_parallel_broadcast_tensor_list([timesteps, sigmas], self.parallel_dims)
        return timesteps, sigmas

    # ------------------------ Checkpointing helpers ------------------------

    def model_dict(self) -> dict[str, torch.nn.Module]:
        return {
            "net": self.net,
            "fake_score": self.net_fake_score,
        }

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        """Load weights for student/EMA (via base class) and for fake_score.

        ``net_fake_score`` is held outside the module registry, so route its keys
        to the fake-score net explicitly and keep them out of the base model load.
        Strictness is enforced by the distillation checkpointer wrapper using the
        returned missing/unexpected keys.
        """
        strict_requested = strict
        main_state_dict = collections.OrderedDict(
            (key, value) for key, value in state_dict.items() if not key.startswith("net_fake_score.")
        )
        base_results: _IncompatibleKeys = super().load_state_dict(main_state_dict, strict=False, assign=assign)

        # Partition the flat state dict by prefix, stripping the fake-score prefix
        # so fake_score.load_state_dict sees bare keys.
        fake_score_state_dict = collections.OrderedDict()

        for k, v in state_dict.items():
            if k.startswith("net_fake_score."):
                fake_score_state_dict[k.replace("net_fake_score.", "")] = v

        # Accumulate missing/unexpected keys from fake_score so the caller gets a
        # unified report matching the contract of nn.Module.load_state_dict.
        missing_keys: list[str] = list(base_results.missing_keys)
        unexpected_keys: list[str] = list(base_results.unexpected_keys)

        fake_score_holder = getattr(self, "_net_fake_score_holder", [])
        is_inference_mode = bool(getattr(self.config.parallelism, "enable_inference_mode", False))
        if fake_score_state_dict and fake_score_holder:
            fake_score_results: _IncompatibleKeys = self.net_fake_score.load_state_dict(
                fake_score_state_dict, strict=False, assign=assign
            )
            missing_keys += [f"net_fake_score.{key}" for key in fake_score_results.missing_keys]
            unexpected_keys += [f"net_fake_score.{key}" for key in fake_score_results.unexpected_keys]
        elif fake_score_state_dict and is_inference_mode:
            log.info("[DMD2RF] load_state_dict: ignoring net_fake_score.* keys in inference mode.")
        elif fake_score_state_dict:
            unexpected_keys += [f"net_fake_score.{key}" for key in fake_score_state_dict]

        if strict_requested and (missing_keys or unexpected_keys):
            log.warning(
                "DMD2RFModel.load_state_dict received strict=True, but this loader returns incompatible keys "
                "for the distillation checkpointer to validate instead of raising directly. "
                f"Missing: {missing_keys[:20]}, Unexpected: {unexpected_keys[:20]}"
            )

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

    # ------------------------ Helpers for DMD2 training ------------------------

    @staticmethod
    def _velocity_to_x0(
        x_t: list[torch.Tensor],
        v_pred: list[torch.Tensor],
        sigma: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Convert velocity predictions to x0 (clean-data) predictions per sample.

        Under rectified-flow interpolation ``x_t = (1 - sigma) * x0 + sigma * eps`` and ``v = eps - x0``,
        we have ``x0 = x_t - sigma * v``.

        All inputs are lists of per-sample tensors (variable shapes across samples).

        Args:
            x_t: Noisy inputs, each ``(C, T_i, H_i, W_i)``.
            v_pred: Predicted velocities, same shapes as *x_t*.
            sigma: Per-sample noise levels, each ``(T_i, 1, 1)`` with zeros for conditioned frames.
                Broadcasts with token shapes.

        Returns:
            Per-sample x0 predictions with the same shapes as *x_t*.
        """
        return [x_t[i] - sigma[i] * v_pred[i] for i in range(len(x_t))]

    def _sample_student_sigma(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """Sample a noise level for the student from the discrete sigma schedule.

        Randomly picks one of the predefined sigma values
        that define the student's multi-step schedule.

        Returns:
            Tensor of shape ``(batch_size, 1)`` with sigma values in ``[0, 1]``.
        """
        t_list = DMD2RFModel._require_fixed_step_sampler_config(self).t_list
        t_tensor = torch.tensor(t_list, **self.tensor_kwargs_fp32)  # [len(t_list)]
        ids = torch.randint(0, len(t_list), (batch_size,), device=t_tensor.device)  # [B]
        return t_tensor[ids].unsqueeze(1)  # [B,1]

    def _sample_random_student_sigma(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """Sample a continuous random student noise level for capacity-distillation DMD2."""
        return torch.rand((batch_size, 1), **self.tensor_kwargs_fp32)  # [B,1]

    def _pack_and_denoise(
        self,
        gen_data_clean: GenerationDataClean,
        gen_data_noised: GenerationDataNoised,
        timesteps: torch.Tensor,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        net_type: str,
    ) -> dict[str, torch.Tensor | object]:
        """Pack a sequence and run a denoiser network.

        Args:
            gen_data_clean: Original clean data (provides metadata: ``raw_state_*``, ``fps_*``,
                ``is_image_batch``, ``num_vision_items_per_sample``).
            gen_data_noised: Pre-computed noised data for all modalities (provides
                ``xt_tokens_*`` for packing and token injection).
            timesteps: RF timesteps ``(B, 1)`` in ``[0, num_train_timesteps]`` range.
            input_text_indexes: Tokenized captions per sample.
            sequence_plans: Conditioning structure per sample.
            net_type: One of ``"student"``, ``"teacher"``, ``"fake_score"``.

        Returns:
            Network output dict (e.g. ``{"preds_vision": [...]}``) .
        """
        # Build a GenerationDataClean proxy using CPU versions of the noised tokens
        # so that _pack_input_sequence can determine shapes and conditioning structure.
        # The actual token values are overwritten below with the CUDA tensors.
        gen_data_for_packing = GenerationDataClean(
            batch_size=gen_data_clean.batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=[vt.cpu() for vt in gen_data_noised.xt_tokens_vision],
            fps_vision=gen_data_clean.fps_vision,
            temporal_positions_vision=gen_data_clean.temporal_positions_vision,
            num_vision_items_per_sample=gen_data_clean.num_vision_items_per_sample,
            raw_state_sound=gen_data_clean.raw_state_sound,
            x0_tokens_sound=(
                [st.cpu() for st in gen_data_noised.xt_tokens_sound]
                if gen_data_noised.xt_tokens_sound is not None
                else None
            ),
            fps_sound=gen_data_clean.fps_sound,
            raw_state_action=gen_data_clean.raw_state_action,
            x0_tokens_action=(
                [at.cpu() for at in gen_data_noised.xt_tokens_action]
                if gen_data_noised.xt_tokens_action is not None
                else None
            ),
            fps_action=gen_data_clean.fps_action,
            action_domain_id=gen_data_clean.action_domain_id,
        )

        saved_llm_special_tokens = self.llm_special_tokens
        self.llm_special_tokens = self._llm_special_tokens_by_net[net_type]
        try:
            packed_sequence = self._pack_input_sequence(
                sequence_plans, input_text_indexes, gen_data_for_packing, timesteps.cpu()
            )
        finally:
            self.llm_special_tokens = saved_llm_special_tokens

        # Overwrite packed tokens with the actual noised values (already on CUDA)
        assert packed_sequence.vision is not None, "Packed vision data is required"
        packed_sequence.vision.tokens = gen_data_noised.xt_tokens_vision

        packed_sequence.to_cuda()

        return self.denoise(
            net=self.denoiser_nets[net_type],
            data_batch_packed=packed_sequence,
        )

    def _gen_data_from_student(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        iteration: int,
    ) -> tuple[GenerationDataClean, PackedSequence, torch.Tensor, dict]:
        """Generate x0 from the student network.

        Dispatches to :meth:`_forward_simulation` or :meth:`_backward_simulation`
        based on ``config.simulation_mode``. ``iteration`` drives the rollout-length cycle
        in backward simulation (ignored by forward simulation).

        Returns:
            Tuple of:
            - ``gen_data_clean_student``: :class:`GenerationDataClean` with student-generated x0
              for all modalities (``x0_tokens_*``).
            - ``packed_student``: :class:`PackedSequence` from the student forward pass; provides
              per-modality ``condition_mask`` used for re-noising and loss computation.
            - ``sigma_student``: The student sigma ``(B, 1)`` used.
            - ``out_student``: Raw network output dict; kept for FSDP dummy terms when a
              modality has no data in the current batch.
        """
        if self.config.simulation_mode in ("forward", "forward_random"):
            return self._forward_simulation(input_text_indexes, sequence_plans, gen_data_clean)
        elif self.config.simulation_mode == "backward":
            return self._backward_simulation(input_text_indexes, sequence_plans, gen_data_clean, iteration)
        else:
            raise NotImplementedError

    def _forward_simulation(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
    ) -> tuple[GenerationDataClean, PackedSequence, torch.Tensor, dict]:
        """Generate x0 via a single student forward pass.

        ``simulation_mode="forward"`` samples sigma from the fixed-step schedule.
        ``simulation_mode="forward_random"`` samples continuous random sigma. Both modes add
        noise to clean data, run the student once, and convert the velocity prediction to x0.
        No gradient flows through any re-noising step; all gradient signal is carried by this
        single forward pass.

        In ``forward`` mode, single-step distillation sees near-pure noise
        (``sigma = t_list[0]``), while multi-step distillation samples sigma from ``t_list``.
        In ``forward_random`` mode, the student sees real data noised to continuous random sigma.
        """
        B = gen_data_clean.batch_size

        # Sample student noise level
        if self.config.simulation_mode == "forward_random":
            sigma_student = self._sample_random_student_sigma(B)  # [B,1]
        else:
            sigma_student = self._sample_student_sigma(B)  # [B,1]
        context_parallel_broadcast_tensor_list([sigma_student], getattr(self, "parallel_dims", None))
        num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps
        timesteps_student = sigma_student * num_train_timesteps  # [B,1] discrete in [0, num_train_timesteps]
        sigmas_student_vision = _expand_per_sample_to_per_vision_item(
            sigma_student, gen_data_clean.num_vision_items_per_sample
        )  # [B_items,1]

        # Pack sequence and add noise (following the standard MoT training pipeline)
        packed_sequence = self._pack_input_sequence(
            sequence_plans, input_text_indexes, gen_data_clean, timesteps_student.cpu()
        )
        sigmas_action_dense = None
        sigmas_sound_dense = None
        gen_data_noised = self._add_noise_to_input(
            gen_data_clean,
            packed_sequence,
            sigmas_student_vision,
            sigmas_action=sigmas_action_dense,
            sigmas_sound=sigmas_sound_dense,
        )
        self._replace_clean_with_noised(packed_sequence, gen_data_noised)
        packed_sequence.to_cuda()

        # Student forward pass
        assert packed_sequence.vision is not None
        out_student = self.denoise(
            net=self.denoiser_nets["student"],
            data_batch_packed=packed_sequence,
        )

        # Vision x0: zero out conditioned frames so their x0 ≈ xt (clean)
        condition_mask_vision = packed_sequence.vision.condition_mask  # list of [T_i,1,1]
        assert isinstance(condition_mask_vision, list)
        num_vision_items = len(condition_mask_vision)
        sigma_vision = [
            sigmas_student_vision[i].view(1, 1, 1) * (1.0 - condition_mask_vision[i]) for i in range(num_vision_items)
        ]  # list of [T_i,1,1]
        xt_vision = [xt.to(**self.tensor_kwargs) for xt in gen_data_noised.xt_tokens_vision]  # list of [C,T_i,H_i,W_i]
        gen_x0_vision = self._velocity_to_x0(
            xt_vision, out_student["preds_vision"], sigma_vision
        )  # list of [C,T_i,H_i,W_i]

        # Action x0 (when action_gen=True and batch contains action data)
        gen_x0_action = None

        # Sound x0 (when sound_gen=True and batch contains sound data)
        gen_x0_sound = None

        gen_data_clean_student = GenerationDataClean(
            batch_size=gen_data_clean.batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=gen_x0_vision,
            fps_vision=gen_data_clean.fps_vision,
            temporal_positions_vision=gen_data_clean.temporal_positions_vision,
            num_vision_items_per_sample=gen_data_clean.num_vision_items_per_sample,
            raw_state_sound=gen_data_clean.raw_state_sound,
            x0_tokens_sound=gen_x0_sound,
            fps_sound=gen_data_clean.fps_sound,
            raw_state_action=gen_data_clean.raw_state_action,
            x0_tokens_action=gen_x0_action,
            fps_action=gen_data_clean.fps_action,
            action_domain_id=gen_data_clean.action_domain_id,
        )

        return gen_data_clean_student, packed_sequence, sigma_student, out_student

    def _ode_step(
        self,
        xt: list[torch.Tensor],  # list of [C,...], current noised tokens
        v_pred: list[torch.Tensor],  # list of [C,...], velocity predictions
        noisy_mask: list[torch.Tensor],  # list of [...], broadcastable to xt; 0 for conditioned frames
        delta_sigma: float,
    ) -> list[torch.Tensor]:
        """Euler ODE step: xt_next = xt + delta_sigma * v * noisy_mask."""
        result = []
        for xt_i, v_i, mask_i in zip(xt, v_pred, noisy_mask):
            xt_next_i = xt_i + delta_sigma * v_i * mask_i
            result.append(xt_next_i.to(**self.tensor_kwargs))
        return result

    def _ode_step_per_sample(
        self,
        xt: list[torch.Tensor],  # list of [C,...], current noised tokens
        v_pred: list[torch.Tensor],  # list of [C,...], velocity predictions
        noisy_mask: list[torch.Tensor],  # list of [...], broadcastable to xt; 0 for conditioned frames
        delta_sigma: list[torch.Tensor],  # list of [...], broadcastable per sample
    ) -> list[torch.Tensor]:  # list of [C,...]
        """Euler ODE step with a different sigma delta per sample."""
        result = []
        for xt_i, v_i, mask_i, delta_i in zip(xt, v_pred, noisy_mask, delta_sigma):
            delta_i = delta_i.to(device=xt_i.device, dtype=xt_i.dtype)  # [...]
            xt_next_i = xt_i + delta_i * v_i * mask_i  # [C,...]
            result.append(xt_next_i.to(**self.tensor_kwargs))  # [C,...]
        return result

    def _sde_step(
        self,
        xt: list[torch.Tensor],  # list of [C,...], current noised tokens
        x0_pred: list[torch.Tensor],  # list of [C,...], predicted clean tokens
        noisy_mask: list[torch.Tensor],  # list of [...], 1 for generated frames, 0 for conditioned
        cond_mask: list[torch.Tensor],  # list of [...], 1 for conditioned frames, 0 for generated
        sigma_next: float,
    ) -> list[torch.Tensor]:
        """SDE re-noise step: re-noise x0 to sigma_next for generated frames; keep conditioned frames."""
        result = []
        noises = [torch.randn_like(x0_i, **self.tensor_kwargs_fp32) for x0_i in x0_pred]  # list of [C,...]
        context_parallel_broadcast_tensor_list(noises, getattr(self, "parallel_dims", None))
        for xt_i, x0_i, noisy_i, cond_i, noise_i in zip(xt, x0_pred, noisy_mask, cond_mask, noises):
            xt_next_i = (
                noisy_i * ((1.0 - sigma_next) * x0_i + sigma_next * noise_i) + cond_i.to(**self.tensor_kwargs) * xt_i
            )  # [C,...]
            xt_next_i = xt_next_i.to(**self.tensor_kwargs)  # [C,...]
            result.append(xt_next_i)
        return result

    def _backward_n_steps(self, max_n_steps: int, iteration: int) -> int:
        """Rollout length: cycle deterministically by per-net update index modulo max.

        Uses ``effective_iteration % max_simulation_steps + 1`` (not iid sampling), so every
        rollout length in ``[1, max_n_steps]`` is visited in a fixed cycle. Cycling on the
        per-net update counter (rather than the global iteration) keeps the cycle complete even
        when ``student_update_freq`` shares a common factor with ``max_n_steps``. Deterministic in
        ``iteration``, hence identical across ranks without a broadcast.
        """
        phase = self.get_phase(iteration)
        update_idx = (
            self.get_student_iteration(iteration) if phase == "student" else self.get_critic_iteration(iteration)
        )
        # get_critic_iteration is -1 at iteration 0 during critic-only warmup; clamp so the
        # cycle starts at n_steps=1 rather than wrapping to max.
        update_idx = max(update_idx, 0)
        return update_idx % max_n_steps + 1

    def _backward_simulation(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        iteration: int,
    ) -> tuple[GenerationDataClean, PackedSequence, torch.Tensor, dict]:
        """Generate x0 via a multi-step denoising rollout from pure noise.

        The rollout always starts from pure noise (``t_list[0]`` must be 1.0, so the RF seed
        ``(1 - sigma) * x0 + sigma * eps`` reduces to ``eps`` with zero clean-data leakage) and a
        iteration-cycled number of schedule steps (schedule prefix). Each step applies the ODE
        Euler step or SDE re-noising to progress toward sigma=0. Conditioned frames are held fixed
        throughout via ``noisy_mask``.

        Gradient flows through the last ``config.backward_grad_steps`` student forward passes
        (1 = last step only, matching predict2_distill; -1 = full BPTT through all steps).

        Returns the same tuple as :meth:`_forward_simulation`.
        """
        backward_grad_steps = self.config.backward_grad_steps
        assert backward_grad_steps == -1 or backward_grad_steps >= 1, (
            f"backward_grad_steps must be -1 (all steps) or >= 1, got {backward_grad_steps}"
        )

        B = gen_data_clean.batch_size
        num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps
        sampler_cfg = DMD2RFModel._require_fixed_step_sampler_config(self)
        sample_type = sampler_cfg.sample_type
        t_list = list(sampler_cfg.t_list)
        assert len(t_list) > 0, "fixed_step_sampler_config.t_list must not be empty"
        full_t_list = t_list if t_list[-1] == 0.0 else t_list + [0.0]
        assert len(full_t_list) > 1, "fixed_step_sampler_config.t_list must contain a nonzero sigma"

        # Pure-noise start contract: the schedule must begin at
        # sigma=1.0 so the rollout seed is pure noise. Under RF interpolation the seed is
        # (1 - sigma) * x0 + sigma * eps; at sigma=1.0 the clean term vanishes (seed == eps),
        # so no clean-data signal leaks into the student generation. t_list[0] < 1.0 would
        # blend x0 into the start state.
        assert abs(full_t_list[0] - 1.0) < 1e-6, (
            f"backward simulation requires t_list[0] == 1.0 for a pure-noise start, got {full_t_list[0]}"
        )

        # Iteration-cycled trajectory length: keep the pure-noise start (schedule prefix) and
        # cycle how many denoising steps to run (by per-net update index) before descending to
        # sigma=0. Fewer steps = a shorter rollout from the same pure-noise start, never a
        # half-noised real-data start.
        nonzero_levels = full_t_list[:-1]  # descending sigmas; nonzero_levels[0] == 1.0
        n_steps = self._backward_n_steps(len(nonzero_levels), iteration)
        full_t_list = nonzero_levels[:n_steps] + [0.0]
        log.debug(f"[DMD2RF] backward_simulation: n_steps={n_steps}, t_list={full_t_list}")
        grad_steps = n_steps if backward_grad_steps == -1 else backward_grad_steps

        # Pack once at sigma_max to obtain condition masks (fixed throughout the rollout)
        sigma_max = torch.full((B, 1), full_t_list[0], **self.tensor_kwargs_fp32)  # [B,1]
        timesteps_max = sigma_max * num_train_timesteps  # [B,1]
        sigmas_max_vision = _expand_per_sample_to_per_vision_item(
            sigma_max, gen_data_clean.num_vision_items_per_sample
        )  # [B_items,1]
        packed_sequence = self._pack_input_sequence(
            sequence_plans, input_text_indexes, gen_data_clean, timesteps_max.cpu()
        )
        # Initialize noised data: generation frames ≈ pure noise, conditioned frames = x0_clean
        sigmas_action_dense_init = None
        sigmas_sound_dense_init = None
        gen_data_noised = self._add_noise_to_input(
            gen_data_clean,
            packed_sequence,
            sigmas_max_vision,
            sigmas_action=sigmas_action_dense_init,
            sigmas_sound=sigmas_sound_dense_init,
        )

        assert packed_sequence.vision is not None
        condition_mask_vision = packed_sequence.vision.condition_mask  # list of [T_i,1,1]
        assert isinstance(condition_mask_vision, list)
        noisy_mask_vision = [1.0 - cm for cm in condition_mask_vision]  # list of [T_i,1,1]

        condition_mask_action = None
        condition_mask_sound = None
        noisy_mask_action = None
        cond_mask_sound = None
        noisy_mask_sound = None
        has_action = False
        has_sound = False

        out_student: dict = {}
        x0_pred_vision: list[torch.Tensor] = []
        x0_pred_action: Optional[list[torch.Tensor]] = None
        x0_pred_sound: Optional[list[torch.Tensor]] = None

        for count, (sigma_cur_val, sigma_next_val) in enumerate(zip(full_t_list[:-1], full_t_list[1:])):
            sigma_cur = torch.full((B, 1), sigma_cur_val, **self.tensor_kwargs_fp32)  # [B,1]
            timesteps_cur = sigma_cur * num_train_timesteps  # [B,1]
            sigmas_cur_vision = _expand_per_sample_to_per_vision_item(
                sigma_cur, gen_data_clean.num_vision_items_per_sample
            )  # [B_items,1]

            # Gradient enabled for the last `grad_steps` steps; all prior steps run under no_grad
            enable_grad = count >= n_steps - grad_steps
            with torch.set_grad_enabled(enable_grad):
                out_student = self._pack_and_denoise(
                    gen_data_clean,
                    gen_data_noised,
                    timesteps_cur,
                    input_text_indexes,
                    sequence_plans,
                    net_type="student",
                )

            # --- Vision x0 ---
            xt_vision = [
                xt.to(**self.tensor_kwargs) for xt in gen_data_noised.xt_tokens_vision
            ]  # list of [C,T_i,H_i,W_i]
            num_vision_items = len(noisy_mask_vision)
            sigma_vision = [
                sigmas_cur_vision[i].view(1, 1, 1) * noisy_mask_vision[i] for i in range(num_vision_items)
            ]  # list of [T_i,1,1]
            x0_pred_vision = self._velocity_to_x0(
                xt_vision, out_student["preds_vision"], sigma_vision
            )  # list of [C,T_i,H_i,W_i]

            # --- Action x0 ---
            x0_pred_action = None
            xt_action: Optional[list[torch.Tensor]] = None

            # --- Sound x0 ---
            x0_pred_sound = None
            xt_sound: Optional[list[torch.Tensor]] = None

            if sigma_next_val > 0.0:
                # Compute xt_next for the next denoising step.
                # Conditioned frames (noisy_mask=0) are kept unchanged via masking.
                if sample_type == "ode":
                    # Euler step: xt_next = xt + (sigma_next - sigma_cur) * v * noisy_mask
                    delta_sigma = sigma_next_val - sigma_cur_val  # negative scalar
                    xt_next_vision = self._ode_step(
                        xt_vision, out_student["preds_vision"], noisy_mask_vision, delta_sigma
                    )  # list of [C,T_i,H_i,W_i]
                    xt_next_action = None
                    xt_next_sound = None
                else:
                    # SDE step: re-noise x0 to sigma_next for generation frames;
                    # keep conditioned frames at their current xt value (= x0_clean).
                    xt_next_vision = self._sde_step(
                        xt_vision, x0_pred_vision, noisy_mask_vision, condition_mask_vision, sigma_next_val
                    )  # list of [C,T_i,H_i,W_i]
                    xt_next_action = None
                    xt_next_sound = None

                # Build updated GenerationDataNoised for the next step.
                # Only xt_tokens_* fields are consumed by _pack_and_denoise; sigmas_vision is
                # set for correctness but is not read by the packing logic.
                sigma_next_tensor = torch.full((B, 1), sigma_next_val, **self.tensor_kwargs_fp32)  # [B,1]
                sigmas_next_vision = _expand_per_sample_to_per_vision_item(
                    sigma_next_tensor, gen_data_clean.num_vision_items_per_sample
                )  # [B_items,1]
                sigmas_vision_next = [
                    sigmas_next_vision[i].view(1, 1, 1) * noisy_mask_vision[i] for i in range(num_vision_items)
                ]  # list of [T_i,1,1]
                sigmas_sound_next = None
                gen_data_noised = GenerationDataNoised(
                    batch_size=B,
                    epsilon_vision=gen_data_noised.epsilon_vision,
                    xt_tokens_vision=xt_next_vision,
                    vt_target_vision=gen_data_noised.vt_target_vision,
                    sigmas_vision=sigmas_vision_next,
                    epsilon_action=gen_data_noised.epsilon_action,
                    xt_tokens_action=xt_next_action,
                    vt_target_action=gen_data_noised.vt_target_action,
                    sigmas_action=gen_data_noised.sigmas_action,
                    epsilon_sound=gen_data_noised.epsilon_sound,
                    xt_tokens_sound=xt_next_sound,
                    vt_target_sound=gen_data_noised.vt_target_sound,
                    sigmas_sound=sigmas_sound_next,
                )

        gen_data_clean_student = GenerationDataClean(
            batch_size=gen_data_clean.batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=x0_pred_vision,
            fps_vision=gen_data_clean.fps_vision,
            temporal_positions_vision=gen_data_clean.temporal_positions_vision,
            num_vision_items_per_sample=gen_data_clean.num_vision_items_per_sample,
            raw_state_sound=gen_data_clean.raw_state_sound,
            x0_tokens_sound=x0_pred_sound,
            fps_sound=gen_data_clean.fps_sound,
            raw_state_action=gen_data_clean.raw_state_action,
            x0_tokens_action=x0_pred_action,
            fps_action=gen_data_clean.fps_action,
            action_domain_id=gen_data_clean.action_domain_id,
        )

        return gen_data_clean_student, packed_sequence, sigma_max, out_student

    # ------------------------ Training step ------------------------

    @staticmethod
    def _validate_vision_only_sequence_plans(sequence_plans: list[SequencePlan]) -> None:
        """Reject action or sound samples from the vision-only public DMD2 path."""
        if any(plan.has_action or plan.has_sound for plan in sequence_plans):
            raise ValueError("DMD2-RF OSS supports only vision T2I/I2V batches")

    def _setup_grad_requirements(self, iteration: int) -> None:
        # NOTE: requires_grad_(True) must be applied to the entire active network, not per-parameter,
        # because torch.compile + activation checkpointing requires all Q/K/V tensors in a layer to
        # share the same requires_grad state. This means grad is computed for ALL params (including
        # frozen backbone), making the effective clip threshold looser — but optimizer.step() only
        # updates the selected params, so weight correctness is unaffected.
        if self.get_phase(iteration) == "student":
            self.net.train().requires_grad_(True)
            self.net_fake_score.eval().requires_grad_(False)
        else:
            self.net.eval().requires_grad_(False)
            self.net_fake_score.train().requires_grad_(True)
        # Teacher must remain frozen + in eval() across all phase switches; the trainer
        # may have recursively re-enabled train() on the parent module.
        self.net_teacher.eval().requires_grad_(False)

    def _extract_raw_captions_for_retokenization(self, data_batch: dict[str, Any]) -> list[str]:
        """Read raw captions from the batch for a denoiser-specific tokenizer."""
        captions = data_batch.get(self.input_caption_key)
        if captions is None and self.input_caption_key != "ai_caption":
            captions = data_batch.get("ai_caption")
        if captions is None:
            raise KeyError(
                f"Cannot retokenize text for asymmetric DMD2: missing '{self.input_caption_key}' in data batch."
            )
        if isinstance(captions, str):
            caption_items = [captions]
        elif isinstance(captions, tuple):
            caption_items = list(captions)
        elif isinstance(captions, list):
            caption_items = []
            for caption in captions:
                if isinstance(caption, list):
                    caption_items.extend(caption)
                else:
                    caption_items.append(caption)
        else:
            raise TypeError(f"Unsupported caption batch type for retokenization: {type(captions)}")
        return [json.dumps(caption) if isinstance(caption, dict) else str(caption) for caption in caption_items]

    def _tokenize_raw_captions_for_net(
        self,
        captions: list[str],
        net_type: str,
        *,
        is_video: bool,
    ) -> list[list[int]]:
        """Tokenize raw captions with the tokenizer attached to ``net_type``."""
        tokenizer = self._vlm_tokenizers_by_net[net_type]
        vlm_config = self._vlm_configs_by_net[net_type]
        return [
            tokenize_caption(
                caption,
                tokenizer,
                is_video=is_video,
                use_system_prompt=vlm_config.use_system_prompt,
            )[:_MAX_NUM_TEXT_TOKENS]
            for caption in captions
        ]

    def _build_input_text_indexes_by_net(
        self,
        data_batch: dict[str, Any],
        student_input_text_indexes: list[list[int]],
    ) -> dict[str, list[list[int]]]:
        """Return tokenized captions for student, teacher, and fake-score denoisers."""
        input_text_indexes_by_net = {
            "student": student_input_text_indexes,
            "teacher": student_input_text_indexes,
            "fake_score": student_input_text_indexes,
        }
        if self.config.retokenize_text_for_teacher or self.config.retokenize_text_for_fake_score:
            captions = self._extract_raw_captions_for_retokenization(data_batch)
            if self.config.retokenize_text_for_teacher:
                input_text_indexes_by_net["teacher"] = self._tokenize_raw_captions_for_net(
                    captions, "teacher", is_video=False
                )
            if self.config.retokenize_text_for_fake_score:
                input_text_indexes_by_net["fake_score"] = self._tokenize_raw_captions_for_net(
                    captions, "fake_score", is_video=False
                )
        return input_text_indexes_by_net

    def training_step(self, data_batch: dict[str, Any], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Perform a single training step for DMD2 distillation.

        Delegates to :meth:`training_step_generator` (student update) or :meth:`training_step_critic`
        (fake-score update) depending on the current iteration phase.
        """
        phase = self.get_phase(iteration)

        # Freeze / unfreeze networks according to current phase
        self._setup_grad_requirements(iteration)

        # Update stats on how many videos the model has seen
        if self.parallel_dims is None or self.parallel_dims.cp_rank == 0:
            self._update_train_stats(data_batch)

        # Load, apply dropout, and tokenize input captions
        student_input_text_indexes = self._load_and_tokenize_text_data(data_batch, iteration)

        # Build sequence plans if not present. SequencePlan has the conditioning information.
        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch,
            input_video_key=self.input_video_key,
            input_image_key=self.input_image_key,
        )

        self._validate_vision_only_sequence_plans(sequence_plans)

        # Get data from raw data batch and tokenize into corresponding tokens for *generation* task
        gen_data_clean = self.get_data_and_condition(data_batch)
        data_resolutions, num_vision_tokens_per_sample = self._get_vision_noise_sampling_metadata(
            data_batch, gen_data_clean
        )
        input_text_indexes_by_net = self._build_input_text_indexes_by_net(
            data_batch,
            student_input_text_indexes,
        )

        # The noise addition and denoising happens inside respective phases
        if phase == "student":
            output_batch, loss = self.training_step_generator(
                input_text_indexes_by_net,
                sequence_plans,
                gen_data_clean,
                data_resolutions,
                num_vision_tokens_per_sample,
                iteration,
            )
        else:
            output_batch, loss = self.training_step_critic(
                input_text_indexes_by_net,
                sequence_plans,
                gen_data_clean,
                data_resolutions,
                num_vision_tokens_per_sample,
                iteration,
            )

        return output_batch, loss

    # -------------------- Generator (student) update --------------------

    def training_step_generator(
        self,
        input_text_indexes_by_net: Mapping[str, list[list[int]]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        data_resolutions: list[str],
        num_vision_tokens_per_sample: list[int],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Student update step following the DMD2 algorithm.

        1. Student generates x0 (forward simulation: noised clean data; backward simulation: multi-step rollout).
        2. Student x0 is re-noised at a random sigma for the critic networks.
        3. Teacher and fake-score predict x0 on the re-noised data.
        4. VSD loss drives the student.

        All vision data is handled as per-sample lists to support variable shapes, consistent with the MoT pipeline.

        """
        B = gen_data_clean.batch_size

        # 1. Student generates x0 (gradient flows through here)
        gen_data_student, packed_student, _, out_student = self._gen_data_from_student(
            input_text_indexes_by_net["student"],
            sequence_plans,
            gen_data_clean,
            iteration,
        )

        # 2. Sample critic noise level and re-noise student output for all modalities
        generated_vision_item_indices = _generated_vision_item_indices(
            gen_data_student.num_vision_items_per_sample,
            len(gen_data_student.x0_tokens_vision),
        )
        timesteps_critic, sigmas_critic = self._sample_critic_noise_level(
            gen_data_student,
            data_resolutions,
            num_vision_tokens_per_sample,
            generated_vision_item_indices,
        )  # [B,1], [B,1]

        sigmas_critic_vision = _expand_per_sample_to_per_vision_item(
            sigmas_critic, gen_data_student.num_vision_items_per_sample
        )  # [B_items,1]
        sigmas_action_critic = None
        sigmas_sound_critic = None
        gen_data_noised_critic = self._add_noise_to_input(
            gen_data_student,
            packed_student,
            sigmas_critic_vision,
            sigmas_action=sigmas_action_critic,
            sigmas_sound=sigmas_sound_critic,
        )

        # 3. Fake-score prediction (no grad -- no GAN in cosmos3)
        with torch.no_grad():
            out_fake = self._pack_and_denoise(
                gen_data_student,
                gen_data_noised_critic,
                timesteps_critic,
                input_text_indexes_by_net["fake_score"],
                sequence_plans,
                net_type="fake_score",
            )
            fake_v = cast(list[torch.Tensor], out_fake["preds_vision"])  # list of [C,T_i,H_i,W_i]
            fake_x0 = self._velocity_to_x0(
                gen_data_noised_critic.xt_tokens_vision,
                fake_v,
                gen_data_noised_critic.sigmas_vision,  # type: ignore[arg-type]
            )

        # 4. Teacher prediction (no grad)
        with torch.no_grad():
            out_teacher = self._pack_and_denoise(
                gen_data_student,
                gen_data_noised_critic,
                timesteps_critic,
                input_text_indexes_by_net["teacher"],
                sequence_plans,
                net_type="teacher",
            )
            teacher_v = cast(list[torch.Tensor], out_teacher["preds_vision"])  # list of [C,T_i,H_i,W_i]
            teacher_v_guided = teacher_v  # list of [C,T_i,H_i,W_i]
            teacher_x0 = self._velocity_to_x0(
                gen_data_noised_critic.xt_tokens_vision,
                teacher_v,
                gen_data_noised_critic.sigmas_vision,  # type: ignore[arg-type]
            )
            log.debug(f"student phase, after pack_and_denoise teacher, teacher_x0: {teacher_x0[0].shape}")
            # Optional classifier-free guidance for teacher
            if self.config.teacher_guidance > 1.0:
                num_vision_items = len(teacher_v)
                # Tokenize the uncond prompt with the SAME modality formatting as the batch
                # (image vs video). Hard-coding is_video=False corrupted the uncond prediction
                # on video batches, biasing the CFG-combined teacher target.
                uncond_is_video = not gen_data_student.is_image_batch
                uncond_text = [
                    tokenize_caption(
                        self.config.teacher_negative_prompt,
                        self._vlm_tokenizers_by_net["teacher"],
                        is_video=uncond_is_video,
                        use_system_prompt=self._vlm_configs_by_net["teacher"].use_system_prompt,
                    )
                    for _ in range(B)
                ]
                out_teacher_uncond = self._pack_and_denoise(
                    gen_data_student,
                    gen_data_noised_critic,
                    timesteps_critic,
                    uncond_text,
                    sequence_plans,
                    net_type="teacher",
                )
                teacher_v_uncond = cast(
                    list[torch.Tensor], out_teacher_uncond["preds_vision"]
                )  # list of [C,T_i,H_i,W_i]
                teacher_x0_uncond = self._velocity_to_x0(
                    gen_data_noised_critic.xt_tokens_vision,
                    teacher_v_uncond,
                    gen_data_noised_critic.sigmas_vision,  # type: ignore[arg-type]
                )
                teacher_v_guided = [
                    teacher_v[i] + (self.config.teacher_guidance - 1.0) * (teacher_v[i] - teacher_v_uncond[i])
                    for i in range(num_vision_items)
                ]  # list of [C,T_i,H_i,W_i]
                teacher_x0 = [
                    teacher_x0[i] + (self.config.teacher_guidance - 1.0) * (teacher_x0[i] - teacher_x0_uncond[i])
                    for i in range(num_vision_items)
                ]  # list of [C,T_i,H_i,W_i]

        # 5. VSD loss — vision (gradient flows into gen_data_student.x0_tokens_vision -> student)
        assert gen_data_student.x0_tokens_vision is not None
        assert packed_student.vision is not None
        generated_x0_tokens_vision = _select_items_by_indices(
            gen_data_student.x0_tokens_vision,
            generated_vision_item_indices,
        )  # list of [C,T_i,H_i,W_i]
        generated_teacher_x0 = _select_items_by_indices(
            teacher_x0, generated_vision_item_indices
        )  # list of [C,T_i,H_i,W_i]
        generated_fake_x0 = _select_items_by_indices(fake_x0, generated_vision_item_indices)  # list of [C,T_i,H_i,W_i]
        noisy_mask_vision = [
            (1.0 - cm).to(gen_data_student.x0_tokens_vision[0].dtype)
            for cm in _select_items_by_indices(
                packed_student.vision.condition_mask,  # type: ignore[union-attr]
                generated_vision_item_indices,
            )
        ]  # list of [T_i,1,1]
        if self.config.vsd_gradient_space == "velocity":
            vsd_grad_vision = [
                teacher_v_guided[i] - fake_v[i] for i in generated_vision_item_indices
            ]  # list of [C,T_i,H_i,W_i]
            vsd_loss = variational_score_distillation_loss_from_gradient(
                generated_x0_tokens_vision,
                vsd_grad_vision,
                weight_reference=generated_teacher_x0,
                loss_mask=noisy_mask_vision,
                reduction=self.config.vsd_loss_reduction,
            )
        elif self.config.vsd_gradient_space == "x0":
            vsd_loss = variational_score_distillation_loss(
                generated_x0_tokens_vision,
                generated_teacher_x0,
                generated_fake_x0,
                loss_mask=noisy_mask_vision,
                reduction=self.config.vsd_loss_reduction,
            )
        else:
            raise ValueError(f"Unknown vsd_gradient_space={self.config.vsd_gradient_space}")
        total_loss = self.config.loss_scale_sid * vsd_loss



        log.debug(
            f"Iteration {iteration}, student phase, after vsd_loss, "
            f"vsd loss: {vsd_loss.item():.6f}, total loss: {total_loss.item():.6f}"
        )

        output_batch: dict[str, torch.Tensor] = {
            "vsd_loss": vsd_loss.detach(),
            "total_generator_loss": total_loss.detach(),
            "sigma": sigmas_critic.detach(),
            "flow_matching_loss_vision_per_instance": vsd_loss.detach().expand(B),  # [B]
            "dmd_loss_generator": total_loss.detach(),
        }
        return output_batch, total_loss

    # -------------------- Critic (fake-score) update --------------------

    def training_step_critic(
        self,
        input_text_indexes_by_net: Mapping[str, list[list[int]]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        data_resolutions: list[str],
        num_vision_tokens_per_sample: list[int],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Fake-score update step following the DMD2 algorithm.

        1. Student generates x0 with gradients stopped.
        2. Student x0 is re-noised at a random sigma.
        3. Fake-score learns to denoise the student's distribution via flow-matching loss.

        All vision data is handled as per-sample lists to support variable shapes, consistent with the MoT pipeline.

        """
        B = gen_data_clean.batch_size

        # 1. Student generates x0 with no gradient for the critic update.
        with torch.no_grad():
            gen_data_student, packed_student, _, _ = self._gen_data_from_student(
                input_text_indexes_by_net["student"],
                sequence_plans,
                gen_data_clean,
                iteration,
            )

        # 2. Sample critic noise level and re-noise student output for all modalities
        generated_vision_item_indices = _generated_vision_item_indices(
            gen_data_student.num_vision_items_per_sample,
            len(gen_data_student.x0_tokens_vision),
        )
        timesteps_critic, sigmas_critic = self._sample_critic_noise_level(
            gen_data_student,
            data_resolutions,
            num_vision_tokens_per_sample,
            generated_vision_item_indices,
        )  # [B,1], [B,1]

        sigmas_critic_vision = _expand_per_sample_to_per_vision_item(
            sigmas_critic, gen_data_student.num_vision_items_per_sample
        )  # [B_items,1]
        timesteps_critic_vision = _expand_per_sample_to_per_vision_item(
            timesteps_critic, gen_data_student.num_vision_items_per_sample
        )  # [B_items,1]
        with torch.no_grad():
            sigmas_action_critic = None
            sigmas_sound_critic = None
            gen_data_noised_critic = self._add_noise_to_input(
                gen_data_student,
                packed_student,
                sigmas_critic_vision,
                sigmas_action=sigmas_action_critic,
                sigmas_sound=sigmas_sound_critic,
            )

        # 3. Fake-score forward (with gradient)
        out_fake = self._pack_and_denoise(
            gen_data_student,
            gen_data_noised_critic,
            timesteps_critic,
            input_text_indexes_by_net["fake_score"],
            sequence_plans,
            net_type="fake_score",
        )

        # 4. Flow-matching loss — vision. active_mean preserves sum/active_count
        # over generated frames; per_instance_sum uses per-instance active-element sum.
        rectified_flow = self.rectified_flow_image if gen_data_student.is_image_batch else self.rectified_flow_video
        fake_score_loss_reduction = self.config.fake_score_loss_reduction
        if fake_score_loss_reduction == "active_mean":
            # Use loss_per_instance (unweighted) to match flow_matching_denoising_loss semantics (no time weighting).
            _, loss_per_instance = self._compute_flow_matching_loss(
                pred=out_fake["preds_vision"],  # type: ignore[arg-type]
                target=gen_data_noised_critic.vt_target_vision,  # type: ignore[arg-type]
                condition_mask=packed_student.vision.condition_mask,  # type: ignore[union-attr]
                timesteps=timesteps_critic_vision,
                has_valid_tokens=True,
                rectified_flow=rectified_flow,
                normalize_by_active=True,
            )
        elif fake_score_loss_reduction == "per_instance_sum":
            _, loss_per_instance = self._flow_matching_per_instance_sum_loss(
                pred=out_fake["preds_vision"],  # type: ignore[arg-type]
                target=gen_data_noised_critic.vt_target_vision,  # type: ignore[arg-type]
                condition_mask=packed_student.vision.condition_mask,  # type: ignore[union-attr]
                has_valid_tokens=True,
            )
        else:
            raise ValueError(f"Unknown fake-score loss reduction: {fake_score_loss_reduction}")
        if int(loss_per_instance.shape[0]) != len(gen_data_student.x0_tokens_vision):
            raise ValueError(
                "Vision per-instance loss must align with flat vision items, "
                f"got {int(loss_per_instance.shape[0])} losses and {len(gen_data_student.x0_tokens_vision)} items."
            )
        generated_loss_per_instance = loss_per_instance[generated_vision_item_indices]  # [B_gen]
        loss_fake_score = generated_loss_per_instance.mean() * self.config.loss_scale_fake_score
        total_loss = loss_fake_score



        log.debug(
            f"Iteration {iteration}, critic phase, after flow_matching_loss, "
            f"fake_score_loss: {loss_fake_score.item():.6f}, total_loss: {total_loss.item():.6f}"
        )

        output_batch: dict[str, torch.Tensor] = {
            "fake_score_loss": loss_fake_score.detach(),
            "total_critic_loss": total_loss.detach(),
            "sigma": sigmas_critic.detach(),
            "flow_matching_loss_vision_per_instance": generated_loss_per_instance.detach(),  # [B_gen]
            "dmd_loss_critic": total_loss.detach(),
        }
        return output_batch, total_loss
