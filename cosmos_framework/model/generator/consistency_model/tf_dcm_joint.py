# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in TF-dCM with one diffusion clock for RGB and LiDAR endpoints."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from typing_extensions import override

from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan
from cosmos_framework.model.generator.consistency_model.tf_dcm import (
    TeacherForcingDCMModel,
    compute_endpoint_consistency_loss,
    sample_teacher_forcing_dcm_ladder,
)
from cosmos_framework.model.generator.mot.causal_cosmos3_vfm_network import build_interactive_multiview_mask_items
from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_broadcast_tensor_list
from cosmos_framework.model.generator.utils.data_and_condition import (
    GenerationDataClean,
    GenerationDataNoised,
    _expand_per_sample_to_per_vision_item,
)
from cosmos_framework.model.generator.utils.kv_cache import TeacherForcingMemoryState


class TeacherForcingJointDCMModel(TeacherForcingDCMModel):
    """Distill both sensors while retaining the original RGB-only model class."""

    @override
    def _denoise_with_replayed_tf(
        self,
        net: torch.nn.Module,
        noisy_pack: PackedSequence,
        gen_data_clean: GenerationDataClean,
        tf_memory: TeacherForcingMemoryState,
    ) -> dict:
        """Preserve RGB and LiDAR roles in the packer's per-sample replay order."""
        if getattr(tf_memory, "selected_clean_gen_token_indexes", None) is not None:
            if getattr(noisy_pack, "teacher_forcing_original_condition_masks_sensors", None) is None:
                # Each sample stores its RGB items before its LiDAR items. A global
                # concatenation of the two modality lists would mix sample owners.
                noisy_pack.teacher_forcing_original_condition_masks_sensors = [
                    item.condition_mask.clone()
                    for sample in build_interactive_multiview_mask_items(noisy_pack)
                    for item in sample
                ]  # list[[T_sensor]]
        return super()._denoise_with_replayed_tf(net, noisy_pack, gen_data_clean, tf_memory)

    @staticmethod
    def _joint_condition_masks(packed: PackedSequence) -> dict[str, list[torch.Tensor]]:
        """Read per-item masks without folding LiDAR sweeps into camera frames."""
        masks: dict[str, list[torch.Tensor]] = {}
        for modality in ("vision", "lidar"):
            stream = getattr(packed, modality)
            if stream is not None:
                if not isinstance(stream.condition_mask, list):
                    raise TypeError(f"Joint TF-dCM requires per-item {modality} condition masks.")
                masks[modality] = stream.condition_mask  # list[[T,1,1]]
        if "vision" not in masks:
            raise ValueError("Joint TF-dCM requires RGB targets.")
        return masks

    def _joint_sigma_masks(
        self,
        sample_sigmas: torch.Tensor,  # [B,1]
        clean: GenerationDataClean,
        masks: dict[str, list[torch.Tensor]],  # modality -> list[[T,1,1]]
    ) -> dict[str, list[torch.Tensor]]:
        """Expand the same sample clock independently over each sensor's items."""
        result: dict[str, list[torch.Tensor]] = {}
        for modality, condition_masks in masks.items():
            counts = getattr(clean, f"num_{modality}_items_per_sample")
            item_sigmas = _expand_per_sample_to_per_vision_item(sample_sigmas, counts)  # [B_items,1]
            if len(condition_masks) != len(item_sigmas):
                raise ValueError(f"Joint TF-dCM {modality} item counts do not match condition masks.")
            result[modality] = self._vision_sigma_masks(item_sigmas, condition_masks)  # list[[T,1,1]]
        return result

    def _joint_noised_state(
        self,
        base: GenerationDataNoised,
        states: dict[str, list[torch.Tensor]],  # modality -> list[[1,C,T,H,W]]
        sigmas: dict[str, list[torch.Tensor]],  # modality -> list[[T,1,1]]
    ) -> GenerationDataNoised:
        """Replace endpoint states while preserving all original noise and control fields."""
        result = replace(base)
        for modality, values in states.items():
            tokens = [value.to(**self.tensor_kwargs) for value in values]  # list[[1,C,T,H,W]]
            setattr(result, f"xt_tokens_{modality}", tokens)
            setattr(result, f"sigmas_{modality}", sigmas[modality])
        return result

    def _joint_teacher_branch(
        self,
        clean_pack: PackedSequence,
        plans: list[SequencePlan],
        text: list[list[int]],
        clean: GenerationDataClean,
        noised: GenerationDataNoised,
        timesteps: torch.Tensor,  # [B,1]
        memory_info: dict[str, Any],
    ) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]]]:
        """Evaluate one teacher branch and release its clean replay cache."""
        memory = self._build_clean_tf_cache(
            net=self.net_teacher,
            packed_sequence=clean_pack,
            gen_data_clean=clean,
            memory_info=memory_info,
            detach_clean_kv=True,
        )
        packed = self._build_noisy_target_pack(plans, text, clean, noised, timesteps, memory_info)
        output = self._denoise_with_replayed_tf(self.net_teacher, packed, clean, memory)
        masks = self._joint_condition_masks(packed)
        predictions = {
            modality: [value.detach() for value in output[f"preds_{modality}"]] for modality in masks
        }  # modality -> list[[C,T,H,W]]
        return predictions, masks

    def _check_joint_masks(
        self,
        reference: dict[str, list[torch.Tensor]],  # modality -> list[[T,1,1]]
        candidate: dict[str, list[torch.Tensor]],  # modality -> list[[T,1,1]]
    ) -> None:
        if reference.keys() != candidate.keys():
            raise ValueError("Joint TF-dCM endpoint modalities differ.")
        for modality in reference:
            self._assert_matching_vision_condition_masks(reference[modality], candidate[modality])

    def _joint_endpoint_loss(
        self,
        endpoints: dict[str, list[torch.Tensor]],  # modality -> list[[1,C,T,H,W]]
        targets: dict[str, list[torch.Tensor]],  # modality -> list[[1,C,T,H,W]]
        masks: dict[str, list[torch.Tensor]],  # modality -> list[[T,1,1]]
        predictions: dict[str, Any],
        is_image_batch: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Use the configured item denominator per sensor and retain gradients for absent sensors."""
        rf_config = self.config.rectified_flow_training_config
        total = torch.zeros((), **self.tensor_kwargs_fp32)  # []
        losses: dict[str, torch.Tensor] = {}
        for modality in ("vision", "lidar"):
            values = predictions.get(f"preds_{modality}")
            if modality in masks and any(bool((mask != 1).any()) for mask in masks[modality]):
                loss, per_item = compute_endpoint_consistency_loss(
                    endpoints[modality],
                    targets[modality],
                    masks[modality],
                    detach_target=self.config.teacher_forcing_dcm_detach_target_endpoint,
                    exclude_fully_conditioned_items=rf_config.exclude_fully_conditioned_items,
                )  # [], [B_items]
                losses[f"tf_dcm_loss_{modality}_per_instance"] = per_item  # [B_items]
                losses[f"flow_matching_loss_{modality}_per_instance"] = per_item  # [B_items]
            elif values is not None:
                loss = sum(value.sum() for value in values) * 0.0  # []
            elif modality == "lidar" and self.config.lidar_state_ch is None:
                continue
            else:
                raise ValueError(f"Joint TF-dCM requires connected {modality} predictions.")
            scale = rf_config.loss_scale
            if modality == "vision" and is_image_batch and rf_config.image_loss_scale is not None:
                scale = rf_config.image_loss_scale
            if modality == "lidar" and rf_config.lidar_loss_scale is not None:
                scale = rf_config.lidar_loss_scale
            total = total + loss * scale  # []
            losses[f"tf_dcm_loss_{modality}"] = loss  # []
            losses[f"flow_matching_loss_{modality}"] = loss  # []
        for modality in ("action", "sound"):
            losses[f"flow_matching_loss_{modality}"] = torch.zeros((), **self.tensor_kwargs_fp32)  # []
        return total, losses

    @override
    def _training_step_teacher_forcing_dcm(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Advance both sensors on the teacher ODE before comparing student endpoints."""
        text, plans, clean, memory_info, resolutions, _ = self._get_training_inputs(data_batch, iteration)
        if self.config.sound_gen or self.config.action_gen:
            raise ValueError("Joint TF-dCM supports RGB/LiDAR generation with conditioning-only actions.")
        if memory_info.get("use_rolling_kv_cache", False):
            raise ValueError("Joint TF-dCM requires independent teacher and student replay caches.")
        shift = self._resolve_teacher_forcing_dcm_shift(
            clean.batch_size, clean.is_image_batch, resolutions, self._num_vision_tokens_per_sample(clean)
        )  # [B]
        flow = self.rectified_flow_image if clean.is_image_batch else self.rectified_flow_video
        timesteps, sigmas = sample_teacher_forcing_dcm_ladder(
            clean.batch_size,
            self.config.teacher_forcing_dcm_total_steps,
            self.config.teacher_forcing_dcm_skipping_interval_steps,
            shift,
            flow.noise_scheduler.config.num_train_timesteps,
            self.tensor_kwargs_fp32,
        )  # [B,N+1], [B,N+1]
        context_parallel_broadcast_tensor_list([timesteps, sigmas], getattr(self, "parallel_dims", None))
        packed = self._pack_input_sequence(
            plans,
            text,
            clean,
            timesteps[:, :1].cpu(),  # [B,1]
            skip_text_tokens=memory_info["skip_text"],
            initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
        )
        self._validate_teacher_forcing_pack(packed)
        masks = self._joint_condition_masks(packed)
        guidance = self.config.teacher_forcing_dcm_teacher_guidance
        if guidance < 1.0:
            raise ValueError("Joint TF-dCM teacher guidance must be at least one.")
        negative_text = (
            [
                list(tokens)
                for plan, tokens in zip(plans, memory_info["_tf_dcm_unconditional_text_indexes"], strict=True)
                for _ in range(len(plan.text_view_ids) if plan.text_view_ids is not None else 1)
            ]
            if guidance > 1.0
            else None
        )
        negative_pack = None
        if negative_text is not None:
            negative_pack = self._pack_input_sequence(
                plans,
                negative_text,
                clean,
                timesteps[:, :1].cpu(),  # [B,1]
                skip_text_tokens=memory_info["skip_text"],
                initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
            )
            self._validate_teacher_forcing_pack(negative_pack)
        item_sigmas = {
            modality: _expand_per_sample_to_per_vision_item(
                sigmas[:, :1], getattr(clean, f"num_{modality}_items_per_sample")
            )
            for modality in masks
        }  # modality -> [B_items,1]
        noised = self._add_noise_to_input(
            clean, packed, item_sigmas["vision"], sigmas_lidar=item_sigmas.get("lidar"), iteration=iteration
        )
        initial_sigmas = self._joint_sigma_masks(sigmas[:, :1], clean, masks)  # modality -> list[[T,1,1]]
        initial = {
            modality: [
                (1.0 - sigma.to(device=x0.device, dtype=torch.float32)) * x0.float()
                + sigma.to(device=x0.device, dtype=torch.float32) * epsilon.float()
                for x0, epsilon, sigma in zip(
                    getattr(clean, f"x0_tokens_{modality}"),
                    getattr(noised, f"epsilon_{modality}"),
                    initial_sigmas[modality],
                    strict=True,
                )
            ]
            for modality in masks
        }  # modality -> list[[1,C,T,H,W]]
        current = initial
        with torch.no_grad():
            for index in range(self.config.teacher_forcing_dcm_skipping_interval_steps):
                current_sigmas = self._joint_sigma_masks(
                    sigmas[:, index : index + 1], clean, masks
                )  # modality -> list[[T,1,1]]
                next_sigmas = self._joint_sigma_masks(
                    sigmas[:, index + 1 : index + 2], clean, masks
                )  # modality -> list[[T,1,1]]
                current_noised = self._joint_noised_state(noised, current, current_sigmas)
                velocity, teacher_masks = self._joint_teacher_branch(
                    packed, plans, text, clean, current_noised, timesteps[:, index : index + 1], memory_info
                )  # modality -> list[[C,T,H,W]], modality -> list[[T,1,1]]
                self._check_joint_masks(masks, teacher_masks)
                if negative_text is not None:
                    assert negative_pack is not None
                    negative_velocity, negative_masks = self._joint_teacher_branch(
                        negative_pack,
                        plans,
                        negative_text,
                        clean,
                        current_noised,
                        timesteps[:, index : index + 1],
                        memory_info,
                    )  # modality -> list[[C,T,H,W]], modality -> list[[T,1,1]]
                    self._check_joint_masks(masks, negative_masks)
                    velocity = {
                        modality: self._blend_teacher_cfg_vision(
                            velocity[modality], negative_velocity[modality], guidance
                        )
                        for modality in masks
                    }  # modality -> list[[C,T,H,W]]
                    del negative_velocity, negative_masks
                current = {
                    modality: self._ode_step_vision(
                        current[modality], velocity[modality], current_sigmas[modality], next_sigmas[modality]
                    )
                    for modality in masks
                }  # modality -> list[[1,C,T,H,W]]
                del current_noised, velocity, teacher_masks
        student_memory = self._build_clean_tf_cache(
            net=self.net,
            packed_sequence=packed,
            gen_data_clean=clean,
            memory_info=memory_info,
            detach_clean_kv=self.config.teacher_forcing_detach_clean_kv,
        )
        self._replace_clean_with_noised(packed, noised)
        packed.to_cuda()
        prediction = self._denoise_with_replayed_tf(self.net, packed, clean, student_memory)
        endpoints = {
            modality: self._velocity_to_x0(initial[modality], prediction[f"preds_{modality}"], initial_sigmas[modality])
            for modality in masks
        }  # modality -> list[[1,C,T,H,W]]
        del negative_pack
        final_sigmas = self._joint_sigma_masks(sigmas[:, -1:], clean, masks)  # modality -> list[[T,1,1]]
        final_noised = self._joint_noised_state(noised, current, final_sigmas)
        final_pack = self._build_noisy_target_pack(plans, text, clean, final_noised, timesteps[:, -1:], memory_info)
        self._check_joint_masks(masks, self._joint_condition_masks(final_pack))
        with torch.no_grad():
            final_prediction = self._denoise_with_replayed_tf(self.net, final_pack, clean, student_memory)
            targets = {
                modality: self._velocity_to_x0(
                    current[modality], final_prediction[f"preds_{modality}"], final_sigmas[modality]
                )
                for modality in masks
            }  # modality -> list[[1,C,T,H,W]]
        loss, losses = self._joint_endpoint_loss(
            endpoints, targets, masks, prediction, clean.is_image_batch
        )  # [], dict
        auxiliary, auxiliary_losses = self._compute_tf_dcm_aux_losses(prediction)  # [], dict
        loss = loss + auxiliary  # []
        losses.update(auxiliary_losses)
        output = {
            "x0": clean.x0_tokens_vision,
            "xt": final_noised.xt_tokens_vision,
            "sigma": _expand_per_sample_to_per_vision_item(
                sigmas[:, -1:], clean.num_vision_items_per_sample
            ),  # [B_items,1]
            "model_pred": final_prediction["preds_vision"],
            "condition_mask_vision": masks["vision"],
            "condition_mask_lidar": masks.get("lidar"),
            "condition_mask_action": final_pack.action.condition_mask if final_pack.action else None,
            "und_token_length": final_pack.text_indexes.shape[0],
            "gen_token_length": final_pack.sequence_length - final_pack.text_indexes.shape[0],
            "vision_token_length": len(final_pack.vision.sequence_indexes),
            "lidar_token_length": len(final_pack.lidar.sequence_indexes) if final_pack.lidar else 0,
            "action_token_length": len(final_pack.action.sequence_indexes) if final_pack.action else 0,
            "sound_token_length": 0,
            "is_image_batch": clean.is_image_batch,
            "batch_size": clean.batch_size,
            "split_lens": final_pack.split_lens,
            "attn_modes": final_pack.attn_modes,
            "tf_dcm_timesteps": timesteps,
            "tf_dcm_sigmas": sigmas,
            "tf_dcm_teacher_guidance": torch.tensor(guidance, **self.tensor_kwargs_fp32),  # []
            **losses,
        }
        return output, loss
