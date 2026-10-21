# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Replayed teacher-forcing discrete consistency model for interactive causal VFM."""

from __future__ import annotations

import collections
import math
from collections.abc import Mapping
from typing import Any

import attrs
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy
from typing_extensions import override

from cosmos_framework.utils import log
from cosmos_framework.model.generator.algorithm.loss.load_balancing import compute_load_balancing_loss
from cosmos_framework.checkpoint.storage_reader import get_storage_reader
from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import (
    DEFAULT_TEMPLATE as DEFAULT_DURATION_FPS_TEMPLATE,
)
from cosmos_framework.data.generator.augmentors.resolution_text_info import DEFAULT_VIDEO_TEMPLATE
from cosmos_framework.model.generator.diffusion.consistency_distillation import (
    sample_dcm_ladder as _sample_dcm_ladder,
)
from cosmos_framework.model.generator.diffusion.consistency_distillation import (
    shift_rectified_flow_time as _shift_rectified_flow_time,
)
from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_broadcast_tensor_list
from cosmos_framework.model.generator.omni_mot_causal_model import (
    OmniMoTCausalModel,
    OmniMoTCausalModelConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.model.generator.utils.data_and_condition import (
    GenerationDataClean,
    GenerationDataNoised,
    _expand_per_sample_to_per_vision_item,
)
from cosmos_framework.model.generator.utils.fsdp_reshard import defer_fsdp_reshard
from cosmos_framework.model.generator.utils.kv_cache import TeacherForcingMemoryState
from cosmos_framework.data.generator.sequence_packing import (
    PackedSequence,
    SequencePlan,
)
from cosmos_framework.data.generator.sequence_packing.autoregressive import resolve_text_system_prompt
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution


def shift_rectified_flow_time(
    sigma_unshifted: torch.Tensor,  # [B,N] or [B]
    shift: torch.Tensor | float,  # [B] or []
) -> torch.Tensor:  # [B,N] or [B]
    """Apply the Cosmos RF time-shift formula to unshifted sigma values."""
    return _shift_rectified_flow_time(sigma_unshifted, shift)  # [B,N] or [B]


def sample_teacher_forcing_dcm_ladder(
    batch_size: int,
    total_steps: int,
    skipping_interval_steps: int,
    shift: torch.Tensor | float,  # [B] or []
    max_timestep: int,
    tensor_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the shifted discrete RF ladder used by replayed TF-dCM."""
    return _sample_dcm_ladder(
        batch_size, total_steps, skipping_interval_steps, shift, max_timestep, tensor_kwargs
    )  # [B,N+1], [B,N+1]


def compute_endpoint_consistency_loss(
    endpoint: list[torch.Tensor],  # list of [C,T,H,W] or [1,C,T,H,W]
    target: list[torch.Tensor],  # list of [C,T,H,W] or [1,C,T,H,W]
    condition_mask: list[torch.Tensor],  # list of [T,1,1]
    *,
    detach_target: bool = True,
    exclude_fully_conditioned_items: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute endpoint MSE per item, optionally retaining conditioned zero-loss items in the mean."""
    losses: list[torch.Tensor] = []
    active_losses: list[torch.Tensor] = []
    for endpoint_i, target_i, condition_i in zip(endpoint, target, condition_mask, strict=True):
        all_conditioned = torch.all(condition_i == 1)  # []
        if all_conditioned.item():
            losses.append(torch.zeros((), dtype=torch.float32, device=endpoint_i.device))  # []
            continue
        endpoint_value = endpoint_i.float()  # [C,T,H,W] or [1,C,T,H,W]
        target_value = target_i.detach().float() if detach_target else target_i.float()  # [C,T,H,W] or [1,C,T,H,W]
        noisy_mask = 1.0 - condition_i.to(device=endpoint_i.device, dtype=torch.float32)  # [T,1,1]
        if endpoint_value.ndim == 5:
            noisy_mask = noisy_mask.unsqueeze(0).unsqueeze(1)  # [1,1,T,1,1]
            channel_spatial_count = endpoint_value.shape[1] * endpoint_value.shape[3] * endpoint_value.shape[4]
        elif endpoint_value.ndim == 4:
            noisy_mask = noisy_mask.unsqueeze(0)  # [1,T,1,1]
            channel_spatial_count = endpoint_value.shape[0] * endpoint_value.shape[2] * endpoint_value.shape[3]
        else:
            raise ValueError(f"TF-dCM endpoint tensors must be 4D or 5D, got shape {tuple(endpoint_i.shape)}.")
        sqerr = (endpoint_value - target_value) ** 2  # [C,T,H,W] or [1,C,T,H,W]
        item_loss = (sqerr * noisy_mask).sum() / channel_spatial_count  # []
        losses.append(item_loss)
        active_losses.append(item_loss)
    if not active_losses:
        raise ValueError("TF-dCM endpoint loss requires at least one generated vision item.")
    per_instance = torch.stack(losses)  # [B_items]
    scalar = torch.stack(active_losses).mean() if exclude_fully_conditioned_items else per_instance.mean()  # []
    return scalar, per_instance


@attrs.define(slots=False)
class TeacherForcingDCMModelConfig(OmniMoTCausalModelConfig):
    """Config for replayed teacher-forcing dCM on the causal model."""

    teacher_forcing_dcm_total_steps: int = 48
    teacher_forcing_dcm_skipping_interval_steps: int = 1
    teacher_forcing_dcm_sample_timestep_shift: float | None = None
    teacher_forcing_dcm_detach_target_endpoint: bool = True
    teacher_forcing_dcm_teacher_guidance: float = 3.0
    # Caption for the teacher CFG negative branch; retain each sample's system
    # prompt. The empty default preserves existing non-transfer recipes.
    teacher_forcing_dcm_teacher_negative_prompt: str = ""
    # Match transfer inference's "same" metadata mode for single-view videos.
    teacher_forcing_dcm_teacher_negative_prompt_keep_metadata: bool = False
    teacher_forcing_dcm_teacher_load_from: Any = None
    teacher_forcing_dcm_teacher_checkpoint_prefix: str = "auto"
    teacher_forcing_dcm_enable_load_balancing_loss: bool = False
    # TF-dCM evaluates the frozen teacher four times and the student three times
    # per step at CFG=3, skip=1. Retaining each network's gathered parameters
    # across its consecutive forwards avoids redundant FSDP all-gathers.
    teacher_forcing_dcm_defer_fsdp_reshard: bool = False


class TeacherForcingDCMModel(OmniMoTCausalModel):
    """Causal VFM model with replayed teacher-forcing dCM training."""

    config: TeacherForcingDCMModelConfig

    def __init__(self, config: TeacherForcingDCMModelConfig):
        self._net_teacher: list[torch.nn.Module] = []
        super().__init__(config)

    @override
    def set_up_model(self) -> None:
        """Build the student and, during TF-dCM training, a frozen causal teacher."""
        super().set_up_model()
        if self.config.causal_training_strategy != "teacher_forcing_dcm":
            return
        if bool(getattr(self.config.parallelism, "enable_inference_mode", False)):
            log.info("[TF-dCM] set_up_model: inference mode; skipping frozen teacher.")
            return

        teacher = self.build_net(dtype=self.precision, mp_policy=MixedPrecisionPolicy())
        teacher.requires_grad_(False)
        teacher.eval()
        self._net_teacher = [teacher]
        if self.config.teacher_forcing_dcm_teacher_load_from is not None:
            self._load_checkpoint_to_tf_dcm_teacher()
        else:
            self._copy_student_to_tf_dcm_teacher()

    @override
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> Any:
        """Load student weights and refresh a student-derived non-persistent teacher."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.causal_training_strategy == "teacher_forcing_dcm" and self._net_teacher:
            self._refresh_tf_dcm_teacher_after_student_load()
        return result

    @override
    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Run the standard causal step or the replayed TF-dCM step."""
        if self.config.causal_training_strategy != "teacher_forcing_dcm":
            return super().training_step(data_batch, iteration)
        return self._training_step_teacher_forcing_dcm(data_batch, iteration)

    @override
    def memory_init_training(
        self,
        gen_data_clean: GenerationDataClean,
        data_batch: dict[str, Any],
        input_text_indexes: list[list[int]],
    ) -> tuple[GenerationDataClean, dict]:
        """Keep teacher CFG tokens with their sample during CP data-window rotation."""
        # Capture metadata from the sampled clip before chunk alignment can
        # shorten raw_state_vision, which aliases the batch's pixel list.
        unconditional_text_indexes = None
        if (
            self.config.causal_training_strategy == "teacher_forcing_dcm"
            and self.config.teacher_forcing_dcm_teacher_guidance > 1.0
        ):
            unconditional_text_indexes = self._make_unconditional_text_indexes(gen_data_clean.batch_size, data_batch)
        gen_data_clean, memory_info = super().memory_init_training(gen_data_clean, data_batch, input_text_indexes)
        if unconditional_text_indexes is not None:
            memory_info["_tf_dcm_unconditional_text_indexes"] = unconditional_text_indexes
        return gen_data_clean, memory_info

    def _refresh_tf_dcm_teacher_after_student_load(self) -> None:
        """Refresh a student-derived TF-dCM teacher after loading student weights."""
        # A teacher loaded from its own checkpoint lives in a plain Python list,
        # so the student's load_state_dict traversal cannot mutate it. Only the
        # teacher derived from the student needs refreshing here.
        if self.config.teacher_forcing_dcm_teacher_load_from is None:
            self._copy_student_to_tf_dcm_teacher()

    @property
    def net_teacher(self) -> torch.nn.Module:
        """Frozen TF-dCM teacher kept out of the routine checkpoint state dict."""
        if not self._net_teacher:
            raise AttributeError("TF-dCM teacher has not been initialized.")
        return self._net_teacher[0]

    def _copy_student_to_tf_dcm_teacher(self) -> None:
        """Initialize the frozen TF-dCM teacher from the current student network."""
        teacher_result = self.net_teacher.load_state_dict(self.net.state_dict(), strict=False)
        unexpected = [key for key in teacher_result.unexpected_keys if not key.endswith("_extra_state")]
        missing = [key for key in teacher_result.missing_keys if not key.endswith("_extra_state")]
        if missing or unexpected:
            raise RuntimeError(
                "Failed to initialize TF-dCM teacher from student; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.net_teacher.requires_grad_(False)
        self.net_teacher.eval()

    @staticmethod
    def _get_load_from_value(load_from: Any, key: str) -> Any:
        """Read a checkpoint config value from either LazyDict or plain dict."""
        if isinstance(load_from, dict):
            return load_from.get(key)
        return getattr(load_from, key, None)

    def _load_checkpoint_to_tf_dcm_teacher(self) -> None:
        """Load the non-persistent TF-dCM teacher from a standalone checkpoint."""
        load_from = self.config.teacher_forcing_dcm_teacher_load_from
        if load_from is None:
            return
        ckpt_path = self._get_load_from_value(load_from, "load_path")
        credential_path = self._get_load_from_value(load_from, "credentials")
        if not ckpt_path:
            raise ValueError("teacher_forcing_dcm_teacher_load_from must define `load_path`.")

        prefix = self.config.teacher_forcing_dcm_teacher_checkpoint_prefix
        if prefix == "auto":
            prefix = "net" if ckpt_path.endswith(".dcp/model") else "net_ema"
        if prefix not in ("net", "net_ema"):
            raise ValueError(
                f"teacher_forcing_dcm_teacher_checkpoint_prefix must be 'auto', 'net', or 'net_ema'; got {prefix!r}."
            )

        log.info(f"[TF-dCM] Loading frozen teacher from {ckpt_path} with prefix {prefix!r}")
        storage_reader = get_storage_reader(ckpt_path, credential_path)
        teacher_state_dict = get_model_state_dict(self.net_teacher)
        metadata = storage_reader.read_metadata()
        checkpoint_keys = set(metadata.state_dict_metadata.keys())
        prefixed_teacher_keys = {f"{prefix}.{key}" for key in teacher_state_dict.keys()}

        missing_keys = [
            key for key in sorted(prefixed_teacher_keys - checkpoint_keys) if not key.endswith("_extra_state")
        ]
        if missing_keys:
            raise RuntimeError(
                f"[TF-dCM] Frozen teacher checkpoint is missing model keys; first missing keys: {missing_keys[:20]}"
            )
        unexpected_keys = sorted(checkpoint_keys - prefixed_teacher_keys)
        if prefix == "net_ema":
            unexpected_keys = [key for key in unexpected_keys if "net." not in key]
        else:
            unexpected_keys = [key for key in unexpected_keys if "net_ema." not in key]
        unexpected_keys = [key for key in unexpected_keys if "_extra_state" not in key]
        if unexpected_keys:
            log.warning(f"[TF-dCM] Unexpected teacher checkpoint keys: {unexpected_keys[:20]}")

        prefixed_state_dict = collections.OrderedDict()
        for key in teacher_state_dict.keys():
            prefixed_state_dict[f"{prefix}.{key}"] = teacher_state_dict[key]
        dcp.load(
            prefixed_state_dict,
            storage_reader=storage_reader,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        for key in teacher_state_dict.keys():
            teacher_state_dict[key] = prefixed_state_dict[f"{prefix}.{key}"]

        log.info(
            set_model_state_dict(
                self.net_teacher,
                teacher_state_dict,
                options=StateDictOptions(strict=False),
            )
        )
        self.net_teacher.requires_grad_(False)
        self.net_teacher.eval()
        del teacher_state_dict, prefixed_state_dict

    def _build_noisy_target_pack(
        self,
        sequence_plans: list[SequencePlan],
        input_text_indexes: list[list[int]],
        gen_data_clean: GenerationDataClean,
        gen_data_noised: GenerationDataNoised,
        timesteps: torch.Tensor,  # [B,1] on the diffusion device or CPU
        memory_info: dict,
    ) -> PackedSequence:
        """Pack a noisy target sequence at a specified RF timestep."""
        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            input_text_indexes,
            gen_data_clean,
            # Layout metadata is constructed on CPU even when the diffusion
            # ladder and endpoint states live on CUDA.
            timesteps.cpu(),  # [B,1] on CPU
            skip_text_tokens=memory_info["skip_text"],
            initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
        )
        self._replace_clean_with_noised(packed_sequence, gen_data_noised)
        packed_sequence.to_cuda()
        return packed_sequence

    def _denoise_with_replayed_tf(
        self,
        net: torch.nn.Module,
        noisy_pack: PackedSequence,
        gen_data_clean: GenerationDataClean,
        tf_memory: TeacherForcingMemoryState,
    ) -> dict:
        """Run a noisy target pass through replayed teacher-forcing memory."""
        if getattr(tf_memory, "selected_clean_gen_token_indexes", None) is not None:
            if noisy_pack.vision is None:
                raise ValueError("Two-way Flex TF-dCM replay requires a vision pack.")
            original_condition_masks = getattr(
                noisy_pack,
                "teacher_forcing_original_condition_masks_vision",
                None,
            )
            if original_condition_masks is None:
                original_condition_masks = [mask.clone() for mask in noisy_pack.vision.condition_mask]
            noisy_pack.teacher_forcing_pass = "noisy"
            noisy_pack.teacher_forcing_original_condition_masks_vision = original_condition_masks
            noisy_pack.teacher_forcing_selected_clean_target_padded_capacity = (
                tf_memory.selected_clean_gen_padded_capacity
            )
        tf_memory.pass_number = 2
        return self.denoise(
            net=net,
            data_batch_packed=noisy_pack,
            memory=tf_memory,
        )

    def _make_teacher_negative_captions(self, batch_size: int, data_batch: dict[str, Any]) -> list[str]:
        """Format negative captions from the owner's pixels before CP data-window rotation."""
        negative_prompt = self.config.teacher_forcing_dcm_teacher_negative_prompt
        if not self.config.teacher_forcing_dcm_teacher_negative_prompt_keep_metadata:
            return [negative_prompt] * batch_size
        if "enable_per_camera_vae_encoding" in data_batch:
            raise ValueError("TF-dCM negative metadata currently supports single-view video batches only.")

        # get_data_and_condition has flattened each sample's [control, target]
        # pixels. The final item belongs to the target; image_size is flattened
        # too, so indexing that field by logical sample would mix sample owners.
        videos = data_batch.get(self.input_video_key)
        item_counts = data_batch.get("num_vision_items_per_sample")
        if item_counts is None:
            item_counts = [1] * batch_size
        if (
            not isinstance(videos, (list, tuple))
            or not isinstance(item_counts, (list, tuple))
            or len(item_counts) != batch_size
            or any(not isinstance(count, int) or count < 1 for count in item_counts)
            or sum(item_counts) != len(videos)
        ):
            raise ValueError("TF-dCM negative metadata requires aligned video items and counts for each sample.")

        # conditioning_fps includes the sampler's frame stride; the source fps
        # may describe a different rate. Both collated tensors and joint-loader
        # lists of singleton tensors preserve one effective FPS per sample.
        fps_values = data_batch.get("conditioning_fps")
        if isinstance(fps_values, torch.Tensor):
            fps_values = fps_values.reshape(-1).tolist()  # [B] -> list[float]
        if not isinstance(fps_values, (list, tuple)) or len(fps_values) != batch_size:
            raise ValueError("TF-dCM negative metadata requires one conditioning_fps value per sample.")

        captions: list[str] = []
        item_offset = 0
        for sample_idx, item_count in enumerate(item_counts):
            target = videos[item_offset + item_count - 1]  # [1,C,T,H,W]
            item_offset += item_count
            if not isinstance(target, torch.Tensor) or target.ndim != 5 or target.shape[0] != 1:
                raise ValueError("TF-dCM negative metadata requires target pixels with shape [1,C,T,H,W].")
            num_frames, height, width = target.shape[-3:]
            if min(num_frames, height, width) < 1:
                raise ValueError("TF-dCM negative metadata requires nonempty target pixels.")
            fps_value = fps_values[sample_idx]
            if isinstance(fps_value, torch.Tensor):
                if fps_value.numel() != 1:
                    raise ValueError("TF-dCM negative metadata requires one conditioning_fps value per sample.")
                fps_value = fps_value.item()
            try:
                fps = float(fps_value)
            except (TypeError, ValueError) as error:
                raise ValueError("TF-dCM negative metadata requires numeric conditioning_fps.") from error
            if not math.isfinite(fps) or fps <= 0 or round(fps) < 1:
                raise ValueError(f"TF-dCM negative metadata requires positive finite conditioning_fps; got {fps}.")
            # Transfer inference rounds output FPS before formatting metadata.
            # Use the full sampled pixel clip, not the latent generation chunk.
            metadata_fps = int(round(fps))
            caption = negative_prompt.strip()
            if num_frames > 1:
                duration_text = DEFAULT_DURATION_FPS_TEMPLATE.format(
                    duration=num_frames / metadata_fps, fps=metadata_fps
                )
                caption = caption.rstrip(".") + ". " + duration_text
            resolution_text = DEFAULT_VIDEO_TEMPLATE.format(height=height, width=width)
            caption = caption.strip().rstrip(".") + ". " + resolution_text
            captions.append(caption.lstrip(".").strip())
        return captions

    def _make_unconditional_text_indexes(
        self,
        batch_size: int,
        data_batch: dict[str, Any],
    ) -> list[list[int]]:
        """Tokenize the teacher's negative caption with caption dropout's system prompts."""
        system_prompt = resolve_text_system_prompt(data_batch)
        if system_prompt is None or isinstance(system_prompt, str):
            system_prompts = [system_prompt] * batch_size
        elif isinstance(system_prompt, (list, tuple)):
            if len(system_prompt) != batch_size:
                raise ValueError(
                    f"TF-dCM requires one system prompt per sample; got {len(system_prompt)} for {batch_size} samples."
                )
            system_prompts = list(system_prompt)
        else:
            raise TypeError(f"TF-dCM system prompts must be strings, got {type(system_prompt).__name__}.")
        if any(prompt is not None and not isinstance(prompt, str) for prompt in system_prompts):
            raise TypeError("TF-dCM system prompts must contain only strings or None.")
        negative_captions = self._make_teacher_negative_captions(batch_size, data_batch)
        return [
            tokenize_caption(
                negative_caption,
                self.vlm_tokenizer,
                is_video=False,
                use_system_prompt=self.vlm_config.use_system_prompt,
                system_prompt=prompt,
            )
            for negative_caption, prompt in zip(negative_captions, system_prompts, strict=True)
        ]

    @staticmethod
    def _blend_teacher_cfg_vision(
        conditional: list[torch.Tensor],  # list of [C,T,H,W]
        unconditional: list[torch.Tensor],  # list of [C,T,H,W]
        guidance: float,
    ) -> list[torch.Tensor]:
        """Blend conditional and unconditional teacher vision velocities."""
        return [
            uncond_i + guidance * (cond_i - uncond_i) for cond_i, uncond_i in zip(conditional, unconditional)
        ]  # list of [C,T,H,W]

    def _denoise_tf_dcm_teacher_with_guidance(
        self,
        teacher_pack: PackedSequence,
        teacher_pack_uncond: PackedSequence | None,
        gen_data_clean: GenerationDataClean,
        teacher_tf_memory: TeacherForcingMemoryState,
        teacher_tf_memory_uncond: TeacherForcingMemoryState | None,
    ) -> dict:
        """Run the frozen TF-dCM teacher, optionally with replayed CFG."""
        guidance = self.config.teacher_forcing_dcm_teacher_guidance
        out_teacher = self._denoise_with_replayed_tf(self.net_teacher, teacher_pack, gen_data_clean, teacher_tf_memory)
        if guidance == 1.0:
            return out_teacher
        if teacher_pack_uncond is None or teacher_tf_memory_uncond is None:
            raise ValueError("TF-dCM teacher CFG requires an unconditional teacher replay cache.")
        out_teacher_uncond = self._denoise_with_replayed_tf(
            self.net_teacher,
            teacher_pack_uncond,
            gen_data_clean,
            teacher_tf_memory_uncond,
        )
        out_teacher = dict(out_teacher)
        out_teacher["preds_vision"] = self._blend_teacher_cfg_vision(
            out_teacher["preds_vision"],
            out_teacher_uncond["preds_vision"],
            guidance,
        )
        return out_teacher

    def _denoise_tf_dcm_teacher_branch(
        self,
        *,
        clean_pack: PackedSequence,
        sequence_plans: list[SequencePlan],
        input_text_indexes: list[list[int]],
        gen_data_clean: GenerationDataClean,
        gen_data_noised: GenerationDataNoised,
        timesteps: torch.Tensor,  # [B,1] on CPU
        memory_info: dict,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run one frozen teacher branch and release its replay cache promptly."""
        teacher_tf_memory = self._build_clean_tf_cache(
            net=self.net_teacher,
            packed_sequence=clean_pack,
            gen_data_clean=gen_data_clean,
            memory_info=memory_info,
            detach_clean_kv=True,
        )
        teacher_pack = self._build_noisy_target_pack(
            sequence_plans=sequence_plans,
            input_text_indexes=input_text_indexes,
            gen_data_clean=gen_data_clean,
            gen_data_noised=gen_data_noised,
            timesteps=timesteps,
            memory_info=memory_info,
        )
        out_teacher = self._denoise_with_replayed_tf(
            self.net_teacher,
            teacher_pack,
            gen_data_clean,
            teacher_tf_memory,
        )
        assert teacher_pack.vision is not None
        condition_mask_teacher = teacher_pack.vision.condition_mask
        assert isinstance(condition_mask_teacher, list)
        preds_vision = [pred.detach() for pred in out_teacher["preds_vision"]]  # list of [C,T,H,W]
        del out_teacher, teacher_pack, teacher_tf_memory
        return preds_vision, condition_mask_teacher

    def _get_data_resolutions(self, data_batch: dict[str, torch.Tensor], batch_size: int) -> list[str] | None:
        """Compute per-sample resolution strings for RF shift lookup."""
        if "image_size" not in data_batch:
            return None
        data_resolutions = []
        for i in range(batch_size):
            img_size = data_batch["image_size"][i]
            if img_size.dim() == 2:
                img_size = img_size[0]
            target_h = int(img_size[0].item())
            target_w = int(img_size[1].item())
            data_resolutions.append(get_vision_data_resolution((target_h, target_w)))
        return data_resolutions

    @staticmethod
    def _num_vision_tokens_per_sample(gen_data_clean: GenerationDataClean) -> list[int]:
        """Return generated-target token counts before merging, one per logical sample.

        Multi-item vision samples such as transfer flatten ``[control, target]``
        into one list, with the generated target stored last. Dynamic RF shifts
        are per logical sample, so count the last item rather than every flat
        control and target item independently.
        """
        vision_items = gen_data_clean.x0_tokens_vision
        assert vision_items is not None, "TF-dCM requires vision tokens."
        item_counts = getattr(gen_data_clean, "num_vision_items_per_sample", None)
        if item_counts is None:
            return [x.shape[2] * x.shape[3] * x.shape[4] for x in vision_items]
        if len(item_counts) != gen_data_clean.batch_size or sum(item_counts) != len(vision_items):
            raise ValueError(
                "num_vision_items_per_sample must align with the logical batch and flat vision items, "
                f"got batch_size={gen_data_clean.batch_size}, counts={item_counts}, and {len(vision_items)} items."
            )

        num_tokens: list[int] = []
        item_offset = 0
        for item_count in item_counts:
            if item_count < 1:
                raise ValueError(f"Vision item counts must be positive, got {item_counts}.")
            x0 = vision_items[item_offset + item_count - 1]  # generated target = last item of the sample
            num_tokens.append(x0.shape[2] * x0.shape[3] * x0.shape[4])
            item_offset += item_count
        return num_tokens

    def _resolve_teacher_forcing_dcm_shift(
        self,
        batch_size: int,
        is_image_batch: bool,
        resolutions: list[str] | None,
        num_tokens: list[int],
    ) -> torch.Tensor:
        """Resolve the per-sample RF shift used for TF-dCM ladder sampling."""
        shift_config = self.config.teacher_forcing_dcm_sample_timestep_shift
        if shift_config is None:
            shift_config = self.config.rectified_flow_training_config.shift

        if isinstance(shift_config, (int, float)):
            return torch.full((batch_size,), float(shift_config), **self.tensor_kwargs_fp32)  # [B]

        shift_dict = dict(shift_config)
        if not is_image_batch and "dynamic_shift_base_num_tokens_video" in shift_dict:
            assert len(num_tokens) == batch_size
            base_num_tokens = shift_dict["dynamic_shift_base_num_tokens_video"]
            return torch.sqrt(torch.tensor(num_tokens, **self.tensor_kwargs_fp32) / base_num_tokens)  # [B]
        if is_image_batch and "dynamic_shift_base_num_tokens_image" in shift_dict:
            assert len(num_tokens) == batch_size
            base_num_tokens = shift_dict["dynamic_shift_base_num_tokens_image"]
            return torch.sqrt(torch.tensor(num_tokens, **self.tensor_kwargs_fp32) / base_num_tokens)  # [B]
        if resolutions is None:
            raise ValueError("Resolutions must be provided when TF-dCM shift is a dict")

        shifts = []
        for resolution in resolutions:
            if resolution not in shift_dict:
                raise ValueError(
                    f"Resolution '{resolution}' not found in TF-dCM shift dict. "
                    f"Available resolutions: {list(shift_dict.keys())}"
                )
            shifts.append(float(shift_dict[resolution]))
        return torch.tensor(shifts, **self.tensor_kwargs_fp32)  # [B]

    @staticmethod
    def _velocity_to_x0(
        x_t: list[torch.Tensor],  # list of [C,T,H,W]
        v_pred: list[torch.Tensor],  # list of [C,T,H,W]
        sigma: list[torch.Tensor],  # list of [T,1,1]
    ) -> list[torch.Tensor]:
        """Convert RF velocity predictions to clean endpoint predictions."""
        return [
            x_t[i].float() - sigma[i].to(device=x_t[i].device, dtype=torch.float32) * v_pred[i].float()  # [C,T,H,W]
            for i in range(len(x_t))
        ]  # list of [C,T,H,W]

    @staticmethod
    def _vision_sigma_masks(
        sigmas: torch.Tensor,  # [B_items,1]
        condition_mask: list[torch.Tensor],  # list of [T,1,1]
    ) -> list[torch.Tensor]:
        """Broadcast sample sigmas over generated vision tokens only."""
        return [
            sigmas[i].view(-1, 1, 1)[: condition_mask[i].shape[0]]
            * (1.0 - condition_mask[i].to(device=sigmas.device, dtype=sigmas.dtype))
            for i in range(len(condition_mask))
        ]  # list of [T,1,1]

    def _make_generation_data_noised_like(
        self,
        base: GenerationDataNoised,
        xt_tokens_vision: list[torch.Tensor],  # list of [C,T,H,W]
        sigmas_vision: list[torch.Tensor],  # list of [T,1,1]
    ) -> GenerationDataNoised:
        """Reuse non-vision noised fields while replacing the current vision state."""
        return GenerationDataNoised(
            batch_size=base.batch_size,
            epsilon_vision=base.epsilon_vision,
            xt_tokens_vision=[xt.to(**self.tensor_kwargs) for xt in xt_tokens_vision],  # list of [C,T,H,W]
            vt_target_vision=base.vt_target_vision,
            sigmas_vision=sigmas_vision,
            epsilon_action=base.epsilon_action,
            xt_tokens_action=base.xt_tokens_action,
            vt_target_action=base.vt_target_action,
            sigmas_action=base.sigmas_action,
            epsilon_sound=base.epsilon_sound,
            xt_tokens_sound=base.xt_tokens_sound,
            vt_target_sound=base.vt_target_sound,
            sigmas_sound=base.sigmas_sound,
            raw_action_dim=base.raw_action_dim,
        )

    @staticmethod
    def _ode_step_vision(
        xt: list[torch.Tensor],  # list of [C,T,H,W]
        v_pred: list[torch.Tensor],  # list of [C,T,H,W]
        sigma_cur: list[torch.Tensor],  # list of [T,1,1]
        sigma_next: list[torch.Tensor],  # list of [T,1,1]
    ) -> list[torch.Tensor]:
        """Advance noised vision tokens with one teacher Euler step."""
        return [
            xt_i.float()
            + (
                sigma_next_i.to(device=xt_i.device, dtype=torch.float32)
                - sigma_cur_i.to(device=xt_i.device, dtype=torch.float32)
            )
            * v_i.float()  # [C,T,H,W]
            for xt_i, v_i, sigma_cur_i, sigma_next_i in zip(xt, v_pred, sigma_cur, sigma_next)
        ]  # list of [C,T,H,W]

    @staticmethod
    def _assert_matching_vision_condition_masks(
        reference: list[torch.Tensor],  # list of [T,1,1]
        candidate: list[torch.Tensor],  # list of [T,1,1]
    ) -> None:
        """Validate that endpoint packs supervise the same vision frames."""
        if len(reference) != len(candidate):
            raise ValueError(
                "TF-dCM endpoint packs must have the same number of vision condition masks; "
                f"got {len(reference)} and {len(candidate)}."
            )
        for idx, (reference_i, candidate_i) in enumerate(zip(reference, candidate)):
            if reference_i.shape != candidate_i.shape or not torch.equal(reference_i, candidate_i):
                raise ValueError(f"TF-dCM endpoint condition masks diverged for vision item {idx}.")

    def _compute_tf_dcm_aux_losses(self, out_net: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute auxiliary LBL losses for the TF-dCM student endpoint pass."""
        total_loss = torch.tensor(0.0, **self.tensor_kwargs_fp32)  # []
        losses_dict = {}
        if not self.config.teacher_forcing_dcm_enable_load_balancing_loss:
            return total_loss, losses_dict
        device_mesh, context_parallel_mesh = self._get_load_balancing_loss_meshes()
        for load_balancing_type in ["und", "gen"]:
            lbl_metadata = out_net.get(f"lbl_metadata_{load_balancing_type}", None)
            if lbl_metadata is None:
                continue
            load_balancing_loss = compute_load_balancing_loss(
                lbl_metadata,
                coeff=getattr(self.config.lbl, f"coeff_{load_balancing_type}"),
                method=self.config.lbl.method,
                device_mesh=device_mesh,
                context_parallel_mesh=context_parallel_mesh,
            )
            if load_balancing_loss is not None:
                total_loss = total_loss + load_balancing_loss  # []
                losses_dict[f"aux_loss_{load_balancing_type}"] = load_balancing_loss
        return total_loss, losses_dict

    def _training_step_teacher_forcing_dcm(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Run replayed teacher-forcing discrete consistency distillation."""
        if not hasattr(self, "net_teacher"):
            raise RuntimeError("TF-dCM requires a frozen `net_teacher`; call set_up_model() before training.")
        input_text_indexes, sequence_plans, gen_data_clean, memory_info, data_resolutions, _ = (
            self._get_training_inputs(data_batch, iteration)
        )
        assert all(x.shape[0] == 1 for x in gen_data_clean.x0_tokens_vision), (
            "Batch size must be 1 for individual samples"
        )
        num_tokens_per_sample = self._num_vision_tokens_per_sample(gen_data_clean)
        dcm_shift = self._resolve_teacher_forcing_dcm_shift(
            batch_size=gen_data_clean.batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            resolutions=data_resolutions,
            num_tokens=num_tokens_per_sample,
        )  # [B]
        rectified_flow_vision = (
            self.rectified_flow_image if gen_data_clean.is_image_batch else self.rectified_flow_video
        )
        max_timestep = rectified_flow_vision.noise_scheduler.config.num_train_timesteps
        timesteps_ladder, sigmas_ladder = sample_teacher_forcing_dcm_ladder(
            batch_size=gen_data_clean.batch_size,
            total_steps=self.config.teacher_forcing_dcm_total_steps,
            skipping_interval_steps=self.config.teacher_forcing_dcm_skipping_interval_steps,
            shift=dcm_shift,
            max_timestep=max_timestep,
            tensor_kwargs=self.tensor_kwargs_fp32,
        )  # [B,N+1], [B,N+1]
        context_parallel_broadcast_tensor_list([timesteps_ladder, sigmas_ladder], getattr(self, "parallel_dims", None))
        # Sequence packing consumes CPU timesteps. Transfer the short ladder
        # once instead of synchronizing and copying it before every one of the
        # conditional, unconditional, and student noisy-pass packs.
        timesteps_ladder_cpu = timesteps_ladder.detach().cpu()  # [B,N+1]

        timesteps_t0 = timesteps_ladder_cpu[:, 0:1]  # [B,1] on CPU
        sigmas_t0 = _expand_per_sample_to_per_vision_item(
            sigmas_ladder[:, 0:1], gen_data_clean.num_vision_items_per_sample
        )  # [B_items,1]
        packed_t0 = self._pack_input_sequence(
            sequence_plans,
            input_text_indexes,
            gen_data_clean,
            timesteps_t0,
            skip_text_tokens=memory_info["skip_text"],
            initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
        )
        self._validate_teacher_forcing_pack(packed_t0)
        teacher_guidance = self.config.teacher_forcing_dcm_teacher_guidance
        if teacher_guidance < 1.0:
            raise ValueError(f"teacher_forcing_dcm_teacher_guidance must be >= 1.0; got {teacher_guidance}.")
        teacher_cfg_enabled = teacher_guidance > 1.0
        if self.config.sound_gen:
            raise ValueError("TF-dCM currently supports vision/action training only; set sound_gen=False.")
        if memory_info.get("use_rolling_kv_cache", False):
            raise ValueError(
                "TF-dCM currently requires independent per-network replay caches; disable KV-cache training."
            )
        if packed_t0.action is not None and any(nfi.numel() > 0 for nfi in packed_t0.action.noisy_frame_indexes):
            raise ValueError("TF-dCM currently supports action tokens as conditioning inputs only.")

        # Expand the CP owner's cached negative tokens over its packed caption slots.
        # Tokenization already captured the full clip metadata before memory initialization.
        uncond_text_indexes = (
            [
                list(tokens)
                for plan, tokens in zip(sequence_plans, memory_info["_tf_dcm_unconditional_text_indexes"], strict=True)
                for _ in range(len(plan.text_view_ids) if plan.text_view_ids is not None else 1)
            ]
            if teacher_cfg_enabled
            else None
        )
        packed_t0_uncond = (
            self._pack_input_sequence(
                sequence_plans,
                uncond_text_indexes,
                gen_data_clean,
                timesteps_t0,
                skip_text_tokens=memory_info["skip_text"],
                initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
            )
            if uncond_text_indexes is not None
            else None
        )
        if packed_t0_uncond is not None:
            self._validate_teacher_forcing_pack(packed_t0_uncond)

        gen_data_noised_t0 = self._add_noise_to_input(gen_data_clean, packed_t0, sigmas_t0)
        assert packed_t0.vision is not None
        condition_mask_t0 = packed_t0.vision.condition_mask
        assert isinstance(condition_mask_t0, list)
        sigma_vision_t0 = self._vision_sigma_masks(sigmas_t0, condition_mask_t0)  # list of [T,1,1]
        xt_vision_t0_fp32 = [
            (1.0 - sigma_i.to(device=x0_i.device, dtype=torch.float32)) * x0_i.float()  # [C,T,H,W]
            + sigma_i.to(device=x0_i.device, dtype=torch.float32) * eps_i.float()
            for x0_i, eps_i, sigma_i in zip(
                gen_data_clean.x0_tokens_vision,
                gen_data_noised_t0.epsilon_vision,
                sigma_vision_t0,
            )
        ]  # list of [C,T,H,W]

        xt_current = xt_vision_t0_fp32  # list of [C,T,H,W]
        with (
            torch.no_grad(),
            defer_fsdp_reshard(
                self.net_teacher,
                after_forward=self.config.teacher_forcing_dcm_defer_fsdp_reshard,
                reshard_root_on_exit=self.config.teacher_forcing_dcm_defer_fsdp_reshard,
            ),
        ):
            for step_idx in range(self.config.teacher_forcing_dcm_skipping_interval_steps):
                timesteps_cur = timesteps_ladder_cpu[:, step_idx : step_idx + 1]  # [B,1] on CPU
                sigmas_cur = _expand_per_sample_to_per_vision_item(
                    sigmas_ladder[:, step_idx : step_idx + 1], gen_data_clean.num_vision_items_per_sample
                )  # [B_items,1]
                sigmas_next = _expand_per_sample_to_per_vision_item(
                    sigmas_ladder[:, step_idx + 1 : step_idx + 2], gen_data_clean.num_vision_items_per_sample
                )  # [B_items,1]
                gen_data_noised_cur = self._make_generation_data_noised_like(
                    gen_data_noised_t0,
                    xt_tokens_vision=xt_current,
                    sigmas_vision=self._vision_sigma_masks(sigmas_cur, condition_mask_t0),
                )
                teacher_velocity, condition_mask_teacher = self._denoise_tf_dcm_teacher_branch(
                    clean_pack=packed_t0,
                    sequence_plans=sequence_plans,
                    input_text_indexes=input_text_indexes,
                    gen_data_clean=gen_data_clean,
                    gen_data_noised=gen_data_noised_cur,
                    timesteps=timesteps_cur,
                    memory_info=memory_info,
                )
                if uncond_text_indexes is not None:
                    assert packed_t0_uncond is not None
                    teacher_velocity_uncond, condition_mask_teacher_uncond = self._denoise_tf_dcm_teacher_branch(
                        clean_pack=packed_t0_uncond,
                        sequence_plans=sequence_plans,
                        input_text_indexes=uncond_text_indexes,
                        gen_data_clean=gen_data_clean,
                        gen_data_noised=gen_data_noised_cur,
                        timesteps=timesteps_cur,
                        memory_info=memory_info,
                    )
                    self._assert_matching_vision_condition_masks(condition_mask_teacher, condition_mask_teacher_uncond)
                    teacher_velocity = self._blend_teacher_cfg_vision(
                        teacher_velocity,
                        teacher_velocity_uncond,
                        teacher_guidance,
                    )  # list of [C,T,H,W]
                    del teacher_velocity_uncond, condition_mask_teacher_uncond
                xt_current = self._ode_step_vision(
                    xt_current,
                    teacher_velocity,
                    self._vision_sigma_masks(sigmas_cur, condition_mask_teacher),
                    self._vision_sigma_masks(sigmas_next, condition_mask_teacher),
                )  # list of [C,T,H,W]
                del gen_data_noised_cur, teacher_velocity, condition_mask_teacher

        with defer_fsdp_reshard(
            self.net,
            after_forward=self.config.teacher_forcing_dcm_defer_fsdp_reshard,
        ):
            student_tf_memory = self._build_clean_tf_cache(
                net=self.net,
                packed_sequence=packed_t0,
                gen_data_clean=gen_data_clean,
                memory_info=memory_info,
                detach_clean_kv=self.config.teacher_forcing_detach_clean_kv,
            )
            self._replace_clean_with_noised(packed_t0, gen_data_noised_t0)
            packed_t0.to_cuda()
            out_student_t0 = self._denoise_with_replayed_tf(self.net, packed_t0, gen_data_clean, student_tf_memory)
            x0_pred_t0 = self._velocity_to_x0(
                xt_vision_t0_fp32, out_student_t0["preds_vision"], sigma_vision_t0
            )  # list of [C,T,H,W]
            del packed_t0_uncond

            timesteps_tn = timesteps_ladder_cpu[:, -1:]  # [B,1] on CPU
            sigmas_tn = _expand_per_sample_to_per_vision_item(
                sigmas_ladder[:, -1:], gen_data_clean.num_vision_items_per_sample
            )  # [B_items,1]
            gen_data_noised_tn = self._make_generation_data_noised_like(
                gen_data_noised_t0,
                xt_tokens_vision=xt_current,
                sigmas_vision=self._vision_sigma_masks(sigmas_tn, condition_mask_t0),
            )
            packed_tn = self._build_noisy_target_pack(
                sequence_plans=sequence_plans,
                input_text_indexes=input_text_indexes,
                gen_data_clean=gen_data_clean,
                gen_data_noised=gen_data_noised_tn,
                timesteps=timesteps_tn,
                memory_info=memory_info,
            )
            with torch.no_grad():
                out_student_tn = self._denoise_with_replayed_tf(self.net, packed_tn, gen_data_clean, student_tf_memory)
            assert packed_tn.vision is not None
            condition_mask_tn = packed_tn.vision.condition_mask
            assert isinstance(condition_mask_tn, list)
            self._assert_matching_vision_condition_masks(condition_mask_t0, condition_mask_tn)
            sigma_vision_tn = self._vision_sigma_masks(sigmas_tn, condition_mask_tn)  # list of [T,1,1]
            with torch.no_grad():
                x0_pred_tn = self._velocity_to_x0(
                    xt_current, out_student_tn["preds_vision"], sigma_vision_tn
                )  # list of [C,T,H,W]

        dcm_loss_vision, dcm_loss_vision_per_instance = compute_endpoint_consistency_loss(
            endpoint=x0_pred_t0,
            target=x0_pred_tn,
            condition_mask=condition_mask_t0,
            detach_target=self.config.teacher_forcing_dcm_detach_target_endpoint,
        )
        rf_cfg = self.config.rectified_flow_training_config
        loss_scale = (
            rf_cfg.image_loss_scale
            if gen_data_clean.is_image_batch and rf_cfg.image_loss_scale is not None
            else rf_cfg.loss_scale
        )
        total_loss = dcm_loss_vision * loss_scale  # []
        losses_dict: dict[str, torch.Tensor] = {
            "tf_dcm_loss_vision": dcm_loss_vision,
            "tf_dcm_loss_vision_per_instance": dcm_loss_vision_per_instance,
            "flow_matching_loss_vision": dcm_loss_vision,
            "flow_matching_loss_vision_per_instance": dcm_loss_vision_per_instance,
            "flow_matching_loss_action": torch.tensor(0.0, **self.tensor_kwargs_fp32),
            "flow_matching_loss_sound": torch.tensor(0.0, **self.tensor_kwargs_fp32),
        }
        if self.config.action_gen and "preds_action" in out_student_tn:
            dummy_action_loss = 0.0 * sum(pred.sum() for pred in out_student_tn["preds_action"])  # []
            total_loss = total_loss + dummy_action_loss
            losses_dict["flow_matching_loss_action"] = dummy_action_loss
        if self.config.sound_gen and "preds_sound" in out_student_tn:
            dummy_sound_loss = 0.0 * sum(pred.sum() for pred in out_student_tn["preds_sound"])  # []
            total_loss = total_loss + dummy_sound_loss
            losses_dict["flow_matching_loss_sound"] = dummy_sound_loss
        aux_loss, aux_losses_dict = self._compute_tf_dcm_aux_losses(out_student_t0)
        total_loss = total_loss + aux_loss
        losses_dict.update(aux_losses_dict)

        output_batch = {
            "x0": gen_data_clean.x0_tokens_vision,
            "xt": gen_data_noised_tn.xt_tokens_vision,
            "sigma": sigmas_tn,
            "model_pred": out_student_tn["preds_vision"],
            "condition_mask_vision": packed_tn.vision.condition_mask,
            "condition_mask_action": packed_tn.action.condition_mask if packed_tn.action else None,
            "und_token_length": packed_tn.text_indexes.shape[0],
            "gen_token_length": packed_tn.sequence_length - packed_tn.text_indexes.shape[0],
            "vision_token_length": len(packed_tn.vision.sequence_indexes),
            "action_token_length": len(packed_tn.action.sequence_indexes) if packed_tn.action else 0,
            "sound_token_length": len(packed_tn.sound.sequence_indexes) if packed_tn.sound else 0,
            "is_image_batch": gen_data_clean.is_image_batch,
            "batch_size": gen_data_clean.batch_size,
            "split_lens": packed_tn.split_lens,
            "attn_modes": packed_tn.attn_modes,
            "tf_dcm_timesteps": timesteps_ladder,
            "tf_dcm_sigmas": sigmas_ladder,
            "tf_dcm_teacher_guidance": torch.tensor(teacher_guidance, **self.tensor_kwargs_fp32),  # []
            **losses_dict,
        }
        return output_batch, total_loss
