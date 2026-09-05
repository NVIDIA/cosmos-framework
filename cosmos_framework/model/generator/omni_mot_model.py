# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import collections
import dataclasses
import inspect
import json
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_framework.utils.flags import DEVICE, Device
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate
from cosmos_framework.utils.lazy_config.registry import locate
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log, misc
from cosmos_framework.utils.count_params import count_params
from cosmos_framework.model.generator.algorithm.loss.flow_matching import (
    ACTION_SLOT_SAMPLE_COUNT_KEY,
    ACTION_SLOT_SAMPLE_LOSS_KEY,
    ActionSlotLossStats,
    compute_flow_matching_loss,
)
from cosmos_framework.model.generator.algorithm.loss.load_balancing import compute_load_balancing_loss
from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.configs.base.defaults.parallelism import PRECISION_TO_TORCH_DTYPE
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionProcessor,
    get_action_processing_records,
)
from cosmos_framework.data.generator.action.utils.unified_action_schema import UNIFIED_ACTION_SLOT_GROUPS
from cosmos_framework.data.generator.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.diffusion.samplers.edm import EDMSampler
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
from cosmos_framework.model.generator.mot.context_parallel_utils import (
    broadcast_context_parallel_object,
    context_parallel_broadcast_tensor_list,
)
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
    InferenceTextKVMemoryState,
    UndKVCache,
    install_inference_memory_attention_dispatch,
    make_inference_text_kv_cache,
    restore_inference_attention_dispatch,
)
from cosmos_framework.model.generator.mot.modeling_utils import has_noisy_tokens
from cosmos_framework.model.generator.mot.parallelize_vfm_network import parallelize_vfm_network
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.model.generator.utils.data_and_condition import (
    GenerationDataClean,
    GenerationDataNoised,
    _expand_per_sample_to_per_vision_item,
    build_dense_sound_schedule,
    unwrap_and_densify,
)
from cosmos_framework.model.generator.utils.load_balancing_stats import LBLConfig
from cosmos_framework.model.generator.utils.memory import MemoryState
from cosmos_framework.model.generator.utils.moe_utils import (
    sync_expert_biases_to_ema,
    sync_router_biases_to_ema,
    update_expert_biases,
    update_router_biases,
    uses_aux_loss_free_load_balancing,
    uses_ema_router_bias,
)
from cosmos_framework.model.generator.utils.safetensors_loader import (
    load_language_model as load_language_model_safetensors,
)
from cosmos_framework.model.generator.vision_encoder import (
    VisionEncoder,
    get_vae_pixel_shapes,
    normalize_uint8_item,
    validate_multiview_length,
)
from cosmos_framework.data.generator.sequence_packing import (
    PackedSequence,
    SequencePlan,
    build_sequence_plans_from_data_batch,
    pack_input_sequence,
)
from cosmos_framework.data.generator.sequence_packing.modality import add_special_tokens
from cosmos_framework.data.generator.sequence_packing.packers import is_item_generated, uses_single_timestep
from cosmos_framework.model.generator.tokenizers.interface import VideoTokenizerInterface
from cosmos_framework.model.generator.upsampler.prompts import build_messages, clean_response
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution, read_positive_int_metadata
from cosmos_framework.utils.generator.dtensor_helper import DTensorFastEmaModelUpdater
from cosmos_framework.utils.generator.model_weights_stats import WeightTrainingStat
from cosmos_framework.utils.generator.parallelism import ParallelDims
from cosmos_framework.utils.generator.quantization import swap_modelopt_fp8_linears_on_meta


def _all_group_ranks_allow(
    local_eligible: bool,
    process_group: torch.distributed.ProcessGroup | None,
    device: torch.device | str | None,
) -> bool:
    """Return whether every rank in ``process_group`` is locally eligible."""
    if process_group is None:
        return local_eligible
    eligible = torch.tensor([local_eligible], device=device, dtype=torch.uint8)
    torch.distributed.all_reduce(eligible, op=torch.distributed.ReduceOp.MIN, group=process_group)
    return bool(eligible.item())


def _any_dp_shard_rank_needs_guidance_path(
    local_needs_guidance_path: bool,
    dp_shard_group: torch.distributed.ProcessGroup | None,
    device: torch.device | str | None,
) -> bool:
    """Enter the guidance path on every FSDP shard rank when any rank needs it."""
    if dp_shard_group is None:
        return local_needs_guidance_path
    needed = torch.tensor([local_needs_guidance_path], device=device, dtype=torch.uint8)
    torch.distributed.all_reduce(needed, op=torch.distributed.ReduceOp.MAX, group=dp_shard_group)
    return bool(needed.item())


def _uses_lidar_primary_tokenizer(data_batch: dict[str, Any]) -> bool:
    """Return whether the primary ``video`` stream carries LiDAR clips (V0 or V1)."""
    value = data_batch.get("vision_tokenizer_type")
    while isinstance(value, (list, tuple)):
        if not value:
            return False
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value == "lidar"


def _vision_condition_masks(packed_sequence: PackedSequence) -> list[torch.Tensor]:
    """Per-vision-item condition masks, empty when the batch generates no camera stream.

    Packing represents an absent modality as ``None``. The LiDAR-only recipe is the first
    batch to generate without a camera, so the walks over vision items read their masks
    here and simply walk nothing, rather than each testing for the absent stream.
    """
    if packed_sequence.vision is None:
        return []
    assert isinstance(packed_sequence.vision.condition_mask, list), "Vision condition mask must be a list of tensors"
    return packed_sequence.vision.condition_mask


def _densify_action_family(
    raw_actions: list | torch.Tensor | None, raw_families: str | list | tuple | None, expected_rows: int
) -> list[str] | None:
    """Align optional dataset names with the dense, action-bearing rows."""
    if raw_actions is None or raw_families is None:
        return None
    actions = raw_actions if isinstance(raw_actions, list) else [raw_actions]
    families = [raw_families] if isinstance(raw_families, str) else raw_families
    if not isinstance(families, (list, tuple)) or len(actions) != len(families):
        return None

    result: list[str] = []
    for action, family in zip(actions, families, strict=True):
        if isinstance(action, list):
            if len(action) != 1:
                return None
            action = action[0]
        if action is None:
            continue
        if not isinstance(family, str) or not family:
            return None
        result.append(family)
    return result if len(result) == expected_rows else None


class OmniMoTModel(ImaginaireModel):
    """
    Mixture of Transformers (MoT) model to be trained with the flow matching objective
    for visual / sound / action generation.
    """

    def __init__(self, config: OmniMoTModelConfig):
        super().__init__()
        # This rank's post-tokenizer payload, encoded once at the start of each
        # CP data window and retained until every CP rank has owned one slot.
        self._cp_local_training_payload: dict[str, Any] | None = None
        # Current owner rank in the CP data window; advances modulo CP size after
        # each successful training step and returns to 0 for the next window.
        self._cp_window_slot: int = 0
        self.config = config
        log.info(f"OmniMoTModel: config {self.config}")

        # 0. Set up precision
        self.set_precision()

        # 1. Set data keys and data information
        self.set_up_data_key()

        # 2. Text, vision, audio, action tokenizers
        self.set_up_tokenizers()

        # 3. FSDP setup. Note: call this before building the model.
        self.set_up_parallelism()

        # 4. Build the denoiser network
        self.set_up_model()

        # 5. Set up training time scheduler and inference time sampler
        self.set_up_scheduler_and_sampler()

    def set_precision(self) -> None:
        self.precision = PRECISION_TO_TORCH_DTYPE[self.config.precision]
        self.tensor_kwargs = {"device": DEVICE, "dtype": self.precision}
        self.tensor_kwargs_fp32 = {"device": DEVICE, "dtype": torch.float32}
        log.warning(f"OmniMoTModel: precision {self.precision}")

        # Disable TF32 for CUDA matrix multiplications since this may impact model quality.
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False

    def set_up_data_key(self) -> None:
        self.input_video_key = self.config.input_video_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key
        self.input_caption_key = self.config.input_caption_key

    @misc.timer("OmniMoTModel: set_up_tokenizers")
    def set_up_tokenizers(self) -> None:
        """
        Variable names follow the naming convention:
        - tokenizer_<modality_type>_gen if used for generation branch
        - tokenizer_<modality_type>_und if used for understanding branch
        """
        # 1. Text tokenizer
        self.vlm_config = self.config.vlm_config

        # Keep the full processor handy for the I2V branch of
        # ``upsample_captions``: ``apply_chat_template`` on a multimodal
        # processor (e.g. ``Qwen3VLProcessor``) emits ``input_ids`` with
        # the image-placeholder tokens, plus the matched
        # ``pixel_values`` / ``image_grid_thw`` tensors that
        # ``generate_reasoner_text`` consumes for image-conditioned
        # prefill.  ``add_special_tokens`` modifies the tokenizer
        # in-place via ``add_tokens``, so ``_vlm_proc.processor.tokenizer``
        # already reflects the Cosmos3 ``<|vision_start|>`` /
        # ``<|vision_end|>`` additions; nothing else to plumb.  For
        # LLM-only configs (``LLMTokenizerProcessor``) this attribute
        # exists but ``apply_chat_template`` is not implemented, which
        # ``upsample_captions`` checks before driving the multimodal
        # path.

        # Annotated ``Any`` because the live processor is duck-typed
        # across ``Qwen3VLProcessor`` (carries ``apply_chat_template`` for
        # the multimodal path) and ``LLMTokenizerProcessor`` (text-only,
        # no ``apply_chat_template``); the upsampler's runtime branch on
        # ``callable(getattr(self.vlm_processor, "apply_chat_template", None))``
        # is the source of truth.  Without the annotation, basedpyright
        # narrows the ``lazy_instantiate`` overload return to
        # ``list[Unknown]`` and reports ``apply_chat_template`` missing.
        self.vlm_processor: Any = lazy_instantiate(self.vlm_config.tokenizer)

        vlm_tokenizer = self.vlm_processor.tokenizer
        vlm_tokenizer, special_tokens = add_special_tokens(vlm_tokenizer)
        self.vlm_tokenizer = vlm_tokenizer

        self.llm_special_tokens = special_tokens
        self.llm_special_tokens["eos_token_id"] = vlm_tokenizer.eos_token_id

        # 2. Vision tokenizer (images/videos) for generation. Reasoner-only
        # inference does not encode or decode generation latents, so it can
        # leave the VAE unloaded.
        self.tokenizer_vision_gen: VideoTokenizerInterface | None = None
        if self.config.load_vision_tokenizer:
            self.tokenizer_vision_gen = lazy_instantiate(self.config.tokenizer)
            assert self.tokenizer_vision_gen.latent_ch == self.config.state_ch, (
                f"vision tokenizer latent_ch {self.tokenizer_vision_gen.latent_ch} != state_shape {self.config.state_ch}"
            )
            if hasattr(self.tokenizer_vision_gen, "reset_dtype"):
                self.tokenizer_vision_gen.reset_dtype()
        else:
            log.info("Vision tokenizer initialization skipped")

        # 2b. LiDAR VAE for the range-view stream, so one sample can carry camera clips
        # alongside LiDAR clips. Its latent_ch deliberately differs from state_ch: the
        # network gives LiDAR its own projections into and out of the sequence, which is
        # why lidar_state_ch has to agree with this VAE, not with state_ch.
        self.tokenizer_lidar_gen: VideoTokenizerInterface | None = None
        if self.config.lidar_tokenizer is not None and self.config.load_vision_tokenizer:
            self.tokenizer_lidar_gen = lazy_instantiate(self.config.lidar_tokenizer)
            if self.config.lidar_state_ch is None:
                raise ValueError("lidar_state_ch must be set when lidar_tokenizer is configured.")
            if self.tokenizer_lidar_gen.latent_ch != self.config.lidar_state_ch:
                raise ValueError(
                    f"LiDAR tokenizer latent_ch {self.tokenizer_lidar_gen.latent_ch} != "
                    f"lidar_state_ch {self.config.lidar_state_ch}; the LiDAR projection heads are sized "
                    "from lidar_state_ch, so a mismatch would silently project the wrong width."
                )
            if self.config.lidar_fps is None or self.config.lidar_fps <= 0:
                raise ValueError(
                    f"lidar_fps must be a positive sweep rate when lidar_tokenizer is configured, got "
                    f"{self.config.lidar_fps}. Without it the packer has no way to convert a LiDAR "
                    "latent index into seconds and would silently place sweeps on the camera's rate."
                )
            if hasattr(self.tokenizer_lidar_gen, "reset_dtype"):
                self.tokenizer_lidar_gen.reset_dtype()
            log.info(f"LiDAR tokenizer initialized: {type(self.tokenizer_lidar_gen).__name__}")

        # 3. Sound/audio tokenizer (optional)
        if self.config.sound_gen:
            assert self.config.sound_tokenizer is not None, "sound_tokenizer must be provided when sound_gen is True"
            self.tokenizer_sound_gen = lazy_instantiate(self.config.sound_tokenizer)
            assert self.config.sound_dim is not None, "sound_dim must be provided when sound_gen is True"
            assert self.tokenizer_sound_gen.latent_ch == self.config.sound_dim, (
                f"sound tokenizer latent_ch {self.tokenizer_sound_gen.latent_ch} != sound_dim {self.config.sound_dim}"
            )
            # The augmentor derives the synchronized sound prefix from config.sound_latent_fps (dataloader
            # workers never see the tokenizer); pin it to the tokenizer so training and inference agree.
            assert (
                self.config.sound_latent_fps * self.tokenizer_sound_gen.temporal_compression_factor
                == self.tokenizer_sound_gen.sample_rate
            ), (
                f"sound_latent_fps {self.config.sound_latent_fps} != sample_rate/hop "
                f"{self.tokenizer_sound_gen.sample_rate}/{self.tokenizer_sound_gen.temporal_compression_factor}"
            )
            if hasattr(self.tokenizer_sound_gen, "reset_dtype"):
                self.tokenizer_sound_gen.reset_dtype()
            log.info(f"Sound tokenizer initialized: {type(self.tokenizer_sound_gen).__name__}")
        else:
            self.tokenizer_sound_gen = None


    def build_net(
        self,
        dtype: torch.dtype,
        *,
        mp_policy: MixedPrecisionPolicy | None = None,
        lora_enabled: bool | None = None,
    ) -> torch.nn.Module:
        """Build the VFM network, parallelize it, and materialize it on device.

        Args:
            dtype: dtype the network's parameters and buffers are *stored* in -- what the
                optimizer steps and what the checkpoint holds. With ``mp_policy`` this is
                the FSDP master dtype and is higher precision than the compute dtype.
            mp_policy: FSDP2 mixed-precision policy (see :func:`parallelize_vfm_network`).
                ``None`` runs forward/backward in ``dtype`` itself.
            lora_enabled: Override for ``config.lora_enabled``.
        """
        # Build model network and parallelize it.
        lora_enabled = self.config.lora_enabled if lora_enabled is None else lora_enabled
        with torch.device("meta"):
            assert self.vlm_config.model_instance is not None, "Model instance should be specified"

            sample_lbl_enabled = self.config.lbl.method == "sample" and (
                self.config.lbl.coeff_und is not None or self.config.lbl.coeff_gen is not None
            )
            lbl_init_kwargs: dict[str, LBLConfig] = {}
            if sample_lbl_enabled:
                # Sample LBL is MoE-only. Dense wrappers reject this kwarg, so fail here
                # with a clear message instead of a TypeError inside lazy_instantiate.
                model_target = self.vlm_config.model_instance["_target_"]
                model_cls = locate(model_target) if isinstance(model_target, str) else model_target
                assert callable(model_cls), f"Expected a callable model target, got {model_target!r}"
                assert "lbl_config" in inspect.signature(model_cls).parameters, (
                    f"lbl.method='sample' requires an MoE model; got {getattr(model_cls, '__qualname__', model_cls)}"
                )
                lbl_init_kwargs = {"lbl_config": self.config.lbl}
            language_model = lazy_instantiate(self.vlm_config.model_instance, **lbl_init_kwargs)

            # NOTE: We pass "RF timesteps" to the network in the same scale as the scheduler
            # (i.e., roughly [0, num_train_timesteps]). The MoT network expects to internally
            # rescale timesteps before embedding; avoid hard-coding 1e-3 by computing it from
            # the configured scheduler resolution.
            num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps
            network_config = Cosmos3VFMNetworkConfig(
                vlm_config=language_model.config,
                latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
                latent_downsample_factor=self.config.latent_downsample_factor,
                latent_channel_size=self.config.state_ch,
                lidar_latent_channel_size=self.config.lidar_state_ch,
                max_latent_h=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_w=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_t=self.config.state_t,
                enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
                enable_vision_modality_embeddings=(
                    self.config.diffusion_expert_config.enable_vision_modality_embeddings
                ),
                enable_media_modality_embedding=(self.config.diffusion_expert_config.enable_media_modality_embedding),
                enable_action_modality_embedding=(self.config.diffusion_expert_config.enable_action_modality_embedding),
                enable_sound_modality_embedding=(self.config.diffusion_expert_config.enable_sound_modality_embedding),
                base_fps=self.config.diffusion_expert_config.base_fps,
                vision_gen=self.config.vision_gen,
                action_gen=self.config.action_gen,
                sound_gen=self.config.sound_gen,
                joint_attn_implementation=self.config.joint_attn_implementation,
                use_multiview_flex_attention=self.config.flex_attention.enabled,
                flex_attention_backend=self.config.flex_attention.backend,
                attention_scope=self.config.flex_attention.mask.attention_scope,
                control_attends_sensor=self.config.flex_attention.mask.control_attends_sensor,
                decomposed_temporal_window_seconds=self.config.flex_attention.mask.decomposed_temporal_window_seconds,
                timestep_scale=1.0 / float(num_train_timesteps) * self.config.diffusion_expert_config.timestep_range,
                action_dim=self.config.max_action_dim,
                num_embodiment_domains=self.config.num_embodiment_domains,
                action_io_projector_type=self.config.action_io_projector_type,
                temporal_compression_factor_vision=(
                    self.tokenizer_vision_gen.temporal_compression_factor
                    if self.tokenizer_vision_gen is not None
                    else self.config.tokenizer.temporal_compression_factor
                ),
                natten_parameter_list=self.config.natten_parameter_list,
                video_temporal_causal=self.config.video_temporal_causal,
                # Sound generation parameters
                sound_dim=self.config.sound_dim,
                sound_latent_fps=self.config.sound_latent_fps,
                enable_input_bias=self.config.enable_input_bias,
            )
            network_config._attn_implementation_internal = "eager"
            net = Cosmos3VFMNetwork(
                language_model=language_model,
                config=network_config,
            )
            net.pad_for_cuda_graphs = self.config.compile.use_cuda_graphs

            # Inject LoRA BEFORE FSDP wrap, while still on meta device. The
            # injector must see unsharded Linear shapes; injecting post-FSDP causes
            # lora_B to be created at the per-rank shard size and crashes at
            # forward time. See `OmniMoTModel.add_lora` for details.
            if lora_enabled:
                net = self.add_lora(
                    net,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_target_modules=self.config.lora_target_modules,
                )

        # Swap ModelOpt FP8 linears BEFORE FSDP wrap and materialization, for the
        # same reason LoRA is injected early. `fully_shard` wraps the whole network
        # into one parameter group, so a linear replaced afterwards leaves the group
        # holding a stale parameter; and `to_empty` below is what sets peak memory,
        # which at bf16 shapes exceeds a single 80 GB device for a Super-class model.
        if self.config.quantization.modelopt_fp8_checkpoint_path:
            swap_modelopt_fp8_linears_on_meta(net, self.config.quantization.modelopt_fp8_target_fqns)

        self.install_attention_dispatch(net)

        # Cast while still on meta (free -- no data to convert) and BEFORE sharding: each
        # FSDP2 unit records its parameters' dtype as it is built, so a cast afterwards
        # would leave ``mp_policy.reduce_dtype`` disagreeing with the sharded params.
        net = net.to(dtype=dtype)

        net = parallelize_vfm_network(
            net,
            parallel_dims=self.parallel_dims,
            compile_config=self.config.compile,
            ac_config=self.config.activation_checkpointing,
            attention_io_layout=getattr(self.config, "attention_io_layout", "sequence_sharded"),
            mp_policy=mp_policy,
        )

        with misc.timer("meta to cuda and broadcast model states"):
            net.to_empty(device=DEVICE)
            if DEVICE == Device.CUDA:
                # Weight initialization is not needed for other devices (cpu,
                # meta), since they are only for checkpoint conversion and smoke
                # tests.
                net.init_weights(buffer_device=DEVICE)
                if lora_enabled:
                    self._init_lora_weights_post_materialization(net)

        return net

    def load_pretrained_model_if_needed(
        self,
        *,
        has_resumable_checkpoint: bool,
        has_load_path: bool,
    ) -> None:
        """Conditionally seed pretrained understanding/reasoner weights at startup.

        OmniMoT has two weight groups: the understanding/reasoner pathway (the
        ``language_model`` backbone, e.g. Qwen3-VL / Cosmos-Reason) and the
        generation pathway (the diffusion MoE experts). This hook runs after the
        model is built and after DCP has had a chance to restore a checkpoint. It
        decides (a) whether the understanding weights still need to be seeded from
        the pretrained HuggingFace source, and (b) whether those weights must be
        copied into the generation pathway.

        Args:
            has_resumable_checkpoint: A ``latest_checkpoint.txt`` exists in the
                load directory, i.e. DCP has already restored the full model from a
                mid-run checkpoint. The understanding weights are normally present
                in such a checkpoint, so the HF load is skipped -- unless
                ``exclude_reasoner_weights_from_checkpoint`` is set, in which case
                those weights were never checkpointed and must be re-seeded here.
            has_load_path: ``checkpoint.load_path`` is set, i.e. DCP has loaded the
                full model from a warm-start path. The understanding weights are
                still re-seeded from HF (e.g. to swap Qwen3-VL -> Cosmos-Reason),
                but the understanding->generation copy is skipped because the
                generation pathway was already populated from ``load_path``.

        The gates combine into three startup scenarios:
          1. Fresh init (neither gate set): seed understanding weights from HF and
             copy them into the generation pathway.
          2. Warm-start (``has_load_path`` only): re-seed understanding weights,
             skip the understanding->generation copy.
          3. Resume (``has_resumable_checkpoint`` set): skip everything, unless
             ``exclude_reasoner_weights_from_checkpoint`` forces re-seeding the
             understanding weights (the copy is still skipped).
        """
        # A checkpoint of any kind (mid-run resume or warm-start load_path) means
        # the generation pathway is already populated, so the understanding->
        # generation copy further below must be skipped.
        has_checkpoint = has_resumable_checkpoint or has_load_path

        pretrained_weights = self.vlm_config.pretrained_weights

        if self.config.exclude_reasoner_weights_from_checkpoint and not pretrained_weights.enabled:
            raise ValueError(
                "Reasoner weights must be loaded from pretrained checkpoint when "
                "exclude_reasoner_weights_from_checkpoint is True. However, "
                "pretrained_weights.enabled is set to False."
            )

        # Seed understanding weights from HF only when the source is enabled and
        # either there is no resumable checkpoint to restore them from, or they
        # were deliberately excluded from the checkpoint (so it cannot contain
        # them and they must be reloaded from the pretrained source).
        load_pretrained_weights = pretrained_weights.enabled and (
            self.config.exclude_reasoner_weights_from_checkpoint or not has_resumable_checkpoint
        )
        if not load_pretrained_weights:
            return

        # Load the language_model (understanding/reasoner backbone) safetensors
        # into the given net, respecting the active parallelism layout.
        def _load_language_model(net: torch.nn.Module):
            load_language_model_safetensors(
                model=net.language_model,
                checkpoint_path=pretrained_weights.backbone_path,
                credential_path=pretrained_weights.credentials_path,
                parallel_dims=self.parallel_dims,
                checkpoint_format=pretrained_weights.checkpoint_format,
            )

        log.info(f"Loading reasoner pathway weights from {pretrained_weights.backbone_path}")
        _load_language_model(self.net)
        # Keep the EMA copy in sync with the freshly seeded understanding weights.
        if self.config.ema.enabled:
            _load_language_model(self.net_ema)
        log.info("Successfully loaded reasoner pathway weights.")

        # Copy understanding -> generation only on a truly fresh init: the config
        # must request it and no checkpoint (resume or warm-start) may have already
        # populated the generation pathway.
        load_pretrained_diffusion_weights = (
            self.config.diffusion_expert_config.load_weights_from_pretrained and not has_checkpoint
        )
        if not load_pretrained_diffusion_weights:
            log.info("Skipping diffusion pathway weights copying.")
            return

        # init_moe() copies the understanding-pathway weights into the generation
        # (diffusion MoE) experts so generation starts from the pretrained backbone.
        log.info("Copying understanding pathway weights to generation pathway.")
        self.net.language_model.init_moe()
        if self.config.ema.enabled:
            self.net_ema.language_model.init_moe()
        log.info("Successfully copied understanding pathway weights to generation pathway.")

    @misc.timer("OmniMoTModel: set_up_model")
    def set_up_model(self):
        assert hasattr(self, "parallel_dims"), "parallel_dims must be set"
        config = self.config

        with misc.timer("Creating PyTorch model and ema if enabled"):
            master_dtype, mp_policy = _get_master_dtype_and_mp_policy(config, self.precision)
            self.net = self.build_net(dtype=master_dtype, mp_policy=mp_policy)
            self._param_count = count_params(self.net, verbose=False)

            self._uses_aux_loss_free_load_balancing: bool = uses_aux_loss_free_load_balancing(self.net)
            self._uses_ema_router_bias: bool = uses_ema_router_bias(self.net)

            if config.ema.enabled:
                self.net_ema = self.build_net(dtype=torch.float32)
                self.net_ema.requires_grad_(False)

                self.net_ema_worker = DTensorFastEmaModelUpdater()

                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

        self.set_up_memory()

        torch.cuda.empty_cache()

    def install_attention_dispatch(self, net: torch.nn.Module) -> None:
        """Install a custom attention dispatch function on the network.

        Called during ``build_net()`` after the network is constructed but
        before parallelization.  The base implementation is a no-op;
        ``OmniMoTCausalModel`` overrides this to install
        ``dispatch_attention_with_memory`` on every attention layer.
        """
        pass

    def set_up_memory(self) -> None:
        """Initialize memory state used during training (e.g. KV caches).

        The base implementation is a no-op.  ``OmniMoTCausalModel`` overrides
        this to allocate a KV cache.
        """
        pass

    def set_up_parallelism(self) -> None:
        """Set up the fsdp for the model."""
        self.parallel_dims = ParallelDims(
            enable_inference_mode=self.config.parallelism.enable_inference_mode,
            world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            dp_shard=self.config.parallelism.data_parallel_shard_degree,
            cfgp=self.config.parallelism.cfg_parallel_shard_degree,
            cp=self.config.parallelism.context_parallel_shard_degree,
            lb=self.config.parallelism.vae_load_balance_group_size,
        )
        self.parallel_dims.build_meshes(device_type=DEVICE)

    def set_up_scheduler_and_sampler(self):
        # Get shift values - support both int and dict-based resolution lookup.
        rf_config = self.config.rectified_flow_training_config

        def resolve_shift(shift_config: Any, dynamic_shift_key: str) -> int:
            if isinstance(shift_config, int):
                return shift_config

            shift_dict = dict(shift_config)
            # Token-count-based dicts hold no per-resolution value; the shift is derived per batch
            # in `_get_train_noise_level_vision`, so initialize the scheduler with the identity
            # shift instead of failing the resolution lookup.
            if dynamic_shift_key in shift_dict:
                return 1
            resolution = self.config.resolution
            if resolution not in shift_dict:
                raise ValueError(
                    f"Resolution '{resolution}' not found in shift dict. Available resolutions: {list(shift_dict.keys())}"
                )
            return shift_dict[resolution]

        shift_video = resolve_shift(rf_config.shift, "dynamic_shift_base_num_tokens_video")
        _shift_image_cfg = getattr(rf_config, "shift_image", None)
        shift_image = resolve_shift(
            _shift_image_cfg if _shift_image_cfg is not None else rf_config.shift,
            "dynamic_shift_base_num_tokens_image",
        )

        # Rectified Flow timestep scheduler and sampler for training (separate for image and video)
        if self.config.vision_gen:
            self.rectified_flow_image = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=rf_config.train_time_image_distribution,
                use_dynamic_shift=rf_config.use_dynamic_shift,
                shift=shift_image,
                train_time_weight_method=rf_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )
            self.rectified_flow_video = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=rf_config.train_time_video_distribution,
                use_dynamic_shift=rf_config.use_dynamic_shift,
                shift=shift_video,
                train_time_weight_method=rf_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )
        if self.config.action_gen:
            self.rectified_flow_action = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_action_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift_video,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )
        if self.config.sound_gen:
            self.rectified_flow_sound = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_sound_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift_video,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )

        # Denoising sampler (solver) for inference
        assert self.config.rectified_flow_inference_config.scheduler_type in ["unipc", "edm"]
        if self.config.rectified_flow_inference_config.scheduler_type == "unipc":
            unipc_sampler_config = UniPCSamplerConfig(
                num_train_timesteps=self.config.rectified_flow_inference_config.num_train_timesteps,
                shift=self.config.rectified_flow_inference_config.shift,
                use_dynamic_shifting=self.config.rectified_flow_inference_config.use_dynamic_shifting,
            )
            self.sampler = UniPCSampler(cfg=unipc_sampler_config, tensor_kwargs=self.tensor_kwargs)
        else:
            self.sampler = EDMSampler()

        # Fixed-step sampler for distilled models (None for base models)
        if self.config.fixed_step_sampler_config is not None:
            cfg = self.config.fixed_step_sampler_config
            self.fixed_step_sampler = FixedStepSampler(
                t_list=list(cfg.t_list),
                sample_type=cfg.sample_type,
                num_train_timesteps=float(self.config.rectified_flow_inference_config.num_train_timesteps),
            )
        else:
            self.fixed_step_sampler = None

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            optimizer_config (LazyDict): The lazy config for the optimizer.
            scheduler_config (LazyDict): The lazy config for the learning rate scheduler.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """

        optimizer = lazy_instantiate(optimizer_config, model=self)
        scheduler = lazy_instantiate(scheduler_config, optimizer=optimizer)
        return optimizer, scheduler

    def _derive_include_end_of_generation_token(self) -> bool:
        impl = self.config.joint_attn_implementation
        assert impl in ("two_way", "three_way"), (
            f"Invalid joint_attn_implementation: {impl}. Must be 'two_way' or 'three_way'."
        )
        return False

    # ------------------------ training hooks ------------------------
    def on_before_optimizer_step(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """Run aux-loss-free load balancing + EMA router de-sink on the gen-tower MoE blocks."""
        del scheduler, optimizer

        dp_mesh = self.parallel_dims.dp_mesh if self.parallel_dims else None

        # Aux-loss-free load balancing (only when enabled).
        if self._uses_aux_loss_free_load_balancing:
            update_expert_biases(
                net=self.net,
                device_mesh=dp_mesh,
            )

        # EMA-tracked token-constant de-sink bias (a checkpointed buffer). Independent of
        # expert-load bias correction; the update reduces the per-step token stats across the DP mesh so
        # the buffer is identical on every rank.
        if self._uses_ema_router_bias:
            update_router_biases(net=self.net, device_mesh=dp_mesh)
        self.sync_ema_buffers()

    def sync_ema_buffers(self) -> None:
        """Mirror checkpointed MoE buffers omitted by the parameter-only EMA worker.

        Callers own any cross-rank source-buffer update before this seam; it only
        copies the resulting persistent state into the checkpointed EMA network.
        """
        if not self.config.ema.enabled:
            return
        if self._uses_aux_loss_free_load_balancing:
            sync_expert_biases_to_ema(net=self.net, net_ema=self.net_ema)
        if self._uses_ema_router_bias:
            sync_router_biases_to_ema(net=self.net, net_ema=self.net_ema)

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        if self.config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    # ------------------------ helpers ------------------------

    def _pack_input_sequence(
        self,
        sequence_plans: list[SequencePlan],
        input_text_indexes: list[list[int]],
        gen_data_clean: GenerationDataClean,
        input_timesteps: torch.Tensor,
        include_end_of_generation_token: bool = False,
        skip_text_tokens: bool = False,
        initial_mrope_temporal_offset: int | float = 0,
    ) -> PackedSequence:
        """Wrap ``pack_input_sequence`` with all config-derived args pre-filled.

        Centralises the config-derived positional/embedding args so callers only
        supply the four per-call arguments (sequence_plans, text tokens, data, timesteps)
        plus three optional flags.
        """
        assert self.tokenizer_vision_gen is not None
        return pack_input_sequence(
            sequence_plans=sequence_plans,
            input_text_indexes=input_text_indexes,
            gen_data_clean=gen_data_clean,
            input_timesteps=input_timesteps,
            special_tokens=self.llm_special_tokens,
            latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
            skip_text_tokens=skip_text_tokens,
            include_end_of_generation_token=include_end_of_generation_token,
            unified_3d_mrope_reset_spatial_ids=self.config.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=self.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
            base_fps=float(self.config.diffusion_expert_config.base_fps),
            sound_base_temporal_compression_factor=self.config.diffusion_expert_config.sound_base_temporal_compression_factor,
            temporal_compression_factor=self.tokenizer_vision_gen.temporal_compression_factor,
            lidar_temporal_compression_factor=(
                self.tokenizer_lidar_gen.temporal_compression_factor if self.tokenizer_lidar_gen is not None else None
            ),
            vision_temporal_position_mode=self.config.diffusion_expert_config.vision_temporal_position_mode,
            align_temporal_positions_across_views=(
                self.config.diffusion_expert_config.align_temporal_positions_across_views
            ),
            video_temporal_causal=self.config.video_temporal_causal,
            action_dim=self.config.max_action_dim,
            initial_mrope_temporal_offset=initial_mrope_temporal_offset,
        )

    def _get_temporal_positions_vision(
        self,
        raw_state_vision: list[torch.Tensor],
        x0_tokens_vision: list[torch.Tensor],
        num_views_per_vision_item: list[int] | None = None,
        frames_per_vision_item: list[int] | None = None,
    ) -> list[torch.Tensor] | None:
        """Return optional per-latent temporal coordinates for vision tokens.

        Multiview items are currently flattened camera-major. Compute UniAE
        positions for one independently encoded camera clip, retain explicit
        view/time axes, and flatten them in the same order as the latents. A
        future timestamp-major layout only needs to transpose those two axes
        before flattening.
        """
        mode = self.config.diffusion_expert_config.vision_temporal_position_mode
        if mode == "latent_index":
            return None
        if mode != "uniae_source_right_edge":
            raise ValueError(
                "Unsupported vision_temporal_position_mode: "
                f"{mode}. Expected 'latent_index' or 'uniae_source_right_edge'."
            )

        assert self.tokenizer_vision_gen is not None
        if (num_views_per_vision_item is None) != (frames_per_vision_item is None):
            raise ValueError("num_views_per_vision_item and frames_per_vision_item must be provided together.")
        if num_views_per_vision_item is None:
            num_views_per_vision_item = [1] * len(raw_state_vision)
            frames_per_vision_item_optional: list[int | None] = [None] * len(raw_state_vision)
        else:
            assert frames_per_vision_item is not None
            if len(num_views_per_vision_item) != len(raw_state_vision) or len(frames_per_vision_item) != len(
                raw_state_vision
            ):
                raise ValueError(
                    "Multiview temporal-position metadata must align with flattened vision items: "
                    f"got {len(num_views_per_vision_item)} view counts, {len(frames_per_vision_item)} frame "
                    f"counts, and {len(raw_state_vision)} vision items."
                )
            frames_per_vision_item_optional = list(frames_per_vision_item)

        temporal_positions_vision: list[torch.Tensor] = []
        for raw_state_vision_i, x0_tokens_vision_i, num_views, frames_per_view in zip(
            raw_state_vision,
            x0_tokens_vision,
            num_views_per_vision_item,
            frames_per_vision_item_optional,
            strict=True,
        ):
            if raw_state_vision_i.dim() == 5:
                num_pixel_frames = int(raw_state_vision_i.shape[2])
            elif raw_state_vision_i.dim() == 4:
                num_pixel_frames = int(raw_state_vision_i.shape[1])
            else:
                raise ValueError(
                    "raw_state_vision items must have shape [B,C,T,H,W] or [C,T,H,W], "
                    f"got shape {tuple(raw_state_vision_i.shape)}."
                )
            num_latent_frames = int(x0_tokens_vision_i.shape[2])
            frame_h = int(raw_state_vision_i.shape[-2])
            frame_w = int(raw_state_vision_i.shape[-1])
            resolution = get_vision_data_resolution((frame_h, frame_w))

            if num_views > 1:
                if frames_per_view is None:
                    raise ValueError("frames_per_view is required when num_views is greater than one.")
                expected_pixel_frames = num_views * frames_per_view
                if num_pixel_frames != expected_pixel_frames:
                    raise ValueError(
                        "Multiview vision length must equal num_views * frames_per_view when computing temporal "
                        f"positions: got T={num_pixel_frames}, num_views={num_views}, "
                        f"frames_per_view={frames_per_view}."
                    )
                if num_latent_frames % num_views != 0:
                    raise ValueError(
                        "Multiview latent length must be divisible by num_views when computing temporal positions: "
                        f"got T={num_latent_frames}, num_views={num_views}."
                    )
                latent_frames_per_view = num_latent_frames // num_views
                temporal_positions_per_view = self.tokenizer_vision_gen.get_latent_temporal_positions(
                    num_pixel_frames=frames_per_view,
                    resolution=resolution,
                    num_latent_frames=latent_frames_per_view,
                )  # [T_latent_view]
                if temporal_positions_per_view is None:
                    raise ValueError(
                        f"{type(self.tokenizer_vision_gen).__name__} does not support "
                        f"vision_temporal_position_mode={mode}."
                    )
                if temporal_positions_per_view.shape[0] != latent_frames_per_view:
                    raise ValueError(
                        "Vision temporal position count must match latent frames per view: "
                        f"got {temporal_positions_per_view.shape[0]} positions for "
                        f"{latent_frames_per_view} latent frames."
                    )
                temporal_positions_by_view = temporal_positions_per_view.unsqueeze(0).expand(
                    num_views, -1
                )  # [V,T_latent_view]
                temporal_positions = temporal_positions_by_view.reshape(-1)  # [V*T_latent_view]
            else:
                temporal_positions = self.tokenizer_vision_gen.get_latent_temporal_positions(
                    num_pixel_frames=num_pixel_frames,
                    resolution=resolution,
                    num_latent_frames=num_latent_frames,
                )  # [T_latent]

            if temporal_positions is None:
                raise ValueError(
                    f"{type(self.tokenizer_vision_gen).__name__} does not support vision_temporal_position_mode={mode}."
                )
            if temporal_positions.shape[0] != num_latent_frames:
                raise ValueError(
                    "Vision temporal position count must match latent frames: "
                    f"got {temporal_positions.shape[0]} positions for {num_latent_frames} latent frames."
                )
            temporal_positions = temporal_positions.to(
                device=x0_tokens_vision_i.device,
                dtype=torch.float32,
            )  # [T_latent]
            temporal_positions_vision.append(temporal_positions)
        return temporal_positions_vision

    # ------------------------ training ------------------------

    def memory_init_training(
        self,
        gen_data_clean: GenerationDataClean,
        data_batch: dict[str, torch.Tensor],
        input_text_indexes: list[list[int]],
    ) -> tuple[GenerationDataClean, dict]:
        """Prepare the memory for a single training step.

        Called at the start of ``training_step`` to give the causal subclass
        an injection point for memory-based segment handling (frame trimming,
        segment bookkeeping, cache resets, packing overrides).

        The base implementation returns *gen_data_clean* unmodified and a
        default memory_info dict that does not support memory-backed training.

        The ``skip_text`` and ``initial_temporal`` offset fields are required,
        and are used for both sequence packing and memory.

        Returns:
            ``(gen_data_clean, memory_info)`` where *memory_info* is a dict with keys:
            ``skip_text``, ``initial_temporal_offset``
        """
        return gen_data_clean, {
            "skip_text": False,
            "initial_temporal_offset": 0,
        }

    def build_memory_state(
        self,
        packed_seq: PackedSequence,
        memory_info: dict,
    ) -> MemoryState | None:
        """Construct a ``MemoryState`` from a packed sequence and context dict.

        Called after packing in ``training_step()``, and before ``denoise()``
        in AR inference.  The base implementation returns ``None`` (no
        persistent memory).  ``OmniMoTCausalModel`` overrides this to build
        the appropriate ``ARMemoryState`` or ``KVCacheTrainMemoryState``.

        Args:
            packed_seq: The packed multi-modal sequence produced by
                ``_pack_input_sequence``.
            memory_info: Context dict returned by ``memory_init_training()``
                (for the training path) or constructed by the AR inference
                caller.  See ``memory_init_training()`` for the base keys.
        """
        return None

    def pre_noise_memory_hook(
        self,
        packed_sequence: PackedSequence,
        gen_data_clean: GenerationDataClean,
        memory_info: dict,
    ) -> dict:
        """Hook called after sequence packing and before noising. Returns (possibly updated) memory_info.

        The packed sequence still contains clean tokens at this point.
        Override in subclasses to run a clean forward pass (e.g. for teacher forcing).
        """
        return memory_info

    def _prepare_training_data(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[
        list[list[int]],
        list[SequencePlan],
        GenerationDataClean,
        dict,
        list[str] | None,
        list[tuple[int, int, int]],
    ]:
        """Run local tokenization and return the typed inputs needed by training.

        ``balance_vae_encode=True`` lets :meth:`get_data_and_condition` spread this step's
        VAE-encode COMPUTE across the ``lb`` group (see ``models/mot/vae_load_balance.py``)
        rather than having every rank encode exactly its own pixels. Only the compute moves:
        ``encode(pixels) -> latent`` is a pure function of the pixel tensor, so no sample
        changes owner and every other per-sample field (caption, image_size, sequence-packing
        slot, sound/action/lidar counterparts) stays paired with its sample exactly as it
        always was.

        Training is the only caller that opts in. Inference and the visualization callbacks
        leave it off deliberately: balancing runs collectives over the whole ``lb`` group, and
        those paths are not guaranteed to run on every rank of it.
        """
        input_text_indexes = self._load_and_tokenize_text_data(data_batch, iteration)
        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch,
            input_video_key=self.input_video_key,
            input_image_key=self.input_image_key,
        )
        per_camera_vae_encoding = "enable_per_camera_vae_encoding" in data_batch
        gen_data_clean = self.get_data_and_condition(
            data_batch,
            retain_raw_state_vision=not per_camera_vae_encoding,
            balance_vae_encode=True,
        )
        # Account for every item encoded by the VAE, even if memory initialization
        # later removes or truncates items before sequence packing.
        vae_pixel_shapes = get_vae_pixel_shapes(gen_data_clean.raw_state_vision)
        gen_data_clean, memory_info = self.memory_init_training(gen_data_clean, data_batch, input_text_indexes)

        # image_size[i] may be (1, 4) from IterativeJointDataLoader or (4,) from custom_collate_fn.
        if "image_size" in data_batch:
            data_resolutions: list[str] | str | None = []
            for i in range(gen_data_clean.batch_size):
                img_size = data_batch["image_size"][i]
                if img_size.dim() == 2:
                    img_size = img_size[0]
                target_h = int(img_size[0].item())
                target_w = int(img_size[1].item())
                data_resolutions.append(get_vision_data_resolution((target_h, target_w)))
        elif sequence_plans and not any(plan.has_vision for plan in sequence_plans):
            # A vision-less batch -- the LiDAR-only recipe -- still samples the per-sample vision
            # timestep, because that is the clock LiDAR borrows. There is no camera frame to
            # bucket, so the noise-schedule shift comes from the run's configured resolution.
            data_resolutions = self.config.resolution
        else:
            data_resolutions = None

        if per_camera_vae_encoding:
            # Training needs only the encoded latents after memory initialization.
            gen_data_clean.raw_state_vision = None
        return input_text_indexes, sequence_plans, gen_data_clean, memory_info, data_resolutions, vae_pixel_shapes

    @staticmethod
    def _pack_training_payload(
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        memory_info: dict,
        data_resolutions: list[str] | str | None,
        vae_pixel_shapes: list[tuple[int, int, int]],
    ) -> dict[str, Any]:
        """Convert typed training inputs into a plain tree for CP broadcast."""
        # Keep shallow tensor references
        gen_data_clean_payload = {
            field.name: getattr(gen_data_clean, field.name) for field in dataclasses.fields(GenerationDataClean)
        }
        # Raw pixels/audio/action/LiDAR are unused after tokenization; omit them from the CP cache.
        for key in ("raw_state_vision", "raw_state_sound", "raw_state_action", "raw_state_lidar"):
            gen_data_clean_payload.pop(key)
        return {
            "input_text_indexes": input_text_indexes,
            "sequence_plans": [dataclasses.asdict(plan) for plan in sequence_plans],
            "gen_data_clean": gen_data_clean_payload,
            "memory_info": memory_info,
            "data_resolutions": data_resolutions,
            "vae_pixel_shapes": vae_pixel_shapes,
        }

    @staticmethod
    def _unpack_training_payload(
        payload: dict[str, Any],
    ) -> tuple[
        list[list[int]],
        list[SequencePlan],
        GenerationDataClean,
        dict,
        list[str] | None,
        list[tuple[int, int, int]],
    ]:
        expected = {
            "input_text_indexes",
            "sequence_plans",
            "gen_data_clean",
            "memory_info",
            "data_resolutions",
            "vae_pixel_shapes",
        }
        if set(payload) != expected:
            raise ValueError(f"Unexpected CP payload keys: {set(payload) ^ expected}")
        input_text_indexes = payload["input_text_indexes"]
        sequence_plans = [SequencePlan(**plan) for plan in payload["sequence_plans"]]
        gen_data_clean = GenerationDataClean(**payload["gen_data_clean"])
        memory_info = payload["memory_info"]
        data_resolutions = payload["data_resolutions"]
        vae_pixel_shapes = payload["vae_pixel_shapes"]
        return input_text_indexes, sequence_plans, gen_data_clean, memory_info, data_resolutions, vae_pixel_shapes

    def _get_training_inputs(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[
        list[list[int]],
        list[SequencePlan],
        GenerationDataClean,
        dict,
        list[str] | None,
        list[tuple[int, int, int]],
    ]:
        """Prepare inputs using a context-parallel (CP) data-window rotation.

        A CP group of size ``C`` processes a window of ``C`` rank-local batches
        over ``C`` consecutive training steps. Each position in that window is
        a *slot*, and CP rank ``s`` owns slot ``s``.

        At slot 0, every rank independently tokenizes/encodes its own raw batch
        once and stores the post-tokenizer payload in
        ``self._cp_local_training_payload``. At slot ``s``, owner rank ``s``
        broadcasts its cached payload, and all CP ranks execute that training
        step on the same owner batch (the packed sequence is sharded across
        those ranks later).

        ``self._cp_window_slot`` advances modulo ``C`` after each step. After
        slot ``C - 1``, the cache is cleared and the slot resets to 0, beginning
        the next window. Thus each rank fetches and encodes one local batch per
        window; later slots reuse the cache rather than encoding raw data again.
        """
        cp_enabled = self.parallel_dims is not None and self.parallel_dims.cp_enabled
        if not cp_enabled:
            if self.parallel_dims is None or self.parallel_dims.cp_rank == 0:
                self._update_train_stats(data_batch)
            return self._prepare_training_data(data_batch, iteration)

        cp_size = self.parallel_dims.cp_mesh.size()
        cp_window_slot = self._cp_window_slot
        if not 0 <= cp_window_slot < cp_size:
            raise ValueError(f"CP data window slot must be in [0, {cp_size}), got {cp_window_slot}.")
        # Slot 0: every CP rank encodes its local batch once and caches the payload.
        # Later slots broadcast the current owner's cache; raw data is ignored.
        if cp_window_slot == 0:
            local_training_data = self._prepare_training_data(data_batch, iteration)
            self._cp_local_training_payload = self._pack_training_payload(*local_training_data)
        if self._cp_local_training_payload is None:
            raise RuntimeError("CP training payload cache is empty before its data window is complete.")
        payload = broadcast_context_parallel_object(
            self._cp_local_training_payload,
            self.parallel_dims,
            owner_rank=cp_window_slot,
        )
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a dictionary CP training payload, got {type(payload).__name__}.")
        input_text_indexes, sequence_plans, gen_data_clean, memory_info, data_resolutions, vae_pixel_shapes = (
            self._unpack_training_payload(payload)
        )
        if cp_window_slot == cp_size - 1:
            self._cp_local_training_payload = None
        if self.parallel_dims.cp_rank == 0:
            self._update_train_stats_from_processed_batch(gen_data_clean)
        self._cp_window_slot = (cp_window_slot + 1) % cp_size
        return input_text_indexes, sequence_plans, gen_data_clean, memory_info, data_resolutions, vae_pixel_shapes

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the rectified-flow (flow-matching) model.

        This method executes one iteration of the model's training. It involves:
        1. Tokenizing generation modalities (vision/action/sound) into latents (tokens).
        2. Sampling a training timestep (t) for each modality and constructing noised latents (xt)
           per the rectified-flow formulation.
        3. Packing text + generation tokens into a single sequence and running the MoT network to predict
           the flow field velocity at the given t.
        4. Computing flow-matching loss (plus optional auxiliary load-balancing losses).

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        """
        input_text_indexes, sequence_plans, gen_data_clean, memory_info, data_resolutions, vae_pixel_shapes = (
            self._get_training_inputs(data_batch, iteration)
        )

        # Calculate number of tokens per sample (before 2x2 merge) for dynamic shift
        # gen_data_clean.x0_tokens_vision: B, C, T, H, W
        assert all(x.shape[0] == 1 for x in gen_data_clean.x0_tokens_vision), (
            "Batch size must be 1 for individual samples"
        )
        num_tokens_per_sample = [x.shape[2] * x.shape[3] * x.shape[4] for x in gen_data_clean.x0_tokens_vision]

        # Sample a random noise level (sigma) and corresponding interpolation coefficient ("timesteps" in RF)
        # Apply shift per sample based on each sample's resolution
        num_vision_latent_frames = [x.shape[2] for x in gen_data_clean.x0_tokens_vision]
        timesteps_vision, sigmas_vision = self._get_train_noise_level_vision(
            batch_size=gen_data_clean.batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            resolutions=data_resolutions,
            num_vision_latent_frames=num_vision_latent_frames,
            num_tokens=num_tokens_per_sample,
            iteration=iteration,
        )  # [B, T_vis] each

        # Optional independent action schedule (sampled from rectified_flow_action with
        # action-specific shift). Only active when the config opts in and the batch contains
        # action data.
        #
        # Mixed-batch indexing: gen_data_clean.x0_tokens_action (and every packed_sequence.action.*
        # field) is *dense* — one entry per sample with has_action=True, in the original batch order
        # but skipping non-action samples. To feed each dense action entry its sample's sigma, we
        # sample σ for the full batch and reindex with action_sample_indices (the batch positions
        # of action-bearing samples). This avoids the mismatch that happens when, e.g., batch
        # sample 1 has action but the dense entry 0 would otherwise read σ from batch position 0.
        rf_cfg = self.config.rectified_flow_training_config
        action_sample_indices = [i for i, plan in enumerate(sequence_plans) if plan.has_action]
        if rf_cfg.independent_action_schedule and action_sample_indices:
            ts_full, sg_full = self._get_train_noise_level_action(
                batch_size=gen_data_clean.batch_size, iteration=iteration
            )  # [B, 1] each
            idx = torch.tensor(action_sample_indices, dtype=torch.long)  # [n_action]
            timesteps_action = ts_full[idx]  # [n_action, 1]
            sigmas_action = sg_full[idx]  # [n_action, 1]
        else:
            timesteps_action, sigmas_action = (None, None)

        # Optional independent sound schedule: sample a scalar sound sigma per batch
        # slot, then reindex to the dense audio-bearing subset.
        sound_sample_indices = [i for i, plan in enumerate(sequence_plans) if getattr(plan, "has_sound", False)]
        if getattr(rf_cfg, "independent_sound_schedule", False) and sound_sample_indices:
            ts_sound_full, sg_sound_full = self._get_train_noise_level_sound(
                batch_size=gen_data_clean.batch_size
            )  # [B,1] each
            timesteps_sound, sigmas_sound = build_dense_sound_schedule(
                sequence_plans,
                gen_data_clean.x0_tokens_sound,
                ts_sound_full,
                sg_sound_full,
            )  # [n_sound,1], [n_sound,1]
        else:
            timesteps_sound, sigmas_sound = (None, None)

        # Broadcast timesteps/sigmas across CP group to ensure consistency
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            src_rank = 0  # use cp rank 0 to broadcast timesteps/sigmas
            cp_group = self.parallel_dims.cp_mesh.get_group()
            global_src_rank = torch.distributed.get_global_rank(cp_group, src_rank)
            timesteps_vision = timesteps_vision.contiguous()
            sigmas_vision = sigmas_vision.contiguous()
            torch.distributed.broadcast(timesteps_vision, src=global_src_rank, group=cp_group)
            torch.distributed.broadcast(sigmas_vision, src=global_src_rank, group=cp_group)
            if sigmas_action is not None:
                timesteps_action = timesteps_action.contiguous()
                sigmas_action = sigmas_action.contiguous()
                torch.distributed.broadcast(timesteps_action, src=global_src_rank, group=cp_group)
                torch.distributed.broadcast(sigmas_action, src=global_src_rank, group=cp_group)
            if sigmas_sound is not None:
                timesteps_sound = timesteps_sound.contiguous()  # [n_sound,1]
                sigmas_sound = sigmas_sound.contiguous()  # [n_sound,1]
                torch.distributed.broadcast(timesteps_sound, src=global_src_rank, group=cp_group)
                torch.distributed.broadcast(sigmas_sound, src=global_src_rank, group=cp_group)

        if timesteps_sound is None:
            # Sound tensors are dense over audio-bearing samples, while the vision timestep/sigma schedule
            # is indexed by original batch position. Reindex here so mixed audio/no-audio batches use each
            # sound sample's own schedule for noising and loss weighting.
            timesteps_sound, sigmas_sound = build_dense_sound_schedule(
                sequence_plans,
                gen_data_clean.x0_tokens_sound,
                timesteps_vision,
                sigmas_vision,
            )  # [n_sound,T_vis] or None, [n_sound,T_vis] or None

        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            input_text_indexes,
            gen_data_clean,
            timesteps_vision.cpu(),
            skip_text_tokens=memory_info["skip_text"],
            initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
        )

        # Under independent_action_schedule, overwrite the vision-based action timestep the
        # packer injected with the action timestep, so the denoiser's action timestep embedding
        # matches the sigma used to noise action tokens.
        if timesteps_action is not None and packed_sequence.action is not None:
            action_has_noisy_tokens = any(nfi.numel() > 0 for nfi in packed_sequence.action.noisy_frame_indexes)
            if action_has_noisy_tokens:
                sample_ts = timesteps_action.squeeze(1).cpu()  # [n_action]
                packed_sequence.action.timesteps = torch.cat(
                    [
                        sample_ts[i : i + 1].expand(nfi.numel())
                        for i, nfi in enumerate(packed_sequence.action.noisy_frame_indexes)
                    ]
                ).to(dtype=torch.float32)  # [N_action_noisy]
                # These per-sample action sigmas can differ even when the vision timesteps the
                # packer saw were uniform, so the fast path in _embed_packed_timesteps no longer
                # holds for this sequence.
                packed_sequence.uses_single_timestep &= uses_single_timestep(sample_ts)
            else:
                timesteps_action, sigmas_action = (None, None)

        # Under independent_sound_schedule, overwrite the vision-based sound timestep the packer
        # injected with the sound timestep, so the denoiser's sound timestep embedding matches
        # the sigma used to noise sound tokens.
        if (
            getattr(rf_cfg, "independent_sound_schedule", False)
            and timesteps_sound is not None
            and packed_sequence.sound is not None
        ):
            sound_has_noisy_tokens = any(nfi.numel() > 0 for nfi in packed_sequence.sound.noisy_frame_indexes)
            if sound_has_noisy_tokens:
                sample_ts = timesteps_sound.squeeze(1).cpu()  # [n_sound]
                packed_sequence.sound.timesteps = torch.cat(
                    [
                        sample_ts[i : i + 1].expand(nfi.numel())
                        for i, nfi in enumerate(packed_sequence.sound.noisy_frame_indexes)
                    ]
                ).to(dtype=torch.float32)  # [N_sound_noisy]
                # As above: independent per-sample sound sigmas invalidate the single-timestep
                # fast path even when the packer saw uniform vision timesteps.
                packed_sequence.uses_single_timestep &= uses_single_timestep(sample_ts)
            else:
                timesteps_sound, sigmas_sound = (None, None)

        # For image editing (multi-item vision), expand per-sample timesteps/sigmas to
        # per-vision-item so downstream noise/loss indexing matches the flat x0_tokens_vision
        # list. No-op when num_vision_items_per_sample is None (standard T2I/T2V/WAM cases).
        # Conditioning items get sigma=0 via their condition_mask, so the actual timestep value
        # for them does not matter.
        # LiDAR shares the sample's vision schedule so both sensors denoise on one clock, then
        # expands to per-LiDAR-item rows. Sampled before the vision expansion below, which
        # rebinds timesteps_vision from per-sample to per-vision-item rows.
        if gen_data_clean.num_lidar_items_per_sample is not None:
            assert timesteps_vision.shape[1] == 1, (
                "LiDAR reuses the sample's vision timestep, which requires a single timestep per sample "
                "(diffusion forcing is not supported for joint camera + LiDAR batches)"
            )
            timesteps_lidar = _expand_per_sample_to_per_vision_item(
                timesteps_vision, gen_data_clean.num_lidar_items_per_sample
            )  # [B_lidar_items, 1]
            sigmas_lidar = _expand_per_sample_to_per_vision_item(
                sigmas_vision, gen_data_clean.num_lidar_items_per_sample
            )  # [B_lidar_items, 1]
        else:
            timesteps_lidar, sigmas_lidar = (None, None)

        timesteps_vision = _expand_per_sample_to_per_vision_item(
            timesteps_vision, gen_data_clean.num_vision_items_per_sample
        )  # [B_items, T_vis]
        sigmas_vision = _expand_per_sample_to_per_vision_item(
            sigmas_vision, gen_data_clean.num_vision_items_per_sample
        )  # [B_items, T_vis]

        memory_info = self.pre_noise_memory_hook(packed_sequence, gen_data_clean, memory_info)

        # Flow matching/diffusion forward process: noise the input signal with the sampled noise level
        gen_data_noised = self._add_noise_to_input(
            gen_data_clean,
            packed_sequence,
            sigmas_vision,
            sigmas_action=sigmas_action,
            sigmas_sound=sigmas_sound,
            sigmas_lidar=sigmas_lidar,
            iteration=iteration,
        )
        self._replace_clean_with_noised(packed_sequence, gen_data_noised)

        # Move packed sequence to CUDA
        packed_sequence.to_cuda()

        # Network forward pass
        memory = self.build_memory_state(packed_sequence, memory_info)  # pylint: disable=assignment-from-none
        out_net = self.denoise(
            data_batch_packed=packed_sequence,
            memory=memory,
        )

        loss, losses_dict = self._compute_losses(
            out_net=out_net,
            data_batch_packed=packed_sequence,
            gen_data_noised=gen_data_noised,
            timesteps=timesteps_vision,
            is_image_batch=gen_data_clean.is_image_batch,
            timesteps_action=timesteps_action,
            timesteps_sound=timesteps_sound,
            timesteps_lidar=timesteps_lidar,
        )

        _vision_tokens = len(packed_sequence.vision.sequence_indexes) if packed_sequence.vision else 0
        _action_tokens = len(packed_sequence.action.sequence_indexes) if packed_sequence.action else 0
        _sound_tokens = len(packed_sequence.sound.sequence_indexes) if packed_sequence.sound else 0

        output_batch = {
            "x0": gen_data_clean.x0_tokens_vision,
            "xt": gen_data_noised.xt_tokens_vision,
            "sigma": sigmas_vision,  # [B_items, T_vis]
            "model_pred": out_net["preds_vision"],
            "condition_mask_vision": packed_sequence.vision.condition_mask if packed_sequence.vision else None,
            "condition_mask_action": packed_sequence.action.condition_mask if packed_sequence.action else None,
            "und_token_length": packed_sequence.text_indexes.shape[0],
            "gen_token_length": packed_sequence.sequence_length - packed_sequence.text_indexes.shape[0],
            "vision_token_length": _vision_tokens,
            "action_token_length": _action_tokens,
            "sound_token_length": _sound_tokens,
            "is_image_batch": gen_data_clean.is_image_batch,
            # Processed metadata follows the same CP broadcast and dense action-row filtering as the loss tensors.
            "_action_family": getattr(gen_data_clean, "action_family", None),
            "batch_size": gen_data_clean.batch_size,
            "split_lens": packed_sequence.split_lens,
            "attn_modes": packed_sequence.attn_modes,
            "vae_pixel_shapes": vae_pixel_shapes,
            **losses_dict,
        }
        if sigmas_action is not None:
            output_batch["sigma_action"] = sigmas_action  # [n_action, 1] — dense over action-bearing samples
        if getattr(rf_cfg, "independent_sound_schedule", False) and sigmas_sound is not None:
            output_batch["sigma_sound"] = sigmas_sound  # [n_sound, 1] — dense over sound-bearing samples

        return output_batch, loss

    def _compute_flow_matching_loss(
        self,
        pred: list[torch.Tensor],
        target: list[torch.Tensor],
        condition_mask: list[torch.Tensor],
        timesteps: torch.Tensor,
        has_valid_tokens: bool,
        rectified_flow: RectifiedFlow,
        raw_action_dim: list[torch.Tensor] | None = None,
        action_valid_mask: list[torch.Tensor] | None = None,
        normalize_by_active: bool = False,
        exclude_fully_conditioned_items: bool = False,
        action_slot_stats: ActionSlotLossStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute flow matching loss for a modality.

        Args:
            pred: Predicted velocity field (list of tensors, one per sample).
            target: Target velocity field (list of tensors, one per sample).
                Under rectified flow the target is ``v = eps - x0``.
            condition_mask: Mask where 1 = clean/conditioning, 0 = noisy/generation (list of tensors).
            timesteps: Diffusion timesteps for time weighting. Shape [B,1] for
                base/teacher_forcing (all frames share one timestep) or [B,T_max]
                for diffusion_forcing (per-frame independent timesteps). Time weights
                are applied per-frame before averaging, so non-uniform weight functions
                are handled correctly.
            has_valid_tokens: Whether this modality has valid noisy tokens.
            rectified_flow: The rectified flow object for time weighting.
            normalize_by_active: When True, normalize per-instance loss by the count of
                active (noisy) elements rather than all elements. Preserves the
                ``sum / active_count`` semantics needed for distillation critics where
                conditioned frames contribute no signal and should not dilute the
                denominator.
            exclude_fully_conditioned_items: When True, omit items with no noisy
                tokens from the final scalar mean while retaining their per-instance
                zero entries.
            action_slot_stats: Optional collector for detached normalized per-sample
                losses over the canonical unified Action slots.

        Returns:
            tuple: A tuple containing two elements:
                - Flow matching loss (or dummy loss for gradient consistency).
                - Per-instance loss (or dummy loss for gradient consistency).
        """
        return compute_flow_matching_loss(
            pred=pred,
            target=target,
            condition_mask=condition_mask,
            timesteps=timesteps,
            has_valid_tokens=has_valid_tokens,
            rectified_flow=rectified_flow,
            tensor_kwargs_fp32=self.tensor_kwargs_fp32,
            raw_action_dim=raw_action_dim,
            action_valid_mask=action_valid_mask,
            normalize_by_active=normalize_by_active,
            exclude_fully_conditioned_items=exclude_fully_conditioned_items,
            action_slot_stats=action_slot_stats,
        )

    def _get_load_balancing_loss_meshes(self) -> tuple[DeviceMesh | None, DeviceMesh | None]:
        """Return the data- and context-parallel meshes used by load balancing loss."""
        parallel_dims = getattr(self, "parallel_dims", None)
        if parallel_dims is None:
            return None, None

        context_parallel_mesh = None
        if parallel_dims.cp_enabled and self.config.attention_io_layout == "sequence_sharded":
            context_parallel_mesh = parallel_dims.cp_mesh
        return parallel_dims.dp_mesh, context_parallel_mesh

    def _loss_averaging_group(self) -> tuple[torch.distributed.ProcessGroup | None, int]:
        """Return the (process_group, size) over which gradients are averaged.

        This must match the group FSDP/DDP averages gradients over so that scaling the
        per-rank loss by ``size`` exactly counteracts the framework's mean reduction.
        FSDP shards/replicates over ``parallel_dims.dp_mesh``; plain DDP averages over
        the whole world. Returns ``(None, 1)`` when running without distributed, in
        which case no reduction (and no scaling) is needed.

        Note: ``ParallelDims.dp_mesh`` is always 2-D ``(dp_replicate, dp_shard)`` with
        ``dp_replicate * dp_shard == world_size``, so FSDP/HSDP averages over the whole
        world. ``DeviceMesh.get_group()`` requires ``mesh_dim`` when ``ndim > 1``; use
        the default WORLD process group (``group=None``) instead of calling ``get_group()``.
        """
        if self.parallel_dims is not None and self.parallel_dims.dp_enabled:
            dp_mesh = self.parallel_dims.dp_mesh
            # WORLD group: dp mesh spans all ranks (see docstring). Avoid get_group() on 2-D mesh.
            return None, dp_mesh.size()
        if torch.distributed.is_initialized():
            # No explicit dp mesh (e.g. bare DDP): gradients average over the whole world.
            return None, torch.distributed.get_world_size()
        return None, 1

    def _sample_level_loss_scale(
        self,
        is_image_batch: bool,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Multiplier that converts rank-level to sample-level loss averaging.

        A packed batch is homogeneous — either all image or all video (video may carry
        accompanying action/audio) — with ``num_samples`` samples on this rank. Each
        iteration the per-modality sample counts are all-reduced over the data-parallel
        group to get the global totals ``N_image`` / ``N_video``, and this rank's batch is
        scaled by ``group_size * num_samples / N_modality``.

        Applied to the whole ``total_loss`` (video + action + audio for video batches),
        this scales every term by the same factor, so the video/action/audio balance set
        by ``loss_scale`` / ``action_loss_weight`` / ``sound_loss_scale`` is preserved,
        while the effective objective becomes a per-sample mean. The base per-modality
        weight already baked into ``total_loss`` (``loss_scale`` for video, ``image_loss_scale``
        for image) cancels out of this multiplier, so it does not appear here. The
        ``group_size`` factor cancels the framework's ``1/group_size`` gradient averaging;
        for balanced batches (``num_samples ≈ N/group_size``) the scale is ≈ 1, so logged
        loss magnitudes stay comparable to rank-level averaging.
        """
        dp_group, group_size = self._loss_averaging_group()

        # counts = [num_local_image_samples, num_local_video_samples]; exactly one is non-zero.
        counts = torch.zeros(2, dtype=torch.float64, device=device)
        counts[0 if is_image_batch else 1] = float(num_samples)
        if group_size > 1:
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=dp_group)

        global_num_samples = (counts[0] if is_image_batch else counts[1]).clamp(min=1.0)
        return (group_size * num_samples) / global_num_samples

    def _compute_losses(
        self,
        out_net: dict,
        data_batch_packed: PackedSequence,
        gen_data_noised: GenerationDataNoised,
        timesteps: torch.Tensor,
        is_image_batch: bool,
        timesteps_action: torch.Tensor | None = None,
        timesteps_sound: torch.Tensor | None = None,
        timesteps_lidar: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute flow matching loss and auxiliary load balancing losses.

        ``timesteps_action`` is an optional ``[n_action, 1]`` override for the action loss
        time-weighting — dense over action-bearing samples, matching ``data_batch_packed.action.*``.
        When None, action reuses ``timesteps`` (vision timesteps, legacy behavior). Set by
        ``training_step`` under ``independent_action_schedule=True``.

        ``timesteps_sound`` is an optional dense sound timestep tensor, matching
        ``data_batch_packed.sound.*``. When None, sound reuses ``timesteps``.
        """
        total_loss = 0.0
        losses_dict = {}
        # ts_action shape: vision fallback [B_items, T_vis] (legacy) or [n_action, 1] (independent).
        ts_action = timesteps if timesteps_action is None else timesteps_action  # [B_items,T_vis] or [n_action,1]
        # ts_sound shape: vision fallback [B_items,T_vis] or dense sound schedule [n_sound,...].
        ts_sound = timesteps if timesteps_sound is None else timesteps_sound  # [B_items,T_vis] or [n_sound,...]

        rf_cfg = self.config.rectified_flow_training_config
        normalize_by_active = rf_cfg.normalize_loss_by_active
        if self.config.vision_gen:
            # Only a batch that generates no camera stream, as the LiDAR-only recipe does, may
            # arrive with vision unpacked; for anything else that would silently sink the vision
            # loss into the dummy loss below.
            assert data_batch_packed.vision is not None or not gen_data_noised.vt_target_vision, (
                "Vision packed data required when the batch carries camera items"
            )
            rectified_flow_vision = self.rectified_flow_image if is_image_batch else self.rectified_flow_video

            # With no camera stream there are no noisy vision tokens, so this returns its dummy
            # loss over the network's zero-weighted vision predictions. That is what keeps
            # vae2llm and llm2vae in the backward graph on every rank, which FSDP requires.
            # Transfer teacher forcing flattens clean controls and the generated target into
            # one vision-item list. Keep the per-item vector aligned for logging, but do not
            # let zero-loss controls dilute the scalar training objective.
            fm_loss_vision, fm_loss_vision_per_instance = compute_flow_matching_loss(
                pred=out_net["preds_vision"],
                target=gen_data_noised.vt_target_vision,
                condition_mask=_vision_condition_masks(data_batch_packed),
                timesteps=timesteps,
                has_valid_tokens=has_noisy_tokens(data_batch_packed.vision),
                rectified_flow=rectified_flow_vision,
                tensor_kwargs_fp32=self.tensor_kwargs_fp32,
                normalize_by_active=normalize_by_active,
                exclude_fully_conditioned_items=getattr(self.config, "causal_training_strategy", None)
                == "teacher_forcing",
            )
            loss_scale = (
                rf_cfg.image_loss_scale if is_image_batch and rf_cfg.image_loss_scale is not None else rf_cfg.loss_scale
            )
            total_loss += fm_loss_vision * loss_scale  # []
            losses_dict["flow_matching_loss_vision"] = fm_loss_vision  # []
            losses_dict["flow_matching_loss_vision_per_instance"] = fm_loss_vision_per_instance  # [N]
        else:
            losses_dict["flow_matching_loss_vision"] = torch.tensor(0.0, **self.tensor_kwargs_fp32)  # []

        # Same condition the network builds its LiDAR projections under, so the two cannot
        # disagree about whether this run has a LiDAR stream to supervise.
        if self.config.vision_gen and self.config.lidar_state_ch is not None:
            if data_batch_packed.lidar is not None:
                assert isinstance(data_batch_packed.lidar.condition_mask, list), (
                    "LiDAR condition mask must be a list of tensors for loss computation"
                )
                assert gen_data_noised.vt_target_lidar is not None, "LiDAR targets required when the batch has LiDAR"
                fm_loss_lidar, _ = compute_flow_matching_loss(
                    pred=out_net["preds_lidar"],
                    target=gen_data_noised.vt_target_lidar,
                    condition_mask=data_batch_packed.lidar.condition_mask,
                    timesteps=timesteps_lidar if timesteps_lidar is not None else timesteps,
                    has_valid_tokens=has_noisy_tokens(data_batch_packed.lidar),
                    rectified_flow=self.rectified_flow_video,
                    tensor_kwargs_fp32=self.tensor_kwargs_fp32,
                    normalize_by_active=normalize_by_active,
                )
                lidar_loss_scale = rf_cfg.lidar_loss_scale if rf_cfg.lidar_loss_scale is not None else rf_cfg.loss_scale
                total_loss += fm_loss_lidar * lidar_loss_scale  # []
                losses_dict["flow_matching_loss_lidar"] = fm_loss_lidar  # []
            else:
                # No LiDAR data in this batch. Connect the network's dummy preds_lidar to the
                # loss so lidar2llm / llm2lidar stay in the backward graph, or FSDP's gradient
                # reduce hangs against the ranks that did draw LiDAR.
                #
                # _decode_grid_stream returns [probe] for an absent stream -- one tensor that
                # has already been through both projections -- which is what makes the sum
                # below a tensor rather than the int that `sum` returns for an empty list.
                # That int would multiply to a plain float, dropping the projections from the
                # graph and producing exactly the hang above, so assert the guarantee here
                # rather than rely on a reader knowing the other file.
                preds_lidar = out_net["preds_lidar"]
                assert preds_lidar, (
                    "preds_lidar must carry the network's zero-weighted probe so lidar2llm / "
                    "llm2lidar stay in the backward graph on a batch with no LiDAR"
                )
                dummy_loss = 0.0 * sum(p.sum() for p in preds_lidar)  # []
                total_loss += dummy_loss  # []
                losses_dict["flow_matching_loss_lidar"] = dummy_loss  # []

        if self.config.action_gen:
            if data_batch_packed.action is not None:
                assert isinstance(data_batch_packed.action.condition_mask, list), (
                    "Action condition mask must be a list of tensors for loss computation"
                )
                assert gen_data_noised.vt_target_action is not None, "Action targets required when action_gen is True"
                action_has_valid_tokens = has_noisy_tokens(data_batch_packed.action)
                num_action_samples = len(out_net["preds_action"])
                slot_stat_zeros = out_net["preds_action"][0].new_zeros(
                    (num_action_samples, len(UNIFIED_ACTION_SLOT_GROUPS)), dtype=torch.float32
                )
                action_slot_stats = ActionSlotLossStats(
                    sample_loss=slot_stat_zeros,
                    sample_count=torch.zeros_like(slot_stat_zeros),
                )
                fm_loss_action, _ = compute_flow_matching_loss(
                    pred=out_net["preds_action"],
                    target=gen_data_noised.vt_target_action,
                    condition_mask=data_batch_packed.action.condition_mask,
                    timesteps=ts_action,
                    has_valid_tokens=action_has_valid_tokens,
                    rectified_flow=self.rectified_flow_action,
                    tensor_kwargs_fp32=self.tensor_kwargs_fp32,
                    raw_action_dim=data_batch_packed.action.raw_action_dim,
                    action_valid_mask=data_batch_packed.action.action_valid_mask,
                    normalize_by_active=normalize_by_active,
                    action_slot_stats=action_slot_stats,
                )

                # Yihuai: In case the video loss is too large (1.5) and covers the action loss (0.05), we scale up the action loss to match the video loss to improve action precision.
                total_loss += fm_loss_action * rf_cfg.action_loss_weight
                losses_dict["flow_matching_loss_action"] = fm_loss_action
                losses_dict[ACTION_SLOT_SAMPLE_LOSS_KEY] = action_slot_stats.sample_loss
                losses_dict[ACTION_SLOT_SAMPLE_COUNT_KEY] = action_slot_stats.sample_count
            else:
                # No action data in this batch. Connect the network's dummy preds_action
                # to the loss so action-specific params
                # (llm2action, action2llm, and action_modality_embed when enabled)
                # stay in the backward graph. Without this, FSDP reduce-scatter /
                # DDP all-reduce will hang when other ranks do have action data.
                dummy_loss = 0.0 * sum(p.sum() for p in out_net["preds_action"])
                total_loss += dummy_loss
                losses_dict["flow_matching_loss_action"] = dummy_loss
        else:
            losses_dict["flow_matching_loss_action"] = torch.tensor(0.0, **self.tensor_kwargs_fp32)

        if self.config.sound_gen:
            if data_batch_packed.sound is not None:
                assert isinstance(data_batch_packed.sound.condition_mask, list), (
                    "Sound condition mask must be a list of tensors for loss computation"
                )
                assert gen_data_noised.vt_target_sound is not None, "Sound targets required when sound_gen is True"
                # Sound preds/targets are (C, T); condition_mask is (T, 1) — transpose to (1, T) for broadcasting
                fm_loss_sound, _ = compute_flow_matching_loss(
                    pred=out_net["preds_sound"],
                    target=gen_data_noised.vt_target_sound,
                    condition_mask=[m.T for m in data_batch_packed.sound.condition_mask],
                    timesteps=ts_sound,
                    has_valid_tokens=has_noisy_tokens(data_batch_packed.sound),
                    rectified_flow=self.rectified_flow_sound,
                    tensor_kwargs_fp32=self.tensor_kwargs_fp32,
                    normalize_by_active=normalize_by_active,
                )
                loss_scale = rf_cfg.sound_loss_scale if rf_cfg.sound_loss_scale is not None else rf_cfg.loss_scale
                total_loss += fm_loss_sound * loss_scale
                losses_dict["flow_matching_loss_sound"] = fm_loss_sound
            else:
                # No sound data in this batch. Connect the network's dummy preds_sound
                # to the loss so enabled sound-specific parameters stay in the backward
                # graph. Without this, FSDP gradient reduce hangs when other ranks do
                # have sound data.
                dummy_loss = 0.0 * sum(p.sum() for p in out_net["preds_sound"])
                total_loss += dummy_loss
                losses_dict["flow_matching_loss_sound"] = dummy_loss
        else:
            losses_dict["flow_matching_loss_sound"] = torch.tensor(0.0, **self.tensor_kwargs_fp32)

        # Sample-level (vs rank-level) loss averaging. Scale the whole batch loss
        # (vision + accompanying action/audio for video batches) by one factor so every
        # image / video sample contributes equally across the data-parallel group. Applied
        # here — after all flow-matching terms, before load balancing — so the video/action/
        # audio balance is preserved and the auxiliary load-balancing losses stay unscaled.
        if rf_cfg.sample_level_loss_averaging and self.config.vision_gen:
            num_samples = len(out_net["preds_vision"])
            sample_level_scale = self._sample_level_loss_scale(
                is_image_batch=is_image_batch,
                num_samples=num_samples,
                device=self.tensor_kwargs_fp32["device"],
            )
            total_loss = total_loss * sample_level_scale.to(dtype=total_loss.dtype)

        # 2. Load balancing auxiliary losses
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
                total_loss += load_balancing_loss
                losses_dict[f"aux_loss_{load_balancing_type}"] = load_balancing_loss

        return total_loss, losses_dict

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        if not isinstance(self.net, WeightTrainingStat):
            return

        is_image = self.is_image_batch(data_batch)
        if self._has_vision_stream(data_batch):
            input_key = self.input_image_key if is_image else self.input_video_key
            value = data_batch[input_key]
            # For image editing data_batch[input_key] is a list-of-lists, not a tensor.
            sample_count = len(value) if isinstance(value, list) else value.shape[0]
        else:
            # The LiDAR-only stream still contributes samples on the video clock. Prefer
            # per-sample item counts after encoding has flattened the LiDAR item list.
            lidar_counts = data_batch.get("num_lidar_items_per_sample")
            sample_count = len(lidar_counts) if lidar_counts is not None else len(data_batch["lidar"])

        if is_image:
            self.net.accum_image_sample_counter += sample_count
        else:
            self.net.accum_video_sample_counter += sample_count

    def _update_train_stats_from_processed_batch(self, gen_data_clean: GenerationDataClean) -> None:
        """Update rank-zero sample counters from the CP-shared processed payload."""
        if not isinstance(self.net, WeightTrainingStat):
            return
        if gen_data_clean.is_image_batch:
            self.net.accum_image_sample_counter += gen_data_clean.batch_size
        else:
            self.net.accum_video_sample_counter += gen_data_clean.batch_size

    def _load_and_tokenize_text_data(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> list[list[int]]:
        """
        Load and tokenize the text data from the data batch.

        Args:
            data_batch (dict[str, torch.Tensor]): The data batch.
            iteration (int): The current iteration number.

        Returns:
            list[torch.Tensor]: The input text tokens.
        """
        input_text_tokens = data_batch["text_token_ids"]
        if isinstance(input_text_tokens, list):
            # Convert text tokens to list of lists of ints
            input_text_tokens = [tokens.tolist() for x in input_text_tokens for tokens in x]
        else:
            input_text_tokens = [tokens.squeeze(0).tolist() for tokens in input_text_tokens]

        return input_text_tokens

    def _get_train_noise_level_vision(
        self,
        batch_size: int,
        is_image_batch: bool,
        num_vision_latent_frames: list[int],
        resolutions: list[str] | str | None = None,
        num_tokens: list[int] | None = None,
        iteration: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the rectified flow interpolation coefficient (timesteps) and obtain the corresponding
        normalized timestep.

        Args:
            batch_size: Batch size for sampling timesteps.
            is_image_batch: Whether this is an image batch (vs video).
            num_vision_latent_frames: Per-sample vision latent frame counts [T_0, ..., T_{B-1}].
                         For causal_training_strategy="diffusion_forcing", resamples B*T_max independent
                         times and returns tensors of shape [B,T_max]. For base/TF strategies, ignored —
                         returns shape [B,1] (all frames share the same sigma).
            resolutions: Resolution string(s) (e.g., "256", "512") for dict-based shift lookup.
                         Can be a single string (applied to all samples) or a list of strings (one per sample).
                         If None, defaults to self.config.resolution (can be used for other modalities).
            num_tokens: Number of tokens for each sample (before 2x2 merge). Needed for dynamic shift.

        Returns:
            (timesteps, sigmas): Both [B,1] for TF/base, or [B,T_max] for diffusion_forcing.
        """

        rectified_flow = self.rectified_flow_image if is_image_batch else self.rectified_flow_video

        assert not self.config.rectified_flow_training_config.use_discrete_rf, (
            "Discrete RF is not supported for Cosmos3"
        )
        # Continuous RF implementation
        max_timestep = rectified_flow.noise_scheduler.config.num_train_timesteps

        # Get shift value(s) - images can override the shared/video shift.
        rf_config = self.config.rectified_flow_training_config
        shift_image = getattr(rf_config, "shift_image", None)
        shift_config = shift_image if is_image_batch and shift_image is not None else rf_config.shift
        if isinstance(shift_config, int):
            # Int-based shift: use directly for all samples
            shifts = torch.full((batch_size,), shift_config, dtype=torch.float32)
        else:
            # Convert to plain dict to avoid traceback-based memory leaks when GC is disabled
            # (OmegaConf's `in` operator uses exception control flow internally).
            shift_dict = dict(shift_config)
            if not is_image_batch and "dynamic_shift_base_num_tokens_video" in shift_dict:
                # Dynamic shift based on token count
                if num_tokens is None or len(num_tokens) != batch_size:
                    raise ValueError(
                        "dynamic_shift_base_num_tokens_video requires one vision token count per sample; "
                        f"got {0 if num_tokens is None else len(num_tokens)} counts for batch size {batch_size}. "
                        "A vision-less LiDAR batch cannot use a vision-token-based dynamic shift."
                    )
                base_num_tokens = shift_dict["dynamic_shift_base_num_tokens_video"]
                shifts = torch.sqrt(torch.tensor(num_tokens, dtype=torch.float32) / base_num_tokens)
            elif is_image_batch and "dynamic_shift_base_num_tokens_image" in shift_dict:
                if num_tokens is None or len(num_tokens) != batch_size:
                    raise ValueError(
                        "dynamic_shift_base_num_tokens_image requires one vision token count per sample; "
                        f"got {0 if num_tokens is None else len(num_tokens)} counts for batch size {batch_size}."
                    )
                base_num_tokens = shift_dict["dynamic_shift_base_num_tokens_image"]
                shifts = torch.sqrt(torch.tensor(num_tokens, dtype=torch.float32) / base_num_tokens)
            else:
                # Dict-based shift: lookup per sample
                if resolutions is None:
                    raise ValueError("Resolutions must be provided when shift is a dict")

                # Normalize to list format
                if isinstance(resolutions, str):
                    resolutions = [resolutions] * batch_size

                assert len(resolutions) == batch_size, (
                    f"Number of resolutions ({len(resolutions)}) must match batch_size ({batch_size})"
                )

                # Lookup shift per sample
                shifts_list = []
                for resolution in resolutions:
                    if resolution not in shift_dict:
                        raise ValueError(
                            f"Resolution '{resolution}' not found in shift dict. Available resolutions: {list(shift_dict.keys())}"
                        )
                    shifts_list.append(shift_dict[resolution])
                shifts = torch.tensor(shifts_list, dtype=torch.float32)

        # Sample noise times: B×T_max for DF (one per video latent frame), B×1 for base/TF
        if self.config.causal_training_strategy == "diffusion_forcing":
            # T_max = max(num_vision_latent_frames) across the batch; trailing entries for shorter
            # sequences are unused (sliced away in _add_noise_to_input).
            T_max = max(num_vision_latent_frames)
            sigmas = (
                rectified_flow.sample_train_time(
                    batch_size * T_max, iteration=iteration, shifts=shifts.repeat_interleave(T_max)
                )
                .to(**self.tensor_kwargs_fp32)
                .reshape(batch_size, T_max)
            )  # [B,T_max]
        else:
            sigmas = (
                rectified_flow.sample_train_time(batch_size, iteration=iteration, shifts=shifts)
                .to(**self.tensor_kwargs_fp32)
                .unsqueeze(1)
            )  # [B,1]

        timesteps = sigmas * max_timestep  # [B,1] or [B,T_max]
        return timesteps, sigmas

    def _get_train_noise_level_action(
        self, batch_size: int, iteration: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(timesteps, sigmas)`` of shape ``[batch_size, 1]`` from ``rectified_flow_action``.

        This helper is locally-scoped: it just draws ``batch_size`` independent σ values and
        applies action-specific shift config. The caller decides what ``batch_size`` means
        semantically — ``training_step`` passes the full batch size and then reindexes to
        the dense action-bearing subset with ``action_sample_indices``.

        ``shift_action`` must be an int (or ``None`` to inherit ``shift``). Dict-keyed
        per-resolution shifts are vision-only — multi-resolution action training would need
        per-sample lookup, which this helper does not implement; if the global ``shift`` is a
        dict and ``shift_action`` is None, this raises so the user sets shift_action explicitly.
        σ is a shared scalar per input slot (no per-frame σ for action).
        """
        rf_cfg = self.config.rectified_flow_training_config
        rf = self.rectified_flow_action
        max_timestep = rf.noise_scheduler.config.num_train_timesteps  # int

        # Resolve shift. shift_action, when provided, must be an int.
        if rf_cfg.shift_action is not None:
            if not isinstance(rf_cfg.shift_action, int):
                raise ValueError(
                    f"shift_action must be an int; got {type(rf_cfg.shift_action).__name__}. "
                    "Dict-keyed per-resolution shifts are vision-only."
                )
            shift_val = rf_cfg.shift_action  # int
        elif isinstance(rf_cfg.shift, int):
            shift_val = rf_cfg.shift  # inherit the global int shift
        else:
            raise ValueError(
                "shift_action=None requires the global `shift` to be an int. When `shift` is a "
                f"dict (multi-resolution vision training), set shift_action explicitly as an int. "
                f"Got shift={rf_cfg.shift!r}."
            )

        shifts = torch.full((batch_size,), shift_val, dtype=torch.float32)
        sigmas = (
            rf.sample_train_time(batch_size, iteration=iteration, shifts=shifts)
            .to(**self.tensor_kwargs_fp32)
            .unsqueeze(1)
        )  # [B,1]
        timesteps = sigmas * max_timestep  # [B,1]
        return timesteps, sigmas

    def _get_train_noise_level_sound(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(timesteps, sigmas)`` of shape ``[batch_size, 1]`` from ``rectified_flow_sound``.

        Sound uses a shared scalar sigma per audio-bearing sample, then training_step
        reindexes the full-batch samples to the dense sound tensor list.
        """
        rf_cfg = self.config.rectified_flow_training_config
        rf = self.rectified_flow_sound
        max_timestep = rf.noise_scheduler.config.num_train_timesteps  # int

        # Resolve shift. shift_sound, when provided, must be an int.
        if rf_cfg.shift_sound is not None:
            if not isinstance(rf_cfg.shift_sound, int):
                raise ValueError(
                    f"shift_sound must be an int; got {type(rf_cfg.shift_sound).__name__}. "
                    "Dict-keyed per-resolution shifts are vision-only."
                )
            shift_val = rf_cfg.shift_sound  # int
        elif isinstance(rf_cfg.shift, int):
            shift_val = rf_cfg.shift  # inherit the global int shift
        else:
            raise ValueError(
                "shift_sound=None requires the global `shift` to be an int. When `shift` is a "
                f"dict (multi-resolution vision training), set shift_sound explicitly as an int. "
                f"Got shift={rf_cfg.shift!r}."
            )

        shifts = torch.full((batch_size,), shift_val, dtype=torch.float32)
        sigmas = rf.sample_train_time(batch_size, shifts=shifts).to(**self.tensor_kwargs_fp32).unsqueeze(1)  # [B,1]
        timesteps = sigmas * max_timestep  # [B,1]
        return timesteps, sigmas

    def _add_noise_to_input(
        self,
        gen_data_clean: GenerationDataClean,
        packed_sequence: PackedSequence,
        sigmas: torch.Tensor,
        sigmas_action: torch.Tensor | None = None,
        sigmas_sound: torch.Tensor | None = None,
        sigmas_lidar: torch.Tensor | None = None,
        iteration: int | None = None,
    ) -> GenerationDataNoised:
        """
        Diffusion / Flow matching forward process: apply noise of given noise level (sigmas) to input data.

        Args:
            gen_data_clean (GenerationDataClean): The input dataclass containing the clean data *latents* (tokens).
            packed_sequence (PackedSequence): Packed sequence with condition masks attached to modalities.
            sigmas (torch.Tensor): The noise levels. Shape [B,1] for base/teacher_forcing (all video
                latent frames share the same sigma) or [B,T_max] for diffusion_forcing (per-latent-frame
                independent sigma). T_max is the number of video latent frames (temporally compressed
                tokens), not RGB frames. In all modes, sigmas are multiplied by (1 - condition_mask)
                so conditioning latent frames get sigma_eff=0 and only non-conditioned frames contribute
                to the loss.
            sigmas_action: Optional ``[n_action, 1]`` override for action noising — dense over
                action-bearing samples, matching ``packed_sequence.action.*``. When None, action
                reuses ``sigmas`` (vision σ, legacy behavior). Set by ``training_step`` when
                ``independent_action_schedule=True``.
            sigmas_sound: Optional dense sound sigma tensor matching ``packed_sequence.sound.*``.
                When None, sound reuses ``sigmas``.
            sigmas_lidar: ``[n_lidar_items, 1]`` sigma per LiDAR item, required when the batch
                carries LiDAR. Rows are the owning sample's vision sigma so both sensors are
                noised at the same point on the flow.

        Returns:
            GenerationDataNoised: A dataclass containing the noise, noisy data (xt), and velocity field (vt).
        """
        # Action sigma defaults to the shared vision sigma (legacy behavior).
        # Legacy (sigmas_action=None): vision σ of shape [B_items, T_vis].
        # Independent (sigmas_action provided): dense action σ of shape [n_action, 1].
        sigmas_for_action = sigmas if sigmas_action is None else sigmas_action  # [B_items,T_vis] or [n_action,1]
        # Sound uses a dense view of the per-sample vision schedule so mixed audio/no-audio
        # batches do not index full-batch sigmas with dense sound positions.
        sigmas_for_sound = sigmas if sigmas_sound is None else sigmas_sound  # [B_items,T_vis] or [n_sound,...]

        # Seeded noise generator (deterministic mode only): keyed on (iteration, rank).
        # Offset +32768 keeps this seed distinct from the sigma seed in sample_train_time.
        noise_gen: torch.Generator | None = None
        if iteration is not None and torch.are_deterministic_algorithms_enabled():
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            noise_gen = torch.Generator(device=self.tensor_kwargs_fp32["device"])
            noise_gen.manual_seed(iteration * 65536 + rank + 32768)

        # Vision
        x0_vision = gen_data_clean.x0_tokens_vision  # list of [C,T,H,W]
        assert x0_vision is not None, "Vision tokens are required for VFM noising."
        epsilon_vision = [
            torch.randn(x0_vision_i.size(), generator=noise_gen, **self.tensor_kwargs_fp32) for x0_vision_i in x0_vision
        ]  # list of [C,T,H,W]
        # Under CP, every rank holds the same x0 (the post-tokenizer payload is broadcast
        # round-robin above) but each samples its own ε from a rank-divergent RNG. Broadcasting
        # from CP rank 0 makes ε (and therefore xt) identical across the CP group so the
        # seq-sharded packed sequence is consistent. No-op when CP is disabled.
        context_parallel_broadcast_tensor_list(epsilon_vision, self.parallel_dims)

        # Derive noisy mask (1 for noised, 0 for clean) for sigmas computation
        condition_mask_vision = _vision_condition_masks(packed_sequence)

        # Compute sigmas per vision item (supports variable shapes).
        # For image editing, x0_tokens_vision is a flat list with multiple items per sample
        # and sigmas has already been expanded to match (see _expand_per_sample_to_per_vision_item).
        # Conditioning latent frames are zeroed via (1 - condition_mask) in all modes (base/TF/DF).
        # view(-1,1,1)[:T_latent]: for base/TF sigmas[i] is (1,), view gives (1,1,1) and the slice is a no-op;
        # for DF sigmas[i] is (T_max,) — one sigma per video latent frame — view gives (T_max,1,1)
        # and [:T_latent] slices to (T_latent,1,1) matching the per-item latent frame count.
        num_vision_items = len(condition_mask_vision)
        noisy_mask_vision = [1.0 - cond_mask for cond_mask in condition_mask_vision]
        sigmas_vision = [
            sigmas[i].view(-1, 1, 1)[: x0_vision[i].shape[2]] * noisy_mask_vision[i] for i in range(num_vision_items)
        ]
        rectified_flow_vision = (
            self.rectified_flow_image if gen_data_clean.is_image_batch else self.rectified_flow_video
        )
        xt_vision, vt_vision = rectified_flow_vision.get_interpolation(
            epsilon_vision, x0_vision, sigmas_vision
        )  # list of [C,T,H,W], list of [C,T,H,W]

        xt_vision = [
            xt_vision_i.to(**self.tensor_kwargs) for xt_vision_i in xt_vision
        ]  # list of [C,T,H,W]; to make tensor compatible with the precision of the model

        # LiDAR: same rectified flow as video, on its own grid latents.
        x0_lidar = gen_data_clean.x0_tokens_lidar  # list of [C,T,H,W]
        if x0_lidar is not None and len(x0_lidar) > 0:
            assert packed_sequence.lidar is not None, "Packed LiDAR data required when LiDAR tokens exist"
            assert isinstance(packed_sequence.lidar.condition_mask, list), (
                "LiDAR condition mask must be a list of tensors for noise scheduling"
            )
            assert sigmas_lidar is not None, "sigmas_lidar required when LiDAR tokens exist"
            epsilon_lidar = [
                torch.randn(x0_i.size(), generator=noise_gen, **self.tensor_kwargs_fp32) for x0_i in x0_lidar
            ]  # list of [C,T,H,W]
            context_parallel_broadcast_tensor_list(epsilon_lidar, self.parallel_dims)
            # sigmas_lidar[i] is (1,) → view (1,1,1), broadcast against condition_mask [T,1,1].
            sigmas_lidar_list = [
                sigmas_lidar[i].view(-1, 1, 1) * (1.0 - packed_sequence.lidar.condition_mask[i])
                for i in range(len(x0_lidar))
            ]  # list of [T,1,1]
            xt_lidar, vt_lidar = self.rectified_flow_video.get_interpolation(epsilon_lidar, x0_lidar, sigmas_lidar_list)
            xt_lidar = [xt_i.to(**self.tensor_kwargs) for xt_i in xt_lidar]  # list of [C,T,H,W]
        else:
            epsilon_lidar = None
            sigmas_lidar_list = None
            xt_lidar = None
            vt_lidar = None

        # Action (x0_tokens_action is already a dense list with no None entries).
        # Gate on action_gen: the dataset may emit action tensors for models that
        # don't consume them (e.g. camera dataset on a vision-only config), in
        # which case packed_sequence.action is None and we must skip this block.
        x0_action = gen_data_clean.x0_tokens_action  # list of [T,action_dim]
        if self.config.action_gen and x0_action is not None and len(x0_action) > 0:
            assert packed_sequence.action is not None, "Packed action data required when action tokens exist"
            assert packed_sequence.action.condition_mask is not None, (
                "Action condition mask required when action tokens exist"
            )
            action_batch_size = len(packed_sequence.action.condition_mask)
            all_actions_are_conditioning = all(
                torch.all(condition_mask == 1).item() for condition_mask in packed_sequence.action.condition_mask
            )
            if all_actions_are_conditioning:
                epsilon_action = [
                    torch.zeros(x0_action_i.size(), **self.tensor_kwargs_fp32) for x0_action_i in x0_action
                ]  # list of [T,action_dim]
                sigmas_action = [
                    torch.zeros_like(condition_mask, dtype=torch.float32, device=condition_mask.device)
                    for condition_mask in packed_sequence.action.condition_mask
                ]  # list of [T,1]
                xt_action = [
                    x0_action_i.to(**self.tensor_kwargs) for x0_action_i in x0_action
                ]  # list of [T,action_dim]
                vt_action = [
                    torch.zeros(x0_action_i.size(), **self.tensor_kwargs_fp32) for x0_action_i in x0_action
                ]  # list of [T,action_dim]
            else:
                epsilon_action = [
                    torch.randn(x0_action_i.size(), generator=noise_gen, **self.tensor_kwargs_fp32)
                    for x0_action_i in x0_action
                ]  # list of [T,action_dim]
                context_parallel_broadcast_tensor_list(epsilon_action, self.parallel_dims)
                # Conditioning action timesteps are zeroed via (1 - condition_mask) in all modes (base/TF/DF).
                # Action timesteps are aligned 1-to-1 with video latent frames, not RGB frames.
                # view(-1,1)[:T_i]: for base/TF sigmas[i] is (1,) → (1,1), slice is a no-op;
                # for DF sigmas[i] is (T_max,) → (T_max,1) → (T_i,1) per-action-timestep sigmas.
                # condition_mask[i] shape [T_i,1]; result broadcasts with x0 shape [T_i,C].
                sigmas_action = [
                    sigmas_for_action[i].view(-1, 1)[: x0_action[i].shape[0]]
                    * (1.0 - packed_sequence.action.condition_mask[i])
                    for i in range(action_batch_size)
                ]  # list of [T_i,1]
                assert sigmas_action is not None
                xt_action, vt_action = self.rectified_flow_action.get_interpolation(
                    epsilon_action, x0_action, sigmas_action
                )  # list of [T,action_dim], list of [T,action_dim]
                xt_action = [
                    xt_action_i.to(**self.tensor_kwargs) for xt_action_i in xt_action
                ]  # list of [T,action_dim]; to make tensor compatible with the precision of the model
            for i in range(len(xt_action)):
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    xt_action[i][:, gen_data_clean.raw_action_dim[i] :] = 0
                if gen_data_clean.action_valid_mask is not None and gen_data_clean.action_valid_mask[i] is not None:
                    valid_mask = gen_data_clean.action_valid_mask[i].to(device=xt_action[i].device)
                    xt_action[i][:, : valid_mask.numel()].mul_(valid_mask.to(dtype=xt_action[i].dtype))

        else:
            epsilon_action = None
            sigmas_action = None
            xt_action = None
            vt_action = None

        # Sound (x0_tokens_sound is a list of [C, T] tensors, or None)
        x0_sound = gen_data_clean.x0_tokens_sound  # list of [sound_channels,T_sound]
        if x0_sound is not None and len(x0_sound) > 0:
            assert packed_sequence.sound is not None, "Packed sound data required when sound tokens exist"
            assert packed_sequence.sound.condition_mask is not None, (
                "Sound condition mask required when sound tokens exist"
            )
            sound_batch_size = len(packed_sequence.sound.condition_mask)
            epsilon_sound = [
                torch.randn(x0_i.size(), generator=noise_gen, **self.tensor_kwargs_fp32) for x0_i in x0_sound
            ]  # list of [C,T_sound]
            context_parallel_broadcast_tensor_list(epsilon_sound, self.parallel_dims)
            # Conditioning frames are zeroed via (1 - condition_mask) in all modes (base/TF/DF).
            # view(-1,1)[:T_sound].T: for base/TF sigmas[i] is (1,) → (1,1) → no-op → (1,1);
            # for DF sigmas[i] is (T_max,) → (T_max,1) → (T_sound,1) → (1,T_sound).
            # condition_mask[i] shape [T_sound,1]; .T gives [1,T_sound]; result broadcasts with x0 [C,T_sound].
            sigmas_sound = [
                sigmas_for_sound[i].view(-1, 1)[: x0_sound[i].shape[1]].T
                * (1.0 - packed_sequence.sound.condition_mask[i].T)
                for i in range(sound_batch_size)
            ]
            assert sigmas_sound is not None
            xt_sound, vt_sound = self.rectified_flow_sound.get_interpolation(epsilon_sound, x0_sound, sigmas_sound)
            xt_sound = [xt_i.to(**self.tensor_kwargs) for xt_i in xt_sound]
        else:
            epsilon_sound = None
            sigmas_sound = None
            xt_sound = None
            vt_sound = None

        # create the GenerationDataNoised object
        gen_data_noised = GenerationDataNoised(
            batch_size=gen_data_clean.batch_size,
            # vision
            epsilon_vision=epsilon_vision,
            xt_tokens_vision=xt_vision,
            vt_target_vision=vt_vision,
            sigmas_vision=sigmas_vision,
            # lidar
            epsilon_lidar=epsilon_lidar,
            xt_tokens_lidar=xt_lidar,
            vt_target_lidar=vt_lidar,
            sigmas_lidar=sigmas_lidar_list,
            # action
            epsilon_action=epsilon_action,
            xt_tokens_action=xt_action,
            vt_target_action=vt_action,
            sigmas_action=sigmas_action,
            raw_action_dim=gen_data_clean.raw_action_dim,
            action_valid_mask=gen_data_clean.action_valid_mask,
            # sound
            epsilon_sound=epsilon_sound,
            xt_tokens_sound=xt_sound,
            vt_target_sound=vt_sound,
            sigmas_sound=sigmas_sound,
        )

        return gen_data_noised

    def _replace_clean_with_noised(
        self,
        packed_sequence: PackedSequence,
        gen_data_noised: GenerationDataNoised,
    ) -> None:
        """Replace packed clean tokens with noised tokens."""
        if packed_sequence.vision is not None:
            packed_sequence.vision.tokens = gen_data_noised.xt_tokens_vision
        if packed_sequence.lidar is not None and gen_data_noised.xt_tokens_lidar is not None:
            packed_sequence.lidar.tokens = gen_data_noised.xt_tokens_lidar
        if packed_sequence.action is not None and gen_data_noised.xt_tokens_action is not None:
            action_all_conditioning = all(
                torch.all(condition_mask == 1).item() for condition_mask in packed_sequence.action.condition_mask
            )
            if not action_all_conditioning:
                packed_sequence.action.tokens = gen_data_noised.xt_tokens_action
        if packed_sequence.sound is not None and gen_data_noised.xt_tokens_sound is not None:
            packed_sequence.sound.tokens = gen_data_noised.xt_tokens_sound

    # ------------------------ Inference Utils ------------------------
    def _get_inference_text_tokens(
        self, data_batch: dict, has_negative_prompt: bool
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Tokenize conditional and unconditional captions for inference.

        Delegates the per-caption chat-template tokenization to
        :meth:`_tokenize_captions` (the same helper backing the public
        :meth:`tokenize_text`) so there is a single source of truth for
        how raw captions become token ids.
        """
        use_system_prompt = self.vlm_config.use_system_prompt
        system_prompt: str | None = data_batch.get("system_prompt")

        cond_tokens = self._tokenize_captions(
            data_batch[self.input_caption_key],
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
            is_video=False,
        )

        if has_negative_prompt:
            neg_key = "neg_" + self.input_caption_key
            if neg_key not in data_batch:
                raise ValueError(f"Negative prompt ({neg_key}) not found")
            uncond_captions = data_batch[neg_key]
        else:
            uncond_captions = [""] * len(cond_tokens)

        uncond_tokens = self._tokenize_captions(
            uncond_captions,
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
            is_video=False,
        )
        return cond_tokens, uncond_tokens

    def _prepare_inference_data(
        self,
        data_batch: dict,
        seed: list[int],
        has_negative_prompt: bool = False,
    ) -> tuple[
        list[SequencePlan],
        GenerationDataClean,
        list[list[int]],
        list[list[int]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        bool,
    ]:
        """
        Prepare all data needed for inference sampling.
        Mirrors training_step's data preparation flow.

        This method:
        1. Builds sequence plans (conditioning information)
        2. Gets data and condition (encodes vision)
        3. Tokenizes text (conditional and unconditional for CFG)
        4. Builds a packed sequence to fetch conditioning masks
        5. Initializes noise with conditioning applied (as lists for variable shapes)
        6. Concatenates action noise with vision noise when the plan has action targets

        Args:
            data_batch: Raw data batch from dataloader.
            seed: Random seed(s) for noise generation.
            has_negative_prompt: If True, use negative prompt for unconditional branch.

        Returns:
            Tuple of:
                - sequence_plans: List of SequencePlan objects
                - gen_data_clean: GenerationDataClean with encoded tokens
                - cond_text_tokens: Conditional text tokens
                - uncond_text_tokens: Unconditional text tokens (for CFG)
                - initial_noise: List of noise tensors (one per sample), each containing
                  flattened vision (and optionally action) noise concatenated
                - condition_reference: List of clean reference tensors flattened
                  in the same order as initial_noise
                - condition_mask: List of masks flattened in the same order as
                  initial_noise, where 1 keeps condition_reference values fixed
                - has_noisy_actions: True when the packed sequence marks action
                  tokens as denoising targets. Determines whether the flattened
                  noise layout is [vision | action | sound] or [vision | sound];
                  callers must thread it to _get_velocity so both sides split the
                  flat state identically.
        """
        # 1. Build sequence plans (same as training)
        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch,
            input_video_key=self.input_video_key,
            input_image_key=self.input_image_key,
        )

        # 2. Get data and condition (same as training)
        # This encodes vision to x0_tokens. Pass each sample's vision conditioning
        # frame indexes so a causal tokenizer can skip encoding pixel frames that only
        # feed generated (non-conditioned) latent positions (e.g. WAM /
        # forward-dynamics condition latent frame 0 only, so only the first pixel frame
        # is encoded instead of the whole clip).
        vision_condition_indexes = [plan.condition_frame_indexes_vision for plan in sequence_plans]
        per_camera_vae_encoding = "enable_per_camera_vae_encoding" in data_batch
        gen_data_clean = self.get_data_and_condition(
            data_batch,
            vision_condition_indexes=vision_condition_indexes,
            retain_raw_state_vision=not per_camera_vae_encoding,
        )
        if gen_data_clean.num_views_per_vision_item is None:
            gen_data_clean.num_views_per_vision_item = self._get_explicit_single_view_metadata(
                data_batch,
                gen_data_clean.num_vision_items_per_sample,
                gen_data_clean.batch_size,
            )

        num_items_per_sample = gen_data_clean.num_vision_items_per_sample  # None for standard T2I/T2V

        # 3. Tokenize text (similar to training's _load_and_tokenize_text_data)
        cond_text_tokens, uncond_text_tokens = self._get_inference_text_tokens(data_batch, has_negative_prompt)

        # 4. Build packed sequence to fetch conditioning masks
        mask_timesteps = torch.zeros((gen_data_clean.batch_size,), dtype=torch.float32)  # [B]
        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            cond_text_tokens,
            gen_data_clean,
            mask_timesteps,
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
        )

        # 5. Initialize vision noise with conditioning
        condition_mask_vision = _vision_condition_masks(packed_sequence)
        assert gen_data_clean.x0_tokens_vision is not None, "Vision data required for inference noise"
        n_sample = (
            len(gen_data_clean.x0_tokens_vision)
            if gen_data_clean.num_vision_items_per_sample is None
            else len(gen_data_clean.num_vision_items_per_sample)
        )

        assert len(seed) == n_sample, (
            f"Seed list length {len(seed)} must have the same length as the number of samples {n_sample}"
        )

        # For image2image, num_items_per_sample could be > 1 (multi-vision),
        # so we need to repeat the seed for each vision item.
        seed_dict = {"vision": [], "action": [], "sound": []}
        for sample_idx in range(n_sample):
            num_vision_items = num_items_per_sample[sample_idx] if num_items_per_sample is not None else 1
            seed_dict["vision"].extend([seed[sample_idx]] * num_vision_items)
            seed_dict["action"].append(seed[sample_idx])
            seed_dict["sound"].append(seed[sample_idx])

        # Generate noise and apply conditioning per vision item (supports variable shapes).
        # Noise and the conditioning blend are kept in fp32 so the sampler accumulates
        # in full precision. x0_token is already fp32 (forced by .float() in
        # get_data_and_condition). The cast to model dtype happens inside velocity_fn.
        noise_vision_list: list[torch.Tensor] = []
        for i, (x0_token, cond_mask) in enumerate(
            zip(gen_data_clean.x0_tokens_vision, condition_mask_vision, strict=True)
        ):
            pure_noise_i = misc.arch_invariant_rand(
                tuple(x0_token.shape),
                self.tensor_kwargs_fp32["dtype"],
                self.tensor_kwargs_fp32["device"],
                seed_dict["vision"][i],  # Different seed per sample for diversity
            )  # [C,T,H,W]
            noise_i = cond_mask * x0_token + (1.0 - cond_mask) * pure_noise_i  # [C,T,H,W]
            noise_vision_list.append(noise_i)

        # 5b. Initialize LiDAR noise the same way, from its own stream's condition masks.
        noise_lidar_list: list[torch.Tensor] | None = None
        if gen_data_clean.x0_tokens_lidar is not None:
            assert packed_sequence.lidar is not None, "Packed LiDAR data required when the batch carries LiDAR"
            assert isinstance(packed_sequence.lidar.condition_mask, list), "LiDAR condition mask required"
            lidar_counts = gen_data_clean.num_lidar_items_per_sample or [1] * n_sample
            seed_lidar = [seed[sample_idx] for sample_idx, count in enumerate(lidar_counts) for _ in range(count)]
            noise_lidar_list = []
            for i, (x0_token, cond_mask) in enumerate(
                zip(gen_data_clean.x0_tokens_lidar, packed_sequence.lidar.condition_mask, strict=True)
            ):
                pure_noise_i = misc.arch_invariant_rand(
                    tuple(x0_token.shape),
                    self.tensor_kwargs_fp32["dtype"],
                    self.tensor_kwargs_fp32["device"],
                    seed_lidar[i],
                )  # [C,T,H,W]
                noise_lidar_list.append(cond_mask * x0_token + (1.0 - cond_mask) * pure_noise_i)  # [C,T,H,W]

        # 6. Initialize action noise if action_gen is True
        has_action = self.config.action_gen and any(plan.has_action for plan in sequence_plans)
        # Actions are denoising targets only when the packer marked action tokens
        # as noisy/supervised (WAM / inverse dynamics leave predicted steps
        # un-conditioned). Forward-dynamics plans and the temporal-causal packer
        # emit all-conditioning action tokens (actions are pure inputs), so there
        # is nothing to denoise — the network's _decode_action returns dummy zero
        # predictions in that case (same has_noisy_tokens predicate).
        has_noisy_actions = has_action and has_noisy_tokens(packed_sequence.action)
        noise_action_list: list[torch.Tensor] | None = None

        if has_noisy_actions:
            assert gen_data_clean.x0_tokens_action is not None, "Action data required when sequence plan has action"
            # Generate action noise per sample (x0_tokens_action is already dense, no None entries)
            noise_action_list = []
            for i, (x0_action, cond_mask_action) in enumerate(
                zip(gen_data_clean.x0_tokens_action, packed_sequence.action.condition_mask, strict=True)
            ):
                # Mirror the vision branch: keep the action noise and the conditioning blend in fp32
                # so the sampler accumulates in full precision and inference matches the fp32 flow
                # interpolation used in training. The cast to model dtype happens inside velocity_fn
                # (at the action encoder boundary in ``_encode_action``).
                pure_noise_action_i = misc.arch_invariant_rand(
                    tuple(x0_action.shape),
                    self.tensor_kwargs_fp32["dtype"],
                    self.tensor_kwargs_fp32["device"],
                    seed_dict["action"][i],  # Different seed per sample for diversity
                )  # [T,action_dim]
                cond_mask_action_fp32 = cond_mask_action.to(**self.tensor_kwargs_fp32)  # [T,1]
                noise_action_i = (
                    cond_mask_action_fp32 * x0_action.to(**self.tensor_kwargs_fp32)
                    + (1.0 - cond_mask_action_fp32) * pure_noise_action_i
                )  # [T,action_dim]
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    noise_action_i[:, gen_data_clean.raw_action_dim[i] :] = 0
                if gen_data_clean.action_valid_mask is not None and gen_data_clean.action_valid_mask[i] is not None:
                    valid_mask = gen_data_clean.action_valid_mask[i].to(device=noise_action_i.device)
                    noise_action_i[:, : valid_mask.numel()].mul_(valid_mask.to(dtype=noise_action_i.dtype))
                noise_action_list.append(noise_action_i)

        # 7. Initialize sound noise if sound_gen is True
        has_sound = self.config.sound_gen and any(plan.has_sound for plan in sequence_plans)
        noise_sound_list: list[torch.Tensor] | None = None

        if has_sound:
            assert gen_data_clean.x0_tokens_sound is not None, "Sound data required when sequence plan has sound"
            assert packed_sequence.sound is not None, "Packed sound data required when sound_gen is True"
            assert packed_sequence.sound.condition_mask is not None, "Sound condition mask required"
            assert isinstance(packed_sequence.sound.condition_mask, list), (
                "Sound condition mask must be a list of tensors for inference noise"
            )

            noise_sound_list = []
            for i, (x0_sound, cond_mask_sound) in enumerate(
                zip(gen_data_clean.x0_tokens_sound, packed_sequence.sound.condition_mask, strict=True)
            ):
                # Mirror the vision branch: keep the sound noise and the conditioning blend in fp32
                # so the sampler accumulates in full precision and inference matches the fp32 flow
                # interpolation used in training (``_add_noise_to_input`` noises sound in fp32).
                # The cast to model dtype happens inside velocity_fn.
                pure_noise_sound_i = misc.arch_invariant_rand(
                    tuple(x0_sound.shape),
                    self.tensor_kwargs_fp32["dtype"],
                    self.tensor_kwargs_fp32["device"],
                    seed_dict["sound"][i],  # Different seed per sample for diversity
                )  # [sound_channels,T_sound]
                # cond_mask_sound is (T, 1), x0_sound is (C, T) — transpose mask for broadcasting
                cond_mask_sound_fp32 = cond_mask_sound.T.to(**self.tensor_kwargs_fp32)  # [1,T_sound]
                noise_sound_i = (
                    cond_mask_sound_fp32 * x0_sound.to(**self.tensor_kwargs_fp32)
                    + (1.0 - cond_mask_sound_fp32) * pure_noise_sound_i
                )  # [sound_channels,T_sound]
                noise_sound_list.append(noise_sound_i)

        # 8. Concatenate vision, LiDAR, action, and sound noise per sample (flattened)
        # Order: [vision | lidar (if present) | action (if present) | sound (if present)],
        # matching the order the packer lays a sample out in.
        # noise_action_list and noise_sound_list are dense (only modality-having samples),
        # so we use separate indexes.
        initial_noise: list[torch.Tensor] = []
        condition_reference: list[torch.Tensor] = []
        condition_mask: list[torch.Tensor] = []
        idx_vision = 0
        idx_lidar = 0
        idx_action = 0
        idx_sound = 0

        for i in range(n_sample):
            parts = []
            condition_reference_parts = []
            condition_mask_parts = []

            # Flatten and concatenate all vision items for this sample
            num_vis = num_items_per_sample[i] if num_items_per_sample is not None else 1
            for _ in range(num_vis):
                parts.append(noise_vision_list[idx_vision].reshape(-1))
                x0_vision = gen_data_clean.x0_tokens_vision[idx_vision]  # [C,T,H,W]
                mask_vision = packed_sequence.vision.condition_mask[idx_vision].to(  # [T,1,1]
                    dtype=x0_vision.dtype, device=x0_vision.device
                )
                condition_reference_parts.append(x0_vision.reshape(-1))  # [N_vision]
                condition_mask_parts.append((mask_vision * torch.ones_like(x0_vision)).reshape(-1))  # [N_vision]
                idx_vision += 1

            if noise_lidar_list is not None and sequence_plans[i].has_lidar:
                assert packed_sequence.lidar is not None
                assert gen_data_clean.x0_tokens_lidar is not None
                lidar_counts = gen_data_clean.num_lidar_items_per_sample
                for _ in range(lidar_counts[i] if lidar_counts is not None else 1):
                    parts.append(noise_lidar_list[idx_lidar].reshape(-1))
                    x0_lidar = gen_data_clean.x0_tokens_lidar[idx_lidar]  # [C,T,H,W]
                    mask_lidar = packed_sequence.lidar.condition_mask[idx_lidar].to(  # [T,1,1]
                        dtype=x0_lidar.dtype, device=x0_lidar.device
                    )
                    condition_reference_parts.append(x0_lidar.reshape(-1))  # [N_lidar]
                    condition_mask_parts.append((mask_lidar * torch.ones_like(x0_lidar)).reshape(-1))  # [N_lidar]
                    idx_lidar += 1

            if noise_action_list is not None and sequence_plans[i].has_action:
                assert packed_sequence.action is not None
                assert packed_sequence.action.condition_mask is not None
                assert gen_data_clean.x0_tokens_action is not None
                parts.append(noise_action_list[idx_action].reshape(-1))
                x0_action = gen_data_clean.x0_tokens_action[idx_action].to(  # [T,D]
                    dtype=noise_action_list[idx_action].dtype,
                    device=noise_action_list[idx_action].device,
                )
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[idx_action] is not None:
                    x0_action = x0_action.clone()  # [T,D]
                    x0_action[:, gen_data_clean.raw_action_dim[idx_action] :] = 0  # [T,D]
                if (
                    gen_data_clean.action_valid_mask is not None
                    and gen_data_clean.action_valid_mask[idx_action] is not None
                ):
                    x0_action = x0_action.clone()  # [T,D]
                    valid_mask = gen_data_clean.action_valid_mask[idx_action].to(device=x0_action.device)
                    x0_action[:, : valid_mask.numel()].mul_(valid_mask.to(dtype=x0_action.dtype))
                mask_action = packed_sequence.action.condition_mask[idx_action].to(  # [T,1]
                    dtype=x0_action.dtype, device=x0_action.device
                )
                condition_reference_parts.append(x0_action.reshape(-1))  # [N_action]
                condition_mask_parts.append((mask_action * torch.ones_like(x0_action)).reshape(-1))  # [N_action]
                idx_action += 1

            if noise_sound_list is not None and sequence_plans[i].has_sound:
                assert packed_sequence.sound is not None
                assert packed_sequence.sound.condition_mask is not None
                assert gen_data_clean.x0_tokens_sound is not None
                parts.append(noise_sound_list[idx_sound].reshape(-1))
                x0_sound = gen_data_clean.x0_tokens_sound[idx_sound].to(  # [C_sound,T_sound]
                    dtype=noise_sound_list[idx_sound].dtype,
                    device=noise_sound_list[idx_sound].device,
                )
                mask_sound = packed_sequence.sound.condition_mask[idx_sound].T.to(  # [1,T_sound]
                    dtype=x0_sound.dtype, device=x0_sound.device
                )
                condition_reference_parts.append(x0_sound.reshape(-1))  # [N_sound]
                condition_mask_parts.append((mask_sound * torch.ones_like(x0_sound)).reshape(-1))  # [N_sound]
                idx_sound += 1

            initial_noise.append(torch.cat(parts, dim=0))  # [N_tokens_flat]
            condition_reference.append(torch.cat(condition_reference_parts, dim=0))  # [N_tokens_flat]
            condition_mask.append(torch.cat(condition_mask_parts, dim=0))  # [N_tokens_flat]

        return (
            sequence_plans,
            gen_data_clean,
            cond_text_tokens,
            uncond_text_tokens,
            initial_noise,
            condition_reference,
            condition_mask,
            has_noisy_actions,
        )

    def _can_reuse_inference_pack_templates(
        self,
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
    ) -> bool:
        """Return whether denoising can reuse request-local packed metadata templates."""
        if gen_data_clean.num_vision_items_per_sample is not None:
            return False
        if self.config.video_temporal_causal:
            return False
        if self.config.sound_gen and any(plan.has_sound for plan in sequence_plans):
            return False
        return True

    def _can_reuse_inference_text_kv(
        self,
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        *,
        reuse_pack_templates: bool,
        has_velocity_postprocess: bool,
    ) -> bool:
        """Return rank-local eligibility for request-local text K/V reuse.

        The caller must MIN-reduce this result over every participating FSDP
        and CFG-parallel group before installing the cache.
        """
        if has_velocity_postprocess or not reuse_pack_templates:
            return False
        # CFG parallelism and FSDP are safe when the caller synchronizes
        # eligibility: caches stay rank-local. CP changes the attention layout
        # and remains excluded pending separate validation.
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            return False
        if self.config.joint_attn_implementation != "two_way":
            return False
        if self.config.video_temporal_causal:
            return False
        if self.config.sound_gen and any(plan.has_sound for plan in sequence_plans):
            return False
        if gen_data_clean.batch_size != 1 or len(sequence_plans) != 1:
            return False
        if gen_data_clean.num_vision_items_per_sample is not None:
            return False
        return True

    def _make_inference_text_kv_cache(self, net: torch.nn.Module | None = None) -> list[UndKVCache]:
        """Create per-layer request-local text K/V caches for one CFG branch."""
        target_net = net or self.net
        return make_inference_text_kv_cache(len(target_net.language_model.model.layers))

    def _copy_timestep_to_template(self, timesteps: torch.Tensor | None, timestep: torch.Tensor) -> None:
        """Copy the sampler timestep into a packed template without a device-to-host round trip."""
        if timesteps is None or timesteps.numel() == 0:
            return
        timestep_values = timestep.reshape(-1).to(device=timesteps.device, dtype=timesteps.dtype)  # [B_or_1]
        timestep_value = timestep_values[0]  # []
        timesteps.copy_(timestep_value.expand_as(timesteps))  # [N_noisy_tokens]

    def _update_inference_pack_template(
        self,
        packed_sequence: PackedSequence,
        noise_x_vision: list[torch.Tensor],
        noise_x_lidar: list[torch.Tensor] | None,
        noise_x_action: list[torch.Tensor] | None,
        noise_x_sound: list[torch.Tensor] | None,
        timestep: torch.Tensor,
    ) -> PackedSequence:
        """Refresh noisy tokens and timesteps in a request-local packed sequence template."""
        if packed_sequence.vision is not None:
            packed_sequence.vision.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_vision]  # list[[C,T,H,W]]
            self._copy_timestep_to_template(packed_sequence.vision.timesteps, timestep)

        if noise_x_lidar is not None:
            assert packed_sequence.lidar is not None, "packed_sequence.lidar must exist when LiDAR noise is present"
            packed_sequence.lidar.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_lidar]  # list[[C,T,H,W]]
            self._copy_timestep_to_template(packed_sequence.lidar.timesteps, timestep)

        if noise_x_action is not None:
            assert packed_sequence.action is not None, "packed_sequence.action must exist when action noise is present"
            packed_sequence.action.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_action]  # list[[T,D]]
            self._copy_timestep_to_template(packed_sequence.action.timesteps, timestep)

        if noise_x_sound is not None:
            assert packed_sequence.sound is not None, "packed_sequence.sound must exist when sound noise is present"
            packed_sequence.sound.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_sound]  # list[[C_sound,T]]
            self._copy_timestep_to_template(packed_sequence.sound.timesteps, timestep)

        return packed_sequence

    @staticmethod
    def _duplicate_gen_data_clean(gd: GenerationDataClean) -> GenerationDataClean:
        """Return a GenerationDataClean with every per-sample field duplicated.

        Used by ``use_batched_cfg`` to feed (cond + uncond) through one forward
        of size 2N instead of two sequential forwards of size N. Field handling
        is generic (driven by the dataclass fields) so it stays correct if
        GenerationDataClean gains/loses fields: ``batch_size`` is doubled,
        scalar flags are copied, per-sample lists are concatenated, and batched
        tensors are duplicated along dim 0. Tensor/list references are reused
        (the forward path is read-only on these fields) so this is O(N) with no
        deep copies.
        """

        def _dup(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return torch.cat([value, value], dim=0)
            if isinstance(value, list):
                return value + value
            return value

        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(gd):
            val = getattr(gd, field.name)
            if field.name == "batch_size":
                kwargs[field.name] = val * 2
            elif field.name == "is_image_batch":
                kwargs[field.name] = val
            else:
                kwargs[field.name] = _dup(val)
        return GenerationDataClean(**kwargs)

    def _get_velocity(
        self,
        *,
        net: torch.nn.Module | None = None,
        noise_x: list[torch.Tensor],
        timestep: torch.Tensor,
        text_tokens: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        skip_text_tokens: bool = False,
        packed_sequence_template: PackedSequence | None = None,
        memory: MemoryState | None = None,
        has_noisy_actions: bool,
    ) -> list[torch.Tensor]:
        """
        Compute velocity prediction for a single sampling step.

        This method handles the full pipeline for one denoising step:
        1. Splits flattened noise_x into vision (and action) parts per sample
        2. Packs the input sequence with current noisy latents
        3. Runs the network via self.denoise()
        4. Applies velocity masks (zeroes out conditioned parts)
        5. Returns flattened velocities (concatenated vision + action per sample)

        Args:
            noise_x: List of noisy latents, each containing concatenated
                     vision (and optionally action) noise.
                     len(noise_x) == B, noise_x[i] is shape (D)
            timestep: Current timestep for each sample
            text_tokens: Tokenized text for each sample
            sequence_plans: Pre-computed sequence plans (from _prepare_inference_data)
            gen_data_clean: Pre-computed clean data (from _prepare_inference_data)
            skip_text_tokens: If True, skip text tokens (for CFG unconditional branch)
            packed_sequence_template: Optional request-local packed metadata template.
            memory: Optional request-local text K/V cache state for this CFG branch.
            has_noisy_actions: Whether actions are denoising targets, i.e. whether
                noise_x was laid out as [vision | action | sound]. Must be the
                packer-derived value from _prepare_inference_data so both sides
                split the flat state identically.

        Returns:
            Stacked flattened velocity tensors (one per sample), each containing
            concatenated vision (and optionally action) velocity
        """
        n_samples = len(noise_x)
        is_image_batch = gen_data_clean.is_image_batch
        has_action = self.config.action_gen and any(plan.has_action for plan in sequence_plans)
        if has_noisy_actions and not has_action:
            raise ValueError("has_noisy_actions=True requires an action-bearing sequence")
        num_items = gen_data_clean.num_vision_items_per_sample  # None for standard T2I/T2V
        has_sound = self.config.sound_gen and any(plan.has_sound for plan in sequence_plans)

        has_lidar = gen_data_clean.x0_tokens_lidar is not None
        num_lidar_items = gen_data_clean.num_lidar_items_per_sample

        # Split flattened noise_x into vision, LiDAR, action, and sound parts per sample
        # Order must match _prepare_inference_data:
        # [vision | lidar (if present) | action (if present) | sound (if present)]
        noise_x_vision: list[torch.Tensor] = []
        noise_x_lidar: list[torch.Tensor] | None = [] if has_lidar else None
        noise_x_action: list[torch.Tensor] | None = [] if has_noisy_actions else None
        noise_x_sound: list[torch.Tensor] | None = [] if has_sound else None

        vision_offset = 0  # tracks position in the flat x0_tokens_vision list
        lidar_offset = 0  # tracks position in the flat x0_tokens_lidar list
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            n_vis = num_items[i] if num_items is not None else 1
            offset = 0
            for j in range(n_vis):
                vision_shape = gen_data_clean.x0_tokens_vision[vision_offset + j].shape
                vision_dim = int(torch.prod(torch.tensor(vision_shape)))
                noise_vision_ij = noise_x[i][offset : offset + vision_dim].reshape(vision_shape)
                noise_x_vision.append(noise_vision_ij)
                offset += vision_dim
            vision_offset += n_vis

            if noise_x_lidar is not None and sequence_plans[i].has_lidar:
                n_lidar = num_lidar_items[i] if num_lidar_items is not None else 1
                for j in range(n_lidar):
                    lidar_shape = gen_data_clean.x0_tokens_lidar[lidar_offset + j].shape
                    lidar_dim = int(torch.prod(torch.tensor(lidar_shape)))
                    noise_x_lidar.append(noise_x[i][offset : offset + lidar_dim].reshape(lidar_shape))
                    offset += lidar_dim
                lidar_offset += n_lidar

            if has_noisy_actions and noise_x_action is not None:
                assert gen_data_clean.x0_tokens_action is not None
                action_shape = gen_data_clean.x0_tokens_action[idx_action].shape
                action_dim = int(torch.prod(torch.tensor(action_shape)))
                noise_x_action.append(noise_x[i][offset : offset + action_dim].reshape(action_shape))  # [T,action_dim]
                offset += action_dim
                idx_action += 1

            # Extract sound if present for this sample
            if has_sound and noise_x_sound is not None and sequence_plans[i].has_sound:
                assert gen_data_clean.x0_tokens_sound is not None
                sound_shape = gen_data_clean.x0_tokens_sound[idx_sound].shape
                sound_dim = int(torch.prod(torch.tensor(sound_shape)))
                noise_x_sound.append(
                    noise_x[i][offset : offset + sound_dim].reshape(sound_shape)
                )  # [sound_channels,T_sound]
                offset += sound_dim
                idx_sound += 1

        if has_action:
            if has_noisy_actions:
                x0_action_tokens = noise_x_action
            else:
                x0_action_tokens = gen_data_clean.x0_tokens_action
        else:
            x0_action_tokens = None

        gen_data_for_packing = GenerationDataClean(
            batch_size=n_samples,
            is_image_batch=is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=noise_x_vision,
            fps_vision=gen_data_clean.fps_vision,
            temporal_positions_vision=gen_data_clean.temporal_positions_vision,
            # Action fields
            raw_state_action=gen_data_clean.raw_state_action if has_action else None,
            x0_tokens_action=x0_action_tokens,
            action_domain_id=gen_data_clean.action_domain_id if has_action else None,
            fps_action=gen_data_clean.fps_action if has_action else None,
            raw_action_dim=gen_data_clean.raw_action_dim if has_action else None,
            action_valid_mask=gen_data_clean.action_valid_mask if has_action else None,
            # Sound fields
            raw_state_sound=gen_data_clean.raw_state_sound if has_sound else None,
            x0_tokens_sound=noise_x_sound if has_sound else None,
            fps_sound=gen_data_clean.fps_sound if has_sound else None,
            num_vision_items_per_sample=num_items,
            num_views_per_vision_item=gen_data_clean.num_views_per_vision_item,
            # LiDAR fields
            raw_state_lidar=gen_data_clean.raw_state_lidar,
            x0_tokens_lidar=noise_x_lidar if has_lidar else None,
            fps_lidar=gen_data_clean.fps_lidar,
            num_lidar_items_per_sample=num_lidar_items,
            # Multi-control transfer: carry per-control weights so the packer can
            # populate vision_item_split_lens / control_weights on the packed
            # sequence. Without this, multi_control_two_way_attention never runs
            # and all controls are blended equally (weights ignored).
            control_weights=gen_data_clean.control_weights,
        )

        if packed_sequence_template is None:
            packed_sequence = self._pack_input_sequence(
                sequence_plans,
                text_tokens,
                gen_data_for_packing,
                timestep.cpu(),
                include_end_of_generation_token=self._derive_include_end_of_generation_token(),
                skip_text_tokens=skip_text_tokens,
            )

            # Set the actual noisy latents (as lists)
            if packed_sequence.vision is not None:
                packed_sequence.vision.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_vision]  # list[[C,T,H,W]]

            if noise_x_lidar is not None:
                assert packed_sequence.lidar is not None, "packed_sequence.lidar must exist when the batch has LiDAR"
                packed_sequence.lidar.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_lidar]  # list[[C,T,H,W]]

            if has_noisy_actions and noise_x_action is not None:
                assert packed_sequence.action is not None, "packed_sequence.action must exist when has_action is True"
                packed_sequence.action.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_action]  # list[[T,D]]
                packed_sequence.action.domain_id = gen_data_clean.action_domain_id

            if has_sound and noise_x_sound is not None:
                assert packed_sequence.sound is not None, "packed_sequence.sound must exist when has_sound is True"
                packed_sequence.sound.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_sound]  # list[[C_sound,T]]

            packed_sequence.to_cuda()
        else:
            packed_sequence = self._update_inference_pack_template(
                packed_sequence_template,
                noise_x_vision,
                noise_x_lidar,
                noise_x_action,
                noise_x_sound,
                timestep,
            )

        # --- Network forward ---
        out = self.denoise(
            net=net,
            data_batch_packed=packed_sequence,
            memory=memory,
        )

        # --- Apply velocity masks ---
        # Zero out velocity for conditioned parts (they don't change during sampling)
        # Compute noisy_mask per sample (supports variable shapes)
        noisy_mask_vision = [1.0 - cond_mask for cond_mask in _vision_condition_masks(packed_sequence)]

        # Apply velocity mask per element - check if each sample has noisy tokens. A batch with
        # no camera stream has no mask to pair with the network's zero-weighted vision
        # prediction, so the zip stops immediately and leaves the velocity list empty.
        velocity_vision: list[torch.Tensor] = []
        for i, (pred, noisy_mask) in enumerate(zip(out["preds_vision"], noisy_mask_vision)):
            # pred: [C,T,H,W], noisy_mask: [T,1,1]
            has_noisy_tokens_i = noisy_mask.sum() > 0
            if has_noisy_tokens_i:
                # Apply mask to prediction
                velocity_vision.append(pred * noisy_mask.to(dtype=pred.dtype, device=pred.device))  # [C,T,H,W]
            else:
                # All tokens are conditioned - velocity should be zero
                velocity_vision.append(torch.zeros_like(pred))  # [C,T,H,W]

        # Handle LiDAR velocity
        velocity_lidar: list[torch.Tensor] | None = None
        if has_lidar and packed_sequence.lidar is not None and isinstance(packed_sequence.lidar.condition_mask, list):
            velocity_lidar = []
            for pred, cond_mask in zip(out["preds_lidar"], packed_sequence.lidar.condition_mask, strict=True):
                # pred: [C,T,H,W], cond_mask: [T,1,1]
                noisy_mask = 1.0 - cond_mask
                if noisy_mask.sum() > 0:
                    velocity_lidar.append(pred * noisy_mask.to(dtype=pred.dtype, device=pred.device))  # [C,T,H,W]
                else:
                    velocity_lidar.append(torch.zeros_like(pred))  # [C,T,H,W]

        # Handle action velocity
        velocity_action: list[torch.Tensor] | None = None
        if (
            has_noisy_actions
            and packed_sequence.action is not None
            and packed_sequence.action.condition_mask is not None
            and isinstance(packed_sequence.action.condition_mask, list)
        ):
            noisy_mask_action = [1.0 - cond_mask for cond_mask in packed_sequence.action.condition_mask]

            velocity_action = []
            for i, (pred, noisy_mask) in enumerate(zip(out["preds_action"], noisy_mask_action)):
                # pred: [T,action_dim], noisy_mask: [T,1]
                has_noisy_tokens_i = noisy_mask.sum() > 0
                if has_noisy_tokens_i:
                    v = pred * noisy_mask.to(dtype=pred.dtype, device=pred.device)  # [T,action_dim]
                else:
                    v = torch.zeros_like(pred)  # [T,action_dim]
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    v[:, gen_data_clean.raw_action_dim[i] :] = 0
                if gen_data_clean.action_valid_mask is not None and gen_data_clean.action_valid_mask[i] is not None:
                    valid_mask = gen_data_clean.action_valid_mask[i].to(device=v.device)
                    v[:, : valid_mask.numel()].mul_(valid_mask.to(dtype=v.dtype))
                velocity_action.append(v)

        # Handle sound velocity
        velocity_sound: list[torch.Tensor] | None = None
        if (
            has_sound
            and packed_sequence.sound is not None
            and packed_sequence.sound.condition_mask is not None
            and isinstance(packed_sequence.sound.condition_mask, list)
        ):
            noisy_mask_sound = [1.0 - cond_mask for cond_mask in packed_sequence.sound.condition_mask]

            velocity_sound = []
            for i, (pred, noisy_mask) in enumerate(zip(out["preds_sound"], noisy_mask_sound)):
                # pred: [sound_channels,T_sound], noisy_mask: [T_sound,1]
                has_noisy_tokens_i = noisy_mask.sum() > 0
                if has_noisy_tokens_i:
                    # noisy_mask is (T, 1), pred is (C, T) — transpose mask for broadcasting
                    velocity_sound.append(
                        pred * noisy_mask.T.to(dtype=pred.dtype, device=pred.device)
                    )  # [sound_channels,T_sound]
                else:
                    velocity_sound.append(torch.zeros_like(pred))  # [sound_channels,T_sound]

        # Concatenate vision, LiDAR, action, and sound velocities per sample (flattened)
        # Order must match _prepare_inference_data: [vision | lidar | action | sound]
        velocity_output: list[torch.Tensor] = []
        vis_offset = 0
        lidar_out_offset = 0
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            parts = []
            n_vis = num_items[i] if num_items is not None else 1

            for _ in range(n_vis):
                parts.append(velocity_vision[vis_offset].reshape(-1))
                vis_offset += 1

            if velocity_lidar is not None and sequence_plans[i].has_lidar:
                for _ in range(num_lidar_items[i] if num_lidar_items is not None else 1):
                    parts.append(velocity_lidar[lidar_out_offset].reshape(-1))
                    lidar_out_offset += 1

            if velocity_action is not None and sequence_plans[i].has_action:
                parts.append(velocity_action[idx_action].reshape(-1))
                idx_action += 1

            if velocity_sound is not None and sequence_plans[i].has_sound:
                parts.append(velocity_sound[idx_sound].reshape(-1))
                idx_sound += 1

            velocity_output.append(torch.cat(parts, dim=0))  # [N_tokens_flat]

        return velocity_output

    def _remove_padding_from_latent(
        self, x0_tokens_vision: list[torch.Tensor], frame_size: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Remove reflection padding from encoded latent vision tokens.

        Each sample in the batch may have different original dimensions, so we process
        each sample individually and return a list of latents with varying spatial sizes.

        The padding coordinates are scaled down by the spatial compression factor since
        we're operating in latent space.

        Args:
            x0_tokens_vision (list[torch.Tensor]): List of encoded latent tensors,
                each of shape (1, C, T, H_latent, W_latent)
                where H_latent, W_latent include scaled padding.
            frame_size (list[torch.Tensor]): List of tensors, each of shape (1,4) or (4,) containing
                [target_h, target_w, orig_h, orig_w] for each sample (in pixel space).

        Returns:
            list[torch.Tensor]: List of cropped latent tokens, each of shape (1, C, T, H_latent_cropped, W_latent_cropped).
                Each element may have different spatial sizes based on original image dimensions.
        """
        batch_size = len(x0_tokens_vision)
        spatial_factor = self.tokenizer_vision_gen.spatial_compression_factor
        cropped_latents = []
        for i in range(batch_size):
            # frame_size: [target_h, target_w, orig_h, orig_w] in pixel space
            # Normalize: frame_size[i] may be (1, 4) from IterativeJointDataLoader
            # or (4,) when loaded from safetensors in the eval/export path.
            fs = frame_size[i]
            if fs.dim() == 2:
                fs = fs[0]
            orig_h = int(fs[2].item())
            orig_w = int(fs[3].item())

            # Scale to latent space
            if orig_h // spatial_factor == 0 or orig_w // spatial_factor == 0:
                log.warning(
                    f"Zero-sized latent found: orig_h: {orig_h}, orig_w: {orig_w}, spatial_factor: {spatial_factor}"
                )

            orig_h_latent = max(orig_h // spatial_factor, 1)
            orig_w_latent = max(orig_w // spatial_factor, 1)

            # Crop to remove padding: x0_tokens_vision[i] shape is (1, C, T, H, W)
            cropped_latent = x0_tokens_vision[i][:, :, :, :orig_h_latent, :orig_w_latent].contiguous()
            cropped_latents.append(cropped_latent)

        return cropped_latents

    def _run_classifier_free_guidance(
        self,
        cond_tokens: list[list[int]],
        uncond_tokens: list[list[int]],
        skip_text_tokens_for_cfg: bool,
        single_velocity_fn: Callable[[list[list[int]], bool], list[torch.Tensor]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run classifier-free guidance, optionally in parallel via CFG parallelism.

        Args:
            cond_tokens: Tokenized text for the conditional branch.
            uncond_tokens: Tokenized text for the unconditional branch.
            skip_text_tokens_for_cfg: If True, skip text tokens in the
                unconditional branch.
            single_velocity_fn: Computes velocity for a given set of tokens.
                Accepts ``(tokens, skip_text_tokens)`` and returns a list of
                velocity tensors (one per sample).

        Returns:
            A tuple ``(cond_v, uncond_v)`` where each element is a list of
            velocity tensors (one per sample).
        """
        if self.parallel_dims is None or not self.parallel_dims.cfgp_enabled:
            return (
                single_velocity_fn(cond_tokens, False),
                single_velocity_fn(uncond_tokens, skip_text_tokens_for_cfg),
            )

        cfgp_rank = self.parallel_dims.cfgp_rank
        cfgp_size = self.parallel_dims.cfgp_size
        cfgp_group = self.parallel_dims.cfgp_mesh.get_group()
        cfgp_peer = (cfgp_rank + 1) % cfgp_size

        if cfgp_rank == 0:
            v_list = single_velocity_fn(cond_tokens, False)
        else:
            v_list = single_velocity_fn(uncond_tokens, skip_text_tokens_for_cfg)

        other_v_list = [torch.empty_like(v_i) for v_i in v_list]

        ops: list[dist.P2POp] = []
        for v_i, other_v_i in zip(v_list, other_v_list):
            ops.append(dist.P2POp(op=dist.isend, tensor=v_i, group_peer=cfgp_peer, group=cfgp_group))
            ops.append(dist.P2POp(op=dist.irecv, tensor=other_v_i, group_peer=cfgp_peer, group=cfgp_group))

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        if cfgp_rank == 0:
            return v_list, other_v_list
        else:
            return other_v_list, v_list

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        net: torch.nn.Module | None = None,
        sampler: Any | None = None,
        guidance: float = 1.5,
        guidance_interval: Optional[list[float]] = None,
        velocity_postprocess_builder: Optional[
            Callable[
                ..., Optional[Callable[[list[torch.Tensor], list[torch.Tensor], torch.Tensor], list[torch.Tensor]]]
            ]
        ] = None,
        seed: list[int] | int = 1,
        n_sample: int | None = None,
        has_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        sigma_max: float = 80.0,
        skip_text_tokens_for_cfg: bool = False,
        normalize_cfg: bool = False,
        use_batched_cfg: bool = False,
        upsample_task: str | None = None,
        upsample_max_new_tokens: int = 2048,
        upsample_temperature: float | None = 0.7,
        upsample_top_k: int | None = 20,
        upsample_top_p: float | None = 0.8,
        upsample_repetition_penalty: float = 1.0,
        upsample_presence_penalty: float = 0.0,
        upsample_seed: int | None = None,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        """
        Generate samples from the batch. Based on given batch, it will automatically
        determine whether to generate image or video samples.

        This method follows the same structure as training_step:
        1. Build sequence plans
        2. Get data and condition (encode vision)
        3. Initialize noise with conditioning (as lists for variable shapes)
        4. Run sampling loop with velocity function
        5. Return latents as lists (supports variable shapes)

        If ``upsample_task`` is not ``None``, the conditional captions are
        upsampled with the corresponding canonical task (``"t2i"``, ``"t2v"``,
        or ``"i2v"``) before sample generation.

        Args:
            data_batch (dict): Raw data batch from the dataloader.
            guidance (float): Classifier-free guidance weight.
            guidance_interval (list[float] | None): Optional timestep interval to apply guidance.
                For the timesteps (ranging between 0-1000) that fall between the interval, we perform CFG, otherwise, we skip the unconditional generation.
            seed (list[int] | int): Random seeds for noise generation. For all new use-cases,
                we use a list of seeds, one for each sample. The length of the list must match
                the number of samples. Legacy use-cases use a single integer seed which is
                incremented by 1 for each sample. But this is not supported anymore, and will
                raise an error if used.
            n_sample (int | None): Number of samples to generate; defaults to batch size.
            has_negative_prompt (bool): If True, use negative prompt for unconditional branch.
            num_steps (int): Number of sampling steps for the diffusion process.
            shift (float): Time shift parameter for the sampler.
            sigma_max (float): Maximum sigma for the EDM sampler.
            skip_text_tokens_for_cfg (bool): If True, skip text tokens in unconditional branch.
            normalize_cfg (bool): If True, normalize the CFG output.
            use_batched_cfg (bool): If True, run cond + uncond as one batched forward of
                size 2N (mutually exclusive with cfg-parallel). Opt-in; default False
                keeps this API conservative for saturated or memory-constrained workloads.
                ``OmniInference`` explicitly passes the configured
                ``SetupOverrides.use_batched_cfg`` policy.
                Reuses a doubled pack template and a shared text K/V cache when eligible.
            upsample_task (str | None): Canonical V4.2 task resolved by the
                inference caller. ``None`` disables native prompt upsampling.
                Otherwise, the conditional captions in
                ``data_batch[self.input_caption_key]`` are expanded via
                :meth:`upsample_captions` (which drives the reasoner tower's
                autoregressive loop) BEFORE downstream tokenization, so all
                subsequent code paths — cond/uncond build, sequence plans,
                velocity_fn, CFG — operate on the upsampled prompts uniformly.
                When ``upsample_task`` is ``"i2v"``, the
                anchor (first) raw frame of each sample's video — extracted
                from ``data_batch[self.input_video_key]`` via
                :meth:`_extract_upsample_conditioning_images` — is also
                forwarded to the upsampler's Qwen3-VL multimodal chat
                template so the rewritten caption is grounded in the
                actual conditioning frame, not just the original caption
                text.  The unconditional captions are NOT upsampled
                (negative prompts and empty strings pass through as-is).
            upsample_max_new_tokens (int): Maximum tokens generated per
                caption during upsampling.  Defaults to 2048.
            upsample_temperature, upsample_top_k, upsample_top_p: Native
                reasoner prompt upsampling sampling controls.
            upsample_repetition_penalty (float): CTRL/HF-style
                multiplicative logit penalty on tokens already seen
                in each caption's history.  ``>1.0`` discourages
                verbatim repetition, ``<1.0`` encourages it, ``1.0``
                (default) is identity and adds zero overhead.
            upsample_presence_penalty (float): OpenAI-style additive
                logit penalty (binary presence, not frequency) on
                tokens already seen.  ``>0`` discourages reuse,
                ``<0`` encourages it, ``0.0`` (default) is identity.
            upsample_seed (int | None): Optional integer seed for the
                native prompt upsampler's sampling RNG.  Forwarded to
                :meth:`upsample_captions` and through to
                :func:`unified_mot._impl_generate_reasoner_text`, which
                seeds a fresh device-local ``torch.Generator`` once and
                threads it into every ``torch.multinomial`` draw.
                ``None`` (default) uses the device's default RNG and is
                bit-identical to the pre-seed behavior.  Greedy
                upsampling (``upsample_temperature`` 0) never reads the
                generator, so the value has no effect in that case.
                NOTE: distinct from the per-sample ``seed`` argument
                above, which seeds the diffusion noise.

        Returns:
            Dict with keys:
                - "vision": List of vision latent tensors (one per sample, variable shapes)
                - "action": List of external-space action tensors
                  (only present when the packed sequence marks action tokens as
                  denoising targets, i.e. WAM / inverse dynamics. Plans whose
                  actions are pure conditioning — forward dynamics, temporal-causal
                  action conditioning — generate no actions and return no key)

        Raises:
            ValueError: If the number of samples does not match the number of noise tensors or seeds.
            ValueError: If the seed is a single integer. This is not supported anymore: `seed` must be
                a list of integers, one for each sample.
        """
        if isinstance(seed, int):
            raise ValueError(
                "Single integer seed is not supported anymore: `seed` must be a list of integers, one for each sample."
            )
        assert isinstance(seed, list)

        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            seed = _broadcast_seed(seed, self.parallel_dims.cp_mesh.get_group(), self.parallel_dims.cp_rank)

        if self.parallel_dims is not None and self.parallel_dims.cfgp_enabled:
            seed = _broadcast_seed(seed, self.parallel_dims.cfgp_mesh.get_group(), self.parallel_dims.cfgp_rank)

        # Optional reasoner-tower prompt upsampling.  See
        # :meth:`_maybe_apply_prompt_upsampling` for the full per-task
        # contract (task inference, image conditioning for i2v, V4.2
        # template specs, FSDP-safe caption broadcast).  Returns a
        # fresh ``data_batch`` (caller's dict is never mutated) with
        # ``self.input_caption_key`` overwritten by the upsampled
        # captions; returns the input unchanged when ``upsample_task=None``.
        data_batch = self._maybe_apply_prompt_upsampling(
            data_batch,
            upsample_task=upsample_task,
            upsample_max_new_tokens=upsample_max_new_tokens,
            upsample_temperature=upsample_temperature,
            upsample_top_k=upsample_top_k,
            upsample_top_p=upsample_top_p,
            upsample_repetition_penalty=upsample_repetition_penalty,
            upsample_presence_penalty=upsample_presence_penalty,
            upsample_seed=upsample_seed,
        )

        # Prepare all data (initial noise as list of flattened tensors per sample)
        (
            sequence_plans,
            gen_data_clean,
            cond_tokens,
            uncond_tokens,
            initial_noise,
            condition_reference,
            condition_mask,
            has_noisy_actions,
        ) = self._prepare_inference_data(data_batch, seed, has_negative_prompt)

        if n_sample is not None:
            assert n_sample == len(initial_noise), (
                f"Number of samples {n_sample} must match number of noise tensors {len(initial_noise)}"
            )
        else:
            n_sample = len(initial_noise)

        assert n_sample == len(seed), f"Number of samples {n_sample} must match number of seeds {len(seed)}"

        # Batched CFG: pre-compute the doubled (cond + uncond) context
        # once so velocity_fn can run both guidance branches as a single forward
        # of size 2N instead of two sequential forwards of size N. The forward
        # path is read-only on these fields, so tensor refs are reused (no copy).
        if use_batched_cfg:
            cfgp_on = self.parallel_dims is not None and self.parallel_dims.cfgp_enabled
            if cfgp_on:
                log.warning("CFG parallelism takes precedence over batched CFG; disabling batched CFG.")
                use_batched_cfg = False
        if use_batched_cfg:
            sequence_plans_doubled = list(sequence_plans) + list(sequence_plans)
            gen_data_clean_doubled = self._duplicate_gen_data_clean(gen_data_clean)
        else:
            sequence_plans_doubled = None
            gen_data_clean_doubled = None

        # Optional per-step velocity postprocess hook. Built once via a builder
        # that receives the prepared inference state. The returned callable (if
        # any) is invoked after the conditional forward on every step and can
        # modify the conditional velocity (e.g. inject control-CFG, attention
        # weighting, etc.). The model itself stays agnostic of what the hook
        # does — all transfer/edit-specific logic lives in the caller.
        velocity_postprocess: Optional[
            Callable[[list[torch.Tensor], list[torch.Tensor], torch.Tensor], list[torch.Tensor]]
        ] = None
        if velocity_postprocess_builder is not None:
            velocity_postprocess = velocity_postprocess_builder(
                model=self,
                net=net,
                cond_tokens=cond_tokens,
                sequence_plans=sequence_plans,
                gen_data_clean=gen_data_clean,
            )

        # In throughput-preset inference, each FSDP rank can hold a different
        # sample. Resolve the shard group before selecting the fast path so a
        # rank-local postprocess/packing difference cannot make peers issue a
        # different number of model forwards and deadlock collectives.
        if (
            self.parallel_dims is not None
            and self.parallel_dims.dp_shard_mesh is not None
            and torch.distributed.is_initialized()
            and self.parallel_dims.dp_shard_mesh.size() > 1
        ):
            _dp_shard_group = self.parallel_dims.dp_shard_mesh.get_group()
            _align_device = self.tensor_kwargs["device"]
        else:
            _dp_shard_group = None
            _align_device = None

        if (
            self.parallel_dims is not None
            and self.parallel_dims.cfgp_enabled
            and self.parallel_dims.cfgp_mesh is not None
            and torch.distributed.is_initialized()
            and self.parallel_dims.cfgp_mesh.size() > 1
        ):
            _cfgp_group = self.parallel_dims.cfgp_mesh.get_group()
            _align_device = self.tensor_kwargs["device"]
        else:
            _cfgp_group = None

        # Static eligibility for the fused cond+uncond fast path (velocity_fn also
        # requires ``_any_needs_guidance_path`` per step). Keep pack / text-KV alloc in
        # sync with this predicate so we do not build unused 2N resources.
        batched_cfg_fast_path = _all_group_ranks_allow(
            use_batched_cfg and velocity_postprocess is None and not skip_text_tokens_for_cfg,
            _dp_shard_group,
            _align_device,
        )

        reuse_pack_templates = self._can_reuse_inference_pack_templates(sequence_plans, gen_data_clean)
        cond_packed_sequence_template: PackedSequence | None = None
        uncond_packed_sequence_template: PackedSequence | None = None
        batched_packed_sequence_template: PackedSequence | None = None
        if reuse_pack_templates:
            include_end_of_generation_token = self._derive_include_end_of_generation_token()
            zero_timesteps = torch.zeros((gen_data_clean.batch_size,), dtype=torch.float32)  # [B]
            cond_packed_sequence_template = self._pack_input_sequence(
                sequence_plans,
                cond_tokens,
                gen_data_clean,
                zero_timesteps,
                include_end_of_generation_token=include_end_of_generation_token,
                skip_text_tokens=False,
            )
            cond_packed_sequence_template.to_cuda()
            if not batched_cfg_fast_path and (guidance != 1.0 or velocity_postprocess is not None):
                uncond_packed_sequence_template = self._pack_input_sequence(
                    sequence_plans,
                    uncond_tokens,
                    gen_data_clean,
                    zero_timesteps,
                    include_end_of_generation_token=include_end_of_generation_token,
                    skip_text_tokens=skip_text_tokens_for_cfg,
                )
                uncond_packed_sequence_template.to_cuda()

        if (
            reuse_pack_templates
            and batched_cfg_fast_path
            and sequence_plans_doubled is not None
            and gen_data_clean_doubled is not None
        ):
            zero_timesteps_doubled = torch.zeros((gen_data_clean_doubled.batch_size,), dtype=torch.float32)  # [2B]
            batched_packed_sequence_template = self._pack_input_sequence(
                sequence_plans_doubled,
                list(cond_tokens) + list(uncond_tokens),
                gen_data_clean_doubled,
                zero_timesteps_doubled,
                include_end_of_generation_token=include_end_of_generation_token,
                skip_text_tokens=False,
            )
            batched_packed_sequence_template.to_cuda()

        # Create a velocity function for a single sample (for use with self.sampler).
        # FSDP collective-sequence alignment (throughput-preset inference).
        #
        # In throughput-preset inference each rank holds a different sample,
        # and different samples can diverge on (a) whether text CFG or a
        # velocity postprocess hook requires a second forward, (b) whether
        # the fused batched path is locally eligible, and (c) ``num_steps``.
        # Any divergence makes the FSDP allgather sequence misalign across
        # ranks, deadlocking NCCL at the 30-min watchdog timeout.
        #
        # We align in three places:
        #   1. Before velocity_fn: MIN-reduce fused-path eligibility so every
        #      rank agrees on batched versus sequential CFG.
        #   2. Inside velocity_fn (per call): MAX-reduce whether any rank
        #      needs CFG/postprocessing; if so, every rank performs the same
        #      forward sequence. Ranks whose local decision was "no CFG" return
        #      ``cond_v`` directly — bit-identical to the original no-CFG path.
        #   3. Around the sampler call: all_reduce the local num_steps;
        #      ranks with local < max issue a dummy sampler call with the
        #      remaining steps to pad the FSDP allgather stream. The
        #      dummy call's output is discarded; ``latents`` is never
        #      re-bound.
        #
        # Guidance-path alignment is scoped to the FSDP shard group (the only
        # process group whose model-forward sequence is at risk). Text-KV
        # eligibility must additionally agree across CFG-parallel peers before
        # either peer installs the request-local attention dispatch.
        reuse_text_kv = self._can_reuse_inference_text_kv(
            sequence_plans,
            gen_data_clean,
            reuse_pack_templates=reuse_pack_templates,
            has_velocity_postprocess=velocity_postprocess is not None,
        )
        reuse_text_kv = _all_group_ranks_allow(
            reuse_text_kv,
            _dp_shard_group,
            _align_device,
        )
        reuse_text_kv = _all_group_ranks_allow(
            reuse_text_kv,
            _cfgp_group,
            _align_device,
        )
        cond_text_kv_cache: list[UndKVCache] | None = None
        uncond_text_kv_cache: list[UndKVCache] | None = None
        batched_text_kv_cache: list[UndKVCache] | None = None
        # Request-scoped: install only for this generate call and restore afterward so we
        # never permanently shadow another dispatch_attention_fn on the model.
        previous_attention_dispatch = None
        try:
            if reuse_text_kv:
                target_net = net or self.net
                if target_net is None:
                    raise RuntimeError(
                        "Cannot install memory-aware attention dispatch: net is None "
                        "(pass net= or ensure self.net is built)."
                    )
                previous_attention_dispatch = install_inference_memory_attention_dispatch(target_net)
                # Cond cache covers no-CFG steps. When the fused path is eligible,
                # CFG steps use ``batched_text_kv_cache`` only — skip allocating the
                # unused uncond stack. Otherwise keep the sequential cond+uncond pair.
                cond_text_kv_cache = self._make_inference_text_kv_cache(target_net)
                if batched_cfg_fast_path:
                    # Shared und K/V for the fused cond+uncond (B=2) pack. Filled on
                    # the first batched forward; later steps run gen-only with
                    # per-sample isolation in ``_attention_gen_with_cached_text``.
                    batched_text_kv_cache = self._make_inference_text_kv_cache(target_net)
                elif guidance != 1.0 or velocity_postprocess_builder is not None:
                    uncond_text_kv_cache = self._make_inference_text_kv_cache(target_net)

            def velocity_fn(noise_x: list[torch.Tensor], timestep: torch.Tensor) -> list[torch.Tensor]:
                # len(noise_x) == B, noise_x[i] is shape (D)
                # timestep is shape (B, 1)
                torch.compiler.cudagraph_mark_step_begin()

                assert timestep.ndim == 2, f"timestep must be 2D, got {timestep.shape}"
                assert timestep.shape == (1, 1), f"timestep must be (1, 1), got {timestep.shape}"

                # Expand timestep to (B, 1)
                timestep = timestep.repeat(len(noise_x), 1)  # [B,1]

                def _single_velocity_fn(tokens: list[list[int]], skip_text_tokens: bool) -> list[torch.Tensor]:
                    nonlocal uncond_text_kv_cache
                    packed_sequence_template = None
                    text_kv_cache: list[UndKVCache] | None = None
                    if reuse_pack_templates:
                        if tokens is cond_tokens and not skip_text_tokens:
                            packed_sequence_template = cond_packed_sequence_template
                            text_kv_cache = cond_text_kv_cache
                        elif tokens is uncond_tokens and skip_text_tokens == skip_text_tokens_for_cfg:
                            packed_sequence_template = uncond_packed_sequence_template
                            if reuse_text_kv and uncond_text_kv_cache is None:
                                uncond_text_kv_cache = self._make_inference_text_kv_cache(net)
                            text_kv_cache = uncond_text_kv_cache
                    memory: MemoryState | None = (
                        InferenceTextKVMemoryState(text_kv_cache) if text_kv_cache is not None else None
                    )
                    return self._get_velocity(
                        net=net,
                        noise_x=noise_x,
                        timestep=timestep,
                        text_tokens=tokens,
                        sequence_plans=sequence_plans,
                        gen_data_clean=gen_data_clean,
                        skip_text_tokens=skip_text_tokens,
                        packed_sequence_template=packed_sequence_template,
                        memory=memory,
                        has_noisy_actions=has_noisy_actions,
                    )

                needs_text_cfg = guidance != 1.0
                if needs_text_cfg and guidance_interval is not None:
                    assert len(guidance_interval) == 2, f"guidance_interval must be [lo, hi], got {guidance_interval}"
                    t_lo, t_hi = guidance_interval
                    needs_text_cfg = t_lo < timestep[0].item() < t_hi

                # FSDP alignment: if ANY rank in the shard group needs a second
                # forward for text CFG or a velocity postprocess hook, every rank
                # must issue two sequential forwards (or one globally eligible
                # batched forward) so the allgather sequence stays aligned.
                needs_guidance_path = needs_text_cfg or velocity_postprocess is not None
                _any_needs_guidance_path = _any_dp_shard_rank_needs_guidance_path(
                    needs_guidance_path,
                    _dp_shard_group,
                    _align_device,
                )

                # Fast path: no rank needs CFG or postprocessing — single forward.
                if not _any_needs_guidance_path:
                    return _single_velocity_fn(cond_tokens, skip_text_tokens=False)

                # Batched-CFG fast path (opt-in): run cond + uncond as one forward of
                # size 2N instead of two sequential N forwards. Removes the per-step
                # python orchestration between the two forwards. Only taken when there
                # is no postprocess hook, text-CFG is needed somewhere in the FSDP
                # shard group this step, and we are not skipping text tokens for the
                # uncond branch. cfg-parallel was already ruled out at setup. Falls
                # through to the standard path otherwise.
                # Under FSDP, every rank takes this path whenever ``_any_needs_guidance_path``
                # so the allgather sequence stays aligned (same 1-forward-per-step
                # count); ranks whose local decision was no-CFG still return cond only.
                if batched_cfg_fast_path and _any_needs_guidance_path:
                    assert sequence_plans_doubled is not None and gen_data_clean_doubled is not None
                    n = len(noise_x)
                    batched_memory: MemoryState | None = (
                        InferenceTextKVMemoryState(batched_text_kv_cache) if batched_text_kv_cache is not None else None
                    )
                    velocities = self._get_velocity(
                        net=net,
                        noise_x=noise_x + noise_x,
                        timestep=torch.cat([timestep, timestep], dim=0),
                        text_tokens=list(cond_tokens) + list(uncond_tokens),
                        sequence_plans=sequence_plans_doubled,
                        gen_data_clean=gen_data_clean_doubled,
                        skip_text_tokens=False,
                        packed_sequence_template=batched_packed_sequence_template,
                        memory=batched_memory,
                        has_noisy_actions=has_noisy_actions,
                    )
                    cond_v = velocities[:n]
                    uncond_v = velocities[n:]
                    if not needs_text_cfg:
                        return cond_v
                    v_pred = [u_i + guidance * (c_i - u_i) for c_i, u_i in zip(cond_v, uncond_v)]
                    if normalize_cfg:
                        v_pred = [
                            v_i * (torch.norm(c_i) / (torch.norm(v_i) + 1e-8)).clamp(min=0.0, max=1.0)
                            for v_i, c_i in zip(v_pred, cond_v)
                        ]
                    return v_pred

                # Fast path: only text-CFG and no postprocess — preserve the
                # cfgp-parallel branch so two-rank CFG parallelism stays available.
                if velocity_postprocess is None:
                    cond_v, uncond_v = self._run_classifier_free_guidance(
                        cond_tokens=cond_tokens,
                        uncond_tokens=uncond_tokens,
                        skip_text_tokens_for_cfg=skip_text_tokens_for_cfg,
                        single_velocity_fn=_single_velocity_fn,
                    )
                    if not needs_text_cfg:
                        # Peers needed CFG so we ran the uncond forward to keep
                        # FSDP allgather aligned; locally we still return cond.
                        return cond_v
                    v_pred = [u_i + guidance * (c_i - u_i) for c_i, u_i in zip(cond_v, uncond_v)]
                    if normalize_cfg:
                        v_pred = [
                            v_i * (torch.norm(c_i) / (torch.norm(v_i) + 1e-8)).clamp(min=0.0, max=1.0)
                            for v_i, c_i in zip(v_pred, cond_v)
                        ]
                    return v_pred

                # Conditional forward, then per-step postprocess hook. Hook runs
                # sequentially; cfgp parallelism not used on this path.
                cond_v_full = _single_velocity_fn(cond_tokens, skip_text_tokens=False)
                cond_v = velocity_postprocess(cond_v_full, noise_x, timestep)

                uncond_v = _single_velocity_fn(uncond_tokens, skip_text_tokens=skip_text_tokens_for_cfg)
                if not needs_text_cfg:
                    # Same alignment story as above for the postprocess branch.
                    return cond_v

                v_pred = [u_i + guidance * (c_i - u_i) for c_i, u_i in zip(cond_v, uncond_v)]

                if normalize_cfg:
                    v_pred = [
                        v_i * (torch.norm(c_i) / (torch.norm(v_i) + 1e-8)).clamp(min=0.0, max=1.0)
                        for v_i, c_i in zip(v_pred, cond_v)
                    ]

                return v_pred

            # Run sampler for all samples at once.
            sampler = sampler or self.sampler
            scheduler_type = self.config.rectified_flow_inference_config.scheduler_type
            if isinstance(sampler, FixedStepSampler):
                num_steps = len(sampler.t_list) - 1
                shift = 0.0

            # FSDP collective-sequence alignment (sampler outer loop). See the
            # large block above the velocity_fn definition for the full
            # rationale. all_reduce on the local num_steps so every rank knows
            # the max; below, ranks with local < max issue a dummy sampler call
            # to pad their FSDP allgather sequence.
            if _dp_shard_group is not None:
                _local_steps_t = torch.tensor([num_steps], device=_align_device, dtype=torch.int32)
                torch.distributed.all_reduce(_local_steps_t, op=torch.distributed.ReduceOp.MAX, group=_dp_shard_group)
                _max_num_steps = int(_local_steps_t.item())
            else:
                _max_num_steps = num_steps
            _extra_num_steps = _max_num_steps - num_steps

            if isinstance(sampler, FixedStepSampler):
                log.info(f"Using sampler: FixedStep (t_list={sampler.t_list}, sample_type={sampler.sample_type})")
            elif scheduler_type == "unipc":
                log.info(f"Using sampler: UniPC (shift={shift}, num_steps={num_steps})")
            else:
                log.info(f"Using sampler: EDM (sigma_max={sigma_max}, num_steps={num_steps})")

            fixed_step_sampler_kwargs = {}
            if isinstance(sampler, FixedStepSampler):
                fixed_step_sampler_kwargs = {
                    "condition_reference": condition_reference,
                    "condition_mask": condition_mask,
                }

            # Mixed-precision diffusion steps: select W8A16/W8A8 once per
            # sampler step; reset (trace + staging cleanup) when the request
            # ends, including on error.
            _mixed_precision_runtime = getattr(self.net, "_mixed_precision_runtime", None)
            _step_callback = _mixed_precision_runtime.set_step if _mixed_precision_runtime is not None else None

            try:
                if isinstance(sampler, FixedStepSampler) or scheduler_type == "unipc":
                    latents = sampler(
                        velocity_fn,
                        initial_noise,
                        num_steps=num_steps,
                        shift=shift,
                        seed=seed,
                        step_callback=_step_callback,
                        **fixed_step_sampler_kwargs,
                    )
                    if _extra_num_steps > 0:
                        # Dummy sampler call to issue (_extra_num_steps × per-step)
                        # FSDP allgathers; output discarded so `latents` keeps the
                        # real result captured above. Slow ranks have _extra_num_steps==0
                        # here, but they're issuing the SAME number of in-sampler
                        # collectives via their longer real call.
                        log.debug(
                            f"FSDP alignment: dummy sampler run with {_extra_num_steps} "
                            f"extra steps (local={num_steps}, max={_max_num_steps})"
                        )
                        if _mixed_precision_runtime is not None:
                            # Deterministic cheap padding: the FSDP-alignment dummy
                            # call must not inherit the last real step's W8A16.
                            _mixed_precision_runtime.set_base_precision()
                        _ = sampler(
                            velocity_fn,
                            latents,
                            num_steps=_extra_num_steps,
                            shift=shift,
                            seed=seed,
                            **fixed_step_sampler_kwargs,
                        )
                else:
                    # EDM Sampler
                    chunk_sizes = [_x.shape[0] for _x in initial_noise]
                    initial_noise = torch.cat(initial_noise, dim=0)

                    def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                        assert sigma.ndim == 0, f"sigma must be 0D, got {sigma.shape}"
                        timestep_rf = sigma * float(self.config.rectified_flow_inference_config.num_train_timesteps)

                        # Convert noise_x to list of tensors for velocity_fn, and then
                        # concatenate the results back into a single tensor.
                        _noise_x = list(torch.split(noise_x, chunk_sizes, dim=0))
                        _velocity_pred = velocity_fn(_noise_x, timestep_rf.reshape(1, 1))
                        velocity_pred = torch.cat(_velocity_pred, dim=0)

                        x0_pred = noise_x - sigma * velocity_pred
                        return x0_pred

                    latents = sampler(
                        x0_fn,
                        initial_noise,
                        num_steps=num_steps,
                        sigma_max=sigma_max,
                        sigma_min=0.002,
                        solver_option="2ab",
                        step_callback=_step_callback,
                    )
                    if _extra_num_steps > 0:
                        # Pad the FSDP allgather sequence with ``_extra_num_steps``
                        # direct ``x0_fn`` calls instead of a second EDM sampler
                        # run. Avoids two EDM-specific footguns:
                        #   (1) ``EDMSampler._forward_impl`` always runs an extra
                        #       ``sample_clean`` denoiser forward (see
                        #       ``cosmos_framework/model/vfm/diffusion/samplers/edm.py``).
                        #       A nested sampler call would add one too many
                        #       forwards on fast ranks, since the slow rank's
                        #       single call also pays the ``sample_clean`` cost.
                        #   (2) ``get_rev_ts(..., num_steps=0)`` divides by zero,
                        #       producing NaN sigmas. The fix's ``extra==1`` edge
                        #       case would need num_steps=0 to balance the count.
                        # Direct ``x0_fn`` calls bypass both: each call routes
                        # through the same ``velocity_fn`` closure (so the
                        # per-call CFG all_reduce still aligns ranks), issues
                        # exactly one model forward, and discards its return.
                        # ``latents`` is the catted single tensor at this point;
                        # the dummy sigma value is irrelevant for collective
                        # alignment because the model's allgather sequence is
                        # determined by tensor shapes, not sigma.
                        log.debug(
                            f"FSDP alignment: padding {_extra_num_steps} dummy x0_fn calls "
                            f"(local={num_steps}, max={_max_num_steps})"
                        )
                        if _mixed_precision_runtime is not None:
                            # Deterministic cheap padding: the FSDP-alignment dummy
                            # call must not inherit the last real step's W8A16.
                            _mixed_precision_runtime.set_base_precision()
                        # ``x0_fn`` expects a sigma in the RF domain (the real EDM
                        # loop converts raw sigmas via ``sigmas_L / (1 + sigmas_L)``
                        # at edm.py:174, landing them in ``(0, 1)``). Mirror that
                        # transform here so the dummy call's timestep stays in the
                        # same numerical domain as a real sampler step. The exact
                        # value doesn't matter for collective alignment, only the
                        # domain.
                        _dummy_sigma = latents.new_tensor(sigma_max / (1.0 + sigma_max))
                        for _ in range(_extra_num_steps):
                            _ = x0_fn(latents, _dummy_sigma)
                    latents = list(torch.split(latents, chunk_sizes, dim=0))
            finally:
                if _mixed_precision_runtime is not None:
                    _mixed_precision_runtime.reset()

            # Split flattened latents back into vision latents, LiDAR latents, external actions,
            # and sound latents. Mirror the per-sample logic from _prepare_inference_data:
            # Order: [vision | lidar (if present) | action (if present) | sound (if present)]
            # lidar/action/sound lists are dense (only modality-having samples), so use separate indexes.
            result_vision: list[torch.Tensor] = []
            result_lidar: list[torch.Tensor] = []
            result_action: list[torch.Tensor] = []
            result_sound: list[torch.Tensor] = []
            action_processing_records = get_action_processing_records(data_batch)
            idx_vision = 0
            idx_lidar = 0
            idx_action = 0
            idx_sound = 0
            num_vision_items = gen_data_clean.num_vision_items_per_sample
            num_lidar_items = gen_data_clean.num_lidar_items_per_sample

            for i in range(n_sample):
                offset = 0

                # Extract vision
                n_vis = num_vision_items[i] if num_vision_items is not None else 1
                for j in range(n_vis):
                    vision_shape = gen_data_clean.x0_tokens_vision[idx_vision + j].shape
                    vision_dim = int(torch.prod(torch.tensor(vision_shape)))
                    # Keep standard single-vision inference dense, including conditioned Action
                    # vision. Multi-item camera/LiDAR inference returns only generated items.
                    if n_vis == 1 or is_item_generated(
                        sequence_plans[i].condition_frame_indexes_vision,
                        item_idx=j,
                        num_items=n_vis,
                        latent_t=int(vision_shape[2]),
                    ):
                        result_vision.append(latents[i][offset : offset + vision_dim].reshape(vision_shape))
                    offset += vision_dim
                idx_vision += n_vis

                # Extract LiDAR if present, on the same rule: every item the plan generates.
                if sequence_plans[i].has_lidar:
                    assert gen_data_clean.x0_tokens_lidar is not None
                    n_lidar = num_lidar_items[i] if num_lidar_items is not None else 1
                    for j in range(n_lidar):
                        lidar_shape = gen_data_clean.x0_tokens_lidar[idx_lidar + j].shape
                        lidar_dim = int(torch.prod(torch.tensor(lidar_shape)))
                        if is_item_generated(
                            sequence_plans[i].condition_frame_indexes_lidar,
                            item_idx=j,
                            num_items=n_lidar,
                            latent_t=int(lidar_shape[2]),
                        ):
                            result_lidar.append(latents[i][offset : offset + lidar_dim].reshape(lidar_shape))
                        offset += lidar_dim
                    idx_lidar += n_lidar

                # Extract action if present
                if has_noisy_actions and sequence_plans[i].has_action:
                    assert gen_data_clean.x0_tokens_action is not None
                    action_shape = gen_data_clean.x0_tokens_action[idx_action].shape
                    action_dim = int(torch.prod(torch.tensor(action_shape)))
                    action_model = latents[i][offset : offset + action_dim].reshape(action_shape)  # [T,D_model]
                    action_record = action_processing_records[i] if i < len(action_processing_records) else None
                    if action_record is None:
                        raise ValueError(
                            f"Generated action output for sample {i} cannot be externalized without "
                            "action_processing_record"
                        )
                    action_external = ActionProcessor.postprocess_action(action_model, action_record)  # [T,D_raw]
                    result_action.append(action_external)
                    offset += action_dim
                    idx_action += 1

                # Extract sound if present
                if self.config.sound_gen and sequence_plans[i].has_sound:
                    assert gen_data_clean.x0_tokens_sound is not None
                    sound_shape = gen_data_clean.x0_tokens_sound[idx_sound].shape
                    sound_dim = int(torch.prod(torch.tensor(sound_shape)))
                    result_sound.append(latents[i][offset : offset + sound_dim].reshape(sound_shape))
                    offset += sound_dim
                    idx_sound += 1

            result: dict[str, list[torch.Tensor]] = {"vision": result_vision}
            if result_lidar:
                result["lidar"] = result_lidar
            if has_noisy_actions and result_action:
                result["action"] = result_action
            if self.config.sound_gen and len(result_sound) > 0:
                result["sound"] = result_sound
            return result
        finally:
            if previous_attention_dispatch is not None:
                restore_inference_attention_dispatch(previous_attention_dispatch)

    def _extract_condition_images_for_visualization(
        self,
        gen_data_clean: GenerationDataClean,
        sequence_plans: list[SequencePlan],
        n_samples: int,
    ) -> list[torch.Tensor | None]:
        """Extract condition images from gen_data_clean for visualization.

        For image editing, raw_state_vision is a flat list of individually-encoded
        images (e.g. [src1, tgt1, src2, tgt2, ...]).  The first vision item for
        each sample is the condition (source) image.  This method extracts it and
        resizes to match the target for side-by-side display.

        Args:
            gen_data_clean: Clean data containing raw vision states.
            sequence_plans: Sequence plans for each sample.
            n_samples: Number of samples to process.

        Returns:
            List of condition image tensors (one per sample with condition frames).
        """
        condition_images: list[torch.Tensor | None] = []

        if gen_data_clean.num_vision_items_per_sample is not None:
            # Multi-item (image editing): raw_state_vision is flat [src1, tgt1, src2, tgt2, ...]
            vision_offset = 0
            for i in range(n_samples):
                num_items = gen_data_clean.num_vision_items_per_sample[i]
                if num_items >= 2:
                    cond_frame = gen_data_clean.raw_state_vision[vision_offset]  # (1, C, 1, H_s, W_s)
                    target_frame = gen_data_clean.raw_state_vision[vision_offset + 1]  # (1, C, 1, H_t, W_t)
                    # Resize condition frame to match target size for visualization
                    if cond_frame.shape[-2:] != target_frame.shape[-2:]:
                        cond_frame = torch.nn.functional.interpolate(
                            cond_frame.squeeze(2),  # (1, C, H, W)
                            size=target_frame.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).unsqueeze(2)  # (1, C, 1, H, W)
                    condition_images.append(cond_frame)
                else:
                    condition_images.append(None)
                vision_offset += num_items
        else:
            # Standard single-item mode: check condition_frame_indexes_vision
            for i in range(n_samples):
                plan = sequence_plans[i]
                if len(plan.condition_frame_indexes_vision) > 0 and gen_data_clean.raw_state_vision is not None:
                    raw_vision = gen_data_clean.raw_state_vision[i]  # (1, C, T, H, W)
                    condition_images.append(raw_vision[:, :, 0:1, :, :])
                else:
                    condition_images.append(None)

        return condition_images

    def _slice_gen_data_clean(self, gen_data_clean: GenerationDataClean, start: int, limit: int) -> GenerationDataClean:
        """Extract a subset of GenerationDataClean for inference.

        The samples in [start:limit] are extracted from the original GenerationDataClean.

        For image editing (``num_vision_items_per_sample`` is set), the sample index refers to
        the *real sample* index. The method computes the correct slice of the flat
        ``x0_tokens_vision`` / ``raw_state_vision`` lists using the item counts and
        preserves ``num_vision_items_per_sample`` on the returned subset so that
        downstream packing works correctly.

        Args:
            gen_data_clean: GenerationDataClean to slice.
            start: Start index of the slice.
            limit: Limit index of the slice.

        Returns:
            Sliced GenerationDataClean.
        """
        # x0_tokens_action can be an empty list for general-video samples, not just None.
        has_action = bool(gen_data_clean.x0_tokens_action)
        has_sound = bool(gen_data_clean.x0_tokens_sound)

        # Determine vision slice for this sample
        num_items = gen_data_clean.num_vision_items_per_sample
        if num_items is not None:
            # Multi-item mode: compute flat-list offset
            vis_start = sum(num_items[:start])  # number of all the vision tokens before the start
            vis_end = sum(num_items[:limit])
            subset_x0_vision = gen_data_clean.x0_tokens_vision[vis_start:vis_end]
            subset_raw_vision = (
                gen_data_clean.raw_state_vision[vis_start:vis_end] if gen_data_clean.raw_state_vision else None
            )
            subset_temporal_positions_vision = (
                gen_data_clean.temporal_positions_vision[vis_start:vis_end]
                if gen_data_clean.temporal_positions_vision
                else None
            )
            subset_num_items = num_items[start:limit]
            vision_item_slice = slice(vis_start, vis_end)
        else:
            # Standard single-item mode
            subset_x0_vision = gen_data_clean.x0_tokens_vision[start:limit]
            subset_raw_vision = (
                gen_data_clean.raw_state_vision[start:limit] if gen_data_clean.raw_state_vision else None
            )
            subset_temporal_positions_vision = (
                gen_data_clean.temporal_positions_vision[start:limit]
                if gen_data_clean.temporal_positions_vision
                else None
            )
            subset_num_items = None
            vision_item_slice = slice(start, limit)
        # Parallel to the flattened vision item list, so it follows the same slice.
        subset_num_views_per_vision_item = (
            gen_data_clean.num_views_per_vision_item[vision_item_slice]
            if gen_data_clean.num_views_per_vision_item is not None
            else None
        )
        fps_vision = gen_data_clean.fps_vision[start:limit] if gen_data_clean.fps_vision is not None else None

        # The LiDAR stream is grouped by sample the same way, so it takes its own slice.
        num_lidar_items = gen_data_clean.num_lidar_items_per_sample
        if gen_data_clean.x0_tokens_lidar is None:
            subset_x0_lidar = None
            subset_raw_lidar = None
            subset_num_lidar_items = None
        else:
            lidar_counts = num_lidar_items if num_lidar_items is not None else [1] * gen_data_clean.batch_size
            lidar_slice = slice(sum(lidar_counts[:start]), sum(lidar_counts[:limit]))
            subset_x0_lidar = gen_data_clean.x0_tokens_lidar[lidar_slice]
            subset_raw_lidar = gen_data_clean.raw_state_lidar[lidar_slice] if gen_data_clean.raw_state_lidar else None
            subset_num_lidar_items = None if num_lidar_items is None else num_lidar_items[start:limit]
        fps_lidar = gen_data_clean.fps_lidar[start:limit] if gen_data_clean.fps_lidar is not None else None

        if has_action:
            subset_raw_action = (
                gen_data_clean.raw_state_action[start:limit] if gen_data_clean.raw_state_action else None
            )
            x0_tokens_action = gen_data_clean.x0_tokens_action[start:limit]
            fps_action = gen_data_clean.fps_action[start:limit] if gen_data_clean.fps_action is not None else None
            action_domain_id = gen_data_clean.action_domain_id[start:limit] if gen_data_clean.action_domain_id else None
            raw_action_dim = gen_data_clean.raw_action_dim[start:limit] if gen_data_clean.raw_action_dim else None
            action_valid_mask = (
                gen_data_clean.action_valid_mask[start:limit] if gen_data_clean.action_valid_mask else None
            )
        else:
            subset_raw_action = None
            x0_tokens_action = None
            fps_action = None
            action_domain_id = None
            raw_action_dim = None
            action_valid_mask = None

        if has_sound:
            subset_raw_sound = gen_data_clean.raw_state_sound[start:limit] if gen_data_clean.raw_state_sound else None
            x0_tokens_sound = gen_data_clean.x0_tokens_sound[start:limit]
            fps_sound = gen_data_clean.fps_sound[start:limit] if gen_data_clean.fps_sound is not None else None
        else:
            subset_raw_sound = None
            x0_tokens_sound = None
            fps_sound = None

        return GenerationDataClean(
            batch_size=limit - start,
            is_image_batch=gen_data_clean.is_image_batch,
            raw_state_vision=subset_raw_vision,
            raw_state_action=subset_raw_action,
            raw_state_sound=subset_raw_sound,
            x0_tokens_vision=subset_x0_vision,
            x0_tokens_action=x0_tokens_action,
            x0_tokens_sound=x0_tokens_sound,
            fps_vision=fps_vision,
            temporal_positions_vision=subset_temporal_positions_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
            action_domain_id=action_domain_id,
            raw_action_dim=raw_action_dim,
            action_valid_mask=action_valid_mask,
            num_vision_items_per_sample=subset_num_items,
            num_views_per_vision_item=subset_num_views_per_vision_item,
            raw_state_lidar=subset_raw_lidar,
            x0_tokens_lidar=subset_x0_lidar,
            fps_lidar=fps_lidar,
            num_lidar_items_per_sample=subset_num_lidar_items,
        )

    @torch.no_grad()
    def validation_step(self, data_batch: dict[str, torch.Tensor], iteration: int):
        pass

    @torch.no_grad()
    def forward(self, xt, t):
        pass

    @staticmethod
    def _get_explicit_single_view_metadata(
        data_batch: dict[str, Any],
        num_vision_items_per_sample: list[int] | None,
        batch_size: int,
    ) -> list[int] | None:
        """Read one-view item metadata without opting into uint8 per-camera VAE encoding."""
        num_views = read_positive_int_metadata(
            data_batch,
            "num_views_per_vision_item",
            expected_count=sum(num_vision_items_per_sample or [1] * batch_size),
        )
        if num_views is not None and any(num_view != 1 for num_view in num_views):
            raise ValueError(
                "Explicit num_views_per_vision_item metadata is only supported for preprocessed "
                f"single-view inputs, got {num_views}."
            )
        return num_views

    @staticmethod
    def _get_multiview_vae_metadata(
        data_batch: dict[str, Any],
        num_vision_items_per_sample: list[int] | None,
        batch_size: int,
    ) -> tuple[list[int] | None, list[int] | None]:
        """Align per-sample multiview metadata with flattened vision items."""
        if "enable_per_camera_vae_encoding" not in data_batch:
            return None, None

        sample_n_views = read_positive_int_metadata(data_batch, "sample_n_views", expected_count=batch_size)
        sample_frames_per_view = read_positive_int_metadata(
            data_batch,
            "num_video_frames_per_view",
            expected_count=batch_size,
        )
        if sample_n_views is None or sample_frames_per_view is None:
            raise ValueError(
                "sample_n_views and num_video_frames_per_view are required when per-camera VAE encoding is enabled."
            )

        vision_item_counts = (
            num_vision_items_per_sample if num_vision_items_per_sample is not None else [1] * batch_size
        )
        if len(vision_item_counts) != batch_size:
            raise ValueError(
                "num_vision_items_per_sample must align with the packed batch: "
                f"got {len(vision_item_counts)} counts for batch size {batch_size}."
            )

        num_views_per_vision_item: list[int] = []
        frames_per_vision_item: list[int] = []
        for sample_idx, num_vision_items in enumerate(vision_item_counts):
            if num_vision_items <= 0:
                raise ValueError(f"Vision item counts must be positive, got {vision_item_counts}.")
            num_views_per_vision_item.extend([sample_n_views[sample_idx]] * num_vision_items)
            frames_per_vision_item.extend([sample_frames_per_view[sample_idx]] * num_vision_items)
        return num_views_per_vision_item, frames_per_vision_item

    def _require_lidar_tokenizer(self) -> VideoTokenizerInterface:
        """Return the LiDAR VAE, or say which config knob is missing."""
        if self.tokenizer_lidar_gen is None:
            raise ValueError(
                "This batch carries a LiDAR stream, but no LiDAR tokenizer is loaded. "
                "Set lidar_tokenizer and lidar_state_ch on the model config."
            )
        return self.tokenizer_lidar_gen

    def _normalize_uint8_vision_item(self, state: torch.Tensor) -> torch.Tensor:
        """Convert one GPU-resident uint8 vision item to fp32 and normalize to ``[-1,1]``."""
        return normalize_uint8_item(state, self.tensor_kwargs_fp32)

    def _encode_vision_item(
        self,
        state: torch.Tensor,
        *,
        num_views: int,
        frames_per_view: int | None,
    ) -> torch.Tensor:  # state: [B,C,T,H,W] or [C,T,H,W], returns [...,C_latent,T_latent,H_latent,W_latent]
        """Encode one vision item, normalizing camera-major views independently when requested.

        When ``frames_per_view`` is present, ``state`` contains ``num_views``
        complete clips concatenated along T. uint8 camera clips are converted to
        fp32 and normalized per view so the full multiview tensor is never
        materialized. Floating-point LiDAR clips (V0 range+intensity or V1 metric
        three-channel) are already tokenizer-native and are encoded as-is.

        Kept on the model rather than moved onto :class:`VisionEncoder` because it is a
        monkeypatch seam: ``posttrain``'s seeded-validation context manager replaces this
        attribute to give each vision item a deterministic RNG seed, so every unbalanced
        encode has to keep flowing through it.
        """
        if frames_per_view is None:
            if num_views != 1:
                raise ValueError("frames_per_view is required when num_views is greater than one.")
            return self.encode(state).contiguous().float()  # [...,C_latent,T_latent,H_latent,W_latent]

        temporal_dim = state.ndim - 3
        validate_multiview_length(state, num_views=num_views, frames_per_view=frames_per_view)

        encoded_views: list[torch.Tensor] = []
        for view_idx in range(num_views):
            view_state = state.narrow(  # [...,C,T_v,H,W]
                temporal_dim,
                view_idx * frames_per_view,
                frames_per_view,
            )
            # Camera pixels arrive as uint8 levels and need the [-1,1] map. A LiDAR
            # clip arrives float in the units its own tokenizer normalizes from, so
            # it is already tokenizer-native and is encoded as-is.
            encode_input = (
                view_state if torch.is_floating_point(view_state) else self._normalize_uint8_vision_item(view_state)
            )  # [...,C,T_v,H,W]
            encoded_view = self.encode(encode_input).contiguous().float()  # [...,C_latent,T_latent_v,H_latent,W_latent]
            # The VAE is frozen/no-grad, so its output does not retain the
            # encode input. Release this reference before the next view.
            del encode_input
            encoded_views.append(encoded_view)

        # Do camera-major repacking for now (instead of timestamp-major).
        return torch.cat(encoded_views, dim=temporal_dim)  # [...,C_latent,V*T_latent_v,H_latent,W_latent]

    def _vision_encoder(self) -> VisionEncoder:
        """Build the encoder for the current tokenizer, mesh and encode entry point.

        Built per use rather than cached in ``__init__``: ``tokenizer_vision_gen`` and
        ``parallel_dims`` are both assigned after construction, and ``IterSpeed`` installs a
        ``MethodTimer`` over ``self.encode`` mid-run. Re-reading them here keeps the encoder
        from pinning a stale collaborator, and costs nothing next to a VAE encode.
        """
        return VisionEncoder(
            tokenizer=self.tokenizer_vision_gen,
            parallel_dims=self.parallel_dims,
            encode_fn=lambda state: self.encode(state),
            fp32_kwargs=self.tensor_kwargs_fp32,
        )

    def _encode_vision_x0_tokens(
        self,
        raw_state_vision: list[torch.Tensor],
        num_vision_items_per_sample: list[int] | None,
        vision_condition_indexes: list[list[int]] | None,
        num_views_per_vision_item: list[int] | None = None,
        frames_per_vision_item: list[int] | None = None,
        balance_vae_encode: bool = False,
    ) -> list[torch.Tensor]:
        """Encode vision items into x0 latent tokens, optionally splitting camera views.

        Default behavior (``vision_condition_indexes is None``) encodes every pixel
        frame of every vision item. This is the path used during training and by any
        caller that does not opt in. Multiview items are camera-major along the raw
        temporal axis; each camera is normalized and encoded independently before
        its latent sequence is concatenated back in the same camera-major order.

        Inference optimization: when ``vision_condition_indexes`` is provided (one
        conditioning-frame-index list per sample, from each ``SequencePlan``) and the
        vision tokenizer is temporally causal, only the pixel-frame prefix required to
        reconstruct the highest conditioned latent frame is encoded. The remaining
        latent frames are generated (pure-noise) positions that the ``condition_mask``
        blend discards, so they are zero-filled instead of encoded. Under a causal VAE
        the kept latents are identical to the corresponding frames of a full-clip
        encode, while the encode processes far fewer frames (e.g. 1 pixel frame instead
        of the full clip for WAM / forward-dynamics modes that only condition latent
        frame 0). Inverse-dynamics conditions every latent frame, so it keeps the full
        encode automatically.

        The optimization falls back to a full encode for multi-vision samples,
        multiview items, non-causal tokenizers, samples with no conditioning frames,
        and any item whose full clip is already the minimal prefix.

        ``balance_vae_encode`` spreads the full-encode path's work across the ``lb`` group
        (see :meth:`VisionEncoder.encode_balanced`). It is ignored on the prefix-encode path above,
        which is inference-only, and only training opts in.
        """

        has_multiview_metadata = num_views_per_vision_item is not None
        if has_multiview_metadata != (frames_per_vision_item is not None):
            raise ValueError("num_views_per_vision_item and frames_per_vision_item must be provided together.")
        if num_views_per_vision_item is None:
            num_views_per_vision_item = [1] * len(raw_state_vision)
            frames_per_vision_item_optional: list[int | None] = [None] * len(raw_state_vision)
        else:
            assert frames_per_vision_item is not None
            if len(num_views_per_vision_item) != len(raw_state_vision) or len(frames_per_vision_item) != len(
                raw_state_vision
            ):
                raise ValueError(
                    "Multiview VAE metadata must align with flattened vision items: "
                    f"got {len(num_views_per_vision_item)} view counts, {len(frames_per_vision_item)} frame "
                    f"counts, and {len(raw_state_vision)} vision items."
                )
            frames_per_vision_item_optional = list(frames_per_vision_item)

        # Only opt in when a caller supplied per-sample conditioning indexes, the
        # samples map 1:1 to single-view vision items (no multi-vision flattening),
        # and the tokenizer is causal (so a pixel prefix reproduces the leading latents).
        optimization_applicable = (
            vision_condition_indexes is not None
            and num_vision_items_per_sample is None
            and not has_multiview_metadata
            and all(num_views == 1 for num_views in num_views_per_vision_item)
            and self.tokenizer_vision_gen is not None
            and self.tokenizer_vision_gen.is_causal
            and len(vision_condition_indexes) == len(raw_state_vision)
        )
        if not optimization_applicable:
            # Training does not provide vision_condition_indexes, so it fully
            # encodes each item in the flattened control/target list.
            if balance_vae_encode:
                # Built only here: callers that never balance (inference, the visualization
                # callbacks) should not need a tokenizer or an lb mesh to encode.
                encoder = self._vision_encoder()
                if encoder.balancing_available():
                    return encoder.encode_balanced(
                        raw_state_vision,
                        num_views_per_vision_item,
                        frames_per_vision_item_optional,
                    )
            return [
                self._encode_vision_item(state, num_views=num_views, frames_per_view=frames_per_view)
                for state, num_views, frames_per_view in zip(
                    raw_state_vision,
                    num_views_per_vision_item,
                    frames_per_vision_item_optional,
                    strict=True,
                )
            ]

        # The prefix-encode optimization zero-fills the generated (pure-noise) latent
        # frames that the condition_mask blend later discards. That is only valid when
        # no gradient is required: torch.is_grad_enabled() is False under both
        # torch.inference_mode() and torch.no_grad(), so this asserts we are on an
        # inference path and never silently corrupts a training forward.
        if torch.is_grad_enabled():
            raise ValueError(
                "prefix-encode optimization is inference-only, but grad is enabled "
                "(expected torch.inference_mode()/torch.no_grad())"
            )

        tokenizer = self.tokenizer_vision_gen
        assert vision_condition_indexes is not None  # narrowed by optimization_applicable above
        x0_tokens_vision: list[torch.Tensor] = []
        for raw_state_vision_i, condition_indexes in zip(
            raw_state_vision,
            vision_condition_indexes,
            strict=True,
        ):
            # The temporal axis is the first of the trailing (T, H, W) dims for both the
            # [B, C, T, H, W] and [C, T, H, W] layouts the tokenizer accepts, and encode
            # preserves rank so the same index locates T in the latent output.
            temporal_dim = raw_state_vision_i.ndim - 3
            num_pixel_frames = int(raw_state_vision_i.shape[temporal_dim])
            num_latent_frames = tokenizer.get_latent_num_frames(num_pixel_frames)

            valid_condition = [idx for idx in condition_indexes if 0 <= idx < num_latent_frames]
            if not valid_condition:
                # No conditioning frames to preserve (e.g. fully generated T2V item):
                # keep the full encode so behavior is unchanged for non-action items.
                x0_tokens_vision.append(self._encode_vision_item(raw_state_vision_i, num_views=1, frames_per_view=None))
                continue

            needed_latent_frames = max(valid_condition) + 1
            needed_pixel_frames = tokenizer.get_pixel_num_frames(needed_latent_frames)
            if needed_pixel_frames <= 0 or tokenizer.get_latent_num_frames(needed_pixel_frames) != needed_latent_frames:
                # Some causal tokenizers prepend enough checkpoint-native
                # context that no positive pixel prefix produces exactly the
                # requested leading latent count. Preserve correctness by
                # encoding the full item; the condition mask still keeps only
                # the declared clean latent indexes. In particular, DCAE maps
                # one observed image to two latent slots, but slot 1 is causal
                # padding that remains generated rather than a second clean
                # observation, so its condition indexes intentionally stay [0].
                x0_tokens_vision.append(self._encode_vision_item(raw_state_vision_i, num_views=1, frames_per_view=None))
                continue
            assert needed_pixel_frames <= num_pixel_frames, (
                f"needed_pixel_frames ({needed_pixel_frames}) cannot be greater than "
                f"num_pixel_frames ({num_pixel_frames})"
            )
            if needed_pixel_frames == num_pixel_frames:
                # The conditioning already spans (nearly) the whole clip; no work saved.
                x0_tokens_vision.append(self._encode_vision_item(raw_state_vision_i, num_views=1, frames_per_view=None))
                continue

            prefix = raw_state_vision_i.narrow(temporal_dim, 0, needed_pixel_frames)  # [...,T_prefix,H,W]
            prefix_latent = self._encode_vision_item(  # [...,T_latent_prefix,H_latent,W_latent]
                prefix,
                num_views=1,
                frames_per_view=None,
            )

            # The scatter below assumes the pixel->latent round trip is exact, i.e. that
            # encoding ``needed_pixel_frames`` yields exactly ``needed_latent_frames``
            # latents. All current causal tokenizers satisfy this, but guard it: an
            # overshoot would make the narrow below request more frames than full_latent
            # holds (confusing runtime error), and an undershoot would silently zero-fill
            # the highest conditioning frame and corrupt the condition_mask blend with no
            # error at all.
            num_prefix_latent_frames = prefix_latent.shape[temporal_dim]
            assert num_prefix_latent_frames == needed_latent_frames, (
                f"causal tokenizer pixel<->latent round trip is not exact: encoding "
                f"{needed_pixel_frames} pixel frames (from get_pixel_num_frames({needed_latent_frames})) "
                f"produced {num_prefix_latent_frames} latent frames, expected {needed_latent_frames}"
            )

            # Scatter the encoded prefix into a full-length latent; the unconditioned
            # tail frames stay zero since the condition_mask blend replaces them with noise.
            full_shape = list(prefix_latent.shape)
            full_shape[temporal_dim] = num_latent_frames
            full_latent = prefix_latent.new_zeros(full_shape)
            full_latent.narrow(temporal_dim, 0, prefix_latent.shape[temporal_dim]).copy_(prefix_latent)
            x0_tokens_vision.append(full_latent)

        return x0_tokens_vision

    def _encode_sound_x0_tokens(
        self,
        raw_state_sound: list[torch.Tensor],
        sequence_plans: list[SequencePlan],
    ) -> list[torch.Tensor]:
        """Encode sound targets, re-encoding VS2VS prefixes from prefix samples only.

        The non-causal tokenizer would leak future context into conditioned prefix latents, so
        a contiguous proper prefix is re-encoded alone (matching inference); empty and full
        prefixes keep the single encode. ``raw_state_sound`` is the dense list from
        ``_normalize_sound_databatch_inplace``; the strict zip enforces its 1:1 alignment with
        the ``has_sound`` plans that the packer's ``idx_sound`` counter relies on.
        """
        x0_tokens_sound = [
            self.encode_sound(sound).contiguous().float() for sound in raw_state_sound
        ]  # list of [sound_channels,T_sound]
        sound_plans = [plan for plan in sequence_plans if plan.has_sound]
        compression_factor = int(self.tokenizer_sound_gen.temporal_compression_factor)
        for sound, x0_sound, plan in zip(raw_state_sound, x0_tokens_sound, sound_plans, strict=True):
            condition_indexes = plan.condition_frame_indexes_sound
            num_condition_frames = len(condition_indexes)
            is_proper_prefix = 0 < num_condition_frames < int(x0_sound.shape[1]) and condition_indexes == list(
                range(num_condition_frames)
            )
            if not is_proper_prefix:
                continue
            prefix_num_samples = num_condition_frames * compression_factor
            prefix_tokens = (
                self.encode_sound(sound[..., :prefix_num_samples]).contiguous().float()
            )  # [sound_channels,T_prefix]
            x0_sound[:, :num_condition_frames].copy_(
                prefix_tokens[:, :num_condition_frames]
            )  # [sound_channels,num_condition_frames]

        return x0_tokens_sound

    def get_data_and_condition(
        self,
        data_batch: dict[str, torch.Tensor],
        vision_condition_indexes: list[list[int]] | None = None,
        retain_raw_state_vision: bool = True,
        balance_vae_encode: bool = False,
    ) -> GenerationDataClean:
        """
        - Get raw data of different modalities from databatch
        - Tokenize into corresponding latents
        - Load other conditioning information if any (fps, etc.)

        Args:
            data_batch: Raw data batch from the dataloader.
            vision_condition_indexes: Optional per-sample list of conditioning latent
                frame indexes (one list per sample, taken from each ``SequencePlan``).
                When provided (inference only), it enables the causal-VAE prefix-encode
                optimization in ``_encode_vision_x0_tokens`` that skips encoding pixel
                frames which only feed generated (non-conditioned) latent positions.
                Leaving it ``None`` (the default, used during training) encodes every
                frame — the original behavior.
            retain_raw_state_vision: Preserve the public normalized
                ``[B,C,V*T,H,W]`` raw-state contract for multiview sampling and
                inference. Training disables this to avoid materializing complete
                normalized multiview inputs.
            balance_vae_encode: Spread this step's VAE-encode compute across the ``lb``
                group instead of encoding exactly this rank's own pixels (see
                ``VisionEncoder.encode_balanced``). Produces identical latents either way; only
                where each encode runs changes. Off by default because it runs collectives
                over the whole group: only training, which every rank enters in lockstep,
                may turn it on.
        """
        if not self._has_vision_stream(data_batch):
            return self._get_lidar_only_data_and_condition(data_batch)

        # Detect whether any sample has multiple vision items (e.g. image editing).
        # If so, track the count per sample before all vision items from this batch are flattened into a list.
        is_image_batch = self.is_image_batch(data_batch)
        sample_vision_list = data_batch[self.input_image_key if is_image_batch else self.input_video_key]

        # we should always get this information here during training. If we can read this field
        # from data_batch it means we are in the visualization callback:
        if "num_vision_items_per_sample" not in data_batch:
            # Each element must be a list/tuple of tensors (not a bare tensor) to count
            # as multi-vision.  A bare tensor's len() returns its first dim size (e.g. C=3),
            # which would incorrectly trigger the multi-vision path for regular video batches.
            has_multiple_vision_per_sample = any(
                isinstance(v, (list, tuple)) and len(v) > 1 for v in sample_vision_list
            )
            num_vision_items_per_sample: list[int] | None = (
                [len(v) for v in sample_vision_list] if has_multiple_vision_per_sample else None
            )
            # information is only stored in the GenerationDataClean object which will be discarded
            # outside the training loop. Error will be raised when the data batch is passed to the
            # visualization callbacks.
            data_batch["num_vision_items_per_sample"] = num_vision_items_per_sample

            # if has_multiple_vision_per_sample, this means that the input media is a list
            # of lists of tensors, we need to flatten it to a list of tensors
            if has_multiple_vision_per_sample:
                media_key = self.input_video_key if not is_image_batch else self.input_image_key
                data_batch[media_key] = [item.unsqueeze(0) for sublist in sample_vision_list for item in sublist]
                if data_batch[media_key][0].dtype == torch.float32 and not is_image_batch:
                    # For video batch, is_preprocessed = True means the video data is normalized.
                    # For the image batch, is_preprocessed = True means the image data is
                    # normalized and augmented with a temporal dimension.
                    data_batch["is_preprocessed"] = True
        else:
            num_vision_items_per_sample = data_batch["num_vision_items_per_sample"]

        batch_size = (
            len(sample_vision_list) if num_vision_items_per_sample is None else len(num_vision_items_per_sample)
        )

        num_views_per_vision_item, frames_per_vision_item = self._get_multiview_vae_metadata(
            data_batch,
            num_vision_items_per_sample,
            batch_size,
        )

        # Vision (image/video) raw state and tokenized latent state.
        media_key = self.input_image_key if is_image_batch else self.input_video_key
        if num_views_per_vision_item is None:
            # Legacy VFM/image path: normalize the complete input when needed and
            # preserve the existing image batch-dimension handling.
            self._normalize_video_databatch_inplace(data_batch)
            self._augment_image_dim_inplace(data_batch)  # converts each image tensor to [1,C,1,H,W]
            raw_state_vision = data_batch[media_key]

        else:
            # Per-camera multiview path: preserve camera-major uint8 pixels here;
            # _encode_vision_item normalizes only one camera at a time.
            raw_state_vision = []
            for item in data_batch[media_key]:
                if isinstance(item, (list, tuple)):
                    if len(item) != 1:
                        raise ValueError(
                            f"Single-item multiview samples must contain exactly one vision tensor, got {len(item)}."
                        )
                    item = item[0]
                if not isinstance(item, torch.Tensor):
                    raise TypeError(f"Multiview vision items must be tensors, got {type(item).__name__}.")
                if item.dim() == 4:  # Unbatched camera-major pixels: [C,V*T,H,W]
                    item = item.unsqueeze(0)  # [1,C,V*T,H,W]
                elif item.dim() != 5:  # Batched camera-major pixels: [B,C,V*T,H,W]
                    raise ValueError(
                        f"Multiview vision items must have shape [C,V*T,H,W] or [B,C,V*T,H,W], got {tuple(item.shape)}."
                    )
                raw_state_vision.append(item)

        x0_tokens_vision = self._encode_vision_x0_tokens(
            raw_state_vision,
            num_vision_items_per_sample,
            vision_condition_indexes,
            num_views_per_vision_item,
            frames_per_vision_item,
            balance_vae_encode=balance_vae_encode,
        )

        frame_size = data_batch.get("image_size", None)
        if frame_size is not None:
            x0_tokens_vision = self._remove_padding_from_latent(x0_tokens_vision, frame_size)

        temporal_positions_vision = self._get_temporal_positions_vision(
            raw_state_vision=raw_state_vision,
            x0_tokens_vision=x0_tokens_vision,
            num_views_per_vision_item=num_views_per_vision_item,
            frames_per_vision_item=frames_per_vision_item,
        )

        # LiDAR range clips: their own VAE, so they never travel among the vision items.
        raw_state_lidar, x0_tokens_lidar, num_lidar_items_per_sample = self._encode_lidar_stream(data_batch, batch_size)

        output_raw_state_vision = raw_state_vision
        if retain_raw_state_vision and num_views_per_vision_item is not None:
            # Camera pixels arrive as uint8 levels and need the [-1,1] map. An item that is
            # already floating point has been normalized by whoever produced it, and running
            # the level map over it a second time would misread its units.
            output_raw_state_vision = [
                state if torch.is_floating_point(state) else self._normalize_uint8_vision_item(state)
                for state in raw_state_vision
            ]  # list[[B,C,V*T,H,W]]

        # Action – extract dense action / domain_id without mutating data_batch,
        # so downstream callbacks can still read the original per-sample domain_ids.
        raw_state_action, action_domain_id, action_family = self._normalize_action_databatch(data_batch)
        x0_tokens_action = raw_state_action
        raw_action_dim = data_batch.get("raw_action_dim", None)
        action_valid_mask = data_batch.get("action_valid_mask", None)

        # Sound/audio - normalize, encode if present and sound_gen is enabled
        raw_state_sound = self._normalize_sound_databatch_inplace(data_batch)
        if raw_state_sound is not None and self.tokenizer_sound_gen is not None:
            # Sound batches always carry plans (``build_sequence_plans_from_data_batch`` asserts it
            # before every training/inference call), so a missing key fails loudly here rather than
            # silently skipping the VS2VS prefix re-encode.
            x0_tokens_sound = self._encode_sound_x0_tokens(raw_state_sound, data_batch["sequence_plan"])
        else:
            x0_tokens_sound = None

        # FPS metadata is used by the sequence packer for mRoPE temporal IDs when
        # FPS modulation is enabled in the training config.
        fps_raw = data_batch.get("conditioning_fps", None)
        if isinstance(fps_raw, list):
            fps_raw = torch.stack(fps_raw).flatten()  # list of scalar tensors -> (B,)
        fps_vision = fps_raw.to(**self.tensor_kwargs) if fps_raw is not None else None
        fps_action = fps_raw.to(**self.tensor_kwargs) if fps_raw is not None else None
        if "conditioning_fps_action" in data_batch:
            fps_action_raw = data_batch["conditioning_fps_action"]
            if isinstance(fps_action_raw, list):
                # serve_policy.py wraps every tensor in a list; align with the conditioning_fps
                # branch above so inference paths don't hit AttributeError on .to().
                fps_action_raw = torch.stack(fps_action_raw).flatten()
            fps_action = fps_action_raw.to(**self.tensor_kwargs)

        # LiDAR sweep rate for mRoPE, a property of the recording rather than of the clip, so
        # it comes from the config the way the LiDAR VAE's compression does.
        fps_lidar = None
        if x0_tokens_lidar is not None:
            fps_lidar = torch.full((batch_size,), float(self.config.lidar_fps), dtype=torch.float32).to(
                **self.tensor_kwargs
            )

        # Sound FPS for RoPE alignment (constant, from config)
        if x0_tokens_sound is not None:
            sound_batch_size = len(x0_tokens_sound)
            fps_sound = torch.full(
                (sound_batch_size,),
                self._get_sound_fps_for_rope(),
                dtype=torch.float32,
            ).to(**self.tensor_kwargs)
        else:
            fps_sound = None

        control_weights: list[list[float]] | None = data_batch.get("control_weights", None)
        return GenerationDataClean(
            batch_size=batch_size,
            is_image_batch=is_image_batch,
            raw_state_vision=output_raw_state_vision,
            raw_state_action=raw_state_action,
            raw_state_sound=raw_state_sound,
            x0_tokens_vision=x0_tokens_vision,
            x0_tokens_action=x0_tokens_action,
            x0_tokens_sound=x0_tokens_sound,
            fps_vision=fps_vision,
            temporal_positions_vision=temporal_positions_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
            action_domain_id=action_domain_id,
            action_family=action_family,
            num_vision_items_per_sample=num_vision_items_per_sample,
            num_views_per_vision_item=num_views_per_vision_item,
            raw_state_lidar=raw_state_lidar,
            x0_tokens_lidar=x0_tokens_lidar,
            fps_lidar=fps_lidar,
            num_lidar_items_per_sample=num_lidar_items_per_sample,
            raw_action_dim=raw_action_dim,
            action_valid_mask=action_valid_mask,
            control_weights=control_weights,
        )

    def _has_vision_stream(self, data_batch: dict[str, Any]) -> bool:
        """Whether the batch carries a camera stream to tokenize.

        The LiDAR-only AV recipe decodes no camera at all, so it arrives with neither the
        image nor the video key and its sole generated stream is ``lidar``.
        """
        return self.input_image_key in data_batch or self.input_video_key in data_batch

    def _get_lidar_only_data_and_condition(self, data_batch: dict[str, Any]) -> GenerationDataClean:
        """Build the clean payload for a batch whose only generated stream is LiDAR.

        ``batch_size`` comes from the LiDAR stream because there are no vision items to count.
        The vision fields are empty rather than absent, and every sample is recorded as owning
        zero vision items, so the packer, the noise schedule and the loss walk the vision
        stream the same way they walk any other batch -- over nothing, in this case.
        """
        lidar_entries = data_batch.get("lidar")
        if lidar_entries is None:
            raise ValueError(
                "A batch with no vision stream must carry a LiDAR stream: found neither "
                f"{self.input_video_key!r}/{self.input_image_key!r} nor 'lidar'."
            )
        num_lidar_items_per_sample = data_batch.get("num_lidar_items_per_sample", None)
        batch_size = len(num_lidar_items_per_sample) if num_lidar_items_per_sample is not None else len(lidar_entries)

        raw_state_lidar, x0_tokens_lidar, num_lidar_items = self._encode_lidar_stream(data_batch, batch_size)
        # The sweep rate is a property of the recording rather than of the clip, so it comes
        # from the config the way the LiDAR VAE's compression does.
        fps_lidar = torch.full((batch_size,), float(self.config.lidar_fps), dtype=torch.float32).to(
            **self.tensor_kwargs
        )
        return GenerationDataClean(
            batch_size=batch_size,
            is_image_batch=False,
            raw_state_vision=[],
            x0_tokens_vision=[],
            num_vision_items_per_sample=[0] * batch_size,
            raw_state_lidar=raw_state_lidar,
            x0_tokens_lidar=x0_tokens_lidar,
            fps_lidar=fps_lidar,
            num_lidar_items_per_sample=num_lidar_items,
            control_weights=data_batch.get("control_weights", None),
        )

    def _encode_lidar_stream(
        self,
        data_batch: dict[str, Any],
        batch_size: int,
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None, list[int] | None]:
        """Encode the batch's LiDAR range clips into x0 latent tokens.

        ``data_batch["lidar"]`` arrives from the loader as one entry per sample — a list of
        tokenizer-native range clips — and is flattened here the way the vision key is. Both
        tokenizer versions receive the same layout, metric range with unit intensity and a
        validity mask, and normalize it inside encode; only the azimuth canvas differs. The
        tokenizer owns that layout. The per-sample counts
        are written back to the batch, both so a second pass over the same batch sees the
        flat form and so the visualization callbacks can regroup the items once the
        training payload is gone.

        Returns:
            The raw range clips, their latents, and the per-sample item counts; all
            ``None`` when the batch carries no LiDAR.
        """
        raw = data_batch.get("lidar", None)
        if raw is None:
            return None, None, None

        num_lidar_items_per_sample = data_batch.get("num_lidar_items_per_sample", None)
        if num_lidar_items_per_sample is None:
            # Per-sample form: group the items and record the counts.
            if len(raw) != batch_size:
                raise ValueError(f"The LiDAR stream needs one entry per sample: got {len(raw)} for {batch_size}.")
            grouped = [list(entry) if isinstance(entry, (list, tuple)) else [entry] for entry in raw]
            num_lidar_items_per_sample = [len(items) for items in grouped]
            flat_items = [item for items in grouped for item in items]
        else:
            # Flat form: already grouped on an earlier pass over this batch.
            num_lidar_items_per_sample = [int(count) for count in num_lidar_items_per_sample]
            flat_items = list(raw)
            if len(num_lidar_items_per_sample) != batch_size:
                raise ValueError(
                    "num_lidar_items_per_sample must have one count per sample: "
                    f"got {len(num_lidar_items_per_sample)} counts for batch size {batch_size}."
                )
            if len(flat_items) != sum(num_lidar_items_per_sample):
                raise ValueError(
                    "LiDAR items must match num_lidar_items_per_sample: "
                    f"got {len(flat_items)} items for {sum(num_lidar_items_per_sample)} expected."
                )

        raw_state_lidar: list[torch.Tensor] = []
        for item in flat_items:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"LiDAR items must be tensors, got {type(item).__name__}.")
            if item.dim() == 4:  # [C,T,H,W]
                item = item.unsqueeze(0)  # [1,C,T,H,W]
            elif item.dim() != 5:
                raise ValueError(f"LiDAR items must have shape [C,T,H,W] or [B,C,T,H,W], got {tuple(item.shape)}.")
            if not torch.is_floating_point(item):
                raise TypeError(f"LiDAR range maps must arrive as floating-point clips, got {item.dtype}.")
            raw_state_lidar.append(item.to(**self.tensor_kwargs_fp32))  # [1,C,T,H,W]

        data_batch["lidar"] = raw_state_lidar
        data_batch["num_lidar_items_per_sample"] = num_lidar_items_per_sample

        x0_tokens_lidar = [self.encode_lidar(state).contiguous().float() for state in raw_state_lidar]
        return raw_state_lidar, x0_tokens_lidar, num_lidar_items_per_sample

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, torch.Tensor]) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the tensor at `input_key` within `data_batch` in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        IS_PREPROCESSED_KEY = "is_preprocessed"
        input_key = self.input_video_key
        # only handle video batch
        if input_key in data_batch:
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                for i in range(len(data_batch[input_key])):
                    assert torch.is_floating_point(data_batch[input_key][i]), "Video data is not in float format."
                    assert torch.all((data_batch[input_key][i] >= -1.0001) & (data_batch[input_key][i] <= 1.0001)), (
                        f"Video data is not in the range [-1, 1]. get data range "
                        f"[{data_batch[input_key][i].min()}, {data_batch[input_key][i].max()}]"
                    )
            else:
                # Worker-normalized batches take the preprocessed branch above.
                for i in range(len(data_batch[input_key])):
                    item = data_batch[input_key][i]
                    if isinstance(item, torch.Tensor):
                        assert item.dtype == torch.uint8, "Video data is not in uint8 format."
                        # Flattened multiview items already include B; preserve the original stacking behavior otherwise.
                        if item.ndim != 5:
                            item = torch.stack([item])  # [1,C,T,H,W]
                        normalized_item = item.to(**self.tensor_kwargs_fp32) / 127.5 - 1.0  # [B,C,T,H,W]
                    else:
                        # Non-flattened batches may still provide one [C,T,H,W] tensor per sample.
                        assert item[0].dtype == torch.uint8, "Video data is not in uint8 format."
                        normalized_item = (  # [B,C,T,H,W]
                            torch.stack(item).to(**self.tensor_kwargs_fp32) / 127.5 - 1.0
                        )
                    data_batch[input_key][i] = normalized_item  # [B,C,T,H,W]
                data_batch[IS_PREPROCESSED_KEY] = True

    def _normalize_action_databatch(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None, list[str] | None]:
        """Extract aligned action, domain_id, and dataset-family lists from the data batch.

        The joint dataloader produces action and domain_id data as
        ``[[tensor], [None], [tensor], ...]`` (each sample wrapped in a
        single-element list).  This method unwraps inner lists and filters
        out ``None`` entries to produce dense lists suitable for the model,
        **without mutating** ``data_batch``.

        Returns:
            Dense action and domain-ID tensors plus dataset names aligned with
            the non-None action rows. Missing or malformed telemetry metadata
            produces ``None`` for the dataset names without affecting training.
        """
        # Keep normalized actions in FP32 through flow interpolation so adding noise and computing
        # the velocity target do not inherit BF16 quantization. The noisy encoder input is cast to
        # model precision at the action encoder boundary in ``_encode_action``.
        dense_action = unwrap_and_densify(data_batch.get("action", None), self.tensor_kwargs_fp32)
        dense_domain_id = unwrap_and_densify(
            data_batch.get("domain_id", None), {"device": self.tensor_kwargs["device"]}
        )
        dense_action_family = (
            _densify_action_family(data_batch.get("action"), data_batch.get("_action_family"), len(dense_action))
            if dense_action is not None
            else None
        )
        return dense_action, dense_domain_id, dense_action_family

    def _normalize_sound_databatch_inplace(self, data_batch: dict[str, torch.Tensor]) -> list[torch.Tensor] | None:
        """Normalize the plans' sound flags in-place and return the dense sound list.

        The joint dataloader produces sound data as
        ``[[tensor], [None], [tensor], ...]`` (each sample wrapped in a single-element
        list).  This method:

        1. Unwraps inner lists: ``[[t], [None], [t]]`` -> ``[t, None, t]``
        2. Clears ``sequence_plan.has_sound`` for samples whose sound is ``None``
           (kept aligned by ``custom_collate_fn`` preserving ``None`` placeholders).
        3. Filters out None entries: ``[t, None, t]`` -> ``[t, t]``
        4. Moves tensors to the model device.
        5. Returns the dense list, or ``None`` (also stored in ``data_batch["sound"]``)
           if no valid sound data remains.

        Alignment invariant: ``custom_collate_fn`` keeps the ``"sound"`` key
        as a list with ``None`` placeholders for samples that lack audio (e.g.
        audio-extraction failures), so the unwrapped ``raw_state_sound`` is
        1:1 with ``sequence_plan``.  ``SoundSequencePlanBuilder`` already sets
        each plan's ``has_sound`` according to that sample's actual sound
        presence, so clearing flags for ``None`` slots here is just defensive.
        The per-sample slots stay in ``data_batch["sound"]``: the online sampling
        callbacks call ``get_data_and_condition`` again on the trainer's batch (and
        on prefix slices of it), which must see the same alignment.
        """
        raw_state_sound = data_batch.get("sound", None)
        sequence_plans = data_batch.get("sequence_plan", None)
        sound_enabled = self.tokenizer_sound_gen is not None

        def _disable_sound_on_plans() -> None:
            if isinstance(sequence_plans, list):
                for plan in sequence_plans:
                    if hasattr(plan, "has_sound"):
                        plan.has_sound = False
                        plan.condition_frame_indexes_sound = []

        if not isinstance(raw_state_sound, list) or len(raw_state_sound) == 0:
            # No sound entries at all (image-only batches, or every sample
            # came from a non-audio stream).  Defensively clear has_sound on
            # any plan that somehow has it set so packing does not look up
            # missing tensors.
            _disable_sound_on_plans()
            data_batch["sound"] = None
            return

        # Unwrap single-element inner lists produced by IterativeJointDataLoader
        if isinstance(raw_state_sound[0], list):
            raw_state_sound = [item[0] if isinstance(item, list) else item for item in raw_state_sound]

        if not sound_enabled:
            # Model is not configured for sound generation: drop tensors and
            # clear any has_sound flags so packing skips the sound path.
            _disable_sound_on_plans()
            data_batch["sound"] = None
            return

        if isinstance(sequence_plans, list):
            if len(sequence_plans) == len(raw_state_sound):
                # Expected path: 1:1 alignment between plans and per-sample
                # sound slots.  Clear has_sound where the per-sample tensor
                # is None so sequence_packing's idx_sound counter stays in
                # sync with the filtered dense list.
                for plan, sound in zip(sequence_plans, raw_state_sound, strict=True):
                    if hasattr(plan, "has_sound") and sound is None:
                        plan.has_sound = False
                        plan.condition_frame_indexes_sound = []
            else:
                # Length mismatch can only happen if some upstream code path
                # (e.g. a stale collate that drops "sound" when any sample is
                # None) leaves the dense list shorter than the plans.  Without
                # 1:1 alignment we cannot safely associate tensors with plans,
                # so we conservatively disable sound for the whole batch.
                # This trades a small amount of training signal for guaranteed
                # correctness — better than silently feeding sound from one
                # sample into another sample's plan.
                log.warning(
                    f"Sound/plan length mismatch ({len(sequence_plans)} plans vs "
                    f"{len(raw_state_sound)} sound entries). Disabling sound for "
                    "this batch.  Check that custom_collate_fn preserves the "
                    "'sound' key with None placeholders."
                )
                _disable_sound_on_plans()
                data_batch["sound"] = None
                return

        # Filter out None entries (samples without audio) and move to device.
        # After the alignment step above, the remaining dense list has the
        # same cardinality as plans with has_sound=True.
        raw_state_sound = [
            s.to(self.tensor_kwargs["device"]) for s in raw_state_sound if s is not None
        ]  # list of [C,T_audio]

        if len(raw_state_sound) == 0:
            _disable_sound_on_plans()
            data_batch["sound"] = None
            return None
        return raw_state_sound

    def _augment_image_dim_inplace(self, data_batch: dict[str, torch.Tensor]) -> None:
        """
        Augments image tensors by adding a temporal dimension (B, C, H, W) -> (B, C, 1, H, W).

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the image data.

        Side Effects:
            Modifies the tensor at `input_key` within `data_batch` in-place.
        """
        IS_PREPROCESSED_KEY = "is_preprocessed"

        input_key = self.input_image_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                for i in range(len(data_batch[input_key])):
                    assert data_batch[input_key][i].shape[2] == 1, (
                        f"Image data is claimed be augmented while its shape is {data_batch[input_key][i].shape} for sample {i}"
                    )
                    assert torch.is_floating_point(data_batch[input_key][i]), "Image data is not in float format."
                    assert torch.all((data_batch[input_key][i] >= -1.0001) & (data_batch[input_key][i] <= 1.0001)), (
                        f"Image data is not in the range [-1, 1]. get data range [{data_batch[input_key][i].min()}, {data_batch[input_key][i].max()}]"
                    )
                return
            else:
                new_image_tensor_list = []
                for i in range(len(data_batch[input_key])):
                    for img_tensor in data_batch[input_key][i]:
                        img_tensor = rearrange(img_tensor, "c h w -> 1 c 1 h w").contiguous()
                        if img_tensor.dtype == torch.uint8:
                            img_tensor = img_tensor.to(**self.tensor_kwargs_fp32) / 127.5 - 1.0
                        new_image_tensor_list.append(img_tensor)
                data_batch[input_key] = new_image_tensor_list
                data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------ Checkpointing ------------------

    def state_dict(
        self,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Return checkpointable model weights using OmniMoT's flat key layout.

        The regular network is saved under ``net.*`` keys.  When EMA is
        enabled, the EMA copy is saved under matching ``net_ema.*`` keys so
        the DCP loader can materialize both trees from one flat state dict.
        The optional ``prefix`` is prepended before those namespaces, matching
        the ``torch.nn.Module.state_dict`` convention.

        The full ``torch.nn.Module.state_dict`` signature (``destination``,
        ``prefix``, ``keep_vars``) is honored so this module behaves correctly
        when a parent module's ``state_dict`` recurses into it: PyTorch ignores
        the child return value and expects the entries to be written into the
        provided ``destination`` mapping.

        If ``exclude_reasoner_weights_from_checkpoint`` is enabled, the
        understanding/reasoner tower keys are omitted from both regular and
        EMA state dicts; generation-pathway weights and VFM heads remain
        checkpointed.
        """
        reg_state_dict = self._net_state_dict(
            self.net,
            prefix=prefix + "net.",
            keep_vars=keep_vars,
        )

        if self.config.ema.enabled:
            ema_state_dict = self._net_state_dict(
                self.net_ema,
                prefix=prefix + "net_ema.",
                keep_vars=keep_vars,
            )
        else:
            ema_state_dict = {}

        if destination is not None:
            destination.update(reg_state_dict)
            destination.update(ema_state_dict)
            return destination

        return {**reg_state_dict, **ema_state_dict}

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.

        Parameters:
            state_dict (Mapping[str, Any]): A dictionary containing separate state
                dictionaries for the model and potentially for an EMA version of the model
                under the keys 'net' and 'net_ema', respectively.
            strict (bool, optional): Must be False. Missing and unexpected keys are
                returned to the caller in an `_IncompatibleKeys` object so the DCP
                wrapper can report them after `set_model_state_dict` completes.
                Passing True raises ValueError.
            assign (bool, optional): Must be False. Assign-mode loading is not
                supported by this checkpoint path; passing True raises ValueError.
                Defaults to False.

        Returns:
            _IncompatibleKeys: A tuple containing the missing and unexpected keys.
        """
        # Note that strict must be set to False to avoid facing errors inside the
        # `set_model_state_dict` function in the parent class. The caller must check
        # the returned `_IncompatibleKeys` to get the missing and unexpected keys,
        # and raise errors if needed.
        if strict:
            raise ValueError("Strict mode is not supported for OmniMoTModel load_state_dict")
        if assign:
            raise ValueError("Assign mode is not supported for OmniMoTModel load_state_dict")

        missing_keys: list[str] = []
        unexpected_keys: list[str] = []

        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.removeprefix("net.")] = v
            elif k.startswith("net_ema.") and self.config.ema.enabled:
                _ema_state_dict[k.removeprefix("net_ema.")] = v
            else:
                # If the key is prefixed with "net_ema." but EMA is not enabled, it
                # is unexpected. If the key is not prefixed with "net." or "net_ema.",
                # it is unexpected.
                unexpected_keys.append(k)

        reg_results = self._load_net_state_dict(self.net, _reg_state_dict)
        missing_keys.extend(f"net.{k}" for k in reg_results.missing_keys)
        unexpected_keys.extend(f"net.{k}" for k in reg_results.unexpected_keys)

        if self.config.ema.enabled:
            ema_results = self._load_net_state_dict(self.net_ema, _ema_state_dict)
            missing_keys.extend(f"net_ema.{k}" for k in ema_results.missing_keys)
            unexpected_keys.extend(f"net_ema.{k}" for k in ema_results.unexpected_keys)

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

    def _net_state_dict(
        self,
        net: torch.nn.Module,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        if self.config.exclude_reasoner_weights_from_checkpoint:
            return {
                k: v
                for k, v in net.state_dict(prefix=prefix, keep_vars=keep_vars).items()
                if not _is_reasoner_state_dict_key(k.removeprefix(prefix))
            }
        else:
            return net.state_dict(prefix=prefix, keep_vars=keep_vars)

    def _load_net_state_dict(
        self,
        net: torch.nn.Module,
        state_dict: Mapping[str, Any],
    ) -> _IncompatibleKeys:
        if self.config.exclude_reasoner_weights_from_checkpoint:
            # Leave pretrained reasoner weights untouched even if an incoming
            # checkpoint contains them, and tolerate their absence when they
            # were intentionally not checkpointed.
            state_dict = collections.OrderedDict(
                (k, v) for k, v in state_dict.items() if not _is_reasoner_state_dict_key(k)
            )

        ret: _IncompatibleKeys = net.load_state_dict(state_dict, strict=False, assign=False)

        if self.config.exclude_reasoner_weights_from_checkpoint:
            missing_keys = [k for k in ret.missing_keys if not _is_reasoner_state_dict_key(k)]
        else:
            missing_keys = ret.missing_keys

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=ret.unexpected_keys)

    # ------------------ public methods ------------------

    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, torch.Tensor]) -> bool:
        """Check if the data_batch contains images (vs. videos).

        We handle two types of data_batch: one from a joint_dataloader where "dataset_name" can
        differentiate image_batch and video_batch, another from a single dataloader which we
        assume as video_data by default.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_video_key in data_batch
        if not (is_image or is_video):
            # The LiDAR-only recipe decodes no camera, so it carries neither key and this is not
            # an image batch. Require its dedicated stream here so a corrupted camera batch that
            # lost its media key still fails at the original classification boundary.
            if "lidar" not in data_batch:
                raise ValueError(
                    "A batch must carry exactly one camera stream or the dedicated LiDAR stream; "
                    f"found neither {self.input_image_key!r}, {self.input_video_key!r}, nor 'lidar'."
                )
            return False
        assert is_image != is_video, (
            "Only one of the input_image_key or input_video_key should be present in the data_batch."
        )
        return is_image

    def _extract_upsample_video_specs(
        self, data_batch: dict, task: str
    ) -> tuple[str, int, int, int | None, int | None]:
        """Derive the V4.2 template specs (``aspect_ratio``, ``resolution_w/h``,
        ``fps``, ``duration_secs``) for :meth:`upsample_captions` from a
        ``data_batch``.

        Used by :meth:`generate_samples_from_batch` to feed the per-batch
        V4.2 template constraints (resolution / aspect ratio / fps /
        duration) into the prompt upsampler so the rewritten caption is
        consistent with the actual clip that will be generated.

        The spatial size is read directly off the first sample's vision
        tensor (``shape[-1]`` for width, ``shape[-2]`` for height), and
        the ``aspect_ratio`` string is reverse-looked-up against the
        canonical ``{IMAGE,VIDEO}_RES_SIZE_INFO`` tables in
        :mod:`cosmos_framework.data.generator.utils` — image table for
        ``"t2i"``, video table otherwise.  Note these tables are
        ``{res: {ar: (W, H)}}`` (the first entry is *width*); the
        existing logging-only lookup in
        :meth:`_get_clean_generation_data` uses ``(h, w)`` which only
        matches square aspect ratios — that's a known logging bug, not
        a regression caused by this helper.

        For the video tasks (``"t2v"``, ``"i2v"``) we also pull ``fps``
        from ``data_batch["conditioning_fps"]`` (taking the first
        sample's scalar — batches are assumed fps-uniform for prompt
        upsampling) and derive ``duration_secs = num_frames // fps``
        where ``num_frames`` is the temporal dimension
        (``shape[-3]``) of the same vision tensor.  For ``"t2i"`` both
        fields are returned as ``None`` so
        :func:`cosmos_framework.model.generator.upsampler.prompts.build_user_text`'s
        ``t2i``-must-have-no-video-args contract is satisfied.

        Args:
            data_batch: The raw data batch handed to
                :meth:`generate_samples_from_batch`.
            task: Canonical upsampler task previously resolved by
                the inference caller (``"t2v"``, ``"t2i"``, or
                ``"i2v"``). Selects which key of ``data_batch`` holds
                the spatial reference tensor and which
                ``RES_SIZE_INFO`` table to use.

        Returns:
            ``(aspect_ratio_str, resolution_w, resolution_h, fps, duration_secs)``,
            where the last two are ``None`` iff ``task == "t2i"``.

        Raises:
            ValueError: If the vision tensor for the resolved modality
                key is missing, if its spatial shape doesn't match any
                ``(W, H)`` entry in the canonical resolution table
                (i.e. the batch is at a non-canonical resolution
                /aspect-ratio combination), or if a video task is
                missing ``conditioning_fps``.
        """
        if task == "t2i":
            vision_key = self.input_image_key
            res_info = IMAGE_RES_SIZE_INFO
            res_info_name = "IMAGE_RES_SIZE_INFO"
        else:
            vision_key = self.input_video_key
            res_info = VIDEO_RES_SIZE_INFO
            res_info_name = "VIDEO_RES_SIZE_INFO"

        vision_data = data_batch.get(vision_key)
        if vision_data is None:
            raise ValueError(
                f"upsample task={task!r} requires data_batch[{vision_key!r}] "
                "for resolution / aspect-ratio inference; found None."
            )
        sample = vision_data[0] if isinstance(vision_data, list) else vision_data
        h = int(sample.shape[-2])
        w = int(sample.shape[-1])

        aspect_ratio: str | None = None
        for ar_map in res_info.values():
            for ar_str, (target_w, target_h) in ar_map.items():
                if (target_w, target_h) == (w, h):
                    aspect_ratio = ar_str
                    break
            if aspect_ratio is not None:
                break
        if aspect_ratio is None:
            raise ValueError(
                f"Cannot infer aspect_ratio for upsample task={task!r}: vision tensor has "
                f"(W={w}, H={h}) which is not a canonical entry in {res_info_name}. "
                "Use a supported (resolution, aspect_ratio) combination from "
                "cosmos_framework.data.generator.utils, or pass explicit specs to "
                "`upsample_captions(...)`."
            )

        if task == "t2i":
            return aspect_ratio, w, h, None, None

        fps_raw = data_batch.get("conditioning_fps")
        if fps_raw is None:
            raise ValueError(
                f"upsample task={task!r} requires data_batch['conditioning_fps'] for fps / duration inference."
            )
        if isinstance(fps_raw, list):
            fps_raw = torch.stack(fps_raw)
        fps_int = int(fps_raw.flatten()[0].item())
        if fps_int <= 0:
            raise ValueError(f"upsample task={task!r}: conditioning_fps must be positive; got {fps_int}.")
        num_frames = int(sample.shape[-3])
        # Integer-floor seconds matches the canonical V4.2 ``M:SS`` rendering
        # in :func:`cosmos_framework.model.generator.upsampler.prompts._format_duration`,
        # which expects an int and rejects fractional seconds.
        duration_secs = max(1, num_frames // fps_int)
        return aspect_ratio, w, h, fps_int, duration_secs

    def _maybe_apply_prompt_upsampling(
        self,
        data_batch: dict,
        *,
        upsample_task: str | None,
        upsample_max_new_tokens: int,
        upsample_temperature: float | None,
        upsample_top_k: int | None,
        upsample_top_p: float | None,
        upsample_repetition_penalty: float = 1.0,
        upsample_presence_penalty: float = 0.0,
        upsample_seed: int | None = None,
    ) -> dict:
        """Rewrite the conditional captions in ``data_batch`` via the
        reasoner-tower prompt upsampler, returning a fresh dict.

        Orchestrates the full per-task upsampling pipeline that
        :meth:`generate_samples_from_batch` runs *before*
        :meth:`_prepare_inference_data` so that downstream tokenization,
        sequence-plan construction, and the CFG ``velocity_fn`` all see
        the upsampled prompts uniformly with no further changes.  Steps:

        1. **Task selection** — the caller resolves the V4.2 task
           (``"t2v"`` / ``"t2i"`` / ``"i2v"``) and passes it via
           ``upsample_task``.
        2. **Image conditioning (i2v only)** — for ``"i2v"``, one
           VLM-ready conditioning image per caption is pulled via
           :meth:`_extract_upsample_conditioning_images` so the V4.2
           i2v chat block actually carries the visual context.
        3. **Template specs** —
           :meth:`_extract_upsample_video_specs` derives
           ``aspect_ratio``, ``resolution_w/h``, and (for video tasks)
           ``fps`` / ``duration_secs`` from the same ``data_batch`` so
           the upsampler's V4.2 ``task_constraints`` block matches the
           clip the diffusion tower will actually generate.
        4. **Upsampling** — :meth:`upsample_captions` runs the reasoner
           AR loop with the V4.2 template, applies ``clean_response``
           post-processing, and falls back to the original caption on
           empty-cleaned output.
        5. **Substitution** — the caller's dict is shallow-copied and
           ``self.input_caption_key`` is overwritten with the upsampled
           captions.  The caller's dict is never mutated in place.

        The unconditional captions are NOT upsampled (negative prompts
        and empty strings pass through as-is); CFG runs with the
        original unconditional branch.  When CP/CFGP is enabled, the
        rank-uniformity of the final captions is handled inside
        :meth:`upsample_captions` (greedy decode + FSDP-lockstep
        per-prompt loop), so this orchestrator does not need to
        broadcast separately.

        Args:
            data_batch: Raw data batch handed to
                :meth:`generate_samples_from_batch`.  Must contain
                ``self.input_caption_key`` and one of
                ``{self.input_image_key, self.input_video_key}`` (so
                the caller can resolve a canonical V4.2 task).
            upsample_task: Canonical V4.2 task resolved by the inference
                caller. ``None`` short-circuits this method to a no-op
                that returns ``data_batch`` unchanged (same object identity).
            upsample_max_new_tokens: Per-caption decode budget passed
                through to :meth:`upsample_captions`.
            upsample_temperature, upsample_top_k, upsample_top_p:
                Sampling controls forwarded to
                :meth:`upsample_captions`.  Greedy decoding
                (``do_sample=False``) is NOT used here —
                ``generate_samples_from_batch`` historically runs the
                upsampler with sampling on (``do_sample=True``) so
                negative-prompt diversity isn't suppressed across the
                batch; see :meth:`upsample_captions` for the
                FSDP-lockstep contract.
            upsample_repetition_penalty: CTRL/HF-style multiplicative
                logit penalty on tokens already seen in each
                caption's history; forwarded to
                :meth:`upsample_captions`.  ``1.0`` (default) is
                identity and adds zero overhead to the reasoner AR
                loop.
            upsample_presence_penalty: OpenAI-style additive logit
                penalty (binary presence, not frequency) on tokens
                already seen; forwarded to :meth:`upsample_captions`.
                ``0.0`` (default) is identity.
            upsample_seed: Optional integer seed for the upsampler's
                sampling RNG; forwarded to :meth:`upsample_captions`
                and through to
                :func:`unified_mot._impl_generate_reasoner_text`.
                ``None`` (default) consumes the device's default RNG
                and is bit-identical to the pre-seed behavior.  Has
                no effect under greedy decoding.  Distinct from the
                per-sample diffusion noise seed handled by
                :meth:`generate_samples_from_batch`.

        Returns:
            ``data_batch`` unchanged if ``upsample_task is None``.
            Otherwise, a shallow copy with ``self.input_caption_key``
            replaced by the upsampled captions list.
        """
        if upsample_task is None:
            return data_batch
        if upsample_task not in {"t2v", "t2i", "i2v"}:
            raise ValueError(f"upsample_task must be one of {{'t2v', 't2i', 'i2v'}} or None; got {upsample_task!r}.")

        upsample_start_time = time.time()
        captions = data_batch[self.input_caption_key]
        log.info(
            "Prompt upsampling options: "
            f"task={upsample_task}, "
            f"max_new_tokens={upsample_max_new_tokens}, "
            f"temperature={upsample_temperature}, "
            f"top_k={upsample_top_k}, "
            f"top_p={upsample_top_p}, "
            f"repetition_penalty={upsample_repetition_penalty}, "
            f"presence_penalty={upsample_presence_penalty}, "
            f"seed={upsample_seed}",
            rank0_only=False,
        )
        # For i2v, hand the upsampler one VLM-ready image per caption so
        # the V4.2 i2v template's chat block actually carries visual context.
        upsample_input_images: list[Any] | None = None
        if upsample_task == "i2v":
            upsample_input_images = self._extract_upsample_conditioning_images(data_batch)

        # V4.2 templates inject the actual clip's aspect ratio, resolution,
        # fps, and duration into their ``task_constraints`` block, so the
        # rewritten caption stays consistent with the clip the diffusion
        # tower will generate.  Derive these specs from ``data_batch``
        # rather than relying on stale defaults.
        (
            upsample_aspect_ratio,
            upsample_res_w,
            upsample_res_h,
            upsample_fps,
            upsample_duration_secs,
        ) = self._extract_upsample_video_specs(data_batch, upsample_task)
        log.info(
            f"Prompt upsampling specs: aspect_ratio={upsample_aspect_ratio!r}, "
            f"resolution_w={upsample_res_w}, resolution_h={upsample_res_h}, "
            f"fps={upsample_fps}, duration_secs={upsample_duration_secs}",
            rank0_only=False,
        )
        upsampled_captions = self.upsample_captions(
            captions,
            max_new_tokens=upsample_max_new_tokens,
            task=upsample_task,
            aspect_ratio=upsample_aspect_ratio,
            resolution_w=upsample_res_w,
            resolution_h=upsample_res_h,
            fps=upsample_fps,
            duration_secs=upsample_duration_secs,
            images=upsample_input_images,
            do_sample=True,
            temperature=upsample_temperature,
            top_k=upsample_top_k,
            top_p=upsample_top_p,
            repetition_penalty=upsample_repetition_penalty,
            presence_penalty=upsample_presence_penalty,
            seed=upsample_seed,
        )
        data_batch = dict(data_batch)
        data_batch[self.input_caption_key] = upsampled_captions
        log.info(f"Prompt upsampling took {time.time() - upsample_start_time:.2f} seconds", rank0_only=False)
        return data_batch

    def _extract_upsample_conditioning_images(self, data_batch: dict) -> list[Any]:
        """Return VLM-ready conditioning images for the i2v prompt upsampler."""
        plans = data_batch.get("sequence_plan")
        if plans is None or self.input_video_key not in data_batch:
            raise ValueError(
                f"I2V prompt upsampling requires both sequence_plan and {self.input_video_key!r} in the data batch."
            )

        prompt_upsampling_images = data_batch.get("_prompt_upsampling_images")
        if prompt_upsampling_images is None:
            raise ValueError(
                "I2V prompt upsampling requires '_prompt_upsampling_images' with one VLM-ready image per caption."
            )
        if len(prompt_upsampling_images) != len(plans):
            raise ValueError(
                "I2V prompt upsampling image count must match sequence_plan count: "
                f"{len(prompt_upsampling_images)} != {len(plans)}."
            )
        return list(prompt_upsampling_images)

    def denoise(
        self,
        net: torch.nn.Module | None = None,
        data_batch_packed: PackedSequence | None = None,
        memory: MemoryState | None = None,
        video_temporal_causal: bool | None = None,
    ) -> dict:
        """
        Runs the MoT network on a packed multi-modal sequence to predict velocity (v) targets.

        Args:
            data_batch_packed: PackedSequence from `pack_input_sequence(...)`.
            memory: Optional pre-built MemoryState for autoregressive generation
                or KV-cache training.
            video_temporal_causal: Per-call attention-mode override forwarded to
                ``Cosmos3VFMNetwork.forward``; ``None`` (default) keeps the
                config-selected mode.

        Returns:
            dict containing:
                - "preds_vision": list[Tensor[C,T,H,W]], one per sample.
                - "preds_lidar": Velocity prediction for the LiDAR stream (if the network carries it).
                - "preds_action": Velocity prediction for action modality (if action_gen enabled).
                - "preds_sound": Velocity prediction for sound modality (if sound_gen enabled).
                - "lbl_metadata_und": Load balancing metadata for understanding pathway (if present).
                - "lbl_metadata_gen": Load balancing metadata for generation pathway (if present).
        """
        net = net or self.net
        out_net = net(
            packed_seq=data_batch_packed,
            memory=memory,
            video_temporal_causal=video_temporal_causal,
        )
        output_dict = dict()
        output_dict["preds_vision"] = out_net["preds_vision"]
        if "preds_lidar" in out_net:
            output_dict["preds_lidar"] = out_net["preds_lidar"]
        if self.config.action_gen and "preds_action" in out_net:
            output_dict["preds_action"] = out_net["preds_action"]
        if self.config.sound_gen and "preds_sound" in out_net:
            output_dict["preds_sound"] = out_net["preds_sound"]
        for key, value in out_net.items():
            if "lbl_metadata_" in key:
                output_dict[key] = value

        return output_dict

    def _tokenize_captions(
        self,
        captions: list[str],
        *,
        use_system_prompt: bool,
        system_prompt: str | None,
        is_video: bool,
    ) -> list[list[int]]:
        """Per-caption chat-template tokenization (ragged, no padding).

        Single source of truth for the per-caption loop shared by
        :meth:`tokenize_text` (which adds padding + tensor-wrapping) and
        :meth:`_get_inference_text_tokens` (which feeds the diffusion
        sampling pipeline that natively consumes ``list[list[int]]``).

        Args:
            captions: Raw text prompts.
            use_system_prompt: Already-resolved boolean (callers handle any
                ``None`` fallback against ``self.vlm_config``).
            system_prompt: Explicit system prompt that overrides
                ``use_system_prompt`` / ``is_video`` when supplied.
            is_video: Selects the video vs image default system prompt
                when ``use_system_prompt=True`` and no explicit
                ``system_prompt`` is given.
        """
        return [
            tokenize_caption(
                c,
                self.vlm_tokenizer,
                is_video=is_video,
                use_system_prompt=use_system_prompt,
                system_prompt=system_prompt,
            )
            for c in captions
        ]

    @torch.no_grad()
    def tokenize_text(
        self,
        text: str | list[str],
        *,
        use_system_prompt: bool | None = None,
        system_prompt: str | None = None,
        is_video: bool = False,
        pad_token_id: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Tokenize one or more text prompts into a ``[B, T]`` long tensor.

        Wraps each prompt in the VLM's chat template via
        :func:`tokenize_caption` (the same helper used by
        :meth:`_get_inference_text_tokens` during conditional sampling), so
        the resulting ids match what the model sees during training /
        inference (e.g. ``<|im_start|>user\\n…<|im_end|>\\n<|im_start|>assistant\\n``
        for Qwen3-style chat models).

        When multiple prompts of different lengths are supplied, the result
        is right-padded with ``pad_token_id``.  Right-padding is the natural
        choice for prefill into :meth:`generate_reasoner_text`, which reads
        the last position's logits to predict the first new token; for that
        to be meaningful, callers should normally pass either a single
        prompt or a list of equal-length prompts.

        Args:
            text: A single string or a list of strings, one per batch item.
            use_system_prompt: Override for ``self.vlm_config.use_system_prompt``.
                When ``None``, falls back to the VLM config flag.
            system_prompt: When supplied, this exact string is used as the
                system prompt and overrides ``use_system_prompt`` /
                ``is_video``.
            is_video: Selects the video vs image default system prompt
                when ``use_system_prompt=True`` and no explicit
                ``system_prompt`` is given.
            pad_token_id: Pad id for right-padding.  Defaults to
                ``self.vlm_tokenizer.pad_token_id`` when set, else
                ``self.llm_special_tokens["eos_token_id"]``, else ``0``.
            device: Optional device for the returned tensor.

        Returns:
            A ``torch.long`` tensor of shape ``[B, T_max]`` ready to be fed
            into :meth:`generate_reasoner_text` or any other token-driven
            entry point.
        """
        prompts: list[str] = [text] if isinstance(text, str) else list(text)
        if len(prompts) == 0:
            raise ValueError("tokenize_text requires at least one prompt.")
        resolved_use_system_prompt: bool = (
            bool(getattr(self.vlm_config, "use_system_prompt", False))
            if use_system_prompt is None
            else use_system_prompt
        )

        token_lists = self._tokenize_captions(
            prompts,
            use_system_prompt=resolved_use_system_prompt,
            system_prompt=system_prompt,
            is_video=is_video,
        )

        if pad_token_id is None:
            candidate = getattr(self.vlm_tokenizer, "pad_token_id", None)
            if candidate is None:
                candidate = self.llm_special_tokens.get("eos_token_id")
            pad_id: int = int(candidate) if candidate is not None else 0
        else:
            pad_id = int(pad_token_id)

        max_len = max(len(ids) for ids in token_lists)
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in token_lists]
        out = torch.tensor(padded, dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out

    @torch.no_grad()
    def detokenize_text(
        self,
        token_ids: torch.Tensor | list[int] | list[list[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> str | list[str]:
        """Decode token ids produced by :meth:`generate_reasoner_text` back to text.

        Accepts the natural output shapes:
          - 1D tensor / ``list[int]``  → returns a single ``str``.
          - 2D tensor / ``list[list[int]]`` → returns ``list[str]`` (one
            per row).

        Args:
            token_ids: Ids to decode.
            skip_special_tokens: Forwarded to
                ``self.vlm_tokenizer.decode``; when ``True`` (default),
                strips chat-template / vision boundary specials so the
                returned string contains only model-generated content.
        """
        # Normalize to nested Python lists.  Tensors are converted via
        # ``.tolist()`` (1D -> list[int]; 2D -> list[list[int]]); raw lists
        # pass through.  We use ``Any`` here to keep the body cleanly typed
        # against ``list`` without the type-checker pessimistically widening
        # to the input union after the isinstance narrow.
        raw: Any = token_ids
        if hasattr(raw, "tolist"):
            if hasattr(raw, "dim") and raw.dim() not in (1, 2):
                raise ValueError(f"token_ids tensor must be 1D or 2D, got shape {tuple(raw.shape)}")
            ids_list: list = raw.tolist()
        elif isinstance(raw, list):
            ids_list = raw
        else:
            raise TypeError(f"Unsupported token_ids type: {type(token_ids)}")

        if len(ids_list) == 0:
            return ""

        decode = self.vlm_tokenizer.decode
        if isinstance(ids_list[0], list):
            return [decode(row, skip_special_tokens=skip_special_tokens) for row in ids_list]
        return decode(ids_list, skip_special_tokens=skip_special_tokens)

    def _broadcast_outputs_to_parallel_groups(self, outputs: list[str]) -> list[str]:
        """Pin ``outputs`` to the lowest-numbered rank's value within each CP / CFGP group.

        Under context parallelism (CP) or CFG parallelism (CFGP), every
        rank within a parallel group is processing the same logical
        batch — CP splits the context dimension across ranks, CFGP splits
        the cfg/uncfg pair.  In a deterministic AR decode (e.g. greedy)
        the per-rank token streams should agree exactly, but numerical
        noise (kernel selection, FSDP all-gather ordering, RNG state in
        the sampling controls, EMA scope) can let those ranks emit
        token streams that differ at one or two greedy-tied positions.
        The public contract for ``generate_reasoner_text`` is "list[str]
        in, list[str] out" with a single canonical response per prompt
        across the parallel group, so we resolve any divergence by
        broadcasting rank 0's strings to every other rank within the
        group via :func:`torch.distributed.broadcast_object_list`.

        Order matches the seed-broadcast pattern used elsewhere in this
        class (CP first, then CFGP) so the final outputs are consistent
        across both groups when both are enabled.  No-op when neither CP
        nor CFGP is enabled (the typical case for pure-FSDP runs) — and
        also when ``parallel_dims`` is unset, e.g. CPU-only smoke tests.
        """
        if self.parallel_dims is None:
            return outputs
        if self.parallel_dims.cp_enabled and self.parallel_dims.cp_mesh is not None:
            cp_group = self.parallel_dims.cp_mesh.get_group()
            cp_bucket: list[Any] = [outputs]
            dist.broadcast_object_list(cp_bucket, group=cp_group, group_src=0)
            outputs = cp_bucket[0]
        if self.parallel_dims.cfgp_enabled and self.parallel_dims.cfgp_mesh is not None:
            cfgp_group = self.parallel_dims.cfgp_mesh.get_group()
            cfgp_bucket: list[Any] = [outputs]
            dist.broadcast_object_list(cfgp_bucket, group=cfgp_group, group_src=0)
            outputs = cfgp_bucket[0]
        return outputs

    @torch.no_grad()
    def generate_reasoner_text(
        self,
        inputs: list[str],
        max_new_tokens: int,
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        prompt_builder: Callable[[str], list[dict[str, Any]]] | None = None,
        do_sample: bool = False,
        temperature: float | None = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int | None = None,
    ) -> list[str]:
        """Autoregressively generate text using only the reasoner tower.

        High-level prompt-driven entry point: for each input prompt this
        method (a) builds a chat-style messages list via ``prompt_builder``
        (or wraps the prompt as a single user message when no callback is
        given), (b) tokenizes it — text-only via :meth:`tokenize_text`, or
        multimodal via ``self.vlm_processor.apply_chat_template`` when
        ``images`` or ``videos`` is supplied (the image path lowers the chat
        into ``input_ids``, ``attention_mask``, ``pixel_values``, and
        ``image_grid_thw``; the video path yields ``pixel_values_videos`` and
        ``video_grid_thw`` instead), (c)
        runs the reasoner-only AR decode loop through
        ``self.net.generate_reasoner_text`` (the lower-level token-driven
        pass-through that delegates to ``unified_mot._impl_generate_reasoner_text``),
        and (d) decodes via :meth:`detokenize_text`.  Returns the list of
        generated strings (same length as ``inputs``).  Empty /
        whitespace-only prompts pass through unchanged; an empty
        generation falls back to the original prompt so downstream
        tokenization stays well-defined.

        Per-prompt iteration is deliberate: the reasoner AR loop's
        ``attention_mask`` is only consumed by the multimodal prefill
        (for ``get_rope_index``) and not by the dense attention on the
        decode side, so right-padding different-length prompts in a
        single batched call would let the shorter prompt's prefill
        predict from a pad token at position ``T-1``, producing garbage
        continuations.  Callers needing batched, same-length token-level
        decoding can drive ``self.net.generate_reasoner_text(input_ids=...)``
        directly.

        Only the und-pathway weights — those WITHOUT the ``_moe_gen``
        suffix — plus ``embed_tokens`` / ``norm`` / ``lm_head`` (and the
        Qwen3-VL ``visual`` tower for the image-conditioned path)
        participate; the generation pathway and all VFM-level multimodal
        embedders / heads are bypassed.

        ``eos_token_id`` and ``pad_token_id`` are resolved internally and
        not exposed as parameters.  EOS comes from
        ``self.llm_special_tokens["eos_token_id"]`` (so generation
        terminates on the assistant-turn boundary natural to the live
        VLM).  Pad mirrors the resolution used by :meth:`tokenize_text`:
        the VLM tokenizer's ``pad_token_id`` if set, else falling back
        to the EOS id (which is harmless here because the prompt-driven
        loop runs each prompt at ``B=1``, so there is never a shorter
        sequence to pad after EOS).

        For EMA inference, wrap the call in ``self.ema_scope()`` as with
        the other generation entry points::

            with model.ema_scope():
                out = model.generate_reasoner_text(prompts, ...)

        Args:
            inputs: List of raw prompt strings.
            max_new_tokens: Decode budget per prompt.
            images: Optional per-prompt conditioning images.  Must have
                the same length as ``inputs`` when provided.  When
                non-``None``, each prompt is routed through the Qwen3-VL
                multimodal chat template — ``pixel_values`` /
                ``image_grid_thw`` / ``attention_mask`` are computed
                internally from this list and the VLM processor.  Each
                entry is forwarded verbatim into
                ``processor.apply_chat_template``, so any input it
                accepts works (file path ``str``, ``PIL.Image.Image``,
                ``np.ndarray``, or a CHW / HWC tensor).
            videos: Optional per-prompt conditioning videos (mutually
                exclusive with ``images``). Each entry must be a
                ``{"frames": [...PIL...], "fps": float}`` payload
                (pre-decoded by the caller, e.g. via
                ``_decode_reasoner_video``). The frames list and fps are
                forwarded into the ``{"type": "video", "video": frames,
                "fps": fps}`` chat block so the processor produces
                ``pixel_values_videos`` / ``video_grid_thw``.
            prompt_builder: Optional callback that maps a raw prompt
                string to a chat-style messages list (e.g.
                :func:`cosmos_framework.model.generator.upsampler.prompts.build_messages`
                for V4.2 caption upsampling).  When ``None``, prompts are
                wrapped as ``[{"role": "user", "content": prompt}]`` with
                no system message.
            do_sample, temperature, top_k, top_p: Sampling controls.
                Greedy (the default) is recommended for deterministic
                upsampling across CP/CFGP ranks; see "Distributed
                semantics" below for how non-greedy or numerically-noisy
                drift between ranks within a parallel group is reconciled.
            repetition_penalty: Multiplicative penalty applied to logits
                at vocab positions already seen in this sample's history
                (prompt + everything generated so far).  ``>1.0``
                discourages repetition, ``<1.0`` encourages it, ``1.0``
                (default) is identity.  Forwarded to
                :func:`unified_mot._impl_generate_reasoner_text`.
            presence_penalty: Additive penalty subtracted from logits at
                vocab positions already seen in this sample's history.
                ``>0`` discourages reuse, ``<0`` encourages it, ``0``
                (default) is identity.  Forwarded to
                :func:`unified_mot._impl_generate_reasoner_text`.
            seed: Optional integer seed for the sampling RNG; forwarded
                verbatim to
                :func:`unified_mot._impl_generate_reasoner_text`, which
                allocates a device-local ``torch.Generator``, seeds it
                once with ``manual_seed(seed)``, and threads it into
                every ``torch.multinomial`` draw.  ``None`` (default)
                consumes the device's default RNG and is bit-identical
                to the pre-seed behavior.  Greedy decoding doesn't read
                the generator, so the value has no effect when
                ``do_sample=False``.  Under CP/CFGP, the same ``seed``
                must be passed on every rank within a parallel group
                for the per-prompt loop to produce identical token
                streams pre-broadcast; the post-decode
                :meth:`_broadcast_outputs_to_parallel_groups` pin masks
                divergences either way, but matching seeds keep the
                ranks doing the same work.

        Returns:
            ``list[str]`` of generated text (same length as ``inputs``).

        Distributed semantics:
            When CP or CFGP is enabled, every rank within a parallel
            group is processing the same logical batch, and is
            *expected* to produce identical strings.  Before returning,
            this method pins ``outputs`` to the lowest-numbered rank's
            value within each group via
            :func:`torch.distributed.broadcast_object_list` (CP first,
            then CFGP — matching the seed-broadcast ordering used
            elsewhere in this class).  This guarantees a single
            canonical ``list[str]`` per call even when numerical noise
            (kernel selection, RNG, FSDP all-gather ordering, EMA scope)
            makes the per-rank token streams disagree at greedy-tied
            positions.  No-op when neither CP nor CFGP is enabled.  See
            :meth:`_broadcast_outputs_to_parallel_groups` for the
            implementation.

        Raises:
            ValueError: If ``images`` length does not match ``inputs``
                length, or if ``videos`` length does not match ``inputs``
                length.  Also raised if both ``images`` and ``videos`` are
                supplied simultaneously (only one medium is allowed per
                call).
            RuntimeError: If ``images`` or ``videos`` is supplied but the
                live VLM processor does not implement
                ``apply_chat_template`` (i.e., the VLM is configured as
                text-only).
        """
        # Decide whether the multimodal flow is in play, and validate the
        # image-list contract here so the failure happens before any
        # decoding work — far easier to debug than a downstream
        # ``apply_chat_template`` error.
        if images is not None and videos is not None:
            raise ValueError(
                "generate_reasoner_text conditions on one medium at a time: pass `images` OR `videos`, not both."
            )
        use_image = images is not None
        use_video = videos is not None
        use_multimodal = use_image or use_video
        media = images if use_image else videos
        if use_multimodal:
            assert media is not None  # narrowed by `use_multimodal`
            if len(media) != len(inputs):
                raise ValueError(
                    f"generate_reasoner_text: media length ({len(media)}) "
                    f"must equal `inputs` length ({len(inputs)}) for the "
                    "vision-conditioned flow."
                )
            if not callable(getattr(self.vlm_processor, "apply_chat_template", None)):
                raise RuntimeError(
                    "generate_reasoner_text(images=... / videos=...) requires a multimodal "
                    "VLM processor (e.g. Qwen3VLProcessor) but the live processor "
                    f"{type(self.vlm_processor).__name__!r} does not implement "
                    "apply_chat_template — the live VLM is configured as text-only."
                )
        # Resolve EOS / pad ids internally so callers don't have to know
        # about VLM-specific id wiring.  EOS comes from the cached VLM
        # special-tokens dict (set in ``set_up_tokenizers``); pad mirrors
        # the resolution used by ``tokenize_text`` (vlm_tokenizer.pad_token_id,
        # falling back to EOS when the tokenizer has no dedicated pad).
        eos_raw = self.llm_special_tokens.get("eos_token_id")
        eos_id: int | None = int(eos_raw) if eos_raw is not None else None
        pad_raw = getattr(self.vlm_tokenizer, "pad_token_id", None)
        if pad_raw is None:
            pad_raw = eos_raw
        pad_id: int | None = int(pad_raw) if pad_raw is not None else None
        device = self.tensor_kwargs.get("device", "cuda")

        outputs: list[str] = []
        for idx, prompt in enumerate(inputs):
            # Empty / whitespace-only prompts pass through unchanged so
            # callers that drop them downstream don't have to special-case
            # the response shape.
            if not prompt or not prompt.strip():
                outputs.append(prompt)
                continue

            # The prompt_builder callback turns the raw prompt into a chat-
            # style messages list (e.g. ``build_messages`` for the V4.2
            # caption upsampler templates).  Default: a single user-role
            # message with no system prompt.  Annotate explicitly to
            # ``list[dict[str, Any]]`` so the multimodal branch's
            # mixed-type content (str + list[dict]) does not get narrowed
            # to ``dict[str, str]`` by the no-callback default.
            messages: list[dict[str, Any]]
            if prompt_builder is not None:
                messages = prompt_builder(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]

            if use_multimodal:
                assert media is not None  # narrowed by `use_multimodal`
                last_user = messages[-1]
                last_text = last_user["content"] if isinstance(last_user.get("content"), str) else ""
                if use_video:
                    media_item: dict[str, Any] = {
                        "type": "video",
                        "video": media[idx]["frames"],
                        "fps": media[idx]["fps"],
                    }
                else:
                    media_item = {"type": "image", "image": media[idx]}
                multimodal_messages = list(messages[:-1])
                multimodal_messages.append(
                    {
                        "role": "user",
                        "content": [media_item, {"type": "text", "text": last_text}],
                    }
                )
                processor_inputs = self.vlm_processor.apply_chat_template(
                    multimodal_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                inner_input_ids = processor_inputs["input_ids"].to(device).unsqueeze(0)
                inner_attention_mask = processor_inputs["attention_mask"].to(device).unsqueeze(0)
                if use_video:
                    inner_pixel_values_videos = processor_inputs["pixel_values_videos"].to(device)
                    inner_video_grid_thw = processor_inputs["video_grid_thw"].to(device)
                    out_ids = self.net.generate_reasoner_text(
                        input_ids=inner_input_ids,
                        max_new_tokens=max_new_tokens,
                        pixel_values_videos=inner_pixel_values_videos,
                        video_grid_thw=inner_video_grid_thw,
                        attention_mask=inner_attention_mask,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        do_sample=do_sample,
                        temperature=temperature if temperature is not None else 1.0,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        seed=seed,
                        return_only_new_tokens=True,
                    )
                else:
                    inner_pixel_values = processor_inputs["pixel_values"].to(device)  # [N_patches,C,H,W]
                    inner_image_grid_thw = processor_inputs["image_grid_thw"].to(device)  # [num_images,3]
                    out_ids = self.net.generate_reasoner_text(
                        input_ids=inner_input_ids,
                        max_new_tokens=max_new_tokens,
                        pixel_values=inner_pixel_values,
                        image_grid_thw=inner_image_grid_thw,
                        attention_mask=inner_attention_mask,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        do_sample=do_sample,
                        temperature=temperature if temperature is not None else 1.0,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        seed=seed,
                        return_only_new_tokens=True,
                    )
            else:
                # Text-only path.  Pull the system prompt (if any) and
                # the last user message text out of the messages list,
                # then route through ``tokenize_text`` to get the chat-
                # templated prompt ids.  Falls back to the raw prompt if
                # the callback returned a non-string user content.
                system_prompt: str | None = None
                user_text: str = prompt
                if messages:
                    first = messages[0]
                    if first.get("role") == "system" and isinstance(first.get("content"), str):
                        system_prompt = first["content"]
                    last_user = messages[-1]
                    if last_user.get("role") == "user" and isinstance(last_user.get("content"), str):
                        user_text = last_user["content"]
                prompt_ids = self.tokenize_text(
                    user_text,
                    system_prompt=system_prompt,
                    device=device,
                )
                out_ids = self.net.generate_reasoner_text(
                    input_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    do_sample=do_sample,
                    temperature=temperature if temperature is not None else 1.0,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    seed=seed,
                    return_only_new_tokens=True,
                )
            decoded = self.detokenize_text(out_ids, skip_special_tokens=True)
            assert isinstance(decoded, list) and len(decoded) == 1, (
                f"detokenize_text returned unexpected shape: {type(decoded)}"
            )
            result = decoded[0].strip()
            # If the reasoner returns an empty string (rare), keep the
            # original prompt so downstream tokenization stays well-defined.
            outputs.append(result if result else prompt)

        # Reconcile any per-rank divergence within CP/CFGP groups so
        # callers see a single canonical ``list[str]`` per call.  No-op
        # when neither CP nor CFGP is enabled.  See
        # :meth:`_broadcast_outputs_to_parallel_groups` for the contract.
        return self._broadcast_outputs_to_parallel_groups(outputs)

    @torch.no_grad()
    def upsample_captions(
        self,
        captions: list[str],
        max_new_tokens: int,
        *,
        task: str = "t2v",
        aspect_ratio: str,
        resolution_w: int,
        resolution_h: int,
        fps: int | None = None,
        duration_secs: int | None = None,
        images: list[Any] | None = None,
        do_sample: bool = False,
        temperature: float | None = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int | None = None,
    ) -> list[str]:
        """Expand each caption via the reasoner tower's autoregressive loop.

        Thin task-aware wrapper over :meth:`generate_reasoner_text`'s
        prompt-driven branch.  The only thing this method adds on top of
        the generic per-prompt loop is the V4.2 chat-template injection:
        each caption is wrapped via
        :func:`cosmos_framework.model.generator.upsampler.prompts.build_messages`
        (which returns ``[system, user]`` with the user content embedding
        the caption inside the canonical V4.2 template — instructions,
        task constraints, and output JSON schema for the requested task).

        Two flows share this entry point:

        - **Text-only** (``task in {"t2v", "t2i"}``, or ``task="i2v"`` with
          ``images=None``): the user content is fed via :meth:`tokenize_text`
          (chat template ``<|im_start|>system\\n…<|im_end|>\\n<|im_start|>user\\n…<|im_end|>\\n<|im_start|>assistant\\n``)
          and the reasoner-only AR loop runs on text tokens alone.
        - **Image-conditioned** (``task="i2v"`` with one image per caption):
          the user content becomes a Qwen3-VL multimodal block —
          ``[{"type": "image", "image": images[i]}, {"type": "text", "text": user_text}]``
          — which ``self.vlm_processor.apply_chat_template`` lowers into
          ``input_ids`` (with image-placeholder tokens at the right
          positions), ``attention_mask``, ``pixel_values``, and
          ``image_grid_thw``.  All four route through
          ``unified_mot._impl_generate_reasoner_text``'s multimodal prefill
          (visual encoder → ``masked_scatter`` of image embeddings → mrope
          position ids) before kicking off the AR decode loop.

        Each raw reasoner output is post-processed by
        :func:`cosmos_framework.model.generator.upsampler.prompts.clean_response`
        before being returned.  The cleaner strips
        ``<think>`` / ``<reasoning>`` / ``<thinking>`` / etc. reasoning
        blocks and any prose preamble that appears before the
        ``` ```json ``` `` fence.  For canonical V4.2 SFT outputs
        (cosmos3 upsampler 8B / 32B), which already emit clean fenced
        JSON with no reasoning markers, this is a documented
        byte-for-byte no-op; for reasoning-style backbones
        (Qwen3-reasoning, DeepSeek-R1, etc.) it removes the chatter so
        the downstream JSON fence is the first content in the returned
        string.  ``clean_response`` is idempotent and NEVER raises.

        Defensive fallback: if the cleaner leaves an empty /
        whitespace-only string behind — e.g. the raw output was
        entirely a stripped-out thinking block with no JSON body — the
        original caption is returned in its place so downstream
        tokenization stays well-defined.  This extends the same
        safety net :meth:`generate_reasoner_text` already provides
        for empty model output (line ~3550) across the
        post-processing stage; without it, ``clean_response`` could
        collapse a non-empty reasoner output to ``""`` and that empty
        string would propagate into
        ``data_batch[self.input_caption_key]`` at the call site in
        :meth:`generate_samples_from_batch`.

        Telemetry: a single rank-local summary log line is emitted
        per call ONLY when at least one sample required stripping or
        triggered the empty-cleaned fallback; on the canonical
        all-clean path the method is silent.  See
        :meth:`generate_reasoner_text` for the underlying per-prompt
        loop contract (image-list validation, EMA scope, etc.).

        Args:
            captions: Raw input captions (typically
                ``data_batch[self.input_caption_key]``).
            task: Canonical upsampler task — ``"t2v"`` (text-to-video,
                default), ``"t2i"`` (text-to-image), or ``"i2v"``
                (image-to-video).  Selects which V4.2 template
                ``build_messages`` returns.
            aspect_ratio: Output clip aspect ratio in comma form,
                e.g. ``"1,1"``, ``"16,9"``, ``"9,16"``, ``"4,3"``,
                ``"3,4"``.  Injected into the ``aspect_ratio``
                constraint of every V4.2 template (all three tasks).
                Required.
            resolution_w: Output frame width in pixels.  Injected into
                the ``resolution`` constraint of every V4.2 template
                (all three tasks).  Required.
            resolution_h: Output frame height in pixels.  Counterpart
                to ``resolution_w``.  Required for all tasks.
            fps: Target frames-per-second for the generated clip.
                Required for the video tasks (``"t2v"``, ``"i2v"``)
                and must be ``None`` for ``"t2i"`` — the underlying
                :func:`cosmos_framework.model.generator.upsampler.prompts.build_user_text`
                raises ``ValueError`` if a video task is missing
                ``fps`` or ``duration_secs``.
            duration_secs: Clip duration in whole seconds (rendered as
                ``M:SS`` inside the template).  Same required/forbidden
                contract as ``fps``.
            images: Optional per-caption conditioning images for the
                image-conditioned (``task="i2v"``) flow.  Must have the
                same length as ``captions`` when provided.  Each entry is
                forwarded verbatim into the Qwen3-VL multimodal chat
                template, so any input the underlying
                ``processor.apply_chat_template`` accepts works (file
                path ``str``, ``PIL.Image.Image``, ``np.ndarray``, or
                a CHW / HWC tensor).  Required for ``task="i2v"`` and
                ignored (with a warning) for the text-only tasks.
            max_new_tokens: Per-caption decode budget.  Sized to comfortably
                fit a one-paragraph expanded prompt.
            do_sample, temperature, top_k, top_p: Sampling controls,
                forwarded to :meth:`generate_reasoner_text`.  Greedy
                decoding (the default) is recommended so upsampling is
                deterministic across CP/CFGP ranks.
            repetition_penalty: Multiplicative penalty applied to logits
                at vocab positions already seen in this caption's
                history (prompt + everything generated so far).
                ``>1.0`` discourages verbatim repetition, ``<1.0``
                encourages it, ``1.0`` (default) is identity and adds
                zero overhead to the AR loop.  Forwarded verbatim to
                :meth:`generate_reasoner_text` and through to
                :func:`unified_mot._impl_generate_reasoner_text`.
            presence_penalty: Additive penalty subtracted from logits
                at vocab positions already seen in this caption's
                history (binary presence, not frequency).  ``>0``
                discourages reuse, ``<0`` encourages it, ``0``
                (default) is identity.  Same forwarding chain as
                ``repetition_penalty``.
            seed: Optional integer seed for the sampling RNG.
                Forwarded verbatim to :meth:`generate_reasoner_text`
                and through to
                :func:`unified_mot._impl_generate_reasoner_text`,
                which seeds a fresh device-local ``torch.Generator``
                once and threads it into every ``torch.multinomial``
                draw.  ``None`` (default) consumes the device's
                default RNG and is bit-identical to the pre-seed
                behavior.  Greedy decoding doesn't read the
                generator, so the value has no effect when
                ``do_sample=False``.

        Returns:
            A list of post-processed upsampled captions, same length as
            ``captions``.

        Raises:
            ValueError: If ``images`` length does not match ``captions``
                length (raised inside :meth:`generate_reasoner_text`).
            RuntimeError: If the multimodal flow is requested but the
                live VLM is LLM-only (raised inside
                :meth:`generate_reasoner_text`).
        """
        # Text-only tasks ignore conditioning images; warn loudly so the
        # caller doesn't mistakenly assume the image affected upsampling.
        if task != "i2v" and images is not None:
            log.warning(f"upsample_captions(task={task!r}) received `images` but only task='i2v' uses them; ignoring.")
            images = None

        def _builder(description: str) -> list[dict[str, Any]]:
            return build_messages(
                task=task,
                description=description,
                aspect_ratio=aspect_ratio,
                resolution_w=resolution_w,
                resolution_h=resolution_h,
                fps=fps,
                duration_secs=duration_secs,
            )

        raw_outputs = self.generate_reasoner_text(
            captions,
            max_new_tokens=max_new_tokens,
            prompt_builder=_builder,
            images=images,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )

        # Defensive post-processing: strip thinking-style markers from
        # the raw reasoner output via ``clean_response``.  This is a
        # documented byte-for-byte no-op on canonical V4.2 SFT outputs
        # (which emit clean fenced JSON with no reasoning markers) and
        # only fires for reasoning-style backbones.  If the cleaner
        # collapses a non-empty reasoner output to an empty /
        # whitespace-only string (e.g., the raw was entirely a
        # ``<think>...</think>`` block with no JSON body), fall back
        # to the original caption so downstream tokenization stays
        # well-defined — without this, an empty string would propagate
        # into ``data_batch[self.input_caption_key]`` at the call site.
        cleaned_outputs: list[str] = []
        n_stripped = 0
        for raw, original in zip(raw_outputs, captions):
            cleaned_text, clean_info = clean_response(raw)
            if not clean_info["was_clean"]:
                n_stripped += 1
            if not cleaned_text.strip():
                cleaned_text = original

            # Stamp the actual generation ``duration`` onto the upsampled
            # JSON object using the duration_secs argument. Only done for
            # T2V and I2V tasks.
            if duration_secs is not None:
                cleaned_text = cleaned_text.removeprefix("```json").removesuffix("```").strip()
                obj = json.loads(cleaned_text)
                assert isinstance(obj, dict), f"JSON parsing failed with error: {type(obj)}"
                obj["duration"] = f"{duration_secs}s"
                cleaned_text = json.dumps(obj)

            cleaned_outputs.append(cleaned_text)

        # Stay silent on the canonical all-clean path; only emit
        # telemetry when something actually happened.  Logged per-rank
        # to match the surrounding upsampling logs in
        # :meth:`generate_samples_from_batch` (line ~2218).
        if n_stripped:
            log.info(
                f"upsample_captions(task={task!r}, n={len(raw_outputs)}): thinking-stripped={n_stripped}",
                rank0_only=False,
            )

        return cleaned_outputs

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer_vision_gen.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer_vision_gen.decode(latent)

    @torch.no_grad()
    def encode_lidar(self, state: torch.Tensor) -> torch.Tensor:
        """Encode a LiDAR range clip with the LiDAR VAE."""
        return self._require_lidar_tokenizer().encode(state)

    @torch.no_grad()
    def decode_lidar(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode LiDAR latents to range pixels with the LiDAR VAE.

        The stream keeps its own VAE's channel count end to end, so a latent arrives here
        at the width its decoder expects. Both tokenizer versions return metric range,
        unit intensity, and a resolved mask, so the visualizers downstream never branch
        on which one is configured.
        """
        return self._require_lidar_tokenizer().decode(latent)

    @torch.no_grad()
    def encode_sound(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform into latent tokens.

        Args:
            waveform: Audio tensor of shape (C, N). A batch dim is added/removed
                      internally since AVAE expects (B, C, N).
                      Mono audio is duplicated to stereo if the tokenizer expects 2 channels.
        """
        assert self.tokenizer_sound_gen is not None, "Sound tokenizer not initialized"
        # Ensure correct number of channels (AVAE typically expects stereo)
        expected_channels = self.tokenizer_sound_gen.audio_channels
        if waveform.shape[0] == 1 and expected_channels == 2:
            waveform = waveform.repeat(2, 1)  # mono → stereo
        elif waveform.shape[0] > expected_channels:
            waveform = waveform[:expected_channels]
        # AVAE expects (B, C, N)
        latent = self.tokenizer_sound_gen.encode(waveform.unsqueeze(0))  # [1,sound_channels,T_sound]
        return latent.squeeze(0)  # [sound_channels,T_sound]

    @torch.no_grad()
    def decode_sound(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode sound latent tokens back to waveform.

        Args:
            latent: Sound latent tensor of shape (C, T). A batch dim is added/removed
                    internally since AVAE expects (B, C, T).
        """
        assert self.tokenizer_sound_gen is not None, "Sound tokenizer not initialized"
        # AVAE expects (B, C, T)
        waveform = self.tokenizer_sound_gen.decode(latent.unsqueeze(0))  # [1,audio_channels,N_samples]
        return waveform.squeeze(0)  # [audio_channels,N_samples]

    def _get_sound_fps_for_rope(self) -> float:
        """Compute the sound FPS to pass to RoPE for temporal alignment with video.

        Returns the sound tokenizer's latent rate (e.g., 25 Hz for 48kHz/1920 hop).
        This is passed as input_fps to the sound RoPE's generate_embeddings(), where
        the FPS modulation formula aligns sound indices with video indices.
        """
        return float(self.config.sound_latent_fps)

    def get_video_height_width(self) -> Tuple[int, int]:
        return VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]

    def get_video_latent_height_width(self) -> Tuple[int, int]:
        height, width = VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]
        return (
            height // self.tokenizer_vision_gen.spatial_compression_factor,
            width // self.tokenizer_vision_gen.spatial_compression_factor,
        )

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")

    def add_lora(
        self,
        network: torch.nn.Module,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: str,
    ) -> torch.nn.Module:
        """Pre-FSDP LoRA injection — see :func:`inject_lora_pre_fsdp` for details."""
        from cosmos_framework.utils.generator.lora import inject_lora_pre_fsdp

        self.lora_alpha = lora_alpha
        return inject_lora_pre_fsdp(
            network,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
        )

    def _init_lora_weights_post_materialization(self, network: torch.nn.Module) -> None:
        """Post-materialization LoRA init — see :func:`init_lora_weights_post_materialization`."""
        from cosmos_framework.utils.generator.lora import init_lora_weights_post_materialization

        init_lora_weights_post_materialization(network)


def _get_master_dtype_and_mp_policy(
    config: OmniMoTModelConfig, precision: torch.dtype
) -> tuple[torch.dtype, MixedPrecisionPolicy | None]:
    """Resolve the parameter storage dtype together with the FSDP2 policy that matches it.

    These are decided in one place because they describe one arrangement from opposite ends:
    the returned dtype is what ``nn.Parameter.data`` holds on each rank -- what the optimizer
    steps and what the checkpoint stores -- while the policy is what FSDP2 casts to around it.
    The parameters are generally kept in a higher-precision "master" dtype (FP32 by default)
    and FSDP2 down-casts to ``precision`` for the forward/backward and the all-gather.

    Three outcomes, in the order they are decided:

    1. ``parallelism.enable_inference_mode`` collapses the master dtype to ``precision``, then
       falls into case 2. With no optimizer to step a master copy and no gradient to reduce
       into it, a higher-precision parameter only costs resident weight memory and an
       all-gather down-cast per forward.
    2. A master dtype equal to ``precision`` yields no policy: nothing is cast, and the
       parameters are stored, computed with, and reduced in the compute dtype. This is the
       pre-mixed-precision behaviour, and setting ``parallelism.fsdp_master_dtype`` to
       ``precision`` is how to opt back out of mixed precision by hand.
    3. Otherwise a policy casts to ``precision`` for compute and all-gather, and reduces
       gradients in ``parallelism.fsdp_reduce_dtype``. Only parameters are cast; forward
       inputs are left alone (see the comment on ``cast_forward_inputs`` below).

    Args:
        config: Read for ``parallelism.enable_inference_mode``, ``parallelism.fsdp_master_dtype``
            and ``parallelism.fsdp_reduce_dtype``.
        precision: The forward/backward compute dtype, i.e. ``MixedPrecisionPolicy.param_dtype``
            -- ``OmniMoTModel.precision``, mapped from ``config.precision``.

    Returns:
        ``(master_dtype, mp_policy)``: the dtype to build and store the network's parameters in
        (pass as ``build_net(dtype=...)``), and the policy to install on each FSDP unit, or
        ``None`` to cast nothing. ``mp_policy`` is ``None`` exactly when ``master_dtype ==
        precision``.
    """
    master_dtype = (
        precision
        if config.parallelism.enable_inference_mode
        else PRECISION_TO_TORCH_DTYPE[config.parallelism.fsdp_master_dtype]
    )
    if master_dtype == precision:
        mp_policy = None
    else:
        if config.parallelism.fsdp_reduce_dtype is None:
            # Follow the master dtype so the reduced gradient lands in the dtype of the
            # shard it writes back into, which is what the optimizer steps in place.
            reduce_dtype = master_dtype
        else:
            reduce_dtype = PRECISION_TO_TORCH_DTYPE[config.parallelism.fsdp_reduce_dtype]
        # ``cast_forward_inputs`` stays off because FSDP2 casts forward inputs with
        # ``_apply_to_tensors``, which recurses into dataclasses and rewrites every
        # floating-point tensor it reaches. That reaches through ``PackedSequence`` into
        # metadata that must keep its own dtype: FP32 rectified-flow timesteps and, under
        # ``enable_fps_modulation``, FP32 mRoPE position ids -- neither survives BF16
        # (positions near the modality margin round to multiples of 64). Activations are
        # already handed to the network in ``precision``, so nothing needs the cast.
        mp_policy = MixedPrecisionPolicy(param_dtype=precision, reduce_dtype=reduce_dtype, cast_forward_inputs=False)
    return master_dtype, mp_policy


def _broadcast_seed(seed: list[int], group: dist.ProcessGroup, rank: int) -> list[int]:
    if rank == 0:
        seed_tensor = torch.tensor(seed, dtype=torch.int64, device=DEVICE)  # [len(seed)]
    else:
        seed_tensor = torch.zeros(len(seed), dtype=torch.int64, device=DEVICE)  # [len(seed)]

    dist.broadcast(seed_tensor, group=group, group_src=0)
    return seed_tensor.tolist()


def _is_reasoner_state_dict_key(key: str) -> bool:
    """Return True for und/reasoner-tower weights nested under ``language_model``.

    Reasoner weights are the understanding-pathway parameters in the MoT
    language tower: ``embed_tokens``, ``norm``, ``lm_head``, ``visual``, and
    every layer weight *without* the ``_moe_gen`` suffix.  Generation-pathway
    duplicates (``*_moe_gen``) and all non-``language_model`` VFM heads are
    excluded from this predicate.
    """
    key = key.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
    if not key.startswith("language_model."):
        return False
    return "_moe_gen" not in key
