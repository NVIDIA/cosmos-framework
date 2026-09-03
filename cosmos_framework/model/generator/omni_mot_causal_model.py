# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal (AR / teacher-forcing) extension of OmniMoTModel.

``OmniMoTCausalModel`` inherits from ``OmniMoTModel`` and adds:

* Autoregressive (AR) video generation methods
  (``iter_samples_from_batch_autoregressive``, ``generate_next_frame``).
* Teacher-forcing attention backends for temporal-causal training.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Generator
from typing import Any, Literal

import attrs
import torch
import torch.distributed as dist
from loguru import logger as log
from typing_extensions import override

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, _broadcast_seed
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.utils.memory import MemoryState
from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.data.generator.sequence_packing.modality import compute_text_split_length
from cosmos_framework.model.generator.attention_io_layout import AttentionIOLayout
from cosmos_framework.model.generator.mot.causal_attention import dispatch_attention_with_memory
from cosmos_framework.model.generator.mot.post_saturation.installer import install_ar_post_saturation_mode
from cosmos_framework.model.generator.mot.post_saturation.runtime import (
    is_ar_post_saturation_cuda_graph_frame,
    is_ar_post_saturation_static_compile_frame,
    reset_ar_post_saturation_runtime_for_generation,
    run_ar_post_saturation_cuda_graph,
    uses_ar_post_saturation_static_compile,
)
from cosmos_framework.model.generator.mot.post_saturation.static_compile import (
    validate_ar_static_und_cache_lengths,
)
from cosmos_framework.model.generator.teacher_forcing import make_teacher_forcing_clean_pack
from cosmos_framework.model.generator.utils.kv_cache import (
    ARMemoryState,
    DualKVCache,
    TeacherForcingMemoryState,
)
from cosmos_framework.model.generator.utils.kv_storage_backend import validate_kv_cache_dtype
from cosmos_framework.model.generator.utils.nvfp4 import resolve_legacy_nvfp4_mode
from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive
from cosmos_framework.utils.generator.data_batch import condition_frame_indexes_vision_from_batch


class GaussianBellTrainTimeWeight:
    """Interactive-local rCM-style RF time weighting."""

    def __init__(self, noise_scheduler: Any):
        self.noise_scheduler = noise_scheduler

    def __call__(self, t: torch.Tensor, tensor_kwargs: dict) -> torch.Tensor:  # t: [B] or [T], returns [B] or [T]
        del tensor_kwargs
        t_rf = t / float(self.noise_scheduler.config.num_train_timesteps)  # [B] or [T]
        floor = torch.exp(torch.tensor(-0.5, device=t.device, dtype=t.dtype))  # []
        return torch.exp(-2 * (t_rf - 0.5) ** 2) - floor  # [B] or [T]


def _get_context_parallel_num_kv_heads(num_kv_heads: int, parallel_dims: Any | None) -> int:
    """Return the KV heads visible to each CP rank after Ulysses all-to-all."""
    if parallel_dims is None or not getattr(parallel_dims, "cp_enabled", False):
        return num_kv_heads

    cp_size = int(parallel_dims.cp_size)
    kv_head_repeats = max(cp_size // num_kv_heads, 1)
    repeated_kv_heads = num_kv_heads * kv_head_repeats
    if repeated_kv_heads % cp_size != 0:
        raise ValueError(
            f"Repeated KV heads ({repeated_kv_heads}) must be divisible by "
            f"context parallel size ({cp_size}); "
            f"got KV heads={num_kv_heads}, repeats={kv_head_repeats}."
        )
    return repeated_kv_heads // cp_size


@attrs.define(slots=False)
class OmniMoTCausalModelConfig(OmniMoTModelConfig):
    """Config for the causal (AR / teacher-forcing) extension of OmniMoTModel.

    Inherits all base-model fields and adds causal-specific settings for
    diffusion forcing, teacher forcing, and inference cache sizes.
    The remaining causal fields (``video_temporal_causal``,
    ``causal_training_strategy``) still live on the parent because the base
    class reads them directly; they will migrate here once the corresponding
    base-class logic moves behind overridable hooks.
    """

    # Tensor layout at the attention boundary when CP is enabled. Both layouts
    # may run the attention kernel with head-sharded Q/K/V:
    # ``sequence_sharded`` keeps surrounding projections/MLP sequence-sharded
    # with Ulysses-style all-to-all into/out of attention, while
    # ``replicated`` keeps current-frame hidden states replicated, slices
    # local heads before attention, then reduces/gathers attention output back
    # to replicated hidden states. The ``replicated`` layout is only tested for
    # interactive models.
    attention_io_layout: AttentionIOLayout = "sequence_sharded"

    # σ_small injected into clean-context frames at AR inference under
    # causal_training_strategy="diffusion_forcing". Under DF training, the per-frame
    # σ sampler has continuous support on (0, 1) and never emits exactly 0, so a
    # truly-clean (σ=0) context frame — the TF-style refresh path — is
    # out-of-distribution at inference. The K/V-seeding pass instead feeds
    # x_in = σ_small · ε + (1 - σ_small) · x to keep cache inputs inside the
    # training distribution. Interpreted as post-shift σ (the actual noise level the
    # model sees), so the timestep passed to the network is σ_small · num_train_timesteps
    # with no further shift-transform. Default is empirical — tune per model.
    # Ignored when causal_training_strategy != "diffusion_forcing".
    sigma_diffusion_forcing: float = 0.02

    # Rolling KV cache size for AR inference, specified in latent frames.
    # None = unbounded (keep all frames). When set, GenKVCache uses a circular
    # buffer of this many latent frames.
    kv_cache_inference_size: int | None = None

    # Number of initial AR-inference generation frames to keep pinned in the
    # KV cache while later frames roll through the remaining cache slots.
    attention_sink_size: int = 0

    # Maximum packed understanding/text KV length used by post-saturation
    # static AR compile. The real prompt length is validated during frame-0
    # prefill, then cached und K/V is padded to this fixed length only for
    # the static compile path so prompt-length differences do not retrace.
    ar_static_und_cache_max_len: int = attrs.field(default=512, validator=attrs.validators.ge(1))

    # Storage format for the AR-inference generation KV cache.
    # None selects BF16 (default, off); "fp8" selects tensor-scale e4m3 FP8.
    kv_cache_dtype: str | None = None

    # Batch decode kernel for the FP8 AR-inference generation KV cache.
    # FP8 encode always uses torch; "triton" selects fused batch decode.
    kv_cache_kernel_impl: str = "triton"

    # NVFP4 quantization of the decoder linear layers, applied right after the checkpoint
    # loads (see ``OmniMoTCausalModel.maybe_convert_linears_to_nvfp4``). None ⇒ off (bf16);
    # "w4a4" ⇒ NVFP4 weights + activations via the vendored FP4 GEMM; "w4a16" ⇒ weight-only via
    # torchao (packed 4-bit weights, bf16 activations). Conversion is post-load
    # (not at __init__) on purpose: DCP loads the bf16 ``weight`` into plain ``nn.Linear``
    # first, and pre-quantized buffers (no ``weight`` param, no lazy-quant) keep the layer
    # ``torch.compile``-clean. ``torch.compile`` / CUDA graphs are optional run-flag add-ons.
    nvfp4_linears: str | None = None

    # Teacher forcing: whether to detach clean K/V in Pass 1.
    # When True, Pass 1 runs under torch.no_grad() and clean K/V
    # are detached (lower memory, no gradient through cache projections).
    # When False, gradients flow through the clean K/V, matching non-replayed
    # teacher forcing at the cost of 2x memory/compute for the clean pass.
    teacher_forcing_detach_clean_kv: bool = False

    # Chunkwise teacher forcing: number of latent frames per causal chunk.
    # The chunk partition is [1, C, C, ...] -- latent frame 0 is always its own
    # singleton chunk (mirroring the VAE's first-frame encoding), so the I2V
    # conditioning frame stays pure clean causal context.
    # Currently implemented for the replayed teacher-forcing path only.
    teacher_forcing_frames_per_chunk: int = 1

    # Probability of removing the complete control vision item from a
    # teacher-forcing transfer sample after VAE encoding. The remaining target
    # item is packed as ordinary single-target teacher forcing, which provides a
    # genuine no-control example instead of conditioning on a zero-valued map.
    # This is disabled by default and has no effect on non-transfer batches.
    teacher_forcing_transfer_control_dropout_rate: float = attrs.field(
        default=0.0,
        validator=attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.le(1.0)),
    )

    # Visibility of the clean transfer control stream. ``global_control`` is
    # backward-compatible full-clip attention. ``causal_control`` follows the
    # same [1, C, C, ...] chunk partition as target teacher forcing. The
    # ``causal_control_with_rgb_history`` mode additionally lets control queries
    # consume clean RGB from strictly earlier teacher-forcing chunks.
    transfer_control_attention_mode: Literal[
        "global_control",
        "causal_control",
        "current_only_control",
        "causal_control_with_rgb_history",
        "current_only_control_with_rgb_history",
    ] = attrs.field(
        default="global_control",
        validator=attrs.validators.in_(
            (
                "global_control",
                "causal_control",
                "current_only_control",
                "causal_control_with_rgb_history",
                "current_only_control_with_rgb_history",
            )
        ),
    )

    # Alternate bidirectional and teacher-forcing steps. Each cycle runs
    # ``moba_bidirectional_steps`` bidirectional steps followed by
    # ``moba_causal_steps`` teacher-forcing steps (default 1:1). The degenerate
    # patterns select the single-attention ablations: (1, 0) trains
    # bidirectional-only, (0, 1) trains causal-only.
    enable_moba: bool = False
    moba_bidirectional_steps: int = 1
    moba_causal_steps: int = 1

    # Whether the rolling-cache + text varlen offsets are clamped to >=1
    # and the resulting (spurious) LSE is masked to -inf via
    # MergeAttentionsBridge inside three_way_attention_with_kv_cache.
    #
    # The clamp + mask are a workaround for attention kernels that return
    # NaN (or otherwise misbehave) when given a zero-length varlen range.
    # Whether this workaround is *necessary* is BACKEND- and ARCHITECTURE-
    # specific:
    #   - fp32 FA varlen kernels need it (return NaN for empty ranges).
    #   - bf16 FA varlen kernels may not need it (return
    #     out=0, lse=-inf cleanly).
    #   - Other combinations: unknown — verify before disabling.
    #
    # When False, the clamp and the torch.where + bridge are skipped,
    # which is slightly faster.
    #
    # Default is ``True`` (safe / conservative).  To check whether your
    # backend / architecture combination actually needs the workaround,
    # run ``cosmos_framework/model/generator/mot/three_way_attention_test.py``
    # with the env var override ``CLAMP_EMPTY_VARLEN_KV=false``; if all
    # tests pass on your platform you can safely set this to False.
    #
    # ``None`` preserves the older auto-detection (True for fp32, False
    # for bf16); kept for backwards compatibility.
    clamp_empty_varlen_kv: bool = True

    # Match rCM defaults: keep grad clipping available for opt-in runs, but
    # do not bound gradients unless explicitly enabled.
    grad_clip: bool = False


def _iter_ar_chunk_ranges(start_frame: int, num_frames: int, chunk_size: int):
    """Yield ``[chunk_start, chunk_end)`` ranges over ``[start_frame, num_frames)``.

    Ranges align to the chunkwise teacher-forcing partition ``[1, C, C, ...]``
    (``C = chunk_size``): latent frame 0 is always its own singleton chunk
    (matching the training first-frame pin so I2V conditioning stays a clean
    singleton), then ``C``-frame chunks ``[1, 1+C), [1+C, 1+2C), ...``. Ranges
    are clipped to ``num_frames``; if ``start_frame`` lands mid-partition (e.g. a
    multi-frame I2V prefix), the first emitted chunk is the partial tail of that
    partition chunk. ``chunk_size == 1`` yields one frame per range (framewise).
    """
    f = start_frame
    while f < num_frames:
        if f == 0:
            chunk_end = 1
        else:
            chunk_end = 1 + ((f - 1) // chunk_size + 1) * chunk_size
        chunk_end = min(chunk_end, num_frames)
        yield f, chunk_end
        f = chunk_end


def _validate_kv_cache_dtype_supports_cuda_graphs(kv_cache_dtype: str | None, cuda_graph_path_active: bool) -> None:
    """Reject FP8 KV cache when the cuda-graph AR path is effectively active.

    The FP8 KV cache backend is only supported on the dynamic AR read path;
    the static cuda-graph path does not support it, so this guard rejects the
    combination instead of silently producing wrong results.
    ``cuda_graph_path_active`` must be the effective flag (compile.enabled and
    compile.use_cuda_graphs): with compile disabled the run falls back to the
    dynamic path, which is FP8-safe and must not be blocked. Kept as a
    module-level function so the guard is unit-testable on CPU without the GPU
    AR inference path.
    """
    validate_kv_cache_dtype(kv_cache_dtype)
    if kv_cache_dtype is not None and cuda_graph_path_active:
        raise NotImplementedError(
            f"kv_cache_dtype={kv_cache_dtype!r} is not supported together with cuda graphs "
            "(compile.enabled and compile.use_cuda_graphs). The FP8 KV cache "
            "is only supported on the dynamic AR read path. Set "
            "kv_cache_dtype=None or disable cuda graphs."
        )


def _validate_attention_sink_config(kv_cache_inference_size: int | None, attention_sink_size: int) -> None:
    """Validate AR-inference attention-sink cache settings."""
    if attention_sink_size < 0:
        raise ValueError(f"attention_sink_size must be >= 0, got {attention_sink_size}")
    if kv_cache_inference_size is None:
        if attention_sink_size != 0:
            raise ValueError("attention_sink_size must be 0 when kv_cache_inference_size is None")
    elif attention_sink_size >= kv_cache_inference_size:
        raise ValueError(
            "attention_sink_size must be less than kv_cache_inference_size, "
            f"got {attention_sink_size}>={kv_cache_inference_size}"
        )


class OmniMoTCausalModel(OmniMoTModel):
    """Mixture of Transformers model with causal (AR) generation capabilities.

    Extends ``OmniMoTModel`` with autoregressive video generation and
    teacher-forcing training backends. Non-causal (base) training and
    inference are inherited unchanged from the parent.
    """

    config: OmniMoTCausalModelConfig

    # Set while a bidirectional MoBA step or dense MoBA inference runs. Read by
    # pre_noise_memory_hook and denoise.
    _bidirectional_step_active: bool = False

    def __init__(self, config: OmniMoTCausalModelConfig):
        super().__init__(config)
        if self.config.enable_moba:
            self._validate_moba_config()

    @override
    def set_up_scheduler_and_sampler(self) -> None:
        """Install interactive-only RF scheduler extensions without touching VFM."""
        train_time_weight = self.config.rectified_flow_training_config.train_time_weight
        if train_time_weight != "gaussian_bell":
            super().set_up_scheduler_and_sampler()
            return

        self.config.rectified_flow_training_config.train_time_weight = "uniform"
        try:
            super().set_up_scheduler_and_sampler()
        finally:
            self.config.rectified_flow_training_config.train_time_weight = train_time_weight

        for rf_name in (
            "rectified_flow_image",
            "rectified_flow_video",
            "rectified_flow_action",
            "rectified_flow_sound",
        ):
            rectified_flow = getattr(self, rf_name, None)
            if rectified_flow is not None:
                rectified_flow.train_time_weight = GaussianBellTrainTimeWeight(rectified_flow.noise_scheduler)

    @override
    def install_attention_dispatch(self, net: torch.nn.Module) -> None:
        """Install ``dispatch_attention_with_memory`` on every attention layer."""
        for layer in net.language_model.model.layers:
            layer.self_attn.dispatch_attention_fn = dispatch_attention_with_memory

    def maybe_convert_linears_to_nvfp4(self) -> None:
        """Convert the decoder linears to NVFP4 in place, driven by ``config.nvfp4_linears``.

        Call once after the checkpoint has loaded (DCP needs plain ``nn.Linear`` at load time;
        see the config field). No-op when unset. The ``COSMOS3_NVFP4=1`` env var is honored as a
        fallback so existing env-based runs keep working, but ``config.nvfp4_linears`` takes
        precedence and is the intended interface.
        """
        mode = resolve_legacy_nvfp4_mode(self.config.nvfp4_linears)
        if mode is None:
            return
        from cosmos_framework.model.generator.utils.nvfp4 import convert_decoder_linears_to_nvfp4

        log.info(
            f"[nvfp4] converting decoder linears -> NVFP4 {mode} (config.nvfp4_linears={self.config.nvfp4_linears!r})"
        )
        convert_decoder_linears_to_nvfp4(self, mode)

    def maybe_install_ar_post_saturation_mode(self) -> None:
        """Install optional AR post-saturation routing after checkpoint load."""
        install_ar_post_saturation_mode(self)

    @override
    def memory_init_training(
        self,
        gen_data_clean: GenerationDataClean,
        data_batch: dict[str, torch.Tensor],
        input_text_indexes: list[list[int]],
    ) -> tuple[GenerationDataClean, dict]:
        """Prepare per-step memory info for causal training.

        Optionally removes the control item from transfer samples, drops trailing
        latent frames so every clip satisfies the chunkwise-TF constraint, then
        returns the default (non-rolling) memory info. Rolling KV-cache
        (segment-based) training is not supported.
        """
        del input_text_indexes
        gen_data_clean = self._maybe_drop_teacher_forcing_transfer_control(gen_data_clean, data_batch)
        # Chunkwise TF requires every latent vision clip to satisfy (T - 1) % C == 0.
        # The dataloaders only VAE-align clip lengths (T arbitrary), so we drop the
        # trailing latent frames here -- in lockstep across all modalities -- then
        # assert the result (a safety net that should now always pass).
        gen_data_clean = self._truncate_for_chunkwise_tf(gen_data_clean)
        self._assert_chunkwise_tf_shape(gen_data_clean)

        return gen_data_clean, {
            "skip_text": False,
            "initial_temporal_offset": 0,
            "dual_kv_cache": None,
            "use_rolling_kv_cache": False,
            "frame_idx": 0,
            "segment_idx": 0,
        }

    def _maybe_drop_teacher_forcing_transfer_control(
        self,
        gen_data_clean: GenerationDataClean,
        data_batch: dict[str, Any],
    ) -> GenerationDataClean:
        """Remove the control item for a sampled teacher-forcing transfer step.

        Transfer data reaches this hook as one logical sample containing two
        flattened vision items, ``[control, target]``. Dropping the first item
        after VAE encoding preserves the normalized dataloader contract while
        making the packed sequence genuinely control-free. This hook therefore
        requires the transfer dataloader to keep ``max_samples_per_batch=1``;
        packing multiple transfer samples would require per-sample dropout.
        """
        dropout_rate = self.config.teacher_forcing_transfer_control_dropout_rate
        if dropout_rate == 0.0 or self.config.causal_training_strategy != "teacher_forcing":
            return gen_data_clean

        dataset_names_raw = data_batch.get("dataset_name")
        if isinstance(dataset_names_raw, str):
            dataset_names = [dataset_names_raw]
        elif isinstance(dataset_names_raw, (list, tuple)):
            dataset_names = [name for name in dataset_names_raw if isinstance(name, str)]
        else:
            dataset_names = []
        is_transfer_batch = bool(dataset_names) and all(
            name == "video_transfer" or name.startswith(("video_transfer_", "driving_transfer_"))
            for name in dataset_names
        )
        if not is_transfer_batch:
            return gen_data_clean

        vision_items = gen_data_clean.x0_tokens_vision
        if gen_data_clean.num_vision_items_per_sample != [2] or vision_items is None or len(vision_items) != 2:
            raise ValueError(
                "Teacher-forcing transfer control dropout requires max_samples_per_batch=1 and one logical "
                "sample with exactly [control, target] vision items."
            )

        parallel_item_lists = {
            "raw_state_vision": gen_data_clean.raw_state_vision,
            "temporal_positions_vision": gen_data_clean.temporal_positions_vision,
            "num_views_per_vision_item": gen_data_clean.num_views_per_vision_item,
        }
        for field_name, values in parallel_item_lists.items():
            if values is not None and len(values) != 2:
                raise ValueError(
                    "Teacher-forcing transfer control dropout requires two entries in every per-item field; "
                    f"{field_name} has {len(values)}."
                )

        if dropout_rate < 1.0:
            dropout_draw = torch.rand(())  # []
            if dropout_draw.item() >= dropout_rate:
                return gen_data_clean

        gen_data_clean.x0_tokens_vision = [vision_items[-1]]  # list[[B,C,T,H,W]]
        if gen_data_clean.raw_state_vision is not None:
            gen_data_clean.raw_state_vision = [gen_data_clean.raw_state_vision[-1]]  # list[[B,C,T_px,H,W]]
        if gen_data_clean.temporal_positions_vision is not None:
            gen_data_clean.temporal_positions_vision = [gen_data_clean.temporal_positions_vision[-1]]  # list[[T]]
        if gen_data_clean.num_views_per_vision_item is not None:
            gen_data_clean.num_views_per_vision_item = [gen_data_clean.num_views_per_vision_item[-1]]
        gen_data_clean.num_vision_items_per_sample = None
        gen_data_clean.control_weights = None
        return gen_data_clean

    def _is_chunkwise_tf(self) -> bool:
        """True when this step runs chunkwise teacher-forcing attention (``C > 1``)."""
        return self.config.teacher_forcing_frames_per_chunk > 1 and self.config.causal_training_strategy in (
            "teacher_forcing",
            "teacher_forcing_dcm",
        )

    def _validate_moba_config(self) -> None:
        """Validate assumptions required by MoBA training. Called once at __init__."""
        if self.config.causal_training_strategy != "teacher_forcing":
            raise ValueError("MoBA requires causal_training_strategy='teacher_forcing'")
        if any(parameter is not None for parameter in self.config.natten_parameter_list or []):
            raise ValueError("MoBA does not support sparse NATTEN parameters")
        if self.config.moba_bidirectional_steps < 0 or self.config.moba_causal_steps < 0:
            raise ValueError(
                "moba_bidirectional_steps and moba_causal_steps must be >= 0, got "
                f"{self.config.moba_bidirectional_steps} and {self.config.moba_causal_steps}"
            )
        if self.config.moba_bidirectional_steps + self.config.moba_causal_steps == 0:
            raise ValueError("moba_bidirectional_steps + moba_causal_steps must be >= 1")

    def _is_moba_bidirectional_step(self, iteration: int) -> bool:
        """True when ``iteration`` falls in the bidirectional part of the MoBA cycle.

        The cycle runs ``moba_bidirectional_steps`` bidirectional steps first, then
        ``moba_causal_steps`` teacher-forcing steps; (1, 0) and (0, 1) reduce to the
        bidirectional-only and causal-only ablations.
        """
        cycle = self.config.moba_bidirectional_steps + self.config.moba_causal_steps
        return iteration % cycle < self.config.moba_bidirectional_steps

    @override
    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Alternate bidirectional and teacher-forcing steps when MoBA is enabled."""
        if not self.config.enable_moba:
            return super().training_step(data_batch, iteration)
        bidirectional_step = self._is_moba_bidirectional_step(iteration)
        self._bidirectional_step_active = bidirectional_step
        try:
            output_batch, loss = super().training_step(data_batch, iteration)
        finally:
            self._bidirectional_step_active = False
        output_batch["bidirectional_step"] = bidirectional_step
        return output_batch, loss

    @torch.no_grad()
    @override
    def generate_samples_from_batch(self, data_batch: dict, *args: Any, **kwargs: Any) -> dict[str, list[torch.Tensor]]:
        """Run dense MoBA inference with its training-time bidirectional attention.

        Causal-only MoBA (``moba_bidirectional_steps == 0``) never trains with
        full attention, so it keeps the standard causal inference path.
        """
        if not self.config.enable_moba or self.config.moba_bidirectional_steps == 0:
            return super().generate_samples_from_batch(data_batch, *args, **kwargs)
        self._bidirectional_step_active = True
        try:
            return super().generate_samples_from_batch(data_batch, *args, **kwargs)
        finally:
            self._bidirectional_step_active = False

    @override
    def denoise(
        self,
        net: torch.nn.Module | None = None,
        data_batch_packed: PackedSequence | None = None,
        memory: MemoryState | None = None,
        video_temporal_causal: bool | None = None,
    ) -> dict:
        """Denoise, forcing full (bidirectional) attention on bidirectional steps."""
        if video_temporal_causal is None and self._bidirectional_step_active:
            video_temporal_causal = False
        return super().denoise(
            net=net,
            data_batch_packed=data_batch_packed,
            memory=memory,
            video_temporal_causal=video_temporal_causal,
        )

    def _truncate_for_chunkwise_tf(self, gen_data_clean: GenerationDataClean) -> GenerationDataClean:
        """Drop trailing frames so every clip satisfies the chunkwise-TF constraint.

        Chunkwise teacher forcing groups latent vision frames into the partition
        ``[1, C, C, ...]`` (``C = teacher_forcing_frames_per_chunk``; latent frame
        0 is its own singleton chunk), so the latent frame count ``T`` must satisfy
        ``(T - 1) % C == 0``. Dataloaders only VAE-align clip lengths (``T``
        arbitrary), so we drop up to ``C - 1`` trailing latent frames here.

        Universal across modalities: each modality is truncated independently from
        its own per-frame layout, so there is no cross-modal index bookkeeping.
        Every modality in a sample encodes the same underlying frame count ``T``
        and is rounded with the same chunk-grid rule, so the truncated tensors
        stay mutually consistent:

        * vision latents ``x0_tokens_vision[i]`` ``[B,C,T,H,W]`` -> ``keep_t`` frames
        * vision pixels  ``raw_state_vision[i]`` -> ``get_pixel_num_frames(keep_t)``
        * action latents ``x0_tokens_action[j]`` ``[(T-1)*tcf, D]`` -> ``(keep_t-1)*tcf``
          rows (``tcf`` action tokens interleaved per vision frame; latent frame 0
          is the null-action conditioning frame and is not stored, hence
          ``(T-1)*tcf`` rows). ``raw_state_action`` aliases ``x0_tokens_action``
          (action is not VAE-encoded), so it follows automatically.

        Sound is packed as a separate, non-chunked split on its own latent rate;
        it is not part of the vision-supertoken chunk grid and needs no truncation
        for chunkwise-TF correctness, so it is left untouched.

        No-op for framewise TF (``C == 1`` -> ``keep_t == T``), non-teacher-forcing
        strategies, or single-frame (image) batches.
        """
        if not self._is_chunkwise_tf() or gen_data_clean.is_image_batch or gen_data_clean.x0_tokens_vision is None:
            return gen_data_clean
        C = self.config.teacher_forcing_frames_per_chunk

        def chunk_aligned_frames(latent_t: int) -> int:
            # Largest keep_t <= latent_t with (keep_t - 1) % C == 0 (frame 0 kept).
            return 1 + C * ((latent_t - 1) // C)

        # Vision latents and their matching pixel frames.
        for i, x0 in enumerate(gen_data_clean.x0_tokens_vision):
            latent_t = x0.shape[2]
            keep_t = chunk_aligned_frames(latent_t)
            if keep_t >= latent_t:
                continue
            gen_data_clean.x0_tokens_vision[i] = x0[:, :, :keep_t].contiguous()
            rv = gen_data_clean.raw_state_vision[i] if gen_data_clean.raw_state_vision is not None else None
            if rv is not None:
                keep_pixels = self.tokenizer_vision_gen.get_pixel_num_frames(keep_t)
                # Temporal dim is -3 for both [B, C, T, H, W] and [C, T, H, W].
                gen_data_clean.raw_state_vision[i] = rv[..., :keep_pixels, :, :].contiguous()

        # Action latents: tcf rows per vision frame for frames 1..T-1 (frame 0 is
        # the null conditioning frame, not stored), i.e. (T-1)*tcf rows. Round the
        # stored real-action frame count down to the chunk grid; this matches each
        # entry's truncated vision because both round the same T with the same rule.
        if gen_data_clean.x0_tokens_action is not None:
            tcf = self.tokenizer_vision_gen.temporal_compression_factor or 1
            for j, act in enumerate(gen_data_clean.x0_tokens_action):
                if act is None:
                    continue
                rows = act.shape[-2]  # (T-1)*tcf
                if rows % tcf != 0:
                    raise ValueError(
                        f"Action latents entry {j} has {rows} rows along the temporal axis, not a multiple of "
                        f"temporal_compression_factor={tcf}; cannot chunk-align for chunkwise TF."
                    )
                keep_rows = C * ((rows // tcf) // C) * tcf  # (keep_t - 1) * tcf
                if keep_rows < rows:
                    gen_data_clean.x0_tokens_action[j] = act[..., :keep_rows, :].contiguous()

        return gen_data_clean

    def _assert_chunkwise_tf_shape(self, gen_data_clean: GenerationDataClean) -> None:
        """Assert each clip is chunkwise-TF-ready (safety net after truncation).

        Runs after ``_truncate_for_chunkwise_tf`` and should always pass; it guards
        against a clip that truncation cannot fix -- one too short to form even a
        single ``C``-frame body chunk. Requires ``(latent_T - 1) % C == 0`` and
        ``latent_T >= 1 + C`` (one singleton frame plus at least one full chunk).

        No-op for framewise TF (``C == 1``), non-teacher-forcing strategies, or
        single-frame (image) batches. The packed-sequence layout is validated
        again in ``_validate_teacher_forcing_pack``; this is the earlier, clearer
        guard on the raw vision latents.
        """
        if not self._is_chunkwise_tf() or gen_data_clean.is_image_batch or gen_data_clean.x0_tokens_vision is None:
            return
        C = self.config.teacher_forcing_frames_per_chunk
        for i, x0 in enumerate(gen_data_clean.x0_tokens_vision):
            latent_t = x0.shape[2]
            if (latent_t - 1) % C != 0 or latent_t < 1 + C:
                raise ValueError(
                    f"Chunkwise teacher forcing requires (latent_T - 1) % {C} == 0 and latent_T >= {1 + C} "
                    f"(latent frame 0 is its own singleton chunk plus at least one full chunk); got "
                    f"latent_T={latent_t} for clip {i}. Clips this short cannot be chunked -- filter them in "
                    "the dataloader."
                )

    @override
    def pre_noise_memory_hook(
        self,
        packed_sequence: PackedSequence,
        gen_data_clean: GenerationDataClean,
        memory_info: dict,
    ) -> dict:
        """Run Pass 1 of teacher forcing: clean forward to capture per-layer K/V.

        Uses a deep copy of the packed sequence so the original remains on CPU
        for the subsequent noise-addition step in ``training_step``.

        Called from OmniMotModel.training_step().
        """
        if self.config.causal_training_strategy != "teacher_forcing":
            return memory_info

        # Bidirectional (full-attention) step of the joint bidirectional/TF
        # recipe: no clean pass, no replayed K/V — the single noisy forward runs
        # with full attention (see the ``denoise`` override).
        if self._bidirectional_step_active:
            return memory_info

        self._validate_teacher_forcing_pack(packed_sequence)

        tf_memory = self._build_clean_tf_cache(
            net=self.net,
            packed_sequence=packed_sequence,
            gen_data_clean=gen_data_clean,
            memory_info=memory_info,
            detach_clean_kv=self.config.teacher_forcing_detach_clean_kv,
        )
        memory_info["_tf_memory_state"] = tf_memory
        return memory_info

    def _cast_generated_tokens_to_precision(self, packed_sequence: PackedSequence) -> None:
        """Cast generated-modality tokens to the model precision before denoising."""
        if packed_sequence.vision is not None:
            packed_sequence.vision.tokens = [token.to(dtype=self.precision) for token in packed_sequence.vision.tokens]
        if packed_sequence.action is not None and packed_sequence.action.tokens:
            packed_sequence.action.tokens = [token.to(dtype=self.precision) for token in packed_sequence.action.tokens]
        if packed_sequence.sound is not None and packed_sequence.sound.tokens:
            packed_sequence.sound.tokens = [token.to(dtype=self.precision) for token in packed_sequence.sound.tokens]

    def _build_clean_tf_cache(
        self,
        net: torch.nn.Module,
        packed_sequence: PackedSequence,
        gen_data_clean: GenerationDataClean,
        memory_info: dict,
        detach_clean_kv: bool,
    ) -> TeacherForcingMemoryState:
        """Build replayed clean K/V for one denoiser network."""
        tf_memory = self._build_tf_memory_state(
            packed_sequence=packed_sequence,
            memory_info=memory_info,
            net=net,
            detach_clean_kv=detach_clean_kv,
        )
        clean_pack = make_teacher_forcing_clean_pack(packed_sequence)
        ctx = torch.no_grad() if detach_clean_kv else contextlib.nullcontext()
        with ctx:
            clean_pack.to_cuda()
            self._cast_generated_tokens_to_precision(clean_pack)
            self.denoise(
                net=net,
                data_batch_packed=clean_pack,
                memory=tf_memory,
            )
        tf_memory.pass_number = 2
        return tf_memory

    def _validate_teacher_forcing_pack(self, packed_seq: PackedSequence) -> None:
        """Fail early for TF layouts unsupported by the replayed K/V path."""
        if not self.config.video_temporal_causal:
            raise ValueError("Teacher-forcing replay requires video_temporal_causal=True.")
        if packed_seq.vision is None or len(packed_seq.vision.token_shapes) not in (1, 2):
            num_layouts = 0 if packed_seq.vision is None else len(packed_seq.vision.token_shapes)
            raise ValueError(
                "Teacher-forcing replay supports one temporal-causal video layout, or exactly two aligned "
                f"transfer layouts (control + target); got {num_layouts}."
            )
        if len(packed_seq.vision.token_shapes) == 2:
            control_shape, target_shape = packed_seq.vision.token_shapes
            if control_shape != target_shape:
                raise ValueError(
                    "Teacher-forcing transfer requires aligned control and target token shapes; "
                    f"got {control_shape} and {target_shape}."
                )
            if packed_seq.action is not None:
                raise ValueError("Teacher-forcing transfer does not support action tokens.")
            if packed_seq.num_vision_items_per_sample != [2]:
                raise ValueError(
                    "Teacher-forcing transfer currently supports one logical sample with [control, target]; "
                    f"got num_vision_items_per_sample={packed_seq.num_vision_items_per_sample}."
                )
            if len(packed_seq.vision.condition_mask) != 2:
                raise ValueError(
                    "Teacher-forcing transfer requires one condition mask per vision item; "
                    f"got {len(packed_seq.vision.condition_mask)} masks."
                )
            if not torch.all(packed_seq.vision.condition_mask[0] == 1):  # []
                raise ValueError("Teacher-forcing transfer requires the control vision item to be fully conditioned.")
            if len(packed_seq.vision_item_split_lens) != 1 or len(packed_seq.vision_item_split_lens[0]) != 2:
                raise ValueError(
                    "Teacher-forcing transfer requires explicit [control, target] token ranges; "
                    f"got vision_item_split_lens={packed_seq.vision_item_split_lens}."
                )

    def _build_tf_memory_state(
        self,
        packed_sequence: PackedSequence,
        memory_info: dict,
        net: torch.nn.Module | None = None,
        detach_clean_kv: bool | None = None,
    ) -> TeacherForcingMemoryState:
        """Construct a ``TeacherForcingMemoryState`` for the two-pass training path."""
        net = self.net if net is None else net
        detach_clean_kv = self.config.teacher_forcing_detach_clean_kv if detach_clean_kv is None else detach_clean_kv
        vision_token_shapes = packed_sequence.vision.token_shapes if packed_sequence.vision else None
        assert vision_token_shapes is not None

        # Create a dummy (empty) KV-cache, so three_way_attention_with_kv_cache is happy.
        num_layers: int = net.num_hidden_layers  # type: ignore[attr-defined]
        dual_kv_cache = [DualKVCache(gen_cache_size=2) for _ in range(num_layers)]
        num_kv_heads = _get_context_parallel_num_kv_heads(net.num_kv_heads, self.parallel_dims)
        context_parallel_size = (
            self.parallel_dims.cp_size if self.parallel_dims and self.parallel_dims.cp_enabled else 1
        )

        return TeacherForcingMemoryState(
            vision_token_shapes=vision_token_shapes,
            num_action_tokens_per_supertoken=packed_sequence.num_action_tokens_per_supertoken,
            null_action_supertokens=packed_sequence.null_action_supertokens,
            segment_idx=0,
            dual_kv_cache=dual_kv_cache,
            num_kv_heads=num_kv_heads,
            head_dim=net.head_dim,
            detach_clean_kv=detach_clean_kv,
            clamp_empty_varlen_kv=self.config.clamp_empty_varlen_kv,
            frames_per_chunk=self.config.teacher_forcing_frames_per_chunk,
            transfer_control_attention_mode=self.config.transfer_control_attention_mode,
            context_parallel_size=context_parallel_size,
        )

    @override
    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    ):
        if not self.config.grad_clip:
            max_norm = 1e12
        return super().clip_grad_norm_(
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    @override
    def build_memory_state(
        self,
        packed_seq: PackedSequence,
        memory_info: dict,
    ) -> MemoryState | None:
        """Construct the appropriate MemoryState for the current forward pass.

        Dispatch logic:
        - Teacher forcing stashed state → return it directly (already pass 2).
        - ``dual_kv_cache`` present and ``use_ar_rolling`` →
          ``ARMemoryState(for_cuda_graphs=True)``.  Static-shape AR
          inference at frame >= 1 (compile + CG).  The und cache must
          already be populated by a frame-0 prefill that took the
          dynamic-shape ``ARMemoryState`` branch.
        - ``dual_kv_cache`` present (else) → ``ARMemoryState`` (dynamic-shape
          AR inference: frame 0 in compile + CG, plus every frame in
          compile-only and no-compile modes).
        - Otherwise → ``None``
        """

        # Returned cached TeacherForcingMemoryState from pass 1 when using TF.
        tf_state: TeacherForcingMemoryState | None = memory_info.get("_tf_memory_state")
        if tf_state is not None:
            return tf_state

        use_ar_rolling: bool = memory_info.get("use_ar_rolling", False)
        post_saturation_static_compile: bool = memory_info.get("post_saturation_static_compile", False)
        coarse_cuda_graph: bool = memory_info.get("coarse_cuda_graph", False)
        stage_gen_cache_writes: bool = memory_info.get("stage_gen_cache_writes", False)
        dual_kv_cache: list[DualKVCache] | None = memory_info["dual_kv_cache"]
        frame_idx: int = memory_info["frame_idx"]
        write_gen_cache: bool = memory_info.get("write_gen_cache", True)
        replicated_attention_io_cp = (
            self.config.attention_io_layout == "replicated"
            and self.parallel_dims is not None
            and self.parallel_dims.cp_enabled
        )
        kv_head_shard_rank = self.parallel_dims.cp_rank if replicated_attention_io_cp else 0
        kv_head_shard_size = self.parallel_dims.cp_size if replicated_attention_io_cp else 1

        # Inference with a KV-cache.
        if dual_kv_cache is not None:
            vision_token_shapes = packed_seq.vision.token_shapes if packed_seq.vision else None
            if use_ar_rolling:
                # Static-shape AR inference at frame >= 1 (compile + CG).
                # ``for_cuda_graphs=True`` makes ``read_for_layer`` return
                # the full preallocated gen buffer + a real-length scalar
                # tensor instead of a sized history slice, so every input
                # to the compiled layer has a constant shape across frames
                # and a single CUDA-graph capture replays.
                assert vision_token_shapes is not None
                assert frame_idx > 0, (
                    "use_ar_rolling requires frame_idx >= 1 (frame 0 must take the dynamic-shape "
                    "ARMemoryState branch — for_cuda_graphs=False — to populate the und cache)"
                )
                return ARMemoryState(
                    dual_kv_cache=dual_kv_cache,
                    frame_idx=frame_idx,
                    vision_token_shapes=vision_token_shapes,
                    num_action_tokens_per_supertoken=packed_seq.num_action_tokens_per_supertoken,
                    null_action_supertokens=packed_seq.null_action_supertokens,
                    for_cuda_graphs=True,
                    num_kv_heads=self.net.num_kv_heads,
                    head_dim=self.net.head_dim,
                    write_gen_cache=write_gen_cache,
                    kv_head_shard_rank=kv_head_shard_rank,
                    kv_head_shard_size=kv_head_shard_size,
                )
            return ARMemoryState(
                dual_kv_cache=dual_kv_cache,
                frame_idx=frame_idx,
                vision_token_shapes=vision_token_shapes,
                num_action_tokens_per_supertoken=packed_seq.num_action_tokens_per_supertoken,
                null_action_supertokens=packed_seq.null_action_supertokens,
                num_kv_heads=self.net.num_kv_heads if replicated_attention_io_cp or coarse_cuda_graph else None,
                head_dim=self.net.head_dim if replicated_attention_io_cp or coarse_cuda_graph else None,
                write_gen_cache=write_gen_cache,
                kv_head_shard_rank=kv_head_shard_rank,
                kv_head_shard_size=kv_head_shard_size,
                post_saturation_static_compile=post_saturation_static_compile,
                static_und_cache_max_len=self.config.ar_static_und_cache_max_len,
                coarse_cuda_graph=coarse_cuda_graph,
                stage_gen_cache_writes=stage_gen_cache_writes,
            )

        # If not using a KV-cache.
        return None

    @staticmethod
    def _first_action_domain_id(gen_data_clean: GenerationDataClean) -> torch.Tensor | None:
        """Return the first action-domain id for batch-size-one AR packing."""
        action_domain_id = getattr(gen_data_clean, "action_domain_id", None)
        if action_domain_id is None or len(action_domain_id) == 0:
            return None
        return action_domain_id[0]  # [1] or []

    @staticmethod
    def _first_raw_action_dim(gen_data_clean: GenerationDataClean) -> torch.Tensor | None:
        """Return the first raw action width for batch-size-one AR packing."""
        raw_action_dims = getattr(gen_data_clean, "raw_action_dim", None)
        if raw_action_dims is None or len(raw_action_dims) == 0:
            return None
        raw_action_dim = raw_action_dims[0]
        return raw_action_dim if raw_action_dim is not None else None  # [] or None

    # ------------------------------------------------------------------
    # Autoregressive generation (moved from OmniMoTModel)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def iter_samples_from_batch_autoregressive(
        self,
        data_batch: dict,
        # "text2image", "text2video", "image2video", "forward_dynamics",
        # "text2video_action_conditioned", or "video_transfer".
        mode: str,
        guidance: float = 1.5,
        seed: int = 1,
        num_steps: int = 35,
        shift: float = 5.0,
        normalize_cfg: bool = False,
        sampler_mode: str = "rf",
        distilled_num_steps: int | None = None,
        sync_num_frames_across_ranks: bool = False,
        sync_process_group: Any | None = None,
        prefix_frame_count: int | None = None,
        max_num_frames: int | None = None,
        on_clean_vision_chunk: Callable[[torch.Tensor], None] | None = None,
        has_negative_prompt: bool = False,
        **kwargs,
    ) -> Generator[dict[str, Any], torch.Tensor | None, None]:
        """
        Generate video frames autoregressively (AR) with KV caching.

        Each latent frame is denoised independently.  After denoising, the frame's
        K/V projections are stored in ``DualKVCache`` so subsequent frames can attend
        to the full temporal history without recomputing it.

        **batch_size constraint**: This function asserts ``batch_size == 1``.
        One ``DualKVCache`` is allocated per transformer layer; all
        internal text-token and action accesses are hardcoded to index ``[0]``.
        Multi-sample generation is achieved by calling this function once per sample
        (with ``seed + i``) from the inference script.

        **KV cache seeding**:
        - ``text2image`` / ``text2video`` (``start_frame=0``): text K/V are stored during
          the first AR iteration (``frame_idx=0``), which includes text tokens in its pack.
        - ``image2video`` / ``forward_dynamics`` (``start_frame=1``): a prefill
          ``denoise`` call at ``frame_idx=0`` seeds the cache with text K/V and the
          conditioned frame 0 vision K/V *before* the AR loop begins.

        Supported modes
        ---------------
        ``text2image``
            Sequence: ``[text] [v0_noisy]``  Single frame denoised (text2video with 1 frame).
        ``text2video``
            Sequence: ``[text] [v0_noisy] [v1_noisy] ...``  All frames denoised.
        ``image2video``
            Sequence: ``[text] [v0*] [v1_noisy] ...``  Frame 0 conditioned.
        ``text2video_action_conditioned``
            Sequence: ``[text] [a_null* v0_noisy] [a0* v1_noisy] ...``  Frame 0
            is generated from text, then actions control later transitions.
        ``forward_dynamics``
            Sequence: ``[text] [v0*] [a0* v1_noisy] [a1* v2_noisy] ...``
            Action ``a{i}`` (shape ``(tcf, D)``) guides the transition ``v{i} → v{i+1}``.
        ``video_transfer``
            Sequence: ``[text] [control_chunk0*] [RGB_chunk0] ...``. The
            current control chunk and clean RGB from prior chunks are cached in
            dependency order for ``causal_control_with_rgb_history``.

        Args:
            data_batch: Raw data batch from dataloader.
            mode: Generation mode — ``"text2image"``, ``"text2video"``, ``"image2video"``,
                or ``"forward_dynamics"``.
            guidance: CFG guidance weight. When ``guidance == 1.0``, the
                unconditional branch is skipped (no extra forward pass).
            seed: Base random seed; each latent frame uses ``seed + frame_idx``.
            num_steps: Number of denoising steps per latent frame.
            shift: Time-shift parameter for the flow scheduler.
            normalize_cfg: Use normalized CFG interpolation instead of standard.
            sampler_mode: ``"rf"`` uses the configured RF sampler; ``"distilled"`` uses
                the configured few-step schedule directly for causal distillation.
            distilled_num_steps: Optional debug truncation for distilled schedules.
            sync_num_frames_across_ranks: Truncate to the distributed minimum latent
                frame count so all ranks issue the same FSDP forward sequence.
            sync_process_group: Optional process group used for rank synchronization.
            prefix_frame_count: Optional fixed number of clean prefix frames for
                ``image2video`` callback sampling.
            max_num_frames: Optional local cap on latent frames to generate.
            on_clean_vision_chunk: Optional callback invoked exactly once for each clean
                output chunk of shape ``[1,C,T,H,W]``, including conditioned prefixes,
                in temporal order.
            has_negative_prompt: Use ``neg_<caption_key>`` from ``data_batch`` for the
                unconditional CFG branch instead of an empty caption.

        Yields:
            dict payloads with:
                - ``"vision"``: one latent frame, shape ``(1, C, 1, H, W)``.
                - ``"action"``: action latents (``forward_dynamics`` only).
        """
        valid_modes = [
            "text2image",
            "text2video",
            "image2video",
            "forward_dynamics",
            "text2video_action_conditioned",
            "video_transfer",
        ]
        assert mode in valid_modes, (
            f"Invalid mode: {mode}. Must be one of: text2image, text2video, image2video, "
            "forward_dynamics, text2video_action_conditioned, video_transfer"
        )
        if sampler_mode not in ("rf", "distilled"):
            raise ValueError(f"sampler_mode must be 'rf' or 'distilled', got {sampler_mode!r}")
        reset_ar_post_saturation_runtime_for_generation(self)

        # Step 1: Prepare data and build sequence plans
        condition_frame_indexes: list[int] = []
        if mode == "image2video":
            if prefix_frame_count is not None:
                if prefix_frame_count < 1:
                    raise ValueError(f"prefix_frame_count must be >= 1, got {prefix_frame_count}")
                condition_frame_indexes = list(range(prefix_frame_count))
            else:
                condition_frame_indexes = condition_frame_indexes_vision_from_batch(data_batch) or [0]
        elif mode == "forward_dynamics":
            condition_frame_indexes = condition_frame_indexes_vision_from_batch(data_batch)
        gen_data_clean = self.get_data_and_condition(
            data_batch,
            vision_condition_indexes=[condition_frame_indexes] if condition_frame_indexes else None,
        )

        batch_size = gen_data_clean.batch_size
        vision_items = gen_data_clean.x0_tokens_vision
        if vision_items is None:
            raise ValueError(f"{mode} AR inference requires vision latents")
        is_transfer = mode == "video_transfer"
        generation_vision_item_idx = 0
        if is_transfer:
            if self.config.transfer_control_attention_mode != "causal_control_with_rgb_history":
                raise ValueError(
                    "This inference-only transfer path supports "
                    "transfer_control_attention_mode='causal_control_with_rgb_history'; "
                    f"got {self.config.transfer_control_attention_mode!r}."
                )
            if self.parallel_dims is not None and getattr(self.parallel_dims, "cp_enabled", False):
                raise ValueError("video_transfer AR inference requires context_parallel_size=1.")
            if gen_data_clean.num_vision_items_per_sample != [2] or len(vision_items) != 2:
                raise ValueError(
                    "video_transfer AR inference requires one logical sample with [control, target] vision items; "
                    f"got num_vision_items_per_sample={gen_data_clean.num_vision_items_per_sample} and "
                    f"{len(vision_items)} flattened items."
                )
            if vision_items[0].shape != vision_items[1].shape:
                raise ValueError(
                    "video_transfer AR inference requires aligned control and target latent shapes; "
                    f"got {tuple(vision_items[0].shape)} and {tuple(vision_items[1].shape)}."
                )
            if gen_data_clean.x0_tokens_action is not None:
                raise ValueError("video_transfer AR inference does not support action tokens")
            generation_vision_item_idx = 1

        num_frames = vision_items[generation_vision_item_idx].shape[2]
        if max_num_frames is not None:
            if max_num_frames < 1:
                raise ValueError(f"max_num_frames must be >= 1, got {max_num_frames}")
            num_frames = min(num_frames, max_num_frames)
        if sync_num_frames_across_ranks and dist.is_available() and dist.is_initialized():
            frame_count = torch.tensor(
                [num_frames],
                dtype=torch.long,
                device=vision_items[generation_vision_item_idx].device,
            )  # [1]
            dist.all_reduce(frame_count, op=dist.ReduceOp.MIN, group=sync_process_group)
            synced_num_frames = int(frame_count.item())
            if synced_num_frames < 1:
                raise ValueError(f"Synchronized AR frame count must be >= 1, got {synced_num_frames}")
            if synced_num_frames != num_frames:
                log.info(f"[AR inference] truncating local latent frames {num_frames} -> {synced_num_frames}")
            num_frames = synced_num_frames

        assert batch_size == 1, "AR generation currently only supports batch_size=1"

        # Step 2: Tokenize text once (conditional + unconditional for CFG)
        cond_text_tokens, uncond_text_tokens = self._get_inference_text_tokens(data_batch, has_negative_prompt)

        # Frames N>=1 skip text (cached in und_cache), so they must pre-seed the mRoPE offset
        # that frame 0 reached right before packing vision. We pass the raw cached text length
        # here; ``pack_input_sequence_autoregressive`` adds the modality margin internally so
        # the convention is uniform across frames. Cond and uncond captions differ in length,
        # so compute both.
        _margin = self.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin
        cond_cached_text_offset = compute_text_split_length(
            len(cond_text_tokens[0]) if cond_text_tokens is not None else 0,
            self.llm_special_tokens,
            has_generation=True,
        )
        uncond_cached_text_offset = compute_text_split_length(
            len(uncond_text_tokens[0]) if uncond_text_tokens is not None else 0,
            self.llm_special_tokens,
            has_generation=True,
        )

        # Broadcast seed across CFGP group so both ranks sample the same noise each frame.
        cfgp_enabled = self.parallel_dims is not None and self.parallel_dims.cfgp_enabled
        if cfgp_enabled:
            seed = _broadcast_seed([seed], self.parallel_dims.cfgp_mesh.get_group(), self.parallel_dims.cfgp_rank)[0]

        # CFG is active whenever guidance != 1.0 (the uncond branch contributes)
        # or CFGP is enabled (rank 1 always runs the uncond branch).
        cfg_active = guidance != 1.0 or cfgp_enabled
        if uses_ar_post_saturation_static_compile(self):
            validate_ar_static_und_cache_lengths(
                cond_text_tokens=cond_text_tokens[0] if cond_text_tokens else None,
                uncond_text_tokens=uncond_text_tokens[0] if uncond_text_tokens else None,
                cfg_active=cfg_active,
                special_tokens=self.llm_special_tokens,
                max_s_und=self.config.ar_static_und_cache_max_len,
            )

        # Step 3: Initialize KV caches.
        # With CFGP: each rank owns one cache (rank 0 = cond, rank 1 = uncond); dual_kv_cache_uncond unused.
        # Without CFGP: allocate a second cache for the sequential uncond branch when CFG is active.
        num_layers: int = self.net.num_hidden_layers  # type: ignore
        # When ``torch.compile`` + CUDA graphs is enabled, the AR loop
        # at frame >= 1 reads cached gen K/V via the static-shape path
        # (``ARMemoryState(for_cuda_graphs=True)`` →
        # ``gen_cache.fetch_kv_padded``) so a single CUDA graph can be
        # captured for the rest of the video.  The captured
        # ``attention_AR_gen_only`` calls ``attention()`` with pre-built
        # varlen offsets (``cu_seqlens_q_t`` / ``cu_seqlens_kv_t``) so
        # the K tensor shape stays static at the padded max while only
        # ``cu_seqlens_kv_t[1]``'s *value* changes per frame.
        # Frame 0 — whether the i2v/forward_dynamics prefill or the
        # t2v/t2i first AR step — always runs eagerly via the
        # dynamic-shape ``ARMemoryState`` so the und cache is populated
        # cheaply.  In eager and compile-only mode (no CG) frame >= 1
        # also runs the dynamic-shape path: text tokens are dropped
        # after frame 0 and ``attention_AR_gen_only`` is a single
        # ``cat``-based dense ``attention()`` call (no varlen kwargs).
        # ``use_ar_rolling_path`` enables the static-shape AR path
        # (``ARMemoryState(for_cuda_graphs=True)``) at frame >= 1; gated
        # by the velocity_fn on ``frame_idx > 0`` so frame 0 stays on
        # the dynamic-shape branch.
        use_ar_rolling_path = (
            self.config.compile.enabled
            and self.config.compile.use_cuda_graphs
            and self.config.compile.ar_post_saturation_mode == "default"
        )
        if is_transfer and self.config.compile.enabled:
            raise ValueError("video_transfer AR inference requires eager attention; run with --no-use-torch-compile.")
        _validate_kv_cache_dtype_supports_cuda_graphs(self.config.kv_cache_dtype, use_ar_rolling_path)

        # Chunkwise AR: generate ``chunk_size`` latent frames jointly per AR step
        # (chunk_size == 1 is the original framewise path). Must match the
        # chunkwise teacher-forcing training value so inference attends with the
        # same receptive field it was trained for.
        chunk_size = self.config.teacher_forcing_frames_per_chunk
        if chunk_size > 1 and use_ar_rolling_path:
            raise ValueError(
                "Chunkwise AR inference (teacher_forcing_frames_per_chunk > 1) is only supported on the "
                "eager dynamic-shape path; run with --no-use-torch-compile (compile.use_cuda_graphs off)."
            )
        if is_transfer and (num_frames < 1 + chunk_size or (num_frames - 1) % chunk_size != 0):
            raise ValueError(
                "video_transfer AR inference must match the trained [1, C, C, ...] latent partition: "
                f"require T >= 1 + C and (T - 1) % C == 0, got T={num_frames}, C={chunk_size}."
            )

        # The static-shape compile-safe path reads via ``fetch_kv_padded``
        # / ``get_padded``, which pad to a fixed size derived from
        # ``gen_cache_size``.  That size must be finite and large enough
        # to hold the entire video.  When the config leaves it unset
        # (None = "unbounded" in eager mode), default to num_frames — but
        # only when the compile-safe path is actually in use.
        _validate_attention_sink_config(self.config.kv_cache_inference_size, self.config.attention_sink_size)
        if is_transfer and self.config.kv_cache_inference_size is not None:
            raise ValueError("video_transfer AR inference requires kv_cache_inference_size=None for full history.")
        if is_transfer and self.config.kv_cache_dtype is not None:
            raise ValueError("video_transfer AR inference requires kv_cache_dtype=None for exact cache values.")
        gen_cache_size = 2 * num_frames + 1 if is_transfer else self.config.kv_cache_inference_size
        if gen_cache_size is None and use_ar_rolling_path:
            gen_cache_size = num_frames
        dual_kv_cache = [
            DualKVCache(
                gen_cache_size=gen_cache_size,
                kv_cache_dtype=self.config.kv_cache_dtype,
                kv_cache_kernel_impl=self.config.kv_cache_kernel_impl,
                attention_sink_size=self.config.attention_sink_size,
            )
            for _ in range(num_layers)
        ]
        dual_kv_cache_uncond = (
            [
                DualKVCache(
                    gen_cache_size=gen_cache_size,
                    kv_cache_dtype=self.config.kv_cache_dtype,
                    kv_cache_kernel_impl=self.config.kv_cache_kernel_impl,
                    attention_sink_size=self.config.attention_sink_size,
                )
                for _ in range(num_layers)
            ]
            if cfg_active and not cfgp_enabled
            else None
        )

        # Step 4: Convert FPS tensors to list[float] for packing functions
        fps_vision_list = gen_data_clean.fps_vision.tolist() if gen_data_clean.fps_vision is not None else [24.0]
        fps_action_list = gen_data_clean.fps_action.tolist() if gen_data_clean.fps_action is not None else [24.0]
        action_domain_id = OmniMoTCausalModel._first_action_domain_id(gen_data_clean)
        raw_action_dim = OmniMoTCausalModel._first_raw_action_dim(gen_data_clean)

        # Step 5: Initialize storage for generated frames
        packed_seq = None  # Will be created for frame 0
        packed_seq_uncond = None  # Will be created when cfg_active
        next_streamed_action: torch.Tensor | None = None

        # Step 6: Determine starting frame and initial vision latent based on mode.
        prefix_vision_latents: list[tuple[int, torch.Tensor]] = []
        if mode in ("text2image", "text2video"):
            start_frame = 0
            has_action = False
            initial_vision_latent = None  # No vision yet
        elif mode == "text2video_action_conditioned":
            start_frame = 0
            has_action = True
            initial_vision_latent = None  # Frame 0 is generated with null action slots.
        elif mode == "video_transfer":
            start_frame = 0
            has_action = False
            initial_vision_latent = None  # Target frame 0 is generated after control chunk 0 is cached.
        elif mode == "image2video":
            has_action = False
            if prefix_frame_count is not None:
                condition_frame_indexes = condition_frame_indexes[:num_frames]
            expected_prefix = list(range(len(condition_frame_indexes)))
            if condition_frame_indexes != expected_prefix:
                raise ValueError(
                    "image2video AR inference only supports contiguous prefix conditioning from frame 0; "
                    f"got condition_frame_indexes_vision={condition_frame_indexes}."
                )
            if condition_frame_indexes[-1] >= num_frames:
                raise ValueError(
                    f"condition_frame_indexes_vision={condition_frame_indexes} exceeds latent frame count {num_frames}."
                )
            start_frame = len(condition_frame_indexes)
            for prefix_frame_idx in condition_frame_indexes:
                prefix_vision_latent = gen_data_clean.x0_tokens_vision[0][
                    :, :, prefix_frame_idx : prefix_frame_idx + 1
                ].to(**self.tensor_kwargs)  # [1,C,1,H,W]
                prefix_vision_latents.append((prefix_frame_idx, prefix_vision_latent))
                if on_clean_vision_chunk is not None:
                    on_clean_vision_chunk(prefix_vision_latent)
                yield {"vision": prefix_vision_latent}  # Emit conditioned prefix frames immediately for streaming
            initial_vision_latent = prefix_vision_latents[0][1]  # [1,C,1,H,W]
        elif mode == "forward_dynamics":
            start_frame = 1
            has_action = True
            initial_vision_latent = gen_data_clean.x0_tokens_vision[0][:, :, 0:1, :, :].to(
                **self.tensor_kwargs
            )  # [1,C,1,H,W]
            prefix_vision_latents.append((0, initial_vision_latent))
            if on_clean_vision_chunk is not None:
                on_clean_vision_chunk(initial_vision_latent)
            # Emit conditioned frame 0 immediately. If the caller resumes with
            # .send(action), this switches to streaming; if it resumes with
            # next()/.send(None), the preloaded action file is used.
            next_streamed_action = yield {"vision": initial_vision_latent}
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # next() resumes this yield with None, so None means "use preloaded
        # actions"; only an actual sent tensor switches the generator to
        # indefinite streaming.
        streaming_actions = next_streamed_action is not None
        if has_action and not streaming_actions and gen_data_clean.x0_tokens_action is None:
            raise ValueError(
                "forward_dynamics and text2video_action_conditioned modes require either preloaded action data "
                "or sent action tensors"
            )

        # Common packing params (reused across all pack_input_sequence_autoregressive calls)
        _tcf: int = self.tokenizer_vision_gen.temporal_compression_factor or 4
        _patch_size: int = self.config.diffusion_expert_config.patch_spatial
        _action_dim: int = self.config.max_action_dim
        _video_tc: bool = self.config.video_temporal_causal
        _enable_fps_mod: bool = self.config.diffusion_expert_config.enable_fps_modulation
        _base_fps: float = float(self.config.diffusion_expert_config.base_fps)
        transfer_history_cache_idx = 0

        # Step 7: Prefill KV cache with text + conditioned prefix frames (image2video / forward_dynamics only).
        # For text2video (start_frame=0) the AR loop seeds the cache at frame_idx=0.
        # For modes where start_frame>0, the AR loop never sees the clean prefix, so we must run
        # forward passes here to store text K/V and prefix vision K/V before the loop begins.
        # The prefill always takes the dynamic-shape ``ARMemoryState`` path.
        # The compile + CG region is the AR loop at frame >= 1, which uses the
        # static-shape ``ARMemoryState(for_cuda_graphs=True)``.  Frame 0 is
        # one-shot, so static-shape constraints aren't needed
        if mode in ("image2video", "forward_dynamics"):
            assert initial_vision_latent is not None
            for prefix_frame_idx, prefix_vision_latent in prefix_vision_latents:
                include_text = prefix_frame_idx == 0
                self._seed_frame_into_kv_cache(
                    frame_latent=prefix_vision_latent,
                    frame_idx=prefix_frame_idx,
                    use_ar_rolling_path=use_ar_rolling_path,
                    dual_kv_cache=dual_kv_cache,
                    dual_kv_cache_uncond=dual_kv_cache_uncond,
                    cond_text_tokens=cond_text_tokens[0] if (cond_text_tokens and include_text) else None,
                    uncond_text_tokens=uncond_text_tokens[0] if (uncond_text_tokens and include_text) else None,
                    cond_cached_text_offset=cond_cached_text_offset,
                    uncond_cached_text_offset=uncond_cached_text_offset,
                    curr_action_latent=None,
                    gen_data_clean=gen_data_clean,
                    fps_vision_list=fps_vision_list,
                    fps_action_list=fps_action_list,
                    seed=seed,
                    cfg_active=cfg_active,
                    cfgp_enabled=cfgp_enabled,
                    tcf=_tcf,
                    patch_size=_patch_size,
                    action_dim=_action_dim,
                    video_tc=_video_tc,
                    enable_fps_mod=_enable_fps_mod,
                    base_fps=_base_fps,
                    modality_margin=_margin,
                )

        # Step 8: AR Generation Loop (chunkwise; chunk_size == 1 is framewise).
        # Chunks follow the training partition [1, C, C, ...] in absolute latent-
        # frame index: frame 0 is its own singleton chunk, then C-frame chunks.
        # Each chunk's frames are denoised jointly -- attention_AR_gen_only uses
        # is_causal=False over the current unit, so the chunk's tokens attend
        # bidirectionally among themselves and fully to all previously generated
        # clean frames in the KV cache (strictly-earlier chunks), matching the
        # chunkwise teacher-forcing receptive field.
        # Streaming runs until the caller closes the generator (framewise only).
        if streaming_actions and chunk_size > 1:
            raise ValueError(
                "Chunkwise AR inference (teacher_forcing_frames_per_chunk > 1) does not support streamed "
                "actions; use a preloaded action file or chunk_size == 1."
            )
        chunk_iter = (
            ((_f, _f + 1) for _f in itertools.count(start_frame))
            if streaming_actions and has_action
            else _iter_ar_chunk_ranges(start_frame, num_frames, chunk_size)
        )
        for chunk_start, chunk_end in chunk_iter:
            chunk_len = chunk_end - chunk_start
            if is_transfer:
                include_text = transfer_history_cache_idx == 0
                control_latent = vision_items[0][:, :, chunk_start:chunk_end].to(
                    **self.tensor_kwargs
                )  # [1,C,T_control,H,W]
                self._seed_frame_into_kv_cache(
                    frame_latent=control_latent,
                    frame_idx=transfer_history_cache_idx,
                    dual_kv_cache=dual_kv_cache,
                    dual_kv_cache_uncond=dual_kv_cache_uncond,
                    cond_text_tokens=cond_text_tokens[0] if (cond_text_tokens and include_text) else None,
                    uncond_text_tokens=uncond_text_tokens[0] if (uncond_text_tokens and include_text) else None,
                    cond_cached_text_offset=cond_cached_text_offset,
                    uncond_cached_text_offset=uncond_cached_text_offset,
                    curr_action_latent=None,
                    gen_data_clean=gen_data_clean,
                    fps_vision_list=fps_vision_list,
                    fps_action_list=fps_action_list,
                    seed=seed,
                    cfg_active=cfg_active,
                    cfgp_enabled=cfgp_enabled,
                    tcf=_tcf,
                    patch_size=_patch_size,
                    action_dim=_action_dim,
                    video_tc=_video_tc,
                    enable_fps_mod=_enable_fps_mod,
                    base_fps=_base_fps,
                    modality_margin=_margin,
                    use_ar_rolling_path=use_ar_rolling_path,
                    position_frame_idx=chunk_start,
                    condition_frame_indexes_vision=list(range(chunk_len)),
                )
                transfer_history_cache_idx += 1
            if not dist.is_initialized() or dist.get_rank() == 0:
                log.info(f"[AR inference] frames [{chunk_start}, {chunk_end}) / {num_frames}")
            # Action prefix for the chunk (forward_dynamics): latent frame f is
            # driven by action a_{f-1}, so concatenate a_{chunk_start-1}..a_{chunk_end-2}
            # (tcf tokens each) into one (chunk_len*tcf, D) prefix.
            if has_action and chunk_start > 0:
                if streaming_actions:
                    # chunk_len == 1 under streaming (chunk_size > 1 is rejected above).
                    if next_streamed_action is None:
                        raise ValueError(f"Missing streamed action tensor for frame_idx={chunk_start}")
                    if tuple(next_streamed_action.shape) != (_tcf, _action_dim):
                        raise ValueError(
                            f"Expected streamed action tensor shape {(_tcf, _action_dim)}, "
                            f"got {tuple(next_streamed_action.shape)}"
                        )
                    curr_action_latent = next_streamed_action.to(
                        device=self.tensor_kwargs["device"],
                        dtype=torch.float32,
                    )  # [tcf,D]
                else:
                    assert gen_data_clean.x0_tokens_action is not None
                    # x0_tokens_action is a dense list; batch_size==1 so take the single entry and slice.
                    curr_action_latent = gen_data_clean.x0_tokens_action[0][
                        (chunk_start - 1) * _tcf : (chunk_end - 1) * _tcf, :
                    ].to(**self.tensor_kwargs)  # a_{chunk_start-1}..a_{chunk_end-2}; [chunk_len*tcf, D]
            else:
                curr_action_latent = None

            # Initialize the whole chunk with noise: [1, C, chunk_len, H, W].
            _ref0 = vision_items[generation_vision_item_idx][:, :, 0:1, :, :].to(**self.tensor_kwargs)  # [1,C,1,H,W]
            _g = torch.Generator(device=_ref0.device).manual_seed(seed + chunk_start)
            curr_vision_latent = torch.empty(
                (1, _ref0.shape[1], chunk_len, _ref0.shape[3], _ref0.shape[4]),
                device=_ref0.device,
                dtype=_ref0.dtype,
            ).normal_(generator=_g)  # [1,C,chunk_len,H,W]

            # Per-frame pack: only project Q/K/V for the new frame, not the whole
            # history.  Drop caption tokens for every frame >= 1 — the
            # dynamic-shape ``ARMemoryState`` (eager / compile-no-CG) and the
            # static-shape ``ARMemoryState(for_cuda_graphs=True)`` (compile + CG)
            # both read und K/V from the cache populated at frame 0.  Removing
            # the und tokens from the pack at frame >= 1 also keeps the und
            # length at 0 inside the gen-only branch of
            # ``MoTDecoderLayer.forward``, so RoPE shapes stay constant across
            # frames.  When text is omitted, pre-seed the mRoPE offset
            # (cached_text_offset) so vision Q matches the cached und K; when
            # text is in the pack (frame 0), pass None so the pack itself
            # advances the offset through text + margin.
            include_text_in_pack = chunk_start == 0 and not is_transfer
            _cond_pack_text = cond_text_tokens[0] if (cond_text_tokens and include_text_in_pack) else None
            packed_seq = pack_input_sequence_autoregressive(
                vision_latent=curr_vision_latent,
                action_latent=curr_action_latent,
                text_tokens=_cond_pack_text,
                timestep=0.0,
                fps_vision=fps_vision_list,
                fps_action=fps_action_list if curr_action_latent is not None else None,
                special_tokens=self.llm_special_tokens,
                latent_patch_size=_patch_size,
                condition_frame_indexes_vision=[],
                condition_frame_indexes_action=[],
                frame_idx=chunk_start,
                temporal_compression_factor=_tcf,
                video_temporal_causal=_video_tc,
                action_dim=_action_dim,
                enable_fps_modulation=_enable_fps_mod,
                base_fps=_base_fps,
                cached_text_offset=None if _cond_pack_text is not None else cond_cached_text_offset,
                unified_3d_mrope_temporal_modality_margin=_margin,
                force_action_tokens=_video_tc and self.config.action_gen,
                action_domain_id=action_domain_id,
                raw_action_dim=raw_action_dim,
            )

            # Pack unconditional frame when CFG is active. Same text_tokens gating
            # as the conditional pack above.
            if cfg_active:
                _uncond_pack_text = uncond_text_tokens[0] if (uncond_text_tokens and include_text_in_pack) else None
                packed_seq_uncond = pack_input_sequence_autoregressive(
                    vision_latent=curr_vision_latent,
                    action_latent=curr_action_latent,
                    text_tokens=_uncond_pack_text,
                    timestep=0.0,
                    fps_vision=fps_vision_list,
                    fps_action=fps_action_list if curr_action_latent is not None else None,
                    special_tokens=self.llm_special_tokens,
                    latent_patch_size=_patch_size,
                    condition_frame_indexes_vision=[],
                    condition_frame_indexes_action=[],
                    frame_idx=chunk_start,
                    temporal_compression_factor=_tcf,
                    video_temporal_causal=_video_tc,
                    action_dim=_action_dim,
                    enable_fps_modulation=_enable_fps_mod,
                    base_fps=_base_fps,
                    cached_text_offset=None if _uncond_pack_text is not None else uncond_cached_text_offset,
                    unified_3d_mrope_temporal_modality_margin=_margin,
                    force_action_tokens=_video_tc and self.config.action_gen,
                    action_domain_id=action_domain_id,
                    raw_action_dim=raw_action_dim,
                )

            # Denoise the whole chunk jointly.
            assert packed_seq is not None
            denoised_chunk = self.generate_next_frame(
                packed_seq=packed_seq,
                packed_seq_uncond=packed_seq_uncond,
                curr_vision_latent=curr_vision_latent,
                curr_action_latent=curr_action_latent,
                cond_text_tokens=cond_text_tokens[0] if cond_text_tokens else [],
                uncond_text_tokens=uncond_text_tokens[0] if uncond_text_tokens else [],
                gen_data_clean=gen_data_clean,
                dual_kv_cache=dual_kv_cache,
                dual_kv_cache_uncond=dual_kv_cache_uncond,
                frame_idx=chunk_start,
                cache_frame_idx=transfer_history_cache_idx if is_transfer else chunk_start,
                num_frames=num_frames,
                guidance=guidance,
                num_steps=num_steps,
                shift=shift,
                seed=seed,
                normalize_cfg=normalize_cfg,
                sampler_mode=sampler_mode,
                distilled_num_steps=distilled_num_steps,
                fps_vision_list=fps_vision_list,
                fps_action_list=fps_action_list,
                use_ar_rolling_path=use_ar_rolling_path,
            )
            if on_clean_vision_chunk is not None:
                on_clean_vision_chunk(denoised_chunk)

            # Refresh the KV cache with each clean denoised frame of the chunk so
            # subsequent chunks attend through in-distribution K/V. The sampler left
            # the cache holding proj(x_{t_final}) + time_embed(t_final), which training
            # never saw on a context frame; without this refresh downstream chunks
            # would attend through a perturbed v_N and collapse to noise.
            # K/V are token-wise projections (independent of attention), so a
            # per-frame seed forward over a single clean frame stores the correct
            # per-frame K/V; seed in increasing frame order. Under TF/none the seed
            # uses the clean frame with σ=0 condition-mask semantics; under DF it
            # injects σ_small noise (matching the denoise-loop K/V layout).
            # Skip the global last frame (no downstream reader). For
            # image2video / forward_dynamics, the singleton prefix frames are
            # seeded by the Step 7 prefill and the loop starts past them.
            for _local_i in range(chunk_len):
                _f = chunk_start + _local_i
                if not (streaming_actions or _f < num_frames - 1):
                    continue
                # Sampler returns fp32; cast to bf16 so vae2llm matches the model graph.
                _frame_latent = denoised_chunk[:, :, _local_i : _local_i + 1].to(**self.tensor_kwargs)  # [1,C,1,H,W]
                _seed_action: torch.Tensor | None = None
                if has_action and _f > 0:
                    if streaming_actions:
                        _seed_action = curr_action_latent  # [tcf,D] (chunk_len == 1 under streaming)
                    else:
                        assert gen_data_clean.x0_tokens_action is not None
                        _seed_action = gen_data_clean.x0_tokens_action[0][(_f - 1) * _tcf : _f * _tcf, :].to(
                            **self.tensor_kwargs
                        )  # a_{_f-1}; [tcf,D]
                self._seed_frame_into_kv_cache(
                    frame_latent=_frame_latent,
                    frame_idx=transfer_history_cache_idx if is_transfer else _f,
                    dual_kv_cache=dual_kv_cache,
                    dual_kv_cache_uncond=dual_kv_cache_uncond,
                    cond_text_tokens=(
                        cond_text_tokens[0] if (cond_text_tokens and _f == 0 and not is_transfer) else None
                    ),
                    uncond_text_tokens=(
                        uncond_text_tokens[0] if (uncond_text_tokens and _f == 0 and not is_transfer) else None
                    ),
                    cond_cached_text_offset=cond_cached_text_offset,
                    uncond_cached_text_offset=uncond_cached_text_offset,
                    curr_action_latent=_seed_action,
                    gen_data_clean=gen_data_clean,
                    fps_vision_list=fps_vision_list,
                    fps_action_list=fps_action_list,
                    seed=seed,
                    cfg_active=cfg_active,
                    cfgp_enabled=cfgp_enabled,
                    tcf=_tcf,
                    patch_size=_patch_size,
                    action_dim=_action_dim,
                    video_tc=_video_tc,
                    enable_fps_mod=_enable_fps_mod,
                    base_fps=_base_fps,
                    modality_margin=_margin,
                    use_ar_rolling_path=use_ar_rolling_path,
                    position_frame_idx=_f,
                )
                if is_transfer:
                    transfer_history_cache_idx += 1

            # Yield each latent frame of the chunk individually so the caller can
            # decode and offload it incrementally instead of materializing the full
            # latent video on GPU first.
            for _local_i in range(chunk_len):
                _f = chunk_start + _local_i
                _frame_out = denoised_chunk[:, :, _local_i : _local_i + 1]  # [1,C,1,H,W]
                if has_action:
                    if streaming_actions:
                        _action_out: torch.Tensor | None = curr_action_latent
                    elif _f > 0:
                        assert gen_data_clean.x0_tokens_action is not None
                        _action_out = gen_data_clean.x0_tokens_action[0][(_f - 1) * _tcf : _f * _tcf, :].to(
                            **self.tensor_kwargs
                        )
                    else:
                        _action_out = None
                    payload = {"vision": _frame_out, "action": _action_out}
                else:
                    payload = {"vision": _frame_out}
                if streaming_actions and has_action:
                    next_streamed_action = yield payload
                else:
                    yield payload

    def _cast_ar_action_tokens_to_projection_dtype(self, packed_seq: PackedSequence) -> None:
        """Cast AR action tokens to the dtype expected by the action projection."""
        if packed_seq.action is None or not packed_seq.action.tokens:
            return
        action2llm = getattr(self.net, "action2llm", None)
        fc = getattr(action2llm, "fc", None)
        weight = getattr(fc, "weight", None)
        action_dtype = getattr(weight, "dtype", torch.float32)
        if not isinstance(action_dtype, torch.dtype):
            action_dtype = torch.float32
        packed_seq.action.tokens = [
            token.to(device=self.tensor_kwargs["device"], dtype=action_dtype) for token in packed_seq.action.tokens
        ]  # list of [T_action,D]

    @torch.no_grad()
    def _seed_frame_into_kv_cache(
        self,
        frame_latent: torch.Tensor,  # [1,C,T,H,W]
        frame_idx: int,
        dual_kv_cache: list[DualKVCache],
        dual_kv_cache_uncond: list[DualKVCache] | None,
        cond_text_tokens: list[int] | None,
        uncond_text_tokens: list[int] | None,
        cond_cached_text_offset: int,
        uncond_cached_text_offset: int,
        curr_action_latent: torch.Tensor | None,
        gen_data_clean: GenerationDataClean,
        fps_vision_list: list[float],
        fps_action_list: list[float],
        seed: int,
        cfg_active: bool,
        cfgp_enabled: bool,
        tcf: int,
        patch_size: int,
        action_dim: int,
        video_tc: bool,
        enable_fps_mod: bool,
        base_fps: float,
        modality_margin: int,
        use_ar_rolling_path: bool = False,
        position_frame_idx: int | None = None,
        condition_frame_indexes_vision: list[int] | None = None,
    ) -> None:
        """Run one forward pass that writes ``frame_latent``'s K/V into
        ``dual_kv_cache[layer].gen_cache[frame_idx]``, under the strategy-appropriate
        input distribution.

        ``use_ar_rolling_path`` mirrors the velocity_fn flag and selects the
        static-shape ``ARMemoryState(for_cuda_graphs=True)`` at ``frame_idx >= 1``
        so the seed (post-denoise gen_cache refresh) shares a single compiled
        ``forward`` and a single CUDA-graph capture with the AR-loop velocity_fn.
        Without this, the seed traces with ``for_cuda_graphs=False`` while the
        AR loop traces with ``for_cuda_graphs=True``: two distinct shape
        signatures, two separate inductor graphs, and a separate cudagraph
        tree captured every frame.  Frame 0 always stays eager
        (``use_ar_rolling=False``) — the und cache is empty there and the
        static-shape branch asserts it populated.

        Args:
            frame_latent: Clean frame/chunk latent ``(1, C, T, H, W)``.
            frame_idx: Logical K/V-cache index.
            cond_text_tokens / uncond_text_tokens: Text tokens to include in the pack,
                or ``None`` to reuse cached ``und_cache`` (for ``frame_idx > 0``).
            cond_cached_text_offset / uncond_cached_text_offset: cached-text mRoPE offset
                used when text tokens are omitted (threaded from ``iter_samples_...``);
                modality margin is added internally by ``pack_input_sequence_autoregressive``.
            curr_action_latent: Action for this frame (forward_dynamics only) or None.
            seed: Base seed; only consulted under ``diffusion_forcing`` for ε sampling.
            position_frame_idx: Global latent-frame position used for mRoPE. Defaults
                to ``frame_idx``. Transfer uses interleaved cache indexes while
                preserving aligned control/target temporal positions.
            condition_frame_indexes_vision: Clean local frame indexes for a
                multi-frame control prefill.
        """
        if position_frame_idx is None:
            position_frame_idx = frame_idx
        strategy = self.config.causal_training_strategy

        if strategy == "diffusion_forcing":
            sigma = self.config.sigma_diffusion_forcing
            # Deterministic ε shared across cond/uncond packs.
            g = torch.Generator(device=frame_latent.device).manual_seed(seed + position_frame_idx)
            eps = torch.empty_like(frame_latent).normal_(generator=g)  # [1,C,T,H,W]
            frame_in = sigma * eps + (1.0 - sigma) * frame_latent  # [1,C,T,H,W]
            # Post-transform σ: pass directly as timestep, no shift-transform applied.
            max_timestep = self.rectified_flow_video.noise_scheduler.config.num_train_timesteps
            timestep_val: float = float(sigma) * float(max_timestep)
            condition_vision: list[int] = (
                list(condition_frame_indexes_vision) if condition_frame_indexes_vision is not None else []
            )
        elif strategy in ("none", "teacher_forcing", "teacher_forcing_dcm"):
            frame_in = frame_latent  # [1,C,T,H,W]
            timestep_val = 0.0
            condition_vision = (
                list(condition_frame_indexes_vision) if condition_frame_indexes_vision is not None else [0]
            )
        else:
            raise ValueError(f"Unknown causal_training_strategy: {strategy!r}")

        def _build_pack(text_tokens: list[int] | None, cached_text_offset: int) -> PackedSequence:
            action_domain_id = OmniMoTCausalModel._first_action_domain_id(gen_data_clean)
            raw_action_dim = OmniMoTCausalModel._first_raw_action_dim(gen_data_clean)
            return pack_input_sequence_autoregressive(
                vision_latent=frame_in,
                action_latent=curr_action_latent,
                text_tokens=text_tokens,
                timestep=timestep_val,
                fps_vision=fps_vision_list,
                fps_action=fps_action_list if curr_action_latent is not None else None,
                special_tokens=self.llm_special_tokens,
                latent_patch_size=patch_size,
                condition_frame_indexes_vision=condition_vision,
                condition_frame_indexes_action=[],
                frame_idx=position_frame_idx,
                temporal_compression_factor=tcf,
                video_temporal_causal=video_tc,
                action_dim=action_dim,
                enable_fps_modulation=enable_fps_mod,
                base_fps=base_fps,
                cached_text_offset=None if text_tokens is not None else cached_text_offset,
                unified_3d_mrope_temporal_modality_margin=modality_margin,
                force_action_tokens=video_tc and self.config.action_gen,
                action_domain_id=action_domain_id,
                raw_action_dim=raw_action_dim,
            )

        cond_pack = _build_pack(cond_text_tokens, cond_cached_text_offset)
        uncond_pack = _build_pack(uncond_text_tokens, uncond_cached_text_offset) if cfg_active else None
        self._cast_ar_action_tokens_to_projection_dtype(cond_pack)
        if uncond_pack is not None:
            self._cast_ar_action_tokens_to_projection_dtype(uncond_pack)

        # Static-shape AR seed at frame_idx >= 1 when CG is active.  Frame 0
        # always runs on the dynamic-shape branch: the und cache is not yet populated and the
        # static-shape branch asserts it.  At frame_idx >= 1 the und cache
        # was populated by the frame-0 prefill, so the seed can share
        # the AR-loop's compiled ``forward`` and CUDA-graph capture.
        use_ar_rolling = use_ar_rolling_path and frame_idx > 0
        post_saturation_static_compile = is_ar_post_saturation_static_compile_frame(self, frame_idx)
        post_saturation_cuda_graph = is_ar_post_saturation_cuda_graph_frame(self, frame_idx)
        memory_info = {
            "dual_kv_cache": dual_kv_cache,
            "use_rolling_kv_cache": False,
            "use_ar_rolling": use_ar_rolling,
            "post_saturation_static_compile": post_saturation_static_compile,
            "frame_idx": frame_idx,
            "write_gen_cache": True,
        }

        if cfgp_enabled:
            pack = cond_pack if self.parallel_dims.cfgp_rank == 0 else uncond_pack
            assert pack is not None
            pack.to_cuda()
            torch.compiler.cudagraph_mark_step_begin()
            memory = self.build_memory_state(pack, memory_info)
            self.denoise(
                data_batch_packed=pack,
                memory=memory,
            )
            return

        cond_pack.to_cuda()
        if post_saturation_cuda_graph:
            run_ar_post_saturation_cuda_graph(
                self,
                kind="refresh",
                branch="conditional",
                packed_seq=cond_pack,
                memory_info=memory_info,
            )
        else:
            torch.compiler.cudagraph_mark_step_begin()
            memory = self.build_memory_state(cond_pack, memory_info)
            self.denoise(
                data_batch_packed=cond_pack,
                memory=memory,
            )
        if dual_kv_cache_uncond is not None:
            assert uncond_pack is not None
            uncond_pack.to_cuda()
            memory_info_uncond = {
                "dual_kv_cache": dual_kv_cache_uncond,
                "use_rolling_kv_cache": False,
                "use_ar_rolling": use_ar_rolling,
                "post_saturation_static_compile": post_saturation_static_compile,
                "frame_idx": frame_idx,
                "write_gen_cache": True,
            }
            if post_saturation_cuda_graph:
                run_ar_post_saturation_cuda_graph(
                    self,
                    kind="refresh",
                    branch="unconditional",
                    packed_seq=uncond_pack,
                    memory_info=memory_info_uncond,
                )
            else:
                torch.compiler.cudagraph_mark_step_begin()
                memory_uncond = self.build_memory_state(uncond_pack, memory_info_uncond)
                self.denoise(
                    data_batch_packed=uncond_pack,
                    memory=memory_uncond,
                )

    def _get_ar_distilled_timestep_schedule(
        self,
        distilled_num_steps: int | None = None,
        frame_idx: int | None = None,
        num_frames: int | None = None,
    ) -> list[float]:
        """Return the causal-distillation few-step sigma schedule used by AR inference."""
        del frame_idx, num_frames
        sampler_cfg = self.config.fixed_step_sampler_config
        full_t_list = list(sampler_cfg.t_list)
        if not full_t_list:
            raise ValueError("fixed_step_sampler_config.t_list must not be empty")
        if full_t_list[-1] != 0.0:
            full_t_list = full_t_list + [0.0]
        if distilled_num_steps is not None:
            if distilled_num_steps < 1:
                raise ValueError(f"distilled_num_steps must be >= 1, got {distilled_num_steps}")
            if distilled_num_steps < len(full_t_list) - 1:
                full_t_list = full_t_list[:distilled_num_steps] + [0.0]
        if len(full_t_list) < 2:
            raise ValueError("fixed_step_sampler_config.t_list must contain a nonzero sigma")
        return [float(t) for t in full_t_list]

    def _run_distilled_ar_sampler(
        self,
        velocity_fn: Any,
        initial_noise: torch.Tensor,
        *,
        seed: int,
        frame_idx: int,
        num_frames: int | None = None,
        distilled_num_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample one AR latent frame from ``fixed_step_sampler_config.t_list``."""
        full_t_list = self._get_ar_distilled_timestep_schedule(
            distilled_num_steps,
            frame_idx=frame_idx,
            num_frames=num_frames,
        )  # [N_steps+1]
        sample_type = self.config.fixed_step_sampler_config.sample_type
        max_timestep = float(self.config.rectified_flow_inference_config.num_train_timesteps)
        x = initial_noise.float()  # [B,N_tokens_flat]
        for step_idx, (sigma_cur, sigma_next) in enumerate(zip(full_t_list[:-1], full_t_list[1:])):
            timestep = torch.full(
                (x.shape[0], 1),
                sigma_cur * max_timestep,
                dtype=torch.float32,
                device=x.device,
            )  # [B,1]
            velocity = velocity_fn(x, timestep).float()  # [B,N_tokens_flat]
            sigma_cur_tensor = torch.as_tensor(sigma_cur, dtype=torch.float32, device=x.device)  # []
            x0_pred = x - sigma_cur_tensor * velocity  # [B,N_tokens_flat]
            if sigma_next == 0.0:
                x = x0_pred  # [B,N_tokens_flat]
                continue
            sigma_next_tensor = torch.as_tensor(sigma_next, dtype=torch.float32, device=x.device)  # []
            if sample_type == "ode":
                delta_sigma = torch.as_tensor(sigma_next - sigma_cur, dtype=torch.float32, device=x.device)  # []
                x = x + delta_sigma * velocity  # [B,N_tokens_flat]
            elif sample_type == "sde":
                # Use a mixed seed so different frame/step pairs cannot collide
                # (e.g. frame 0 step 1 vs frame 1 step 0).
                step_seed = int(seed) + int(frame_idx) * 1_000_003 + int(step_idx) * 9_176
                generator = torch.Generator(device=x.device).manual_seed(step_seed)
                noise = torch.empty_like(x).normal_(generator=generator)  # [B,N_tokens_flat]
                x = (1.0 - sigma_next_tensor) * x0_pred + sigma_next_tensor * noise  # [B,N_tokens_flat]
            else:
                raise ValueError(f"Unsupported distilled sample_type: {sample_type!r}")
        return x  # [B,N_tokens_flat]

    @torch.no_grad()
    def generate_next_frame(
        self,
        packed_seq: PackedSequence,
        packed_seq_uncond: PackedSequence | None,
        curr_vision_latent: torch.Tensor,
        curr_action_latent: torch.Tensor | None,
        cond_text_tokens: list[int],
        uncond_text_tokens: list[int],
        gen_data_clean: GenerationDataClean,
        dual_kv_cache: list[DualKVCache],
        dual_kv_cache_uncond: list[DualKVCache] | None,
        guidance: float,
        num_steps: int,
        shift: float,
        seed: int,
        fps_vision_list: list[float],
        fps_action_list: list[float],
        frame_idx: int | None = None,
        cache_frame_idx: int | None = None,
        num_frames: int | None = None,
        normalize_cfg: bool = False,
        sampler_mode: str = "rf",
        distilled_num_steps: int | None = None,
        use_ar_rolling_path: bool = False,
    ) -> torch.Tensor:
        """
        Denoise a single frame using AR generation with cumulative pack.

        This function is called iteratively by iter_samples_from_batch_autoregressive()
        to denoise each frame one at a time. The packed_seq parameter contains all
        previous frames plus the current frame, providing full temporal context.

        Args:
            packed_seq: PackedSequence containing [text][v0]...[v{frame_idx}] with current frame noisy
            curr_vision_latent: Current frame to denoise. Shape: (1, C, 1, H, W)
            curr_action_latent: Action for current frame. Shape: (tcf, D) or None
            cond_text_tokens: Conditional text tokens for CFG
            uncond_text_tokens: Unconditional text tokens for CFG
            gen_data_clean: Generation data with FPS and metadata
            dual_kv_cache: Dual KV cache for conditional path
            dual_kv_cache_uncond: Dual KV cache for unconditional path. Required
                when ``guidance != 1.0`` without CFGP; unused under CFGP.
            guidance: CFG guidance weight. When ``guidance == 1.0`` (and CFGP is
                off), the unconditional forward is skipped.
            num_steps: Number of denoising steps
            shift: Time shift parameter
            seed: Random seed
            fps_vision_list: FPS list for vision tokens
            fps_action_list: FPS list for action tokens
            frame_idx: Current temporal frame/chunk index (default: 0).
            cache_frame_idx: Logical K/V-cache index. Defaults to ``frame_idx``;
                transfer uses an interleaved control/RGB index.
            num_frames: Total latent frame count for frame-aware distilled schedules.
            normalize_cfg: Normalize CFG output
            sampler_mode: ``"rf"`` for multi-step RF sampling, ``"distilled"`` for
                direct causal-distillation few-step sampling.
            distilled_num_steps: Optional debug truncation for distilled schedules.

        Returns:
            Denoised frame latent. Shape: (1, C, 1, H, W)
        """
        # Handle None frame_idx as 0
        if frame_idx is None:
            frame_idx = 0
        if cache_frame_idx is None:
            cache_frame_idx = frame_idx

        # packed_seq now passed as parameter, containing all frames up to current
        # No need to create pack here - it's maintained by the caller

        def cast_pack_action_tokens(pack: PackedSequence) -> None:
            """Keep action conditioning in the dtype expected by action2llm."""
            if pack.action is not None and pack.action.tokens:
                self._cast_ar_action_tokens_to_projection_dtype(pack)

        # Helper to set noisy latents in pack. The pack holds one vision *item*
        # (a single clip) whose temporal extent is 1 frame (framewise) or
        # ``chunk_len`` frames (chunkwise); the whole chunk shares one timestep.
        def set_pack_noise(
            pack: PackedSequence,
            noise_x_vision: torch.Tensor,
            timestep: torch.Tensor,
        ) -> None:
            """Set noisy latents for the current frame/chunk in the pack (one vision item)."""
            if pack.vision is not None:
                # tokens is list[Tensor] with exactly one entry (one vision item / clip).
                assert len(pack.vision.tokens) == 1, (
                    f"AR packing: expected 1 vision item, got {len(pack.vision.tokens)}"
                )
                pack.vision.tokens = [noise_x_vision.to(**self.tensor_kwargs)]
                # One timestep entry per noisy patch (Th*Tw per latent frame, across
                # all frames of the chunk); the chunk is denoised at a single σ.
                n_vision_patches = len(pack.vision.mse_loss_indexes)
                pack.vision.timesteps = (
                    timestep.flatten()[0]
                    .repeat(n_vision_patches)
                    .to(
                        device=self.tensor_kwargs["device"],
                        dtype=torch.float32,
                    )
                )  # [N_noisy_vision]

            if curr_action_latent is not None and pack.action is not None:
                # For action, we keep it clean (condition) - no noise added
                assert len(pack.action.tokens) == 1, (
                    f"AR packing: expected 1 action item, got {len(pack.action.tokens)}"
                )
                pack.action.tokens = [
                    curr_action_latent.to(device=self.tensor_kwargs["device"], dtype=torch.float32)
                ]  # [TCF,D]
                # 1 entry per noisy action step (frame_token_stride=1 for action)
                n_action_steps = len(pack.action.mse_loss_indexes)
                pack.action.timesteps = torch.zeros(
                    n_action_steps,
                    device=self.tensor_kwargs["device"],
                    dtype=torch.float32,
                )  # [N_noisy_action]

            cast_pack_action_tokens(pack)
            pack.to_cuda()

        # Define velocity function for denoising
        # KV cache enabled: retrieve previous frames' K/V, don't store until final step
        def velocity_fn(noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            """
            Velocity function for sampler.

            Computes velocity prediction for the current noisy latent at given timestep.
            Handles both conditional (with text) and unconditional (without text) forward passes
            for Classifier-Free Guidance (CFG).
            """
            assert timestep.shape == (1, 1), f"Expected (1, 1), got {timestep.shape}"

            # Reshape noise_x to vision shape
            vision_shape = curr_vision_latent.shape
            noise_x_vision = noise_x.reshape(vision_shape)  # [B,C,1,H,W]

            cfgp_enabled = self.parallel_dims is not None and self.parallel_dims.cfgp_enabled

            # AR path selection (single ``attention_AR_gen_only`` kernel
            # for all inference modes):
            #
            # - frame_idx == 0: dynamic-shape ``ARMemoryState`` seeds
            #   the und cache.  At i2v/forward_dynamics the prefill
            #   above already handled frame 0, so this branch is unused
            #   there.
            # - frame_idx > 0 with compile + CG: static-shape
            #   ``ARMemoryState(for_cuda_graphs=True)`` (signalled via
            #   ``use_ar_rolling=True``).  Constant shapes across frames
            #   so a single CUDA-graph capture replays.  The attention
            #   itself is the same ``attention_AR_gen_only`` function;
            #   it picks the static-shape branch when the memory_value
            #   carries a preallocated gen buffer.
            # - frame_idx > 0 without CG (eager or compile-only):
            #   dynamic-shape ``ARMemoryState`` (``use_ar_rolling=False``).
            #
            use_ar_rolling = use_ar_rolling_path and cache_frame_idx > 0
            post_saturation_static_compile = is_ar_post_saturation_static_compile_frame(self, cache_frame_idx)
            post_saturation_cuda_graph = is_ar_post_saturation_cuda_graph_frame(self, cache_frame_idx)
            memory_info = {
                "dual_kv_cache": dual_kv_cache,
                "use_rolling_kv_cache": False,
                "use_ar_rolling": use_ar_rolling,
                "post_saturation_static_compile": post_saturation_static_compile,
                "frame_idx": cache_frame_idx,
                # Do not persist noisy current-frame K/V from intermediate
                # denoising steps into the rolling AR cache.
                # The clean denoised frame is written exactly once by
                # ``_seed_frame_into_kv_cache`` after sampling finishes.
                "write_gen_cache": False,
            }

            if cfgp_enabled:
                # CFGP: each rank runs one branch in parallel, then P2P-exchange velocities.
                # Rank 0 = conditional, rank 1 = unconditional.
                cfgp_rank = self.parallel_dims.cfgp_rank
                cfgp_size = self.parallel_dims.cfgp_size
                cfgp_group = self.parallel_dims.cfgp_mesh.get_group()

                if cfgp_rank == 0:
                    set_pack_noise(packed_seq, noise_x_vision, timestep)
                    torch.compiler.cudagraph_mark_step_begin()
                    # memory_info has the local (conditional for rank 0) kv_cache.
                    memory = self.build_memory_state(packed_seq, memory_info)
                    out = self.denoise(
                        data_batch_packed=packed_seq,
                        memory=memory,
                    )
                else:
                    assert packed_seq_uncond is not None
                    set_pack_noise(packed_seq_uncond, noise_x_vision, timestep)
                    torch.compiler.cudagraph_mark_step_begin()
                    # memory_info has the local (unconditional for rank 1) kv_cache
                    memory = self.build_memory_state(packed_seq_uncond, memory_info)
                    out = self.denoise(
                        data_batch_packed=packed_seq_uncond,
                        memory=memory,
                    )

                # P2P exchange: same pattern as _run_classifier_free_guidance()
                v = torch.stack(out["preds_vision"]).contiguous()  # [1,C,1,H,W]
                other_v = torch.empty_like(v)  # [1,C,1,H,W]
                cfgp_peer = (cfgp_rank + 1) % cfgp_size
                reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(op=dist.isend, tensor=v, group_peer=cfgp_peer, group=cfgp_group),
                        dist.P2POp(op=dist.irecv, tensor=other_v, group_peer=cfgp_peer, group=cfgp_group),
                    ]
                )
                for req in reqs:
                    req.wait()
                velocity_cond = v if cfgp_rank == 0 else other_v  # [1,C,1,H,W]
                velocity_uncond = other_v if cfgp_rank == 0 else v  # [1,C,1,H,W]
            else:
                # Sequential path (no CFGP)
                set_pack_noise(packed_seq, noise_x_vision, timestep)
                if post_saturation_cuda_graph:
                    out_cond = run_ar_post_saturation_cuda_graph(
                        self,
                        kind="denoise",
                        branch="conditional",
                        packed_seq=packed_seq,
                        memory_info=memory_info,
                    )
                else:
                    torch.compiler.cudagraph_mark_step_begin()
                    # memory_info holds the conditional kv-cache
                    memory_cond = self.build_memory_state(packed_seq, memory_info)
                    out_cond = self.denoise(
                        data_batch_packed=packed_seq,
                        memory=memory_cond,
                    )
                velocity_cond = torch.stack(out_cond["preds_vision"])  # [1,C,1,H,W]

                # Skip the unconditional forward when guidance == 1.0 — the CFG
                # formula collapses to velocity_cond, so computing it is pure waste.
                if guidance == 1.0:
                    return velocity_cond.flatten(start_dim=1)  # [B,N_tokens_flat]

                assert packed_seq_uncond is not None, "packed_seq_uncond required when guidance != 1.0 without CFGP"
                assert dual_kv_cache_uncond is not None, (
                    "dual_kv_cache_uncond required when guidance != 1.0 without CFGP"
                )
                set_pack_noise(packed_seq_uncond, noise_x_vision, timestep)
                memory_info_uncond = {
                    "dual_kv_cache": dual_kv_cache_uncond,
                    "use_rolling_kv_cache": False,
                    "use_ar_rolling": use_ar_rolling,
                    "post_saturation_static_compile": post_saturation_static_compile,
                    "frame_idx": cache_frame_idx,
                    # Do not persist noisy unconditional K/V from intermediate
                    # denoising steps into the rolling AR cache.
                    # The clean denoised frame is written exactly once by
                    # ``_seed_frame_into_kv_cache`` after sampling finishes.
                    "write_gen_cache": False,
                }
                if post_saturation_cuda_graph:
                    out_uncond = run_ar_post_saturation_cuda_graph(
                        self,
                        kind="denoise",
                        branch="unconditional",
                        packed_seq=packed_seq_uncond,
                        memory_info=memory_info_uncond,
                    )
                else:
                    torch.compiler.cudagraph_mark_step_begin()
                    memory_uncond = self.build_memory_state(packed_seq_uncond, memory_info_uncond)
                    out_uncond = self.denoise(
                        data_batch_packed=packed_seq_uncond,
                        memory=memory_uncond,
                    )
                velocity_uncond = torch.stack(out_uncond["preds_vision"])  # [1,C,1,H,W]

            if normalize_cfg:
                # Normalized CFG: interpolate between uncond and cond
                velocity_pred = (1 - guidance) * velocity_uncond + guidance * velocity_cond  # [1,C,1,H,W]
            else:
                # Standard CFG
                velocity_pred = velocity_uncond + guidance * (velocity_cond - velocity_uncond)  # [1,C,1,H,W]

            # Return flattened velocity
            return velocity_pred.flatten(start_dim=1)  # [B,N_tokens_flat]

        # Initialize noise
        initial_noise = curr_vision_latent.flatten(start_dim=1)  # [B,N_tokens_flat]

        # Run sampler
        if sampler_mode == "distilled":
            denoised_flat = self._run_distilled_ar_sampler(
                velocity_fn,
                initial_noise,
                seed=seed,
                frame_idx=frame_idx,
                num_frames=num_frames,
                distilled_num_steps=distilled_num_steps,
            )  # [B,N_tokens_flat]
        elif sampler_mode == "rf" and self.config.rectified_flow_inference_config.scheduler_type == "unipc":
            denoised_flat = self.sampler(
                velocity_fn,
                initial_noise,
                num_steps=num_steps,
                shift=shift,
                seed=seed + frame_idx,
            )  # [1,C*H*W]
        elif sampler_mode == "rf":
            # EDM sampler
            def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                timestep_rf = sigma * float(self.config.rectified_flow_inference_config.num_train_timesteps)
                timestep_rf = timestep_rf.unsqueeze(0)  # [1,1]
                velocity_pred = velocity_fn(noise_x, timestep_rf)  # [1,C*H*W]
                x0_pred = noise_x - sigma * velocity_pred  # [1,C*H*W]
                return x0_pred

            denoised_flat = self.sampler(
                x0_fn,
                initial_noise,
                num_steps=num_steps,
                sigma_max=80.0,
                sigma_min=0.002,
                solver_option="2ab",
            )
        else:
            raise ValueError(f"Unsupported sampler_mode: {sampler_mode!r}")

        # Reshape to frame shape
        denoised_frame = denoised_flat.reshape(curr_vision_latent.shape)  # [B,C,1,H,W]

        return denoised_frame
