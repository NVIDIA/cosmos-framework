# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared multiview Transfer autoregressive backend and replay context."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch

from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.utils.memory import MemoryState
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan
from cosmos_framework.model.generator.mot.causal_flex_attention import (
    MultiviewTransferARCurrentRole,
    MultiviewTransferARMemoryLayout,
    build_multiview_transfer_ar_memory_layout,
)
from cosmos_framework.model.generator.teacher_forcing import mark_modality_as_clean_condition
from cosmos_framework.model.generator.utils.kv_cache import FlexARMemoryState
from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive

MultiviewTransferARKVCache = list[tuple[torch.Tensor, torch.Tensor] | None]
_ARBranch = Literal["conditional", "unconditional"]


class MultiviewTransferARHost(Protocol):
    """Model surface required by the shared multiview Transfer AR backend."""

    config: Any
    llm_special_tokens: dict[str, int]
    net: torch.nn.Module
    parallel_dims: Any
    tensor_kwargs: dict[str, Any]
    tokenizer_vision_gen: Any

    def _pack_input_sequence(
        self,
        sequence_plans: list[SequencePlan],
        input_text_indexes: list[list[int]],
        gen_data_clean: GenerationDataClean,
        input_timesteps: torch.Tensor,
        include_end_of_generation_token: bool = False,
        skip_text_tokens: bool = False,
        initial_mrope_temporal_offset: int | float = 0,
    ) -> PackedSequence: ...

    def _cast_generated_tokens_to_precision(self, packed_sequence: PackedSequence) -> None: ...

    def denoise(
        self,
        net: torch.nn.Module | None = None,
        data_batch_packed: PackedSequence | None = None,
        memory: MemoryState | None = None,
        video_temporal_causal: bool | None = None,
    ) -> dict[str, Any]: ...


@dataclass
class MultiviewTransferARSession:
    """Mutable backend-owned state for one multiview Transfer AR rollout."""

    token_shapes: tuple[tuple[int, int, int], ...]
    target_condition_mask: torch.Tensor  # [V*T,1,1]
    num_views: int
    frames_per_view: int
    frames_per_chunk: int
    condition_count: int
    memory_seq_len: int
    control_frame_ranges: list[tuple[int, int]]
    target_condition_frame_ranges: list[tuple[int, int]]
    history_frame_ranges: list[tuple[int, int]]
    conditional_cache: MultiviewTransferARKVCache  # Sequential conditional or CFGP rank-local K/V: [1,M,H_kv,D]
    unconditional_cache: MultiviewTransferARKVCache | None  # K/V: [1,M,H_kv,D] or None
    cfg_active: bool
    cfgp_enabled: bool


@dataclass(frozen=True)
class MultiviewTransferARReplayContext:
    """Immutable layout snapshot plus shared detached cache for one replayed chunk."""

    cache: MultiviewTransferARKVCache  # K/V: [1,M,H_kv,D]
    token_shapes: tuple[tuple[int, int, int], ...]
    target_condition_mask: torch.Tensor  # [V*T,1,1]
    num_views: int
    frames_per_view: int
    frames_per_chunk: int
    condition_count: int
    history_frame_ranges: tuple[tuple[int, int], ...]
    memory_seq_len: int
    text_tokens: tuple[int, ...]
    fps_vision: tuple[float, ...]


class MultiviewTransferARBackend:
    """Own multiview Transfer packing, Flex KV layout, cache writes, and replay state."""

    def __init__(self, host: MultiviewTransferARHost) -> None:
        self.host = host

    def build_prefill_pack(
        self,
        *,
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        text_tokens: list[int],
        materialized_target_frame_ranges: Sequence[tuple[int, int]] | None = None,
    ) -> PackedSequence:
        """Build one full-geometry clean prefill pack for cache capture."""
        pack = self.host._pack_input_sequence(
            sequence_plans,
            [text_tokens],
            gen_data_clean,
            torch.zeros(1, dtype=torch.float32),  # [1]
        )
        if pack.vision is None:
            raise ValueError("Multiview transfer AR prefill requires packed vision data.")
        original_masks = [mask.clone() for mask in pack.vision.condition_mask]  # list[[V*T,1,1]]
        pack.teacher_forcing_pass = "clean"
        pack.teacher_forcing_original_condition_masks_vision = original_masks
        if materialized_target_frame_ranges is not None:
            # Preserve the full two-item geometry while preventing real queries
            # from reading ungenerated target suffix values.
            pack.teacher_forcing_materialized_target_frame_ranges = tuple(materialized_target_frame_ranges)
        pack.to_cuda()
        self.host._cast_generated_tokens_to_precision(pack)
        return pack

    def create_session(
        self,
        *,
        prefill_pack: PackedSequence,
        num_views: int,
        frames_per_view: int,
        condition_count: int,
        cfg_active: bool,
        cfgp_enabled: bool,
    ) -> MultiviewTransferARSession:
        """Allocate backend state from a validated prefill pack."""
        if prefill_pack.vision is None or len(prefill_pack.vision.token_shapes) != 2:
            raise ValueError("Multiview transfer AR requires packed [control, target] vision metadata.")
        flex_backend = getattr(self.host.net, "flex_backend", None)
        if flex_backend is None:
            raise ValueError("Multiview transfer AR requires an initialized FlexAttention backend.")
        control_shape, target_shape = prefill_pack.vision.token_shapes
        total_memory_tokens = control_shape[0] * control_shape[1] * control_shape[2]
        total_memory_tokens += target_shape[0] * target_shape[1] * target_shape[2]
        kv_alignment = int(flex_backend.block_size[1])
        memory_seq_len = ((total_memory_tokens + kv_alignment - 1) // kv_alignment) * kv_alignment
        target_condition_ranges = [(0, condition_count)] if condition_count else []
        num_layers = int(self.host.net.num_hidden_layers)  # type: ignore[attr-defined]
        return MultiviewTransferARSession(
            token_shapes=tuple(prefill_pack.vision.token_shapes),
            target_condition_mask=prefill_pack.vision.condition_mask[1],  # [V*T,1,1]
            num_views=num_views,
            frames_per_view=frames_per_view,
            frames_per_chunk=int(self.host.config.teacher_forcing_frames_per_chunk),
            condition_count=condition_count,
            memory_seq_len=memory_seq_len,
            control_frame_ranges=[],
            target_condition_frame_ranges=target_condition_ranges,
            history_frame_ranges=[],
            conditional_cache=[None] * num_layers,
            unconditional_cache=[None] * num_layers if cfg_active and not cfgp_enabled else None,
            cfg_active=cfg_active,
            cfgp_enabled=cfgp_enabled,
        )

    @staticmethod
    def build_memory_layout(session: MultiviewTransferARSession) -> MultiviewTransferARMemoryLayout:
        """Resolve the current fixed-slot Flex memory layout."""
        return build_multiview_transfer_ar_memory_layout(
            token_shapes=list(session.token_shapes),
            target_condition_mask=session.target_condition_mask,
            num_views=session.num_views,
            frames_per_chunk=session.frames_per_chunk,
            control_frame_ranges=session.control_frame_ranges,
            target_condition_frame_ranges=session.target_condition_frame_ranges,
            history_frame_ranges=session.history_frame_ranges,
            memory_seq_len=session.memory_seq_len,
            device=session.target_condition_mask.device,
        )

    def build_current_pack(
        self,
        *,
        vision_latent: torch.Tensor,  # [1,C,V*chunk_len,H,W]
        text_tokens: list[int],
        fps_vision: list[float],
        num_views: int,
        frames_per_view: int,
        chunk_start: int,
        memory_layout: MultiviewTransferARMemoryLayout,
        current_role: MultiviewTransferARCurrentRole,
    ) -> PackedSequence:
        """Pack one synchronized chunk at its absolute camera-major mRoPE positions."""
        if vision_latent.shape[2] % num_views != 0:
            raise ValueError(
                f"Multiview transfer chunk latent_t={vision_latent.shape[2]} must be divisible by num_views={num_views}."
            )
        chunk_len = vision_latent.shape[2] // num_views
        temporal_positions = torch.cat(
            [
                torch.arange(
                    view_idx * frames_per_view + chunk_start,
                    view_idx * frames_per_view + chunk_start + chunk_len,
                    dtype=torch.float32,
                )
                for view_idx in range(num_views)
            ]
        )  # [V*chunk_len]
        pack = pack_input_sequence_autoregressive(
            vision_latent=vision_latent,
            action_latent=None,
            text_tokens=text_tokens,
            timestep=0.0,
            fps_vision=fps_vision,
            fps_action=None,
            special_tokens=self.host.llm_special_tokens,
            latent_patch_size=self.host.config.diffusion_expert_config.patch_spatial,
            condition_frame_indexes_vision=[],
            frame_idx=0,
            temporal_compression_factor=self.host.tokenizer_vision_gen.temporal_compression_factor or 4,
            video_temporal_causal=False,
            action_dim=self.host.config.max_action_dim,
            enable_fps_modulation=self.host.config.diffusion_expert_config.enable_fps_modulation,
            base_fps=float(self.host.config.diffusion_expert_config.base_fps),
            unified_3d_mrope_temporal_modality_margin=(
                self.host.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin
            ),
            vision_temporal_positions=temporal_positions,
            num_views=num_views,
        )
        pack.to_cuda()
        pack.multiview_transfer_ar_metadata = {
            "current_frame_start": chunk_start,
            "frames_per_view": frames_per_view,
            "frames_per_chunk": self.host.config.teacher_forcing_frames_per_chunk,
            "current_role": current_role,
            "memory_layout": memory_layout,
        }
        if current_role == "clean_target":
            if pack.vision is None:
                raise ValueError("Multiview transfer clean-history packing requires vision tokens.")
            mark_modality_as_clean_condition(pack.vision)
        self.host._cast_generated_tokens_to_precision(pack)
        return pack

    def capture_memory(
        self,
        *,
        pack: PackedSequence,
        cache: MultiviewTransferARKVCache,
        memory_seq_len: int,
        write_indexes: torch.Tensor,  # [S_write]
        write_offset: int,
        cache_write_indexes: torch.Tensor | None = None,  # [S_write] or None
    ) -> None:
        """Run a clean pass and commit selected generated-token K/V."""
        memory = FlexARMemoryState(
            num_layers=int(self.host.net.num_hidden_layers),  # type: ignore[attr-defined]
            memory_seq_len=memory_seq_len,
            cache=cache,
            write_indexes=write_indexes,
            write_offset=write_offset,
            cache_write_indexes=cache_write_indexes,
        )
        self.host.denoise(data_batch_packed=pack, memory=memory)

    @staticmethod
    def merge_memory(
        *,
        destination: MultiviewTransferARKVCache,
        source: MultiviewTransferARKVCache,
        cache_indexes: torch.Tensor,  # [S_write]
    ) -> None:
        """Copy selected fixed-slot K/V from a scratch no-memory clean pass."""
        if len(destination) != len(source):
            raise ValueError(f"Expected matching cache layers, got {len(destination)} and {len(source)}.")
        for layer_idx, source_kv in enumerate(source):
            if source_kv is None:
                raise ValueError(f"Clean replay did not capture K/V for layer {layer_idx}.")
            source_k, source_v = source_kv
            destination_kv = destination[layer_idx]
            if destination_kv is None:
                destination_k = torch.zeros_like(source_k)  # [1,S_memory,H_kv,D]
                destination_v = torch.zeros_like(source_v)  # [1,S_memory,H_kv,D]
                destination[layer_idx] = (destination_k, destination_v)
            else:
                destination_k, destination_v = destination_kv
            layer_cache_indexes = cache_indexes.to(device=source_k.device, dtype=torch.long)  # [S_write]
            selected_k = torch.index_select(source_k, 1, layer_cache_indexes)  # [1,S_write,H_kv,D]
            selected_v = torch.index_select(source_v, 1, layer_cache_indexes)  # [1,S_write,H_kv,D]
            destination_k.index_copy_(1, layer_cache_indexes, selected_k)  # [1,S_memory,H_kv,D]
            destination_v.index_copy_(1, layer_cache_indexes, selected_v)  # [1,S_memory,H_kv,D]

    def capture_prefill(
        self,
        *,
        session: MultiviewTransferARSession,
        pack: PackedSequence,
        destination: MultiviewTransferARKVCache,
        memory_layout: MultiviewTransferARMemoryLayout,
    ) -> None:
        """Capture prefill without allowing it to read partially built AR memory."""
        scratch_cache: MultiviewTransferARKVCache = [None] * len(destination)
        self.capture_memory(
            pack=pack,
            cache=scratch_cache,
            memory_seq_len=session.memory_seq_len,
            write_indexes=memory_layout.prefill_source_token_indexes,
            write_offset=0,
            cache_write_indexes=memory_layout.prefill_cache_token_indexes,
        )
        self.merge_memory(
            destination=destination,
            source=scratch_cache,
            cache_indexes=memory_layout.prefill_cache_token_indexes,
        )

    def prime_control_cache(
        self,
        *,
        session: MultiviewTransferARSession,
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        conditional_text_tokens: list[int],
        unconditional_text_tokens: list[int] | None,
    ) -> None:
        """Capture control/condition K/V against the currently materialized target history."""
        session.control_frame_ranges[:] = [(0, session.frames_per_view)]
        conditional_pack = self.build_prefill_pack(
            sequence_plans=sequence_plans,
            gen_data_clean=gen_data_clean,
            text_tokens=conditional_text_tokens,
            materialized_target_frame_ranges=session.history_frame_ranges,
        )
        unconditional_pack = None
        if session.cfg_active:
            if unconditional_text_tokens is None:
                raise ValueError("CFG multiview transfer AR requires unconditional text tokens.")
            unconditional_pack = self.build_prefill_pack(
                sequence_plans=sequence_plans,
                gen_data_clean=gen_data_clean,
                text_tokens=unconditional_text_tokens,
                materialized_target_frame_ranges=session.history_frame_ranges,
            )
        self.capture_control_cache(
            session=session,
            conditional_pack=conditional_pack,
            unconditional_pack=unconditional_pack,
        )

    def capture_control_cache(
        self,
        *,
        session: MultiviewTransferARSession,
        conditional_pack: PackedSequence,
        unconditional_pack: PackedSequence | None,
    ) -> None:
        """Capture already-built control prefills into every active branch cache."""
        session.control_frame_ranges[:] = [(0, session.frames_per_view)]
        memory_layout = self.build_memory_layout(session)
        if session.cfgp_enabled:
            local_pack = conditional_pack if int(self.host.parallel_dims.cfgp_rank) == 0 else unconditional_pack
            if local_pack is None:
                raise RuntimeError("CFGP multiview transfer AR is missing its rank-local prefill pack.")
            self.capture_prefill(
                session=session,
                pack=local_pack,
                destination=session.conditional_cache,
                memory_layout=memory_layout,
            )
            return
        self.capture_prefill(
            session=session,
            pack=conditional_pack,
            destination=session.conditional_cache,
            memory_layout=memory_layout,
        )
        if session.unconditional_cache is not None:
            if unconditional_pack is None:
                raise RuntimeError("CFG multiview transfer AR is missing its unconditional prefill pack.")
            self.capture_prefill(
                session=session,
                pack=unconditional_pack,
                destination=session.unconditional_cache,
                memory_layout=memory_layout,
            )

    def make_memory(
        self,
        session: MultiviewTransferARSession,
        *,
        branch: _ARBranch = "conditional",
    ) -> FlexARMemoryState:
        """Create the read-only Flex memory view for a denoising branch."""
        if session.cfgp_enabled:
            if self.host.parallel_dims is None:
                raise ValueError("CFGP multiview transfer AR requires initialized parallel dimensions.")
            cfgp_rank = int(self.host.parallel_dims.cfgp_rank)
            if cfgp_rank not in (0, 1):
                raise ValueError(f"CFGP multiview transfer AR requires rank 0 or 1, got {cfgp_rank}.")
            local_branch: _ARBranch = "conditional" if cfgp_rank == 0 else "unconditional"
            if branch != local_branch:
                raise ValueError(f"CFGP rank {cfgp_rank} owns the {local_branch} branch, not {branch}.")
            # CFGP stores the rank-local branch in the only allocated cache.
            cache = session.conditional_cache
        else:
            cache = session.conditional_cache if branch == "conditional" else session.unconditional_cache
        if cache is None:
            raise ValueError(f"Multiview transfer AR has no {branch} cache.")
        return FlexARMemoryState(
            num_layers=int(self.host.net.num_hidden_layers),  # type: ignore[attr-defined]
            memory_seq_len=session.memory_seq_len,
            cache=cache,
        )

    def commit_clean_chunk(
        self,
        *,
        session: MultiviewTransferARSession,
        denoised_chunk: torch.Tensor,  # [1,C,V*chunk_len,H,W]
        chunk_start: int,
        chunk_end: int,
        conditional_text_tokens: list[int],
        unconditional_text_tokens: list[int] | None,
        fps_vision: list[float],
    ) -> None:
        """Write one finalized target chunk into branch caches and advance history."""
        memory_layout = self.build_memory_layout(session)
        conditional_pack = self.build_current_pack(
            vision_latent=denoised_chunk,
            text_tokens=conditional_text_tokens,
            fps_vision=fps_vision,
            num_views=session.num_views,
            frames_per_view=session.frames_per_view,
            chunk_start=chunk_start,
            memory_layout=memory_layout,
            current_role="clean_target",
        )
        unconditional_pack = None
        if session.cfg_active:
            if unconditional_text_tokens is None:
                raise ValueError("CFG multiview transfer AR requires unconditional text tokens.")
            unconditional_pack = self.build_current_pack(
                vision_latent=denoised_chunk,
                text_tokens=unconditional_text_tokens,
                fps_vision=fps_vision,
                num_views=session.num_views,
                frames_per_view=session.frames_per_view,
                chunk_start=chunk_start,
                memory_layout=memory_layout,
                current_role="clean_target",
            )
        target_shape = session.token_shapes[1]
        chunk_len = chunk_end - chunk_start
        spatial_tokens = target_shape[1] * target_shape[2]
        chunk_token_count = session.num_views * chunk_len * spatial_tokens
        write_indexes = torch.arange(
            chunk_token_count,
            device=session.target_condition_mask.device,
            dtype=torch.long,
        )  # [chunk_tokens]
        cache_write_indexes = memory_layout.target_cache_token_indexes((chunk_start, chunk_end))  # [chunk_tokens]
        if session.cfgp_enabled:
            local_pack = conditional_pack if int(self.host.parallel_dims.cfgp_rank) == 0 else unconditional_pack
            if local_pack is None:
                raise RuntimeError("CFGP multiview transfer AR is missing its rank-local clean chunk pack.")
            self.capture_memory(
                pack=local_pack,
                cache=session.conditional_cache,
                memory_seq_len=session.memory_seq_len,
                write_indexes=write_indexes,
                write_offset=0,
                cache_write_indexes=cache_write_indexes,
            )
        else:
            self.capture_memory(
                pack=conditional_pack,
                cache=session.conditional_cache,
                memory_seq_len=session.memory_seq_len,
                write_indexes=write_indexes,
                write_offset=0,
                cache_write_indexes=cache_write_indexes,
            )
            if session.unconditional_cache is not None:
                if unconditional_pack is None:
                    raise RuntimeError("CFG multiview transfer AR is missing its unconditional clean chunk pack.")
                self.capture_memory(
                    pack=unconditional_pack,
                    cache=session.unconditional_cache,
                    memory_seq_len=session.memory_seq_len,
                    write_indexes=write_indexes,
                    write_offset=0,
                    cache_write_indexes=cache_write_indexes,
                )
        session.history_frame_ranges.append((chunk_start, chunk_end))

    @staticmethod
    def slice_chunk(
        vision_tokens: torch.Tensor,  # [1,C,V*T,H,W]
        *,
        num_views: int,
        frames_per_view: int,
        chunk_start: int,
        chunk_end: int,
    ) -> torch.Tensor:  # [1,C,V*chunk_len,H,W]
        """Slice one synchronized camera-major chunk from every view."""
        if vision_tokens.ndim != 5 or vision_tokens.shape[0] != 1:
            raise ValueError(f"Expected [1,C,V*T,H,W] latents, got shape {tuple(vision_tokens.shape)}.")
        if num_views < 1 or vision_tokens.shape[2] != num_views * frames_per_view:
            raise ValueError(
                f"Expected camera-major latent_t={num_views * frames_per_view}, got {vision_tokens.shape[2]}."
            )
        if not 0 <= chunk_start < chunk_end <= frames_per_view:
            raise ValueError(f"Invalid synchronized chunk [{chunk_start},{chunk_end}) for T={frames_per_view}.")
        return torch.cat(
            [
                vision_tokens[
                    :,
                    :,
                    view_idx * frames_per_view + chunk_start : view_idx * frames_per_view + chunk_end,
                ]
                for view_idx in range(num_views)
            ],
            dim=2,
        )  # [1,C,V*chunk_len,H,W]

    @staticmethod
    def scatter_chunk(
        destination: torch.Tensor,  # [1,C,V*T,H,W]
        chunk: torch.Tensor,  # [1,C,V*chunk_len,H,W]
        *,
        num_views: int,
        frames_per_view: int,
        chunk_start: int,
        chunk_end: int,
    ) -> None:
        """Write one synchronized camera-major chunk into its per-view ranges."""
        chunk_len = chunk_end - chunk_start
        if destination.ndim != 5 or destination.shape[2] != num_views * frames_per_view:
            raise ValueError(f"Invalid multiview destination shape {tuple(destination.shape)}.")
        if chunk.ndim != 5 or chunk.shape[2] != num_views * chunk_len:
            raise ValueError(f"Invalid multiview chunk shape {tuple(chunk.shape)}.")
        for view_idx in range(num_views):
            source_start = view_idx * chunk_len
            target_start = view_idx * frames_per_view + chunk_start
            destination[:, :, target_start : target_start + chunk_len].copy_(
                chunk[:, :, source_start : source_start + chunk_len]
            )  # [1,C,chunk_len,H,W]

    @staticmethod
    def snapshot_replay_context(
        session: MultiviewTransferARSession,
        *,
        text_tokens: list[int],
        fps_vision: list[float],
    ) -> MultiviewTransferARReplayContext:
        """Snapshot layout metadata while sharing the detached cache storage."""
        return MultiviewTransferARReplayContext(
            cache=session.conditional_cache,
            token_shapes=session.token_shapes,
            target_condition_mask=session.target_condition_mask,
            num_views=session.num_views,
            frames_per_view=session.frames_per_view,
            frames_per_chunk=session.frames_per_chunk,
            condition_count=session.condition_count,
            history_frame_ranges=tuple(session.history_frame_ranges),
            memory_seq_len=session.memory_seq_len,
            text_tokens=tuple(text_tokens),
            fps_vision=tuple(fps_vision),
        )

    def build_replay_pack_and_memory(
        self,
        *,
        context: MultiviewTransferARReplayContext,
        vision_latent: torch.Tensor,  # [1,C,V*chunk_len,H,W]
        chunk_start: int,
        timestep: float,
    ) -> tuple[PackedSequence, FlexARMemoryState]:
        """Rebuild one differentiable current-chunk input from a backend replay context."""
        target_condition_ranges = [(0, context.condition_count)] if context.condition_count else []
        memory_layout = build_multiview_transfer_ar_memory_layout(
            token_shapes=list(context.token_shapes),
            target_condition_mask=context.target_condition_mask,
            num_views=context.num_views,
            frames_per_chunk=context.frames_per_chunk,
            control_frame_ranges=[(0, context.frames_per_view)],
            target_condition_frame_ranges=target_condition_ranges,
            history_frame_ranges=list(context.history_frame_ranges),
            memory_seq_len=context.memory_seq_len,
            device=context.target_condition_mask.device,
        )
        pack = self.build_current_pack(
            vision_latent=vision_latent,
            text_tokens=list(context.text_tokens),
            fps_vision=list(context.fps_vision),
            num_views=context.num_views,
            frames_per_view=context.frames_per_view,
            chunk_start=chunk_start,
            memory_layout=memory_layout,
            current_role="current_target",
        )
        if pack.vision is None:
            raise ValueError("Multiview transfer replay requires packed vision tokens.")
        num_vision_patches = len(pack.vision.mse_loss_indexes)
        pack.vision.timesteps = torch.full(
            (num_vision_patches,),
            timestep,
            device=self.host.tensor_kwargs["device"],
            dtype=torch.float32,
        )  # [N_noisy_vision]
        memory = FlexARMemoryState(
            num_layers=int(self.host.net.num_hidden_layers),  # type: ignore[attr-defined]
            memory_seq_len=context.memory_seq_len,
            cache=context.cache,
        )
        return pack, memory
