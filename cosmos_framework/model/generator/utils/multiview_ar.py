# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared RGB and joint RGB/LiDAR multiview Transfer autoregressive backends and replay context."""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

import torch
import torch.distributed as dist

from cosmos_framework.model.generator.mot.causal_flex_attention import (
    _ROLE_CLEAN_TARGET,
    _ROLE_CURRENT_TARGET,
    FlexQueryMetadata,
    MultiviewTransferARCurrentRole,
    MultiviewTransferARMemoryLayout,
    TeacherForcingFlexMetadata,
    build_multiview_transfer_ar_memory_layout,
)
from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_broadcast_tensor_list
from cosmos_framework.model.generator.omni_mot_model import _broadcast_seed, _per_view_caption_groups
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.utils.kv_cache import (
    JointChunkMemory,
    MultiviewARMemoryState,
    TeacherForcingMemoryState,
)
from cosmos_framework.model.generator.utils.memory import MemoryState
from cosmos_framework.model.generator.utils.rolling_kv.rolling_prompt import RollingPromptSchedule
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan, build_sequence_plans_from_data_batch
from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive
from cosmos_framework.data.generator.sequence_packing.packers import mark_modality_as_clean_condition
from cosmos_framework.utils.generator.multiview import slice_camera_major_frames

if TYPE_CHECKING:
    from cosmos_framework.model.generator.mot.flex_attention import SensorMaskItem
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

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
    text_view_ids: tuple[int, ...] | None = None
    target_view_ids: torch.Tensor | None = None  # [V] physical rig IDs


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
    text_tokens: tuple[tuple[int, ...], ...]
    fps_vision: tuple[float, ...]
    text_view_ids: tuple[int, ...] | None = None
    target_view_ids: torch.Tensor | None = None  # [V] physical rig IDs


class MultiviewTransferARBackend:
    """Own multiview Transfer packing, replay KV layout, cache writes, and replay state."""

    def __init__(self, host: MultiviewTransferARHost) -> None:
        self.host = host

    def build_prefill_pack(
        self,
        *,
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        text_tokens: list[list[int]],
        materialized_target_frame_ranges: Sequence[tuple[int, int]] | None = None,
    ) -> PackedSequence:
        """Build one full-geometry clean prefill pack for cache capture."""
        pack = self.host._pack_input_sequence(
            sequence_plans,
            text_tokens,
            gen_data_clean,
            torch.zeros(1, dtype=torch.float32),  # [1]
        )
        if pack.vision is None:
            raise ValueError("Multiview transfer AR prefill requires packed vision data.")
        original_masks = [mask.clone() for mask in pack.vision.condition_mask]  # list[[V*T,1,1]]
        # Clean prefill must not add a diffusion embedding to materialized RGB history.
        # Keep the original conditioning mask for prefix validation and cache geometry.
        mark_modality_as_clean_condition(pack.vision)
        pack.vision.condition_mask = original_masks
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
        text_view_ids: list[int] | None = None,
    ) -> MultiviewTransferARSession:
        """Allocate backend state from a validated prefill pack."""
        if prefill_pack.vision is None or len(prefill_pack.vision.token_shapes) != 2:
            raise ValueError("Multiview transfer AR requires packed [control, target] vision metadata.")
        flex_backend = getattr(self.host.net, "flex_backend", None)
        maskless_replay = getattr(self.host.net, "teacher_forcing_maskless", False)
        if flex_backend is None and not maskless_replay:
            raise ValueError("Multiview transfer AR requires an initialized FlexAttention backend.")
        if text_view_ids is not None and text_view_ids != list(range(num_views)):
            raise ValueError(
                f"Multiview transfer AR text must cover camera-major views {list(range(num_views))}, "
                f"got {text_view_ids}."
            )
        control_shape, target_shape = prefill_pack.vision.token_shapes
        total_memory_tokens = control_shape[0] * control_shape[1] * control_shape[2]
        total_memory_tokens += target_shape[0] * target_shape[1] * target_shape[2]
        kv_alignment = 1 if maskless_replay else int(flex_backend.block_size[1])
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
            text_view_ids=tuple(text_view_ids) if text_view_ids is not None else None,
            target_view_ids=prefill_pack.vision_view_ids[1]
            if prefill_pack.vision_view_ids is not None
            else None,  # [V]
        )

    @staticmethod
    def _current_pack_text_tokens(
        text_tokens: list[list[int]],
        text_view_ids: tuple[int, ...] | None,
    ) -> list[int] | list[list[int]]:
        """Restore the sample-level or per-view AR text payload shape."""
        if text_view_ids is None:
            if len(text_tokens) != 1:
                raise ValueError(f"Sample-level multiview AR text requires one caption, got {len(text_tokens)}.")
            return text_tokens[0]
        if len(text_tokens) != len(text_view_ids):
            raise ValueError(
                f"Per-view multiview AR text carries {len(text_tokens)} captions but {len(text_view_ids)} view IDs."
            )
        return text_tokens

    @staticmethod
    def build_memory_layout(session: MultiviewTransferARSession) -> MultiviewTransferARMemoryLayout:
        """Resolve the current fixed-slot replay memory layout."""
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
        text_tokens: list[list[int]],
        text_view_ids: tuple[int, ...] | None,
        fps_vision: list[float],
        num_views: int,
        frames_per_view: int,
        chunk_start: int,
        memory_layout: MultiviewTransferARMemoryLayout,
        current_role: MultiviewTransferARCurrentRole,
        vision_view_ids: torch.Tensor | None = None,  # [V] physical rig IDs, distinct from caption-local text_view_ids
    ) -> PackedSequence:
        """Pack one synchronized chunk at its shared camera-local mRoPE positions."""
        if vision_latent.shape[2] % num_views != 0:
            raise ValueError(
                f"Multiview transfer chunk latent_t={vision_latent.shape[2]} must be divisible by num_views={num_views}."
            )
        chunk_len = vision_latent.shape[2] // num_views
        temporal_positions = torch.arange(
            chunk_start,
            chunk_start + chunk_len,
            dtype=torch.float32,
        ).repeat(num_views)  # [V*chunk_len]
        ar_text_tokens = self._current_pack_text_tokens(text_tokens, text_view_ids)
        pack = pack_input_sequence_autoregressive(
            vision_latent=vision_latent,
            action_latent=None,
            text_tokens=ar_text_tokens,
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
            text_view_ids=list(text_view_ids) if text_view_ids is not None else None,
        )
        pack.vision_view_ids = [vision_view_ids] if vision_view_ids is not None else None  # list[[V]] or None
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
        memory = MultiviewARMemoryState(
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
        conditional_text_tokens: list[list[int]],
        unconditional_text_tokens: list[list[int]] | None,
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
    ) -> MultiviewARMemoryState:
        """Create the read-only multiview memory view for a denoising branch."""
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
        return MultiviewARMemoryState(
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
        conditional_text_tokens: list[list[int]],
        unconditional_text_tokens: list[list[int]] | None,
        fps_vision: list[float],
    ) -> None:
        """Write one finalized target chunk into branch caches and advance history."""
        memory_layout = self.build_memory_layout(session)
        conditional_pack = self.build_current_pack(
            vision_latent=denoised_chunk,
            text_tokens=conditional_text_tokens,
            text_view_ids=session.text_view_ids,
            vision_view_ids=session.target_view_ids,
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
                text_view_ids=session.text_view_ids,
                vision_view_ids=session.target_view_ids,
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
        text_tokens: list[list[int]],
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
            text_tokens=tuple(tuple(tokens) for tokens in text_tokens),
            fps_vision=tuple(fps_vision),
            text_view_ids=session.text_view_ids,
            target_view_ids=session.target_view_ids,
        )

    def build_replay_pack_and_memory(
        self,
        *,
        context: MultiviewTransferARReplayContext,
        vision_latent: torch.Tensor,  # [1,C,V*chunk_len,H,W]
        chunk_start: int,
        timestep: float,
    ) -> tuple[PackedSequence, MultiviewARMemoryState]:
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
            text_tokens=[list(tokens) for tokens in context.text_tokens],
            text_view_ids=context.text_view_ids,
            vision_view_ids=context.target_view_ids,
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
        memory = MultiviewARMemoryState(
            num_layers=int(self.host.net.num_hidden_layers),  # type: ignore[attr-defined]
            memory_seq_len=context.memory_seq_len,
            cache=context.cache,
        )
        return pack, memory


class JointARCache:
    """Handle persistent-K/V layout and attention metadata for a joint AR pipeline.

    This coordinator maps current RGB/LiDAR chunks to cache slots, restores their
    original position IDs, and builds token labels for maskless attention.
    JointChunkMemory owns the actual per-layer K/V buffers.

    Each layer's K and V buffers have shape [1,S,H_kv/CP,D], where S is the allocated
    RGB+LiDAR token capacity, including unwritten slots.
    Storage along S and its matching metadata are ordered as follows::

        K/V slots:      [ RGB control | RGB target | LiDAR control | LiDAR target ]
        Metadata slots: [ RGB control | RGB target | LiDAR control | LiDAR target ]
                              same slot index identifies the same token

        Within each RGB item:   [view 0: frames_0,frame_1..][view 1: ...]...
        Within each LiDAR item: [frames 0,1...]

    Each frame contains contiguous spatial tokens.
    Unwritten slots have padding roles and are excluded from attention groups.
    """

    reference: PackedSequence
    memory: JointChunkMemory
    metadata: dict[str, torch.Tensor] | None
    current_metadata: dict[str, torch.Tensor] | None
    item_layouts: list[tuple[int, int, int, int]]
    rgb_frames: tuple[int, ...]
    lidar_frames: tuple[int, ...]
    spans: list[tuple[int, int, int]]

    def __init__(self, reference: PackedSequence, num_layers: int) -> None:
        self.reference = reference
        assert reference.vision is not None and reference.lidar is not None
        views = reference.num_views_per_vision_item
        assert views is not None
        if len(reference.vision.token_shapes) != 2 or len(reference.lidar.token_shapes) != 2 or len(views) != 2:
            raise ValueError("Joint AR cache requires one control and one target item for each of RGB and LiDAR.")
        rgb_control_shape, rgb_target_shape = reference.vision.token_shapes
        lidar_control_shape, lidar_target_shape = reference.lidar.token_shapes
        # Cache order is RGB control, RGB target, LiDAR control, LiDAR target.
        # RGB items have per-item camera counts; LiDAR has no camera-view axis.
        items = (
            (rgb_control_shape, views[0]),
            (rgb_target_shape, views[1]),
            (lidar_control_shape, 1),
            (lidar_target_shape, 1),
        )
        self.item_layouts = []
        capacity = 0
        for (total_frames, height, width), num_views in items:
            slots = total_frames // num_views
            self.item_layouts.append((capacity, num_views, slots, height * width))
            capacity += num_views * slots * height * width
        self.memory = JointChunkMemory(num_layers, capacity)
        self.metadata = None
        self.current_metadata = None
        self.rgb_frames = ()
        self.lidar_frames = ()
        self.spans = []

    def prepare(self, pack: PackedSequence, rgb_frames: tuple[int, ...], lidar_frames: tuple[int, ...]) -> None:
        """Prepare current-chunk cache writes, positional encoding, and attention visibility.

        1. Compute spans mapping current-chunk K/V into persistent history for
           the branch's clean forward.
        2. Restore current tokens' original position IDs from the full reference
           pack before RoPE runs. Packing only a chunk can give its tokens local
           positions. Restoring the original IDs keeps current Q/K in the same
           positional coordinate system as the cached historical keys.
        3. Record original RGB/LiDAR frame IDs and attach this coordinator to the
           pack. The network hook uses those IDs and stored history labels to
           build attention visibility for current tokens and cached history in
           both noisy denoising and clean forwards.

        Position IDs preserve positional encoding; metadata preserves attention
        visibility. This method does not write K/V or enable cache writes.
        """
        self.rgb_frames, self.lidar_frames = rgb_frames, lidar_frames
        self.spans = []
        source = 0
        for modality_name, frame_ids in (("vision", rgb_frames), ("lidar", lidar_frames)):
            modality = getattr(pack, modality_name)
            original = getattr(self.reference, modality_name)
            assert modality is not None and original is not None
            current_offset = original_offset = 0
            for local_item, (total_frames, height, width) in enumerate(original.token_shapes):
                item = local_item + (0 if modality_name == "vision" else 2)
                base, views, slots, spatial = self.item_layouts[item]
                original_frames = total_frames // views
                selected = [view * original_frames + frame for view in range(views) for frame in frame_ids]
                frame_tensor = torch.tensor(
                    selected, dtype=torch.long, device=original.sequence_indexes.device
                )  # [V*C]
                original_indexes = original.sequence_indexes[
                    original_offset : original_offset + total_frames * spatial
                ]  # [V*T*S]
                selected_indexes = original_indexes.reshape(total_frames, spatial)[frame_tensor].flatten()  # [V*C*S]
                length = len(selected) * spatial
                current_indexes = modality.sequence_indexes[current_offset : current_offset + length]  # [V*C*S]
                pack.position_ids[:, current_indexes] = self.reference.position_ids[:, selected_indexes].to(
                    pack.position_ids.device
                )  # [3,V*C*S]
                for view in range(views):
                    for frame in frame_ids:
                        destination = base + (view * slots + frame) * spatial
                        if (
                            self.spans
                            and self.spans[-1][0] + self.spans[-1][2] == source
                            and self.spans[-1][1] + self.spans[-1][2] == destination
                        ):
                            previous = self.spans[-1]
                            self.spans[-1] = (previous[0], previous[1], previous[2] + spatial)
                        else:
                            self.spans.append((source, destination, spatial))
                        source += spatial
                current_offset += length
                original_offset += total_frames * spatial
        setattr(pack, "joint_ar_cache", self)

    def augment_metadata(
        self,
        metadata: TeacherForcingFlexMetadata,
        sensor_items: list[list[SensorMaskItem]],
        chunk_size: int,
        *,
        noisy: bool,
    ) -> TeacherForcingFlexMetadata:
        """Build attention-visibility metadata for current tokens and persistent history.

        The returned metadata describes current sensor queries and keys ordered as follows::

            Query labels: [current sensor chunk]
            Key labels:   [text | current sensor chunk | persistent history]

        The network uses this metadata to build the maskless attention plan.
        """
        names = [field.name for field in fields(FlexQueryMetadata)]
        current = {name: getattr(metadata.query, name).clone() for name in names}  # each [Q]
        if noisy:
            current["token_role_id"] = torch.where(
                current["token_role_id"] == _ROLE_CLEAN_TARGET,
                _ROLE_CURRENT_TARGET,
                current["token_role_id"],
            )  # [Q]
        offset = 0
        for item_index, item in enumerate(sensor_items[0]):
            frame_ids = self.rgb_frames if item_index < 2 else self.lidar_frames
            spatial = item.token_shape[1] * item.token_shape[2]
            logical = (
                torch.tensor(frame_ids, dtype=torch.long, device=current["frame_id"].device)
                .repeat(item.num_views)
                .repeat_interleave(spatial)
            )  # [V*C*S]
            length = logical.numel()
            current["frame_id"][offset : offset + length] = logical  # [V*C*S]
            current["timestamp"][offset : offset + length] = logical * item.seconds_per_frame  # [V*C*S]
            current["causal_step_id"][offset : offset + length] = torch.ceil(
                logical.double() * item.seconds_per_frame / (chunk_size * sensor_items[0][0].seconds_per_frame) - 1e-5
            ).long()  # [V*C*S]
            offset += length
        self.current_metadata = current
        values = {}
        for name in names:
            parts = [getattr(metadata, name)[: metadata.num_und], current[name]]  # [UND], [Q]
            if self.metadata is not None:
                parts.append(self.metadata[name])  # [M]
            values[name] = torch.cat(parts)  # [UND+Q+M]
        return replace(metadata, **values, query=FlexQueryMetadata(**current))

    def commit_metadata(self) -> None:
        """Publish occupied slots only after every layer has written its clean K/V."""
        assert self.current_metadata is not None
        # Stored targets are clean memory, never noisy keys in later chunks.
        self.current_metadata["is_noisy"] = torch.zeros_like(self.current_metadata["is_noisy"])  # [Q]
        if self.metadata is None:
            self.metadata = {
                name: torch.full(
                    (self.memory.memory_seq_len,),
                    False if value.dtype == torch.bool else -1,
                    device=value.device,
                    dtype=value.dtype,
                )
                for name, value in self.current_metadata.items()
            }  # each [M]
        for name, value in self.current_metadata.items():  # value: [Q]
            for source, destination, length in self.spans:
                self.metadata[name][destination : destination + length] = value[source : source + length]  # [L]
        self.memory.spans = None


@dataclass(frozen=True)
class JointARChunk:
    """A complete training-time causal chunk on the shared RGB/LiDAR clock."""

    step: int
    vision_frames: tuple[int, ...]
    lidar_frames: tuple[int, ...]
    vision_prefix_end: int
    lidar_prefix_end: int


def joint_ar_chunks(
    *,
    vision_frames: int,
    lidar_frames: int,
    vision_seconds_per_frame: float,
    lidar_seconds_per_frame: float,
    frames_per_chunk: int,
) -> list[JointARChunk]:
    """Match the teacher-forcing singleton-zero and shared-time chunk boundaries."""
    if min(vision_frames, lidar_frames, frames_per_chunk) < 1:
        raise ValueError("Joint AR requires positive frame counts and chunk size")
    if not all(math.isfinite(value) and value > 0 for value in (vision_seconds_per_frame, lidar_seconds_per_frame)):
        raise ValueError("Joint AR requires finite positive sensor periods")
    chunk_seconds = frames_per_chunk * vision_seconds_per_frame
    vision_steps = [math.ceil(frame / frames_per_chunk - 1e-5) for frame in range(vision_frames)]
    lidar_steps = [math.ceil(frame * lidar_seconds_per_frame / chunk_seconds - 1e-5) for frame in range(lidar_frames)]
    chunks: list[JointARChunk] = []
    for step in sorted(set(vision_steps) | set(lidar_steps)):
        chunks.append(
            JointARChunk(
                step=step,
                vision_frames=tuple(frame for frame, value in enumerate(vision_steps) if value == step),
                lidar_frames=tuple(frame for frame, value in enumerate(lidar_steps) if value == step),
                vision_prefix_end=sum(value <= step for value in vision_steps),
                lidar_prefix_end=sum(value <= step for value in lidar_steps),
            )
        )
    return chunks


def _conditioned_target(
    latent: torch.Tensor,  # [1,C,T,H,W]
    mask: torch.Tensor,  # [T,1,1]
) -> torch.Tensor:  # [1,C,T,H,W]
    """Discard every unconditioned target value before any model forward."""
    condition = mask.to(device=latent.device, dtype=torch.bool).reshape(1, 1, -1, 1, 1)  # [1,1,T,1,1]
    return torch.where(condition, latent, torch.zeros_like(latent))  # [1,C,T,H,W]


def _validate_persistent_chunks(chunks: list[JointARChunk], vision_condition: int, lidar_condition: int) -> None:
    """Check whether persistent K/V can replace replay for the complete chunk schedule.

    Persistent K/V reuses stored clean history instead of rebuilding it through replay,
    but cannot currently handle:

    1. A causal chunk missing RGB or LiDAR frames; both sensors must be present.
    2. User-provided RGB/LiDAR target conditioning extending beyond the first
       causal chunk. Later conditioned queries may read earlier noisy target
       representations in replay; substituting stored clean K/V is not equivalent.

    For either case, use the replay K/V mechanism by setting joint_ar_use_persistent_kv=False.
    This check raises rather than silently falling back to replay.
    """
    if any(not chunk.vision_frames or not chunk.lidar_frames for chunk in chunks):
        raise ValueError(
            "Persistent joint K/V currently requires both sensors (RGB and LiDAR) in every causal chunk. "
            "Use the replay K/V mechanism: set joint_ar_use_persistent_kv=False."
        )
    if vision_condition > chunks[0].vision_prefix_end or lidar_condition > chunks[0].lidar_prefix_end:
        raise ValueError(
            "Persistent joint K/V currently supports target conditioning only in the first causal chunk; "
            "later conditioning can read noisy replay history. Use the replay K/V mechanism: "
            "set joint_ar_use_persistent_kv=False."
        )


def _prefix_data(
    data: GenerationDataClean,
    *,
    vision_target: torch.Tensor,  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target: torch.Tensor,  # [1,Cl,Tl,Hl,Wl]
    num_views: int,
    chunk: JointARChunk,
) -> GenerationDataClean:
    """Keep complete sensor chunks, with no future target or control positions."""
    assert data.x0_tokens_vision is not None and data.x0_tokens_lidar is not None
    if data.temporal_positions_vision is not None:
        raise ValueError("Joint AR currently requires the training latent-index temporal positions")
    return replace(
        data,
        raw_state_vision=None,
        raw_state_lidar=None,
        x0_tokens_vision=[
            slice_camera_major_frames(
                data.x0_tokens_vision[0], num_views, 0, chunk.vision_prefix_end
            ),  # [1,Cv,V*Tv_prefix,Hv,Wv]
            slice_camera_major_frames(vision_target, num_views, 0, chunk.vision_prefix_end),  # [1,Cv,V*Tv_prefix,Hv,Wv]
        ],
        x0_tokens_lidar=[
            data.x0_tokens_lidar[0][:, :, : chunk.lidar_prefix_end],  # [1,Cl,Tl_prefix,Hl,Wl]
            lidar_target[:, :, : chunk.lidar_prefix_end],  # [1,Cl,Tl_prefix,Hl,Wl]
        ],
    )


def _build_replay_branch(
    host: OmniMoTCausalModel,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    text_tokens: list[list[int]],
) -> tuple[PackedSequence, TeacherForcingMemoryState]:
    """Use the actual joint training clean/noisy passes, including caption isolation."""
    pack = host._pack_input_sequence(plans, text_tokens, data, torch.zeros(1, dtype=torch.float32))  # timestep: [1]
    pack.to_cuda()
    host._cast_generated_tokens_to_precision(pack)
    host._validate_teacher_forcing_pack(pack)
    # Replay recomputes clean K/V for the complete causal prefix.
    memory = host._build_clean_tf_cache(
        net=host.net,
        packed_sequence=pack,
        gen_data_clean=data,
        memory_info={},
        detach_clean_kv=True,
    )
    return pack, memory


def _robot_multiview_action_prefix(
    data: GenerationDataClean,
    generated_views: list[torch.Tensor],
    *,
    num_views: int,
    frames_per_view: int,
    end: int,
    temporal_factor: int,
) -> GenerationDataClean:
    """Materialize one complete robot prefix in the exact training layout."""
    assert data.x0_tokens_action is not None and len(data.x0_tokens_action) == 1
    action = data.x0_tokens_action[0]
    actions_per_view = (frames_per_view - 1) * temporal_factor
    prefix_actions_per_view = (end - 1) * temporal_factor
    prefix_action = torch.cat(
        [
            action[view * actions_per_view : view * actions_per_view + prefix_actions_per_view]
            for view in range(num_views)
        ],
        dim=0,
    )
    prefix_domains = data.action_domain_id
    if prefix_domains is not None:
        domain = prefix_domains[0].reshape(-1)
        if domain.numel() == num_views * actions_per_view:
            domain = torch.cat(
                [
                    domain[view * actions_per_view : view * actions_per_view + prefix_actions_per_view]
                    for view in range(num_views)
                ]
            )
            prefix_domains = [domain]
    return replace(
        data,
        raw_state_vision=None,
        raw_state_action=[prefix_action],
        x0_tokens_vision=[view[:, :, :end] for view in generated_views],
        x0_tokens_action=[prefix_action],
        action_domain_id=prefix_domains,
    )


def iter_robot_multiview_action_ar(
    host: OmniMoTCausalModel,
    *,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    conditional_text: list[list[int]],
    unconditional_text: list[list[int]] | None,
    guidance: float,
    seed: int,
    num_steps: int,
    shift: float,
    normalize_cfg: bool,
    sampler_mode: Literal["rf", "distilled"],
    distilled_num_steps: int | None,
    max_num_frames: int | None,
) -> Generator[dict[str, Any], None, None]:
    """Generate heterogeneous robot camera views with full-prefix replay.

    Each denoising chunk uses the same per-view three-way replay plus
    same-instant inter-view RGB attention as Stage-1 training. Rebuilding the
    complete prefix is deliberate for this no-training sanity path: it avoids
    introducing a second, approximate cache layout while preserving the latest
    replay policy exactly.
    """
    if len(plans) != 1 or not plans[0].has_vision or not plans[0].has_action:
        raise ValueError("Robot multiview AR requires one RGB+action forward-dynamics sample.")
    if plans[0].condition_frame_indexes_vision != [0]:
        raise ValueError(
            "Robot multiview AR requires synchronized frame 0 as its only RGB condition; "
            f"got {plans[0].condition_frame_indexes_vision}."
        )
    if not host._uses_robot_multiview_threeway() or host.config.compile.enabled:
        raise ValueError("Robot multiview AR requires eager robot three-way attention.")
    if host._get_teacher_forcing_kv_implementation() != "multiview_threeway_kv":
        raise ValueError(
            "Robot multiview AR requires the Stage-1 multiview_threeway_kv path; "
            "legacy maskless/Flex replay is intentionally unsupported."
        )
    if host.config.multiview_action_conditioning is None:
        raise ValueError("Robot multiview AR requires multiview_action_conditioning.")
    if data.x0_tokens_vision is None or data.x0_tokens_action is None:
        raise ValueError("Robot multiview AR requires encoded RGB and action streams.")

    num_views = len(host.config.multiview_action_conditioning.view_codes)
    if data.num_vision_items_per_sample != [num_views] or len(data.x0_tokens_vision) != num_views:
        raise ValueError(
            f"Robot multiview AR requires one RGB item per configured view ({num_views}); "
            f"got {data.num_vision_items_per_sample}."
        )
    frames_per_view = data.x0_tokens_vision[0].shape[2]
    if any(view.ndim != 5 or view.shape[0] != 1 or view.shape[2] != frames_per_view for view in data.x0_tokens_vision):
        raise ValueError("Robot multiview AR views must be aligned [1,C,T,H,W] tensors.")
    temporal_factor = host.tokenizer_vision_gen.temporal_compression_factor or 4
    actions_per_view = (frames_per_view - 1) * temporal_factor
    action = data.x0_tokens_action[0]
    if action.ndim != 2 or action.shape[0] != num_views * actions_per_view:
        raise ValueError(
            "Robot multiview AR requires camera-major full-rate actions: "
            f"got {tuple(action.shape)}, expected rows={num_views * actions_per_view}."
        )
    output_frames = frames_per_view if max_num_frames is None else min(frames_per_view, max_num_frames)
    if output_frames < 1:
        raise ValueError(f"max_num_frames must leave at least one frame, got {max_num_frames}.")
    chunk_size = int(host.config.teacher_forcing_frames_per_chunk)
    if output_frames > 1 and (output_frames - 1) % chunk_size:
        raise ValueError(
            f"Robot multiview AR must match the Stage-1 [1,C,C,...] partition: got T={output_frames}, C={chunk_size}."
        )
    if normalize_cfg and guidance != 1.0:
        raise ValueError("Robot multiview replay sanity currently requires normalize_cfg=False.")

    generated_views = [torch.zeros_like(view, dtype=torch.float32) for view in data.x0_tokens_vision]
    for generated, source in zip(generated_views, data.x0_tokens_vision, strict=True):
        generated[:, :, :1] = source[:, :, :1].to(device=generated.device, dtype=generated.dtype)
    yield {"vision_items": [view[:, :, :1] for view in generated_views]}
    if output_frames == 1:
        return

    for chunk_start in range(1, output_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, output_frames)
        chunk_len = chunk_end - chunk_start
        forward_data = _robot_multiview_action_prefix(
            data,
            generated_views,
            num_views=num_views,
            frames_per_view=frames_per_view,
            end=chunk_end,
            temporal_factor=temporal_factor,
        )
        cond_pack, cond_memory = _build_replay_branch(host, plans, forward_data, conditional_text)
        uncond_branch = (
            _build_replay_branch(host, plans, forward_data, unconditional_text)
            if guidance != 1.0 and unconditional_text is not None
            else None
        )

        view_shapes = [(1, view.shape[1], chunk_len, view.shape[3], view.shape[4]) for view in generated_views]
        view_sizes = [math.prod(shape) for shape in view_shapes]
        generator = torch.Generator(device=generated_views[0].device).manual_seed(seed + chunk_start)
        initial_noise = torch.randn(
            (1, sum(view_sizes)),
            device=generated_views[0].device,
            dtype=torch.float32,
            generator=generator,
        )

        def branch_velocity(
            pack: PackedSequence,
            memory: TeacherForcingMemoryState,
            noise: torch.Tensor,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
            noisy_views: list[torch.Tensor] = []
            offset = 0
            for view, shape, size in zip(forward_data.x0_tokens_vision or [], view_shapes, view_sizes, strict=True):
                noisy = view.clone()
                noisy[:, :, chunk_start:chunk_end] = noise[:, offset : offset + size].reshape(shape)
                noisy_views.append(noisy)
                offset += size
            host._update_inference_pack_template(
                packed_sequence=pack,
                noise_x_vision=noisy_views,
                noise_x_lidar=None,
                noise_x_radar=None,
                noise_x_action=None,
                noise_x_sound=None,
                timestep=timestep,
            )
            prediction = host.denoise(data_batch_packed=pack, memory=memory)
            predicted_views = prediction["preds_vision"]
            if len(predicted_views) != num_views:
                raise ValueError(f"Expected {num_views} RGB predictions, got {len(predicted_views)}.")
            return torch.cat(
                [prediction[:, :, chunk_start:chunk_end].reshape(1, -1) for prediction in predicted_views],
                dim=1,
            )

        def velocity(noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            conditional = branch_velocity(cond_pack, cond_memory, noise, timestep)
            if uncond_branch is None:
                return conditional
            unconditional = branch_velocity(*uncond_branch, noise, timestep)
            return unconditional + guidance * (conditional - unconditional)

        denoised = host._run_ar_sampler(
            velocity,
            initial_noise,
            sampler_mode=sampler_mode,
            num_steps=num_steps,
            shift=shift,
            seed=seed,
            sample_idx=chunk_start,
            num_frames=output_frames,
            distilled_num_steps=distilled_num_steps,
            schedule_frame_idx=chunk_start,
        )
        if not torch.isfinite(denoised).all():
            raise FloatingPointError(f"Robot multiview AR produced nonfinite latents at chunk {chunk_start}.")
        offset = 0
        clean_chunks: list[torch.Tensor] = []
        for generated, shape, size in zip(generated_views, view_shapes, view_sizes, strict=True):
            clean = denoised[:, offset : offset + size].reshape(shape)
            generated[:, :, chunk_start:chunk_end] = clean
            clean_chunks.append(clean)
            offset += size
        for local_frame in range(chunk_len):
            yield {"vision_items": [chunk[:, :, local_frame : local_frame + 1] for chunk in clean_chunks]}


def _current_chunk_data(
    data: GenerationDataClean,
    vision_target: torch.Tensor,  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target: torch.Tensor,  # [1,Cl,Tl,Hl,Wl]
    num_views: int,
    chunk: JointARChunk,
) -> GenerationDataClean:
    """Materialize only this shared-clock chunk, keeping control/target pairing."""
    assert data.x0_tokens_vision is not None and data.x0_tokens_lidar is not None
    frames = vision_target.shape[2] // num_views
    if data.temporal_positions_vision is not None:
        raise ValueError("Joint AR currently requires the training latent-index temporal positions")
    indexes = [view * frames + frame for view in range(num_views) for frame in chunk.vision_frames]
    return replace(
        data,
        raw_state_vision=None,
        raw_state_lidar=None,
        x0_tokens_vision=[
            data.x0_tokens_vision[0][:, :, indexes],
            vision_target[:, :, indexes],
        ],  # each [1,Cv,V*C_rgb,Hv,Wv]
        x0_tokens_lidar=[
            data.x0_tokens_lidar[0][:, :, list(chunk.lidar_frames)],
            lidar_target[:, :, list(chunk.lidar_frames)],
        ],  # each [1,Cl,C_lidar,Hl,Wl]
    )


def _build_persistent_branch(
    host: OmniMoTCausalModel,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    text: list[list[int]],
    cache: JointARCache,
    chunk: JointARChunk,
    *,
    clean: bool = False,
) -> tuple[PackedSequence, JointChunkMemory]:
    """Prepare current-chunk inputs for attention against persistent history K/V."""
    local_plans = [
        replace(
            plans[0],
            condition_frame_indexes_vision=[
                i for i, frame in enumerate(chunk.vision_frames) if frame in plans[0].condition_frame_indexes_vision
            ],
            condition_frame_indexes_lidar=[
                i for i, frame in enumerate(chunk.lidar_frames) if frame in plans[0].condition_frame_indexes_lidar
            ],
        )
    ]
    pack = host._pack_input_sequence(local_plans, text, data, torch.zeros(1))  # timestep: [1]
    host._validate_teacher_forcing_pack(pack)
    assert pack.vision is not None and pack.lidar is not None
    original_masks = [
        mask.clone() for modality in (pack.vision, pack.lidar) for mask in modality.condition_mask
    ]  # list of [V*C,1,1]
    cache.prepare(pack, chunk.vision_frames, chunk.lidar_frames)
    if clean:
        for modality in (pack.vision, pack.lidar):
            mark_modality_as_clean_condition(modality)
    pack.teacher_forcing_pass = "clean" if clean else "noisy"
    pack.teacher_forcing_original_condition_masks_sensors = original_masks
    pack.to_cuda()
    host._cast_generated_tokens_to_precision(pack)
    return pack, cache.memory


def _commit_persistent_joint_chunk(
    host: OmniMoTCausalModel,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    branches: list[tuple[list[list[int]], JointARCache]],
    chunk: JointARChunk,
) -> None:
    """Store a finalized chunk's clean K/V and attention metadata in persistent history.

    Prior persistent history metadata stays unchanged.
    """
    for text, cache in branches:
        pack, memory = _build_persistent_branch(host, plans, data, text, cache, chunk, clean=True)
        cache.memory.spans = cache.spans
        try:
            host.denoise(data_batch_packed=pack, memory=memory)
            cache.commit_metadata()
        finally:
            cache.memory.spans = None


def sample_joint_transfer_ar(
    host: OmniMoTCausalModel,
    *,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    conditional_text: list[list[int]],
    unconditional_text: list[list[int]] | None,
    guidance: float,
    seed: int,
    num_steps: int,
    shift: float,
    sampler_mode: Literal["rf", "distilled"] | None = None,
    distilled_num_steps: int | None = None,
    prompt_schedule: RollingPromptSchedule | None = None,
    joint_ar_use_persistent_kv: bool = False,
) -> dict[str, list[torch.Tensor]]:  # returns vision: [1,Cv,V*Tv,Hv,Wv], lidar: [1,Cl,Tl,Hl,Wl]
    """Joint AR sampling with reference replay or opt-in persistent clean history.

    Both targets are denoised together. Each clean replay contains only conditions
    and previously generated values; ungenerated targets are zero placeholders.
    The existing noisy-pass mask reads only strictly earlier clean chunks, and its
    current-target edges stay inside the current chunk. A complete prefix is used
    so controls can recompute their training-time history-dependent K/V. This is
    deliberately a separate inference path: it does not alter training or the
    optimized RGB-only AR cache.

    ``joint_ar_use_persistent_kv=True`` selects current-chunk-only maskless
    execution. Each finalized chunk is committed once; later denoising reads
    those immutable keys/values instead of recomputing historical queries.
    This temporary opt-in keeps replay available during numerical qualification.
    """
    fixed_sampler = getattr(host.config, "fixed_step_sampler_config", None)
    if sampler_mode is None:
        sampler_mode = "distilled" if fixed_sampler is not None else "rf"
    if sampler_mode not in ("rf", "distilled"):
        raise ValueError("Joint AR sampler_mode must be 'rf' or 'distilled'.")
    if sampler_mode == "rf" and fixed_sampler is not None:
        raise ValueError("A joint student checkpoint requires sampler_mode='distilled'.")
    if sampler_mode == "distilled":
        if fixed_sampler is None:
            raise ValueError("Joint distilled sampling requires a fixed-step student configuration.")
    if len(plans) != 1 or not plans[0].has_lidar or not plans[0].has_vision:
        raise ValueError("Joint AR requires exactly one RGB+LiDAR sequence")
    if plans[0].has_action or plans[0].has_sound:
        raise ValueError("Joint AR supports only RGB and LiDAR targets")
    if not host._uses_multiview_replay_kv() or host.config.compile.enabled:
        raise ValueError("Joint AR requires eager multiview teacher forcing")
    if host.parallel_dims is not None and host.parallel_dims.cfgp_enabled:
        raise ValueError("Joint AR currently uses serial CFG branches; set cfg_parallel_shard_degree=1")
    policy = host._get_teacher_forcing_replay_policy()
    if policy.control_visibility != "causal":
        raise ValueError("Joint AR prefix replay requires causal controls")
    if data.num_vision_items_per_sample != [2] or data.num_lidar_items_per_sample != [2]:
        raise ValueError("Joint AR requires control and target items for each sensor modality")
    if data.x0_tokens_vision is None or data.x0_tokens_lidar is None:
        raise ValueError("Joint AR requires encoded camera and LiDAR streams")
    views = data.num_views_per_vision_item
    if views is None or len(views) != 2 or views[0] < 1 or views[0] != views[1]:
        raise ValueError("Joint AR requires matching per-camera VAE metadata")
    num_views = views[0]
    if len(data.x0_tokens_vision) != 2 or len(data.x0_tokens_lidar) != 2:
        raise ValueError("Joint AR requires exactly two encoded items per modality")
    for items in (data.x0_tokens_vision, data.x0_tokens_lidar):
        if items[0].ndim != 5 or items[0].shape[0] != 1 or items[0].shape != items[1].shape:
            raise ValueError("Joint AR requires aligned [1,C,T,H,W] control and target latents")
    if data.x0_tokens_vision[1].shape[2] % num_views:
        raise ValueError("Joint AR camera-major latent count must be divisible by its view count")
    if guidance != 1.0 and unconditional_text is None:
        raise ValueError("Joint AR guidance requires unconditional captions")
    if seed < 0 or num_steps < 1 or not math.isfinite(guidance) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Joint AR requires a nonnegative seed and finite valid sampling settings")

    # Rank-local VAE encoding can differ even with identical pixels and seeds.
    # CP shards must read one canonical set of controls and conditioned targets
    # before the initial pack and every subsequent clean/noisy replay.
    # Keep the existing Flex inference input behavior unchanged.
    if host._get_teacher_forcing_kv_implementation() == "multiview_maskless_kv":
        context_parallel_broadcast_tensor_list(data.x0_tokens_vision, host.parallel_dims)  # each [1,Cv,V*Tv,Hv,Wv]
        context_parallel_broadcast_tensor_list(data.x0_tokens_lidar, host.parallel_dims)  # each [1,Cl,Tl,Hl,Wl]

    if getattr(host.config, "rolling_kv_cache_chunks", None) is not None:
        if joint_ar_use_persistent_kv:
            raise ValueError(
                "rolling_kv_cache_chunks uses a separate cache path; disable --joint-ar-use-persistent-kv."
            )
        from cosmos_framework.model.generator.utils.rolling_kv.rolling_transfer_ar import (
            sample_rolling_joint_transfer_ar,
        )

        return sample_rolling_joint_transfer_ar(
            host,
            plans=plans,
            data=data,
            conditional_text=conditional_text,
            unconditional_text=unconditional_text,
            guidance=guidance,
            seed=seed,
            num_steps=num_steps,
            shift=shift,
            sampler_mode=sampler_mode,
            distilled_num_steps=distilled_num_steps,
            prompt_schedule=prompt_schedule,
        )

    if prompt_schedule is not None:
        raise ValueError("Segmented prompts require rolling KV inference.")

    initial_pack = host._pack_input_sequence(plans, conditional_text, data, torch.zeros(1))  # timestep: [1]
    if initial_pack.vision is None or initial_pack.lidar is None:
        raise ValueError("The joint inference pack omitted a required modality")
    # Keep clock and prefix helpers independent of full transformer construction.
    from cosmos_framework.model.generator.mot.causal_cosmos3_vfm_network import build_interactive_multiview_mask_items

    sensor_items = build_interactive_multiview_mask_items(initial_pack)
    if len(sensor_items) != 1 or len(sensor_items[0]) != 4:
        raise ValueError("Joint AR requires the training [RGB control, RGB target, LiDAR control, LiDAR target] layout")
    rgb_control, rgb_item, lidar_control, lidar_item = sensor_items[0]
    if not (rgb_control.is_control and lidar_control.is_control) or rgb_item.is_control or lidar_item.is_control:
        raise ValueError("Joint AR sensor control/target roles differ from the training layout")
    vision_condition = multiview_conditioned_prefix_length(initial_pack.vision.condition_mask[1], num_views)
    lidar_condition = multiview_conditioned_prefix_length(initial_pack.lidar.condition_mask[1], 1)
    vision_target = _conditioned_target(data.x0_tokens_vision[1], initial_pack.vision.condition_mask[1]).to(
        device=host.tensor_kwargs["device"], dtype=torch.float32
    )  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target = _conditioned_target(data.x0_tokens_lidar[1], initial_pack.lidar.condition_mask[1]).to(
        device=host.tensor_kwargs["device"], dtype=torch.float32
    )  # [1,Cl,Tl,Hl,Wl]
    frames_per_view = vision_target.shape[2] // num_views
    chunks = joint_ar_chunks(
        vision_frames=frames_per_view,
        lidar_frames=lidar_target.shape[2],
        vision_seconds_per_frame=rgb_item.seconds_per_frame,
        lidar_seconds_per_frame=lidar_item.seconds_per_frame,
        frames_per_chunk=int(host.config.teacher_forcing_frames_per_chunk),
    )
    persistent_branches = None
    if joint_ar_use_persistent_kv:
        if host._get_teacher_forcing_kv_implementation() != "multiview_maskless_kv":
            raise ValueError("Persistent joint K/V requires maskless attention")
        if host.config.kv_cache_inference_size is not None or host.config.attention_sink_size != 0:
            raise ValueError("Persistent joint K/V is unbounded in this phase; leave the window unset and sink at zero")
        _validate_persistent_chunks(chunks, vision_condition, lidar_condition)
        persistent_branches = [(conditional_text, JointARCache(initial_pack, host.net.num_hidden_layers))]
        if guidance != 1.0 and unconditional_text is not None:
            reference = host._pack_input_sequence(plans, unconditional_text, data, torch.zeros(1))  # timestep: [1]
            persistent_branches.append((unconditional_text, JointARCache(reference, host.net.num_hidden_layers)))
    del initial_pack
    for chunk in chunks:
        _sample_joint_chunk(
            host,
            plans=plans,
            data=data,
            conditional_text=conditional_text,
            unconditional_text=unconditional_text,
            guidance=guidance,
            seed=seed,
            num_steps=num_steps,
            shift=shift,
            vision_target=vision_target,
            lidar_target=lidar_target,
            num_views=num_views,
            frames_per_view=frames_per_view,
            vision_condition=vision_condition,
            lidar_condition=lidar_condition,
            chunk=chunk,
            sampler_mode=sampler_mode,
            distilled_num_steps=distilled_num_steps,
            persistent_branches=persistent_branches,
            # The last chunk need not be retained: no later chunk reads its K/V.
            retain_history=persistent_branches is not None and chunk is not chunks[-1],
        )
    return {"vision": [vision_target], "lidar": [lidar_target]}


def _sample_joint_chunk(
    host: OmniMoTCausalModel,
    *,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    conditional_text: list[list[int]],
    unconditional_text: list[list[int]] | None,
    guidance: float,
    seed: int,
    num_steps: int,
    shift: float,
    vision_target: torch.Tensor,  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target: torch.Tensor,  # [1,Cl,Tl,Hl,Wl]
    num_views: int,
    frames_per_view: int,
    vision_condition: int,
    lidar_condition: int,
    chunk: JointARChunk,
    sampler_mode: Literal["rf", "distilled"] = "rf",
    distilled_num_steps: int | None = None,
    persistent_branches: list[tuple[list[list[int]], JointARCache]] | None = None,
    retain_history: bool = True,
) -> None:
    """Denoise the current RGB/LiDAR chunk and cache its clean K/V if persistent history is needed."""
    vision_frames = [frame for frame in chunk.vision_frames if frame >= vision_condition]
    lidar_frames = [frame for frame in chunk.lidar_frames if frame >= lidar_condition]
    if not vision_frames and not lidar_frames:
        if persistent_branches is not None and retain_history:
            current = _current_chunk_data(data, vision_target, lidar_target, num_views, chunk)
            _commit_persistent_joint_chunk(host, plans, current, persistent_branches, chunk)
        return
    forward_data = (
        _prefix_data(data, vision_target=vision_target, lidar_target=lidar_target, num_views=num_views, chunk=chunk)
        if persistent_branches is None
        else _current_chunk_data(data, vision_target, lidar_target, num_views, chunk)
    )
    assert forward_data.x0_tokens_vision is not None and forward_data.x0_tokens_lidar is not None
    vision_indexes = [view * chunk.vision_prefix_end + frame for view in range(num_views) for frame in vision_frames]
    final_vision_indexes = [view * frames_per_view + frame for view in range(num_views) for frame in vision_frames]
    local_lidar_frames = lidar_frames
    if persistent_branches is not None:
        vision_indexes = [
            view * len(chunk.vision_frames) + i
            for view in range(num_views)
            for i, frame in enumerate(chunk.vision_frames)
            if frame >= vision_condition
        ]
        local_lidar_frames = [i for i, frame in enumerate(chunk.lidar_frames) if frame >= lidar_condition]
    vision_shape = (1, vision_target.shape[1], len(vision_indexes), *vision_target.shape[-2:])
    lidar_shape = (1, lidar_target.shape[1], len(lidar_frames), *lidar_target.shape[-2:])
    vision_size = math.prod(vision_shape)
    lidar_size = math.prod(lidar_shape)
    generator = torch.Generator(device=vision_target.device).manual_seed(seed + chunk.step)
    initial_noise = torch.randn(
        (1, vision_size + lidar_size), device=vision_target.device, dtype=torch.float32, generator=generator
    )  # [1,D_vision+D_lidar]
    if persistent_branches is None:
        cond_pack, cond_memory = _build_replay_branch(host, plans, forward_data, conditional_text)
        uncond_branch = (
            _build_replay_branch(host, plans, forward_data, unconditional_text)
            if guidance != 1.0 and unconditional_text is not None
            else None
        )
    else:
        branches = [
            _build_persistent_branch(host, plans, forward_data, text, cache, chunk)
            for text, cache in persistent_branches
        ]
        cond_pack, cond_memory = branches[0]
        uncond_branch = branches[1] if len(branches) > 1 else None

    def branch_velocity(
        pack: PackedSequence,
        memory: TeacherForcingMemoryState | JointChunkMemory,
        noise: torch.Tensor,  # [1,D_vision+D_lidar]
        timestep: torch.Tensor,  # [1,1]
    ) -> torch.Tensor:  # [1,D_vision+D_lidar]
        assert pack.vision is not None and pack.lidar is not None
        noisy_vision = forward_data.x0_tokens_vision[1].clone()  # [1,Cv,V*Tv_forward,Hv,Wv]
        noisy_lidar = forward_data.x0_tokens_lidar[1].clone()  # [1,Cl,Tl_forward,Hl,Wl]
        noisy_vision[:, :, vision_indexes] = noise[:, :vision_size].reshape(vision_shape)  # [1,Cv,V*Tv_chunk,Hv,Wv]
        noisy_lidar[:, :, local_lidar_frames] = noise[:, vision_size:].reshape(lidar_shape)  # [1,Cl,Tl_chunk,Hl,Wl]
        host._update_inference_pack_template(
            packed_sequence=pack,
            noise_x_vision=[forward_data.x0_tokens_vision[0], noisy_vision],  # list of [1,Cv,V*Tv_forward,Hv,Wv]
            noise_x_lidar=[forward_data.x0_tokens_lidar[0], noisy_lidar],  # list of [1,Cl,Tl_forward,Hl,Wl]
            noise_x_radar=None,
            noise_x_action=None,
            noise_x_sound=None,
            timestep=timestep,  # [1,1]
        )
        prediction = host.denoise(data_batch_packed=pack, memory=memory)
        vision_velocity = prediction["preds_vision"][1]  # [1,Cv,V*Tv_forward,Hv,Wv]
        lidar_velocity = prediction["preds_lidar"][1]  # [1,Cl,Tl_forward,Hl,Wl]
        if vision_velocity.shape != noisy_vision.shape or lidar_velocity.shape != noisy_lidar.shape:
            raise ValueError("Joint model predictions must retain each input item's [1,C,T,H,W] shape")
        # Grid-stream decoding retains its singleton batch dimension. Select time,
        # preserving every channel before flattening for the shared diffusion solver.
        return torch.cat(
            [
                vision_velocity[:, :, vision_indexes].reshape(1, -1),
                lidar_velocity[:, :, local_lidar_frames].reshape(1, -1),
            ],
            dim=1,
        )  # [1,D_vision+D_lidar]

    def velocity(noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # [1,D], [1,1] -> [1,D]
        conditional = branch_velocity(cond_pack, cond_memory, noise, timestep)  # [1,D]
        if uncond_branch is None:
            return conditional  # [1,D]
        unconditional = branch_velocity(*uncond_branch, noise, timestep)  # [1,D]
        return unconditional + guidance * (conditional - unconditional)  # [1,D]

    # Match the joint SF-DMD RGB clock, including a final LiDAR-only chunk.
    # Keep chunk-based noise seeds independent of the schedule's frame index.
    schedule_frame_idx = chunk.vision_frames[0] if chunk.vision_frames else frames_per_view - 1
    denoised = host._run_ar_sampler(
        velocity,
        initial_noise,
        sampler_mode=sampler_mode,
        num_steps=num_steps,
        shift=shift,
        seed=seed,
        sample_idx=chunk.step,
        num_frames=frames_per_view,
        distilled_num_steps=distilled_num_steps,
        schedule_frame_idx=schedule_frame_idx,
    )  # [1,D_vision+D_lidar]
    if not torch.isfinite(denoised).all():
        raise FloatingPointError(f"Joint AR produced nonfinite latents in causal chunk {chunk.step}")
    vision_target[:, :, final_vision_indexes] = denoised[:, :vision_size].reshape(
        vision_shape
    )  # [1,Cv,V*Tv_chunk,Hv,Wv]
    lidar_target[:, :, lidar_frames] = denoised[:, vision_size:].reshape(lidar_shape)  # [1,Cl,Tl_chunk,Hl,Wl]
    if persistent_branches is not None and retain_history:
        current = _current_chunk_data(data, vision_target, lidar_target, num_views, chunk)
        # No later query reads the last chunk; do not run an unused clean forward.
        _commit_persistent_joint_chunk(host, plans, current, persistent_branches, chunk)


def _iter_multiview_logical_frames(
    vision_latent: torch.Tensor,
    *,
    num_views: int,
) -> Generator[torch.Tensor, None, None]:  # vision_latent: [B,C,V*T,H,W], yields [B,C,V,H,W]
    """Yield one camera-major latent time step across all views at a time."""
    if num_views < 1:
        raise ValueError(f"num_views must be positive, got {num_views}.")
    if vision_latent.ndim != 5:
        raise ValueError(f"Expected multiview latent rank 5, got shape {tuple(vision_latent.shape)}.")
    flat_frame_count = int(vision_latent.shape[2])
    if flat_frame_count % num_views != 0:
        raise ValueError(f"Camera-major latent length {flat_frame_count} must be divisible by num_views={num_views}.")
    frames_per_view = flat_frame_count // num_views
    vision_by_view = vision_latent.reshape(  # [B,C,V,T,H,W]
        vision_latent.shape[0],
        vision_latent.shape[1],
        num_views,
        frames_per_view,
        vision_latent.shape[3],
        vision_latent.shape[4],
    )
    for frame_idx in range(frames_per_view):
        logical_frame = vision_by_view[:, :, :, frame_idx : frame_idx + 1].reshape(  # [B,C,V,H,W]
            vision_latent.shape[0],
            vision_latent.shape[1],
            num_views,
            vision_latent.shape[3],
            vision_latent.shape[4],
        )
        yield logical_frame


def multiview_conditioned_prefix_length(
    target_condition_mask: torch.Tensor,
    num_views: int,
    *,
    frames_per_view: int | None = None,
) -> int:
    """Return a synchronized contiguous-prefix length from a camera-major mask."""
    if frames_per_view is None:
        frames_per_view = target_condition_mask.numel() // num_views if num_views > 0 else 0
    expected_frames = num_views * frames_per_view
    if num_views < 1 or frames_per_view < 1 or target_condition_mask.numel() != expected_frames:
        raise ValueError(
            "Expected one target condition value per camera-major frame; "
            f"got shape {tuple(target_condition_mask.shape)} for V={num_views}, T={frames_per_view}."
        )
    condition_grid = target_condition_mask.to(dtype=torch.bool).reshape(num_views, frames_per_view)  # [V,T]
    if not torch.equal(condition_grid, condition_grid[0:1].expand_as(condition_grid)):
        raise ValueError("Multiview transfer AR requires synchronized target conditioning across every view.")
    condition_indexes = torch.nonzero(condition_grid[0], as_tuple=False).squeeze(1)  # [T_condition]
    expected_indexes = torch.arange(condition_indexes.numel(), device=condition_indexes.device)  # [T_condition]
    if not torch.equal(condition_indexes, expected_indexes):
        raise ValueError(
            "Multiview transfer AR only supports contiguous prefix conditioning from frame 0; "
            f"got condition frame indexes {condition_indexes.tolist()}."
        )
    return condition_indexes.numel()


def _submit_multiview_conditioned_prefix(
    target_latent: torch.Tensor,
    *,
    num_views: int,
    frames_per_view: int,
    condition_count: int,
    output_frames: int,
    on_clean_vision_chunk: Callable[[torch.Tensor], None] | None,
) -> torch.Tensor | None:  # target_latent: [B,C,V*T,H,W], returns [B,C,V*T_prefix,H,W] or None
    """Return the camera-major conditioned prefix and optionally submit it for decode."""
    if condition_count == 0:
        return None
    if num_views < 1 or frames_per_view < 1:
        raise ValueError(f"Expected positive multiview dimensions, got V={num_views}, T={frames_per_view}.")
    if target_latent.ndim != 5 or target_latent.shape[2] != num_views * frames_per_view:
        raise ValueError(
            "Expected camera-major target shape [B,C,V*T,H,W], "
            f"got {tuple(target_latent.shape)} for V={num_views}, T={frames_per_view}."
        )
    if not 0 <= condition_count <= frames_per_view:
        raise ValueError(f"condition_count must be in [0, {frames_per_view}], got {condition_count}.")
    if not 1 <= output_frames <= frames_per_view:
        raise ValueError(f"output_frames must be in [1, {frames_per_view}], got {output_frames}.")

    prefix_frame_count = min(condition_count, output_frames)
    conditioned_prefix = slice_camera_major_frames(
        target_latent, num_views, 0, prefix_frame_count
    )  # [B,C,V*T_prefix,H,W]
    if on_clean_vision_chunk is not None:
        on_clean_vision_chunk(conditioned_prefix)
    return conditioned_prefix


def generate_multiview_transfer_ar_chunk(
    host: OmniMoTCausalModel,
    *,
    cond_pack: PackedSequence,
    uncond_pack: PackedSequence | None,
    cond_cache: list[tuple[torch.Tensor, torch.Tensor] | None],
    uncond_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None,
    curr_vision_latent: torch.Tensor,  # [1,C,V*chunk_len,H,W]
    guidance: float,
    num_steps: int,
    shift: float,
    seed: int,
    chunk_start: int,
    num_frames: int,
    normalize_cfg: bool,
    sampler_mode: str,
    distilled_num_steps: int | None,
    memory_seq_len: int,
) -> torch.Tensor:  # [1,C,V*chunk_len,H,W]
    """Denoise one synchronized multiview chunk against the cached K/V suffix."""
    # Imported lazily because the shared model dispatches into this backend.
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    cond_memory = MultiviewARMemoryState(
        num_layers=host.net.num_hidden_layers,
        memory_seq_len=memory_seq_len,
        cache=cond_cache,
    )
    uncond_memory = (
        MultiviewARMemoryState(
            num_layers=host.net.num_hidden_layers,
            memory_seq_len=memory_seq_len,
            cache=uncond_cache,
        )
        if uncond_cache is not None
        else None
    )

    def run_branch(
        pack: PackedSequence,
        noise_vision: torch.Tensor,  # [1,C,V*chunk_len,H,W]
        timestep: torch.Tensor,  # [1,1]
        branch: _ARBranch,
    ) -> torch.Tensor:  # [1,C,V*chunk_len,H,W]
        OmniMoTCausalModel._set_ar_vision_noise(host, pack, noise_vision, timestep)
        cfgp_enabled = host.parallel_dims is not None and host.parallel_dims.cfgp_enabled
        # Under CFGP each rank has one branch-local cache in ``cond_memory``.
        memory = cond_memory if cfgp_enabled or branch == "conditional" else uncond_memory
        assert memory is not None
        output = host.denoise(data_batch_packed=pack, memory=memory)
        return torch.stack(output["preds_vision"])  # [1,C,V*chunk_len,H,W]

    def velocity_fn(noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return OmniMoTCausalModel._predict_ar_velocity_with_cfg(
            host,
            noise_x=noise_x,
            timestep=timestep,
            vision_shape=curr_vision_latent.shape,
            packed_seq=cond_pack,
            packed_seq_uncond=uncond_pack,
            guidance=guidance,
            normalize_cfg=normalize_cfg,
            run_branch=run_branch,
        )  # [1,N_tokens_flat]

    initial_noise = curr_vision_latent.flatten(start_dim=1)  # [1,N_tokens_flat]
    denoised_flat = OmniMoTCausalModel._run_ar_sampler(
        host,
        velocity_fn,
        initial_noise,
        sampler_mode=sampler_mode,
        num_steps=num_steps,
        shift=shift,
        seed=seed,
        sample_idx=chunk_start,
        num_frames=num_frames,
        distilled_num_steps=distilled_num_steps,
    )  # [1,N_tokens_flat]
    return denoised_flat.reshape(curr_vision_latent.shape)  # [1,C,V*chunk_len,H,W]


@torch.no_grad()
def iter_samples_multiview_transfer_autoregressive(
    host: OmniMoTCausalModel,
    *,
    data_batch: dict,
    guidance: float,
    seed: int,
    num_steps: int,
    shift: float,
    normalize_cfg: bool,
    sampler_mode: str,
    distilled_num_steps: int | None,
    sync_num_frames_across_ranks: bool,
    sync_process_group: Any | None,
    max_num_frames: int | None,
    on_clean_vision_chunk: Callable[[torch.Tensor], None] | None,
    has_negative_prompt: bool,
) -> Generator[dict[str, Any], torch.Tensor | None, None]:
    """Generate the target for a two-item camera-major multiview transfer sample."""
    # Imported lazily because the shared model dispatches into this backend.
    from cosmos_framework.model.generator.omni_mot_causal_model import _iter_ar_chunk_ranges

    if not host._uses_multiview_replay_kv():
        raise ValueError("Multiview transfer AR requires replayed two-way multiview teacher-forcing configuration.")
    if host.config.action_gen or host.config.sound_gen:
        raise ValueError("Multiview transfer AR supports vision-only models.")
    sequence_plans = build_sequence_plans_from_data_batch(
        data_batch=data_batch,
        input_video_key=host.input_video_key,
        input_image_key=host.input_image_key,
    )
    if len(sequence_plans) != 1:
        raise ValueError(f"Multiview transfer AR requires batch_size=1, got {len(sequence_plans)} samples.")
    caption_groups = _per_view_caption_groups(data_batch[host.input_caption_key])
    host._apply_inference_caption_plan(sequence_plans, caption_groups)
    vision_condition_indexes = [sequence_plans[0].condition_frame_indexes_vision]
    gen_data_clean = host.get_data_and_condition(
        data_batch,
        vision_condition_indexes=vision_condition_indexes,
        retain_raw_state_vision=False,
    )
    host._release_inference_raw_vision(data_batch, gen_data_clean)
    if gen_data_clean.x0_tokens_vision is None or len(gen_data_clean.x0_tokens_vision) != 2:
        num_items = 0 if gen_data_clean.x0_tokens_vision is None else len(gen_data_clean.x0_tokens_vision)
        raise ValueError(f"Multiview transfer AR requires [control, target], got {num_items} vision items.")
    # Match joint inference: all CP shards must read the owner's encoded
    # controls and conditioned targets, even when local VAE results differ.
    if host._get_teacher_forcing_kv_implementation() == "multiview_maskless_kv":
        context_parallel_broadcast_tensor_list(
            gen_data_clean.x0_tokens_vision, host.parallel_dims
        )  # each [1,C,V*T,H,W]
    if gen_data_clean.num_vision_items_per_sample != [2]:
        raise ValueError(
            "Multiview transfer AR requires one sample with exactly two vision items; "
            f"got {gen_data_clean.num_vision_items_per_sample}."
        )
    if gen_data_clean.num_views_per_vision_item is None or len(gen_data_clean.num_views_per_vision_item) != 2:
        raise ValueError("Multiview transfer AR requires per-camera VAE metadata for both vision items.")
    num_views = gen_data_clean.num_views_per_vision_item[0]
    if gen_data_clean.num_views_per_vision_item != [num_views, num_views]:
        raise ValueError(f"Control and target must use the same views, got {gen_data_clean.num_views_per_vision_item}.")
    control_latent, target_latent = gen_data_clean.x0_tokens_vision
    if control_latent.shape != target_latent.shape:
        raise ValueError(
            f"Control and target latent shapes must match, got {control_latent.shape} and {target_latent.shape}."
        )
    if target_latent.shape[2] % num_views != 0:
        raise ValueError(f"Target latent_t={target_latent.shape[2]} is not divisible by {num_views} views.")
    frames_per_view = target_latent.shape[2] // num_views
    output_frames = frames_per_view if max_num_frames is None else min(frames_per_view, max_num_frames)
    if output_frames < 1:
        raise ValueError(f"max_num_frames must leave at least one frame, got {max_num_frames}.")
    if sync_num_frames_across_ranks and dist.is_available() and dist.is_initialized():
        frame_count = torch.tensor([output_frames], device=target_latent.device, dtype=torch.long)  # [1]
        dist.all_reduce(frame_count, op=dist.ReduceOp.MIN, group=sync_process_group)
        output_frames = int(frame_count.item())

    cond_text_tokens, uncond_text_tokens = host._get_inference_text_tokens(
        data_batch,
        has_negative_prompt,
        caption_groups,
    )
    cfgp_enabled = host.parallel_dims is not None and host.parallel_dims.cfgp_enabled
    if cfgp_enabled:
        seed = _broadcast_seed([seed], host.parallel_dims.cfgp_mesh.get_group(), host.parallel_dims.cfgp_rank)[0]
    cfg_active = guidance != 1.0 or cfgp_enabled
    if getattr(host.config, "rolling_kv_cache_chunks", None) is not None:
        from cosmos_framework.model.generator.utils.rolling_kv.rolling_transfer_ar import (
            iter_rolling_transfer_ar,
        )

        if normalize_cfg and cfg_active:
            raise ValueError("Rolling replay requires normalize_cfg=False when guidance is active.")
        if sampler_mode not in ("rf", "distilled"):
            raise ValueError("Rolling replay sampler_mode must be rf or distilled.")
        for part in iter_rolling_transfer_ar(
            host,
            plans=sequence_plans,
            data=gen_data_clean,
            conditional_text=cond_text_tokens,
            unconditional_text=uncond_text_tokens,
            guidance=guidance,
            seed=seed,
            num_steps=num_steps,
            shift=shift,
            sampler_mode=sampler_mode,
            distilled_num_steps=distilled_num_steps,
            output_frames=output_frames,
            on_clean_vision_chunk=on_clean_vision_chunk,
            prompt_schedule=data_batch.get("rolling_prompt_schedule"),
        ):
            for frame in _iter_multiview_logical_frames(part["vision"], num_views=num_views):
                yield {"vision": frame}
        return
    fps_vision = gen_data_clean.fps_vision.tolist() if gen_data_clean.fps_vision is not None else [24.0]
    backend = host._make_multiview_transfer_ar_backend()

    def build_prefill_pack(
        text_tokens: list[list[int]],
        *,
        materialized_target_frame_ranges: Sequence[tuple[int, int]] | None = None,
    ) -> PackedSequence:
        return backend.build_prefill_pack(
            sequence_plans=sequence_plans,
            gen_data_clean=gen_data_clean,
            text_tokens=text_tokens,
            materialized_target_frame_ranges=materialized_target_frame_ranges,
        )

    cond_prefill = build_prefill_pack(
        cond_text_tokens,
        materialized_target_frame_ranges=[],
    )
    assert cond_prefill.vision is not None
    target_condition_mask = cond_prefill.vision.condition_mask[1]  # [V*T,1,1]
    condition_count = multiview_conditioned_prefix_length(
        target_condition_mask,
        num_views=num_views,
        frames_per_view=frames_per_view,
    )
    generated_target = target_latent.to(**host.tensor_kwargs).clone()  # [1,C,V*T,H,W]
    gen_data_clean.x0_tokens_vision[1] = generated_target
    materialized_condition_count = min(condition_count, output_frames)
    session = backend.create_session(
        prefill_pack=cond_prefill,
        num_views=num_views,
        frames_per_view=frames_per_view,
        condition_count=materialized_condition_count,
        cfg_active=cfg_active,
        cfgp_enabled=cfgp_enabled,
        text_view_ids=sequence_plans[0].text_view_ids,
    )

    controls_read_rgb = host._get_teacher_forcing_replay_policy().controls_read_strict_past_clean_rgb
    if not controls_read_rgb:
        uncond_prefill = None
        if cfg_active:
            assert uncond_text_tokens is not None
            uncond_prefill = build_prefill_pack(
                uncond_text_tokens,
                materialized_target_frame_ranges=[],
            )
        backend.capture_control_cache(
            session=session,
            conditional_pack=cond_prefill,
            unconditional_pack=uncond_prefill,
        )

    conditioned_prefix = _submit_multiview_conditioned_prefix(
        generated_target,
        num_views=num_views,
        frames_per_view=frames_per_view,
        condition_count=condition_count,
        output_frames=output_frames,
        on_clean_vision_chunk=on_clean_vision_chunk,
    )
    if conditioned_prefix is not None:
        for prefix_frame in _iter_multiview_logical_frames(conditioned_prefix, num_views=num_views):
            yield {"vision": prefix_frame}
    for chunk_start, chunk_end in _iter_ar_chunk_ranges(
        condition_count, output_frames, host.config.teacher_forcing_frames_per_chunk
    ):
        chunk_len = chunk_end - chunk_start
        if controls_read_rgb:
            backend.prime_control_cache(
                session=session,
                sequence_plans=sequence_plans,
                gen_data_clean=gen_data_clean,
                conditional_text_tokens=cond_text_tokens,
                unconditional_text_tokens=uncond_text_tokens,
            )
        memory_layout = backend.build_memory_layout(session)
        noise_generator = torch.Generator(device=target_latent.device).manual_seed(seed + chunk_start)
        chunk_noise = torch.empty(
            (
                1,
                target_latent.shape[1],
                num_views * chunk_len,
                target_latent.shape[3],
                target_latent.shape[4],
            ),
            device=target_latent.device,
            dtype=host.tensor_kwargs["dtype"],
        ).normal_(generator=noise_generator)  # [1,C,V*chunk_len,H,W]
        cond_pack = backend.build_current_pack(
            vision_latent=chunk_noise,
            text_tokens=cond_text_tokens,
            text_view_ids=session.text_view_ids,
            vision_view_ids=session.target_view_ids,
            fps_vision=fps_vision,
            num_views=num_views,
            frames_per_view=frames_per_view,
            chunk_start=chunk_start,
            memory_layout=memory_layout,
            current_role="current_target",
        )
        uncond_pack = (
            backend.build_current_pack(
                vision_latent=chunk_noise,
                text_tokens=uncond_text_tokens,
                text_view_ids=session.text_view_ids,
                vision_view_ids=session.target_view_ids,
                fps_vision=fps_vision,
                num_views=num_views,
                frames_per_view=frames_per_view,
                chunk_start=chunk_start,
                memory_layout=memory_layout,
                current_role="current_target",
            )
            if cfg_active
            else None
        )
        denoised_chunk = host._generate_multiview_transfer_ar_chunk(
            cond_pack=cond_pack,
            uncond_pack=uncond_pack,
            cond_cache=session.conditional_cache,
            uncond_cache=session.unconditional_cache,
            curr_vision_latent=chunk_noise,
            guidance=guidance,
            num_steps=num_steps,
            shift=shift,
            seed=seed,
            chunk_start=chunk_start,
            num_frames=output_frames,
            normalize_cfg=normalize_cfg,
            sampler_mode=sampler_mode,
            distilled_num_steps=distilled_num_steps,
            memory_seq_len=session.memory_seq_len,
        )
        backend.scatter_chunk(
            generated_target,
            denoised_chunk,
            num_views=num_views,
            frames_per_view=frames_per_view,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        if on_clean_vision_chunk is not None:
            on_clean_vision_chunk(denoised_chunk)

        if chunk_end < output_frames:
            backend.commit_clean_chunk(
                session=session,
                denoised_chunk=denoised_chunk.to(**host.tensor_kwargs),  # [1,C,V*chunk_len,H,W]
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                conditional_text_tokens=cond_text_tokens,
                unconditional_text_tokens=uncond_text_tokens,
                fps_vision=fps_vision,
            )

        # Expose progress after the chunk is ready for future AR steps. Each
        # event represents one latent time step across every synchronized view.
        for output_frame in _iter_multiview_logical_frames(denoised_chunk, num_views=num_views):
            yield {"vision": output_frame}
