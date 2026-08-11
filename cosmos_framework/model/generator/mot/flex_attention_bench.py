# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Benchmark the multiview FlexAttention mask over the fused ``[UND | GEN]`` key stream.

Measures the costs that ``cosmos3_vfm_network`` pays when ``flex_attention.enabled`` is
on, for a packed multiview batch:

1. ``mask_build`` -- ``build_multiview_block_mask``, paid **once per step**
   (outside the decoder layers, shared by all of them);
2. ``flex_fwd`` / ``flex_fwd_bwd`` -- ``flex_attention``, paid **once per
   decoder layer**;
3. ``flash_fwd`` / ``flash_fwd_bwd`` plus ``mask_build_flash`` -- the same two
   costs on FlexAttention's FlashAttention-4 backend, with ``--flash`` (see below);
4. ``dense_fwd`` / ``dense_fwd_bwd`` -- the varlen baseline it replaces, i.e. every
   GEN token attending to every token of its own sample, causal (understanding) tokens
   included. This is the reference point for the speedup: FlexAttention computes a
   strict subset of those pairs.

Both sides run the shape ``two_way_attention``'s full pass runs: GEN tokens query the
concatenated ``[UND | GEN]`` stream, so the attention is rectangular and the causal
tokens are keys that every GEN token of the same sample reaches. ``--num-causal-tokens``
sets how many of them each sample carries (the caption/reasoner prefix; 301 by default),
and the stream is padded to a key-block multiple exactly as the packer's
``causal_seq_alignment`` does, so the UND/GEN boundary lands on a block boundary and the
gen->und quadrant keeps the fully-unmasked fast path.

The restriction the multiview mask expresses is confined to the GEN->GEN quadrant, so
that quadrant is the whole of the speedup: gen->und is dense for flex and dense alike.
At the default sizes it is also a few hundred keys against 161,920, so passing
``--num-causal-tokens 0`` (the pure GEN->GEN benchmark) barely moves either number --
what the causal prefix mostly buys is that the shape being timed is the shape a step
runs, padding blocks included.

Default scenario (the one this script was written for): 11 cameras at 720p, one
fully-conditioning camera item (the control/clean stream) plus one noisy camera
item (the video being generated), both on the same ``(view, frame)`` grid. That is
the transfer layout the mask was designed for, so the mask keeps

* noisy -> noisy: what ``--noisy-attention-scopes`` admits, by default the full square
  within the sample (the dominant term),
* noisy -> conditioning: same ``(frame, view)`` only,
* conditioning -> conditioning: same ``(frame, view)`` and same item,
* conditioning -> noisy: never.

Noisy attention scopes (``--noisy-attention-scopes``)
-----------------------------------------------------

The noisy square above is the largest quadrant of the mask, so the ``noisy_attention_scope``
config knob -- which restricts it to the query's own camera (``same_view``) or to its own
camera and its own instant (``same_view_or_frame``) -- is the one mask choice that changes
what a step costs rather than only what it may look at. Passing several scopes runs the whole
benchmark once per scope over the identical pack and adds a comparison table of what each one
bought, against the first scope listed:

* ``pairs`` is the ratio of allowed token pairs, i.e. the speedup a mask whose rules landed on
  block boundaries would give: ``V`` for ``same_view``, ``V*F / (F + V - 1)`` for
  ``same_view_or_frame`` over the noisy square, diluted by the quadrants the scope leaves
  alone;
* ``blocks`` is the ratio of *visited blocks*, i.e. what this mask at this block size actually
  leaves the kernel free to skip, and therefore the ratio to read the wall clock against. It
  trails ``pairs`` by whatever the scope's boundaries cost in partially-masked blocks, which
  are charged in full: ``same_view`` keeps nearly all of its saving, since a camera's tokens
  are one contiguous run of the camera-major layout, while ``same_view_or_frame`` gives more
  of it back, since its same-frame stripes are single ``(frame, view)`` cells (920 tokens at
  the default geometry, against 128- or 256-token blocks) and each stripe pays a partial block
  at both ends;
* ``measured`` is the wall clock, and tracking the ``blocks`` column is the property being
  checked. Falling well short of it says the row is no longer attention-bound at that size,
  not that the mask failed to narrow; coming in below ``1.00x`` says the narrower mask did not
  get sparser, which at these sizes should not happen.

Token geometry follows the multiview AV configs: the Wan2.2 tokenizer's 16x spatial and 4x
temporal compression, and ``patch_spatial=2`` from the diffusion expert. A 720x1280 frame
therefore becomes a 45x80 latent, patchified to 23x40 = 920 tokens, and ``F`` pixel frames
per camera become ``1 + (F - 1) // 4`` latent frames. Both factors stay overridable for
other tokenizers, e.g. ``--latent-downsample-factor 8`` for the Wan2.1 4x8x8 VAE. With 11
cameras, 29 pixel frames each and two items that is 2 * 11 * 8 * 920 = 161,920 GEN tokens
in one sample, which the 301 causal tokens extend to 162,304 keys once block-padded to 384
-- so the dense baseline is a 161,920 x 162,304 attention and the numbers below are large
by construction.

Each timed row reports achieved TFLOP/s and the matching MFU, i.e. that throughput
over the device's dense BF16 peak (auto-detected, or set with ``--peak-tflops``).
This is the utilization of *this one operator*, not model MFU: the numerator counts
only the attention matmuls of this one pass, so it says how well the kernel uses the
machine on the work it does, not how much of a training step is useful math.

Three caveats when reading the output. The FlexAttention kernel skips whole blocks, so
its cost -- and hence its TFLOP/s and MFU -- tracks the *visited block* fraction rather
than the (lower) fraction of allowed token pairs; both are reported, per backend, since
the two use different block sizes.

Nor are visited blocks all alike, which is why each mask reports its grid as full, partial
and empty blocks: the empty ones are the computation the mask removes, and the split between
the other two is what the removal cost. A partial block is charged one full tile of math by
the FLOP count above but costs more than a full one to run, because the ``mask_mod`` is
evaluated over every position in it and the unmasked fast path is gone -- twice over in the
backward, which evaluates the mask in both of its passes. So a mask with a high partial
share posts a lower MFU at the same visited-block count, and a narrower
``noisy_attention_scope`` can be genuinely faster while reporting *worse* utilization: it
gets there by turning full blocks into partial ones.

And ``mask_build`` evaluates the ``mask_mod`` over every block pair, so at these sequence
lengths it is not free; it does no attention math, so it has no TFLOP/s or MFU and the
report instead amortizes it over ``--num-layers``.

FlashAttention-4 backend (``--flash``)
--------------------------------------

FlexAttention can lower onto FlashAttention-4 (CuTeDSL) instead of its Triton kernels on
Hopper and Blackwell, via ``kernel_options={"BACKEND": "FLASH"}``. Install the kernels
into the training container with::

    pip install --pre "flash-attn-4[cu13]"   # CUDA 12.x: drop the [cu13] extra

FA4 publishes only pre-releases so far (4.0.0bN), and its ``nvidia-cutlass-dsl`` pin is a
``.dev`` build, so ``--pre`` is required for the resolve to find either.

That pulls ``nvidia-cutlass-dsl`` and JIT-compiles at runtime, so there is no build step,
but it does need a torch new enough to carry the Inductor CuTeDSL codegen (a recent
nightly at the time of writing); on an older torch the ``BACKEND`` kernel option is
rejected and the ``flash_*`` rows report that in the status column rather than failing
the sweep.

Once those hold, a training run picks the backend up on its own:
``flex_attention.resolve_flex_backend`` answers the same question the ``--flash`` rows
assume the answer to, and ``flex_attention.backend="auto"`` (the default) takes it. This
sweep stays explicit because its job is to compare the two, not to run the faster one.

Two things make the flash rows more than a kernel_options flip:

* Block size. FA4 drives its tile scheduler from the block mask, and on Blackwell each
  CTA owns two query tiles (``q_stage=2``), so the smallest region it can skip is 256x128
  rather than the Triton path's 128x128 (Hopper keeps 128x128). The flash rows therefore
  run against their own, coarser mask -- hence the separate ``mask_build_flash`` row and
  the separate visited-block percentage -- and the padded GEN length is raised to that Q
  block so both backends attend over exactly the same tokens.
* Coarser blocks visit more token pairs for the same mask, so a like-for-like read of the
  two backends is the wall-clock (``median(ms)``) columns; their TFLOP/s are each scored
  against the work that backend actually does.

No row requests the log-sum-exp, which is what keeps FA4 in scope at all. ``two_way_attention``
fuses the gen->und keys into this same call rather than merging a second attention term into
it, so a step never asks for the LSE; and the FA4 backward has no rule for differentiating it,
refusing to lower a backward graph that carries a tangent for it -- which asking for the LSE
creates whether or not anything downstream uses it (this benchmark's loss is ``out.sum()``, so
the tangent is zeros, and it still fails). What the rows time is therefore what a step pays.

Two further FA4 limitations, neither of which this mask trips: the backward is not
deterministic when block-sparsity is on, and gradients for buffers captured by a
``score_mod``/``mask_mod`` are unsupported (this mask captures only integer metadata,
which needs no gradient).

Examples:

    # Default: 11 cameras, 720p, 29 pixel frames/camera, 1 cond + 1 noisy item.
    python -m cosmos_framework.model.generator.mot.flex_attention_bench

    # Frame-count scaling study, forward + backward, skipping the (slow) dense baseline.
    python -m cosmos_framework.model.generator.mot.flex_attention_bench \
        --pixel-frames-per-view 9 17 29 57 --backward --skip-dense

    # Triton vs FlashAttention-4, forward + backward, against each other only.
    python -m cosmos_framework.model.generator.mot.flex_attention_bench \
        --flash --backward --skip-dense

    # What the narrower noisy->noisy scopes buy, forward + backward. The dense baseline is
    # identical for every scope and by far the slowest row, so a scope sweep skips it.
    python -m cosmos_framework.model.generator.mot.flex_attention_bench \
        --noisy-attention-scopes all_views same_view_or_frame same_view --backward --skip-dense

    # The same sweep down to the masks alone: their build cost and their sparsity, with no
    # q/k/v allocated and no attention run, so it also fits where the dense shapes do not.
    python -m cosmos_framework.model.generator.mot.flex_attention_bench \
        --noisy-attention-scopes all_views same_view_or_frame same_view --mask-only

    # I2V-style conditioning: first latent frame of every camera kept clean.
    python -m cosmos_framework.model.generator.mot.flex_attention_bench \
        --noisy-cond-frames-per-view 1

    # A longer caption prefix, and the GEN->GEN quadrant on its own for comparison.
    python -m cosmos_framework.model.generator.mot.flex_attention_bench \
        --num-causal-tokens 1024
    python -m cosmos_framework.model.generator.mot.flex_attention_bench \
        --num-causal-tokens 0

    # Mask cost and sparsity only (no q/k/v allocated, no attention run).
    python -m cosmos_framework.model.generator.mot.flex_attention_bench \
        --num-views 11 --pixel-frames-per-view 57 --mask-only

    # 480p, two control streams, batch of two packed samples.
    python -m cosmos_framework.model.generator.mot.flex_attention_bench \
        --resolution-hw 480 832 --num-cond-items 2 --num-samples 2
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
import tyro
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from cosmos_framework.model.attention import attention
from cosmos_framework.configs.base.defaults.flex_attention import NoisyAttentionScope
from cosmos_framework.model.generator.mot.flex_attention import (
    FlexBackend,
    _get_flash_flex_backend,
    _get_triton_flex_backend,
    build_multiview_block_mask,
    flex_attention,
)

# Compression of the Wan2.2 4x16x16 tokenizer (``Wan2pt2VAEInterface``). A config that
# overrides either has to be mirrored with ``--latent-downsample-factor`` /
# ``--temporal-compression-factor``.
WAN2PT2_SPATIAL_COMPRESSION = 16
WAN2PT2_TEMPORAL_COMPRESSION = 4

DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# Dense (non-sparse) BF16 peak per GPU, matching the values the tokenizer performance
# callback logs MFU against. FP16 peaks equal BF16 on all of these parts, so the same
# table serves --dtype fp16. Blackwell Ultra repeats Blackwell's numbers because its uplift
# is in dense NVFP4 and attention (the SFUs), not in BF16; the superchip parts are the
# higher of each pair because they clock their GPU higher than the HGX board does.
# Substrings are matched against torch.cuda.get_device_name in order, so every Grace pairing
# has to precede the board name it contains ("gb300" before "b300"), or a superchip would be
# scored against the board's lower peak.
DENSE_BF16_PEAK_TFLOPS: dict[str, float] = {
    "gb300": 2500.0,
    "b300": 2250.0,
    "gb200": 2500.0,
    "b200": 2250.0,
    "h200": 989.0,
    "h100": 989.0,
    "a100": 312.0,
}


def stream_alignments(backends: Sequence[FlexBackend]) -> tuple[int, int]:
    """``(GEN, UND)`` padding multiples that suit every backend in ``backends``.

    The pack is padded once and all of the backends run against those same tokens, so each
    stream answers to the lcm of what the backends need individually: FA4's query tile where
    they differ (256 on Blackwell against Triton's 128) and the block both step where they do
    not (the key block, which FA4 leaves at 128 -- it buys its coarser tiles in the query
    dimension alone). Padding to the coarser value is what keeps the rows comparable, since
    the backends then differ only in their kernels and in the granularity of their masks.
    """
    return (
        math.lcm(*(backend.full_seq_alignment for backend in backends)),
        math.lcm(*(backend.causal_seq_alignment for backend in backends)),
    )


@dataclass(frozen=True)
class MultiviewScenario:
    """A packed multiview batch, described the way the dataset and packer produce it.

    Every sample owns ``num_noisy_items`` generated items plus ``num_cond_items``
    fully-conditioning control items, all sharing one ``(num_views,
    latent_frames_per_view)`` grid -- the mask requires that, since conditioning
    tokens reach noisy tokens by matching ``(frame, view)``.

    ``noisy_cond_frames_per_view`` models I2V-style conditioning inside a generated
    item: its leading N latent frames of *every* camera are clean, which is what
    ``condition_frame_indexes_vision`` expands to in the camera-major layout.

    ``latent_downsample_factor`` is the dataset-side name for the tokenizer's
    ``spatial_compression_factor``; the multiview AV configs pass the same number to both.

    ``num_causal_tokens`` is the understanding (caption/reasoner) prefix each sample
    contributes to the key stream, which GEN tokens of that sample all attend to. It is
    counted per sample, like every other token knob here.

    ``noisy_attention_scope`` is the one field that changes the mask rather than the pack:
    the tokens, the padding and every other quadrant are identical across scopes, so two
    scenarios differing only in it are timing the same attention under a narrower mask.

    ``seq_alignment`` and ``causal_seq_alignment`` are the packer's two padding multiples,
    ``full_seq_alignment`` and ``causal_seq_alignment``, and they come from the backends
    actually being benchmarked (:func:`stream_alignments`) rather than from any fixed block
    size: the GEN stream supplies the mask's rows and so answers to the query block, the UND
    prefix is keys only and answers to the key block, and padding it is what puts the
    UND/GEN boundary on a block boundary.

    Every field is required: :class:`BenchConfig` is the only thing that builds one of these,
    and it is where the defaults live, documented as the CLI flags they are.
    """

    num_views: int
    pixel_frames_per_view: int
    resolution_hw: tuple[int, int]
    latent_downsample_factor: int
    temporal_compression_factor: int
    patch_spatial: int
    num_noisy_items: int
    num_cond_items: int
    noisy_cond_frames_per_view: int
    num_causal_tokens: int
    num_samples: int
    seq_alignment: int
    causal_seq_alignment: int
    noisy_attention_scope: NoisyAttentionScope

    def __post_init__(self) -> None:
        # Positivity is all these two need here: they are the backends' own block sizes, and
        # build_multiview_block_mask checks each padded length against the block that tiles it.
        if self.seq_alignment < 1:
            raise ValueError(f"seq_alignment must be >= 1, got {self.seq_alignment}.")
        if self.causal_seq_alignment < 1:
            raise ValueError(f"causal_seq_alignment must be >= 1, got {self.causal_seq_alignment}.")
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}.")
        if self.pixel_frames_per_view < 1:
            raise ValueError(f"pixel_frames_per_view must be >= 1, got {self.pixel_frames_per_view}.")
        if self.num_noisy_items < 1:
            raise ValueError(f"num_noisy_items must be >= 1, got {self.num_noisy_items}.")
        if self.num_cond_items < 0:
            raise ValueError(f"num_cond_items must be >= 0, got {self.num_cond_items}.")
        if self.num_causal_tokens < 0:
            raise ValueError(f"num_causal_tokens must be >= 0, got {self.num_causal_tokens}.")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}.")
        if not 0 <= self.noisy_cond_frames_per_view < self.latent_frames_per_view:
            raise ValueError(
                "noisy_cond_frames_per_view must leave at least one noisy frame per camera: "
                f"got {self.noisy_cond_frames_per_view} of {self.latent_frames_per_view} latent frames."
            )

    @property
    def latent_frames_per_view(self) -> int:
        """Causal-VAE latent frame count for one camera clip.

        Mirrors ``Wan2pt2VAEInterface.get_latent_num_frames``, with the compression factor
        taken from the field rather than pinned at 4.
        """
        return 1 + (self.pixel_frames_per_view - 1) // self.temporal_compression_factor

    @property
    def noisy_frames_per_view(self) -> int:
        """Latent frames per camera that are generated, i.e. all but the I2V-style clean ones."""
        return self.latent_frames_per_view - self.noisy_cond_frames_per_view

    @property
    def noisy_cells_in_scope(self) -> int:
        """``(frame, view)`` cells of noisy tokens one noisy query reaches under its scope.

        The noisy grid is ``num_views`` by :attr:`noisy_frames_per_view` and every one of its
        cells holds the same number of noisy tokens, so this count over that grid's size is
        exactly the fraction of the noisy square ``noisy_attention_scope`` keeps: all of it,
        ``1/V`` of it, or the cross through the grid, ``(F + V - 1)/(V*F)``.
        """
        frames = self.noisy_frames_per_view
        return {
            "all_views": self.num_views * frames,
            "same_view": frames,
            "same_view_or_frame": frames + self.num_views - 1,
        }[self.noisy_attention_scope]

    @property
    def patch_hw(self) -> tuple[int, int]:
        """Patchified latent grid per frame, matching ``PackedSequenceBuilder``'s ceil-div."""
        height, width = self.resolution_hw
        latent_h = height // self.latent_downsample_factor
        latent_w = width // self.latent_downsample_factor
        return math.ceil(latent_h / self.patch_spatial), math.ceil(latent_w / self.patch_spatial)

    @property
    def spatial_tokens(self) -> int:
        """GEN tokens contributed by a single latent frame of a single camera."""
        patch_h, patch_w = self.patch_hw
        return patch_h * patch_w

    @property
    def latent_t(self) -> int:
        """Camera-major latent time axis of one item: all frames of view 0, then view 1, ..."""
        return self.num_views * self.latent_frames_per_view

    @property
    def items_per_sample(self) -> int:
        return self.num_noisy_items + self.num_cond_items

    @property
    def tokens_per_item(self) -> int:
        return self.latent_t * self.spatial_tokens

    @property
    def tokens_per_sample(self) -> int:
        return self.items_per_sample * self.tokens_per_item

    @property
    def real_tokens(self) -> int:
        return self.num_samples * self.tokens_per_sample

    @property
    def seq_len(self) -> int:
        """Block-padded GEN sequence length, as ``full_seq_alignment=seq_alignment`` produces."""
        return math.ceil(self.real_tokens / self.seq_alignment) * self.seq_alignment

    @property
    def real_causal_tokens(self) -> int:
        return self.num_samples * self.num_causal_tokens

    @property
    def causal_seq_len(self) -> int:
        """Block-padded UND stream length, as ``causal_seq_alignment`` produces."""
        return math.ceil(self.real_causal_tokens / self.causal_seq_alignment) * self.causal_seq_alignment

    @property
    def kv_seq_len(self) -> int:
        """Length of the fused key/value stream: the padded UND prefix, then the padded GEN stream."""
        return self.causal_seq_len + self.seq_len

    @property
    def token_shape(self) -> tuple[int, int, int]:
        patch_h, patch_w = self.patch_hw
        return (self.latent_t, patch_h, patch_w)

    def mask_blocks(self, backend: FlexBackend) -> tuple[int, int]:
        """``(rows, columns)`` of the block mask ``backend`` builds over this pack.

        Every backend tiles the same tokens at its own block size, so a coarser one has fewer
        rows for the same pack. This grid is what ``build_multiview_block_mask`` evaluates the
        ``mask_mod`` over, i.e. what the ``mask_build`` rows time.
        """
        q_block, kv_block = backend.block_size
        return self.seq_len // q_block, self.kv_seq_len // kv_block

    @property
    def dense_mask_bytes(self) -> int:
        """Bytes a mask build that evaluates every token pair would need.

        ``create_block_mask`` works that way: it fills a ``[1, 1, seq_len, kv_seq_len]``
        bool tensor before reducing it to blocks, one byte per token pair, and Inductor does
        not fuse that intermediate away. ``build_multiview_block_mask`` collapses the
        predicate onto the metadata runs instead, so ``mask_build``'s peak should stay orders
        of magnitude below this; anywhere near it means the dense path is back.
        """
        return self.seq_len * self.kv_seq_len


@dataclass(frozen=True)
class PackInputs:
    """The mask-building arguments a packed batch of this scenario would carry."""

    token_shapes: list[tuple[int, int, int]]
    condition_masks: list[Tensor]  # one [latent_t] bool mask per item
    num_vision_items_per_sample: list[int]
    num_views_per_vision_item: list[int]
    full_q_offsets: Tensor  # [num_samples+1], int32: cumulative per-sample GEN offsets
    causal_offsets: Tensor  # [num_samples+1], int32: cumulative per-sample UND offsets
    dense_q_offsets: Tensor  # [num_samples+1], int32: GEN queries per sample
    dense_kv_offsets: Tensor  # [num_samples+1], int32: UND + GEN keys per sample
    dense_max_seqlen_q: int
    dense_max_seqlen_kv: int


def build_pack_inputs(scenario: MultiviewScenario, device: torch.device) -> PackInputs:
    """Materialize the per-item metadata for ``build_multiview_block_mask``.

    Items are laid out noisy-first then control, matching the transfer packing order;
    the order only names the ``cond_type_id`` values, since every mask rule is gated on
    ``same_sample``.

    The ``dense_*`` offsets describe the same attention for the varlen baseline: GEN
    queries against their own sample's UND and GEN keys, which is what
    ``two_way_attention`` runs against ``get_all_seq`` when no mask is in play. Both
    streams carry trailing block padding, and varlen leaves rows outside its cumulative
    ranges unwritten in both directions, so the last sample's range is stretched to the
    end of each buffer. The alternative -- padding as its own segment, as the packer
    describes it -- needs a segment on both sides, and the two streams are padded
    independently, so it would mean handing a kernel a zero-length query segment. The
    pairs this adds are the padding rows against one sample's keys, a rounding error next
    to the counts below, and they keep every output row written.
    """
    condition_masks: list[Tensor] = []
    for _ in range(scenario.num_samples):
        for _ in range(scenario.num_noisy_items):
            mask = torch.zeros(scenario.latent_t, dtype=torch.bool, device=device)  # [latent_t]
            if scenario.noisy_cond_frames_per_view:
                # Camera-major axis: the leading frames of every view, not the leading
                # frames of the flattened item.
                frames_per_view = scenario.latent_frames_per_view
                for view_idx in range(scenario.num_views):
                    start = view_idx * frames_per_view
                    mask[start : start + scenario.noisy_cond_frames_per_view] = True
            condition_masks.append(mask)
        for _ in range(scenario.num_cond_items):
            condition_masks.append(torch.ones(scenario.latent_t, dtype=torch.bool, device=device))  # [latent_t]

    num_items = scenario.num_samples * scenario.items_per_sample
    offsets = [scenario.tokens_per_sample * i for i in range(scenario.num_samples + 1)]
    causal_offsets = [scenario.num_causal_tokens * i for i in range(scenario.num_samples + 1)]

    keys_per_sample = scenario.num_causal_tokens + scenario.tokens_per_sample
    dense_q_offsets = [scenario.tokens_per_sample * i for i in range(scenario.num_samples)] + [scenario.seq_len]
    dense_kv_offsets = [keys_per_sample * i for i in range(scenario.num_samples)] + [scenario.kv_seq_len]

    def _max_segment(cumulative: list[int]) -> int:
        return max(end - start for start, end in zip(cumulative, cumulative[1:]))

    return PackInputs(
        token_shapes=[scenario.token_shape] * num_items,
        condition_masks=condition_masks,
        num_vision_items_per_sample=[scenario.items_per_sample] * scenario.num_samples,
        num_views_per_vision_item=[scenario.num_views] * num_items,
        full_q_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),  # [num_samples+1]
        causal_offsets=torch.tensor(causal_offsets, dtype=torch.int32, device=device),  # [num_samples+1]
        dense_q_offsets=torch.tensor(dense_q_offsets, dtype=torch.int32, device=device),  # [num_samples+1]
        dense_kv_offsets=torch.tensor(dense_kv_offsets, dtype=torch.int32, device=device),  # [num_samples+1]
        dense_max_seqlen_q=_max_segment(dense_q_offsets),
        dense_max_seqlen_kv=_max_segment(dense_kv_offsets),
    )


def allowed_pair_count(scenario: MultiviewScenario) -> int:
    """Token pairs the multiview mask lets through, counted exactly from the geometry.

    Every ``(frame, view)`` cell holds ``spatial_tokens`` tokens per item. With ``V``
    views, ``F`` frames per view, ``S`` spatial tokens, ``C_n`` noisy items, ``C_c``
    control items and ``c`` leading clean frames per view inside each noisy item, per
    sample:

    * noisy -> noisy covers the share of the noisy square that
      ``noisy_attention_scope`` keeps: each of the ``V * (F - c)`` noisy cells reaches
      :attr:`MultiviewScenario.noisy_cells_in_scope` of them, and every cell holds
      ``C_n * S`` noisy tokens;
    * noisy -> conditioning is confined to the query's own cell, which holds
      ``C_c * S`` control tokens;
    * conditioning -> conditioning is confined to the cell *and* the item, so each
      conditioning item contributes ``S**2`` per cell it appears in: control items in
      all ``V * F`` cells, noisy items only in their ``V * c`` clean cells;
    * conditioning -> noisy contributes nothing;
    * every GEN token, noisy or conditioning, reaches all of its sample's causal tokens,
      the one quadrant the mask leaves unrestricted.
    """
    views = scenario.num_views
    frames = scenario.latent_frames_per_view
    spatial = scenario.spatial_tokens
    clean = scenario.noisy_cond_frames_per_view
    noisy_items = scenario.num_noisy_items
    cond_items = scenario.num_cond_items

    noisy_cells = views * (frames - clean)
    cell_noisy_tokens = noisy_items * spatial
    noisy_to_noisy = noisy_cells * scenario.noisy_cells_in_scope * cell_noisy_tokens**2
    noisy_to_cond = noisy_cells * cell_noisy_tokens * (cond_items * spatial)
    cond_to_cond = (views * frames * cond_items + views * clean * noisy_items) * spatial**2
    gen_to_und = scenario.tokens_per_sample * scenario.num_causal_tokens
    return scenario.num_samples * (noisy_to_noisy + noisy_to_cond + cond_to_cond + gen_to_und)


def dense_pair_count(scenario: MultiviewScenario) -> int:
    """Token pairs the dense baseline computes: each sample's GEN queries against all its keys.

    Block-diagonal per sample and rectangular within a block, since the keys are the
    sample's causal tokens as well as its GEN tokens.
    """
    keys_per_sample = scenario.num_causal_tokens + scenario.tokens_per_sample
    return scenario.num_samples * scenario.tokens_per_sample * keys_per_sample


@dataclass(frozen=True)
class MaskBlocks:
    """How a built mask divides its block grid, which is where the saving is visible.

    Every block of the ``grid_blocks`` the mask spans ends up in one of three states, and each
    costs the kernel something different:

    * ``full`` -- fully unmasked, run on the kernel's no-mask path;
    * ``partial`` -- cut through by the rules, so the same tile of math *plus* an evaluation of
      the ``mask_mod`` over every position in it, of which only some results survive;
    * ``empty`` -- skipped outright, and the whole of what the mask buys.

    :attr:`visited_pairs` charges full and partial blocks alike, which is what makes it the
    honest cost proxy for achieved TFLOP/s and MFU, with :func:`allowed_pair_count` as the
    lower bound a perfectly block-aligned mask would reach.

    :attr:`partial_fraction` is the reason a narrower ``noisy_attention_scope`` can post a
    lower MFU than the scope it beats. Narrowing turns full blocks into empty ones, but it also
    turns some into partial ones -- the boundaries of the rule land inside blocks -- and those
    are both more expensive than a full block and less useful, so the visited-pair count that
    scores them buys less time than it did for the wider scope.
    """

    partial: int
    full: int
    grid_blocks: int
    q_block_size: int
    kv_block_size: int

    @property
    def blocks(self) -> int:
        """Blocks the kernel visits: the full ones and the partial ones."""
        return self.partial + self.full

    @property
    def empty(self) -> int:
        """Blocks the kernel skips entirely."""
        return self.grid_blocks - self.blocks

    @property
    def visited_pairs(self) -> int:
        return self.blocks * self.q_block_size * self.kv_block_size

    @property
    def partial_fraction(self) -> float:
        """Share of the *visited* blocks that the rules cut through."""
        return self.partial / self.blocks if self.blocks else float("nan")

    @property
    def skipped_fraction(self) -> float:
        """Share of the whole grid the mask skips, i.e. the block sparsity."""
        return self.empty / self.grid_blocks if self.grid_blocks else float("nan")


def count_mask_blocks(block_mask: BlockMask) -> MaskBlocks:
    """Count ``block_mask``'s full, partial and (by subtraction) empty blocks.

    The block size and the sequence lengths are read off the mask rather than assumed, since
    the FA4 backend needs a coarser block (256x128 on Blackwell) and therefore both visits more
    pairs and spans a smaller grid for the same rules.
    """
    full_blocks = getattr(block_mask, "full_kv_num_blocks", None)
    q_block_size, kv_block_size = block_mask.BLOCK_SIZE
    q_len, kv_len = block_mask.seq_lengths
    return MaskBlocks(
        partial=int(block_mask.kv_num_blocks.sum()),
        full=0 if full_blocks is None else int(full_blocks.sum()),
        grid_blocks=math.ceil(q_len / q_block_size) * math.ceil(kv_len / kv_block_size),
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )


def mask_stats_suffix(backend_name: str) -> str:
    """Stats-key suffix for the mask built at the ``backend_name`` backend's block size.

    The Triton mask owns the unsuffixed keys, so a consumer that never passes ``--flash`` sees
    the same stats it always did.
    """
    return "" if backend_name == "triton" else f"_{backend_name}"


def resolve_peak_tflops(device: torch.device, override: float | None) -> float | None:
    """Dense BF16 peak for ``device``, or ``None`` when the hardware is unrecognized.

    Without a peak there is no MFU to report, so an unknown accelerator degrades to
    TFLOP/s only; ``--peak-tflops`` covers that case and also lets a run be scored
    against a different reference (e.g. the 2250 TFLOP/s GB200 figure Cosmos3 Table 8
    uses instead of the 2500 dense peak).
    """
    if override is not None:
        return override
    if device.type != "cuda":
        return None
    device_name = torch.cuda.get_device_name(device).lower()
    for name, peak in DENSE_BF16_PEAK_TFLOPS.items():
        if name in device_name:
            return peak
    return None


def attention_flops(pairs: int, num_q_heads: int, head_dim: int, include_backward: bool) -> float:
    """FLOPs for ``pairs`` attended token pairs: QK^T and P@V, 2 FLOPs per multiply-add.

    Backward is charged at ~2x the forward, as in ``benchmark_fmha``.
    """
    flops = 4.0 * pairs * num_q_heads * head_dim
    return flops * 3.0 if include_backward else flops


def make_qkv(
    seq_len: int,
    kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Packed q/k/v in the heads-last layout the attention path uses.

    The queries are the padded GEN stream; the keys and values are the fused
    ``[UND | GEN]`` stream, hence the two lengths.

    ``requires_grad`` is always on so the forward-only rows measure a *training*
    forward (residuals saved), which is what a decoder layer actually runs.
    """
    q = torch.randn(1, seq_len, num_q_heads, head_dim, dtype=dtype, device=device, generator=generator).requires_grad_(
        True
    )  # [1,S,Hq,D]
    k = torch.randn(
        1, kv_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device, generator=generator
    ).requires_grad_(True)  # [1,S_kv,Hkv,D]
    v = torch.randn(
        1, kv_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device, generator=generator
    ).requires_grad_(True)  # [1,S_kv,Hkv,D]
    return q, k, v


def time_call(
    call: Callable[[], object],
    *,
    label: str,
    warmup: int,
    iters: int,
    device: torch.device,
    reraise: bool,
) -> tuple[list[float], int, str | None]:
    """Time ``call`` with CUDA events.

    Returns ``(per-iter latencies in ms, peak allocated bytes, error message)``; on
    failure the latencies are empty and the message says why, so one OOM row does not
    kill the whole sweep. ``reraise`` gives up that isolation to keep the traceback,
    which is what tells an allocation inside an Inductor kernel apart from one inside
    eager ``create_mask``.

    ``warmup`` has to cover the ``torch.compile`` of both ``create_block_mask`` and
    ``flex_attention`` (one specialisation per block-aligned shape). Any returned tensor
    is read back on the host after timing so Inductor cannot dead-code-eliminate the
    work being measured.
    """
    sink: object = None
    try:
        for _ in range(warmup):
            sink = call()
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latencies_ms: list[float] = []
        torch.cuda.nvtx.range_push(f"flex_bench.{label}")
        for _ in range(iters):
            start.record()
            sink = call()
            end.record()
            torch.cuda.synchronize(device)
            latencies_ms.append(start.elapsed_time(end))  # ms
        torch.cuda.nvtx.range_pop()
        if isinstance(sink, Tensor):
            _ = sink.sum().item()
    except Exception as e:  # noqa: BLE001 - a row that OOMs or is unsupported must not stop the sweep
        if reraise:
            raise
        return [], 0, f"{type(e).__name__}: {e}"

    return latencies_ms, torch.cuda.max_memory_allocated(device), None


def _clear_grads(*tensors: Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


@dataclass
class Row:
    """One timed variant of the benchmark."""

    variant: str
    pairs: int
    include_backward: bool
    latencies_ms: list[float]
    peak_bytes: int
    error: str | None

    @property
    def status(self) -> str:
        return self.error if self.error is not None else "ok"

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else float("nan")

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else float("nan")

    def tflops_per_s(self, num_q_heads: int, head_dim: int) -> float:
        if not self.latencies_ms or self.pairs == 0:
            return float("nan")
        flops = attention_flops(self.pairs, num_q_heads, head_dim, self.include_backward)
        return flops / (self.median_ms * 1e-3) / 1e12

    def mfu(self, num_q_heads: int, head_dim: int, peak_tflops: float | None) -> float:
        """Achieved TFLOP/s as a fraction of the device's dense peak, for this operator alone."""
        if peak_tflops is None:
            return float("nan")
        return self.tflops_per_s(num_q_heads, head_dim) / peak_tflops


def run_scenario(
    scenario: MultiviewScenario,
    config: BenchConfig,
    device: torch.device,
    triton_backend: FlexBackend,
    flash_backend: FlexBackend | None,
) -> tuple[list[Row], dict]:
    """Build the mask(s) for ``scenario`` and time every requested variant.

    The backends come from the caller because ``scenario``'s padding was derived from them:
    each mask is built at the block size of the backend that runs it, which is the geometry
    those same padded lengths have to tile.

    ``flash_backend`` turns the FlashAttention-4 rows on, and ``None`` leaves them off: the
    backend iterates its own (coarser) block size, so it needs a second mask built at that
    granularity, which is timed separately because a step pays that build too.

    Returns the rows plus a stats dict describing the pack and the mask sparsity.
    """
    pack = build_pack_inputs(scenario, device)
    dtype = DTYPE_MAP[config.dtype]
    generator = torch.Generator(device=device).manual_seed(config.seed)

    def _time_mask_build(label: str, block_size: tuple[int, int]) -> tuple[BlockMask | None, Row]:
        """Time the mask build at ``block_size``; returns the mask (``None`` if it failed)."""

        def build() -> BlockMask:
            return build_multiview_block_mask(
                seq_len=scenario.seq_len,
                full_q_offsets=pack.full_q_offsets,
                token_shapes=pack.token_shapes,
                condition_masks=pack.condition_masks,
                num_vision_items_per_sample=pack.num_vision_items_per_sample,
                num_views_per_vision_item=pack.num_views_per_vision_item,
                device=device,
                block_size=block_size,
                num_und=scenario.causal_seq_len,
                causal_offsets=pack.causal_offsets,
                noisy_attention_scope=scenario.noisy_attention_scope,
            )

        latencies, peak, error = time_call(
            build,
            label=label,
            warmup=max(1, config.warmup),
            iters=config.iters,
            device=device,
            reraise=config.raise_on_error,
        )
        # A mask build does no attention math, so it carries no FLOP count.
        row = Row(label, 0, False, latencies, peak, error)
        return (None if error is not None else build()), row

    rows: list[Row] = []
    stats: dict = {
        "allowed_pairs": allowed_pair_count(scenario),
        "dense_pairs": dense_pair_count(scenario),
    }

    def _record_mask_stats(mask: BlockMask, backend: FlexBackend) -> None:
        """Fold one built mask's block counts into ``stats``, keyed by the backend that runs it."""
        blocks = count_mask_blocks(mask)
        suffix = mask_stats_suffix(backend.name)
        stats[f"visited_pairs{suffix}"] = blocks.visited_pairs
        stats[f"full_blocks{suffix}"] = blocks.full
        stats[f"partial_blocks{suffix}"] = blocks.partial
        stats[f"empty_blocks{suffix}"] = blocks.empty
        stats[f"block_sparsity{suffix}_pct"] = float(mask.sparsity())

    block_mask, mask_row = _time_mask_build("mask_build", triton_backend.block_size)
    rows.append(mask_row)
    if block_mask is None:
        return rows, stats
    _record_mask_stats(block_mask, triton_backend)

    flash_mask: BlockMask | None = None
    if flash_backend is not None:
        flash_block_size = flash_backend.block_size
        stats["flash_block_size"] = f"{flash_block_size[0]}x{flash_block_size[1]}"
        flash_mask, flash_mask_row = _time_mask_build("mask_build_flash", flash_block_size)
        rows.append(flash_mask_row)
        if flash_mask is None:
            return rows, stats
        _record_mask_stats(flash_mask, flash_backend)

    if config.mask_only:
        return rows, stats

    q, k, v = make_qkv(
        scenario.seq_len,
        scenario.kv_seq_len,
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device,
        generator,
    )

    def _flex(
        mask: BlockMask,
        backend: FlexBackend,
        include_backward: bool,
    ) -> Callable[[], Tensor]:
        """One timed attention call, without the log-sum-exp.

        The fused attention path in ``two_way_attention`` does not ask for the LSE, and asking
        for it here would also put it in the graph as a differentiable output, so AOTAutograd
        would hand the backward a (zero) LSE tangent -- which the FA4 backward has no dLSE rule
        for and refuses to lower at all. See the module docstring.
        """

        def call() -> Tensor:
            out = flex_attention(q, k, v, mask, backend)  # [1,S,Hq,D]
            assert isinstance(out, Tensor)
            if include_backward:
                # Grads accumulate across iterations on purpose: clearing them would make
                # every backward re-allocate the q/k/v grad buffers inside the timed region.
                out.sum().backward()
            return out

        return call

    def _dense(include_backward: bool) -> Callable[[], Tensor]:
        """The varlen baseline, also without the LSE, so the comparison stays like-for-like.

        Reads the same buffers as the flex rows, with the key stream taken per sample rather
        than per stream: the offsets treat it as ``[und | gen]`` runs, which is the order
        ``get_all_seq`` produces for this pass and is the same bytes for a single sample.
        Only the segment lengths matter to a timing baseline, so the row order does not.
        """

        def call() -> Tensor:
            out = attention(
                query=q,
                key=k,
                value=v,
                cumulative_seqlen_Q=pack.dense_q_offsets,
                cumulative_seqlen_KV=pack.dense_kv_offsets,
                max_seqlen_Q=pack.dense_max_seqlen_q,
                max_seqlen_KV=pack.dense_max_seqlen_kv,
            )  # [1,S,Hq,D]
            assert isinstance(out, Tensor)
            if include_backward:
                out.sum().backward()
            return out

        return call

    pairs_flex = stats["visited_pairs"]
    variants: list[tuple[str, Callable[[], Tensor], int, bool]] = [
        ("flex_fwd", _flex(block_mask, triton_backend, False), pairs_flex, False),
    ]
    if config.backward:
        variants.append(("flex_fwd_bwd", _flex(block_mask, triton_backend, True), pairs_flex, True))
    if flash_mask is not None:
        assert flash_backend is not None  # flash_mask is only built from it
        pairs_flash = stats["visited_pairs_flash"]
        variants.append(("flash_fwd", _flex(flash_mask, flash_backend, False), pairs_flash, False))
        if config.backward:
            variants.append(("flash_fwd_bwd", _flex(flash_mask, flash_backend, True), pairs_flash, True))
    if not config.skip_dense:
        variants.append(("dense_fwd", _dense(False), stats["dense_pairs"], False))
        if config.backward:
            variants.append(("dense_fwd_bwd", _dense(True), stats["dense_pairs"], True))

    for variant, call, pairs, include_backward in variants:
        latencies, peak, error = time_call(
            call,
            label=variant,
            warmup=config.warmup,
            iters=config.iters,
            device=device,
            reraise=config.raise_on_error,
        )
        rows.append(Row(variant, pairs, include_backward, latencies, peak, error))
        _clear_grads(q, k, v)

    return rows, stats


def format_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))


def print_scenario(scenario: MultiviewScenario, stats: dict, backends: Sequence[FlexBackend]) -> None:
    patch_h, patch_w = scenario.patch_hw
    height, width = scenario.resolution_hw
    print(
        f"Scenario: {scenario.num_views} cameras x {scenario.pixel_frames_per_view} pixel frames "
        f"-> {scenario.latent_frames_per_view} latent frames/camera @ {height}x{width}"
    )
    print(
        f"  per latent frame: {patch_h}x{patch_w} = {scenario.spatial_tokens} tokens "
        f"(VAE {scenario.latent_downsample_factor}x spatial / {scenario.temporal_compression_factor}x temporal, "
        f"patch {scenario.patch_spatial}) | items/sample: {scenario.num_noisy_items} noisy "
        f"({scenario.noisy_cond_frames_per_view} clean frames/camera) + {scenario.num_cond_items} control | "
        f"samples: {scenario.num_samples}"
    )
    print(
        f"  tokens: {scenario.tokens_per_item:,}/item, {scenario.tokens_per_sample:,}/sample, "
        f"{scenario.real_tokens:,} real, {scenario.seq_len:,} block-padded (to {scenario.seq_alignment})"
    )
    print(
        f"  causal: {scenario.num_causal_tokens:,} UND tokens/sample, {scenario.real_causal_tokens:,} real, "
        f"{scenario.causal_seq_len:,} block-padded (to {scenario.causal_seq_alignment}) -> "
        f"{scenario.seq_len:,} queries x {scenario.kv_seq_len:,} fused [UND | GEN] keys"
    )
    grids = ", ".join(
        f"{backend.name} {rows}x{columns} of {backend.block_size[0]}x{backend.block_size[1]}"
        for backend, (rows, columns) in ((backend, scenario.mask_blocks(backend)) for backend in backends)
    )
    print(
        f"  mask build: {grids} blocks; peak must stay far below "
        f"the {scenario.dense_mask_bytes / 1024**3:.1f} GiB a dense [1,1,S,S_kv] evaluation would need"
    )
    dense_pairs = stats["dense_pairs"]
    allowed = stats["allowed_pairs"]
    print(
        f"  mask (noisy scope {scenario.noisy_attention_scope!r}): allowed pairs {allowed:.3e} "
        f"of dense {dense_pairs:.3e} = {100.0 * allowed / dense_pairs:.1f}%"
    )
    # One line per mask built: what the kernel is left to visit, and how much of that it has
    # to evaluate the mask_mod over, which is where a narrow scope loses its MFU.
    for backend in backends:
        suffix = mask_stats_suffix(backend.name)
        visited = stats.get(f"visited_pairs{suffix}")
        if visited is None:
            continue
        q_block, kv_block = backend.block_size
        full = stats[f"full_blocks{suffix}"]
        partial = stats[f"partial_blocks{suffix}"]
        empty = stats[f"empty_blocks{suffix}"]
        print(
            f"    {backend.name} {q_block}x{kv_block}: {full:,} full + {partial:,} partial + {empty:,} empty "
            f"of {full + partial + empty:,} blocks -> {100.0 * empty / (full + partial + empty):.1f}% skipped, "
            f"{100.0 * partial / (full + partial):.1f}% of the visited partial"
        )
        print(
            f"      visits {100.0 * visited / dense_pairs:.1f}% of the dense token pairs, "
            f"of which {100.0 * allowed / visited:.1f}% are allowed"
        )


def print_rows(rows: list[Row], config: BenchConfig, peak_tflops: float | None) -> None:
    headers = ["variant", "median(ms)", "min(ms)", "TFLOP/s", "MFU", "peakGB", "status"]
    widths = [19, 11, 10, 10, 7, 8, 48]
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for row in rows:
        tflops = row.tflops_per_s(config.num_q_heads, config.head_dim)
        mfu = row.mfu(config.num_q_heads, config.head_dim, peak_tflops)
        print(
            format_row(
                [
                    row.variant,
                    f"{row.median_ms:.3f}",
                    f"{row.min_ms:.3f}",
                    "-" if math.isnan(tflops) else f"{tflops:.1f}",
                    "-" if math.isnan(mfu) else f"{100.0 * mfu:.1f}%",
                    f"{row.peak_bytes / 1024**3:.2f}",
                    row.status,
                ],
                widths,
            )
        )


def print_projection(rows: list[Row], num_layers: int) -> None:
    """Project the per-layer timings onto one full decoder step.

    Each attention row is charged the mask build of the backend it runs on, since the two
    masks differ in block size and hence in build cost.
    """
    by_variant = {row.variant: row for row in rows if row.latencies_ms}
    for flex_name, mask_name, dense_name in (
        ("flex_fwd", "mask_build", "dense_fwd"),
        ("flex_fwd_bwd", "mask_build", "dense_fwd_bwd"),
        ("flash_fwd", "mask_build_flash", "dense_fwd"),
        ("flash_fwd_bwd", "mask_build_flash", "dense_fwd_bwd"),
    ):
        flex = by_variant.get(flex_name)
        if flex is None:
            continue
        mask = by_variant.get(mask_name)
        step_ms = flex.median_ms * num_layers + (mask.median_ms if mask else 0.0)
        line = f"Per step ({num_layers} layers{' + mask' if mask else ''}): {flex_name} = {step_ms / 1e3:.3f} s"
        dense = by_variant.get(dense_name)
        if dense is not None:
            line += (
                f" vs {dense_name} = {dense.median_ms * num_layers / 1e3:.3f} s "
                f"({dense.median_ms / flex.median_ms:.2f}x speedup per layer)"
            )
        print(line)


@dataclass(frozen=True)
class ScopeRun:
    """One point of the ``noisy_attention_scopes`` sweep, kept for :func:`print_scope_comparison`."""

    scenario: MultiviewScenario
    rows: list[Row]
    stats: dict

    def timed(self, variant: str) -> Row | None:
        """The named row, or ``None`` when it was skipped or failed (and so has no timings)."""
        for row in self.rows:
            if row.variant == variant and row.latencies_ms:
                return row
        return None


# The masks a scope sweep can be scored against. The flash rows run against FA4's coarser
# mask, so how much of the narrowing they are free to skip is a different number from the
# Triton rows'.
SCOPE_COMPARISON_MASKS: tuple[str, ...] = ("triton", "flash")

# The timed attention rows the comparison covers, each with the mask above that bounds it.
SCOPE_COMPARISON_VARIANTS: tuple[tuple[str, str], ...] = (
    ("flex_fwd", "triton"),
    ("flex_fwd_bwd", "triton"),
    ("flash_fwd", "flash"),
    ("flash_fwd_bwd", "flash"),
)


def print_scope_comparison(runs: Sequence[ScopeRun]) -> None:
    """Compare a ``noisy_attention_scopes`` sweep against its first entry.

    The scope narrows the noisy->noisy quadrant and nothing else, and every scope runs the
    same pack, so the pairs it drops are the whole of the saving. Two ratios bound what a
    kernel can return for them, which is what the first table reports per scope:

    * ``pairs`` -- the allowed-pair ratio, i.e. the speedup a mask whose rules landed on block
      boundaries would give. An upper bound: a partially-masked block costs a full one.
    * ``blocks`` -- the visited-block ratio of the mask actually built, i.e. the work the
      kernel is left free to skip. Below ``pairs`` by what the scope's boundaries cost, which
      is little for ``same_view`` (one contiguous run per camera) and more for
      ``same_view_or_frame`` (a partial block at each end of every same-frame stripe).
    * ``skipped%`` -- the share of the mask's whole block grid that is empty, i.e. the
      computation the scope removes outright. Read with ``partial%`` it says how the narrowing
      landed: blocks the rules missed entirely (skipped, free) against blocks they cut through
      (partial, charged in full).
    * ``partial%`` -- the share of the visited blocks the rules cut through. It is the
      reason a narrow scope can be faster than the baseline and still post a *lower* MFU:
      narrowing converts full blocks into partial ones, which are charged the same tile of
      math by ``blocks`` and by the FLOP count but cost more to run, since the ``mask_mod``
      has to be evaluated over every position in them and the unmasked fast path is gone.
      The backward pays that per-block overhead again in each of its two passes, so its rows
      lose the most. A coarser block size raises this share for the same rules, which is why
      the flash column runs higher than the Triton one.

    The second table puts the wall clock next to the ``blocks`` bound, as ``of blocks``: a row
    near 100% converted the sparsity into time, which is the property worth checking. Well
    below it says the row is no longer attention-bound at this size (mask build, launch
    overhead, the backward's mask-independent traffic) or is paying for the partial blocks
    above, not that the mask failed to narrow -- and a ``measured`` under 1.00x says the
    narrower mask did not get sparser at all, which at these sizes should not happen.
    """
    base = runs[0]
    scope_width = max(18, *(len(run.scenario.noisy_attention_scope) for run in runs))
    masks = [name for name in SCOPE_COMPARISON_MASKS if f"visited_pairs{mask_stats_suffix(name)}" in base.stats]
    print()
    print(f"Scope comparison, against the {base.scenario.noisy_attention_scope!r} baseline:")

    headers = ["noisy scope", "allowed%", "pairs"]
    widths = [scope_width, 9, 8]
    for name in masks:
        headers += [f"{name} skipped%", f"{name} partial%", f"{name} visited%", f"{name} blocks"]
        widths += [len(name) + 9, len(name) + 9, len(name) + 9, max(8, len(name) + 7)]
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for run in runs:
        dense_pairs = run.stats["dense_pairs"]
        allowed = run.stats["allowed_pairs"]
        cells = [
            run.scenario.noisy_attention_scope,
            f"{100.0 * allowed / dense_pairs:.2f}",
            _format_ratio(_ratio(base.stats["allowed_pairs"], allowed)),
        ]
        for name in masks:
            suffix = mask_stats_suffix(name)
            visited = run.stats.get(f"visited_pairs{suffix}")
            partial = run.stats.get(f"partial_blocks{suffix}", 0)
            blocks = partial + run.stats.get(f"full_blocks{suffix}", 0)
            grid = blocks + run.stats.get(f"empty_blocks{suffix}", 0)
            cells += [
                "-" if not grid else f"{100.0 * run.stats[f'empty_blocks{suffix}'] / grid:.1f}",
                "-" if not blocks else f"{100.0 * partial / blocks:.1f}",
                "-" if visited is None else f"{100.0 * visited / dense_pairs:.2f}",
                _format_ratio(_ratio(base.stats[f"visited_pairs{suffix}"], visited)),
            ]
        print(format_row(cells, widths))

    timed = [(variant, mask) for variant, mask in SCOPE_COMPARISON_VARIANTS if base.timed(variant) is not None]
    if not timed:
        return

    headers = ["variant", "noisy scope", "median(ms)", "measured", "of blocks"]
    widths = [14, scope_width, 11, 9, 10]
    print()
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for variant, mask in timed:
        base_row = base.timed(variant)
        assert base_row is not None  # timed() answered for the baseline above
        visited_key = f"visited_pairs{mask_stats_suffix(mask)}"
        for run in runs:
            row = run.timed(variant)
            if row is None:
                continue
            measured = base_row.median_ms / row.median_ms
            blocks = _ratio(base.stats.get(visited_key), run.stats.get(visited_key))
            print(
                format_row(
                    [
                        variant,
                        run.scenario.noisy_attention_scope,
                        f"{row.median_ms:.3f}",
                        f"{measured:.2f}x",
                        "-" if math.isnan(blocks) else f"{100.0 * measured / blocks:.0f}%",
                    ],
                    widths,
                )
            )


def _ratio(baseline: int | None, value: int | None) -> float:
    """``baseline / value`` as a speedup, or NaN when either count is missing or zero."""
    if not baseline or not value:
        return float("nan")
    return baseline / value


def _format_ratio(ratio: float) -> str:
    return "-" if math.isnan(ratio) else f"{ratio:.2f}x"


@dataclass
class BenchConfig:
    """Benchmark the multiview FlexAttention GEN-tower self-attention."""

    num_views: int = 11
    """Cameras per vision item."""
    pixel_frames_per_view: list[int] = field(default_factory=lambda: [29])
    """Pixel frames per camera; one benchmark is run per value."""
    resolution_hw: tuple[int, int] = (720, 1280)
    """Pixel resolution of one camera frame, as (height, width)."""
    latent_downsample_factor: int = WAN2PT2_SPATIAL_COMPRESSION
    """VAE spatial compression; defaults to the Wan2.2 tokenizer's factor."""
    temporal_compression_factor: int = WAN2PT2_TEMPORAL_COMPRESSION
    """VAE temporal compression; defaults to the Wan2.2 tokenizer's factor."""
    patch_spatial: int = 2
    """Latent patch size of the diffusion expert."""
    num_noisy_items: int = 1
    """Generated (noisy) camera items per sample."""
    num_cond_items: int = 1
    """Fully-conditioning control camera items per sample."""
    noisy_cond_frames_per_view: int = 0
    """Leading latent frames per camera kept clean inside each noisy item (I2V-style conditioning)."""
    num_causal_tokens: int = 301
    """Causal (understanding/caption) tokens per sample, prefixed to the key stream; 0 for GEN->GEN only."""
    num_samples: int = 1
    """Samples packed into the batch."""
    noisy_attention_scopes: list[NoisyAttentionScope] = field(default_factory=lambda: ["all_views"])
    """Noisy->noisy scopes to benchmark; one run per value, all compared against the first."""
    num_q_heads: int = 32
    """Query heads (Cosmos3 16B/8B: 32)."""
    num_kv_heads: int = 8
    """KV heads (GQA; Cosmos3 16B/8B: 8)."""
    head_dim: int = 128
    """Attention head dimension."""
    num_layers: int = 36
    """Decoder layers, used for the per-step projection."""
    dtype: Literal["bf16", "fp16"] = "bf16"
    """Precision of q/k/v."""
    peak_tflops: float | None = None
    """Dense peak TFLOP/s per GPU for the MFU column; defaults to the detected device."""

    backward: bool = False
    """Also time forward + backward (training cost)."""
    flash: bool = False
    """Also time FlexAttention's FlashAttention-4 backend (Hopper/Blackwell; needs flash-attn-4)."""
    skip_dense: bool = False
    """Skip the dense varlen baseline, the slowest and most memory-hungry row."""
    mask_only: bool = False
    """Only build the mask: reports its cost and sparsity without allocating q/k/v."""
    warmup: int = 2
    """Warmup iterations; must cover the torch.compile of the mask build and the kernel."""
    iters: int = 10
    """Timed iterations per variant."""
    seed: int = 1234
    """Seed for q/k/v generation."""
    device: str = "cuda"
    """Device to benchmark on."""
    json: bool = False
    """Emit one JSON object per row instead of a table."""
    raise_on_error: bool = False
    """Let a failing variant propagate with its traceback instead of recording it in the status column."""

    def scenario(
        self,
        pixel_frames: int,
        seq_alignment: int,
        causal_seq_alignment: int,
        noisy_attention_scope: NoisyAttentionScope,
    ) -> MultiviewScenario:
        """The scenario for one point of the ``pixel_frames_per_view`` x scope sweep."""
        return MultiviewScenario(
            seq_alignment=seq_alignment,
            causal_seq_alignment=causal_seq_alignment,
            noisy_attention_scope=noisy_attention_scope,
            num_views=self.num_views,
            pixel_frames_per_view=pixel_frames,
            resolution_hw=self.resolution_hw,
            latent_downsample_factor=self.latent_downsample_factor,
            temporal_compression_factor=self.temporal_compression_factor,
            patch_spatial=self.patch_spatial,
            num_noisy_items=self.num_noisy_items,
            num_cond_items=self.num_cond_items,
            noisy_cond_frames_per_view=self.noisy_cond_frames_per_view,
            num_causal_tokens=self.num_causal_tokens,
            num_samples=self.num_samples,
        )


def emit_json_rows(
    scenario: MultiviewScenario,
    rows: list[Row],
    stats: dict,
    config: BenchConfig,
    peak_tflops: float | None,
) -> None:
    """One JSON object per row, carrying the geometry and mask stats it was measured under.

    Every scope of the sweep emits its own rows, tagged with the scope, so a consumer can take
    the ratios :func:`print_scope_comparison` prints without re-deriving the geometry.
    """
    for row in rows:
        print(
            json.dumps(
                {
                    "variant": row.variant,
                    "noisy_attention_scope": scenario.noisy_attention_scope,
                    "num_views": scenario.num_views,
                    "pixel_frames_per_view": scenario.pixel_frames_per_view,
                    "latent_frames_per_view": scenario.latent_frames_per_view,
                    "seq_len": scenario.seq_len,
                    "real_tokens": scenario.real_tokens,
                    "num_causal_tokens": scenario.num_causal_tokens,
                    "causal_seq_len": scenario.causal_seq_len,
                    "kv_seq_len": scenario.kv_seq_len,
                    "dense_mask_bytes": scenario.dense_mask_bytes,
                    "median_ms": row.median_ms,
                    "min_ms": row.min_ms,
                    "tflops_per_s": row.tflops_per_s(config.num_q_heads, config.head_dim),
                    "mfu": row.mfu(config.num_q_heads, config.head_dim, peak_tflops),
                    "peak_tflops": peak_tflops,
                    "peak_bytes": row.peak_bytes,
                    "error": row.error,
                    **stats,
                }
            )
        )


def main(config: BenchConfig) -> None:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this benchmark requires a GPU.")
    device = torch.device(config.device)
    peak_tflops = resolve_peak_tflops(device, config.peak_tflops)

    triton_backend = _get_triton_flex_backend()
    flash_backend: FlexBackend | None = None
    if config.flash:
        try:
            flash_backend = _get_flash_flex_backend(device)
        except ValueError as e:
            # Unsupported hardware is a flag error, not a benchmark result: every flash row
            # would fail identically, and the padding below depends on the answer.
            raise SystemExit(str(e)) from e
    # One padded pack for every backend timed, so they all attend over the same tokens.
    backends = [triton_backend] if flash_backend is None else [triton_backend, flash_backend]
    seq_alignment, causal_seq_alignment = stream_alignments(backends)

    if not config.json:
        peak_label = (
            "unknown, pass --peak-tflops for MFU"
            if peak_tflops is None
            else f"{peak_tflops:g} TFLOP/s dense {config.dtype}"
        )
        print(f"Device: {torch.cuda.get_device_name(device)} (peak: {peak_label})")
        print(
            f"Model: Hq={config.num_q_heads} Hkv={config.num_kv_heads} D={config.head_dim} dtype={config.dtype} "
            f"layers={config.num_layers} | Timing: warmup={config.warmup} iters={config.iters}"
        )
        if flash_backend is not None:
            triton_q_block, triton_kv_block = triton_backend.block_size
            flash_q_block, flash_kv_block = flash_backend.block_size
            print(
                f"Backends: Triton ({triton_q_block}x{triton_kv_block} blocks) and FlashAttention-4 "
                f"({flash_q_block}x{flash_kv_block} blocks), over one pack padded to multiples of "
                f"{seq_alignment} (GEN) and {causal_seq_alignment} (UND)"
            )

    for pixel_frames in config.pixel_frames_per_view:
        # One ScopeRun per noisy_attention_scope, all over the same pack, so the comparison
        # below attributes every difference between them to the mask.
        runs: list[ScopeRun] = []
        for noisy_attention_scope in config.noisy_attention_scopes:
            scenario = config.scenario(pixel_frames, seq_alignment, causal_seq_alignment, noisy_attention_scope)
            rows, stats = run_scenario(scenario, config, device, triton_backend, flash_backend)
            runs.append(ScopeRun(scenario, rows, stats))

            if config.json:
                emit_json_rows(scenario, rows, stats, config, peak_tflops)
                continue

            print()
            print_scenario(scenario, stats, backends)
            print_rows(rows, config, peak_tflops)
            print_projection(rows, config.num_layers)

        if len(runs) > 1 and not config.json:
            print_scope_comparison(runs)


if __name__ == "__main__":
    main(tyro.cli(BenchConfig, description=__doc__))
