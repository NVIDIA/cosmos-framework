# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Time the maskless multiview folds against the paths they are an alternative to.

Three ways to attend one multiview sample, all timed over the same packed batch and all
including the reasoner's own causal self-attention, since every one of them runs it:

* ``dense_full`` -- ``two_way_attention`` with no mask, i.e. every GEN token attends to its
  whole sample. What a multiview run does today with FlexAttention off, and the cost both
  decompositions exist to beat.
* ``flex_decomposed`` -- ``multiview_attention`` under the multiview block mask at
  ``attention_scope="decomposed"``: one masked kernel over the fused ``[UND | GEN]`` stream.
* ``multiview_maskless`` -- ``multiview_attention`` under a ``MultiviewMasklessPlan``: three
  unmasked kernels (same
  view, cross view, gen->und) merged by log-sum-exp, no ``BlockMask`` anywhere.

``mask_build`` times the block mask the flex row needs. It is charged separately because
production builds it once per forward, outside the decoder layers, and every layer then
shares it -- so its per-layer share is that cost over ``--num-layers``, which the projection
line reports.

The two decompositions do not attend the same pairs, and the ``pairs`` column says so: the
maskless one takes the query's own ``(view, frame)`` cell twice, once from each of its two
sensor passes, where the mask takes it once. That is ``spatial_tokens`` extra keys per query
-- one cell out of the ``F + V - 1`` the scope admits -- and it is the deliberate
approximation documented on ``multiview_maskless_attention``, not a benchmarking artifact.
TFLOP/s is therefore each row's own useful work over its own time; the ``vs dense`` column is
what a caller actually chooses between.

Inference by default, and a training step under ``--train``: the folds' backward is correct
(``MergeAttentionsBridge``, see ``multiview_maskless_attention``), so every row can be timed
either way, and all of them are timed the same way as each other.

Example:

    PYTHONPATH=. python -m cosmos_framework.model.generator.mot.multiview_maskless_attention_bench \
        --num-views 7 --pixel-frames-per-view 29 61 121
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import tyro
from torch.nn.attention.flex_attention import BlockMask

from cosmos_framework.utils.misc import set_torch_compile_options
from cosmos_framework.configs.base.defaults.multiview_attention import (
    AttentionScope,
    FlexGeometryPreference,
)
from cosmos_framework.model.generator.mot.attention import build_packed_sequence, two_way_attention
from cosmos_framework.model.generator.mot.flex_attention import (
    CaptionMaskItem,
    FlexBackend,
    SensorMaskItem,
    build_multiview_block_mask,
    resolve_flex_backend,
)
from cosmos_framework.model.generator.mot.flex_attention_bench import (
    WAN2PT2_SPATIAL_COMPRESSION,
    WAN2PT2_TEMPORAL_COMPRESSION,
    MultiviewScenario,
    Row,
    resolve_peak_tflops,
    time_call,
)
from cosmos_framework.model.generator.mot.multiview_attention import multiview_attention
from cosmos_framework.model.generator.mot.multiview_maskless_attention import build_multiview_maskless_plan
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_full_only_seq,
    get_gen_seq,
)

_DTYPES: dict[str, torch.dtype] = {"bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class BenchConfig:
    """Benchmark the maskless multiview folds against the flex and dense paths."""

    num_views: int = 7
    """Cameras in the rig, i.e. views of the one generated item."""
    pixel_frames_per_view: list[int] = field(default_factory=lambda: [29, 61, 121])
    """Pixel frames per camera; one benchmark is run per value."""
    resolution_hw: tuple[int, int] = (720, 1280)
    """Pixel resolution of one camera frame, as (height, width)."""
    latent_downsample_factor: int = WAN2PT2_SPATIAL_COMPRESSION
    """VAE spatial compression; defaults to the Wan2.2 tokenizer's factor."""
    temporal_compression_factor: int = WAN2PT2_TEMPORAL_COMPRESSION
    """VAE temporal compression; defaults to the Wan2.2 tokenizer's factor."""
    patch_spatial: int = 2
    """Latent patch size of the diffusion expert."""
    num_causal_tokens: int = 301
    """Caption tokens prefixed to the key stream, which every GEN token of the sample attends."""
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
    backend: FlexGeometryPreference = "auto"
    """Which kernels the flex rows run on; "auto" takes FlashAttention-4 where it is usable.

    A mask geometry, not an attention: the maskless row builds no mask and takes its padding
    alignments from whichever geometry this resolves to.
    """
    peak_tflops: float | None = None
    """Dense peak TFLOP/s per GPU for the MFU column; defaults to the detected device."""
    skip_dense: bool = False
    """Skip the unmasked full-sample row, which is the slowest at every size."""
    compile: bool = False
    """Run every variant under torch.compile, as the decoder layer does.

    This is the flex row's production setting rather than a tuning knob: FlexAttention's
    FlashAttention-4 backend lowers the multiview mask only inside a compiled region, so an
    eager run either falls back to the Triton kernels or fails to lower at all. The dense rows
    are compiled alongside it so the comparison stays like for like.
    """
    duck_shape: bool = False
    """Whether two tensor dims of equal size share one symbol, as ``use_duck_shape`` decides.

    False is what the AV multiview recipes run: ``config_factory`` sets
    ``compile_config.use_duck_shape=False``, which ``ImaginaireTrainer`` passes to
    ``set_torch_compile_options``. PyTorch's own default is True, so a benchmark that sets
    nothing does *not* represent those runs.

    It is not a tuning knob for the flex rows, it decides whether they lower at all. With duck
    shaping on, the mask's per-token fields share a symbol with the GEN stream they were
    measured from, Inductor binds that symbol into the ``mask_mod`` subgraph as a scalar, and
    the FlashAttention-4 lowering refuses it. Only visible under ``--dynamic``, since static
    shapes give it nothing to share.
    """
    dynamic: bool = False
    """Compile with a symbolic sequence length instead of specialising on this one.

    What a training run wants: the token count varies between steps, and specialising on each
    one recompiles every layer per shape it meets. The head dims stay pinned either way --
    ``mark_static`` below -- because FlexAttention's lowering cannot fold a symbolic head count
    and the FlashAttention-4 templates are instantiated per head width.

    Expect the flex rows to feel this more than the dense ones: FA4 refuses a mask whose
    ``mask_mod`` captures a symbolic scalar at all ("NYI: score_mod or mask_mod captures a
    dynamic scalar"), which is the same lowering that ``--compile`` exists to reach.
    """
    warmup: int = 2
    """Warmup iterations; must cover the torch.compile of the flex kernel."""
    iters: int = 10
    """Timed iterations per variant."""
    seed: int = 1234
    """Seed for q/k/v generation."""
    device: str = "cuda"
    """Device to benchmark on."""
    raise_on_error: bool = False
    """Let a failing variant propagate with its traceback instead of recording it in the status column."""
    samples: list[str] | None = None
    """A ragged batch, one ``views x pixel_frames x caption_tokens`` per sample.

    Overrides the ``pixel_frames_per_view`` sweep, which times one sample at a time. Samples may
    differ in every one of the three, which is the case the batch axis cannot express: the
    decomposition then addresses its groups with varlen ranges instead, and those decline cuDNN,
    so this is a different backend as well as a different launch shape from the single-sample
    rows. Example: ``--samples 11x101x2048 4x53x1536 7x33x712 1x105x4098``.
    """
    camera_control: bool = False
    """Give every sample a control item ahead of its target, as a WSM transfer pack carries.

    The control item matches its target's geometry, doubling the sample's GEN tokens. It joins
    its target's view groups and stays out of the cross-instant one, which is what the mask's
    view rules come to under ``control_attends_sensor`` -- the setting the joint camera + LiDAR
    transfer recipe uses. It also costs the same-view pass its identity gather, since a view's
    control and target tokens then sit in two runs of the packed stream.
    """
    per_view_captions: bool = False
    """Give each camera view its own caption instead of one caption per sample.

    The sample's caption budget -- the third field of a ``--samples`` spec, or
    ``num_causal_tokens`` -- is split evenly across its views rather than grown, so the UND
    stream is the same length either way and the rows stay comparable to the shared-caption
    ones. What changes is the routing: a GEN token reads its own view's caption instead of the
    whole run, which shrinks the gen->und pass for every variant, and the decomposition takes
    its caption gather instead of the batched path.
    """
    train: bool = False
    """Time a training step (forward plus backward) instead of an inference forward.

    Every row keeps its own grad, so the comparison is what a training step of that attention
    actually costs: the packs' streams become leaves, and each timed call runs the forward and
    then backpropagates the GEN output's sum. FLOP/s and MFU charge the backward at 2x the
    forward, as ``attention_flops`` does throughout.
    """

    def scenario(self, pixel_frames: int, backend: FlexBackend) -> MultiviewScenario:
        """The scenario for one point of the ``pixel_frames_per_view`` sweep.

        One generated camera item, with a control item ahead of it under ``--camera-control``
        and none otherwise. The scope is pinned to ``"decomposed"`` because that is the mask
        this row is an alternative to; the folds also express ``"same_view"``, which is a
        cheaper row this sweep does not cover.
        """
        return MultiviewScenario(
            seq_alignment=backend.full_seq_alignment,
            causal_seq_alignment=backend.causal_seq_alignment,
            attention_scope="decomposed",
            num_views=self.num_views,
            pixel_frames_per_view=pixel_frames,
            resolution_hw=self.resolution_hw,
            latent_downsample_factor=self.latent_downsample_factor,
            temporal_compression_factor=self.temporal_compression_factor,
            patch_spatial=self.patch_spatial,
            num_noisy_items=1,
            num_cond_items=1 if self.camera_control else 0,
            noisy_cond_frames_per_view=0,
            num_causal_tokens=self.num_causal_tokens,
            num_samples=1,
        )


def parse_samples(specs: Sequence[str], config: BenchConfig, backend: FlexBackend) -> list[MultiviewScenario]:
    """One scenario per ``views x pixel_frames x caption_tokens`` spec.

    Each carries a single sample, so its properties describe that sample alone and the ragged
    builders below combine them. Everything the samples must agree on -- resolution, the VAE
    factors, the patch size, the block alignments -- comes from the shared config rather than
    the spec, since those are properties of the tokenizer and the kernel and not of the clip.
    """
    scenarios: list[MultiviewScenario] = []
    for spec in specs:
        fields = spec.lower().split("x")
        if len(fields) != 3:
            raise SystemExit(f"--samples takes views x pixel_frames x caption_tokens, got {spec!r}.")
        views, pixel_frames, causal_tokens = (int(field) for field in fields)
        scenarios.append(
            MultiviewScenario(
                seq_alignment=backend.full_seq_alignment,
                causal_seq_alignment=backend.causal_seq_alignment,
                attention_scope="decomposed",
                num_views=views,
                pixel_frames_per_view=pixel_frames,
                resolution_hw=config.resolution_hw,
                latent_downsample_factor=config.latent_downsample_factor,
                temporal_compression_factor=config.temporal_compression_factor,
                patch_spatial=config.patch_spatial,
                num_noisy_items=1,
                num_cond_items=1 if config.camera_control else 0,
                noisy_cond_frames_per_view=0,
                num_causal_tokens=causal_tokens,
                num_samples=1,
            )
        )
    return scenarios


def caption_lens(scenario: MultiviewScenario, per_view_captions: bool) -> list[int] | None:
    """The sample's captions as token counts, or ``None`` for the usual one-per-sample layout.

    The captions must tile the causal split exactly -- ``_build_caption_offsets`` checks it --
    so the budget is divided across the views with the remainder spread over the first few
    rather than dropped.
    """
    if not per_view_captions:
        return None
    views = scenario.num_views
    base, extra = divmod(scenario.real_causal_tokens, views)
    return [base + (1 if index < extra else 0) for index in range(views)]


def caption_items(scenario: MultiviewScenario, per_view_captions: bool) -> list[CaptionMaskItem] | None:
    """The same layout as the mask reads it: one item per caption, tagged with its view."""
    lens = caption_lens(scenario, per_view_captions)
    if lens is None:
        return None
    return [CaptionMaskItem(view_id=view, num_tokens=num_tokens) for view, num_tokens in enumerate(lens)]


def sensor_items(scenario: MultiviewScenario, device: torch.device) -> list[SensorMaskItem]:
    """A sample's items for the mask: its control items, then its target.

    Every item but the last conditions the one that follows it, the convention the packer and
    ``_multiview_sensor_mask_items`` share. No conditioning frames: this scope treats
    conditioning and noisy tokens alike, so the mask only has to accept the item's latent axis.
    """
    total = scenario.sensor_mask_items
    return [
        SensorMaskItem(
            token_shape=scenario.token_shape,
            condition_mask=torch.zeros(scenario.latent_t, dtype=torch.bool, device=device),
            num_views=scenario.num_views,
            view_offset=0,
            is_control=index < total - 1,
            # One camera rig on one clock, which is what these scenarios time.
            seconds_per_frame=1.0,
            caption_access="camera",
        )
        for index in range(total)
    ]


def multiview_maskless_plan(
    scenarios: Sequence[MultiviewScenario],
    device: torch.device,
    per_view_captions: bool = False,
    padded_gen_tokens: int | None = None,
):
    """The plan for these scenarios, one entry per item with the control items marked."""
    num_views: list[int] = []
    token_shapes: list[tuple[int, int, int]] = []
    control: list[bool] = []
    counts: list[int] = []
    for scenario in scenarios:
        total = scenario.sensor_mask_items
        counts.append(total)
        for index in range(total):
            num_views.append(scenario.num_views)
            token_shapes.append(scenario.token_shape)
            control.append(index < total - 1)
    # Captions as ``(view_id, num_tokens)`` per sample, the form the gate hands the builder.
    captions = (
        [list(enumerate(cast(list, caption_lens(scenario, True)))) for scenario in scenarios]
        if per_view_captions
        else None
    )
    return build_multiview_maskless_plan(
        num_views,
        token_shapes,
        device=device,
        items_per_sample=counts,
        is_control=control,
        # The rule ``counted_pairs`` counts and the masks below are built with: a control token
        # and its target share a view group outright, so the group is one varlen segment.
        control_attends_sensor=True,
        view_axis=[0] * len(num_views),
        captions=captions,
        padded_gen_tokens=padded_gen_tokens,
    )


def counted_pairs(scenarios: Sequence[MultiviewScenario], variant: str, per_view_captions: bool = False) -> int:
    """Token pairs a variant attends, counted over ``(item, view, frame)`` cells.

    The closed forms in ``variant_pairs`` describe a sample of one item; a control item adds a
    second, so this counts the cells instead. ``flex_decomposed`` admits a pair when it shares a
    view or -- both being sensor tokens -- a frame, each pair once; ``multiview_maskless`` gives
    the same pair a multiplicity, since the two passes each take the cells they share. Both add
    every GEN token's own captions.
    """
    total = 0
    for scenario in scenarios:
        items = scenario.sensor_mask_items
        spatial, frames, views = scenario.spatial_tokens, scenario.latent_frames_per_view, scenario.num_views
        cells = [
            (index < items - 1, view, frame)
            for index in range(items)
            for view in range(views)
            for frame in range(frames)
        ]
        gen_tokens = len(cells) * spatial
        if variant == "dense_full":
            gen_pairs = gen_tokens * gen_tokens
        else:
            # A single-view sample owns one same-view group, so its cross-instant groups sit
            # inside it and the pass is skipped: no double count, and no instant term at all.
            neutralised = variant == "multiview_maskless" and views == 1
            multiplicity = variant == "multiview_maskless" and not neutralised
            gen_pairs = 0
            for q_control, q_view, q_frame in cells:
                for k_control, k_view, k_frame in cells:
                    same_view = q_view == k_view
                    both_sensor = (not q_control) and (not k_control)
                    if variant == "flex_all_views":
                        # The scope widens the sensor->sensor rule and only that one: it is read
                        # inside ``in_scope``, which every control rule bypasses, so a pair
                        # touching a control token is a view rule under this scope too. Charging
                        # the whole square -- as this did, alongside ``dense_full`` -- is right
                        # only for a sample with no control item, where every cell is a sensor
                        # and ``both_sensor`` already admits everything.
                        gen_pairs += int(both_sensor or same_view) * spatial * spatial
                        continue
                    same_instant = (not neutralised) and both_sensor and q_frame == k_frame
                    if multiplicity:
                        gen_pairs += (int(same_view) + int(same_instant)) * spatial * spatial
                    elif same_view or same_instant:
                        gen_pairs += spatial * spatial
        # Per-view captions make the und key set a property of the query's view, so the
        # gen->und term is per cell rather than one product over the whole causal run.
        # dense_full is maskless, so a per-view caption layout narrows nothing for it: every GEN
        # token still reads the whole causal run. What the layout does change for it is the UND
        # self-attention, which is one document per caption instead of one per sample -- not
        # counted here, for the same reason it is not counted under the shared layout.
        lens = None if variant == "dense_full" else caption_lens(scenario, per_view_captions)
        if lens is None:
            total += gen_tokens * scenario.real_causal_tokens + gen_pairs
        else:
            total += spatial * sum(lens[view] for _, view, _ in cells) + gen_pairs
    return total


def build_ragged_packs(
    scenarios: Sequence[MultiviewScenario],
    config: BenchConfig,
    device: torch.device,
) -> tuple[SequencePack, SequencePack, SequencePack]:
    """Pack a ragged batch the way the packer lays one down: each sample's UND run, then its GEN.

    The splits alternate causal/full per sample, which is what makes the GEN stream the samples'
    GEN runs concatenated -- the order both the plan's offsets and its gather are written in.
    """
    und_lens = [scenario.real_causal_tokens for scenario in scenarios]
    gen_lens = [scenario.real_tokens for scenario in scenarios]
    generator = torch.Generator(device=device).manual_seed(config.seed)
    dtype = _DTYPES[config.dtype]

    split_lens: list[int] = []
    und_indexes: list[int] = []
    gen_indexes: list[int] = []
    start = 0
    for und_len, gen_len in zip(und_lens, gen_lens):
        split_lens.extend((und_len, gen_len))
        und_indexes.extend(range(start, start + und_len))
        gen_indexes.extend(range(start + und_len, start + und_len + gen_len))
        start += und_len + gen_len

    def _pack(num_heads: int) -> SequencePack:
        tokens = torch.randn(start, num_heads, config.head_dim, dtype=dtype, device=device, generator=generator)
        pack = build_packed_sequence(
            "two_way",
            packed_sequence=tokens,
            attn_modes=["causal", "full"] * len(scenarios),
            split_lens=split_lens,
            sample_lens=[und + gen for und, gen in zip(und_lens, gen_lens)],
            packed_und_token_indexes=cast(torch.LongTensor, torch.tensor(und_indexes, dtype=torch.long, device=device)),
            packed_gen_token_indexes=cast(torch.LongTensor, torch.tensor(gen_indexes, dtype=torch.long, device=device)),
            num_heads=num_heads,
            head_dim=config.head_dim,
            num_layers=1,
            full_seq_alignment=scenarios[0].seq_alignment,
            causal_seq_alignment=scenarios[0].causal_seq_alignment,
            text_caption_lens=(
                [cast(list, caption_lens(scenario, True)) for scenario in scenarios]
                if config.per_view_captions
                else None
            ),
        )[0]
        if config.train:
            for key in ("causal_seq", "full_only_seq"):
                pack[key].requires_grad_(True)
        return pack

    return _pack(config.num_q_heads), _pack(config.num_kv_heads), _pack(config.num_kv_heads)


def build_ragged_mask(
    scenarios: Sequence[MultiviewScenario],
    packs: tuple[SequencePack, ...],
    backend: FlexBackend,
    attention_scope: AttentionScope,
    per_view_captions: bool = False,
) -> BlockMask:
    """The block mask for a ragged batch at one scope: one sensor item per sample."""
    full_only_seq, full_q_offsets = get_full_only_seq(packs[0])
    causal_seq, causal_offsets = get_causal_seq(packs[0])
    return build_multiview_block_mask(
        gen_seq_len=full_only_seq.shape[0],
        full_q_offsets=full_q_offsets,
        sensor_mask_items=[sensor_items(scenario, full_only_seq.device) for scenario in scenarios],
        caption_mask_items=(
            [cast(list, caption_items(scenario, True)) for scenario in scenarios] if per_view_captions else None
        ),
        device=full_only_seq.device,
        block_size=backend.block_size,
        und_seq_len=causal_seq.shape[0],
        causal_offsets=causal_offsets,
        attention_scope=attention_scope,
        decomposed_temporal_window_seconds=None,
        # The same rule the maskless plan is built with, so the rows compare one attention and
        # not two. It was ``False`` here while the plan could only express ``True``, which under
        # ``--camera-control`` left the flex rows denying the control->sensor quadrant that
        # ``counted_pairs`` charges them for and the maskless row computes.
        control_attends_sensor=True,
    )


def ragged_variant_pairs(scenarios: Sequence[MultiviewScenario], variant: str) -> int:
    """Token pairs a variant attends over a ragged batch: each sample's own, summed.

    A sample's GEN tokens never reach another sample's, in any variant, so the batch's cost is
    the sum of the per-sample costs ``variant_pairs`` counts.
    """
    return sum(variant_pairs(scenario, variant) for scenario in scenarios)


def build_packs(
    scenario: MultiviewScenario,
    config: BenchConfig,
    device: torch.device,
) -> tuple[SequencePack, SequencePack, SequencePack]:
    """Pack one sample's q/k/v the way the network packs a multiview batch.

    One causal split then one full split, which is what the packer emits per sample and what
    both attention paths read. The streams are padded to the flex backend's two block
    multiples so the one pack serves every row: the dense paths trim or ignore that padding,
    and only the flex row requires it.
    """
    und_len, gen_len = scenario.real_causal_tokens, scenario.real_tokens
    generator = torch.Generator(device=device).manual_seed(config.seed)
    dtype = _DTYPES[config.dtype]

    def _pack(num_heads: int) -> SequencePack:
        tokens = torch.randn(
            und_len + gen_len, num_heads, config.head_dim, dtype=dtype, device=device, generator=generator
        )  # [N_und+N_gen,heads,head_dim]
        pack = build_packed_sequence(
            "two_way",
            packed_sequence=tokens,
            attn_modes=["causal", "full"],
            split_lens=[und_len, gen_len],
            sample_lens=[und_len + gen_len],
            packed_und_token_indexes=cast(torch.LongTensor, torch.arange(und_len, dtype=torch.long, device=device)),
            packed_gen_token_indexes=cast(
                torch.LongTensor, torch.arange(und_len, und_len + gen_len, dtype=torch.long, device=device)
            ),
            num_heads=num_heads,
            head_dim=config.head_dim,
            num_layers=1,
            full_seq_alignment=scenario.seq_alignment,
            causal_seq_alignment=scenario.causal_seq_alignment,
            text_caption_lens=([cast(list, caption_lens(scenario, True))] if config.per_view_captions else None),
        )[0]
        if config.train:
            # The streams are what a decoder layer's projections hand the attention path, so
            # they are the leaves a training step's gradient actually lands on.
            for key in ("causal_seq", "full_only_seq"):
                pack[key].requires_grad_(True)
        return pack

    return _pack(config.num_q_heads), _pack(config.num_kv_heads), _pack(config.num_kv_heads)


def build_mask(
    scenario: MultiviewScenario,
    packs: tuple[SequencePack, ...],
    backend: FlexBackend,
    attention_scope: AttentionScope,
    per_view_captions: bool = False,
) -> BlockMask:
    """The block mask for this pack at one scope, as ``cosmos3_vfm_network`` builds it."""
    full_only_seq, full_q_offsets = get_full_only_seq(packs[0])
    causal_seq, causal_offsets = get_causal_seq(packs[0])
    return build_multiview_block_mask(
        gen_seq_len=full_only_seq.shape[0],
        full_q_offsets=full_q_offsets,
        sensor_mask_items=[sensor_items(scenario, full_only_seq.device)],
        caption_mask_items=[cast(list, caption_items(scenario, True))] if per_view_captions else None,
        device=full_only_seq.device,
        block_size=backend.block_size,
        und_seq_len=causal_seq.shape[0],
        causal_offsets=causal_offsets,
        attention_scope=attention_scope,
        decomposed_temporal_window_seconds=None,
        # The same rule the maskless plan is built with, so the rows compare one attention and
        # not two. It was ``False`` here while the plan could only express ``True``, which under
        # ``--camera-control`` left the flex rows denying the control->sensor quadrant that
        # ``counted_pairs`` charges them for and the maskless row computes.
        control_attends_sensor=True,
    )


def variant_pairs(scenario: MultiviewScenario, variant: str) -> int:
    """GEN-query token pairs a variant attends, counted from the geometry.

    Every variant gives each of the ``N`` GEN queries the sample's ``U`` caption tokens, and
    differs only in how many GEN keys it adds:

    * ``dense_full``: all ``N`` of them.
    * ``flex_decomposed``: the ``F + V - 1`` cells the scope admits, ``S`` tokens each --
      the query's own view (``F`` cells) and its own frame (``V`` cells), the shared cell
      counted once.
    * ``multiview_maskless``: ``F + V`` cells, because its two passes each take that shared
      cell and the merge keeps both copies.

    The reasoner's own causal self-attention is left out: it is the same ``U`` tokens'
    triangle in every row, and at 301 caption tokens against hundreds of thousands of GEN
    tokens it rounds away.
    """
    num_gen, num_und = scenario.real_tokens, scenario.real_causal_tokens
    cells = {
        "dense_full": None,
        # attention_scope="all_views" lets a GEN token reach every GEN token of its sample,
        # which is the set the unmasked dense path attends: same pairs, through the masked
        # kernel. That is what makes this row a measure of what the mask machinery costs when
        # it buys no sparsity at all.
        "flex_all_views": None,
        "flex_decomposed": scenario.latent_frames_per_view + scenario.num_views - 1,
        "multiview_maskless": scenario.latent_frames_per_view + scenario.num_views,
    }[variant]
    gen_keys = num_gen if cells is None else cells * scenario.spatial_tokens
    return num_gen * (num_und + gen_keys)


def run_scenario(
    scenario: MultiviewScenario,
    config: BenchConfig,
    device: torch.device,
    backend: FlexBackend,
) -> tuple[list[Row], list[Row]]:
    """Time every variant over one pack, plus each flex row's mask build."""
    packs = build_packs(scenario, config, device)
    masks, mask_rows = _build_masks(
        config, lambda scope: build_mask(scenario, packs, backend, scope, config.per_view_captions), device
    )
    # One sample per scenario, so this is the plan's dense path; it is built here rather than
    # per call for the same reason the network builds it once per forward.
    plan = multiview_maskless_plan(
        [scenario], device, config.per_view_captions, int(get_full_only_seq(packs[0])[0].shape[0])
    )

    calls: dict[str, Callable[[], object]] = {}
    if not config.skip_dense:
        calls["dense_full"] = _timed_or_trained(config, lambda: two_way_attention(*packs))
    for variant, block_mask in masks.items():
        calls[variant] = _timed_or_trained(
            config,
            lambda mask=block_mask: multiview_attention(*packs, flex_block_mask=mask, flex_backend=backend),
        )
    calls["multiview_maskless"] = _timed_or_trained(config, lambda: multiview_attention(*packs, maskless_plan=plan))

    if config.compile:
        # Only the token count varies between steps in production, and the head dims have to
        # stay concrete: FlexAttention's lowering cannot fold a symbolic head count and the
        # FlashAttention-4 templates are instantiated per head width. MoTAttention.forward
        # specialises them there; here the streams reach the compiled callable as arguments, so
        # they are marked before the call instead.
        for pack in packs:
            for stream in (pack["causal_seq"], pack["full_only_seq"]):
                torch._dynamo.mark_static(stream, 1)
                torch._dynamo.mark_static(stream, 2)
        calls = {variant: torch.compile(call, dynamic=config.dynamic) for variant, call in calls.items()}

    rows: list[Row] = []
    for variant, call in calls.items():
        latencies, peak_bytes, error = time_call(
            call,
            label=variant,
            warmup=config.warmup,
            iters=config.iters,
            device=device,
            reraise=config.raise_on_error,
        )
        rows.append(
            Row(
                variant=variant,
                pairs=counted_pairs([scenario], variant, config.per_view_captions),
                include_backward=config.train,
                latencies_ms=latencies,
                peak_bytes=peak_bytes,
                error=error,
            )
        )
    return rows, mask_rows


def _build_masks(
    config: BenchConfig,
    build: Callable[[AttentionScope], BlockMask],
    device: torch.device,
) -> tuple[dict[str, BlockMask], list[Row]]:
    """Build and time one mask per flex row, returning the masks and their build rows.

    Charged separately from the attention because production builds each mask once per forward,
    outside the decoder layers, and every layer then shares it -- so its per-layer share is that
    cost over ``--num-layers``, which the projection line reports.
    """
    scopes: dict[str, AttentionScope] = {"flex_decomposed": "decomposed", "flex_all_views": "all_views"}
    masks: dict[str, BlockMask] = {}
    rows: list[Row] = []
    for variant, scope in scopes.items():
        with torch.no_grad():
            latencies, peak, error = time_call(
                lambda scope=scope: build(scope),
                label=f"mask_build:{scope}",
                warmup=config.warmup,
                iters=config.iters,
                device=device,
                reraise=config.raise_on_error,
            )
            if error is None:
                masks[variant] = build(scope)
        rows.append(
            Row(
                variant=f"mask_build({scope})",
                pairs=0,
                include_backward=False,
                latencies_ms=latencies,
                peak_bytes=peak,
                error=error,
            )
        )
    return masks, rows


@torch.no_grad()
def _timed_or_trained(config: BenchConfig, forward):
    """Shared by both runners: the forward alone, or the forward and its backward."""
    if not config.train:

        def infer() -> object:
            with torch.no_grad():
                return forward()

        return infer

    def train() -> object:
        out = forward()
        get_gen_seq(out).sum().backward()
        return out

    return train


def run_ragged(
    scenarios: Sequence[MultiviewScenario],
    config: BenchConfig,
    device: torch.device,
    backend: FlexBackend,
) -> tuple[list[Row], list[Row]]:
    """Time every variant over one ragged batch, plus each flex row's mask build."""
    packs = build_ragged_packs(scenarios, config, device)
    masks, mask_rows = _build_masks(
        config, lambda scope: build_ragged_mask(scenarios, packs, backend, scope, config.per_view_captions), device
    )

    plan = multiview_maskless_plan(
        scenarios, device, config.per_view_captions, int(get_full_only_seq(packs[0])[0].shape[0])
    )

    calls: dict[str, Callable[[], object]] = {}
    if not config.skip_dense:
        calls["dense_full"] = _timed_or_trained(config, lambda: two_way_attention(*packs))
    for variant, block_mask in masks.items():
        calls[variant] = _timed_or_trained(
            config,
            lambda mask=block_mask: multiview_attention(*packs, flex_block_mask=mask, flex_backend=backend),
        )
    calls["multiview_maskless"] = _timed_or_trained(config, lambda: multiview_attention(*packs, maskless_plan=plan))

    if config.compile:
        for pack in packs:
            for stream in (pack["causal_seq"], pack["full_only_seq"]):
                torch._dynamo.mark_static(stream, 1)
                torch._dynamo.mark_static(stream, 2)
        calls = {variant: torch.compile(call, dynamic=config.dynamic) for variant, call in calls.items()}

    rows: list[Row] = []
    for variant, call in calls.items():
        latencies, peak_bytes, error = time_call(
            call,
            label=variant,
            warmup=config.warmup,
            iters=config.iters,
            device=device,
            reraise=config.raise_on_error,
        )
        rows.append(
            Row(
                variant=variant,
                pairs=counted_pairs(scenarios, variant, config.per_view_captions),
                include_backward=config.train,
                latencies_ms=latencies,
                peak_bytes=peak_bytes,
                error=error,
            )
        )
    return rows, mask_rows


def print_ragged(scenarios: Sequence[MultiviewScenario], backend: FlexBackend) -> None:
    """One line per sample, then the batch totals the kernels actually see."""
    print(f"\nRagged batch of {len(scenarios)} samples | flex backend: {backend.name} {backend.block_size}")
    for index, scenario in enumerate(scenarios):
        patch_h, patch_w = scenario.patch_hw
        print(
            f"  [{index}] {scenario.num_views} views x {scenario.pixel_frames_per_view} pixel frames "
            f"({scenario.latent_frames_per_view} latent) -> {patch_h}x{patch_w}={scenario.spatial_tokens} per cell "
            f"| GEN {scenario.real_tokens} UND {scenario.real_causal_tokens}"
        )
    print(
        f"  batch: GEN {sum(s.real_tokens for s in scenarios)} tokens, "
        f"UND {sum(s.real_causal_tokens for s in scenarios)}"
    )


def print_scenario(scenario: MultiviewScenario, backend: FlexBackend) -> None:
    patch_h, patch_w = scenario.patch_hw
    print(
        f"\n{scenario.num_views} views x {scenario.pixel_frames_per_view} pixel frames "
        f"({scenario.latent_frames_per_view} latent) at {scenario.resolution_hw[0]}x{scenario.resolution_hw[1]}"
        f" -> {patch_h}x{patch_w}={scenario.spatial_tokens} tokens per cell"
    )
    print(
        f"  GEN {scenario.real_tokens} tokens (padded {scenario.seq_len}), UND {scenario.real_causal_tokens} "
        f"(padded {scenario.causal_seq_len}) | flex backend: {backend.name} {backend.block_size}"
    )


def print_rows(rows: list[Row], mask_rows: list[Row], config: BenchConfig, peak_tflops: float | None) -> None:
    header = ("variant", "pairs", "median ms", "TFLOP/s", "MFU", "peak MB", "vs dense", "status")
    baseline = next((row for row in rows if row.variant == "dense_full" and row.latencies_ms), None)
    cells: list[tuple[str, ...]] = []
    for row in [*rows, *mask_rows]:
        tflops = row.tflops_per_s(config.num_q_heads, config.head_dim)
        mfu = row.mfu(config.num_q_heads, config.head_dim, peak_tflops)
        speedup = (
            f"{baseline.median_ms / row.median_ms:.2f}x"
            if baseline is not None and row.latencies_ms and row not in mask_rows
            else "-"
        )
        cells.append(
            (
                row.variant,
                f"{row.pairs:,}" if row.pairs else "-",
                f"{row.median_ms:.3f}" if row.latencies_ms else "-",
                f"{tflops:.1f}" if row.pairs and row.latencies_ms else "-",
                f"{mfu:.1%}" if row.pairs and row.latencies_ms and peak_tflops else "-",
                f"{row.peak_bytes / 1e6:.0f}",
                speedup,
                row.status if row.error is None else row.status[:48],
            )
        )
    widths = [max(len(header[i]), max(len(row[i]) for row in cells)) for i in range(len(header))]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for row_cells in cells:
        print("  " + "  ".join(c.rjust(w) for c, w in zip(row_cells, widths)))


def print_projection(rows: list[Row], mask_rows: list[Row], num_layers: int) -> None:
    """Per-step cost: every layer's attention, plus the one mask build each flex path shares."""
    scopes = {"flex_decomposed": "mask_build(decomposed)", "flex_all_views": "mask_build(all_views)"}
    builds = {row.variant: row for row in mask_rows}
    print(f"  per step ({num_layers} layers):", end="")
    for row in rows:
        if not row.latencies_ms:
            continue
        total = row.median_ms * num_layers
        build = builds.get(scopes.get(row.variant, ""))
        if build is not None and build.latencies_ms:
            total += build.median_ms
        print(f"  {row.variant} {total:.1f} ms", end="")
    print()


def main(config: BenchConfig) -> None:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this benchmark requires a GPU.")
    # Before anything compiles: the shape env is global, and the flex rows' lowering turns on it.
    set_torch_compile_options(recompile_limit=32, use_duck_shape=config.duck_shape)
    device = torch.device(config.device)
    backend = resolve_flex_backend(device, config.backend)
    peak_tflops = resolve_peak_tflops(device, config.peak_tflops)

    peak_label = "unknown, pass --peak-tflops for MFU" if peak_tflops is None else f"{peak_tflops:g} TFLOP/s dense"
    print(f"Device: {torch.cuda.get_device_name(device)} (peak: {peak_label})")
    print(
        f"Model: Hq={config.num_q_heads} Hkv={config.num_kv_heads} D={config.head_dim} dtype={config.dtype} "
        f"| Timing: warmup={config.warmup} iters={config.iters}, "
        f"{'training (forward + backward)' if config.train else 'inference (no grad)'}"
        f" | compile: {'dynamic' if config.dynamic else 'static'}, duck_shape={config.duck_shape}"
    )

    if config.samples:
        scenarios = parse_samples(config.samples, config, backend)
        rows, mask_rows = run_ragged(scenarios, config, device, backend)
        print_ragged(scenarios, backend)
        print_rows(rows, mask_rows, config, peak_tflops)
        print_projection(rows, mask_rows, config.num_layers)
        return

    for pixel_frames in config.pixel_frames_per_view:
        scenario = config.scenario(pixel_frames, backend)
        rows, mask_rows = run_scenario(scenario, config, device, backend)
        print_scenario(scenario, backend)
        print_rows(rows, mask_rows, config, peak_tflops)
        print_projection(rows, mask_rows, config.num_layers)


if __name__ == "__main__":
    main(tyro.cli(BenchConfig, description=__doc__))
