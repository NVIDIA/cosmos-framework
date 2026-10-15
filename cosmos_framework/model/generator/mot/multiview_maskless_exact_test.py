# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exact and overlapping maskless modes against independent dense edge counts and gradients."""

import math
from typing import Any

import pytest
import torch

from cosmos_framework.configs.base.defaults.multiview_attention import (
    MultiviewAttentionConfig,
    MultiviewAttentionMaskConfig,
    TemporalWindow,
    temporal_window_bounds,
)
from cosmos_framework.model.generator.mot import multiview_maskless_attention as maskless
from cosmos_framework.model.generator.mot.flex_attention import _multiview_pair_predicate, _StreamFields
from cosmos_framework.model.generator.mot.multiview_attention import resolve_multiview_backend
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_full_only_seq,
    sequence_pack_from_packed_sequence,
)

# sample, modality axis, view, instant (or capture time), control -- independent of the plan's groups.
Token = tuple[int, int, int, float, bool]


def _reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,Nq,H,D], [1,Nk,Hkv,D] -> [1,Nq,H,D], [1,Nq,H]
    groups = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(groups, dim=2)  # [1,Nk,H,D]
    v = v.repeat_interleave(groups, dim=2)  # [1,Nk,H,D]
    q_segment = torch.searchsorted(
        kwargs["cumulative_seqlen_Q"][1:], torch.arange(q.shape[1], device=q.device), right=True
    )  # [Nq]
    k_segment = torch.searchsorted(
        kwargs["cumulative_seqlen_KV"][1:], torch.arange(k.shape[1], device=k.device), right=True
    )  # [Nk]
    allowed = q_segment[:, None] == k_segment[None, :]  # [Nq,Nk]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])  # [1,H,Nq,Nk]
    scores = scores.masked_fill(~allowed, -torch.inf)  # [1,H,Nq,Nk]
    out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v)  # [1,Nq,H,D]
    return out, scores.logsumexp(-1).transpose(1, 2)  # [1,Nq,H,D], [1,Nq,H]


def _reference_merge(
    *, outputs: list[torch.Tensor], lse_tensors: list[torch.Tensor], **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor]:  # list[[1,N,H,D]], list[[1,N,H]]
    normalizers = torch.stack(lse_tensors)  # [branches,1,N,H]
    out = (torch.stack(outputs) * normalizers.softmax(0).unsqueeze(-1)).sum(0)  # [1,N,H,D]
    return out, normalizers.logsumexp(0)  # [1,N,H,D], [1,N,H]


def _batch(
    case: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    control_attends_sensor: bool,
    per_view_captions: bool,
    window: TemporalWindow | None = None,
    deduplicate_cross_view: bool = True,
    include_first_frame: bool = False,
) -> tuple[list[torch.Tensor], list[SequencePack], maskless.MultiviewMasklessPlan, list[Token], list[tuple[int, int]]]:
    # Item = (views, frames, spatial tokens, modality axis, control, seconds/frame).
    rgb = (7, 5, 2, 0, False, 2 / 15)
    if case == "rgb":
        samples = [[rgb]]
    elif case == "rgb11":
        samples = [[(11, 5, 2, 0, False, 2 / 15)]]
    elif case == "single":
        samples = [[(1, 3, 2, 0, False, 2 / 15)]]
    elif case == "ragged_joint":
        samples = [
            [
                (3, 3, 2, 0, True, 2 / 15),
                (3, 3, 2, 0, False, 2 / 15),
                (1, 4, 1, 1, True, 0.1),
                (1, 4, 1, 1, False, 0.1),
            ],
            [(1, 2, 3, 0, False, 2 / 15)],
            [(2, 4, 1, 0, False, 2 / 15)],
        ]
    else:
        raise ValueError(case)
    layout: list[Token] = []
    text_layout: list[tuple[int, int]] = []
    split_lens: list[int] = []
    sample_lens: list[int] = []
    und_indices: list[int] = []
    gen_indices: list[int] = []
    captions: list[list[tuple[int, int]]] = []
    cursor = 0
    for sample, items in enumerate(samples):
        caption = [(view, 2) for view in range(items[0][0])] if per_view_captions else [(-1, 5)]
        captions.append(caption)
        text_layout.extend((sample, view) for view, length in caption for _ in range(length))
        und_len = sum(length for _, length in caption)
        gen_len = sum(views * frames * spatial for views, frames, spatial, _, _, _ in items)
        split_lens.extend((und_len, gen_len))
        sample_lens.append(und_len + gen_len)
        und_indices.extend(range(cursor, cursor + und_len))
        gen_indices.extend(range(cursor + und_len, cursor + und_len + gen_len))
        cursor += und_len + gen_len
        for views, frames, spatial, axis, control, period in items:
            frame_times = (torch.arange(frames, dtype=torch.float32, device="cpu") * period).tolist()  # [F] -> list
            for view in range(views):
                for frame in range(frames):
                    instant = (
                        frame_times[frame]
                        if window is not None
                        else math.floor((frame + 0.5) * period / items[0][5] + 1e-6)
                    )
                    layout.extend([(sample, axis, view, instant, control)] * spatial)
    tensors = [
        torch.randn(cursor, heads, 64, device=device, dtype=dtype, requires_grad=True)  # [N,H,D]
        for heads in (2, 1, 1)
    ]
    packs = [
        sequence_pack_from_packed_sequence(
            tensor,
            attn_modes=["causal", "full"] * len(samples),
            split_lens=split_lens,
            sample_lens=sample_lens,
            packed_und_token_indexes=torch.tensor(und_indices, device=device),  # [N_text]
            packed_gen_token_indexes=torch.tensor(gen_indices, device=device),  # [N_gen]
            text_caption_lens=[[length for _, length in caption] for caption in captions]
            if per_view_captions
            else None,
        )
        for tensor in tensors
    ]
    items = [item for sample_items in samples for item in sample_items]
    plan = maskless.build_multiview_maskless_plan(
        [item[0] for item in items],
        [(views * frames, 1, spatial) for views, frames, spatial, _, _, _ in items],
        device=device,
        items_per_sample=[len(sample_items) for sample_items in samples],
        is_control=[item[4] for item in items],
        control_attends_sensor=control_attends_sensor,
        view_axis=[item[3] for item in items],
        seconds_per_frame=[item[5] for item in items],
        captions=captions if per_view_captions else None,
        padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
        deduplicate_cross_view=deduplicate_cross_view,
        decomposed_temporal_window_seconds=window,
        decomposed_temporal_window_includes_first_frame=include_first_frame,
    )
    return tensors, packs, plan, layout, text_layout


def _pair_multiplicity(
    query: Token,
    key: Token,
    control_attends_sensor: bool,
    window: TemporalWindow | None = None,
    deduplicate_cross_view: bool = True,
    include_first_frame: bool = False,
) -> int:
    qs, qa, qv, qt, qc = query
    ks, ka, kv, kt, kc = key
    same_view = qa == ka and qv == kv
    if window is None:
        in_time = qt == kt
    else:
        # Every item's first frame starts at capture time 0.
        in_time = window[0] - 1e-4 <= kt - qt <= window[1] + 1e-4 or (include_first_frame and kt == 0.0)
    if qs != ks:
        return 0
    count = int(same_view and (not qc or kc or control_attends_sensor)) + int(not qc and not kc and in_time)
    return int(count > 0) if deduplicate_cross_view else count


def _dense_reference(
    packs: list[SequencePack],
    layout: list[Token],
    text_layout: list[tuple[int, int]],
    control_attends_sensor: bool,
    window: TemporalWindow | None = None,
    deduplicate_cross_view: bool = True,
    include_first_frame: bool = False,
) -> torch.Tensor:  # [N_gen,H*D]
    q, k, v = [get_full_only_seq(pack)[0][: len(layout)].double() for pack in packs]  # each [N_gen,H,D]
    text_k, text_v = [get_causal_seq(pack)[0][: len(text_layout)].double() for pack in packs[1:]]  # each [N_text,Hkv,D]
    keys = torch.cat((k, text_k)).repeat_interleave(q.shape[1] // k.shape[1], dim=1)  # [N_keys,H,D]
    values = torch.cat((v, text_v)).repeat_interleave(q.shape[1] // v.shape[1], dim=1)  # [N_keys,H,D]
    # Both modes bypass the temporal branch for single-view-only samples.
    group_counts = {
        sample: len({(axis, view) for s, axis, view, _, _ in layout if s == sample}) for sample, *_ in layout
    }
    multiplicity = torch.tensor(
        [
            [
                _pair_multiplicity(
                    query,
                    key,
                    control_attends_sensor,
                    window,
                    deduplicate_cross_view or group_counts[query[0]] == 1,
                    include_first_frame,
                )
                for key in layout
            ]
            + [query[0] == sample and (view == -1 or query[1] != 0 or query[2] == view) for sample, view in text_layout]
            for query in layout
        ],
        device=q.device,
    )  # [N_gen,N_keys]
    scores = torch.einsum("qhd,khd->hqk", q, keys) / math.sqrt(q.shape[-1])  # [H,N_gen,N_keys]
    scores = scores + multiplicity.to(scores.dtype).log()  # [H,N_gen,N_keys]
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).flatten(-2)  # [N_gen,H*D]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("case", ["rgb", "rgb11", "single", "ragged_joint"])
@pytest.mark.parametrize("control_attends_sensor", [True, False])
@pytest.mark.parametrize("per_view_captions", [True, False])
@pytest.mark.parametrize("deduplicate_cross_view", [True, False])
@pytest.mark.parametrize(
    "window",
    [None, (0.0, 0.0), (-0.1, 0.0), (-0.4, 0.0), (-0.4, 0.4), (-0.2, 0.2), (-0.4, 0.1), (0.1, 0.3), (10.0, 11.0)],
)
@pytest.mark.parametrize("include_first_frame", [False, True])
def test_maskless_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    control_attends_sensor: bool,
    per_view_captions: bool,
    window: TemporalWindow | None,
    deduplicate_cross_view: bool,
    include_first_frame: bool,
) -> None:
    if include_first_frame and window is None:
        pytest.skip("The first-frame rule widens a window; test_first_frame_rule_requires_a_window covers None.")
    monkeypatch.setattr(maskless, "attention", _reference_attention)
    monkeypatch.setattr(maskless, "merge_attentions", _reference_merge)
    torch.manual_seed(51)
    tensors, packs, plan, layout, text_layout = _batch(
        case,
        torch.device("cpu"),
        torch.float64,
        control_attends_sensor=control_attends_sensor,
        per_view_captions=per_view_captions,
        window=window,
        deduplicate_cross_view=deduplicate_cross_view,
        include_first_frame=include_first_frame,
    )
    actual = maskless.multiview_maskless_gen_attention(*packs, plan=plan)  # [N_padded,H*D]
    expected = _dense_reference(
        packs, layout, text_layout, control_attends_sensor, window, deduplicate_cross_view, include_first_frame
    )  # [N_gen,H*D]
    torch.testing.assert_close(actual[: len(layout)], expected, atol=1e-10, rtol=1e-10)
    assert torch.count_nonzero(actual[len(layout) :]) == 0
    upstream = torch.randn_like(expected)  # [N_gen,H*D]
    actual_grads = torch.autograd.grad(
        (actual[: len(layout)] * upstream).sum(), tensors, retain_graph=True
    )  # list[[N,H,D]]
    expected_grads = torch.autograd.grad((expected * upstream).sum(), tensors)  # list[[N,H,D]]
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-9)

    # Check multiplicity separately from numerics, including odd view counts,
    # distinct sensor clocks and single-view samples within a mixed batch.
    if not deduplicate_cross_view and window is None:
        return  # Legacy grouping was already checked numerically above.
    group_counts = {
        sample: len({(axis, view) for s, axis, view, _, _ in layout if s == sample}) for sample, *_ in layout
    }
    edges: set[tuple[int, int]] = set()
    for partition in plan.cross_view_partitions:
        assert len(set(partition.query_indices.tolist())) == partition.query_indices.numel()
        if window is None:
            assert len(set(partition.key_indices.tolist())) == partition.key_indices.numel()
        for qs, qe, ks, ke in zip(
            partition.query_offsets[:-1],
            partition.query_offsets[1:],
            partition.key_offsets[:-1],
            partition.key_offsets[1:],
            strict=True,
        ):
            for qi in partition.query_indices[qs:qe].tolist():
                for ki in partition.key_indices[ks:ke].tolist():
                    assert (qi, ki) not in edges
                    edges.add((qi, ki))
    expected_edges = {
        (qi, ki)
        for qi, query in enumerate(layout)
        for ki, key in enumerate(layout)
        if not query[4]
        and not key[4]
        and group_counts[query[0]] > 1
        and (not deduplicate_cross_view or query[1:3] != key[1:3])
        # Subtract the same-view contribution to isolate this temporal pass.
        and _pair_multiplicity(query, key, control_attends_sensor, window, False, include_first_frame)
        > int(query[0] == key[0] and query[1:3] == key[1:3])
    }
    assert edges == expected_edges
    max_depth = math.ceil(math.log2(max(group_counts.values())))
    assert len(plan.cross_view_partitions) <= (max_depth if deduplicate_cross_view else 1)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("views", [1, 2, 3, 4, 5, 7, 8, 11])
def test_hierarchy_covers_each_directed_view_pair_once(views: int) -> None:
    # Different cell sizes and multiple discontiguous spans per view exercise
    # the rectangular Q/K lengths independently of the production layout builder.
    cells: dict[int, list[tuple[int, int]]] = {}
    labels: dict[int, int] = {}
    cursor = 0
    for view in range(views):
        cells[view] = [(cursor, cursor + view + 1), (cursor + view + 3, cursor + view + 4)]
        for start, end in cells[view]:
            labels.update({index: view for index in range(start, end)})
        cursor += view + 5
    partitions = maskless._cross_view_partitions({(0, 0): cells}, torch.device("cpu"), deduplicate_cross_view=True)
    assert len(partitions) == math.ceil(math.log2(views))
    observed: set[tuple[int, int]] = set()
    for partition in partitions:
        assert partition.query_indices.unique().numel() == partition.query_indices.numel()
        assert partition.key_indices.unique().numel() == partition.key_indices.numel()
        for qs, qe, ks, ke in zip(
            partition.query_offsets[:-1],
            partition.query_offsets[1:],
            partition.key_offsets[:-1],
            partition.key_offsets[1:],
            strict=True,
        ):
            for query in partition.query_indices[qs:qe].tolist():
                for key in partition.key_indices[ks:ke].tolist():
                    assert (query, key) not in observed
                    observed.add((query, key))
    assert observed == {(q, k) for q, qv in labels.items() for k, kv in labels.items() if qv != kv}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [None, (-0.4, 0.0), (-0.4, 0.2)])
@pytest.mark.parametrize("deduplicate_cross_view", [True, False])
def test_maskless_compiles_as_one_graph(
    monkeypatch: pytest.MonkeyPatch,
    window: TemporalWindow | None,
    deduplicate_cross_view: bool,
) -> None:
    monkeypatch.setattr(maskless, "attention", _reference_attention)
    monkeypatch.setattr(maskless, "merge_attentions", _reference_merge)
    _, packs, plan, _, _ = _batch(
        "ragged_joint",
        torch.device("cpu"),
        torch.float64,
        control_attends_sensor=False,
        per_view_captions=True,
        window=window,
        deduplicate_cross_view=deduplicate_cross_view,
    )
    compiled = torch.compile(maskless.multiview_maskless_gen_attention, backend="eager", fullgraph=True)
    actual = compiled(*packs, plan=plan)  # [N_padded,H*D]
    expected = maskless.multiview_maskless_gen_attention(*packs, plan=plan)  # [N_padded,H*D]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.L0
@pytest.mark.CPU
def test_exact_mode_is_opt_in() -> None:
    config = MultiviewAttentionConfig(backend="maskless")
    assert config.deduplicate_cross_view is False
    assert config.mask.decomposed_temporal_window_seconds is None
    assert config.mask.decomposed_temporal_window_includes_first_frame is False


@pytest.mark.L0
@pytest.mark.CPU
def test_first_frame_rule_requires_a_window() -> None:
    with pytest.raises(ValueError, match="needs decomposed_temporal_window_seconds"):
        maskless.build_multiview_maskless_plan(
            [2],
            [(6, 1, 1)],
            device=torch.device("cpu"),
            seconds_per_frame=[0.1],
            deduplicate_cross_view=True,
            decomposed_temporal_window_includes_first_frame=True,
        )
    for scope, window in (("decomposed", None), ("same_view", (-0.4, 0.0))):
        config = MultiviewAttentionConfig(
            backend="maskless",
            deduplicate_cross_view=True,
            mask=MultiviewAttentionMaskConfig(
                attention_scope=scope,
                decomposed_temporal_window_seconds=window,
                decomposed_temporal_window_includes_first_frame=True,
            ),
        )
        reason = maskless.maskless_unavailable_reason(config)
        assert reason is not None and "decomposed_temporal_window_includes_first_frame" in reason


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("preference", ["auto", "flex_triton", "flex_flash"])
def test_first_frame_rule_is_refused_by_the_mask_backends(preference: str) -> None:
    config = MultiviewAttentionConfig(
        backend=preference,
        mask=MultiviewAttentionMaskConfig(
            attention_scope="decomposed",
            decomposed_temporal_window_seconds=(-0.4, 0.0),
            decomposed_temporal_window_includes_first_frame=True,
        ),
    )
    with pytest.raises(ValueError, match="requires backend='maskless'"):
        resolve_multiview_backend(torch.device("cpu"), preference, config=config)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "window", [(0.0, 0.0), (-0.1, 0.0), (-0.4, 0.0), (-1.7, 0.0), (-0.4, 0.4), (-0.2, 0.2), (-0.4, 0.1), (0.1, 0.3)]
)
@pytest.mark.parametrize("deduplicate_cross_view", [True, False])
def test_window_pair_counts_match_actual_flex_predicate(window: TemporalWindow, deduplicate_cross_view: bool) -> None:
    # Camera and LiDAR clocks, plus boundary cases on both sides of the 1e-4
    # tolerance. Distinct axes keep each sensor's physical identity separate.
    rates = [2 / 15, 0.1, 0.39995, 0.40005, 0.40015]
    frames = [5, 7, 2, 2, 2]
    times = torch.cat(
        [torch.arange(count, dtype=torch.float32) * rate for count, rate in zip(frames, rates, strict=True)]
    )  # [N]
    view_ids = torch.repeat_interleave(torch.arange(len(frames)), torch.tensor(frames))  # [N]
    plan = maskless.build_multiview_maskless_plan(
        [1] * len(frames),
        [(count, 1, 1) for count in frames],
        device=torch.device("cpu"),
        items_per_sample=[len(frames)],
        view_axis=list(range(len(frames))),
        seconds_per_frame=rates,
        deduplicate_cross_view=deduplicate_cross_view,
        decomposed_temporal_window_seconds=window,
    )
    fields = _StreamFields(
        sample_id=torch.zeros_like(view_ids),  # [N]
        frame_id=torch.cat([torch.arange(count) for count in frames]),  # [N]
        view_id=view_ids,  # [N]
        is_noisy=torch.ones_like(view_ids, dtype=torch.bool),  # [N]
        is_control=torch.zeros_like(view_ids, dtype=torch.bool),  # [N]
        is_und=torch.zeros_like(view_ids, dtype=torch.bool),  # [N]
        timestamp=times,  # [N]
        caption_scope=torch.zeros_like(view_ids),  # [N]
    )
    predicate = _multiview_pair_predicate(fields, fields, "decomposed", window)
    indexes = torch.arange(times.numel())  # [N]
    expected = predicate(torch.tensor(0), torch.tensor(0), indexes[:, None], indexes[None, :])  # [N,N]
    counts = (view_ids[:, None] == view_ids[None, :]).int()  # [N,N], same-view pass
    for partition in plan.cross_view_partitions:
        assert partition.query_indices.unique().numel() == partition.query_indices.numel()
        for qs, qe, ks, ke in zip(
            partition.query_offsets[:-1],
            partition.query_offsets[1:],
            partition.key_offsets[:-1],
            partition.key_offsets[1:],
            strict=True,
        ):
            query = partition.query_indices[qs:qe]  # [Nq]
            key = partition.key_indices[ks:ke]  # [Nk]
            counts[query[:, None], key[None, :]] += 1  # [N,N]
    expected_counts = expected.int()  # [N,N]
    if not deduplicate_cross_view:
        bounds = temporal_window_bounds(window)
        assert bounds is not None
        start, end = torch.tensor(bounds, dtype=torch.float32).unbind()  # each []
        offsets = times[None, :] - times[:, None]  # [N,N]
        overlap = (
            (view_ids[:, None] == view_ids[None, :]) & (offsets >= start - 1e-4) & (offsets <= end + 1e-4)
        )  # [N,N]
        expected_counts += overlap.int()  # [N,N]
    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    assert plan.decomposed_temporal_window_seconds == window


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("window", [0, 0.4, -0.1, float("nan"), float("inf")])
def test_window_rejects_scalar_duration(window: Any) -> None:
    with pytest.raises(ValueError, match="requires two finite bounds"):
        maskless.build_multiview_maskless_plan(
            [2],
            [(6, 1, 1)],
            device=torch.device("cpu"),
            seconds_per_frame=[0.1],
            deduplicate_cross_view=True,
            decomposed_temporal_window_seconds=window,
        )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("deduplicate_cross_view", [True, False])
def test_window_requires_explicit_clock(deduplicate_cross_view: bool) -> None:
    with pytest.raises(ValueError, match="requires explicit seconds_per_frame"):
        maskless.build_multiview_maskless_plan(
            [2],
            [(6, 1, 1)],
            device=torch.device("cpu"),
            deduplicate_cross_view=deduplicate_cross_view,
            decomposed_temporal_window_seconds=(-0.4, 0.0),
        )


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires real GPU attention and merge kernels")
@pytest.mark.parametrize("case", ["rgb", "rgb11", "single", "ragged_joint"])
@pytest.mark.parametrize("compile_graph", [False, True])
@pytest.mark.parametrize("window", [None, (-0.4, 0.0), (-0.4, 0.2), (-0.2, 0.2)])
@pytest.mark.parametrize("deduplicate_cross_view", [True, False])
def test_maskless_real_kernel_gradients(
    case: str,
    compile_graph: bool,
    window: TemporalWindow | None,
    deduplicate_cross_view: bool,
) -> None:
    torch.manual_seed(51)
    tensors, packs, plan, layout, text_layout = _batch(
        case,
        torch.device("cuda"),
        torch.bfloat16,
        control_attends_sensor=False,
        per_view_captions=True,
        window=window,
        deduplicate_cross_view=deduplicate_cross_view,
    )
    forward = maskless.multiview_maskless_gen_attention
    if compile_graph:
        # Each parameter set represents a different recipe/layout, not recompilation
        # within one training run. Keep Dynamo's per-code cache independent between tests.
        torch.compiler.reset()
        forward = torch.compile(forward, fullgraph=True)
    actual = forward(*packs, plan=plan)[: len(layout)]  # [N_gen,H*D]
    expected = _dense_reference(packs, layout, text_layout, False, window, deduplicate_cross_view)  # [N_gen,H*D]
    torch.testing.assert_close(actual.double(), expected, atol=2e-2, rtol=2e-2)
    upstream = torch.randn_like(actual)  # [N_gen,H*D]
    # Evaluate the reference gradients before the real merge patches saved storage.
    expected_grads = torch.autograd.grad((expected * upstream).sum(), tensors, retain_graph=True)  # list[[N,H,D]]
    actual_grads = torch.autograd.grad((actual * upstream).sum(), tensors)  # list[[N,H,D]]
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-2, rtol=3e-2)
