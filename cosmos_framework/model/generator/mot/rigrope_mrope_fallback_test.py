# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""RigRoPE's per-sample mRoPE fallback and its KV-group head split, against pure plans."""

import dataclasses

import pytest
import torch

from cosmos_framework.model.generator.mot import multiview_maskless_attention as dense
from cosmos_framework.model.generator.mot.camera_relative_pose_test import _reference_attention, _reference_merge
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import _rigrope_kv_heads
from cosmos_framework.model.generator.mot.rigrope import RopeRigPE, apply_mrope_rotary
from cosmos_framework.data.generator.sequence_packing.runtime import get_full_only_seq, sequence_pack_from_packed_sequence

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_HEADS = 2
_HEAD_DIM = 128
# Two samples of one two-view item each: 2 frames x 1 x 2 tokens per view.
_TOKENS_PER_SAMPLE = 8


def _two_multiview_samples() -> tuple[list[torch.Tensor], list[dict], dense.MultiviewMasklessPlan]:
    lengths = [2 + _TOKENS_PER_SAMPLE] * 2
    text_indices = [0, 1, 10, 11]
    gen_indices = [*range(2, 10), *range(12, 20)]
    tensors = [torch.randn(sum(lengths), _HEADS, _HEAD_DIM, requires_grad=True) for _ in range(3)]  # each [N,H,D]
    packs = [
        sequence_pack_from_packed_sequence(
            tensor,
            attn_modes=["causal", "full"] * 2,
            split_lens=[2, _TOKENS_PER_SAMPLE] * 2,
            sample_lens=lengths,
            packed_und_token_indexes=torch.tensor(text_indices),  # [N_text]
            packed_gen_token_indexes=torch.tensor(gen_indices),  # [N_gen]
        )
        for tensor in tensors
    ]
    plan = dense.build_multiview_maskless_plan(
        [2, 2],
        [(4, 1, 2), (4, 1, 2)],
        device=torch.device("cpu"),
        items_per_sample=[1, 1],
        is_control=[False, False],
        padded_gen_tokens=get_full_only_seq(packs[0])[0].shape[0],
    )
    assert not plan.cross_view_empty
    return tensors, packs, plan


def _heads(output: torch.Tensor, head: int) -> torch.Tensor:  # [N_gen,H*D] -> [N_gen,D]
    return output[:, head * _HEAD_DIM : (head + 1) * _HEAD_DIM]


@pytest.mark.parametrize("geometry_sample", [0, 1])
def test_sample_without_geometry_keeps_pretrained_mrope(monkeypatch: pytest.MonkeyPatch, geometry_sample: int) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(5)
    tensors, packs, plan = _two_multiview_samples()
    angles = RopeRigPE(_HEAD_DIM)(torch.randn(plan.padded_gen_tokens, 8)).reshape(-1, _HEAD_DIM)  # [N_gen,D]
    base_angles = torch.randn(plan.padded_gen_tokens, _HEAD_DIM // 2).repeat(1, 2)  # [N_gen,D]
    rigrope = dataclasses.replace(
        plan,
        mrope_cos=base_angles.cos(),  # [N_gen,D]
        mrope_sin=base_angles.sin(),  # [N_gen,D]
        rigrope_cos=angles.cos(),  # [N_gen,D]
        rigrope_sin=angles.sin(),  # [N_gen,D]
    )
    valid = torch.zeros(plan.padded_gen_tokens, dtype=torch.bool)  # [N_gen]
    posed = slice(geometry_sample * _TOKENS_PER_SAMPLE, (geometry_sample + 1) * _TOKENS_PER_SAMPLE)
    fallback = slice((1 - geometry_sample) * _TOKENS_PER_SAMPLE, (2 - geometry_sample) * _TOKENS_PER_SAMPLE)
    valid[posed] = True
    mixed = dataclasses.replace(rigrope, rigrope_valid=valid)
    baseline_packs = [dict(pack) for pack in packs]
    for index in (0, 1):
        baseline_packs[index]["full_only_seq"] = apply_mrope_rotary(
            get_full_only_seq(packs[index])[0], base_angles.cos(), base_angles.sin()
        )  # [N_gen,H,D]

    baseline = dense.multiview_maskless_gen_attention(*baseline_packs, plan=plan)  # [N_gen,H*D]
    full_rigrope = dense.multiview_maskless_gen_attention(*packs, plan=rigrope)  # [N_gen,H*D]
    actual = dense.multiview_maskless_gen_attention(*packs, plan=mixed)  # [N_gen,H*D]

    torch.testing.assert_close(actual[fallback], baseline[fallback], atol=0, rtol=0)
    torch.testing.assert_close(actual[posed], full_rigrope[posed], atol=0, rtol=0)
    assert not torch.allclose(full_rigrope[fallback], baseline[fallback])
    weights = torch.randn_like(actual[fallback])  # [8,H*D]
    got = torch.autograd.grad((actual[fallback] * weights).sum(), tensors, retain_graph=True)  # tuple[[N,H,D]]
    want = torch.autograd.grad((baseline[fallback] * weights).sum(), tensors)  # tuple[[N,H,D]]
    for got_grad, want_grad in zip(got, want, strict=True):
        torch.testing.assert_close(got_grad, want_grad, atol=1e-6, rtol=1e-5)


def test_head_split_keeps_mrope_on_the_other_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(6)
    _, packs, plan = _two_multiview_samples()
    angles = RopeRigPE(_HEAD_DIM)(torch.randn(plan.padded_gen_tokens, 8)).reshape(-1, _HEAD_DIM)  # [N_gen,D]
    base_angles = torch.randn(plan.padded_gen_tokens, _HEAD_DIM // 2).repeat(1, 2)  # [N_gen,D]
    rigrope = dataclasses.replace(
        plan,
        mrope_cos=base_angles.cos(),  # [N_gen,D]
        mrope_sin=base_angles.sin(),  # [N_gen,D]
        rigrope_cos=angles.cos(),  # [N_gen,D]
        rigrope_sin=angles.sin(),  # [N_gen,D]
    )
    q_heads, k_heads = dense.rigrope_local_head_masks(_HEADS, _HEADS, 1)  # [H], [H_kv]
    split = dataclasses.replace(rigrope, rigrope_q_heads=q_heads, rigrope_k_heads=k_heads)
    baseline_packs = [dict(pack) for pack in packs]
    for index in (0, 1):
        baseline_packs[index]["full_only_seq"] = apply_mrope_rotary(
            get_full_only_seq(packs[index])[0], base_angles.cos(), base_angles.sin()
        )  # [N_gen,H,D]

    baseline = dense.multiview_maskless_gen_attention(*baseline_packs, plan=plan)  # [N_gen,H*D]
    full_rigrope = dense.multiview_maskless_gen_attention(*packs, plan=rigrope)  # [N_gen,H*D]
    actual = dense.multiview_maskless_gen_attention(*packs, plan=split)  # [N_gen,H*D]

    torch.testing.assert_close(_heads(actual, 0), _heads(full_rigrope, 0), atol=0, rtol=0)
    torch.testing.assert_close(_heads(actual, 1), _heads(baseline, 1), atol=0, rtol=0)


def test_no_rotation_fallback_keeps_each_head_on_one_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """RigRoPE heads skip rotation for a sample without geometry; mRoPE heads keep mRoPE."""
    monkeypatch.setattr(dense, "attention", _reference_attention)
    monkeypatch.setattr(dense, "merge_attentions", _reference_merge)
    torch.manual_seed(7)
    _, packs, plan = _two_multiview_samples()
    posed, unposed = slice(0, _TOKENS_PER_SAMPLE), slice(_TOKENS_PER_SAMPLE, 2 * _TOKENS_PER_SAMPLE)
    coords = torch.randn(plan.padded_gen_tokens, 8)  # [N_gen,8]
    # The network gives a sample without geometry zero features, hence zero angles.
    coords[unposed] = 0.0
    angles = RopeRigPE(_HEAD_DIM)(coords).reshape(-1, _HEAD_DIM)  # [N_gen,D]
    base_angles = torch.randn(plan.padded_gen_tokens, _HEAD_DIM // 2).repeat(1, 2)  # [N_gen,D]
    mrope = {"mrope_cos": base_angles.cos(), "mrope_sin": base_angles.sin()}  # each [N_gen,D]
    rigrope = dataclasses.replace(plan, rigrope_cos=angles.cos(), rigrope_sin=angles.sin(), **mrope)
    unrotated = dataclasses.replace(
        plan,
        rigrope_cos=torch.ones_like(angles),  # [N_gen,D]
        rigrope_sin=torch.zeros_like(angles),  # [N_gen,D]
        **mrope,
    )
    q_heads, k_heads = dense.rigrope_local_head_masks(_HEADS, _HEADS, 1)  # [H], [H_kv]
    split = dataclasses.replace(rigrope, rigrope_q_heads=q_heads, rigrope_k_heads=k_heads)
    baseline_packs = [dict(pack) for pack in packs]
    for index in (0, 1):
        baseline_packs[index]["full_only_seq"] = apply_mrope_rotary(
            get_full_only_seq(packs[index])[0], base_angles.cos(), base_angles.sin()
        )  # [N_gen,H,D]

    baseline = dense.multiview_maskless_gen_attention(*baseline_packs, plan=plan)  # [N_gen,H*D]
    full_rigrope = dense.multiview_maskless_gen_attention(*packs, plan=rigrope)  # [N_gen,H*D]
    content_only = dense.multiview_maskless_gen_attention(*packs, plan=unrotated)  # [N_gen,H*D]
    actual = dense.multiview_maskless_gen_attention(*packs, plan=split)  # [N_gen,H*D]

    torch.testing.assert_close(_heads(actual, 0)[posed], _heads(full_rigrope, 0)[posed], atol=0, rtol=0)
    torch.testing.assert_close(_heads(actual, 0)[unposed], _heads(content_only, 0)[unposed], atol=0, rtol=0)
    torch.testing.assert_close(_heads(actual, 1), _heads(baseline, 1), atol=0, rtol=0)
    assert not torch.allclose(_heads(actual, 0)[unposed], _heads(baseline, 0)[unposed])


def test_head_mask_must_match_the_local_heads() -> None:
    rigrope = torch.zeros(4, 2, 8)  # [N,H,D]
    with pytest.raises(ValueError, match="local heads"):
        dense._rigrope_or_mrope(rigrope, rigrope, None, torch.ones(3, dtype=torch.bool))


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "cp_size"),
    [(32, 8, 1), (32, 8, 2), (32, 8, 8), (32, 8, 16), (32, 8, 32), (16, 2, 4), (8, 8, 4)],
)
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_local_head_masks_follow_context_parallel_head_sharding(
    num_heads: int, num_kv_heads: int, cp_size: int, fraction: float
) -> None:
    rigrope_kv_heads = int(fraction * num_kv_heads)
    if rigrope_kv_heads == 0:
        pytest.skip("fraction selects no whole KV group")
    group = num_heads // num_kv_heads
    repeats = max(cp_size // num_kv_heads, 1)
    # Reproduce ``context_parallel_attention``'s layout: repeat each KV head, then shard both
    # head axes contiguously; global query head h reads global KV head h // group.
    kv_owner = torch.arange(num_kv_heads).repeat_interleave(repeats)  # [H_kv*repeats]
    q_masks, kv_masks = [], []
    for rank in range(cp_size):
        q_mask, kv_mask = dense.rigrope_local_head_masks(
            num_heads, num_kv_heads, rigrope_kv_heads, cp_rank=rank, cp_size=cp_size
        )  # [H_local], [H_kv_local]
        q_per_rank, kv_per_rank = num_heads // cp_size, num_kv_heads * repeats // cp_size
        assert q_mask.shape == (q_per_rank,) and kv_mask.shape == (kv_per_rank,)
        local_kv_owner = kv_owner[rank * kv_per_rank : (rank + 1) * kv_per_rank]  # [H_kv_local]
        torch.testing.assert_close(kv_mask, local_kv_owner < rigrope_kv_heads)
        for local_head in range(q_per_rank):
            global_head = rank * q_per_rank + local_head
            # The local KV slot this query head reads under the kernel's GQA grouping.
            local_kv = local_head // (q_per_rank // kv_per_rank)
            assert int(local_kv_owner[local_kv]) == global_head // group
            assert bool(q_mask[local_head]) == bool(kv_mask[local_kv]) == (global_head // group < rigrope_kv_heads)
        q_masks.append(q_mask)
        kv_masks.append(kv_mask)
    assert int(torch.cat(q_masks).sum()) == rigrope_kv_heads * group


def test_head_fraction_names_whole_kv_groups() -> None:
    assert _rigrope_kv_heads(1.0, 8) is None
    assert _rigrope_kv_heads(0.5, 8) == 4
    assert _rigrope_kv_heads(0.125, 8) == 1
    for fraction in (0.3, 0.05):
        with pytest.raises(ValueError, match="whole number"):
            _rigrope_kv_heads(fraction, 8)
