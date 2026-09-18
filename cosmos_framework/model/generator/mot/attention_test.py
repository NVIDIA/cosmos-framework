# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
import math
import random  # noqa: I001 - release import rewriting changes the package sort order.
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask

import cosmos_framework.model.generator.mot.attention as attention
from cosmos_framework.model.attention import attention as imaginaire_attention
from cosmos_framework.model.attention import multi_dimensional_attention_varlen
from cosmos_framework.model.attention.natten import NATTEN_SUPPORTED
from cosmos_framework.model.attention.varlen import generate_multi_dim_varlen_parameters
from cosmos_framework.utils.misc import set_torch_compile_options
from cosmos_framework.model.generator.mot import multiview_attention as multiview_attention_module
from cosmos_framework.model.generator.mot import multiview_maskless_attention
from cosmos_framework.model.generator.mot.attention import (
    build_packed_sequence,
)
from cosmos_framework.model.generator.mot.flex_attention import (
    FlexBackend,
    SensorMaskItem,
    build_multiview_block_mask,
    resolve_flex_backend,
)
from cosmos_framework.model.generator.mot.flex_attention_test import _FLASH_UNAVAILABLE_MARKERS
from cosmos_framework.model.generator.mot.multiview_attention import multiview_attention
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_all_seq,
    get_all_seq_unpadded,
    get_caption_seq_offsets,
    get_causal_seq,
    get_full_only_seq,
    get_gen_seq,
    get_num_real_samples,
    get_und_seq,
    has_pad_segment,
    prepare_sequence_pack_metadata,
    sequence_pack_from_packed_sequence,
    set_gen_seq,
    set_und_seq,
    zeros_like,
)
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence

MAX_SEQ_LEN = 24
SEQS_PER_BATCH = 4


def _foreign_split_info(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "max_causal_len": 1,
        "max_full_len": 1,
        "max_sample_len": 2,
        "split_lens": [1, 1],
        "attn_modes": ["causal", "full"],
        "sample_lens": [2],
        "is_three_way": False,
        "vision_token_shapes": None,
        "action_token_shapes": None,
        "num_action_tokens_per_supertoken": 0,
        "null_action_supertokens": False,
        "control_stream_token_ranges": None,
        "noisy_token_range": None,
        "control_weights": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def unwrap(fn):
    import torch.utils._pytree as pytree

    def unwrap_fn(a, s):
        args, kwargs = pytree.tree_unflatten(a, s)
        return fn(*args, **kwargs)

    return unwrap_fn


def wrap(fn):
    import torch.utils._pytree as pytree

    def wrap_fn(*args, **kwargs):
        a, s = pytree.tree_flatten((args, kwargs))
        return fn(a, s)

    return wrap_fn


def _test_attention_impls(
    impl_1: str,
    impl_2: str,
    atol_self: float = 1e-4,
    rtol_self: float = 0,
    atol_cmp: float = 1e-1,
    rtol_cmp: float = 0,
    atol_bwd_self: float = 1e-1,
    rtol_bwd_self: float = 0,
    atol_bwd_cmp: float = 1.5,
    rtol_bwd_cmp: float = 0,
):
    random.seed(42)
    torch.manual_seed(42)

    # Reset cache for every new test to avoid reusing cache from previous ones
    torch.compiler.reset()

    IMPL_TO_FN = {
        "two_way": attention.two_way_attention,
        "three_way": attention.three_way_attention,
    }

    assert impl_1 in IMPL_TO_FN
    assert impl_2 in IMPL_TO_FN
    assert impl_1 != impl_2

    fn_1 = IMPL_TO_FN[impl_1]
    fn_2 = IMPL_TO_FN[impl_2]

    use_compile = True
    test_backward: bool = True
    device = torch.device("cuda")
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    text_on_und_mode_only = True
    num_layers = 1

    # smaller seq length to expose off-by-one errors
    sample_lens = torch.randint(4, MAX_SEQ_LEN, (SEQS_PER_BATCH,), device=device, dtype=torch.int32)
    sample_lens = sample_lens.tolist()

    full_length = int(sum(sample_lens))

    # Generate `split_ids` with two splits per sample: always include 0, and a random int within range as intermediate for each sample in `sample_lens`.
    # packed_und_token_indexes takes the first split plust the first and last token of the second split.
    split_lens = []
    start = 0
    packed_und_token_indexes = []
    packed_gen_token_indexes = []
    position_ids = []
    attn_modes = ["causal", "full"] * len(sample_lens)
    token_shapes = []
    for length in sample_lens:
        assert length >= 4, f"sample_len must be >= 4, got {length}"

        und_extra = 1 if text_on_und_mode_only else 0
        gen_minus = 0 if text_on_und_mode_only else 1

        causal_len = int(torch.randint(1, length - 2 + und_extra, ()))
        split_lens.extend((causal_len, length - causal_len))

        und_len = causal_len if text_on_und_mode_only else causal_len + 1

        packed_und_token_indexes.extend(range(start, start + und_len))
        # generation part (latent noise)
        packed_gen_token_indexes.extend(range(start + und_len, start + length - gen_minus))
        if not text_on_und_mode_only:
            # final <IMGEND> token
            packed_und_token_indexes.append(start + length - 1)

        position_ids.extend(range(length))
        start += length

        token_shapes.append((1, length))

    real_len = sum(sample_lens)

    # Precompute LongTensor indices and common kwargs
    packed_und_idx_t = cast(torch.LongTensor, torch.tensor(packed_und_token_indexes, device=device, dtype=torch.long))
    packed_gen_idx_t = cast(torch.LongTensor, torch.tensor(packed_gen_token_indexes, device=device, dtype=torch.long))

    # Builders: return only the pack; retrieve the attention_meta explicitly when needed
    def _make_pack_multiview_dense(x, impl: str):
        return build_packed_sequence(
            impl,
            packed_sequence=x,
            attn_modes=attn_modes,
            split_lens=split_lens,
            sample_lens=sample_lens,
            packed_und_token_indexes=packed_und_idx_t,
            packed_gen_token_indexes=packed_gen_idx_t,
            num_heads=num_q_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            token_shapes=token_shapes,
        )[0]

    def make_pack_two_way(x):
        return _make_pack_multiview_dense(x, "two_way")

    def make_pack_three_way(x):
        return _make_pack_multiview_dense(x, "three_way")

    IMPL_TO_MAKE_PACK = {
        "two_way": make_pack_two_way,
        "three_way": make_pack_three_way,
    }

    packed_und_token_indexes = torch.tensor(packed_und_token_indexes, device=device, dtype=torch.int32)
    packed_gen_token_indexes = torch.tensor(packed_gen_token_indexes, device=device, dtype=torch.int32)
    position_ids = torch.tensor(position_ids, device=device, dtype=torch.int32)

    packed_qkv11 = torch.randn(
        full_length,
        num_q_heads + 2 * num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=test_backward,
    )
    packed_qkv12 = packed_qkv11.detach().clone().requires_grad_(test_backward)
    packed_qkv21 = packed_qkv11.detach().clone().requires_grad_(test_backward)
    packed_qkv22 = packed_qkv11.detach().clone().requires_grad_(test_backward)

    def split_qkv(qkv, make_pack):
        query = qkv[:, :num_q_heads, :]
        key = qkv[:, num_q_heads : num_q_heads + num_kv_heads, :]
        value = qkv[:, num_q_heads + num_kv_heads :, :]

        query_packed = make_pack(query.clone())
        key_packed = make_pack(key.clone())
        value_packed = make_pack(value.clone())

        if test_backward:
            # if we are running backward we cannot modify in place.
            query_packed2 = zeros_like(query_packed)
            key_packed2 = zeros_like(key_packed)
            value_packed2 = zeros_like(value_packed)

            set_gen_seq(query_packed2, get_gen_seq(query_packed))
            set_gen_seq(key_packed2, get_gen_seq(key_packed))
            set_gen_seq(value_packed2, get_gen_seq(value_packed))
        else:
            query_packed2 = query_packed
            key_packed2 = key_packed
            value_packed2 = value_packed

        # tweak non-causal tokens to see if they are properly masked
        set_und_seq(query_packed2, 2 * get_und_seq(query_packed))
        set_und_seq(key_packed2, 2 * get_und_seq(key_packed))
        set_und_seq(value_packed2, 2 * get_und_seq(value_packed))

        return query_packed2, key_packed2, value_packed2

    make_pack_1 = IMPL_TO_MAKE_PACK[impl_1]
    make_pack_2 = IMPL_TO_MAKE_PACK[impl_2]

    query_factored_1, key_factored_1, value_factored_1 = split_qkv(packed_qkv11, make_pack_1)
    query_factored_2, key_factored_2, value_factored_2 = split_qkv(packed_qkv21, make_pack_1)

    query_joint_1, key_joint_1, value_joint_1 = split_qkv(packed_qkv12, make_pack_2)
    query_joint_2, key_joint_2, value_joint_2 = split_qkv(packed_qkv22, make_pack_2)

    def compile(x):
        if use_compile:
            return torch.compile(x, fullgraph=True, backend="eager")
        else:
            return x

    class AttentionWrapper(torch.nn.Module):
        def __init__(self, attention_func, sdpa_func=None):
            super().__init__()
            self.attention_func = attention_func
            self.sdpa_func = sdpa_func

        def forward(self, *args, **kwargs):
            if self.sdpa_func is not None:
                kwargs["sdpa_func"] = self.sdpa_func
            return self.attention_func(*args, **kwargs)

    # NOTE: we should try and maintain only one copy of QKV offsets if they're identical
    # between queries and key/values, since this enables the "don't care" mask, which enables
    # more attention backends in I4 attention.
    if query_factored_1["_causal_seq_offsets"].equal(key_factored_1["_causal_seq_offsets"]) and query_factored_1[
        "_causal_seq_offsets"
    ].equal(value_factored_1["_causal_seq_offsets"]):
        key_factored_1["_causal_seq_offsets"] = query_factored_1["_causal_seq_offsets"]
        value_factored_1["_causal_seq_offsets"] = query_factored_1["_causal_seq_offsets"]

    if query_joint_1["_causal_seq_offsets"].equal(key_joint_1["_causal_seq_offsets"]) and query_joint_1[
        "_causal_seq_offsets"
    ].equal(value_joint_1["_causal_seq_offsets"]):
        key_joint_1["_causal_seq_offsets"] = query_joint_1["_causal_seq_offsets"]
        value_joint_1["_causal_seq_offsets"] = query_joint_1["_causal_seq_offsets"]

    if query_factored_2["_causal_seq_offsets"].equal(key_factored_2["_causal_seq_offsets"]) and query_factored_2[
        "_causal_seq_offsets"
    ].equal(value_factored_2["_causal_seq_offsets"]):
        key_factored_2["_causal_seq_offsets"] = query_factored_2["_causal_seq_offsets"]
        value_factored_2["_causal_seq_offsets"] = query_factored_2["_causal_seq_offsets"]

    if query_joint_2["_causal_seq_offsets"].equal(key_joint_2["_causal_seq_offsets"]) and query_joint_2[
        "_causal_seq_offsets"
    ].equal(value_joint_2["_causal_seq_offsets"]):
        key_joint_2["_causal_seq_offsets"] = query_joint_2["_causal_seq_offsets"]
        value_joint_2["_causal_seq_offsets"] = query_joint_2["_causal_seq_offsets"]

    kwargs_1 = {}
    kwargs_2 = {}

    # natten_metadata is a required argument, but setting it to None implements standard self attn.
    if impl_1 == "three_way":
        kwargs_1["natten_metadata"] = None
    elif impl_2 == "three_way":
        kwargs_2["natten_metadata"] = None

    output1_factored = compile(AttentionWrapper(fn_1))(
        query_factored_1,
        key_factored_1,
        value_factored_1,
        **kwargs_1,
    )
    torch.cuda.synchronize()
    output1_joint = compile(AttentionWrapper(fn_1))(
        query_joint_1,
        key_joint_1,
        value_joint_1,
        **kwargs_1,
    )
    torch.cuda.synchronize()

    output2_factored = compile(AttentionWrapper(fn_2))(
        query_factored_2,
        key_factored_2,
        value_factored_2,
        **kwargs_2,
    )
    torch.cuda.synchronize()
    output2_joint = compile(AttentionWrapper(fn_2))(
        query_joint_2,
        key_joint_2,
        value_joint_2,
        **kwargs_2,
    )
    torch.cuda.synchronize()

    # Independent packs for the same implementation should be the same.
    torch.testing.assert_close(
        get_all_seq_unpadded(output1_factored)[:real_len],
        get_all_seq_unpadded(output1_joint)[:real_len],
        atol=atol_self,
        rtol=rtol_self,
    )
    torch.testing.assert_close(
        get_all_seq_unpadded(output2_factored)[:real_len],
        get_all_seq_unpadded(output2_joint)[:real_len],
        atol=atol_self,
        rtol=rtol_self,
    )

    # impl 1 vs impl 2. needs more tolerance
    torch.testing.assert_close(
        get_all_seq_unpadded(output2_factored)[:real_len],
        get_all_seq_unpadded(output1_factored)[:real_len],
        atol=atol_cmp,
        rtol=rtol_cmp,
    )
    torch.testing.assert_close(
        get_all_seq_unpadded(output2_joint)[:real_len],
        get_all_seq_unpadded(output1_joint)[:real_len],
        atol=atol_cmp,
        rtol=rtol_cmp,
    )

    if test_backward:
        get_all_seq_unpadded(output1_joint)[:real_len].sum().backward()
        get_all_seq_unpadded(output2_joint)[:real_len].sum().backward()
        get_all_seq_unpadded(output1_factored)[:real_len].sum().backward()
        get_all_seq_unpadded(output2_factored)[:real_len].sum().backward()

        # should be close but not necessarily exactly the same because of aggregation order in bwd
        torch.testing.assert_close(
            packed_qkv11.grad[:real_len], packed_qkv12.grad[:real_len], atol=atol_bwd_self, rtol=rtol_bwd_self
        )
        torch.testing.assert_close(
            packed_qkv21.grad[:real_len], packed_qkv22.grad[:real_len], atol=atol_bwd_self, rtol=rtol_bwd_self
        )

        # different attention implementations, needs more tolerance
        torch.testing.assert_close(
            packed_qkv11.grad[:real_len], packed_qkv21.grad[:real_len], atol=atol_bwd_cmp, rtol=rtol_bwd_cmp
        )


@pytest.mark.L0
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="NATTEN is not available, or too old.")
def test_two_way_attention_vs_three_way_attention():
    _test_attention_impls("two_way", "three_way")


class _SampleCountTripwire:
    """Stands in for the sample count and fails whatever compares it.

    Obtaining the count is free; it is the comparison against a constant that specializes the
    compiled graph, so that is what this catches.
    """

    def __gt__(self, other: object) -> bool:
        raise AssertionError("the sample count was compared, which specializes the compiled graph on it")


def _use_varlen_with(num_samples: object, *, has_caption_offsets: bool = False) -> bool:
    return attention._use_varlen(
        cast(int, num_samples),
        has_caption_offsets=has_caption_offsets,
    )


@pytest.mark.L0
def test_use_varlen_does_not_compare_the_sample_count_while_training() -> None:
    """A pack's sample count varies from step to step, so comparing it costs a recompile.

    ``_use_varlen`` only wants the count to take a dense-attention shortcut that is gated to
    inference, and short-circuit evaluation is what keeps training away from it: the grad-mode
    test is the left operand of an ``or``, so training never reaches the comparison at all.
    """
    assert _use_varlen_with(_SampleCountTripwire()) is True


@pytest.mark.L0
def test_use_varlen_takes_the_dense_path_for_a_single_sample_without_grad() -> None:
    # One sample, so the varlen ranges would describe the whole tensor and buy nothing.
    with torch.no_grad():
        assert _use_varlen_with(1) is False


@pytest.mark.L0
def test_use_varlen_stays_varlen_for_several_samples_without_grad() -> None:
    # Two samples, which the dense API has no way to keep apart.
    with torch.no_grad():
        assert _use_varlen_with(2) is True


@pytest.mark.L0
def test_use_varlen_stays_varlen_for_caption_offsets_without_grad() -> None:
    with torch.no_grad():
        assert _use_varlen_with(_SampleCountTripwire(), has_caption_offsets=True) is True


@pytest.mark.L0
def test_und_self_attention_passes_caption_boundaries_during_single_sample_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-grad B=1 fast path still hands all caption ranges to causal attention.

    Asked of the multiview pathway's shared UND pass, which is where per-view captions live:
    a pack only carries their boundaries under separate_view_text_tokenization, and that is
    admitted on the multiview pathway alone.
    """
    caption_lens = [[3, 5]]
    und_indexes = torch.arange(8, dtype=torch.long)  # [N_und]
    gen_indexes = torch.arange(8, 16, dtype=torch.long)  # [N_gen]
    metadata = prepare_sequence_pack_metadata(
        sample_lens=[16],
        split_lens=[8, 8],
        attn_modes=["causal", "full"],
        packed_und_token_indexes=und_indexes,
        device=torch.device("cpu"),
        text_caption_lens=caption_lens,
    )

    def make_pack(values: torch.Tensor) -> SequencePack:
        return sequence_pack_from_packed_sequence(
            packed_sequence=values,
            attn_modes=["causal", "full"],
            split_lens=[8, 8],
            sample_lens=[16],
            packed_und_token_indexes=und_indexes,
            packed_gen_token_indexes=gen_indexes,
            prepared_metadata=metadata,
            text_caption_lens=caption_lens,
        )

    calls: list[dict[str, Any]] = []

    def fake_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del key, value
        calls.append(kwargs)
        return torch.zeros_like(query)

    monkeypatch.setattr(multiview_attention_module, "attention", fake_attention)
    qkv = torch.zeros(3, 16, 1, 2)  # [QKV,N,heads,head_dim]
    packs = tuple(make_pack(qkv[index]) for index in range(3))

    with torch.no_grad():
        multiview_attention_module.und_self_attention(*packs)

    assert len(calls) == 1
    # The last range is the packer's trailing padding segment; the first two are the
    # independent caption ranges whose loss in the old dense shortcut caused cross-caption
    # attention.
    assert calls[0]["cumulative_seqlen_Q"].tolist() == [0, 3, 8, 9]
    assert calls[0]["cumulative_seqlen_KV"].tolist() == [0, 3, 8, 9]
    assert calls[0]["max_seqlen_Q"] == 5


@pytest.mark.L0
def test_build_packed_sequence_rejects_flex():
    device = torch.device("cpu")
    packed_sequence = torch.randn(4, 8, device=device)  # [N,D]
    packed_und_token_indexes = torch.tensor([0, 1], device=device, dtype=torch.long)  # [N_und]
    packed_gen_token_indexes = torch.tensor([2, 3], device=device, dtype=torch.long)  # [N_gen]

    with pytest.raises(ValueError, match="Must be 'two_way' or 'three_way'"):
        build_packed_sequence(
            "flex",
            packed_sequence=packed_sequence,
            attn_modes=["causal", "full"],
            split_lens=[2, 2],
            sample_lens=[4],
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            num_heads=1,
            head_dim=8,
            num_layers=1,
        )


@pytest.mark.L0
def test_prepared_sequence_pack_metadata_is_reused() -> None:
    device = torch.device("cpu")
    packed_sequence = torch.randn(4, 8, device=device)  # [N,D]
    packed_und_token_indexes = torch.tensor([0, 1], device=device, dtype=torch.long)  # [N_und]
    packed_gen_token_indexes = torch.tensor([2, 3], device=device, dtype=torch.long)  # [N_gen]
    metadata = prepare_sequence_pack_metadata(
        sample_lens=[4],
        split_lens=[2, 2],
        attn_modes=["causal", "full"],
        packed_und_token_indexes=packed_und_token_indexes,
        device=device,
    )

    first_pack = sequence_pack_from_packed_sequence(
        packed_sequence=packed_sequence,
        attn_modes=["causal", "full"],
        split_lens=[2, 2],
        sample_lens=[4],
        packed_und_token_indexes=packed_und_token_indexes,
        packed_gen_token_indexes=packed_gen_token_indexes,
        prepared_metadata=metadata,
    )
    second_pack = sequence_pack_from_packed_sequence(
        packed_sequence=packed_sequence,
        attn_modes=["causal", "full"],
        split_lens=[2, 2],
        sample_lens=[4],
        packed_und_token_indexes=packed_und_token_indexes,
        packed_gen_token_indexes=packed_gen_token_indexes,
        prepared_metadata=metadata,
    )

    assert first_pack["_causal_indices"] is metadata.causal_indices
    assert second_pack["_causal_indices"] is metadata.causal_indices
    torch.testing.assert_close(get_all_seq_unpadded(first_pack), get_all_seq_unpadded(second_pack))


@pytest.mark.L0
def test_sequence_pack_padding_keeps_sample_ids_aligned() -> None:
    device = torch.device("cpu")
    packed_sequence = torch.randn(4, 8, device=device)  # [N,D]
    packed_und_token_indexes = torch.tensor([0, 2], device=device, dtype=torch.long)  # [N_und]
    packed_gen_token_indexes = torch.tensor([1, 3], device=device, dtype=torch.long)  # [N_gen]

    pack = sequence_pack_from_packed_sequence(
        packed_sequence=packed_sequence,
        attn_modes=["causal", "full", "causal", "full"],
        split_lens=[1, 1, 1, 1],
        sample_lens=[2, 2],
        packed_und_token_indexes=packed_und_token_indexes,
        packed_gen_token_indexes=packed_gen_token_indexes,
        causal_seq_alignment=4,
        full_seq_alignment=4,
    )

    assert get_und_seq(pack).shape[0] == pack["_causal_sample_ids"].shape[0] == 4
    assert get_gen_seq(pack).shape[0] == pack["_full_only_sample_ids"].shape[0] == 4
    torch.testing.assert_close(pack["_causal_sample_ids"], torch.tensor([0, 1, 2, 2]))
    torch.testing.assert_close(pack["_full_only_sample_ids"], torch.tensor([0, 1, 2, 2]))


@pytest.mark.L0
def test_prepared_sequence_pack_metadata_rejects_another_layout() -> None:
    device = torch.device("cpu")
    packed_sequence = torch.randn(4, 8, device=device)  # [N,D]
    packed_und_token_indexes = torch.tensor([0, 1], device=device, dtype=torch.long)  # [N_und]
    packed_gen_token_indexes = torch.tensor([2, 3], device=device, dtype=torch.long)  # [N_gen]
    metadata = prepare_sequence_pack_metadata(
        sample_lens=[4],
        split_lens=[2, 2],
        attn_modes=["causal", "full"],
        packed_und_token_indexes=packed_und_token_indexes,
        device=device,
    )

    with pytest.raises(ValueError, match="does not match"):
        sequence_pack_from_packed_sequence(
            packed_sequence=packed_sequence,
            attn_modes=["causal", "full"],
            split_lens=[1, 3],
            sample_lens=[4],
            packed_und_token_indexes=packed_und_token_indexes[:1],
            packed_gen_token_indexes=packed_gen_token_indexes,
            prepared_metadata=metadata,
        )


@pytest.mark.L0
def test_dispatch_attention_accepts_structurally_compatible_split_info(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_output = object()

    def fake_two_way_attention(*args: object, **kwargs: object) -> object:
        return expected_output

    monkeypatch.setattr(attention, "two_way_attention", fake_two_way_attention)
    foreign_split_info = _foreign_split_info()

    output, kv_to_store = attention.dispatch_attention(
        object(),
        object(),
        object(),
        foreign_split_info,
    )

    assert output is expected_output
    assert kv_to_store is None


@pytest.mark.L0
def test_dispatch_attention_rejects_incomplete_split_info() -> None:
    foreign_split_info = _foreign_split_info()
    del foreign_split_info.control_weights

    with pytest.raises(TypeError, match="Unsupported attention metadata"):
        attention.dispatch_attention(
            object(),
            object(),
            object(),
            foreign_split_info,
        )


@pytest.mark.L0
def test_decoder_layer_optimized_path_empty_und_tensor_shape():
    """Empty und tensors in the optimized AR path must be 2D, not 1D.

    In the optimized path (frame > 0, KV cache active), the decoder layer creates empty
    und tensors for all intermediate und variables.  These tensors are stored as
    ``causal_seq`` in the output SequencePack, and the *next* decoder layer
    calls ``get_und_seq(input)`` to retrieve them.  If they are 1D ``[0]``, a subsequent
    RMSNorm ``weight [H] * hidden_states [0]`` triggers:
        RuntimeError: The size of tensor a (H) must match tensor b (0) at non-singleton dim 0
    because broadcasting requires one dim to be 1, but H != 0.

    The fix is ``.new_empty(0, X.shape[-1])`` which yields 2D ``[0, H]``.
    """
    hidden_dim = 32
    device = torch.device("cpu")
    dtype = torch.float32

    # Old (buggy): torch.empty(0, ...) produces 1D [0]
    old_und = torch.empty(0, device=device, dtype=dtype)  # [0]
    assert old_und.shape == (0,), "sanity: old code creates 1D tensor"

    # Simulate RMSNorm: weight [H] * hidden_states [0]  → fails
    weight = torch.ones(hidden_dim, device=device, dtype=dtype)  # [H]
    with pytest.raises(RuntimeError):
        _ = weight * old_und  # [H] * [0] → dimension mismatch

    # New (fixed): .new_empty(0, H) produces 2D [0, H]
    ref = torch.randn(4, hidden_dim, device=device, dtype=dtype)  # [S_gen, H]
    new_und = ref.new_empty(0, ref.shape[-1])  # [0, H]
    assert new_und.shape == (0, hidden_dim), "fix: 2D tensor with correct hidden dim"

    # RMSNorm on 2D empty tensor succeeds (result is also [0, H])
    norm_out = weight * new_und  # [H] * [0, H] → [0, H]
    assert norm_out.shape == (0, hidden_dim)

    # Verify round-trip through SequencePack preserves 2D shape.
    # from_mode_splits(und, gen, meta) stores und as causal_seq; get_und_seq retrieves it.
    meta = {"causal_seq": new_und, "full_only_seq": ref}
    retrieved = get_und_seq(meta)  # type: ignore[arg-type]
    assert retrieved.shape == (0, hidden_dim), "get_und_seq must return 2D tensor"


def _split_info_for_multi_control_test() -> attention.SplitInfo:
    return attention.SplitInfo(split_lens=[1, 1], attn_modes=["causal", "full"], sample_lens=[2], actual_len=2)


def _annotate_multi_control_ranges_for_test(
    attention_meta: attention.SplitInfo, packed_seq: PackedSequence, *, n_gen: int
) -> None:
    pytest.importorskip("transformers", reason="cosmos3_vfm_network requires the Cosmos3 network dependencies.")
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import _annotate_multi_control_ranges

    _annotate_multi_control_ranges(attention_meta, packed_seq, n_gen=n_gen)


@pytest.mark.L0
def test_multi_control_range_annotation_ignores_single_control_weight() -> None:
    attention_meta = _split_info_for_multi_control_test()
    packed_seq = PackedSequence(vision_item_split_lens=[[3, 4]], control_weights=[[1.0]])

    _annotate_multi_control_ranges_for_test(attention_meta, packed_seq, n_gen=7)

    assert attention_meta.control_stream_token_ranges is None
    assert attention_meta.noisy_token_range is None
    assert attention_meta.control_weights is None


@pytest.mark.L0
def test_multi_control_range_annotation_sets_ranges_for_multiple_controls() -> None:
    attention_meta = _split_info_for_multi_control_test()
    packed_seq = PackedSequence(vision_item_split_lens=[[2, 3, 5]], control_weights=[[0.25, 0.75]])

    _annotate_multi_control_ranges_for_test(attention_meta, packed_seq, n_gen=10)

    assert attention_meta.control_stream_token_ranges == [(0, 2), (2, 5)]
    assert attention_meta.noisy_token_range == (5, 10)
    assert attention_meta.control_weights == [0.25, 0.75]


@pytest.mark.L0
def test_multi_control_range_annotation_rejects_per_view_captions() -> None:
    """Multi-control routing has no per-caption boundaries, so refuse the pack that needs them.

    Setting the ranges sends the pack to ``multi_control_two_way_attention``, which reads the
    per-sample causal offsets and never the per-caption ones. A per-view pack would run with
    every caption attending every other -- no error, no wrong-looking loss -- which is the
    failure per-view captions exist to prevent.
    """
    attention_meta = _split_info_for_multi_control_test()
    packed_seq = PackedSequence(
        vision_item_split_lens=[[2, 3, 5]],
        control_weights=[[0.25, 0.75]],
        text_caption_lens=[[3, 2]],
        text_caption_view_ids=[[0, 1]],
    )

    with pytest.raises(ValueError, match="per-view captions and multiple control streams"):
        _annotate_multi_control_ranges_for_test(attention_meta, packed_seq, n_gen=10)

    assert attention_meta.control_stream_token_ranges is None, "the pack must not be annotated"


@pytest.mark.L0
def test_multi_control_range_annotation_allows_a_single_caption_per_sample() -> None:
    """The guard keys on the per-view layout, not on the presence of caption bookkeeping."""
    attention_meta = _split_info_for_multi_control_test()
    packed_seq = PackedSequence(
        vision_item_split_lens=[[2, 3, 5]],
        control_weights=[[0.25, 0.75]],
        text_caption_lens=[[5]],
        text_caption_view_ids=[[-1]],
    )

    _annotate_multi_control_ranges_for_test(attention_meta, packed_seq, n_gen=10)

    assert attention_meta.control_stream_token_ranges == [(0, 2), (2, 5)]


@pytest.mark.L0
def test_multi_control_range_annotation_rejects_inconsistent_token_count() -> None:
    attention_meta = _split_info_for_multi_control_test()
    packed_seq = PackedSequence(vision_item_split_lens=[[2, 3, 5]], control_weights=[[0.25, 0.75]])

    with pytest.raises(AssertionError, match="packing inconsistency"):
        _annotate_multi_control_ranges_for_test(attention_meta, packed_seq, n_gen=9)


# ── two_way_attention on the multiview FlexAttention mask ────────────────────
# The generator's full attention has two implementations of "every GEN token attends to
# its whole sample": the dense varlen kernel, and a single FlexAttention call over the
# fused ``[UND | GEN]`` key stream under the multiview supertoken mask. The mask adds a
# restriction on the GEN->GEN quadrant that no dense kernel can encode -- but only when
# there is conditioning to restrict. With every GEN token noisy the two express the same
# thing, which makes the dense path an independent reference for the flex one, end to
# end: the same packs, the same q/k/v, one kernel against the other.
#
# ``flex_attention_test`` covers what the mask *contains*, on CPU and against a per-token
# reference, conditioning included. What it cannot reach is the rest of the path: the fused
# key concatenation, the kernel options, the backward, and what becomes of all of it inside
# the ``torch.compile``d decoder block when the sequence length changes from step to step,
# which is every step of a real training run.


@dataclass(frozen=True)
class _MultiviewShape:
    """The token geometry of one packed multiview batch.

    One UND (caption) split and one GEN (vision) split per sample, the GEN split holding a
    single camera-major vision item of ``(latent_t, patch_h, patch_w)`` laid out over
    ``num_views`` cameras -- the layout the packer produces and
    ``build_multiview_block_mask`` describes.
    """

    und_lens: tuple[int, ...]
    token_shapes: tuple[tuple[int, int, int], ...]
    num_views: tuple[int, ...]

    @property
    def gen_lens(self) -> tuple[int, ...]:
        """GEN tokens per sample."""
        return tuple(latent_t * patch_h * patch_w for latent_t, patch_h, patch_w in self.token_shapes)

    @property
    def real_len(self) -> int:
        """Tokens in the pack, before either stream is padded."""
        return sum(self.und_lens) + sum(self.gen_lens)


# Seven batches sized so that *both* padded stream lengths differ from one to the next, under
# the Triton backend's 128-token block and the FlashAttention-4 query block of 256 alike: GEN
# 224/352/608/800/1200/119600/454480 tokens pad to 256/384/640/896/1280/119680/454528 on Triton
# and to 256/512/768/1024/1280/119808/454656 on FA4, UND 12/200/300/450/580/720/880 pad to
# 128/256/384/512/640/768/896. The padded lengths are what the kernels and the mask are shaped
# by, so real token counts that happened to round to the same multiple would leave the compiled
# graph seeing one shape twice.
#
# The last two are a training geometry rather than a scaled-up toy. One 101-frame 720p clip per
# camera is 26 latent frames (``1 + (101 - 1) // 4`` at the VAE's temporal factor of 4, the way
# ``lance_sft_video`` computes it) of 23x40 patches (720x1280 under 16x VAE and 2x patchify, the
# height rounded up), so 23,920 tokens per camera. The first packs 2- and 3-camera samples (130
# latent frames, ~120k tokens); the second is the full 11-camera MADS rig beside an 8-camera
# sample (286 and 208 latent frames, ~455k tokens between them), which is where
# ``av_wsm_transfer_16b``'s 11-view variant lives: it packs ~1.01M tokens per sample and shards
# them to ~253k per rank at CP=4, the same order as the 263k-token sample here. The sample is
# the unit that matters, since the mask is block diagonal across samples and so is what any one
# kernel launch attends over. Neither the mask nor the kernels care about absolute size, but the
# compiled graph's guards do: this is where a symbol that a small shape let pass has to hold,
# and where a per-shape recompile costs real time rather than being a line in a log.
#
# That last shape is also the sweep's cost floor -- ~455k tokens is a few GB of activations
# across the two attention paths and seconds of kernel time, in each of the four
# parametrizations -- so it is the one to trim first if this test ever has to get cheaper.
#
# Seven is also about as many as this sweep can hold. The static case turns every shape into
# its own Dynamo cache entry, and ``_trainer_shape_env`` leaves the recompile limit at torch's
# default of 8, past which Dynamo stops compiling and falls back to eager -- which would read
# here as a shape that failed to specialize.
#
# The tail is what the dynamic case asserts on, since it can only look past the framework's
# one-off 0/1 specialization recompile, so the list is kept long enough for that tail to be
# more than a single shape.
#
# ``latent_t`` is divisible by the view count of its item, which build_multiview_block_mask
# requires: the frame and view ids it derives are a (num_views, frames_per_view) grid over the
# item's frames.
#
# The sample count is 2 throughout. Dynamo guards on the length of the per-sample metadata
# lists, so varying it would recompile for a reason that has nothing to do with sequence
# length, and the graph count below could no longer say anything.
_MULTIVIEW_SHAPES = (
    _MultiviewShape(und_lens=(7, 5), token_shapes=((8, 4, 4), (6, 4, 4)), num_views=(4, 3)),
    _MultiviewShape(und_lens=(130, 70), token_shapes=((12, 4, 4), (10, 4, 4)), num_views=(4, 5)),
    _MultiviewShape(und_lens=(180, 120), token_shapes=((20, 4, 4), (18, 4, 4)), num_views=(4, 3)),
    _MultiviewShape(und_lens=(300, 150), token_shapes=((28, 4, 4), (22, 4, 4)), num_views=(4, 2)),
    _MultiviewShape(und_lens=(340, 240), token_shapes=((40, 4, 4), (35, 4, 4)), num_views=(5, 5)),
    # 101-frame 720p: 2 and 3 cameras, 26 latent frames each, on the 23x40 patch grid.
    _MultiviewShape(und_lens=(400, 320), token_shapes=((52, 23, 40), (78, 23, 40)), num_views=(2, 3)),
    # The same clips over the full 11-camera MADS rig, and over 8 cameras.
    _MultiviewShape(und_lens=(500, 380), token_shapes=((286, 23, 40), (208, 23, 40)), num_views=(11, 8)),
)


def _multiview_pack(x: torch.Tensor, shape: _MultiviewShape, backend: FlexBackend) -> SequencePack:
    """Pack ``x`` as the network packs a multiview batch, padded for ``backend``.

    The two alignments differ and are taken from the backend rather than picked: the GEN
    stream supplies the mask's rows and answers to the (coarser, on FA4) query block, while
    the UND stream is keys only and answers to the key block.
    """
    split_lens: list[int] = []
    und_indexes: list[int] = []
    gen_indexes: list[int] = []
    start = 0
    for und_len, gen_len in zip(shape.und_lens, shape.gen_lens):
        split_lens.extend((und_len, gen_len))
        und_indexes.extend(range(start, start + und_len))
        gen_indexes.extend(range(start + und_len, start + und_len + gen_len))
        start += und_len + gen_len

    return build_packed_sequence(
        "two_way",
        packed_sequence=x,
        attn_modes=["causal", "full"] * len(shape.und_lens),
        split_lens=split_lens,
        sample_lens=[und_len + gen_len for und_len, gen_len in zip(shape.und_lens, shape.gen_lens)],
        packed_und_token_indexes=cast(torch.LongTensor, torch.tensor(und_indexes, dtype=torch.long, device=x.device)),
        packed_gen_token_indexes=cast(torch.LongTensor, torch.tensor(gen_indexes, dtype=torch.long, device=x.device)),
        num_heads=x.shape[-2],
        head_dim=x.shape[-1],
        num_layers=1,
        full_seq_alignment=backend.full_seq_alignment,
        causal_seq_alignment=backend.causal_seq_alignment,
    )[0]


def _multiview_block_mask(pack: SequencePack, shape: _MultiviewShape, *, block_size: tuple[int, int]) -> BlockMask:
    """Build the GEN-tower mask from ``pack`` the way ``cosmos3_vfm_network`` does.

    No conditioning frames, so every GEN token is noisy. That is what leaves the dense path
    a valid reference: noisy->noisy is full within a sample and gen->und covers the rest, so
    the supertoken mask collapses to "every GEN token attends to its own sample".
    """
    full_only_seq, full_q_offsets = get_full_only_seq(pack)
    causal_seq, causal_offsets = get_causal_seq(pack)
    return build_multiview_block_mask(
        gen_seq_len=full_only_seq.shape[0],
        full_q_offsets=full_q_offsets,
        sensor_mask_items=[
            # One item per sample, none of it conditioning.
            [
                SensorMaskItem(
                    token_shape=token_shape,
                    condition_mask=torch.zeros(token_shape[0], dtype=torch.bool),
                    num_views=num_views,
                    view_offset=0,
                    is_control=False,
                    seconds_per_frame=1.0,
                    caption_access="camera",
                )
            ]
            for token_shape, num_views in zip(shape.token_shapes, shape.num_views)
        ],
        caption_mask_items=None,
        device=full_only_seq.device,
        block_size=block_size,
        und_seq_len=causal_seq.shape[0],
        causal_offsets=causal_offsets,
        attention_scope="all_views",
        decomposed_temporal_window_seconds=None,
        control_attends_sensor=False,
    )


@dataclass(frozen=True)
class _MultiviewBatch:
    """A packed multiview batch: one q/k/v leaf, its three packs, and the mask built for them."""

    qkv: torch.Tensor  # [3, real_len, heads, head_dim]; the only leaf, so grads land here
    packs: tuple[SequencePack, SequencePack, SequencePack]
    block_mask: BlockMask


def _multiview_batch(
    shape: _MultiviewShape, *, backend: FlexBackend, device: torch.device, seed: int
) -> _MultiviewBatch:
    """Pack a fresh q/k/v leaf for ``shape`` and build the GEN mask off it.

    ``seed`` fixes the values, so two calls with the same seed give two independent leaves
    holding identical tensors -- one per attention path, which is what lets the gradients be
    compared without the two graphs sharing a ``.grad`` accumulator.

    bf16 and a 64-wide head because the FlashAttention-4 CuTeDSL kernels are bf16/fp16 only
    and are compiled for 64- and 128-wide heads. It is the dtype training runs in anyway, so
    it is the comparison worth making.
    """
    torch.manual_seed(seed)
    qkv = torch.randn(3, shape.real_len, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    packs = tuple(_multiview_pack(qkv[i], shape, backend) for i in range(3))
    for pack in packs:
        for stream in (pack["causal_seq"], pack["full_only_seq"]):
            # Only the token count varies between steps, and the head dims must stay concrete:
            # FlexAttention's lowering cannot fold a symbolic head count, and the
            # FlashAttention-4 templates are instantiated per head width. In production these are
            # concrete because ``MoTAttention.forward`` specialises them (it has to, since Dynamo
            # lifts a module's int attributes into SymInts); here the streams are arguments to the
            # compiled callable, so ``dynamic=True`` would symbolise all three of their dims and
            # nothing downstream would pin the last two back. Marking eagerly, before the call, is
            # what reaches Dynamo in time to keep it from allocating symbols for them at all.
            torch._dynamo.mark_static(stream, 1)
            torch._dynamo.mark_static(stream, 2)
    return _MultiviewBatch(
        qkv=qkv,
        packs=cast(tuple[SequencePack, SequencePack, SequencePack], packs),
        block_mask=_multiview_block_mask(packs[0], shape, block_size=backend.block_size),
    )


class _FlexAttentionMeta:
    """The two fields ``dispatch_attention`` reads off ``SplitInfo`` to reach the flex path.

    Production hands the mask and the backend to the compiled decoder block as attributes of
    the ``SplitInfo`` it already passes, and ``two_way_attention`` picks them up from there.
    Handing them to ``torch.compile`` as arguments of their own is not the same thing:
    ``dynamic=True`` makes integer *inputs* symbolic, so the backend's block size and the
    mask's own lengths reach the graph as symbols rather than as the numbers the
    FlashAttention-4 template has to specialise on -- ``BLOCK_SIZE=(s75, s33)`` and a KV
    length unrelated to the key stream's, in the rejection this harness used to produce. One
    object, its fields replaced per shape, keeps them where production keeps them.
    """

    def __init__(self) -> None:
        self.flex_block_mask: BlockMask | None = None
        self.flex_backend: FlexBackend | None = None


def _flex_two_way(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_meta: _FlexAttentionMeta,
) -> SequencePack:
    """The multiview pathway on its masked backend, reading the mask off the metadata as production does."""
    return multiview_attention(
        packed_query_states,
        packed_key_states,
        packed_value_states,
        flex_block_mask=attention_meta.flex_block_mask,
        flex_backend=attention_meta.flex_backend,
    )


@contextlib.contextmanager
def _trainer_shape_env() -> Iterator[None]:
    """Compile under the shape-env settings a training job runs with, duck shaping included.

    ``ImaginaireTrainer.__init__`` calls ``set_torch_compile_options``, which turns duck
    shaping off; pytest leaves torch's default on. That is not a detail the flash backend can
    ignore: with duck shaping on, the mask's per-token fields share a symbol with the GEN
    stream they were measured from, Inductor binds that symbol into the ``mask_mod`` subgraph
    as a scalar, and the FlashAttention-4 lowering refuses it (``NYI: score_mod or mask_mod
    captures a dynamic scalar``). A training run lowers onto that backend across dozens of
    padded geometries in one graph, so the difference belongs here rather than in the backend.

    Restores the setting on the way out, since it is global: leaving duck shaping off would
    silently change how every later test in the session allocates symbols.
    """
    from torch.fx.experimental import _config as fx_config

    previous_duck_shape = fx_config.use_duck_shape
    # Mirrors duck shaping only. The trainer also raises the recompile limit, which this test
    # has no reason to want: it asserts an exact graph count, so a limit that hides
    # recompilations would hide the thing being measured.
    previous_recompile_limit = torch._dynamo.config.recompile_limit
    set_torch_compile_options(recompile_limit=previous_recompile_limit, use_duck_shape=False)
    try:
        yield
    finally:
        set_torch_compile_options(recompile_limit=previous_recompile_limit, use_duck_shape=previous_duck_shape)


class _GraphCounter:
    """A ``torch.compile`` backend that counts the graphs Dynamo hands it, then defers to Inductor.

    Feeding several sequence lengths through one compiled callable only shows anything if they
    share a graph rather than each specializing their own, and the count is what distinguishes
    the two. Inductor still does the compiling, so the kernels under test are the ones a
    training step runs -- which the flex path needs, since that is where FlexAttention lowers
    onto Triton or the FlashAttention-4 CuTeDSL templates.
    """

    def __init__(self) -> None:
        self.graphs = 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> object:
        # Imported here rather than at module scope: this is a private Inductor entry point,
        # and a rename should fail this one test instead of the whole module's collection.
        from torch._inductor.compile_fx import compile_fx

        self.graphs += 1
        return compile_fx(gm, example_inputs)


@contextlib.contextmanager
def _flex_lowering_or_skip(backend: FlexBackend) -> Iterator[None]:
    """Skip when the FlashAttention-4 lowering refuses the graph, rather than failing.

    Only the ``flash`` backend can get here: the Triton kernels always lower, so a failure on
    them is this code's and is reported. Anything that does not name the backend is re-raised
    either way -- a wrong result, a rejected shape or a mask the kernel disagrees with all have
    to fail. What names the backend is :data:`_FLASH_UNAVAILABLE_MARKERS`, shared with
    ``flex_attention_test`` so that both suites' idea of an unavailable backend stays one thing.
    """
    try:
        yield
    except Exception as e:  # noqa: BLE001 - narrowed by the backend and marker checks below
        message = f"{type(e).__name__}: {e}"
        if backend.name != "flash" or not any(marker in message for marker in _FLASH_UNAVAILABLE_MARKERS):
            raise
        pytest.skip(f"FlexAttention cannot lower onto the FlashAttention-4 backend here -- {message}")


# The level is per parametrization rather than on the test, because the two settings cost very
# different amounts. The static case compiles a graph per shape -- seven Inductor compilations,
# against the two the dynamic case shares across the whole sweep -- which does not fit the 60s
# per-test timeout the L0 job runs with. The nightly L1 job allows 600s. Only the
# dynamic setting is the training default, so it is the one worth paying for on every merge
# request.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="NATTEN is not available, or too old.")
@pytest.mark.parametrize(
    "compile_dynamic",
    [
        pytest.param(True, id="compile-dynamic", marks=pytest.mark.L0),
        pytest.param(False, id="compile-static", marks=pytest.mark.L1),
    ],
)
@pytest.mark.parametrize("backend_preference", ["flex_triton", "flex_flash"])
def test_two_way_attention_flex_matches_dense_across_batch_shapes(
    backend_preference: str, compile_dynamic: bool
) -> None:
    """The compiled flex path agrees with the dense one, forward and backward, at every shape.

    Seven batches of different padded lengths go through one compiled callable, which is how
    this code is reached in practice: ``parallelize_unified_mot.apply_compile`` wraps every
    decoder block in ``torch.compile(fullgraph=True, dynamic=config.compile_dynamic)``, and the
    packer's output length follows the batch.

    The first is the shapes themselves, which is why both settings of that knob are covered:
    they promise different things. ``compile_dynamic=True``, the training default, has to carry
    the sequence length symbolically and reuse its graph for every later batch; ``False``,
    specializes and pays a recompile per shape.
    """
    device = torch.device("cuda")
    try:
        backend = resolve_flex_backend(device, backend_preference)
    except ValueError as e:
        pytest.skip(str(e))
    torch.compiler.reset()
    counter = _GraphCounter()
    compiled_flex_two_way = torch.compile(_flex_two_way, fullgraph=True, backend=counter, dynamic=compile_dynamic)
    # One instance for every shape, its fields replaced per shape, as the network reuses
    # the SplitInfo it annotates: a fresh object per call would guard on a new identity and
    # recompile, which the graph count below would report as a shape problem.
    attention_meta = _FlexAttentionMeta()
    attention_meta.flex_backend = backend
    # Graphs each shape compiles, rather than the running total: what the two settings promise
    # is about the shapes after the first, and the first one's count is not theirs to make.
    graphs_added_per_shape: list[int] = []

    with _trainer_shape_env():
        for index, shape in enumerate(_MULTIVIEW_SHAPES):
            graphs_before_shape = counter.graphs
            label = f"{backend.name} backend, {sum(shape.gen_lens)} GEN / {sum(shape.und_lens)} UND tokens"
            dense_batch = _multiview_batch(shape, backend=backend, device=device, seed=index)
            flex_batch = _multiview_batch(shape, backend=backend, device=device, seed=index)

            dense_pack = attention.two_way_attention(*dense_batch.packs)
            attention_meta.flex_block_mask = flex_batch.block_mask
            with _flex_lowering_or_skip(backend):
                flex_pack = compiled_flex_two_way(*flex_batch.packs, attention_meta)

            # get_all_seq_unpadded gathers the two towers back into packed token order, the form the decoder
            # layer passes on. Real tokens only: the dense full branch leaves the padding rows
            # unwritten (its varlen offsets stop at the last real token) where the flex branch
            # writes them from the -1 sentinel, so they are neither comparable nor read downstream.
            dense_out = get_all_seq_unpadded(dense_pack)[: shape.real_len].float()
            flex_out = get_all_seq_unpadded(flex_pack)[: shape.real_len].float()
            torch.testing.assert_close(
                flex_out,
                dense_out,
                atol=1e-2,
                rtol=1e-2,
                msg=lambda m, at=label: f"forward, {at}: {m}",
            )

            torch.manual_seed(1000 + index)
            weights = torch.randn_like(dense_out)
            with _flex_lowering_or_skip(backend):
                (flex_out * weights).sum().backward()
            (dense_out * weights).sum().backward()

            assert flex_batch.qkv.grad is not None and dense_batch.qkv.grad is not None
            # Looser than the forward: the two kernels reduce over the sample in different orders
            # and the leaves are bf16, so the gradients agree to rather fewer digits than the
            # outputs do. A mask that admitted the wrong tokens would move them by O(1).
            for name, flex_grad, dense_grad in zip(("dq", "dk", "dv"), flex_batch.qkv.grad, dense_batch.qkv.grad):
                torch.testing.assert_close(
                    flex_grad.float(),
                    dense_grad.float(),
                    atol=1e-2,
                    rtol=1e-2,
                    msg=lambda m, n=name, at=label: f"{n}, {at}: {m}",
                )

            graphs_added_per_shape.append(counter.graphs - graphs_before_shape)

    assert graphs_added_per_shape[0] > 0, "Nothing was compiled: the flex path did not reach the counting backend."
    if compile_dynamic:
        # The second shape is allowed one more graph, and it is not about the sequence length:
        # Dynamo specializes 0/1-valued properties rather than symbolising them, so the first
        # trace bakes in the packed streams' storage offsets and hands out a general graph when
        # a later pack violates that guard (``2 <= args[0]['causal_seq'].storage_offset()``, in
        # TORCH_LOGS=recompiles on a training run). That is paid once -- a training run pays it
        # in its first iteration, where every layer shares the frame, and then reuses the graph
        # across dozens of geometries. A length-driven recompile instead fires for every new
        # shape, so it is the tail of this sweep that tells the two apart.
        assert not any(graphs_added_per_shape[2:]), (
            f"compile_dynamic=True compiled {graphs_added_per_shape} graphs per batch shape: the sweep has to "
            "converge on one symbolic graph, and a shape that still compiles its own after the first two means "
            "the sequence length is being specialized -- a recompile on every training step."
        )
    else:
        assert all(added > 0 for added in graphs_added_per_shape[1:]), (
            f"compile_dynamic=False compiled {graphs_added_per_shape} graphs per batch shape: each shape is "
            "supposed to specialize its own, so a shape that reused an earlier graph is not being specialized "
            "on the sequence length the way that setting says it is."
        )


def _two_way_pack(
    x: torch.Tensor,
    und_lens: Sequence[int],
    gen_lens: Sequence[int],
    *,
    full_seq_alignment: int,
    causal_seq_alignment: int,
) -> SequencePack:
    """Pack ``x`` as one UND and one GEN split per sample, padded to the two alignments.

    The alignments are the knob, rather than a ``FlexBackend``: what the two tests below need
    is a pack whose GEN stream is longer than its real token count, and the dense path they
    exercise is the one taken when no flex mask comes along. Context parallel reaches the same
    state through ``cp_world_size``, which ``_get_padded_size`` folds into the same alignment.
    """
    split_lens: list[int] = []
    und_indexes: list[int] = []
    gen_indexes: list[int] = []
    start = 0
    for und_len, gen_len in zip(und_lens, gen_lens):
        split_lens.extend((und_len, gen_len))
        und_indexes.extend(range(start, start + und_len))
        gen_indexes.extend(range(start + und_len, start + und_len + gen_len))
        start += und_len + gen_len

    return build_packed_sequence(
        "two_way",
        packed_sequence=x,
        attn_modes=["causal", "full"] * len(und_lens),
        split_lens=split_lens,
        sample_lens=[und_len + gen_len for und_len, gen_len in zip(und_lens, gen_lens)],
        packed_und_token_indexes=cast(torch.LongTensor, torch.tensor(und_indexes, dtype=torch.long, device=x.device)),
        packed_gen_token_indexes=cast(torch.LongTensor, torch.tensor(gen_indexes, dtype=torch.long, device=x.device)),
        num_heads=x.shape[-2],
        head_dim=x.shape[-1],
        num_layers=1,
        full_seq_alignment=full_seq_alignment,
        causal_seq_alignment=causal_seq_alignment,
    )[0]


def _natten_varlen_multi_dim_supported() -> bool:
    """Whether ``multi_dimensional_attention_varlen`` can actually run here.

    ``NATTEN_SUPPORTED`` is the coarser gate -- NATTEN present and new enough for the dense
    multi-dim ops. Varlen multi-dim landed later, and ``generate_multi_dim_varlen_parameters``
    raises rather than degrades when it is missing, so a test guarded on the coarse flag alone
    fails on an older NATTEN instead of skipping.
    """
    if not NATTEN_SUPPORTED:
        return False
    try:
        from cosmos_framework.model.attention.natten import NATTEN_VARLEN_MULTI_DIM_VERSION, natten_version_satisfies

        return bool(natten_version_satisfies(NATTEN_VARLEN_MULTI_DIM_VERSION))
    except Exception:
        return False


_NATTEN_VARLEN_MULTI_DIM = _natten_varlen_multi_dim_supported()

# 2**16, which bf16 holds exactly, so a row that still carries it compares equal on the nose.
# A finite marker rather than NaN on purpose: NaN only catches an unwritten row when the
# recycled block happened to hold NaN, while this catches one whatever the kernel does or does
# not write, and tells "written as zero" apart from "left as it was found".
_POISON = 65536.0


def _stage_poisoned_blocks(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, fill: float = _POISON, count: int = 8
) -> set[int]:
    """Leave ``fill``-filled blocks of ``shape`` in the caching allocator; return their addresses.

    A varlen kernel writes only the rows its cumulative ranges cover, so if it allocates its
    output with ``empty_like(q)`` the rows past the last offset keep whatever the block already
    held. Freshly recycled blocks are where that content comes from in a training step, and
    this stages several of them at exactly the size the kernel is about to ask for.

    The addresses come back because staging the blocks is not the same as the kernel *getting*
    one. Everything else the call allocates on the way -- the causal pass's own output, the
    ``get_all_seq_unpadded`` gather, the key concatenations -- competes for the same size class, so the
    output buffer may well be a block none of this ever touched. A test that assumed otherwise
    would read a freshly-zeroed page as proof that the kernel wrote it. :func:`_assert_from_a
    _staged_block` is what turns that assumption into a check.
    """
    blocks = [torch.full(shape, fill, dtype=dtype, device=device) for _ in range(count)]
    addresses = {block.data_ptr() for block in blocks}
    del blocks  # Back to the allocator's cache, poison and all.
    return addresses


def _skip_unless_from_a_staged_block(out: torch.Tensor, addresses: set[int]) -> None:
    """Skip unless ``out`` lives in one of the staged blocks, so its content means something.

    ``two_way_attention`` reshapes the kernel's output before packing it, but ``squeeze`` and
    ``flatten`` on a contiguous tensor are views, so the address survives to here. If it is not
    one of the staged ones then the buffer was never poisoned and its padded rows say nothing
    about whether the kernel wrote them -- which is a skip, not a pass.
    """
    if out.data_ptr() not in addresses:
        pytest.skip(
            "The kernel's output buffer was not one of the staged blocks, so its padded rows "
            "carry no evidence either way. Re-run, or widen the staging, to get a verdict."
        )


# Backends that ``cosmos_framework.model.attention.attention`` may dispatch a varlen call to; ``choose_backend``
# decides which one is used. We only test NATTEN here, since production runs exclusively use it.
# Flash3 on H100/H200 is expected to fail this test, as it leaves rows past the cumulative ranges
# unwritten. NATTEN instead zeros out those rows. We no longer rely on this property for
# correctness, but retain the test in case it becomes relevant again in the future.
_VARLEN_BACKENDS = ("natten",)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The varlen attention kernels require a GPU.")
@pytest.mark.parametrize("backend", _VARLEN_BACKENDS)
def test_varlen_attention_writes_query_rows_past_its_cumulative_ranges(backend: str) -> None:
    """The varlen primitive on its own: are rows outside ``cu_seqlens_Q`` written, or left as found?

    This is the question the two pack-level tests below inherit, asked where it can actually be
    answered. Inside ``two_way_attention`` the output buffer competes with everything else the
    call allocates at that size, so the staged block rarely reaches the kernel and the tests
    skip. Here the varlen call is the only thing allocating 64 KB, so the staged block usually
    does reach it -- and the backends return ``out`` as ``[total_tokens, H, Dv]`` sized by the
    *padded* query count, which is exactly the buffer in question.

    ``backend`` is pinned rather than left to ``choose_backend`` because the answer is the
    kernel's, not the frontend's: NATTEN, flash3 and flash2 are separate implementations behind
    one entry point, and one of them zeroing the uncovered rows says nothing about the others.

    A query length past ``cu_seqlens_Q[-1]`` is the shape the packer produces whenever it pads
    the GEN stream: ``sequence_pack_from_packed_sequence`` pads ``full_only_seq`` while
    ``_full_only_seq_offsets`` still stops at the last real token.
    """
    device = torch.device("cuda")
    heads, head_dim = 4, 64
    real, padded = 96, 128

    # Two samples covering [0, 96); rows 96..127 of q are outside every range. Only the query
    # stream is padded, which is the shape the two-way dense full pass has: its keys come from
    # get_all_seq_unpadded, which holds real tokens only.
    offsets = torch.tensor([0, 48, real], device=device, dtype=torch.int32)

    # Retried for the same reason the backward companion is: whether the staged block reaches the
    # kernel is the allocator's business, and a miss is a skip, which is not an answer. Fresh
    # tensors each round so nothing is held across attempts to compete for the size class.
    inner = None
    for attempt in range(24):
        torch.manual_seed(attempt)
        q = torch.randn(1, padded, heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, real, heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, real, heads, head_dim, device=device, dtype=torch.bfloat16)

        torch.cuda.synchronize()
        addresses = _stage_poisoned_blocks((padded, heads, head_dim), torch.bfloat16, device, count=64)
        try:
            out = imaginaire_attention(
                q,
                k,
                v,
                cumulative_seqlen_Q=offsets,
                cumulative_seqlen_KV=offsets,
                max_seqlen_Q=48,
                max_seqlen_KV=48,
                backend=backend,
            )
        except (ValueError, NotImplementedError, RuntimeError, AssertionError) as e:
            pytest.skip(f"The {backend} backend cannot run this varlen case here: {type(e).__name__}: {e}")

        candidate = out.squeeze(0)  # [padded,H,D], the kernel's own buffer
        if candidate.data_ptr() in addresses:
            inner = candidate.clone()
            break

    if inner is None:
        pytest.skip(
            "The output buffer never landed on a staged block across 24 attempts, so its rows past the "
            "ranges carry no evidence either way."
        )

    tail = inner[real:]  # [pad_rows,heads,head_dim]
    poisoned_rows = int((tail == _POISON).flatten(1).any(dim=1).sum())
    assert not poisoned_rows, (
        f"{poisoned_rows} of the {padded - real} query rows past cu_seqlens_Q[-1] came back holding the "
        f"poison their buffer was staged with: the {backend} varlen forward leaves them exactly as it found "
        "them. Any caller that pads its query stream past its offsets inherits whatever the recycled block "
        "held -- which is the two-way dense full pass, whose queries are the padded GEN stream."
    )


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The varlen attention kernels require a GPU.")
@pytest.mark.parametrize("backend", _VARLEN_BACKENDS)
def test_varlen_attention_backward_writes_query_grad_rows_past_its_ranges(backend: str) -> None:
    """The other half of the same question, on the backward: is ``dQ`` written past the ranges?

    The forward companion above settles the output buffer. The gradient is the half that would
    actually cost something: ``dQ`` rows for padded queries reach the q-projection's weight
    gradient, which sums over every row, so a non-finite one there lands on the whole weight
    rather than on the padding alone -- ``0 * NaN`` is NaN, not zero.

    ``dQ`` is allocated at the same ``[total_tokens, H, D]`` as the forward output, so the same
    staging works on it, and the same address check says whether the staging reached it.
    """
    device = torch.device("cuda")
    heads, head_dim = 4, 64
    real, padded = 96, 128
    offsets = torch.tensor([0, 48, real], device=device, dtype=torch.int32)

    # All three streams are padded, so all three gradients have rows past the ranges. The packer
    # pads both streams too -- causal_seq is the K/V of the causal pass -- so dK and dV carry the
    # same question dQ does, and testing only dQ would leave two thirds of it open.
    #
    # Each gradient competes for its size class with everything else the backward allocates, so a
    # single staging attempt usually misses and the address check skips, which says nothing.
    # Retrying with a fresh graph makes "the allocator did not cooperate" a question of patience
    # rather than a verdict. ``torch.autograd.grad`` rather than ``.backward()`` so the gradients
    # arrive as the backward produced them, with no AccumulateGrad in the way that might hand
    # back a copy of a buffer the kernel never touched.
    landed: dict[str, torch.Tensor] = {}
    for attempt in range(24):
        torch.manual_seed(attempt)
        q, k, v = (
            torch.randn(1, padded, heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )
        try:
            out = imaginaire_attention(
                q,
                k,
                v,
                cumulative_seqlen_Q=offsets,
                cumulative_seqlen_KV=offsets,
                max_seqlen_Q=48,
                max_seqlen_KV=48,
                backend=backend,
            )
        except (ValueError, NotImplementedError, RuntimeError, AssertionError) as e:
            pytest.skip(f"The {backend} backend cannot run this varlen case here: {type(e).__name__}: {e}")

        # Zero on the padded rows, as a real loss leaves them: it reads only the real tokens.
        grad_out = torch.randn_like(out)
        grad_out[:, real:] = 0

        torch.cuda.synchronize()
        addresses = _stage_poisoned_blocks((padded, heads, head_dim), torch.bfloat16, device, count=64)
        grads = torch.autograd.grad(out, (q, k, v), grad_out)
        for name, grad in zip(("dQ", "dK", "dV"), grads):
            if name not in landed and grad.squeeze(0).data_ptr() in addresses:
                landed[name] = grad.squeeze(0).clone()
        if len(landed) == 3:
            break

    if not landed:
        pytest.skip(
            "No gradient landed on a staged block across 24 attempts, so their rows past the ranges carry no "
            "evidence either way."
        )

    # Every gradient is reported, not just the first to fail. Which of the three a backend leaves
    # alone is the whole finding -- dQ maps to the padded query stream and dK/dV to the padded key
    # stream, and the passes in attention.py pad those independently -- so aborting on whichever
    # sorts first would hide most of the answer.
    verdicts: list[str] = []
    for name in ("dQ", "dK", "dV"):
        grad = landed.get(name)
        if grad is None:
            verdicts.append(f"{name}: untested (never landed on a staged block)")
            continue
        tail = grad[real:]  # [pad_rows,heads,head_dim]
        poisoned_rows = int((tail == _POISON).flatten(1).any(dim=1).sum())
        finite = bool(torch.isfinite(grad).all())
        verdicts.append(f"{name}: {poisoned_rows}/{padded - real} padded rows left as staged, finite={finite}")

    left_untouched = [
        name
        for name in ("dQ", "dK", "dV")
        if landed.get(name) is not None and bool((landed[name][real:] == _POISON).any())
    ]
    assert not left_untouched, (
        f"The {backend} varlen backward leaves {', '.join(left_untouched)} unwritten past the cumulative "
        f"ranges. Full verdict -- {'; '.join(verdicts)}. Those rows reach the matching projection's weight "
        "gradient, where dW sums over every row, so one non-finite row there takes out every weight rather "
        "than just the padding."
    )
    assert len(landed) == 3, (
        f"Only {sorted(landed)} landed on a staged block, so the rest are untested here. Full verdict -- "
        f"{'; '.join(verdicts)}. Re-run for a verdict on all three."
    )


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The varlen attention kernels require a GPU.")
@pytest.mark.skipif(
    not _NATTEN_VARLEN_MULTI_DIM,
    reason="Varlen multi-dimensional NATTEN requires NATTEN >= 0.21.9.dev0.",
)
def test_natten_varlen_attention_writes_rows_past_its_token_layouts() -> None:
    """The same question for NATTEN, which production training runs.

    NATTEN does not take ``cu_seqlens`` at all: ``generate_multi_dim_varlen_parameters`` derives
    its metadata from ``token_layout_list``, the per-sample spatial layouts, and those describe
    real tokens only -- ``build_natten_metadata`` builds them from ``vision_token_shapes``. So a
    padded GEN stream reaches ``multi_dimensional_attention_varlen`` with more rows than the
    layouts account for, which is the state ``sequence_packing/natten.py`` flags as a standing
    TODO ("we're assuming ... no padding in between ... We should either make sure this never
    happens, or have static checks in place").

    ``three_way_attention`` merges this output with the gen->und pass through
    ``merge_attentions``, so whatever lands in those rows propagates from there.
    """
    device = torch.device("cuda")
    heads, head_dim = 4, 64
    # Two samples of 4 supertokens x 16 spatial tokens, the shape build_natten_metadata
    # produces for temporal-causal packs: (T, num_action + H*W).
    token_layout_list = [(4, 16), (4, 16)]
    real = sum(t * s for t, s in token_layout_list)
    padded = real + 32

    metadata = generate_multi_dim_varlen_parameters(
        token_layout_list=token_layout_list,
        head_dim=head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=False,
        is_causal=(True, False),
    )

    torch.manual_seed(0)
    q = torch.randn(1, padded, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, padded, heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, padded, heads, head_dim, device=device, dtype=torch.bfloat16)

    torch.cuda.synchronize()
    addresses = _stage_poisoned_blocks((padded, heads, head_dim), torch.bfloat16, device, count=32)

    out = multi_dimensional_attention_varlen(q, k, v, metadata=metadata)

    inner = cast(torch.Tensor, out).squeeze(0)  # [padded,H,D]
    _skip_unless_from_a_staged_block(inner, addresses)

    tail = inner[real:]
    still_poisoned = int((tail == _POISON).any(dim=-1).sum())
    assert not still_poisoned, (
        f"{still_poisoned} of the {padded - real} rows past NATTEN's token layouts came back holding the "
        "poison their buffer was staged with: NATTEN leaves them as it found them, so a padded GEN stream "
        "carries whatever the recycled block held into merge_attentions and on to o_proj_moe_gen."
    )


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The varlen attention kernels require a GPU.")
def test_two_way_dense_gen_pass_writes_its_padded_query_rows() -> None:
    """Every row of the dense GEN pass's output is written, padding included.

    ``two_way_attention``'s causal pass reads the pad segment folded into ``_causal_seq_offsets`` when the
    pack carries one, so its padding is covered by a trailing segment and the kernel writes it.
    The GEN pass does not: it keys against ``get_all_seq_unpadded``, which holds real tokens only and
    whose ``sample_offsets`` have no matching extra segment, so it runs on the plain
    ``_full_only_seq_offsets``. Those stop at the last real GEN token while ``full_q`` is the
    padded stream, leaving the tail rows outside every cumulative range.

    What makes that worth a test rather than a comment is where those rows go next:
    ``unified_mot`` feeds the whole padded GEN stream into ``o_proj_moe_gen``, and a dense MLP
    is row-wise, so nothing between here and the projection re-zeros them.
    """
    device = torch.device("cuda")
    und_lens, gen_lens = (12, 20), (100, 140)
    real_len = sum(und_lens) + sum(gen_lens)
    real_gen = sum(gen_lens)

    qkv = torch.randn(3, real_len, 4, 64, device=device, dtype=torch.bfloat16)
    packs = cast(
        tuple[SequencePack, SequencePack, SequencePack],
        tuple(
            _two_way_pack(qkv[i], und_lens, gen_lens, full_seq_alignment=128, causal_seq_alignment=128)
            for i in range(3)
        ),
    )

    padded_gen = int(get_gen_seq(packs[0]).shape[0])
    assert padded_gen > real_gen, (
        f"The GEN stream came out unpadded ({padded_gen} rows for {real_gen} real tokens), so this test would "
        "assert nothing. Raise full_seq_alignment until the packer pads it."
    )

    # flash allocates the varlen output as empty_like(q), i.e. [padded_gen, heads, head_dim].
    addresses = _stage_poisoned_blocks((padded_gen, qkv.shape[-2], qkv.shape[-1]), torch.bfloat16, device)

    # No flex mask, so this takes the dense branch -- the one under test.
    out = attention.two_way_attention(*packs)

    gen_out = get_gen_seq(out)
    _skip_unless_from_a_staged_block(gen_out, addresses)
    tail = gen_out[real_gen:]
    still_poisoned = int((tail == _POISON).any(dim=-1).sum())
    assert not still_poisoned, (
        f"{still_poisoned} of the GEN stream's {padded_gen - real_gen} padded rows came back holding the "
        "poison the output buffer was staged with, so the dense GEN pass left them unwritten and they carry "
        "whatever the recycled block did. They reach o_proj_moe_gen from here."
    )
    assert torch.isfinite(tail).all(), (
        f"{int((~torch.isfinite(tail)).any(dim=-1).sum())} of the GEN stream's {padded_gen - real_gen} padded "
        "rows came back non-finite."
    )


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The varlen attention kernels require a GPU.")
def test_padded_gen_rows_do_not_poison_a_downstream_weight_gradient() -> None:
    """A projection reading the padded GEN stream gets a finite weight gradient.

    This is the consequence the row-level test above only implies. ``dW = X.T @ dOut`` sums over
    every row the projection saw, padding included. The loss never reads a padded row, so its
    ``dOut`` is zero there -- but zero times a non-finite ``X`` is NaN, not zero, and that NaN
    lands on the whole weight gradient rather than on the padded rows alone. The real tokens'
    contribution is finite by construction here, so a non-finite ``.grad`` can only have come
    through the padding.
    """
    device = torch.device("cuda")
    und_lens, gen_lens = (12, 20), (100, 140)
    real_len = sum(und_lens) + sum(gen_lens)
    real_gen = sum(gen_lens)

    qkv = torch.randn(3, real_len, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    packs = cast(
        tuple[SequencePack, SequencePack, SequencePack],
        tuple(
            _two_way_pack(qkv[i], und_lens, gen_lens, full_seq_alignment=128, causal_seq_alignment=128)
            for i in range(3)
        ),
    )
    padded_gen = int(get_gen_seq(packs[0]).shape[0])
    assert padded_gen > real_gen, "The GEN stream has to be padded for this test to assert anything."

    # Poison with NaN here rather than with the finite marker: this test is about what a
    # non-finite activation does to the gradient, so it stages the case that would.
    addresses = _stage_poisoned_blocks(
        (padded_gen, qkv.shape[-2], qkv.shape[-1]), torch.bfloat16, device, fill=float("nan")
    )

    out = attention.two_way_attention(*packs)

    # Stands in for o_proj_moe_gen, which unified_mot hands the whole padded GEN stream.
    # get_gen_seq is already [tokens, heads * head_dim]: two_way_attention flattens the head
    # axes before it packs the result.
    gen_out = get_gen_seq(out)
    _skip_unless_from_a_staged_block(gen_out, addresses)
    proj = torch.nn.Linear(gen_out.shape[-1], 8, device=device, dtype=torch.float32)
    projected = proj(gen_out.float())
    # Only the real rows reach the loss, exactly as the packer's loss indexes arrange.
    projected[:real_gen].sum().backward()

    assert proj.weight.grad is not None
    assert torch.isfinite(proj.weight.grad).all(), (
        "The projection's weight gradient came back non-finite. The padded GEN rows carry a "
        "non-finite activation and dW sums over them, so 0 * NaN puts NaN on every weight."
    )


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The varlen attention kernels require a GPU.")
def test_two_way_dense_full_pass_covers_its_padded_queries() -> None:
    """The dense full pass's cumulative ranges reach as far as its padded query stream does.

    The structural half of the padding story, which needs no allocator luck to assert. Whether a
    kernel leaves an uncovered row alone is the kernel's business and varies by backend (flash3
    does, NATTEN does not), so the pack has to cover them either way -- and covering them takes a
    segment on *both* sides, since a query range with no matching key range is an empty softmax.

    Before ``sample_offsets`` carried a pad segment the causal pass had that pairing and the dense full
    pass did not, which is the asymmetry this pins: its keys come from the interleaved stream,
    whose ``sample_offsets`` had no pad segment to pair the GEN queries' one against.
    """
    device = torch.device("cuda")
    und_lens, gen_lens = (12, 20), (100, 140)
    real_len = sum(und_lens) + sum(gen_lens)

    qkv = torch.randn(3, real_len, 4, 64, device=device, dtype=torch.bfloat16)
    pack = _two_way_pack(qkv[0], und_lens, gen_lens, full_seq_alignment=128, causal_seq_alignment=128)

    padded_gen = int(get_gen_seq(pack).shape[0])
    padded_und = int(get_und_seq(pack).shape[0])
    assert padded_gen > sum(gen_lens) and padded_und > sum(und_lens), (
        "Both streams have to be padded for the pad segments to exist at all."
    )

    assert has_pad_segment(pack), (
        "A padded two-way pack needs the interleaved stream's pad segment, or the dense full pass "
        "has nothing to pair its padded GEN queries against."
    )
    q_offsets = pack["_full_only_seq_offsets"]
    kv_offsets = pack["sample_offsets"]

    # The query ranges reach the end of the padded GEN stream, the key ranges the end of the
    # padded interleaved stream, so no row of either sits outside every range.
    assert int(q_offsets[-1]) == padded_gen, (
        f"The GEN query ranges stop at {int(q_offsets[-1])} but the stream holds {padded_gen} rows."
    )
    assert int(kv_offsets[-1]) == padded_und + padded_gen, (
        f"The key ranges stop at {int(kv_offsets[-1])} but the interleaved stream holds {padded_und + padded_gen} rows."
    )
    padded_all_seq, _, _ = get_all_seq(pack)
    assert padded_all_seq.shape[0] == int(kv_offsets[-1]), (
        "The padded interleaved stream and the offsets describing it have to agree on their length."
    )

    # Same segment count on both sides, so every query segment has exactly one key segment --
    # including the trailing pad segment, which would otherwise be an empty softmax.
    assert q_offsets.shape[0] == kv_offsets.shape[0], (
        f"The full pass pairs {q_offsets.shape[0] - 1} query segments against {kv_offsets.shape[0] - 1} key segments."
    )
    # And that trailing segment is non-empty on both sides.
    assert int(q_offsets[-1]) > int(q_offsets[-2]), "The GEN pad segment is empty."
    assert int(kv_offsets[-1]) > int(kv_offsets[-2]), "The interleaved pad segment is empty."

    # max_seqlen has to cover the pad segment too: a varlen kernel tiles up to it.
    assert pack["max_full_len"] >= padded_gen - sum(gen_lens)
    assert pack["max_sample_len"] >= (padded_und - sum(und_lens)) + (padded_gen - sum(gen_lens))


@pytest.mark.L0
@pytest.mark.CPU
def test_get_num_real_samples_excludes_the_pad_segment() -> None:
    """The pad segment is a pseudo-sample and must not be counted as one.

    ``sample_offsets`` describes it as a segment like any other, so the raw segment count is one
    too many on a padded pack. That difference is load-bearing: ``_use_varlen`` branches on this
    count, and counting the segment would take a one-sample pack to two and put it on the varlen
    path instead of the dense one.
    """
    device = torch.device("cpu")
    und_lens, gen_lens = (12, 20), (100, 140)
    x = torch.randn(sum(und_lens) + sum(gen_lens), 4, 64, device=device)
    pack = _two_way_pack(x, und_lens, gen_lens, full_seq_alignment=128, causal_seq_alignment=128)

    assert has_pad_segment(pack)
    assert pack["sample_offsets"].shape[0] - 1 == len(und_lens) + 1, (
        "the offsets should describe one segment per sample plus the pad segment"
    )
    assert get_num_real_samples(pack) == len(und_lens)


@pytest.mark.L0
@pytest.mark.CPU
def test_get_num_real_samples_counts_every_segment_without_a_pad_segment() -> None:
    """With no pad segment every segment is a real sample, so nothing is subtracted."""
    device = torch.device("cpu")
    gen_len = 100
    x = torch.randn(gen_len, 4, 64, device=device)
    pack = build_packed_sequence(
        "two_way",
        packed_sequence=x,
        attn_modes=["full"],
        split_lens=[gen_len],
        sample_lens=[gen_len],
        packed_und_token_indexes=cast(torch.LongTensor, torch.empty(0, dtype=torch.long, device=device)),
        packed_gen_token_indexes=cast(torch.LongTensor, torch.arange(gen_len, dtype=torch.long, device=device)),
        num_heads=x.shape[-2],
        head_dim=x.shape[-1],
        num_layers=1,
        full_seq_alignment=1,
        causal_seq_alignment=1,
    )[0]

    assert not has_pad_segment(pack)
    assert get_num_real_samples(pack) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_pad_segments_are_emitted_as_a_complete_set() -> None:
    """A padded pack carries every pad segment or none, never some of them.

    The three pairs are one mechanism: the causal pass reads the causal offsets, the dense full
    pass reads the full offsets against the interleaved ones, and a pack holding only the first
    two would leave the full pass on plain offsets while the causal pass was covered. That is the
    asymmetry the interleaved segment was added to remove, and a partial set would reinstate it
    without failing anything, so the constructor asserts instead of emitting a subset.
    """
    device = torch.device("cpu")
    und_lens, gen_lens = (12, 20), (100, 140)
    x = torch.randn(sum(und_lens) + sum(gen_lens), 4, 64, device=device)
    pack = _two_way_pack(x, und_lens, gen_lens, full_seq_alignment=128, causal_seq_alignment=128)

    present = [key for key in _PAD_SEGMENT_KEYS if key in pack]
    assert present == list(_PAD_SEGMENT_KEYS), f"Padded pack carries only {present}."

    # The three offset tensors describe the same number of segments, so every pass pairs its
    # query segments one to one with its key segments, pad segment included. The two towers hold
    # theirs folded into their own offsets; only the interleaved stream keeps a separate key,
    # because ``sample_offsets`` still has to describe get_all_seq_unpadded's real-tokens-only stream.
    assert (
        pack["_causal_seq_offsets"].shape[0]
        == pack["_full_only_seq_offsets"].shape[0]
        == pack["sample_offsets"].shape[0]
    )
    # And the max lengths are ints, not the tensors they sit beside -- the pairing that a typo
    # here would silently swap, since both are just dict entries.
    for key in ("max_causal_len", "max_full_len", "max_sample_len"):
        assert isinstance(pack[key], int), f"{key} should be an int, got {type(pack[key]).__name__}."
    for key in ("_causal_seq_offsets", "_full_only_seq_offsets", "sample_offsets"):
        assert isinstance(pack[key], torch.Tensor), f"{key} should be a tensor, got {type(pack[key]).__name__}."


# Every offsets array folds its pad segment in, so no key name records the fold any more; this
# flag is what ``runtime.has_pad_segment`` reads.
_PAD_SEGMENT_KEYS = ("_has_pad_segment",)


@pytest.mark.L0
@pytest.mark.CPU
def test_paired_splits_always_reserve_a_pad_segment() -> None:
    """Whenever the pad segment applies, it is reserved up front rather than only when
    alignment happened to force padding.

    The segment's one row is folded into the padded length before rounding, so a pack whose
    real token count already sits on an alignment boundary still gets it. Reserving it only
    when rounding produced spare rows was the bug: an already-aligned pack then carried no
    segment, and the padded rows it did have went back to being unwritten by the varlen kernel.
    Alignment 1 is the sharpest case -- nothing to round up to, so the old code reserved
    nothing at all.
    """
    device = torch.device("cpu")
    und_lens, gen_lens = (12, 20), (100, 140)
    x = torch.randn(sum(und_lens) + sum(gen_lens), 4, 64, device=device)
    pack = _two_way_pack(x, und_lens, gen_lens, full_seq_alignment=1, causal_seq_alignment=1)

    missing = [key for key in _PAD_SEGMENT_KEYS if key not in pack]
    assert not missing, f"A pack with one causal and one full split per sample must reserve {missing}."

    # The reserved row is real: both streams outgrow their token counts, so the trailing
    # segment each pad offset describes is non-empty and no row sits outside every range.
    assert get_und_seq(pack).shape[0] > sum(und_lens)
    assert get_gen_seq(pack).shape[0] > sum(gen_lens)
    assert int(pack["_causal_seq_offsets"][-1]) == get_und_seq(pack).shape[0]
    assert int(pack["_full_only_seq_offsets"][-1]) == get_gen_seq(pack).shape[0]


@pytest.mark.L0
@pytest.mark.CPU
def test_pack_without_paired_splits_carries_no_pad_segments() -> None:
    """The other side of the contract: nothing to pair, so no pad segments and callers fall back.

    The pad segment pairs the two streams segment for segment, so it only applies when every
    sample contributes both a causal and a full split. An AR no-text pack carries full splits
    only (see ``test_prepare_sequence_pack_metadata_no_causal_splits``), leaving the causal side
    with nothing to pair against, so the constructor emits no pad segment at all.
    """
    device = torch.device("cpu")
    gen_len = 100
    x = torch.randn(gen_len, 4, 64, device=device)
    pack = build_packed_sequence(
        "two_way",
        packed_sequence=x,
        attn_modes=["full"],
        split_lens=[gen_len],
        sample_lens=[gen_len],
        packed_und_token_indexes=cast(torch.LongTensor, torch.empty(0, dtype=torch.long, device=device)),
        packed_gen_token_indexes=cast(torch.LongTensor, torch.arange(gen_len, dtype=torch.long, device=device)),
        num_heads=x.shape[-2],
        head_dim=x.shape[-1],
        num_layers=1,
        full_seq_alignment=1,
        causal_seq_alignment=1,
    )[0]

    for key in _PAD_SEGMENT_KEYS:
        assert key not in pack, f"A pack with no causal splits should not carry {key}."

    # And the towers keep pad-free offsets: one entry per split plus the terminator, with the last
    # entry at the real token count rather than the padded one.
    assert pack["_full_only_seq_offsets"].shape[0] == 2
    assert int(pack["_full_only_seq_offsets"][-1]) == gen_len


@dataclass(frozen=True)
class _MultiviewMasklessBatch:
    """A single-sample multiview pack, plus the real (unpadded) tensors a reference needs."""

    packs: tuple[SequencePack, SequencePack, SequencePack]
    gen_q: torch.Tensor  # [V*F*S,heads,head_dim]
    gen_k: torch.Tensor  # [V*F*S,kv_heads,head_dim]
    gen_v: torch.Tensor  # [V*F*S,kv_heads,head_dim]
    und_k: torch.Tensor  # [N_und,kv_heads,head_dim]
    und_v: torch.Tensor  # [N_und,kv_heads,head_dim]


def _multiview_maskless_batch(
    *,
    und_len: int,
    token_shape: tuple[int, int, int],
    num_views: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    seed: int,
) -> _MultiviewMasklessBatch:
    """Pack one camera-major multiview sample the way the network packs one.

    The streams are padded to the Triton backend's 128-token blocks even though this path
    needs no alignment at all: a production pack is padded (for CUDA graphs, or for the flex
    mask), and the trailing rows are what the dense passes have to trim rather than attend.
    ``num_kv_heads`` below the query head count is the grouped-query layout the models run,
    which the view fold has to keep aligned.
    """
    torch.manual_seed(seed)
    gen_len = token_shape[0] * token_shape[1] * token_shape[2]
    real_len = und_len + gen_len
    q = torch.randn(real_len, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(real_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(real_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)

    shape = _MultiviewShape(und_lens=(und_len,), token_shapes=(token_shape,), num_views=(num_views,))
    backend = resolve_flex_backend(device, "flex_triton")
    packs = (
        _multiview_pack(q, shape, backend),
        _multiview_pack(k, shape, backend),
        _multiview_pack(v, shape, backend),
    )
    return _MultiviewMasklessBatch(
        packs=cast(tuple[SequencePack, SequencePack, SequencePack], packs),
        gen_q=q[und_len:],
        gen_k=k[und_len:],
        gen_v=v[und_len:],
        und_k=k[:und_len],
        und_v=v[:und_len],
    )


def _padded_gen_tokens(pack: SequencePack) -> int:
    """The GEN stream's padded length, which a plan's partitions have to cover.

    The plan addresses the pack's stream as it stands, padding included, so this is part of the
    geometry it describes rather than something the attention path works out per call.
    """
    return int(get_full_only_seq(pack)[0].shape[0])


def _plan(
    num_views: int,
    token_shape: tuple[int, int, int],
    device: torch.device,
    padded: int | None = None,
    attention_scope: str = "decomposed",
):
    """The single-sample plan."""
    return multiview_maskless_attention.build_multiview_maskless_plan(
        [num_views], [token_shape], device=device, padded_gen_tokens=padded, attention_scope=attention_scope
    )


def _multiview_maskless_reference(
    batch: _MultiviewMasklessBatch,
    *,
    num_views: int,
    frames_per_view: int,
    spatial_tokens: int,
    cross_view: bool = True,
) -> torch.Tensor:
    """What merging the three passes has to come out as, in float64.

    ``merge_attentions`` merges as if the passes' key sets had been concatenated, so a GEN key
    that is both in the query's view and at its frame appears twice in that concatenation. Two
    copies of one key with one score is the same distribution as one copy carrying twice the
    weight, which is what the multiplicity below encodes: 2 on the query's own ``(view, frame)``
    cell, 1 on the rest of its view and the rest of its frame, 0 elsewhere, and 1 on every
    caption token. The overlap is deliberate -- see ``multiview_maskless_attention``.
    """
    gen_q = batch.gen_q.double()  # [N_gen,heads,head_dim]
    heads, head_dim = gen_q.shape[1], gen_q.shape[2]
    # Grouped-query attention: every key head serves a contiguous run of query heads.
    group_size = heads // batch.gen_k.shape[1]
    gen_k = batch.gen_k.double().repeat_interleave(group_size, dim=1)  # [N_gen,heads,head_dim]
    gen_v = batch.gen_v.double().repeat_interleave(group_size, dim=1)  # [N_gen,heads,head_dim]
    und_k = batch.und_k.double().repeat_interleave(group_size, dim=1)  # [N_und,heads,head_dim]
    und_v = batch.und_v.double().repeat_interleave(group_size, dim=1)  # [N_und,heads,head_dim]

    device = gen_q.device
    # Camera-major ids: view-outer, frame-inner, spatial-innermost.
    view_id = torch.arange(num_views, device=device).repeat_interleave(frames_per_view * spatial_tokens)  # [N_gen]
    frame_id = torch.arange(frames_per_view, device=device).repeat_interleave(spatial_tokens).repeat(num_views)
    same_view = view_id[:, None] == view_id[None, :]  # [N_gen,N_gen]
    same_frame = frame_id[:, None] == frame_id[None, :]  # [N_gen,N_gen]
    # ``cross_view=False`` is the single-view case: the instant groups sit inside the one view
    # group, so the pass is skipped and each key is counted once, as the mask counts it.
    multiplicity = same_view.double()  # [N_gen,N_gen]
    if cross_view:
        multiplicity = multiplicity + same_frame.double()

    scale = head_dim**-0.5
    gen_scores = torch.einsum("ihd,jhd->hij", gen_q, gen_k) * scale  # [heads,N_gen,N_gen]
    und_scores = torch.einsum("ihd,jhd->hij", gen_q, und_k) * scale  # [heads,N_gen,N_und]
    peak = torch.maximum(gen_scores.max(dim=-1).values, und_scores.max(dim=-1).values)  # [heads,N_gen]
    gen_weights = multiplicity[None] * torch.exp(gen_scores - peak[..., None])  # [heads,N_gen,N_gen]
    und_weights = torch.exp(und_scores - peak[..., None])  # [heads,N_gen,N_und]

    numerator = torch.einsum("hij,jhd->ihd", gen_weights, gen_v) + torch.einsum(
        "hij,jhd->ihd", und_weights, und_v
    )  # [N_gen,heads,head_dim]
    denominator = gen_weights.sum(-1) + und_weights.sum(-1)  # [heads,N_gen]
    return numerator / denominator.transpose(0, 1)[..., None]  # [N_gen,heads,head_dim]


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [
        pytest.param(4, 4, id="mha"),
        # Both folds reshape q, k and v by axes the head count is not one of, so a grouped-query
        # layout has to survive them untouched -- and the cross-view fold does copy. This is the
        # case that catches a fold that mixed the head axis into the reshape.
        pytest.param(4, 2, id="gqa"),
    ],
)
@torch.no_grad()
def test_multiview_maskless_attention_matches_a_dense_reference(num_q_heads: int, num_kv_heads: int) -> None:
    """The three merged passes come out as one softmax over their concatenated key sets.

    The reference is that concatenation written densely in float64 (``_multiview_maskless_reference``),
    which is what pins both folds at once: a view fold that crossed views, or a frame fold that
    crossed frames, changes which keys a query sums over and no tolerance would hide it.
    """
    device = torch.device("cuda")
    num_views, frames_per_view, patch_h, patch_w = 3, 2, 2, 3
    token_shape = (num_views * frames_per_view, patch_h, patch_w)
    batch = _multiview_maskless_batch(
        und_len=5,
        token_shape=token_shape,
        num_views=num_views,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        device=device,
        seed=0,
    )

    out_pack = multiview_attention(
        *batch.packs, maskless_plan=_plan(num_views, token_shape, device, _padded_gen_tokens(batch.packs[0]))
    )

    gen_out = get_gen_seq(out_pack)  # [N_full,heads*head_dim]
    num_gen_tokens = batch.gen_q.shape[0]
    expected = _multiview_maskless_reference(
        batch,
        num_views=num_views,
        frames_per_view=frames_per_view,
        spatial_tokens=patch_h * patch_w,
    ).flatten(-2, -1)  # [N_gen,heads*head_dim]

    # bf16 q/k/v against a float64 reference: the observed gap is ~3e-3 at worst on a 0.16-rms
    # output, i.e. rounding. Anything that actually folded the wrong axis moves whole keys in
    # and out of the sum and lands orders of magnitude outside this.
    torch.testing.assert_close(gen_out[:num_gen_tokens].double(), expected, atol=1e-2, rtol=1e-2)
    # The pack was padded to the backend's block, and those rows belong to no view: the passes
    # trim them, so what lands there is the zero the re-pad writes rather than a stale row.
    assert torch.equal(gen_out[num_gen_tokens:], torch.zeros_like(gen_out[num_gen_tokens:]))


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [pytest.param(4, 4, id="mha"), pytest.param(4, 2, id="gqa")],
)
@torch.no_grad()
def test_multiview_maskless_attention_same_view_scope_matches_a_dense_reference(
    num_q_heads: int, num_kv_heads: int
) -> None:
    """``"same_view"`` is the view fold alone, and counts every key exactly once.

    The reference is the same concatenation as the decomposed case with the frame fold dropped
    (``cross_view=False``), which leaves a multiplicity of 1 across the query's own view and 0
    elsewhere. That is the mask's own counting, so unlike ``"decomposed"`` this scope is not a
    distinct attention pattern -- one partition cannot overlap itself, so there is no own-cell
    double weight for inclusion-exclusion to subtract back out.
    """
    device = torch.device("cuda")
    num_views, frames_per_view, patch_h, patch_w = 3, 2, 2, 3
    token_shape = (num_views * frames_per_view, patch_h, patch_w)
    batch = _multiview_maskless_batch(
        und_len=5,
        token_shape=token_shape,
        num_views=num_views,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        device=device,
        seed=0,
    )
    plan = _plan(
        num_views,
        token_shape,
        device,
        _padded_gen_tokens(batch.packs[0]),
        attention_scope="same_view",
    )
    # The cross-instant partition is not built at all, rather than built and merged at no
    # weight: the second kernel never runs, which is the whole cost difference between the two.
    assert plan.cross_view_empty
    assert plan.cross_view_gather is None
    assert plan.attention_scope == "same_view"

    out_pack = multiview_attention(*batch.packs, maskless_plan=plan)

    gen_out = get_gen_seq(out_pack)  # [N_full,heads*head_dim]
    num_gen_tokens = batch.gen_q.shape[0]
    expected = _multiview_maskless_reference(
        batch,
        num_views=num_views,
        frames_per_view=frames_per_view,
        spatial_tokens=patch_h * patch_w,
        cross_view=False,
    ).flatten(-2, -1)  # [N_gen,heads*head_dim]

    torch.testing.assert_close(gen_out[:num_gen_tokens].double(), expected, atol=1e-2, rtol=1e-2)
    assert torch.equal(gen_out[num_gen_tokens:], torch.zeros_like(gen_out[num_gen_tokens:]))


def _same_view_caption_reference(
    batch: _MultiviewMasklessBatch,
    group_id: torch.Tensor,
    reads_captions: torch.Tensor,
) -> torch.Tensor:
    """The ``"same_view"`` fold in float64, with the captions read only by the rows that may.

    ``group_id`` is each GEN token's ``(sample, axis, view)`` group and ``reads_captions`` a
    ``[N_gen, N_und]`` mask of which caption tokens each GEN token reads -- per-token rather than
    per-row so the one reference serves both caption layouts: the sample-level one, where a row
    reads all of them or none, and the per-view one, where a camera reads its own view's span.
    One partition cannot overlap itself, so every GEN key is counted once; a caption key is
    counted once where the mask admits it and not at all elsewhere, which is what
    ``lidar_attends_captions=False`` has to come out as.
    """
    gen_q = batch.gen_q.double()  # [N_gen,heads,head_dim]
    heads, head_dim = gen_q.shape[1], gen_q.shape[2]
    group_size = heads // batch.gen_k.shape[1]
    gen_k = batch.gen_k.double().repeat_interleave(group_size, dim=1)  # [N_gen,heads,head_dim]
    gen_v = batch.gen_v.double().repeat_interleave(group_size, dim=1)  # [N_gen,heads,head_dim]
    und_k = batch.und_k.double().repeat_interleave(group_size, dim=1)  # [N_und,heads,head_dim]
    und_v = batch.und_v.double().repeat_interleave(group_size, dim=1)  # [N_und,heads,head_dim]

    scale = head_dim**-0.5
    gen_scores = torch.einsum("ihd,jhd->hij", gen_q, gen_k) * scale  # [heads,N_gen,N_gen]
    und_scores = torch.einsum("ihd,jhd->hij", gen_q, und_k) * scale  # [heads,N_gen,N_und]
    peak = torch.maximum(gen_scores.max(dim=-1).values, und_scores.max(dim=-1).values)  # [heads,N_gen]
    same_group = (group_id[:, None] == group_id[None, :]).double()  # [N_gen,N_gen]
    gen_weights = same_group[None] * torch.exp(gen_scores - peak[..., None])  # [heads,N_gen,N_gen]
    und_weights = reads_captions.double()[None] * torch.exp(und_scores - peak[..., None])  # [heads,N_gen,N_und]

    numerator = torch.einsum("hij,jhd->ihd", gen_weights, gen_v) + torch.einsum("hij,jhd->ihd", und_weights, und_v)
    denominator = gen_weights.sum(-1) + und_weights.sum(-1)  # [heads,N_gen]
    return numerator / denominator.transpose(0, 1)[..., None]  # [N_gen,heads,head_dim]


# One sample owning a camera pair on the cameras' view axis beside a range clip on its own, the
# joint layout ``lidar_attends_captions`` exists for. The camera item is 2 views x 2 frames x 2
# spatial tokens and the range item 3 sweeps x 2, so the GEN stream is 8 camera tokens then 6
# range ones.
_JOINT_CAMERA_ITEM = (4, 1, 2)
_JOINT_LIDAR_ITEM = (3, 1, 2)
_JOINT_CAMERA_TOKENS = 8
_JOINT_LIDAR_TOKENS = 6
_JOINT_GEN_TOKENS = _JOINT_CAMERA_TOKENS + _JOINT_LIDAR_TOKENS
# Camera view 0, camera view 1, then the sweep -- the groups the same-view fold partitions into.
_JOINT_GROUP_ID = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]


def _joint_cam_lidar_plan(device: torch.device, padded: int, *, lidar_attends_captions: bool):
    """The joint sample's plan under ``"same_view"``, which counts every key once."""
    return multiview_maskless_attention.build_multiview_maskless_plan(
        [2, 1],
        [_JOINT_CAMERA_ITEM, _JOINT_LIDAR_ITEM],
        device=device,
        items_per_sample=[2],
        view_axis=[0, 1],
        padded_gen_tokens=padded,
        attention_scope="same_view",
        caption_access=["camera", "all_captions" if lidar_attends_captions else "no_captions"],
    )


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@pytest.mark.parametrize("lidar_attends_captions", [True, False])
@torch.no_grad()
def test_multiview_maskless_attention_honours_lidar_attends_captions(lidar_attends_captions: bool) -> None:
    """The folds read the flag: with it off a sweep's rows come out having read no caption.

    Both values go through one reference, which differs only in whether the range clip's rows
    carry the caption keys. That is the whole of what the flag means, and running the same batch
    both ways is what shows the difference is the flag rather than the geometry.
    """
    device = torch.device("cuda")
    batch = _multiview_maskless_batch(
        und_len=5,
        # The pack carries stream lengths, not items: the joint geometry is the plan's.
        token_shape=(_JOINT_GEN_TOKENS, 1, 1),
        num_views=1,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        device=device,
        seed=0,
    )
    plan = _joint_cam_lidar_plan(
        device,
        _padded_gen_tokens(batch.packs[0]),
        lidar_attends_captions=lidar_attends_captions,
    )

    out_pack = multiview_attention(*batch.packs, maskless_plan=plan)
    gen_out = get_gen_seq(out_pack)  # [N_full,heads*head_dim]

    # One caption for the sample, so a row reads all of it or none of it.
    reads_row = torch.tensor(
        [True] * _JOINT_CAMERA_TOKENS + [lidar_attends_captions] * _JOINT_LIDAR_TOKENS, device=device
    )  # [N_gen]
    expected = _same_view_caption_reference(
        batch,
        torch.tensor(_JOINT_GROUP_ID, device=device),
        reads_row[:, None].expand(-1, batch.und_k.shape[0]),  # [N_gen,N_und]
    ).flatten(-2, -1)  # [N_gen,heads*head_dim]

    torch.testing.assert_close(gen_out[:_JOINT_GEN_TOKENS].double(), expected, atol=1e-2, rtol=1e-2)
    assert torch.equal(gen_out[_JOINT_GEN_TOKENS:], torch.zeros_like(gen_out[_JOINT_GEN_TOKENS:]))


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@torch.no_grad()
def test_lidar_attends_captions_is_the_only_difference_between_the_two_plans() -> None:
    """The flag changes the LiDAR rows and leaves the camera rows untouched.

    Asserted against the other run rather than against a reference: the camera half of the
    stream is what a reader has to be sure the flag does not quietly reach.
    """
    device = torch.device("cuda")
    batch = _multiview_maskless_batch(
        und_len=5,
        token_shape=(_JOINT_GEN_TOKENS, 1, 1),
        num_views=1,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        device=device,
        seed=0,
    )
    padded = _padded_gen_tokens(batch.packs[0])
    reading = get_gen_seq(
        multiview_attention(
            *batch.packs, maskless_plan=_joint_cam_lidar_plan(device, padded, lidar_attends_captions=True)
        )
    )
    text_free = get_gen_seq(
        multiview_attention(
            *batch.packs, maskless_plan=_joint_cam_lidar_plan(device, padded, lidar_attends_captions=False)
        )
    )

    (
        torch.testing.assert_close(
            reading[:_JOINT_CAMERA_TOKENS], text_free[:_JOINT_CAMERA_TOKENS], atol=0.0, rtol=0.0
        ),
        "the cameras keep their captions either way",
    )
    assert not torch.allclose(
        reading[_JOINT_CAMERA_TOKENS:_JOINT_GEN_TOKENS].double(),
        text_free[_JOINT_CAMERA_TOKENS:_JOINT_GEN_TOKENS].double(),
        atol=1e-3,
        rtol=1e-3,
    ), "the sweep's rows have to change when its captions are taken away"


# The per-view caption layout over the same joint sample: camera view 0 is described by the
# first two UND tokens and view 1 by the next three, covering the whole 5-token UND stream.
_JOINT_PER_VIEW_CAPTIONS = [[(0, 2), (1, 3)]]


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@pytest.mark.parametrize("lidar_attends_captions", [True, False])
@torch.no_grad()
def test_multiview_maskless_attention_honours_the_flag_under_per_view_captions(lidar_attends_captions: bool) -> None:
    """The other caption layout: a camera reads its own view's caption, the sweep all or none.

    This is the branch that keys each same-view group against its own run of captions, rather
    than the sample-level subset, so it is a different code path reaching the same rule.
    """
    device = torch.device("cuda")
    batch = _multiview_maskless_batch(
        und_len=5,
        token_shape=(_JOINT_GEN_TOKENS, 1, 1),
        num_views=1,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        device=device,
        seed=0,
    )
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [2, 1],
        [_JOINT_CAMERA_ITEM, _JOINT_LIDAR_ITEM],
        device=device,
        items_per_sample=[2],
        view_axis=[0, 1],
        padded_gen_tokens=_padded_gen_tokens(batch.packs[0]),
        attention_scope="same_view",
        captions=_JOINT_PER_VIEW_CAPTIONS,
        caption_access=["camera", "all_captions" if lidar_attends_captions else "no_captions"],
    )
    # The per-view layout keys per group, so it never builds the sample-level subset.
    assert plan.caption_gather is not None
    assert plan.gen_to_und_gather is None

    out_pack = multiview_attention(*batch.packs, maskless_plan=plan)
    gen_out = get_gen_seq(out_pack)  # [N_full,heads*head_dim]

    # Camera view 0 reads UND tokens 0-1, view 1 reads 2-4, and the sweep reads all five or none.
    reads = torch.zeros(_JOINT_GEN_TOKENS, 5, dtype=torch.bool, device=device)  # [N_gen,N_und]
    reads[0:4, 0:2] = True
    reads[4:8, 2:5] = True
    reads[8:, :] = lidar_attends_captions
    expected = _same_view_caption_reference(
        batch,
        torch.tensor(_JOINT_GROUP_ID, device=device),
        reads,
    ).flatten(-2, -1)  # [N_gen,heads*head_dim]

    torch.testing.assert_close(gen_out[:_JOINT_GEN_TOKENS].double(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.L0
def test_caption_partition_gives_a_text_free_sweep_an_empty_run() -> None:
    """Under per-view captions the flag shows up as the LiDAR group reading no caption at all."""
    plans = {
        flag: multiview_maskless_attention.build_multiview_maskless_plan(
            [2, 1],
            [_JOINT_CAMERA_ITEM, _JOINT_LIDAR_ITEM],
            device=torch.device("cpu"),
            items_per_sample=[2],
            view_axis=[0, 1],
            padded_gen_tokens=_JOINT_GEN_TOKENS,
            attention_scope="same_view",
            captions=_JOINT_PER_VIEW_CAPTIONS,
            caption_access=["camera", "all_captions" if flag else "no_captions"],
        )
        for flag in (True, False)
    }
    # Groups in order: camera view 0, camera view 1, the sweep. The first two are untouched.
    reading, text_free = plans[True], plans[False]
    # Only groups that read a caption appear: with the sweep reading all of them it is the third
    # run, and with it reading none it is absent from both sides rather than present and empty.
    assert torch.equal(reading.caption_offsets, torch.tensor([0, 2, 5, 10], dtype=torch.int32))
    assert torch.equal(text_free.caption_offsets, torch.tensor([0, 2, 5], dtype=torch.int32))
    assert torch.equal(reading.caption_gather, torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))
    assert torch.equal(text_free.caption_gather, torch.tensor([0, 1, 2, 3, 4]))
    assert reading.caption_max_len == 5 and text_free.caption_max_len == 3
    # The query side loses the sweep's rows too, which is what keeps any group from being
    # handed to the kernel with no keys at all.
    assert reading.caption_q_gather.numel() == _JOINT_GEN_TOKENS
    assert text_free.caption_q_gather.numel() == _JOINT_CAMERA_TOKENS


@pytest.mark.L0
def test_caption_partition_refuses_a_camera_view_no_caption_describes() -> None:
    """A camera reads the caption written for its view or the sample-level one -- never neither.

    Dropping caption-less groups from the pass makes the failure silent otherwise: a camera whose
    caption the pack never recorded would simply leave the gen->und pass and train with no text
    conditioning. The mask refuses the same layout in ``_build_und_view_ids``.
    """
    with pytest.raises(ValueError, match="view 2 reads no caption at all"):
        multiview_maskless_attention.build_multiview_maskless_plan(
            [3],
            [(3, 1, 1)],
            device=torch.device("cpu"),
            items_per_sample=[1],
            view_axis=[0],
            attention_scope="same_view",
            # Three camera views, captions for two of them.
            captions=[[(0, 2), (1, 2)]],
            caption_access=["camera"],
        )


@pytest.mark.L0
def test_multiview_maskless_plan_keeps_the_whole_stream_when_lidar_attends_captions() -> None:
    """The default builds no subset at all, so the gen->und pass stays keyed per sample."""
    plan = _joint_cam_lidar_plan(torch.device("cpu"), _JOINT_GEN_TOKENS, lidar_attends_captions=True)

    assert plan.gen_to_und_gather is None


@pytest.mark.L0
def test_multiview_maskless_plan_drops_lidar_from_the_sample_level_gen_to_und_pass() -> None:
    """With the flag off the pass runs over the camera tokens alone, keyed per sample."""
    plan = _joint_cam_lidar_plan(torch.device("cpu"), _JOINT_GEN_TOKENS, lidar_attends_captions=False)

    assert plan.gen_to_und_gather is not None
    # No padding here (the plan is built at the real length), so the subset is the camera
    # tokens exactly -- the sweep's six are what it leaves out.
    assert torch.equal(plan.gen_to_und_gather, torch.arange(_JOINT_CAMERA_TOKENS))
    assert plan.gen_to_und_max_len == _JOINT_CAMERA_TOKENS


@pytest.mark.L0
def test_multiview_maskless_plan_rejects_a_scope_the_folds_do_not_express() -> None:
    """``"all_views"`` is one pass per sample, not a partition of one, so it is not built here."""
    with pytest.raises(ValueError, match="attention_scope='all_views' is not one this fold expresses"):
        multiview_maskless_attention.build_multiview_maskless_plan(
            [2], [(4, 1, 1)], device=torch.device("cpu"), attention_scope="all_views"
        )


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [pytest.param(4, 4, id="mha"), pytest.param(4, 2, id="gqa")],
)
def test_multiview_maskless_attention_gradients_match_a_dense_reference(num_q_heads: int, num_kv_heads: int) -> None:
    """The merged backward is the reference's backward, which is what the bridges buy.

    ``merge_attentions`` fixes each branch's backward by writing the merged output and LSE into
    the storage the branch's kernel saved, found by data pointer. Every autograd node between
    kernel and merge defeats that -- a copying permute, and a storage-sharing reshape just as
    much, since what the merge patches is the node's output and not the kernel's saved pair.
    Both fold-backs therefore run through ``MergeAttentionsBridge``.

    Without them the forward is unchanged and only the gradients move, so a forward-only test
    cannot see it: measured, the two sensor branches land ~100% off this reference while
    gen->und stays right, because that branch is the one with no reshape. The tolerance is the
    forward test's, for the same bf16 reason.
    """
    device = torch.device("cuda")
    num_views, frames_per_view, patch_h, patch_w = 3, 2, 2, 3
    token_shape = (num_views * frames_per_view, patch_h, patch_w)
    und_len = 5
    batch = _multiview_maskless_batch(
        und_len=und_len,
        token_shape=token_shape,
        num_views=num_views,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        device=device,
        seed=0,
    )
    num_gen_tokens = batch.gen_q.shape[0]
    torch.manual_seed(1)
    seed_grad = torch.randn(num_gen_tokens, num_q_heads * 64, device=device, dtype=torch.bfloat16)

    # The pack's own streams are the leaves, so the gradient measured is the one the decoder
    # layer would hand back to the projections that produced them.
    leaves: dict[str, torch.Tensor] = {}
    for name, pack in zip("qkv", batch.packs):
        for key in ("causal_seq", "full_only_seq"):
            pack[key].requires_grad_(True)
            leaves[f"{name}.{key}"] = pack[key]

    out_pack = multiview_attention(
        *batch.packs, maskless_plan=_plan(num_views, token_shape, device, _padded_gen_tokens(batch.packs[0]))
    )
    get_gen_seq(out_pack)[:num_gen_tokens].backward(seed_grad)

    reference_leaves = {name: leaf.detach().clone().requires_grad_(True) for name, leaf in leaves.items()}
    reference = _multiview_maskless_reference(
        _MultiviewMasklessBatch(
            packs=batch.packs,
            gen_q=reference_leaves["q.full_only_seq"][:num_gen_tokens],
            gen_k=reference_leaves["k.full_only_seq"][:num_gen_tokens],
            gen_v=reference_leaves["v.full_only_seq"][:num_gen_tokens],
            und_k=reference_leaves["k.causal_seq"][:und_len],
            und_v=reference_leaves["v.causal_seq"][:und_len],
        ),
        num_views=num_views,
        frames_per_view=frames_per_view,
        spatial_tokens=patch_h * patch_w,
    )
    reference.flatten(-2, -1).backward(seed_grad.double())

    # The query's causal stream takes no gradient: only the GEN rows were seeded, and the
    # reasoner's own self-attention output is not among them.
    assert leaves["q.causal_seq"].grad is None
    for name in ("q.full_only_seq", "k.full_only_seq", "v.full_only_seq", "k.causal_seq", "v.causal_seq"):
        actual, expected = leaves[name].grad, reference_leaves[name].grad
        assert actual is not None and expected is not None
        # Both cast up: the reference computes in float64 but its gradient lands back on a
        # bf16 leaf, so `expected` is bf16 too and the comparison is bf16-vs-bf16 either way.
        torch.testing.assert_close(
            actual.double(), expected.double(), atol=1e-2, rtol=1e-2, msg=lambda m, n=name: f"{n}: {m}"
        )


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.parametrize(
    "save_ops_regex",
    [
        pytest.param(["fmha"], id="save_fmha"),
        pytest.param([], id="recompute_all"),
    ],
)
def test_multiview_maskless_attention_gradients_survive_activation_checkpointing(
    save_ops_regex: list[str],
) -> None:
    """The merge's backward still lands on the tensors the kernels read, under selective AC.

    ``merge_attentions`` repairs each branch by overwriting the storage that branch's kernel
    saved, and ``MergeAttentionsBridge`` restores that link across the folds' reshapes. Both
    rest on the patch reaching the tensor the kernel will read -- and activation checkpointing
    decides separately whether that tensor was stashed during the forward or is regenerated
    during the backward. Nothing makes PyTorch's checkpointing respect an in-place patch of a
    saved tensor, and when this contract broke before it moved gradients ~70% off with no error
    and nothing odd in the loss, so it is checked rather than reasoned about.

    Both policies are covered because both are live. ``["fmha"]`` is what the AV multiview
    recipes configure (``config_factory`` sets ``mode="selective"`` and that regex is the
    default), and it makes all four attention calls MUST_SAVE -- the three sensor passes plus
    the causal one -- against the mask's one, which is where this path's extra memory goes. The
    empty list recomputes them instead, the change that would recover it.

    The reference is the same module compiled *without* checkpointing, so the only variable is
    the checkpointing itself and the comparison can be exact. It is compiled because that is
    how a decoder layer runs: eager + non-reentrant checkpointing raises ``CheckpointError``
    outright, since NATTEN's merge backward reads ``ctx.saved_tensors`` three times
    (``natten/attn_merge.py:175-177``) and the saved-tensor hooks allow one unpack.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
    )
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    device = torch.device("cuda")
    num_views, frames_per_view, patch_h, patch_w = 3, 2, 2, 3
    token_shape = (num_views * frames_per_view, patch_h, patch_w)
    patterns = [re.compile(pattern) for pattern in save_ops_regex]

    def _policy(ctx, func, *args, **kwargs):
        name = getattr(func, "__name__", str(func))
        if any(pattern.search(name) for pattern in patterns):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_RECOMPUTE

    def _grads(checkpointed: bool) -> dict[str, torch.Tensor]:
        batch = _multiview_maskless_batch(
            und_len=5,
            token_shape=token_shape,
            num_views=num_views,
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=64,
            device=device,
            seed=0,
        )
        leaves: dict[str, torch.Tensor] = {}
        for name, pack in zip("qkv", batch.packs):
            for key in ("causal_seq", "full_only_seq"):
                pack[key].requires_grad_(True)
                leaves[f"{name}.{key}"] = pack[key]
        plan = _plan(num_views, token_shape, device, _padded_gen_tokens(batch.packs[0]))
        num_gen_tokens = batch.gen_q.shape[0]

        class _Layer(torch.nn.Module):
            def forward(self) -> torch.Tensor:
                out = multiview_attention(*batch.packs, maskless_plan=plan)
                return get_gen_seq(out)[:num_gen_tokens]

        layer: torch.nn.Module = _Layer()
        if checkpointed:
            # The wrapper parallelize_unified_mot puts around a decoder layer.
            layer = ptd_checkpoint_wrapper(
                layer,
                context_fn=lambda: create_selective_checkpoint_contexts(_policy),
                preserve_rng_state=False,
            )
        torch.manual_seed(1)
        seed_grad = torch.randn(num_gen_tokens, 8 * 64, device=device, dtype=torch.bfloat16)
        torch.compile(layer)().backward(seed_grad)
        return {name: leaf.grad for name, leaf in leaves.items() if leaf.grad is not None}

    expected = _grads(checkpointed=False)
    actual = _grads(checkpointed=True)

    assert expected, "no leaf took a gradient, so this would pass vacuously"
    assert set(expected) == set(actual), "checkpointing changed which leaves take a gradient"
    for name, want in expected.items():
        # Exact: recomputation reruns the same kernels on the same inputs, so any difference is
        # the patch having missed its tensor rather than arithmetic.
        torch.testing.assert_close(
            actual[name].double(), want.double(), atol=0, rtol=0, msg=lambda m, n=name: f"{n}: {m}"
        )


@pytest.mark.L0
@pytest.mark.CPU
@torch.no_grad()
def test_multiview_maskless_attention_rejects_a_geometry_the_pack_contradicts() -> None:
    """A token count or view count the pack does not carry is refused, not folded into.

    The folds are pure reshapes, so a wrong geometry does not fail on its own: it would carve
    the same tokens into different cells and attend across the wrong ones. Both halves of the
    geometry are therefore checked against the pack.
    """
    device = torch.device("cpu")
    token_shape = (4, 2, 2)
    batch = _multiview_maskless_batch(
        und_len=5,
        token_shape=token_shape,
        num_views=2,
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=64,
        device=device,
        seed=0,
    )
    with pytest.raises(ValueError, match="not divisible by num_views"):
        _plan(3, token_shape, device)
    with pytest.raises(ValueError, match="but the pack holds"):
        multiview_attention(*batch.packs, maskless_plan=_plan(2, (4, 2, 3), device))


@pytest.mark.L0
@pytest.mark.CPU
@torch.no_grad()
def test_multiview_maskless_attention_rejects_a_plan_the_pack_contradicts() -> None:
    """A plan describing a different batch than the pack holds is refused, not folded into."""
    device = torch.device("cpu")
    shape = _MultiviewShape(und_lens=(5, 7), token_shapes=((4, 2, 2), (4, 2, 2)), num_views=(2, 2))
    x = torch.randn(shape.real_len, 4, 64, device=device, dtype=torch.bfloat16)
    backend = resolve_flex_backend(device, "flex_triton")
    packs = tuple(_multiview_pack(x, shape, backend) for _ in range(3))

    one_sample = multiview_maskless_attention.build_multiview_maskless_plan([2], [(4, 2, 2)], device=device)
    with pytest.raises(ValueError, match="samples but the pack holds"):
        multiview_attention(*packs, maskless_plan=one_sample)

    wrong_tokens = multiview_maskless_attention.build_multiview_maskless_plan(
        [2, 2], [(4, 2, 2), (4, 2, 3)], device=device
    )
    with pytest.raises(ValueError, match="but the pack holds"):
        multiview_attention(*packs, maskless_plan=wrong_tokens)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [pytest.param(4, 4, id="mha"), pytest.param(4, 2, id="gqa")],
)
@torch.no_grad()
def test_multiview_maskless_attention_folds_a_ragged_batch(num_q_heads: int, num_kv_heads: int) -> None:
    """Samples differing in views, frames and resolution each attend within themselves.

    The ragged path cannot use the batch axis -- the groups are different lengths -- so it
    addresses them with varlen ranges, and reaches the cross-view groups through a gather. Both
    are pure index math, which is exactly the kind of thing that fails silently: a gather off by
    one sample's base, or offsets that ran the partitions together, still produces a plausible
    tensor. The reference is each sample's own dense answer, computed independently and
    concatenated, so any leakage across the sample boundary shows up immediately.
    """
    device = torch.device("cuda")
    # (num_views, token_shape), deliberately unequal. The third is the single-view shape every
    # LiDAR sample takes: its same-view group is the whole sample, so its cross-view pass adds
    # nothing but the own-frame double-count -- which the reference models, and which is the
    # distortion the V=1 neutralisation is meant to remove.
    samples = ((2, (4, 2, 2)), (3, (6, 2, 3)), (1, (3, 2, 2)))
    und_lens = (5, 7, 4)
    shape = _MultiviewShape(
        und_lens=und_lens,
        token_shapes=tuple(token_shape for _, token_shape in samples),
        num_views=tuple(views for views, _ in samples),
    )
    torch.manual_seed(0)
    qkv = [
        torch.randn(shape.real_len, heads, 64, device=device, dtype=torch.bfloat16)
        for heads in (num_q_heads, num_kv_heads, num_kv_heads)
    ]
    backend = resolve_flex_backend(device, "flex_triton")
    packs = cast(
        tuple[SequencePack, SequencePack, SequencePack],
        tuple(_multiview_pack(tensor, shape, backend) for tensor in qkv),
    )
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        shape.num_views, shape.token_shapes, device=device, padded_gen_tokens=_padded_gen_tokens(packs[0])
    )

    out_pack = multiview_attention(*packs, maskless_plan=plan)
    gen_out = get_gen_seq(out_pack)  # [N_full,heads*head_dim]

    # The pack lays each sample down as its UND run then its GEN run; the GEN stream the
    # attention path sees is those GEN runs concatenated, which is the order to rebuild here.
    und_starts, gen_starts, cursor = [], [], 0
    for und_len, gen_len in zip(shape.und_lens, shape.gen_lens):
        und_starts.append(cursor)
        gen_starts.append(cursor + und_len)
        cursor += und_len + gen_len

    gen_cursor = 0
    for index, ((views, (latent_t, patch_h, patch_w)), und_len) in enumerate(zip(samples, und_lens)):
        gen_len = shape.gen_lens[index]
        sample = _MultiviewMasklessBatch(
            packs=packs,
            gen_q=qkv[0][gen_starts[index] : gen_starts[index] + gen_len],
            gen_k=qkv[1][gen_starts[index] : gen_starts[index] + gen_len],
            gen_v=qkv[2][gen_starts[index] : gen_starts[index] + gen_len],
            und_k=qkv[1][und_starts[index] : und_starts[index] + und_len],
            und_v=qkv[2][und_starts[index] : und_starts[index] + und_len],
        )
        expected = _multiview_maskless_reference(
            sample,
            num_views=views,
            frames_per_view=latent_t // views,
            spatial_tokens=patch_h * patch_w,
            # A single-view sample owns one view group, so its cross-instant pass is skipped.
            cross_view=views > 1,
        ).flatten(-2, -1)  # [N_gen_i,heads*head_dim]
        actual = gen_out[gen_cursor : gen_cursor + gen_len]
        torch.testing.assert_close(actual.double(), expected, atol=1e-2, rtol=1e-2)
        gen_cursor += gen_len

    assert torch.equal(gen_out[gen_cursor:], torch.zeros_like(gen_out[gen_cursor:]))


@pytest.mark.L0
@pytest.mark.GPU
@torch.no_grad()
def test_multiview_maskless_attention_compiles_with_the_pack_as_an_input() -> None:
    """Compile it the way a decoder layer does: packs arriving as arguments, not as a closure.

    A closure over the packs lets Dynamo specialise their contents, so a host-side read of a
    value *inside* a pack tensor is folded to a constant and nothing complains. As a graph
    input the same read is an unbacked symbol, and comparing one against a Python int is a
    data-dependent guard Dynamo refuses outright:

        Could not guard on data-dependent expression Ne(u0, 549120)

    which is how it reached a 16-node run rather than a test. The counts this needs are host
    side in the pack (``get_num_real_tokens``); this pins that they are read from there.

    A ragged batch with padding, because that is what makes the counts differ from the stream
    lengths at all.
    """
    device = torch.device("cuda")
    samples = ((2, (4, 2, 2)), (3, (6, 2, 3)))
    shape = _MultiviewShape(
        und_lens=(5, 7),
        token_shapes=tuple(token_shape for _, token_shape in samples),
        num_views=tuple(views for views, _ in samples),
    )
    torch.manual_seed(0)
    qkv = [torch.randn(shape.real_len, heads, 64, device=device, dtype=torch.bfloat16) for heads in (8, 4, 4)]
    backend = resolve_flex_backend(device, "flex_triton")
    packs = cast(
        tuple[SequencePack, SequencePack, SequencePack],
        tuple(_multiview_pack(tensor, shape, backend) for tensor in qkv),
    )
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        shape.num_views, shape.token_shapes, device=device, padded_gen_tokens=_padded_gen_tokens(packs[0])
    )

    expected = get_gen_seq(multiview_attention(*packs, maskless_plan=plan))
    compiled = torch.compile(multiview_attention)
    actual = get_gen_seq(compiled(*packs, maskless_plan=plan))

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@torch.no_grad()
def test_multiview_maskless_attention_joins_a_camera_and_a_lidar_item_by_capture_time() -> None:
    """A joint sample attends across sensors by quantised capture time, not by frame index.

    Camera at 7.5Hz latent against sweeps at 10Hz: the rates do not share an index, so each
    token's frame *midpoint* is quantised onto the camera's grid and the cross-instant partition
    groups by that. The reference recomputes those group ids from the two rates directly rather
    than reading them off the plan, so a plan that quantised differently -- by frame start, or
    onto the wrong anchor -- fails here even though its shapes stay valid.

    Note what the multiplicity captures: two sweeps landing on one camera instant are in both
    partitions at once (same range item, same instant), so they double-weight each other exactly
    as a camera token double-weights its own (view, frame) cell.
    """
    device = torch.device("cuda")
    num_q_heads, num_kv_heads, head_dim = 4, 2, 64
    cam_views, cam_frames, lidar_frames, spatial = 2, 3, 4, 1
    cam_rate, lidar_rate = 1.0 / 7.5, 1.0 / 10.0
    cam_tokens, lidar_tokens = cam_views * cam_frames * spatial, lidar_frames * spatial
    gen_len, und_len = cam_tokens + lidar_tokens, 5

    # One sample, one causal split then one full split holding both items back to back -- the
    # layout the packer produces for a joint sample. The token_shape here only has to multiply
    # out to the sample's GEN length; the real per-item geometry goes to the plan.
    shape = _MultiviewShape(und_lens=(und_len,), token_shapes=((gen_len, 1, 1),), num_views=(1,))
    torch.manual_seed(0)
    qkv = [
        torch.randn(shape.real_len, heads, head_dim, device=device, dtype=torch.bfloat16)
        for heads in (num_q_heads, num_kv_heads, num_kv_heads)
    ]
    backend = resolve_flex_backend(device, "flex_triton")
    packs = cast(
        tuple[SequencePack, SequencePack, SequencePack],
        tuple(_multiview_pack(tensor, shape, backend) for tensor in qkv),
    )
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [cam_views, 1],
        [(cam_views * cam_frames, 1, spatial), (lidar_frames, 1, spatial)],
        device=device,
        padded_gen_tokens=_padded_gen_tokens(packs[0]),
        seconds_per_frame=[cam_rate, lidar_rate],
        items_per_sample=[2],
        view_axis=[0, 1],  # a camera's view 0 and the sweep's only view are not the same view
    )
    assert plan.num_gen_tokens == gen_len

    out_pack = multiview_attention(*packs, maskless_plan=plan)
    gen_out = get_gen_seq(out_pack)[:gen_len]  # [N_gen,heads*head_dim]

    # ── the reference: group ids rebuilt from the two rates ────────────────────
    same_view, instant = [], []
    for view in range(cam_views):  # camera item: view-outer, frame-inner, spatial-innermost
        for frame in range(cam_frames):
            for _ in range(spatial):
                same_view.append(("cam", view))
                instant.append(math.floor((frame + 0.5) * cam_rate / cam_rate))
    for sweep in range(lidar_frames):  # range item: one view over its own grid
        for _ in range(spatial):
            same_view.append(("lidar", 0))
            instant.append(math.floor((sweep + 0.5) * lidar_rate / cam_rate))
    assert instant == [0, 1, 2, 0, 1, 2, 0, 1, 1, 2], "7.5Hz vs 10Hz puts two sweeps on instant 1."

    group = 1.0 / head_dim**0.5
    gen_q = qkv[0][und_len : und_len + gen_len].double()  # [N_gen,heads,head_dim]
    repeat = num_q_heads // num_kv_heads
    gen_k = qkv[1][und_len : und_len + gen_len].double().repeat_interleave(repeat, dim=1)
    gen_v = qkv[2][und_len : und_len + gen_len].double().repeat_interleave(repeat, dim=1)
    und_k = qkv[1][:und_len].double().repeat_interleave(repeat, dim=1)
    und_v = qkv[2][:und_len].double().repeat_interleave(repeat, dim=1)

    multiplicity = torch.tensor(
        [
            [float(same_view[i] == same_view[j]) + float(instant[i] == instant[j]) for j in range(gen_len)]
            for i in range(gen_len)
        ],
        device=device,
        dtype=torch.float64,
    )  # [N_gen,N_gen]
    gen_scores = torch.einsum("ihd,jhd->hij", gen_q, gen_k) * group
    und_scores = torch.einsum("ihd,jhd->hij", gen_q, und_k) * group
    peak = torch.maximum(gen_scores.max(dim=-1).values, und_scores.max(dim=-1).values)
    gen_weights = multiplicity[None] * torch.exp(gen_scores - peak[..., None])
    und_weights = torch.exp(und_scores - peak[..., None])
    numerator = torch.einsum("hij,jhd->ihd", gen_weights, gen_v) + torch.einsum("hij,jhd->ihd", und_weights, und_v)
    expected = numerator / (gen_weights.sum(-1) + und_weights.sum(-1)).transpose(0, 1)[..., None]

    torch.testing.assert_close(gen_out.double(), expected.flatten(-2, -1), atol=1e-2, rtol=1e-2)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@torch.no_grad()
def test_multiview_maskless_attention_takes_wsm_and_hdmap_control_items() -> None:
    """A joint transfer sample: WSM control + RGB target, HD-map control + LiDAR target.

    Every control rule in the mask is a *view* rule and never an instant one, so a control token
    joins its target's view groups and is absent from the cross-instant partition entirely --
    sliced out as a key and as a query alike, which is what keeps a sensor query from reaching a
    control key through the instant rule. Under ``control_attends_sensor`` the four view rules
    (sensor->sensor, sensor->control, control->control, control->sensor) fill in the whole
    square, so one unmasked pass over ``(sample, axis, view)`` expresses all of them.

    The reference states that as a multiplicity: one for sharing a view, one more for sharing an
    instant, the latter only between two sensor tokens.
    """
    device = torch.device("cuda")
    num_q_heads, num_kv_heads, head_dim = 4, 2, 64
    cam_rate, lidar_rate = 1.0 / 7.5, 1.0 / 10.0
    # WSM control and RGB target on 2 camera views x 2 frames; HD-map control and LiDAR target
    # on the range axis' single view x 3 sweeps. One spatial token each, to keep the reference
    # readable -- the folds never look at the spatial axis except to size a cell.
    items = [
        dict(views=2, latent_t=4, rate=cam_rate, control=True, axis=0),  # WSM
        dict(views=2, latent_t=4, rate=cam_rate, control=False, axis=0),  # RGB target
        dict(views=1, latent_t=3, rate=lidar_rate, control=True, axis=1),  # HD-map rangemap
        dict(views=1, latent_t=3, rate=lidar_rate, control=False, axis=1),  # LiDAR target
    ]
    gen_len, und_len = sum(int(i["latent_t"]) for i in items), 5

    shape = _MultiviewShape(und_lens=(und_len,), token_shapes=((gen_len, 1, 1),), num_views=(1,))
    torch.manual_seed(0)
    qkv = [
        torch.randn(shape.real_len, heads, head_dim, device=device, dtype=torch.bfloat16)
        for heads in (num_q_heads, num_kv_heads, num_kv_heads)
    ]
    backend = resolve_flex_backend(device, "flex_triton")
    packs = cast(
        tuple[SequencePack, SequencePack, SequencePack],
        tuple(_multiview_pack(tensor, shape, backend) for tensor in qkv),
    )
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [int(i["views"]) for i in items],
        [(int(i["latent_t"]), 1, 1) for i in items],
        device=device,
        seconds_per_frame=[float(i["rate"]) for i in items],
        items_per_sample=[len(items)],
        is_control=[bool(i["control"]) for i in items],
        view_axis=[int(i["axis"]) for i in items],
        padded_gen_tokens=_padded_gen_tokens(packs[0]),
    )
    assert plan.same_view_gather is not None, "A control item splits a view into two runs."

    out_pack = multiview_attention(*packs, maskless_plan=plan)
    gen_out = get_gen_seq(out_pack)[:gen_len]  # [N_gen,heads*head_dim]

    # ── the reference: per-token view id, sensor flag and instant, rebuilt from the geometry ──
    view_of, sensor_of, instant_of = [], [], []
    anchor_rate = float(items[0]["rate"])
    for spec in items:
        views, frames = int(spec["views"]), int(spec["latent_t"]) // int(spec["views"])
        for view in range(views):  # view-outer, frame-inner
            for frame in range(frames):
                view_of.append((int(spec["axis"]), view))
                sensor_of.append(not bool(spec["control"]))
                instant_of.append(math.floor((frame + 0.5) * float(spec["rate"]) / anchor_rate + 1e-6))
    assert sensor_of.count(True) == 4 + 3, "Only the two target items are sensors."

    multiplicity = torch.tensor(
        [
            [
                float(view_of[i] == view_of[j])
                + float(sensor_of[i] and sensor_of[j] and instant_of[i] == instant_of[j])
                for j in range(gen_len)
            ]
            for i in range(gen_len)
        ],
        device=device,
        dtype=torch.float64,
    )  # [N_gen,N_gen]

    scale = 1.0 / head_dim**0.5
    repeat = num_q_heads // num_kv_heads
    gen_q = qkv[0][und_len : und_len + gen_len].double()
    gen_k = qkv[1][und_len : und_len + gen_len].double().repeat_interleave(repeat, dim=1)
    gen_v = qkv[2][und_len : und_len + gen_len].double().repeat_interleave(repeat, dim=1)
    und_k = qkv[1][:und_len].double().repeat_interleave(repeat, dim=1)
    und_v = qkv[2][:und_len].double().repeat_interleave(repeat, dim=1)
    gen_scores = torch.einsum("ihd,jhd->hij", gen_q, gen_k) * scale
    und_scores = torch.einsum("ihd,jhd->hij", gen_q, und_k) * scale
    peak = torch.maximum(gen_scores.max(dim=-1).values, und_scores.max(dim=-1).values)
    gen_weights = multiplicity[None] * torch.exp(gen_scores - peak[..., None])
    und_weights = torch.exp(und_scores - peak[..., None])
    numerator = torch.einsum("hij,jhd->ihd", gen_weights, gen_v) + torch.einsum("hij,jhd->ihd", und_weights, und_v)
    expected = numerator / (gen_weights.sum(-1) + und_weights.sum(-1)).transpose(0, 1)[..., None]

    torch.testing.assert_close(gen_out.double(), expected.flatten(-2, -1), atol=1e-2, rtol=1e-2)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@torch.no_grad()
def test_multiview_maskless_attention_reads_one_caption_per_view() -> None:
    """Each camera view attends the caption written for it and no other.

    The gen->und pass borrows the same-view partition for its queries, so its key set is per
    view rather than per sample. That is what per-view captions need and what a single pass over
    the whole causal run cannot give: without it a view would read its neighbour's caption,
    which produces a perfectly plausible tensor and no error at all.

    The reference states the rule as a mask over caption keys -- a GEN token weights the
    captions of its own view at one and every other at zero -- beside the usual GEN
    multiplicity. A pass keying every token against the whole run would put ones everywhere in
    that block and land far outside the tolerance.
    """
    device = torch.device("cuda")
    num_q_heads, num_kv_heads, head_dim = 4, 2, 64
    views, frames, spatial = 2, 2, 1
    caption_lens = [3, 2]  # view 0 gets three caption tokens, view 1 gets two
    gen_len, und_len = views * frames * spatial, sum(caption_lens)

    torch.manual_seed(0)
    qkv = [
        torch.randn(und_len + gen_len, heads, head_dim, device=device, dtype=torch.bfloat16)
        for heads in (num_q_heads, num_kv_heads, num_kv_heads)
    ]

    def _pack(tensor: torch.Tensor) -> SequencePack:
        return build_packed_sequence(
            "two_way",
            packed_sequence=tensor,
            attn_modes=["causal", "full"],
            split_lens=[und_len, gen_len],
            sample_lens=[und_len + gen_len],
            packed_und_token_indexes=cast(torch.LongTensor, torch.arange(und_len, dtype=torch.long, device=device)),
            packed_gen_token_indexes=cast(
                torch.LongTensor, torch.arange(und_len, und_len + gen_len, dtype=torch.long, device=device)
            ),
            num_heads=tensor.shape[-2],
            head_dim=head_dim,
            num_layers=1,
            text_caption_lens=[caption_lens],
        )[0]

    packs = cast(tuple[SequencePack, SequencePack, SequencePack], tuple(_pack(t) for t in qkv))
    assert get_caption_seq_offsets(packs[0]) is not None, "Per-view captions need their boundaries."
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [views],
        [(views * frames, 1, spatial)],
        device=device,
        captions=[[(0, caption_lens[0]), (1, caption_lens[1])]],
        padded_gen_tokens=_padded_gen_tokens(packs[0]),
    )
    assert plan.caption_gather is not None
    # One run per same-view group that reads a caption, and no others: the group the pack's
    # padding forms reads none, so it leaves the pass rather than being keyed against a
    # zero-length run, and the scatter gives those rows a weight the merge ignores.
    assert torch.diff(plan.caption_offsets).tolist() == list(caption_lens)

    out_pack = multiview_attention(*packs, maskless_plan=plan)
    gen_out = get_gen_seq(out_pack)[:gen_len]  # [N_gen,heads*head_dim]

    # ── the reference ─────────────────────────────────────────────────────────
    view_of = [view for view in range(views) for _ in range(frames * spatial)]
    instant_of = [frame for _ in range(views) for frame in range(frames) for _ in range(spatial)]
    caption_view_of = [view for view, length in enumerate(caption_lens) for _ in range(length)]

    scale = 1.0 / head_dim**0.5
    repeat = num_q_heads // num_kv_heads
    gen_q = qkv[0][und_len:].double()
    gen_k = qkv[1][und_len:].double().repeat_interleave(repeat, dim=1)
    gen_v = qkv[2][und_len:].double().repeat_interleave(repeat, dim=1)
    und_k = qkv[1][:und_len].double().repeat_interleave(repeat, dim=1)
    und_v = qkv[2][:und_len].double().repeat_interleave(repeat, dim=1)

    gen_mult = torch.tensor(
        [
            [float(view_of[i] == view_of[j]) + float(instant_of[i] == instant_of[j]) for j in range(gen_len)]
            for i in range(gen_len)
        ],
        device=device,
        dtype=torch.float64,
    )
    und_mult = torch.tensor(
        [[float(view_of[i] == caption_view_of[j]) for j in range(und_len)] for i in range(gen_len)],
        device=device,
        dtype=torch.float64,
    )

    gen_scores = torch.einsum("ihd,jhd->hij", gen_q, gen_k) * scale
    und_scores = torch.einsum("ihd,jhd->hij", gen_q, und_k) * scale
    peak = torch.maximum(gen_scores.max(dim=-1).values, und_scores.max(dim=-1).values)
    gen_weights = gen_mult[None] * torch.exp(gen_scores - peak[..., None])
    und_weights = und_mult[None] * torch.exp(und_scores - peak[..., None])
    numerator = torch.einsum("hij,jhd->ihd", gen_weights, gen_v) + torch.einsum("hij,jhd->ihd", und_weights, und_v)
    expected = numerator / (gen_weights.sum(-1) + und_weights.sum(-1)).transpose(0, 1)[..., None]

    torch.testing.assert_close(gen_out.double(), expected.flatten(-2, -1), atol=1e-2, rtol=1e-2)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@pytest.mark.parametrize("num_views", [pytest.param(1, id="single_view"), pytest.param(3, id="multi_view")])
@torch.no_grad()
def test_multiview_maskless_attention_counts_a_single_view_sample_once(num_views: int) -> None:
    """A single-view sample attends each key once, as the mask does; a multi-view one does not.

    With one view the cross-instant groups sit inside the one view group, so merging them would
    weight the query's own instant twice and buy nothing -- there is no second view to reach.
    That sample is sliced out of the pass entirely, which leaves it attending its item uniformly,
    exactly what ``attention_scope="decomposed"`` degenerates to at one view.

    Parametrised against three views so the assertion is a contrast rather than a claim about
    this implementation: the same code must double-count there, since the two passes then admit
    genuinely different keys and the shared cell is the documented price of that.
    """
    device = torch.device("cuda")
    frames_per_view, patch_h, patch_w = 3, 2, 2
    token_shape = (num_views * frames_per_view, patch_h, patch_w)
    batch = _multiview_maskless_batch(
        und_len=5,
        token_shape=token_shape,
        num_views=num_views,
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=64,
        device=device,
        seed=0,
    )
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [num_views], [token_shape], device=device, padded_gen_tokens=_padded_gen_tokens(batch.packs[0])
    )
    assert plan.cross_view_empty == (num_views == 1)

    out_pack = multiview_attention(*batch.packs, maskless_plan=plan)
    num_gen_tokens = batch.gen_q.shape[0]
    expected = _multiview_maskless_reference(
        batch,
        num_views=num_views,
        frames_per_view=frames_per_view,
        spatial_tokens=patch_h * patch_w,
        cross_view=num_views > 1,
    ).flatten(-2, -1)

    torch.testing.assert_close(get_gen_seq(out_pack)[:num_gen_tokens].double(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 60 * 1024**3,
    reason="The stream this sizes is ~20GB of q/k/v and outputs.",
)
@torch.no_grad()
def test_multiview_maskless_attention_runs_a_gen_stream_past_the_varlen_index_limit() -> None:
    """A GEN stream long enough to overflow a varlen sequence index still runs.

    The gen->und pass keys the whole stream as one range, so its ``max_seqlen_Q`` is that
    stream's length -- and the varlen kernels index a sequence with int32, which overflows once
    ``max_seqlen * heads * head_dim`` reaches 2**31: 524288 queries at 32 heads of 128 wide.
    Past that the kernel takes an illegal address rather than raising, and asynchronously, so it
    surfaces at whatever synchronises next and reads as a fault in an unrelated op.

    One sample never needs the ranges, so the pass takes the dense form and the limit does not
    apply. This pins that: a transfer sample doubles its own GEN stream, and 11 views of 26
    frames at 920 tokens a cell is 526240 -- just past the line, which is how the benchmark
    found it. Sized to the real recipe rather than to the threshold so it stays a description of
    something that runs in production.
    """
    device = torch.device("cuda")
    views, frames, spatial, items = 11, 26, 920, 2
    latent_t = views * frames
    gen_len, und_len = latent_t * spatial * items, 2048
    assert gen_len * 32 * 128 >= 2**31, "This case only means anything past the int32 limit."

    shape = _MultiviewShape(und_lens=(und_len,), token_shapes=((gen_len, 1, 1),), num_views=(1,))
    torch.manual_seed(0)
    qkv = [torch.randn(shape.real_len, heads, 128, device=device, dtype=torch.bfloat16) for heads in (32, 8, 8)]
    backend = resolve_flex_backend(device, "flex_triton")
    packs = cast(
        tuple[SequencePack, SequencePack, SequencePack],
        tuple(_multiview_pack(tensor, shape, backend) for tensor in qkv),
    )
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [views] * items,
        [(latent_t, 23, 40)] * items,
        device=device,
        items_per_sample=[items],
        is_control=[index < items - 1 for index in range(items)],
        view_axis=[0] * items,
    )

    out_pack = multiview_attention(*packs, maskless_plan=plan)
    torch.cuda.synchronize()  # the fault this guards against is asynchronous

    gen_out = get_gen_seq(out_pack)[:gen_len]
    assert torch.isfinite(gen_out.float()).all()


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="The attention kernels require a GPU.")
@pytest.mark.skipif(not NATTEN_SUPPORTED, reason="merge_attentions requires NATTEN.")
@torch.no_grad()
def test_multiview_maskless_plan_gather_is_a_permutation() -> None:
    """The cross-view gather and its inverse undo each other, over a ragged batch.

    Cheap to state and the thing every ragged fold rests on: if these are not inverses, the
    bridge's backward writes the merged output into the wrong rows and nothing raises.
    """
    device = torch.device("cuda")
    plan = multiview_maskless_attention.build_multiview_maskless_plan(
        [2, 3, 1], [(4, 2, 2), (6, 2, 3), (3, 1, 4)], device=device
    )
    gather = plan.cross_view_gather
    assert gather is not None
    # The cross-instant pass covers the samples that own more than one view group, each of their
    # tokens exactly once. The third sample here is single-view, so its instant groups would sit
    # inside its one view group and it is sliced out -- 16 + 36 tokens in, its 12 out.
    covered = 4 * 2 * 2 + 6 * 2 * 3
    assert plan.num_gen_tokens == covered + 3 * 1 * 4
    assert torch.equal(gather.sort().values, torch.arange(covered, device=device))
    # One item per view axis per sample leaves a view's tokens contiguous, so the same-view
    # groups already tile the packed stream and that pass needs no gather at all.
    assert plan.same_view_gather is None
    # Every group boundary lands where the lengths say, and the partitions cover the stream.
    assert int(plan.same_view_offsets[-1]) == plan.num_gen_tokens
    assert int(plan.cross_view_offsets[-1]) == covered


if __name__ == "__main__":
    test_two_way_attention_vs_three_way_attention()
