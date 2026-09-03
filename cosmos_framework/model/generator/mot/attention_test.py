# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import random  # noqa: I001 - release import rewriting changes the package sort order.
import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask

import cosmos_framework.model.generator.mot.attention as attention
from cosmos_framework.model.attention import attention as imaginaire_attention
from cosmos_framework.model.attention import multi_dimensional_attention_varlen
from cosmos_framework.model.attention.natten import NATTEN_SUPPORTED
from cosmos_framework.model.attention.varlen import generate_multi_dim_varlen_parameters
from cosmos_framework.utils.misc import set_torch_compile_options
from cosmos_framework.model.generator.mot.attention import (
    build_packed_sequence,
)
from cosmos_framework.model.generator.mot.flex_attention import (
    FlexBackend,
    MaskItem,
    build_multiview_block_mask,
    resolve_flex_backend,
)
from cosmos_framework.model.generator.mot.flex_attention_test import _FLASH_UNAVAILABLE_MARKERS
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_all_seq,
    get_all_seq_padded,
    get_causal_seq,
    get_full_only_seq,
    get_gen_seq,
    get_und_seq,
    prepare_sequence_pack_metadata,
    sequence_pack_from_packed_sequence,
    set_gen_seq,
    set_und_seq,
    zeros_like,
)

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
    def _make_pack_decomposed(x, impl: str):
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
        return _make_pack_decomposed(x, "two_way")

    def make_pack_three_way(x):
        return _make_pack_decomposed(x, "three_way")

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
        get_all_seq(output1_factored)[:real_len], get_all_seq(output1_joint)[:real_len], atol=atol_self, rtol=rtol_self
    )
    torch.testing.assert_close(
        get_all_seq(output2_factored)[:real_len], get_all_seq(output2_joint)[:real_len], atol=atol_self, rtol=rtol_self
    )

    # impl 1 vs impl 2. needs more tolerance
    torch.testing.assert_close(
        get_all_seq(output2_factored)[:real_len], get_all_seq(output1_factored)[:real_len], atol=atol_cmp, rtol=rtol_cmp
    )
    torch.testing.assert_close(
        get_all_seq(output2_joint)[:real_len], get_all_seq(output1_joint)[:real_len], atol=atol_cmp, rtol=rtol_cmp
    )

    if test_backward:
        get_all_seq(output1_joint)[:real_len].sum().backward()
        get_all_seq(output2_joint)[:real_len].sum().backward()
        get_all_seq(output1_factored)[:real_len].sum().backward()
        get_all_seq(output2_factored)[:real_len].sum().backward()

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
    """Stands in for ``sample_offsets`` and fails whatever reads the sample count off it."""

    @property
    def shape(self) -> tuple[int, ...]:
        raise AssertionError("the sample count was read, which specializes the compiled graph on it")


def _use_varlen_with(sample_offsets: object) -> bool:
    return attention._use_varlen(cast(torch.Tensor, sample_offsets))


@pytest.mark.L0
def test_use_varlen_does_not_read_the_sample_count_while_training() -> None:
    """A pack's sample count varies from step to step, so reading it costs a recompile.

    ``_use_varlen`` only wants the count to take a dense-attention shortcut that is gated to
    inference, and short-circuit evaluation is what keeps training away from it: the grad-mode
    test is the left operand of an ``or``, so training never reaches the count at all.
    """
    assert _use_varlen_with(_SampleCountTripwire()) is True


@pytest.mark.L0
def test_use_varlen_takes_the_dense_path_for_a_single_sample_without_grad() -> None:
    # [0, 4] is one sample, so the varlen ranges would describe the whole tensor and buy nothing.
    with torch.no_grad():
        assert _use_varlen_with(torch.tensor([0, 4], dtype=torch.int32)) is False


@pytest.mark.L0
def test_use_varlen_stays_varlen_for_several_samples_without_grad() -> None:
    # [0, 2, 4] is two samples, which the dense API has no way to keep apart.
    with torch.no_grad():
        assert _use_varlen_with(torch.tensor([0, 2, 4], dtype=torch.int32)) is True


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
    torch.testing.assert_close(get_all_seq(first_pack), get_all_seq(second_pack))


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
        seq_len=full_only_seq.shape[0],
        full_q_offsets=full_q_offsets,
        items_per_sample=[
            # One item per sample, none of it conditioning.
            [
                MaskItem(
                    token_shape=token_shape,
                    condition_mask=torch.zeros(token_shape[0], dtype=torch.bool),
                    num_views=num_views,
                )
            ]
            for token_shape, num_views in zip(shape.token_shapes, shape.num_views)
        ],
        device=full_only_seq.device,
        block_size=block_size,
        num_und=causal_seq.shape[0],
        causal_offsets=causal_offsets,
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
    """``two_way_attention`` on the flex path, reading the mask off the metadata as production does."""
    return attention.two_way_attention(
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
@pytest.mark.parametrize("backend_preference", ["triton", "flash"])
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

            # get_all_seq gathers the two towers back into packed token order, the form the decoder
            # layer passes on. Real tokens only: the dense full branch leaves the padding rows
            # unwritten (its varlen offsets stop at the last real token) where the flex branch
            # writes them from the -1 sentinel, so they are neither comparable nor read downstream.
            dense_out = get_all_seq(dense_pack)[: shape.real_len].float()
            flex_out = get_all_seq(flex_pack)[: shape.real_len].float()
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
    ``get_all_seq`` gather, the key concatenations -- competes for the same size class, so the
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
    # get_all_seq, which holds real tokens only.
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

    ``two_way_attention``'s causal pass switches to ``_causal_seq_offsets_pad_segment`` when the
    pack carries one, so its padding is covered by a trailing segment and the kernel writes it.
    The GEN pass does not: it keys against ``get_all_seq``, which holds real tokens only and
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

    Before ``_sample_offsets_pad_segment`` the causal pass had that pairing and the dense full
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

    assert "_sample_offsets_pad_segment" in pack, (
        "A padded two-way pack needs the interleaved stream's pad segment, or the dense full pass "
        "has nothing to pair its padded GEN queries against."
    )
    q_offsets = pack["_full_only_seq_offsets_pad_segment"]
    kv_offsets = pack["_sample_offsets_pad_segment"]

    # The query ranges reach the end of the padded GEN stream, the key ranges the end of the
    # padded interleaved stream, so no row of either sits outside every range.
    assert int(q_offsets[-1]) == padded_gen, (
        f"The GEN query ranges stop at {int(q_offsets[-1])} but the stream holds {padded_gen} rows."
    )
    assert int(kv_offsets[-1]) == padded_und + padded_gen, (
        f"The key ranges stop at {int(kv_offsets[-1])} but the interleaved stream holds {padded_und + padded_gen} rows."
    )
    padded_all_seq, _, _ = get_all_seq_padded(pack)
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
    assert pack["max_full_len_pad_segment"] >= padded_gen - sum(gen_lens)
    assert pack["max_sample_len_pad_segment"] >= (padded_und - sum(und_lens)) + (padded_gen - sum(gen_lens))


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

    pad_segment_keys = [
        "_causal_seq_offsets_pad_segment",
        "max_causal_len_pad_segment",
        "_full_only_seq_offsets_pad_segment",
        "max_full_len_pad_segment",
        "_sample_offsets_pad_segment",
        "max_sample_len_pad_segment",
    ]
    present = [key for key in pad_segment_keys if key in pack]
    assert present == pad_segment_keys, f"Padded pack carries only {present}."

    # The three offset tensors describe the same number of segments, so every pass pairs its
    # query segments one to one with its key segments, pad segment included.
    assert (
        pack["_causal_seq_offsets_pad_segment"].shape[0]
        == pack["_full_only_seq_offsets_pad_segment"].shape[0]
        == pack["_sample_offsets_pad_segment"].shape[0]
    )
    # And the max lengths are ints, not the tensors they sit beside -- the pairing that a typo
    # here would silently swap, since both are just dict entries.
    for key in ("max_causal_len_pad_segment", "max_full_len_pad_segment", "max_sample_len_pad_segment"):
        assert isinstance(pack[key], int), f"{key} should be an int, got {type(pack[key]).__name__}."
    for key in ("_causal_seq_offsets_pad_segment", "_full_only_seq_offsets_pad_segment", "_sample_offsets_pad_segment"):
        assert isinstance(pack[key], torch.Tensor), f"{key} should be a tensor, got {type(pack[key]).__name__}."


_PAD_SEGMENT_KEYS = (
    "_causal_seq_offsets_pad_segment",
    "max_causal_len_pad_segment",
    "_full_only_seq_offsets_pad_segment",
    "max_full_len_pad_segment",
    "_sample_offsets_pad_segment",
    "max_sample_len_pad_segment",
)


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
    assert int(pack["_causal_seq_offsets_pad_segment"][-1]) == get_und_seq(pack).shape[0]
    assert int(pack["_full_only_seq_offsets_pad_segment"][-1]) == get_gen_seq(pack).shape[0]


@pytest.mark.L0
@pytest.mark.CPU
def test_pack_without_paired_splits_carries_no_pad_segments() -> None:
    """The other side of the contract: nothing to pair, so no pad segments and callers fall back.

    The pad segment pairs the two streams segment for segment, so it only applies when every
    sample contributes both a causal and a full split. An AR no-text pack carries full splits
    only (see ``test_prepare_sequence_pack_metadata_no_causal_splits``), leaving the causal side
    with nothing to pair against, so the constructor emits none of the six fields.
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


if __name__ == "__main__":
    test_two_way_attention_vs_three_way_attention()
