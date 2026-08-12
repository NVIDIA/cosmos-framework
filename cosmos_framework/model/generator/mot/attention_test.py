# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import random  # noqa: I001 - release import rewriting changes the package sort order.
import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask

import cosmos_framework.model.generator.mot.attention as attention
from cosmos_framework.model.attention.natten import NATTEN_SUPPORTED
from cosmos_framework.utils.misc import set_torch_compile_options
from cosmos_framework.model.generator.mot.attention import (
    build_packed_sequence,
)
from cosmos_framework.model.generator.mot.flex_attention import (
    FlexBackend,
    build_multiview_block_mask,
    resolve_flex_backend,
)
from cosmos_framework.model.generator.mot.flex_attention_test import _FLASH_UNAVAILABLE_MARKERS
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_all_seq,
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


def _varlen_kwargs_with(sample_offsets: object) -> dict[str, object]:
    offsets = torch.tensor([0, 4], dtype=torch.int32)  # [num_samples+1]
    return attention._varlen_kwargs(
        cast(torch.Tensor, sample_offsets),
        cumulative_seqlen_Q=offsets,
        cumulative_seqlen_KV=offsets,
        max_seqlen_Q=4,
        max_seqlen_KV=4,
    )


@pytest.mark.L0
def test_varlen_kwargs_does_not_read_the_sample_count_while_training() -> None:
    """A pack's sample count varies from step to step, so reading it costs a recompile.

    ``_varlen_kwargs`` only wants the count to take a dense-attention shortcut that is gated to
    inference, and short-circuit evaluation is what keeps training away from it.
    """
    varlen_kwargs = _varlen_kwargs_with(_SampleCountTripwire())

    assert set(varlen_kwargs) == {"cumulative_seqlen_Q", "cumulative_seqlen_KV", "max_seqlen_Q", "max_seqlen_KV"}


@pytest.mark.L0
def test_varlen_kwargs_takes_the_dense_path_for_a_single_sample_without_grad() -> None:
    with torch.no_grad():
        assert _varlen_kwargs_with(torch.tensor([0, 4], dtype=torch.int32)) == {}


@pytest.mark.L0
def test_varlen_kwargs_stays_varlen_for_several_samples_without_grad() -> None:
    with torch.no_grad():
        assert _varlen_kwargs_with(torch.tensor([0, 2, 4], dtype=torch.int32)) != {}


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
        token_shapes=list(shape.token_shapes),
        condition_masks=[torch.zeros(latent_t, dtype=torch.bool) for latent_t, _, _ in shape.token_shapes],
        num_vision_items_per_sample=[1] * len(shape.und_lens),
        num_views_per_vision_item=list(shape.num_views),
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


if __name__ == "__main__":
    test_two_way_attention_vs_three_way_attention()
