# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

import cosmos_framework.model.generator.reasoner_features as reasoner_features
from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.model.attention.utils import is_blackwell_dc, is_hopper
from cosmos_framework.model.generator.mot.attention import build_packed_sequence
from cosmos_framework.model.generator.mot.unified_mot import (
    Qwen3VLMoTConfig,
    Qwen3VLTextForCausalLM,
    prune_und_pathway_,
)
from cosmos_framework.model.generator.reasoner_features import (
    CapturingReasonerKVMemoryState,
    ReasonerFeatureBatch,
    ReasonerFeatureRequest,
    ReasonerLayerKV,
    StaticReasonerKVMemoryState,
    StaticReasonerKVMemoryValue,
    extract_reasoner_feature_batch,
)


def _pack(
    sequence: torch.Tensor,
    und_offsets: torch.Tensor,
    gen_offsets: torch.Tensor,
    *,
    is_sharded: bool = False,
) -> dict:
    num_samples = gen_offsets.numel() - 1
    return {
        "causal_seq": sequence.new_empty((0, *sequence.shape[1:])),
        "full_only_seq": sequence,
        "sample_offsets": torch.arange(num_samples + 1, dtype=torch.int32),
        "max_sample_len": 1,
        "max_causal_len": int(torch.diff(und_offsets).max()),
        "max_full_len": int(torch.diff(gen_offsets).max()),
        "_causal_indices": torch.empty(0, dtype=torch.int64),
        "_full_indices": torch.arange(sequence.shape[0]),
        "_causal_seq_offsets": und_offsets,
        "_full_only_seq_offsets": gen_offsets,
        "_num_causal_tokens": int(und_offsets[-1]),
        "_num_full_tokens": int(gen_offsets[-1]),
        "is_sharded": is_sharded,
    }


def _two_way_mask() -> SimpleNamespace:
    return SimpleNamespace(
        is_three_way=False,
        control_stream_token_ranges=None,
        flex_block_mask=None,
        multiview_maskless=None,
    )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_canonical_batch_validates_shapes_detaches_and_round_trips_cache_tensors() -> None:
    cross_k = torch.arange(24.0, requires_grad=True).reshape(2, 3, 2, 2)
    cross_v = (cross_k + 100).detach().requires_grad_()
    offsets = torch.tensor([0, 2, 3], dtype=torch.int32)

    batch = ReasonerFeatureBatch.from_stacked(cross_k, cross_v, offsets, ("first", "second"))

    assert batch.num_layers == 2
    assert batch.num_samples == 2
    assert batch.sequence_length == 3
    assert all(not tensor.requires_grad for tensor in (*batch.cross_k, *batch.cross_v))

    restored = ReasonerFeatureBatch.from_cache_tensors(
        batch.to_cache_tensors(),
        fingerprints=batch.fingerprints,
    )
    restored_k, restored_v = restored.to_stacked()
    assert torch.equal(restored_k, cross_k)
    assert torch.equal(restored_v, cross_v)
    assert torch.equal(restored.causal_offsets, offsets)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_extractor_rejects_multiple_causal_documents_per_request() -> None:
    request = ReasonerFeatureRequest(
        sample_key="per-view",
        token_ids=torch.tensor([1, 2]),
        position_ids=torch.zeros(3, 2),
        causal_offsets=torch.tensor([0, 1, 2]),
        fingerprint="fingerprint",
    )

    with pytest.raises(NotImplementedError, match="one causal document"):
        extract_reasoner_feature_batch(object(), [request])


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize(
    ("cross_k", "cross_v", "error"),
    [
        (torch.zeros(3, 2), torch.zeros(3, 2), "shape"),
        (torch.zeros(3, 2, 4), torch.zeros(4, 2, 4), "shapes must match"),
        (torch.zeros(3, 2, 4), torch.zeros(3, 3, 4), "shapes must match"),
    ],
)
def test_layer_kv_rejects_invalid_shapes(
    cross_k: torch.Tensor,
    cross_v: torch.Tensor,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        ReasonerLayerKV(cross_k, cross_v)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_static_state_rejects_sequence_pack_offset_mismatch() -> None:
    features = ReasonerFeatureBatch(
        (torch.zeros(3, 1, 2),),
        (torch.zeros(3, 1, 2),),
        torch.tensor([0, 2, 3], dtype=torch.int32),
        ("first", "second"),
    )
    state = StaticReasonerKVMemoryState(features)
    hidden = torch.zeros(3, 1, 2)
    pack = _pack(
        hidden,
        torch.tensor([0, 1, 3], dtype=torch.int32),
        torch.tensor([0, 1, 3], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="causal offsets do not match"):
        state.init(pack, torch.device("cpu"))


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_static_state_rejects_reasoner_gen_sample_count_mismatch() -> None:
    features = ReasonerFeatureBatch(
        (torch.zeros(3, 1, 2),),
        (torch.zeros(3, 1, 2),),
        torch.tensor([0, 2, 3], dtype=torch.int32),
        ("first", "second"),
    )
    state = StaticReasonerKVMemoryState(features)
    hidden = torch.zeros(3, 1, 2)
    pack = _pack(
        hidden,
        torch.tensor([0, 2, 3], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="sample counts disagree"):
        state.init(pack, torch.device("cpu"))


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_multi_sample_cached_attention_isolates_samples_and_zero_fills_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    und_offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    gen_offsets = torch.tensor([0, 1, 3], dtype=torch.int32)
    cached_k = torch.tensor([10.0, 11.0, 20.0]).reshape(3, 1, 1).requires_grad_()
    cached_v = cached_k.detach().clone().requires_grad_()
    features = ReasonerFeatureBatch((cached_k,), (cached_v,), und_offsets, ("first", "second"))
    state = StaticReasonerKVMemoryState(features)

    q_gen = torch.tensor([1.0, 2.0, 3.0, 999.0]).reshape(4, 1, 1).requires_grad_()
    k_gen = torch.tensor([100.0, 200.0, 201.0, 999.0]).reshape(4, 1, 1).requires_grad_()
    v_gen = k_gen.detach().clone().requires_grad_()
    state.init(_pack(q_gen, und_offsets, gen_offsets), torch.device("cpu"))
    memory_value = state.read_for_layer(0)
    captured: dict[str, object] = {}

    def fake_attention(
        *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        captured.update(query=query, key=key, value=value, **kwargs)
        return query + 5

    monkeypatch.setattr(reasoner_features, "attention", fake_attention)
    output, kv_to_store = reasoner_features.dispatch_attention_with_reasoner_features(
        _pack(q_gen, und_offsets, gen_offsets),
        _pack(k_gen, und_offsets, gen_offsets),
        _pack(v_gen, und_offsets, gen_offsets),
        _two_way_mask(),
        memory_value=memory_value,
    )

    assert kv_to_store is None
    assert isinstance(memory_value, StaticReasonerKVMemoryValue)
    assert captured["query"].flatten().tolist() == [1.0, 2.0, 3.0]
    assert captured["key"].flatten().tolist() == [10.0, 11.0, 100.0, 20.0, 200.0, 201.0]
    assert captured["value"].flatten().tolist() == [10.0, 11.0, 100.0, 20.0, 200.0, 201.0]
    assert torch.equal(captured["cumulative_seqlen_Q"], gen_offsets)
    assert torch.equal(captured["cumulative_seqlen_KV"], torch.tensor([0, 3, 6], dtype=torch.int32))
    assert captured["max_seqlen_Q"] == 2
    assert captured["max_seqlen_KV"] == 3
    assert output["full_only_seq"].flatten().tolist() == [6.0, 7.0, 8.0, 0.0]

    output["full_only_seq"][:3].sum().backward()
    assert q_gen.grad is not None
    assert k_gen.grad is None  # fake attention consumes only Q; cached tensors stay detached either way
    assert cached_k.grad is None
    assert cached_v.grad is None


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_capture_state_detaches_inline_kv_and_switches_to_gen_only() -> None:
    und_offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    gen_offsets = torch.tensor([0, 1, 3], dtype=torch.int32)
    state = CapturingReasonerKVMemoryState(1, fingerprints=("first", "second"))
    state.init(_pack(torch.zeros(3, 1, 1), und_offsets, gen_offsets), torch.device("cpu"))

    assert not state.is_gen_only()
    captured_k = torch.tensor([10.0, 11.0, 20.0]).reshape(1, 3, 1, 1).requires_grad_()
    captured_v = (captured_k + 1).detach().requires_grad_()
    generated = torch.zeros(1, 3, 1, 1)
    state.write_for_layer(0, (generated, generated, captured_k, captured_v))

    assert state.is_gen_only()
    batch = state.to_feature_batch()
    assert not batch.cross_k[0].requires_grad
    assert not batch.cross_v[0].requires_grad
    assert batch.cross_k[0].untyped_storage().data_ptr() != captured_k.untyped_storage().data_ptr()
    assert batch.cross_v[0].untyped_storage().data_ptr() != captured_v.untyped_storage().data_ptr()
    assert batch.cross_k[0].flatten().tolist() == [10.0, 11.0, 20.0]

    state.init(_pack(torch.zeros(3, 1, 1), und_offsets, gen_offsets), torch.device("cpu"))
    assert isinstance(state.read_for_layer(0), StaticReasonerKVMemoryValue)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_cached_dispatch_fails_closed_for_context_parallel_pack() -> None:
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    features = ReasonerFeatureBatch((torch.zeros(2, 1, 1),), (torch.zeros(2, 1, 1),), offsets, ("sample",))
    state = StaticReasonerKVMemoryState(features)
    sequence = torch.zeros(2, 1, 1)
    ordinary_pack = _pack(sequence, offsets, offsets)
    state.init(ordinary_pack, torch.device("cpu"))
    sharded_pack = _pack(sequence, offsets, offsets, is_sharded=True)

    with pytest.raises(ValueError, match="context-parallel"):
        reasoner_features.dispatch_attention_with_reasoner_features(
            sharded_pack,
            sharded_pack,
            sharded_pack,
            _two_way_mask(),
            memory_value=state.read_for_layer(0),
        )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_install_and_restore_reasoner_feature_dispatch() -> None:
    attention_module = SimpleNamespace(dispatch_attention_fn=reasoner_features.dispatch_attention)
    net = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=attention_module)],
            )
        )
    )

    previous = reasoner_features.install_reasoner_feature_attention_dispatch(net)
    assert attention_module.dispatch_attention_fn is reasoner_features.dispatch_attention_with_reasoner_features

    reasoner_features.restore_reasoner_feature_attention_dispatch(previous)
    assert attention_module.dispatch_attention_fn is reasoner_features.dispatch_attention


def _tiny_two_way_model_pack(
    packed_sequence: torch.Tensor,
) -> tuple[dict, object, torch.Tensor]:
    """Build a two-sample, two-way pack in original ``[UND_i, GEN_i]`` order."""
    und_lens = (3, 4)
    gen_lens = (5, 3)
    split_lens: list[int] = []
    sample_lens: list[int] = []
    und_indices: list[int] = []
    gen_indices: list[int] = []
    position_ids: list[torch.Tensor] = []
    offset = 0
    for und_len, gen_len in zip(und_lens, gen_lens):
        split_lens.extend((und_len, gen_len))
        sample_lens.append(und_len + gen_len)
        und_indices.extend(range(offset, offset + und_len))
        gen_indices.extend(range(offset + und_len, offset + und_len + gen_len))
        position_ids.append(torch.arange(und_len + gen_len, device=packed_sequence.device))
        offset += und_len + gen_len

    pack, attention_mask, natten_metadata = build_packed_sequence(
        "two_way",
        packed_sequence=packed_sequence,
        attn_modes=["causal", "full", "causal", "full"],
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=torch.tensor(und_indices, device=packed_sequence.device),
        packed_gen_token_indexes=torch.tensor(gen_indices, device=packed_sequence.device),
        num_heads=4,
        head_dim=64,
        num_layers=2,
        is_image_batch=True,
    )
    assert natten_metadata is None
    return pack, attention_mask, torch.cat(position_ids)


@pytest.mark.level(1)
@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not torch.cuda.is_available() or (not is_hopper() and not is_blackwell_dc()),
    reason="MoT attention parity requires a Hopper or Blackwell CUDA GPU",
)
def test_und_only_extractor_matches_joint_capture() -> None:
    """The reasoner-only prefill emits the same per-layer K/V as the joint path."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(4321)
    config = Qwen3VLMoTConfig(
        {
            "text_config": {
                "vocab_size": 128,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000.0,
                "max_position_embeddings": 128,
                "tie_word_embeddings": False,
            }
        }
    )
    model = Qwen3VLTextForCausalLM(config).to(device=device, dtype=dtype).eval()
    model.model.rotary_emb.init_weights(buffer_device=device)
    token_ids = (torch.tensor([4, 5, 6], device=device), torch.tensor([10, 11, 12, 13], device=device))
    requests = tuple(
        ReasonerFeatureRequest(
            sample_key=f"sample-{sample_idx}",
            token_ids=tokens,
            position_ids=torch.arange(tokens.numel(), device=device),
            causal_offsets=torch.tensor([0, tokens.numel()], device=device),
            fingerprint=f"fingerprint-{sample_idx}",
        )
        for sample_idx, tokens in enumerate(token_ids)
    )

    extracted = extract_reasoner_feature_batch(model, requests)

    base_embeddings = torch.randn(15, 256, device=device, dtype=dtype)
    und_indexes = torch.tensor([0, 1, 2, 8, 9, 10, 11], device=device)
    base_embeddings[und_indexes] = model.model.embed_tokens(torch.cat(token_ids))
    joint_pack, joint_mask, position_ids = _tiny_two_way_model_pack(base_embeddings)
    capture = CapturingReasonerKVMemoryState(2, fingerprints=("fingerprint-0", "fingerprint-1"))
    owner = SimpleNamespace(language_model=model)
    previous_dispatchers = reasoner_features.install_reasoner_feature_attention_dispatch(owner)
    try:
        with torch.no_grad():
            model(joint_pack, attention_mask=joint_mask, position_ids=position_ids, memory=capture)
        captured = capture.to_feature_batch()
    finally:
        reasoner_features.restore_reasoner_feature_attention_dispatch(previous_dispatchers)

    assert captured.causal_offsets.tolist() == extracted.causal_offsets.tolist()
    for layer_idx in range(2):
        torch.testing.assert_close(extracted.cross_k[layer_idx], captured.cross_k[layer_idx], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(extracted.cross_v[layer_idx], captured.cross_v[layer_idx], rtol=2e-2, atol=2e-2)


@pytest.mark.level(1)
@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not torch.cuda.is_available() or (not is_hopper() and not is_blackwell_dc()),
    reason="MoT attention parity requires a Hopper or Blackwell CUDA GPU",
)
def test_tiny_unified_mot_pruned_structure_matches_full_cached_forward_and_gradients() -> None:
    """A structurally pruned consumer preserves cached GEN outputs and gradients."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    config = Qwen3VLMoTConfig(
        {
            "text_config": {
                "vocab_size": 128,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000.0,
                "max_position_embeddings": 128,
                "tie_word_embeddings": False,
            }
        }
    )
    full_model = Qwen3VLTextForCausalLM(config)
    generator_only_model = Qwen3VLTextForCausalLM(config)
    prune_und_pathway_(generator_only_model)

    generator_only_state_names = set(generator_only_model.state_dict())
    assert generator_only_state_names
    assert all("moe_gen" in name for name in generator_only_state_names)
    full_generator_state = {
        name: tensor for name, tensor in full_model.state_dict().items() if name in generator_only_state_names
    }
    assert full_generator_state.keys() == generator_only_state_names
    generator_only_model.load_state_dict(full_generator_state, strict=True)

    full_model = full_model.to(device=device, dtype=dtype).train()
    generator_only_model = generator_only_model.to(device=device, dtype=dtype).train()
    # Casting the module also casts non-persistent buffers; RoPE deliberately
    # keeps inv_freq in FP32, so restore it exactly as production materialization does.
    full_model.model.rotary_emb.init_weights(buffer_device=device)
    generator_only_model.model.rotary_emb.init_weights(buffer_device=device)
    for name, parameter in full_model.named_parameters():
        parameter.requires_grad_("moe_gen" in name)
    full_generator_parameters = {
        name: parameter for name, parameter in full_model.named_parameters() if parameter.requires_grad
    }
    pruned_generator_parameters = dict(generator_only_model.named_parameters())
    assert full_generator_parameters.keys() == pruned_generator_parameters.keys() == generator_only_state_names

    base_embeddings = torch.randn(15, 256, device=device, dtype=dtype)
    target = torch.randn(8, 256, device=device, dtype=torch.float32)
    capture = CapturingReasonerKVMemoryState(2, fingerprints=("sample-0", "sample-1"))
    full_owner = SimpleNamespace(language_model=full_model)
    pruned_owner = SimpleNamespace(language_model=generator_only_model)
    full_dispatchers = reasoner_features.install_reasoner_feature_attention_dispatch(full_owner)
    pruned_dispatchers = reasoner_features.install_reasoner_feature_attention_dispatch(pruned_owner)
    try:
        capture_pack, capture_mask, capture_position_ids = _tiny_two_way_model_pack(base_embeddings.clone())
        with torch.no_grad():
            full_model(
                capture_pack,
                attention_mask=capture_mask,
                position_ids=capture_position_ids,
                memory=capture,
            )
        assert capture.is_gen_only()

        full_pack, full_mask, full_position_ids = _tiny_two_way_model_pack(base_embeddings.clone())
        full_output, _ = full_model(
            full_pack,
            attention_mask=full_mask,
            position_ids=full_position_ids,
            memory=capture,
        )
        full_gen = get_gen_seq(full_output)[: full_pack["_num_full_tokens"]]
        full_loss = (full_gen.float() - target).square().mean()
        full_loss.backward()
        full_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in full_generator_parameters.items()
            if parameter.grad is not None
        }
        assert full_gradients.keys() == full_generator_parameters.keys()

        pruned_pack, pruned_mask, pruned_position_ids = _tiny_two_way_model_pack(base_embeddings.clone())
        pruned_output, _ = generator_only_model(
            pruned_pack,
            attention_mask=pruned_mask,
            position_ids=pruned_position_ids,
            memory=capture,
        )
        pruned_gen = get_gen_seq(pruned_output)[: pruned_pack["_num_full_tokens"]]
        pruned_loss = (pruned_gen.float() - target).square().mean()
        pruned_loss.backward()
        pruned_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in pruned_generator_parameters.items()
            if parameter.grad is not None
        }

        torch.testing.assert_close(pruned_gen, full_gen, rtol=2e-2, atol=2e-2)
        assert pruned_gradients.keys() == full_gradients.keys()
        for name in full_gradients:
            torch.testing.assert_close(
                pruned_gradients[name],
                full_gradients[name],
                rtol=3e-2,
                atol=3e-3,
                msg=lambda message, name=name: f"Generator gradient mismatch for {name}: {message}",
            )
        assert all(parameter.grad is None for name, parameter in full_model.named_parameters() if "moe_gen" not in name)
    finally:
        reasoner_features.restore_reasoner_feature_attention_dispatch(pruned_dispatchers)
        reasoner_features.restore_reasoner_feature_attention_dispatch(full_dispatchers)
