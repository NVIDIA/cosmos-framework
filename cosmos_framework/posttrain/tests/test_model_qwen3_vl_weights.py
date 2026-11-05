# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLWeightMapper, WeightTensorSpec


class _StateModel:
    def __init__(self, state: dict[str, torch.Tensor], *, model_type: str, tied: bool, num_experts: int = 0) -> None:
        self._state = state
        self.config = SimpleNamespace(
            model_type=model_type,
            tie_word_embeddings=tied,
            text_config=SimpleNamespace(num_experts=num_experts),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state)


def _dense_state() -> dict[str, torch.Tensor]:
    embedding = torch.arange(64, dtype=torch.float32).reshape(16, 4)  # [V,H]
    return {
        "model.language_model.embed_tokens.weight": embedding,
        "lm_head.weight": embedding,
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),  # [H,H]
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(4, 4),  # [H,H]
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(4, 4),  # [H,H]
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4),  # [F,H]
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(8, 4),  # [F,H]
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(4),  # [H]
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(12, 4),  # [3H,H]
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_dense_manifest_deduplicates_tied_head_and_describes_packed_targets() -> None:
    mapper = Qwen3VLWeightMapper("qwen3_vl", tie_word_embeddings=True)
    specs = mapper.manifest(_StateModel(_dense_state(), model_type="qwen3_vl", tied=True))
    by_name = {spec.wire_name: spec for spec in specs}

    assert "lm_head.weight" not in by_name
    embedding = by_name["model.language_model.embed_tokens.weight"]
    assert set(embedding.engine_targets) == {
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    }

    qkv_specs = [spec for spec in specs if spec.layout == "packed_qkv"]
    assert {spec.packed_part for spec in qkv_specs} == {"q", "k", "v"}
    assert {spec.engine_targets[0] for spec in qkv_specs} == {"language_model.model.layers.0.self_attn.qkv_proj.weight"}
    gate_up_specs = [spec for spec in specs if spec.layout == "packed_gate_up"]
    assert {spec.packed_part for spec in gate_up_specs} == {"gate", "up"}


@pytest.mark.L0
@pytest.mark.CPU
def test_weight_mapper_rejects_name_and_destination_collisions() -> None:
    tensor = torch.zeros(2, 2)  # [H,H]
    model = _StateModel(
        {
            "model.visual.proj.weight": tensor,
            "model.visual._orig_mod.proj.weight": tensor,
        },
        model_type="qwen3_vl",
        tied=False,
    )
    mapper = Qwen3VLWeightMapper("qwen3_vl")
    with pytest.raises(ValueError, match="normalization collision"):
        mapper.manifest(model)

    duplicate_targets = [
        WeightTensorSpec("a", "a", (2,), "torch.float32", ("packed",)),
        WeightTensorSpec("b", "b", (2,), "torch.float32", ("packed",)),
    ]
    with pytest.raises(ValueError, match="ambiguous contributors"):
        mapper.validate_manifest(duplicate_targets)

    witness = mapper.witness_spec(_StateModel(_dense_state(), model_type="qwen3_vl", tied=True))
    assert witness.wire_name == "model.language_model.layers.0.self_attn.q_proj.weight"
    assert witness.destination_name == "language_model.model.layers.0.self_attn.qkv_proj.weight"
    assert witness.shape == (4, 4)
    assert (witness.layout, witness.packed_part) == ("packed_qkv", "q")


@pytest.mark.L0
@pytest.mark.CPU
def test_moe_manifest_and_bucket_preserve_native_fused_layout() -> None:
    gate_up = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6)  # [E,H,2F]
    down = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)  # [E,F,H]
    state = {
        "model.language_model.embed_tokens.weight": torch.randn(16, 4),  # [V,H]
        "lm_head.weight": torch.randn(16, 4),  # [V,H]
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(2, 4),  # [E,H]
        "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.mlp.experts.down_proj": down,
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),  # [Q,H]
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(4, 4),  # [K,H]
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(4, 4),  # [V,H]
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(12, 4),  # [3H,H]
    }
    model = _StateModel(state, model_type="qwen3_vl_moe", tied=False, num_experts=2)
    mapper = Qwen3VLWeightMapper("qwen3_vl_moe")
    specs = mapper.manifest(model)
    experts = [spec for spec in specs if ".mlp.experts." in spec.wire_name]

    assert {spec.layout for spec in experts} == {"moe_gate_up_fused", "moe_down_fused"}
    # Checkpoint-container form: the routed-expert parameters sit on the `experts` module; the
    # rollout resolves them onto its live engine layout, so no vLLM nesting is named here.
    assert {spec.engine_targets[0] for spec in experts} == {
        "language_model.model.layers.0.mlp.experts.w13_weight",
        "language_model.model.layers.0.mlp.experts.w2_weight",
    }
    assert mapper.witness_spec(model).destination_name == "language_model.model.layers.0.self_attn.qkv_proj.weight"
    expert_names = {spec.wire_name for spec in experts}
    bucket = [entry for entry in mapper.wire_manifest(model) if entry[0] in expert_names]
    exported = mapper.export_bucket(model, bucket)
    assert torch.equal(exported[experts[0].wire_name], state[experts[0].source_name])
    assert torch.equal(exported[experts[1].wire_name], state[experts[1].source_name])
    assert exported["model.language_model.layers.0.mlp.experts.gate_up_proj"].shape == (2, 4, 6)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("name", "weight", "message"),
    [
        ("shared_expert.down_proj.weight", torch.randn(4, 3), "no stock vLLM destination"),  # [H,F]
        ("experts.gate_up_proj", torch.randn(2, 4, 5), "even final dimension"),  # [E,H,odd]
    ],
)
def test_invalid_moe_extensions_fail_before_export(name: str, weight: torch.Tensor, message: str) -> None:
    state = {
        "model.language_model.embed_tokens.weight": torch.randn(16, 4),  # [V,H]
        "lm_head.weight": torch.randn(16, 4),  # [V,H]
        f"model.language_model.layers.0.mlp.{name}": weight,
    }
    model = _StateModel(state, model_type="qwen3_vl_moe", tied=False, num_experts=2)
    with pytest.raises(ValueError, match=message):
        Qwen3VLWeightMapper("qwen3_vl_moe").manifest(model)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("projection_names", "message"),
    [
        (("q_proj", "k_proj"), "q/k/v contributors"),
        (("gate_proj",), "gate/up contributors"),
    ],
)
def test_wire_manifest_rejects_incomplete_packed_contributors(
    projection_names: tuple[str, ...],
    message: str,
) -> None:
    state: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(16, 4),  # [V,H]
        "lm_head.weight": torch.randn(16, 4),  # [V,H]
    }
    for name in projection_names:
        section = "self_attn" if name in ("q_proj", "k_proj") else "mlp"
        state[f"model.language_model.layers.0.{section}.{name}.weight"] = torch.randn(4, 4)  # [H,H]
    model = _StateModel(state, model_type="qwen3_vl", tied=False)

    with pytest.raises(ValueError, match=message):
        Qwen3VLWeightMapper("qwen3_vl").wire_manifest(model)


@pytest.mark.L0
@pytest.mark.CPU
def test_wire_protocol_canonicalizes_wrappers_and_exports_unchanged_tensor() -> None:
    state = _dense_state()
    original_name = "model.visual.blocks.0.attn.qkv.weight"
    wrapped_name = "model.visual._orig_mod.blocks.0.attn.qkv.weight"
    tensor = state.pop(original_name)
    state[wrapped_name] = tensor
    model = _StateModel(state, model_type="qwen3_vl", tied=False)
    mapper = Qwen3VLWeightMapper("qwen3_vl")

    wire_manifest = mapper.wire_manifest(model)
    assert original_name in {name for name, _shape, _dtype in wire_manifest}
    assert wrapped_name not in {name for name, _shape, _dtype in wire_manifest}
    exported = mapper.export_bucket(model, [entry for entry in wire_manifest if entry[0] == original_name])
    assert torch.equal(exported[original_name], tensor)
    assert mapper.destination_names(model) == tuple(
        sorted({target for spec in mapper.manifest(model) for target in spec.engine_targets})
    )
