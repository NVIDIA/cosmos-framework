# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker, SFTPolicyInput
from cosmos_framework.posttrain.model.protocol import VLMPolicy
from cosmos_framework.posttrain.model.qwen3_vl import (
    Qwen3VLBuildHooks,
    Qwen3VLPolicy,
    _coerce_native_config,
    _qwen3_vl_causal_mask,
    build_qwen3_vl_model,
    build_qwen3_vl_policy,
    override_qwen3_vl_forward,
    validate_qwen3_vl_parallelism,
)
from cosmos_framework.posttrain.model.qwen3_vl_moe import override_qwen3_vl_moe_for_padded_batches
from cosmos_framework.posttrain.tests.helpers.sft_packer import local_tokenizer


class _Config:
    tie_word_embeddings: bool = False


class _ForwardOutput:
    def __init__(self, logits: torch.Tensor) -> None:  # logits: [B,T,V]
        self.logits = logits


class _TinyQwen3VL(torch.nn.Module):
    def __init__(self, config: Any | None = None) -> None:
        super().__init__()
        self.config = config or _Config()
        self.embed = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32, bias=False)
        self.seen: dict[str, Any] = {}

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> _ForwardOutput:  # input_ids: [B,T]
        self.seen = dict(kwargs)
        hidden = self.embed(input_ids)  # [B,T,H]
        logits = self.head(hidden)  # [B,T,V]
        return _ForwardOutput(logits)


def _pure_fsdp_dims(**overrides: int) -> SimpleNamespace:
    values = {
        "world_size": 2,
        "dp_shard": 2,
        "dp_replicate": 1,
        "tp": 1,
        "ep": 1,
        "cp": 1,
        "pp": 1,
        "cfgp": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.L0
@pytest.mark.CPU
def test_qwen3_vl_policy_forwards_complete_multimodal_batch() -> None:
    net = _TinyQwen3VL()
    policy = Qwen3VLPolicy(net, "qwen3_vl")
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),  # [B,T]
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),  # [B,T]
        "pixel_values": torch.randn(8, 6),  # [N_patch,D]
        "image_grid_thw": torch.tensor([[1, 2, 4]], dtype=torch.long),  # [N_image,3]
        "logprob_masks": torch.tensor([[False, True, False]]),  # [B,T]
    }

    output = policy.forward_policy_batch(batch)

    assert output.logits.shape == (1, 3, 32)
    assert set(net.seen) == {"attention_mask", "pixel_values", "image_grid_thw", "use_cache"}
    assert net.seen["use_cache"] is False
    assert isinstance(policy, VLMPolicy)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        *(({axis: 2}, axis) for axis in ("tp", "ep", "cp", "pp", "cfgp")),
        ({"dp_shard": 1}, "rank width"),
        ({"dp_replicate": 2}, "rank width"),
        ({"dp_replicate": 0}, "positive"),
        ({"dp_replicate": -1, "dp_shard": -2}, "positive"),
        ({"world_size": 0, "dp_shard": 0}, "positive"),
    ],
)
def test_qwen3_vl_parallelism_rejects_unsupported_meshes(overrides: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_qwen3_vl_parallelism(_pure_fsdp_dims(**overrides))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(("replicate", "shard"), [(1, 1), (1, 8), (2, 4), (16, 64), (32, 32), (8, 1)])
def test_qwen3_vl_parallelism_accepts_one_factorized_policy_world(replicate: int, shard: int) -> None:
    validate_qwen3_vl_parallelism(_pure_fsdp_dims(world_size=replicate * shard, dp_replicate=replicate, dp_shard=shard))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("replicate", [1, 2])
def test_builder_runs_injected_native_lifecycle_in_order(monkeypatch: pytest.MonkeyPatch, replicate: int) -> None:
    from cosmos_framework.posttrain.model import qwen3_vl

    events: list[str] = []
    config = _Config()

    def model_factory(native_config: Any) -> torch.nn.Module:
        assert native_config is config
        events.append("construct")
        return _TinyQwen3VL(native_config)

    def prepare_model(_model: torch.nn.Module) -> None:
        events.append("prepare")

    def parallelize(_model: torch.nn.Module, dims: Any) -> None:
        assert dims.world_size == replicate * 2
        assert dims.dp_replicate == replicate
        events.append("parallelize")

    def materialize(_model: torch.nn.Module) -> None:
        events.append("materialize")

    def load_weights(_model: torch.nn.Module, path: str, credentials: str | None, dims: Any) -> None:
        assert path == "/checkpoint"
        assert credentials == "cred"
        assert dims.dp_shard == 2
        assert dims.dp_replicate == replicate
        events.append("load")

    monkeypatch.setattr(qwen3_vl, "_native_qwen3_vl_classes", lambda _model_type: (_Config, _TinyQwen3VL))
    hooks = Qwen3VLBuildHooks(
        model_factory=model_factory,
        prepare_model=prepare_model,
        parallelize=parallelize,
        materialize=materialize,
        load_weights=load_weights,
    )

    policy = build_qwen3_vl_policy(
        model_type="qwen3_vl",
        model_config=config,
        checkpoint_path="/checkpoint",
        credential_path="cred",
        parallel_dims=_pure_fsdp_dims(world_size=replicate * 2, dp_replicate=replicate),
        dtype=torch.float32,
        hooks=hooks,
        initialize_on_meta=False,
    )

    assert isinstance(policy, Qwen3VLPolicy)
    assert events == ["construct", "prepare", "parallelize", "materialize", "load"]


@pytest.mark.L0
@pytest.mark.CPU
def test_native_config_restores_live_moe_expert_count_from_lossy_hf_dict() -> None:
    class NativeConfig:
        @classmethod
        def from_dict(cls, values: dict[str, Any]) -> SimpleNamespace:
            return SimpleNamespace(
                text_config=SimpleNamespace(**values["text_config"]),
                vision_config=SimpleNamespace(),
            )

    hf_config = SimpleNamespace(
        text_config=SimpleNamespace(num_experts=128),
        to_dict=lambda: {
            "text_config": {
                "num_experts": None,
                "num_experts_per_tok": 8,
                "num_local_experts": 128,
                "pad_token_id": None,
            }
        },
    )

    native_config = _coerce_native_config(NativeConfig, hf_config, "eager")

    assert native_config.text_config.num_experts == 128
    assert native_config.text_config.num_local_experts == 128
    assert native_config.text_config.pad_token_id is None


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("distributed", "FSDP/activation-checkpointing hook"),
        ("meta_without_weights", "meta-initialized"),
    ],
)
def test_builder_rejects_incomplete_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    from cosmos_framework.posttrain.model import qwen3_vl

    monkeypatch.setattr(qwen3_vl, "_native_qwen3_vl_classes", lambda _model_type: (_Config, _TinyQwen3VL))
    kwargs: dict[str, Any] = {}
    if case == "distributed":
        kwargs = {
            "checkpoint_path": "/checkpoint",
            "parallel_dims": _pure_fsdp_dims(),
            "hooks": Qwen3VLBuildHooks(materialize=lambda _model: None, load_weights=lambda *_args: None),
            "initialize_on_meta": False,
        }
    with pytest.raises(ValueError, match=message):
        build_qwen3_vl_policy(
            model_type="qwen3_vl",
            model_config=_Config(),
            **kwargs,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_dummy_vision_forward_override_accepts_i4_native_qwen3_vl() -> None:
    """Text/empty FSDP lanes use the same vision collective sequence as image-bearing peers."""
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLModel

    model = Qwen3VLModel.__new__(Qwen3VLModel)
    torch.nn.Module.__init__(model)

    override_qwen3_vl_forward(model)

    assert model.forward.__func__.__name__ == "patched_forward"


@pytest.mark.L0
@pytest.mark.CPU
def test_native_moe_adapter_flattens_and_restores_padded_hidden_states() -> None:
    """The vendored sparse block's packed tuple contract is adapted before decoder/FSDP wrapping."""
    from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
        Qwen3VLMoeTextSparseMoeBlock,
    )

    block = Qwen3VLMoeTextSparseMoeBlock.__new__(Qwen3VLMoeTextSparseMoeBlock)
    torch.nn.Module.__init__(block)

    def _packed_forward(
        _self: Any,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str]:  # hidden_states: [N,H], returns [N,H], metadata
        assert hidden_states.ndim == 2
        assert token_mask is not None and token_mask.shape == (6,)
        return hidden_states + 1.0, "lbl"

    block.forward = MethodType(_packed_forward, block)
    root = torch.nn.Module()
    root.add_module("block", block)

    assert override_qwen3_vl_moe_for_padded_batches(root) == 1
    block._posttrain_token_mask = torch.ones(6, dtype=torch.bool)  # [B*T]
    hidden = torch.zeros((2, 3, 4), dtype=torch.float32)  # [B,T,H]
    output = block(hidden)  # [B,T,H]

    assert output.shape == hidden.shape
    assert torch.equal(output, torch.ones_like(hidden))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe"])
@pytest.mark.parametrize(("replicate", "shard"), [(1, 8), (2, 4), (8, 1)])
def test_native_qwen_builder_passes_hsdp_mesh_and_degrees_to_fsdp(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    replicate: int,
    shard: int,
) -> None:
    """Both families keep the configured HSDP factorization through the native wrapping seam."""
    from cosmos_framework.model.generator import parallelize_vlm
    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
    from cosmos_framework.posttrain.model import qwen3_vl

    resolved_config = SimpleNamespace(model_type=model_type, tie_word_embeddings=False)
    dims = _pure_fsdp_dims(world_size=replicate * shard, dp_replicate=replicate, dp_shard=shard)
    dims.dp_enabled = True
    dims.dp_replicate_enabled = replicate > 1
    dims.dp_mesh = object()
    dims.dp_shard_mesh = object()
    net = _TinyQwen3VL()
    captured_config: list[Any] = []
    wrapped: list[tuple[torch.nn.Module, Any]] = []
    sentinel = object()
    apply_fsdp = parallelize_vlm.apply_fsdp

    def record_fsdp(model: Any, slots: list[Any], parallel_dims: Any, config: Any, precision: str) -> None:
        assert parallel_dims is dims
        assert precision == "bfloat16"
        captured_config.append(config)
        apply_fsdp(model, slots, parallel_dims, config, precision)

    def record_fully_shard(model: torch.nn.Module, **kwargs: Any) -> None:
        assert kwargs["mp_policy"].reduce_dtype == torch.float32
        wrapped.append((model, kwargs["mesh"]))

    def build_with_native_hooks(**kwargs: Any) -> Any:
        assert kwargs["parallel_dims"] is dims
        assert kwargs["model_type"] == model_type
        kwargs["hooks"].parallelize(net, dims)
        return sentinel

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda _path: resolved_config)
    monkeypatch.setattr(parallelize_vlm, "apply_ac", lambda _model, _config: [(net, "head", net.head)])
    monkeypatch.setattr(parallelize_vlm, "apply_fsdp", record_fsdp)
    monkeypatch.setattr(parallelize_vlm, "fully_shard", record_fully_shard)
    monkeypatch.setattr(qwen3_vl, "build_qwen3_vl_policy", build_with_native_hooks)

    assert build_qwen3_vl_model(Qwen3VLModelConfig(name_or_path="test/checkpoint"), dims) is sentinel
    assert len(captured_config) == 1
    assert captured_config[0].data_parallel_replicate_degree == replicate
    assert captured_config[0].data_parallel_shard_degree == shard
    expected_mesh = dims.dp_mesh if replicate > 1 else dims.dp_shard_mesh
    assert wrapped == [(net.head, expected_mesh), (net, expected_mesh)]


@pytest.mark.L0
@pytest.mark.CPU
def test_native_qwen_builder_rejects_invalid_mesh_before_checkpoint_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig

    def fail_lookup(_path: str) -> None:
        pytest.fail("invalid mesh must fail before accessing the checkpoint")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", fail_lookup)
    with pytest.raises(ValueError, match="rank width"):
        build_qwen3_vl_model(
            Qwen3VLModelConfig(name_or_path="test/checkpoint"),
            _pure_fsdp_dims(dp_replicate=2),
        )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("rows", "shards", "nonempty", "last_rows"),
    [(128, 1024, 128, 1), (4304, 128, 127, 20), (128, 64, 64, 2)],
)
def test_native_checkpoint_slices_match_fsdp_empty_and_uneven_shards(
    rows: int, shards: int, nonempty: int, last_rows: int
) -> None:
    """Loading preserves every row even when Ultra expert tensors have more ranks than experts."""
    from cosmos_framework.model.generator.utils.safetensors_loader import _shard_first_dim

    source = torch.arange(rows, dtype=torch.float32)[:, None]  # [E,1]
    parts = [_shard_first_dim(source, shards, rank) for rank in range(shards)]  # each [E_local,1]
    assert sum(part.shape[0] > 0 for part in parts) == nonempty
    assert parts[nonempty - 1].shape == (last_rows, 1)
    assert all(part.shape == (0, 1) for part in parts[nonempty:])
    assert torch.equal(torch.cat(parts), source)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("lengths", [(5, 5), (3, 5)], ids=["all_valid", "right_padding"])
@pytest.mark.parametrize("mask_dtype", [torch.int64, torch.int32, torch.bool], ids=["int64", "int32", "bool"])
def test_native_causal_mask_accepts_sft_padding_masks(lengths: tuple[int, int], mask_dtype: torch.dtype) -> None:
    """The real SFT collator's integer mask must retain causal AND valid-key semantics."""
    length = max(lengths)
    packer = QwenSFTDataPacker(local_tokenizer(), max_length=length)
    samples = [SFTPolicyInput(list(range(1, size + 1)), [-100] + list(range(2, size + 1))) for size in lengths]
    batch = packer.sft_collate_fn(samples, length)
    assert batch["attention_mask"].dtype == torch.int64
    attention_mask = batch["attention_mask"].to(mask_dtype)  # [B,T]
    input_embeds = torch.zeros(len(lengths), length, 4, dtype=torch.bfloat16)  # [B,T,D]
    cache_position = torch.arange(length)  # [T]

    actual = _qwen3_vl_causal_mask(
        SimpleNamespace(_attn_implementation="eager"), input_embeds, attention_mask, cache_position, None
    )  # [B,1,T,T]
    minimum = torch.finfo(input_embeds.dtype).min
    expected = torch.tensor(
        [
            [[[0.0 if key <= query and key < size else minimum for key in range(length)] for query in range(length)]]
            for size in lengths
        ],
        dtype=input_embeds.dtype,
    )  # [B,1,T,T]
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.CPU
def test_native_causal_mask_preserves_four_dimensional_additive_mask() -> None:
    input_embeds = torch.zeros(1, 2, 4, dtype=torch.bfloat16)  # [B,T,D]
    additive = torch.tensor([[[[0.0, -100.0], [-0.5, 0.0]]]], dtype=input_embeds.dtype)  # [B,1,T,T]
    cache_position = torch.arange(input_embeds.shape[1])  # [T]

    actual = _qwen3_vl_causal_mask(
        SimpleNamespace(_attn_implementation="eager"), input_embeds, additive, cache_position, None
    )  # [B,1,T,T]

    assert actual is additive
