# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Component-level tests for the Nemotron 3 Dense VL text backbone modules.

CPU-only, no GPU or credentials needed — covers the config, RMSNorm, MLP, and
rotary-embedding building blocks.

Usage:
    pytest cosmos_framework/model/vfm/vlm/nemotron_3_dense_vl/nemotron_3_dense_vl_test.py -s -v
"""

import torch

from cosmos_framework.model.vfm.vlm.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.vfm.vlm.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding,
    Nemotron3DenseVLMLP,
    Nemotron3DenseVLPreTrainedModel,
    Nemotron3DenseVLRMSNorm,
    apply_rotary_pos_emb_partial,
    rotate_half,
)


CONFIG_JSON = "cosmos_framework/model/vfm/vlm/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"


def _make_small_config(**overrides) -> Nemotron3DenseVLTextConfig:
    """Build a small config suitable for fast CPU tests."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        max_position_embeddings=512,
        mlp_hidden_act="relu2",
    )
    defaults.update(overrides)
    return Nemotron3DenseVLTextConfig(**defaults)


# ---------------------------------------------------------------------------
# Component-level tests (CPU-only, no credentials)
# ---------------------------------------------------------------------------


class TestNemotron3DenseVLTextConfig:
    def test_defaults(self) -> None:
        cfg = Nemotron3DenseVLTextConfig()
        assert cfg.vocab_size == 131072
        assert cfg.hidden_size == 2048
        assert cfg.intermediate_size == 9216
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 16
        assert cfg.head_dim == 128
        assert cfg.num_key_value_heads == 8
        assert cfg.mlp_hidden_act == "relu2"
        assert cfg.mlp_bias is False
        assert cfg.attention_bias is False
        assert cfg.enable_rope is True
        assert cfg.enable_mrope is True
        assert cfg.mrope_section == [24, 20, 20]
        assert cfg.rope_theta == 100_000_000.0
        assert cfg.tie_word_embeddings is False

    def test_rms_norm_eps_alias(self) -> None:
        cfg = Nemotron3DenseVLTextConfig(layer_norm_epsilon=1e-6)
        assert cfg.rms_norm_eps == 1e-6

    def test_from_json_file(self) -> None:
        cfg = Nemotron3DenseVLTextConfig.from_json_file(CONFIG_JSON)
        assert cfg.vocab_size == 131072
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 28
        assert cfg.mlp_hidden_act == "relu2"
        assert cfg.mrope_section == [24, 20, 20]

    def test_custom_overrides(self) -> None:
        cfg = Nemotron3DenseVLTextConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            head_dim=64,
        )
        assert cfg.hidden_size == 512
        assert cfg.num_hidden_layers == 4
        assert cfg.num_attention_heads == 8
        assert cfg.head_dim == 64


class TestNemotron3DenseVLRMSNorm:
    def test_output_shape(self) -> None:
        norm = Nemotron3DenseVLRMSNorm(hidden_size=64, eps=1e-5)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_dtype_preservation(self) -> None:
        norm = Nemotron3DenseVLRMSNorm(hidden_size=32)
        x_fp16 = torch.randn(1, 5, 32, dtype=torch.float16)
        out = norm(x_fp16)
        assert out.dtype == torch.float16

    def test_unit_weight_is_identity_for_normalized(self) -> None:
        """With weight=1 and input already unit-norm, output should closely match input."""
        norm = Nemotron3DenseVLRMSNorm(hidden_size=16)
        x = torch.randn(1, 1, 16)
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x_unit = x / rms
        out = norm(x_unit)
        assert torch.allclose(out.float(), x_unit.float(), atol=1e-4)

    def test_extra_repr(self) -> None:
        norm = Nemotron3DenseVLRMSNorm(hidden_size=64, eps=1e-6)
        s = norm.extra_repr()
        assert "(64,)" in s
        assert "1e-06" in s


class TestNemotron3DenseVLMLP:
    def test_output_shape(self) -> None:
        cfg = _make_small_config()
        mlp = Nemotron3DenseVLMLP(cfg)
        x = torch.randn(2, 10, cfg.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape

    def test_relu2_activation_is_nonnegative(self) -> None:
        """relu(x)^2 is always >= 0."""
        cfg = _make_small_config()
        mlp = Nemotron3DenseVLMLP(cfg)
        x = torch.randn(4, 8, cfg.hidden_size)
        intermediate = mlp.act_fn(mlp.up_proj(x))
        assert (intermediate >= 0).all()

    def test_no_bias_by_default(self) -> None:
        cfg = _make_small_config(mlp_bias=False)
        mlp = Nemotron3DenseVLMLP(cfg)
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None

    def test_with_bias(self) -> None:
        cfg = _make_small_config(mlp_bias=True)
        mlp = Nemotron3DenseVLMLP(cfg)
        assert mlp.up_proj.bias is not None
        assert mlp.down_proj.bias is not None


class TestRotateHalf:
    def test_output_shape(self) -> None:
        x = torch.randn(2, 4, 8)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_self_inverse_with_negation(self) -> None:
        """rotate_half(rotate_half(x)) == -x."""
        x = torch.randn(3, 5, 16)
        out = rotate_half(rotate_half(x))
        assert torch.allclose(out, -x)


class TestApplyRotaryPosEmbPartial:
    def test_full_rotation(self) -> None:
        """When rot_dim == head_dim, all channels are rotated."""
        seq_len, n_heads, head_dim = 10, 4, 16
        q = torch.randn(seq_len, n_heads, head_dim)
        k = torch.randn(seq_len, n_heads, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        q_out, k_out = apply_rotary_pos_emb_partial(q, k, cos, sin, unsqueeze_dim=1)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_partial_rotation_passthrough(self) -> None:
        """When rot_dim < head_dim, the remainder channels pass through unchanged."""
        seq_len, n_heads, head_dim = 8, 2, 32
        rot_dim = 16
        q = torch.randn(seq_len, n_heads, head_dim)
        k = torch.randn(seq_len, n_heads, head_dim)
        cos = torch.randn(seq_len, rot_dim)
        sin = torch.randn(seq_len, rot_dim)

        q_out, k_out = apply_rotary_pos_emb_partial(q, k, cos, sin, unsqueeze_dim=1)

        assert torch.allclose(q_out[..., rot_dim:], q[..., rot_dim:])
        assert torch.allclose(k_out[..., rot_dim:], k[..., rot_dim:])

    def test_zero_angle_is_identity(self) -> None:
        """With cos=1, sin=0, the rotated output should equal the input."""
        seq_len, n_heads, head_dim = 6, 2, 16
        q = torch.randn(seq_len, n_heads, head_dim)
        k = torch.randn(seq_len, n_heads, head_dim)
        cos = torch.ones(seq_len, head_dim)
        sin = torch.zeros(seq_len, head_dim)

        q_out, k_out = apply_rotary_pos_emb_partial(q, k, cos, sin, unsqueeze_dim=1)
        assert torch.allclose(q_out, q, atol=1e-6)
        assert torch.allclose(k_out, k, atol=1e-6)


class TestMultiModalRotaryEmbedding:
    def test_output_shapes(self) -> None:
        cfg = _make_small_config()
        rope = MultiModalRotaryEmbedding(cfg)
        seq_len = 12
        x = torch.randn(1, seq_len, cfg.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        cos, sin = rope(x, position_ids)
        assert cos.shape[-1] == cfg.head_dim
        assert sin.shape[-1] == cfg.head_dim

    def test_mrope_3d_position_ids(self) -> None:
        """With 3D position_ids (3, batch, seq_len) the mrope interleaving path runs."""
        cfg = _make_small_config()
        rope = MultiModalRotaryEmbedding(cfg)
        seq_len = 8
        x = torch.randn(1, seq_len, cfg.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

        cos, sin = rope(x, position_ids)
        assert cos.shape[-1] == cfg.head_dim
        assert sin.shape[-1] == cfg.head_dim

    def test_init_weights(self) -> None:
        cfg = _make_small_config()
        rope = MultiModalRotaryEmbedding(cfg)
        orig_inv_freq = rope.inv_freq.clone()
        rope.init_weights(buffer_device=None)
        assert torch.allclose(rope.inv_freq, orig_inv_freq)

    def test_deterministic(self) -> None:
        cfg = _make_small_config()
        rope = MultiModalRotaryEmbedding(cfg)
        seq_len = 10
        x = torch.randn(1, seq_len, cfg.hidden_size)
        pos = torch.arange(seq_len).unsqueeze(0)
        cos1, sin1 = rope(x, pos)
        cos2, sin2 = rope(x, pos)
        assert torch.allclose(cos1, cos2)
        assert torch.allclose(sin1, sin2)


class TestNemotron3DenseVLPreTrainedModel:
    def test_config_class(self) -> None:
        assert Nemotron3DenseVLPreTrainedModel.config_class == Nemotron3DenseVLTextConfig

    def test_base_model_prefix(self) -> None:
        assert Nemotron3DenseVLPreTrainedModel.base_model_prefix == "model"
