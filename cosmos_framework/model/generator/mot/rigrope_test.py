# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""Analytic checks for the fixed VRAD frequency and Plucker conventions."""

import math

import pytest
import torch

from cosmos_framework.model.generator.mot.rigrope import (
    RopeRigPE,
    apply_mrope_rotary,
    apply_rig_rotary,
    resize_rig_features,
    rig_features,
    spread_over_even_pairs,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("head_dim", [16, 32, 128, 130])
def test_fixed_frequencies_and_no_parameters(head_dim: int) -> None:
    encoding = RopeRigPE(head_dim)
    assert list(encoding.parameters()) == []
    # The fixed matrix must not become a new required key in strict pre-RigRoPE warm starts.
    assert list(encoding.state_dict()) == []
    counts = torch.count_nonzero(encoding.freq_matrix, dim=1).tolist()  # [8]
    assert counts == [head_dim // 16 + int(i < (head_dim // 2) % 8) for i in range(8)]
    for row in encoding.freq_matrix:
        frequencies = row[row != 0]  # [K]
        assert float(frequencies[0]) == pytest.approx(math.pi)
        if len(frequencies) > 1:
            assert float(frequencies[-1]) == pytest.approx(8 * math.pi)
    restored = RopeRigPE(head_dim)
    # Strict loading an old checkpoint with no RigRoPE key leaves the deterministic buffer intact.
    restored.load_state_dict({}, strict=True)
    torch.testing.assert_close(restored.freq_matrix, encoding.freq_matrix, rtol=0, atol=0)


def test_frequency_range_spans_the_configured_exponents() -> None:
    encoding = RopeRigPE(128, max_freq_exponent=1.0, min_freq_exponent=-2.0)
    for row in encoding.freq_matrix:
        frequencies = row[row != 0]  # [8]
        assert float(frequencies[0]) == pytest.approx(math.pi / 4)
        assert float(frequencies[-1]) == pytest.approx(2 * math.pi)
    # Opposite-facing unit directions differ by 2 per component: the slowest slot must not wrap.
    assert 2 * float(encoding.freq_matrix[encoding.freq_matrix != 0].min()) < 2 * math.pi
    rebuilt = encoding.freq_matrix.clone()  # [8,D/2]
    encoding.freq_matrix.zero_()
    encoding.reset_parameters()
    torch.testing.assert_close(encoding.freq_matrix, rebuilt, rtol=0, atol=0)


@pytest.mark.parametrize("exponents", [(1.0, 0.0), (0.0, math.inf)])
def test_frequency_range_rejects_inverted_or_infinite_exponents(exponents: tuple[float, float]) -> None:
    low, high = exponents
    with pytest.raises(ValueError, match="exponent"):
        RopeRigPE(16, max_freq_exponent=high, min_freq_exponent=low)


def test_vrad_moment_normalization_zero_depth_and_time() -> None:
    directions = torch.tensor([[[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]]])  # [1,2,3]
    origin = torch.tensor([5.0, 0.0, 0.0])  # [3]
    times = torch.tensor([0.0, 0.1, 0.2])  # [3]
    features = rig_features(directions, origin, times)  # [3,1,2,8]
    torch.testing.assert_close(features[0, 0, 0, :7], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert torch.count_nonzero(features[:, :, :, 6]) == 0
    assert torch.count_nonzero(features[:, :, 1, 3:6]) == 0  # Origin lies on the second ray.
    torch.testing.assert_close(features[:, 0, 0, 7], times)
    # VRAD intentionally removes metric baseline length by independently normalizing moment.
    torch.testing.assert_close(features, rig_features(directions, origin * 10, times))


def test_feature_interpolation_is_per_view_without_renormalization() -> None:
    features = torch.zeros(2, 1, 2, 2, 8)  # [V,T,H,W,8]
    features[0, :, 0, :, 0] = 1  # [T,W]
    features[0, :, 1, :, 1] = 1  # [T,W]
    features[1, :, :, :, 2] = 1  # [T,H,W]
    resized = resize_rig_features(features, (1, 1, 1))  # [2,1,1,1,8]
    torch.testing.assert_close(resized[0, 0, 0, 0, :3], torch.tensor([0.5, 0.5, 0.0]))
    torch.testing.assert_close(resized[1, 0, 0, 0, :3], torch.tensor([0.0, 0.0, 1.0]))


def test_rotary_relative_phase_and_gradients() -> None:
    torch.manual_seed(19)
    q = torch.randn(5, 2, 16, requires_grad=True)  # [N,H,D]
    k = torch.randn_like(q, requires_grad=True)  # [N,H,D]
    coords = torch.randn(5, 8)  # [N,8]
    angles = RopeRigPE(16)(coords).reshape(5, 16)  # [N,D]
    q_ = apply_rig_rotary(q, angles.cos(), angles.sin())  # [N,H,D]
    k_ = apply_rig_rotary(k, angles.cos(), angles.sin())  # [N,H,D]
    # Independent complex-number reference uses split-half pairs.
    q_complex = torch.complex(q[..., :8], q[..., 8:])  # [N,H,D/2]
    k_complex = torch.complex(k[..., :8], k[..., 8:])  # [N,H,D/2]
    phase = torch.exp(1j * angles[..., :8])[:, None, :]  # [N,1,D/2]
    expected = ((q_complex[0] * phase[0]).conj() * (k_complex[3] * phase[3])).real.sum(-1)  # [H]
    actual = (q_[0] * k_[3]).sum(-1)  # [H]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    expected_grad = torch.autograd.grad(expected.sum(), (q, k), retain_graph=True)  # tuple[[N,H,D]]
    actual_grad = torch.autograd.grad(actual.sum(), (q, k))  # tuple[[N,H,D]]
    for actual_item, expected_item in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_item, expected_item, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(apply_rig_rotary(q, torch.ones_like(angles), torch.zeros_like(angles)), q)


def test_even_pair_spread_keeps_every_component_on_the_prope_pairs() -> None:
    head_dim = 128
    encoding = RopeRigPE(head_dim // 2)
    # Half width halves each descriptor component's frequencies rather than dropping components.
    assert torch.count_nonzero(encoding.freq_matrix, dim=1).tolist() == [4] * 8
    coords = torch.randn(5, 8)  # [N,8]
    half = encoding(coords).reshape(5, head_dim // 2)  # [N,D/2]
    full = spread_over_even_pairs(half)  # [N,D]
    pairs = torch.arange(head_dim) % (head_dim // 2) % 2 == 0  # [D]
    assert torch.count_nonzero(full[:, ~pairs]) == 0
    for pair in range(head_dim // 4):
        torch.testing.assert_close(full[:, 2 * pair], half[:, pair], rtol=0, atol=0)
        torch.testing.assert_close(full[:, 2 * pair + head_dim // 2], half[:, pair], rtol=0, atol=0)
    with pytest.raises(ValueError, match="divisible by 4"):
        spread_over_even_pairs(torch.zeros(5, 6))


def test_even_pair_rigrope_beside_mrope_keeps_scores_relative() -> None:
    torch.manual_seed(23)
    head_dim = 32
    q, k = torch.randn(2, 1, 1, head_dim, dtype=torch.float64).unbind(0)  # each [1,1,D]
    pairs = torch.arange(head_dim) % (head_dim // 2) % 2 == 0  # [D]
    rig = spread_over_even_pairs(torch.randn(2, head_dim // 2, dtype=torch.float64))  # [2,D]
    mrope = torch.randn(2, head_dim // 2, dtype=torch.float64).repeat(1, 2)  # [2,D]

    def score(rig_angles: torch.Tensor, mrope_angles: torch.Tensor) -> torch.Tensor:
        def rotate(x: torch.Tensor, token: int) -> torch.Tensor:  # [1,1,D]
            rig_token, mrope_token = rig_angles[token : token + 1], mrope_angles[token : token + 1]  # each [1,D]
            geometry = apply_rig_rotary(x, rig_token.cos(), rig_token.sin())
            position = apply_mrope_rotary(x, mrope_token.cos(), mrope_token.sin())
            return torch.where(pairs, geometry, position)

        return (rotate(q, 0) * rotate(k, 1)).sum()

    rig_shift = spread_over_even_pairs(torch.randn(1, head_dim // 2, dtype=torch.float64))  # [1,D]
    mrope_shift = torch.randn(1, head_dim // 2, dtype=torch.float64).repeat(1, 2)  # [1,D]
    # Shifting both tokens by one offset in either encoding leaves the score unchanged.
    torch.testing.assert_close(score(rig + rig_shift, mrope + mrope_shift), score(rig, mrope), atol=1e-5, rtol=1e-5)
