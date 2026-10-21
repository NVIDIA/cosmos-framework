# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""WeightMapper.export_state materializes full tensors: DTensor -> full_tensor(); plain tensor
unchanged; both detached. This is the SyncPlan value source (broadcast ships whole tensors).

`export_state`, `manifest` and `export_bucket` share `_full`, so a gather or pad change lands on
all three at once; what is pinned here is the gather and the `.net` unwrap."""

from __future__ import annotations

import pytest
import torch


class _FakeDTensor:
    def __init__(self, full):
        self._full = full

    def full_tensor(self):
        return self._full

    def detach(self):
        return self  # export detaches AFTER materializing; identity keeps the test simple


class _M:
    def __init__(self, sd):
        self._sd = sd

    def state_dict(self):
        return self._sd


@pytest.mark.L0
@pytest.mark.CPU
def test_export_state_materializes_full_tensor():
    from cosmos_framework.posttrain.comm.weight_transfer import WeightMapper

    full = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    m = _M({"w": _FakeDTensor(full), "b": torch.ones(3)})
    out = WeightMapper().export_state(m)
    assert torch.equal(out["w"], full)  # DTensor gathered to full
    assert torch.equal(out["b"], torch.ones(3))  # plain tensor passes through
    assert not out["w"].requires_grad


@pytest.mark.L0
@pytest.mark.CPU
def test_full_state_dict_unwraps_net_wrapper():
    # A wrapped model (LLMTrainer's FSDP HFModel holds the module under .net) must export the RAW
    # module's un-prefixed keys, matching the raw module the rollout loads into (else keys mismatch).
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.weight_transfer import WeightMapper

    raw = nn.Linear(4, 3)

    class _Wrapper:
        def __init__(self, net):
            self.net = net

    out = WeightMapper().export_state(_Wrapper(raw))
    assert set(out) == {"weight", "bias"}  # raw keys, NOT net.weight/net.bias
    assert torch.equal(out["weight"], raw.weight.detach())


@pytest.mark.L0
@pytest.mark.CPU
def test_padded_rows_are_trimmed_in_both_the_manifest_and_the_export():
    """A model that pads for its own sharding ships the UNPADDED width, in both places.

    FSDP pads a narrow Linear so no rank gets an empty shard; `state_dict` then reports the padded
    shape while the consumer holds the unpadded one. The manifest sizes the staging buffer and the
    export fills it, so trimming one without the other is worse than trimming neither -- it sizes
    for `rows` and writes `rows + pad`. Both are asserted here for that reason.
    """
    from cosmos_framework.posttrain.comm.weight_transfer import model_manifest, resolve_weight_mapper

    class _Padded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(12.0).reshape(4, 3))
            self.unpadded_rows = {"weight": 2}  # rows 2..3 are sharding pad

    model = _Padded()
    assert model_manifest(model) == [("weight", (2, 3), str(model.weight.dtype))]

    exported = resolve_weight_mapper(model).export_bucket(model, [("weight", (2, 3), str(model.weight.dtype))])
    assert torch.equal(exported["weight"], torch.arange(6.0).reshape(2, 3))


@pytest.mark.L0
@pytest.mark.CPU
def test_a_model_that_declares_no_padding_is_untouched():
    """The pad seam is opt-in: absent `unpadded_rows` means every row is real."""
    from cosmos_framework.posttrain.comm.weight_transfer import model_manifest

    model = torch.nn.Linear(3, 4, bias=False)
    assert model_manifest(model) == [("weight", (4, 3), str(model.weight.dtype))]
