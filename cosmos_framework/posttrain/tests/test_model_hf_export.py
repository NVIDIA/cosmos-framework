# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for HFModel.export (Task 10): HF safetensors round-trip.

Single-device (CPU) here: build a tiny GPT2 via HFModel, export it, reload with
``AutoModelForCausalLM.from_pretrained`` and assert bit-exact weights + config.json.
The sharded (FSDP) gather path is exercised on the cluster GPU suite, not here.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_export_round_trip(tmp_path):
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.model.hf import HFModel

    # Save a tiny GPT2 so HFModel can AutoConfig + load it back.
    src = tmp_path / "src"
    GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=256, n_positions=32)).save_pretrained(src)

    m = HFModel(str(src), dtype=torch.float32, attn_implementation="eager")
    m.setup(None)  # single device (no ParallelDims): materialize + tie + load

    out = tmp_path / "out"
    m.export(str(out))  # not distributed -> rank-0 write path, no barrier

    # HF wrote a config + safetensors that re-load into the standard AutoClass.
    assert (out / "config.json").exists()
    ref = AutoModelForCausalLM.from_pretrained(str(out))

    src_sd = m.net.state_dict()
    ref_sd = ref.state_dict()
    assert set(src_sd) == set(ref_sd)
    for k, v in src_sd.items():
        assert torch.equal(v.cpu(), ref_sd[k].cpu()), f"mismatch on {k}"
