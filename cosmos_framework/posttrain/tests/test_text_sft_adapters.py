# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Non-Qwen text templates train through the existing objective and LLMTrainer."""

import copy

import pytest
import torch
import transformers

from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
from cosmos_framework.posttrain.model.adapters import assert_trainability
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition
from cosmos_framework.posttrain.tests.test_data_packer_text_families import target, tokenizer


@pytest.mark.parametrize(
    "family,format",
    [("llama", "chatml"), ("mistral", "mistral_inst"), ("gemma", "gemma_turns"), ("gemma2", "gemma_turns")],
)
@pytest.mark.parametrize("method,rslora", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_template_objective_native_trainer_updates_and_exact_resume(
    tmp_path, monkeypatch, family, format, method, rslora
):
    torch.manual_seed(37)
    tok = tokenizer(format)
    base = tmp_path / "base"
    cfg = transformers.AutoConfig.for_model(
        family,
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
        max_position_embeddings=512,
    )
    source = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
    source.save_pretrained(base)
    tok.save_pretrained(base)
    adapters = AdapterConfig(
        components=[
            AdapterComponentConfig(
                targets=["model.layers.*.self_attn.q_proj", "model.layers.*.self_attn.v_proj"],
                rank=2,
                method=method,
                use_rslora=rslora,
            )
        ]
    )
    model_config = HFModelConfig(name_or_path=str(base), dtype="float32", attn_impl="sdpa", grad_ckpt=True)
    packer = build_sft_packer(str(base), max_length=512)
    row = Trajectory(id="text", status=Status.COMPLETED, transitions=[Transition(obs=ObsBundle(), targets=target())])

    def trainer(resume=False):
        policy = build_hf_model(model_config, adapters=adapters, adapter_resume=resume)
        opt = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.003)
        return LLMTrainer(
            policy,
            make_objective(packer),
            opt,
            checkpoint_format="consolidated",
            save_async=False,
            require_gradients=True,
        )

    original = trainer()
    before = copy.deepcopy(original.model.net.state_dict())
    for _ in range(3):
        report = original.train([row])
        assert torch.isfinite(torch.as_tensor(report["train/loss_avg"]))
    assert_trainability(original.model.net, original.opt)
    for name, param in original.model.net.named_parameters():
        assert torch.equal(before[name], param) is not param.requires_grad, name
    checkpoint = str(tmp_path / "checkpoint")
    original.save(checkpoint, data_state={"next_row": 3})
    original.train([row])
    expected = copy.deepcopy(original.model.net.state_dict())
    expected_opt = copy.deepcopy(original.opt.state_dict())
    monkeypatch.setattr(torch, "svd_lowrank", lambda *a, **kw: pytest.fail("resume repeated PiSSA decomposition"))
    restored = trainer(resume=True)
    assert restored.load(checkpoint) == 3
    restored.train([row])
    for name, value in restored.model.net.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    for key, values in restored.opt.state_dict()["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, expected_opt["state"][key][name], atol=0, rtol=0)
