# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real public SFT construction, relocated portable import, and subsequent resume."""

import json
import shutil
from copy import deepcopy

import pytest
import torch

from cosmos_framework.posttrain.checkpoint.adapters import load_adapter_artifact, save_adapter_artifact
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
from cosmos_framework.posttrain.entrypoints.sft import build_trainer, validate_config
from cosmos_framework.posttrain.roles.env.sft import build_sft_env
from cosmos_framework.posttrain.tests.test_adapter_families import text_snapshot
from cosmos_framework.utils.config import load_config

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("strict", [False, True])
def test_relocated_portable_preserves_live_identity_and_exact_resume(tmp_path, monkeypatch, strict):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source, relocated = tmp_path / "source", tmp_path / "relocated"
    text_snapshot(source, "qwen3")
    shutil.copytree(source, relocated)
    data_path = tmp_path / "data"
    data_path.mkdir()
    (data_path / "train.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "id": str(i),
                    "messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": "Answer"}],
                }
            )
            for i in range(4)
        )
        + "\n"
    )
    monkeypatch.setenv("SFT_DATA_ROOT", str(data_path))
    recipe = "cosmos_framework/posttrain/configs/sft_family_adapters.py"

    def build(base, *, warm="", resume="off", max_length=256, strict_mode=strict):
        monkeypatch.setenv("SFT_MODEL_PATH", str(base))
        config = load_config(
            recipe,
            [
                "--",
                "rl.validation.enabled=false",
                "rl.grad_accum=1",
                "rl.max_steps=4",
                "rl.master_dtype=float32",
                "rl.param_dtype=float32",
                f"rl.resume={resume}",
                f"rl.max_length={max_length}",
            ],
        )
        config.rl.adapters.warm_start = warm
        config.rl.deterministic = True
        config.rl.deterministic_strict = strict_mode
        rl = validate_config(config)
        data = build_sft_env(rl.data, rl.seed)
        lifecycle = CheckpointLifecycle.from_config(config, rl, run_out_dir=str(tmp_path / "run"))
        trainer = build_trainer(config, rl, data, None, lifecycle)
        return trainer, data

    original, data = build(source)
    original.train(data.get_samples(1))
    portable = tmp_path / "portable"
    save_adapter_artifact(original.model.net, str(portable))
    artifact_before = {p.name: p.read_bytes() for p in portable.iterdir() if p.is_file()}
    old_identity = deepcopy(original.model.net._posttrain_adapter_identity)
    # A direct loader caller cannot substitute a genuine preprocessing mismatch
    # using the relocation option, even with the original tokenizer.
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer

    tokenizer = build_sft_packer(str(source), max_length=256).tokenizer
    original.model.net._posttrain_adapter_identity["media"]["max_length"] = 128
    with pytest.raises(ValueError, match="media compatibility mismatch"):
        load_adapter_artifact(original.model.net, str(portable), compatibility_tokenizer=tokenizer)
    original.model.net._posttrain_adapter_identity = deepcopy(old_identity)
    warmed, warmed_data = build(relocated, warm=str(portable))
    live_identity = deepcopy(warmed.model.net._posttrain_adapter_identity)
    assert live_identity["media"] != old_identity["media"]
    assert live_identity["base"] == old_identity["base"]
    assert live_identity["processor"] == old_identity["processor"]
    tokenizer_identity = live_identity["media"]["packer"]["tokenizer"]
    assert tokenizer_identity["settings"]["name_or_path"] == str(relocated)
    for name, value in original.model.net.state_dict().items():
        torch.testing.assert_close(value, warmed.model.net.state_dict()[name], atol=0, rtol=0)
    warmed.train(warmed_data.get_samples(1))
    checkpoint = tmp_path / "iter_000001"
    warmed.save(str(checkpoint), data_state=warmed_data.state_dict())
    warmed.finalize_checkpoints()
    expected_loss = warmed.train(warmed_data.get_samples(1))["train/loss_avg"]
    resumed, resumed_data = build(relocated, resume=str(checkpoint))
    assert resumed.model.net._posttrain_adapter_identity == live_identity
    actual_loss = resumed.train(resumed_data.get_samples(1))["train/loss_avg"]
    assert actual_loss == expected_loss
    for name, value in warmed.model.net.state_dict().items():
        torch.testing.assert_close(value, resumed.model.net.state_dict()[name], atol=0, rtol=0)
    with pytest.raises((ValueError, RuntimeError), match="objective exact-resume identity differs"):
        build(relocated, resume=str(checkpoint), strict_mode=not strict)
    assert {p.name: p.read_bytes() for p in portable.iterdir() if p.is_file()} == artifact_before
    with pytest.raises((ValueError, RuntimeError), match="media compatibility mismatch"):
        build(relocated, warm=str(portable), max_length=128)
    # Equal paths/labels cannot conceal a changed immutable base.
    from safetensors.torch import load_file, save_file

    weights = relocated / "model.safetensors"
    state = load_file(str(weights))
    first = next(iter(state))
    state[first].view(-1)[0] += 1
    save_file(state, str(weights))
    with pytest.raises((ValueError, RuntimeError), match="compatibility mismatch.*base"):
        build(relocated, warm=str(portable))
