# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise dense HF reloads and the collective path used by checkpoint export."""

import json
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from cosmos_framework.callbacks.hf_export import HFExportCallback
from cosmos_framework.utils.generator.lora import LoraInjectedLinear, inject_lora_pre_fsdp

pytestmark = pytest.mark.L0


def _model(*, lora=True, use_rslora=False, dtype=torch.float32):
    torch.manual_seed(123)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).to(dtype=dtype).eval()
    if lora:
        inject_lora_pre_fsdp(
            model,
            lora_rank=4,
            lora_alpha=8.0,
            lora_target_modules="q_proj,v_proj",
            lora_use_rslora=use_rslora,
            lora_precision="float32",
        )
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoraInjectedLinear):
                    module.lora_A.weight.normal_(std=0.1)
                    module.lora_B.weight.normal_(std=0.1)
    return model


def _wrapped(model):
    return SimpleNamespace(model=SimpleNamespace(model=model))


def _callback(dtype="float32"):
    callback = HFExportCallback(dtype=dtype)
    callback.config = SimpleNamespace(
        checkpoint=SimpleNamespace(
            hf_export=SimpleNamespace(upload_to_object_store=SimpleNamespace(enabled=False), hf_repo_id="")
        )
    )
    callback._MAX_SHARD_BYTES = 1024  # Exercise multiple shards and their index.
    return callback


@pytest.mark.parametrize("use_rslora", [False, True])
@pytest.mark.parametrize(
    "base_dtype,export_dtype", [(torch.float32, "float32"), (torch.bfloat16, "bfloat16"), (torch.bfloat16, "float32")]
)
@torch.no_grad()
def test_lora_export_reloads_as_dense_hf_model(tmp_path, use_rslora, base_dtype, export_dtype):
    model = _model(use_rslora=use_rslora, dtype=base_dtype)
    tokens = torch.tensor([[1, 4, 9, 2]])
    expected = model(tokens).logits
    # A zero-initialized adapter would make a broken base-only export pass.
    base_logits = _model(lora=False, dtype=base_dtype)(tokens).logits
    assert (expected - base_logits).abs().max() > 0.01
    before = {name: parameter.clone() for name, parameter in model.named_parameters()}
    identities = {name: (id(parameter), parameter.requires_grad) for name, parameter in model.named_parameters()}
    modes = {name: module.training for name, module in model.named_modules()}

    callback = _callback(export_dtype)
    chunks, manifest, total_size = callback._gather_weights(_wrapped(model))
    exported = {name: tensor.clone() for chunk in chunks for name, tensor in chunk.items()}
    assert len(chunks) > 1
    assert set(exported) == set(_model(lora=False).state_dict())
    assert total_size == sum(t.numel() * t.element_size() for t in exported.values())
    callback._do_save_and_upload(chunks, manifest, total_size, model.config, "", str(tmp_path), 1)
    with (tmp_path / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)
    assert index == {"metadata": {"total_size": total_size}, "weight_map": manifest}
    assert all((tmp_path / shard).is_file() for shard in manifest.values())
    reloaded, loading = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        local_files_only=True,
        dtype=getattr(torch, export_dtype),
        attn_implementation="eager",
        output_loading_info=True,
    )
    assert not loading["unexpected_keys"]
    assert not loading["missing_keys"]
    assert not loading["mismatched_keys"]
    tolerance = 2e-3 if base_dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(reloaded(tokens).logits.float(), expected.float(), atol=tolerance, rtol=0.02)

    # Export must not merge the live adapter a second time or change trainability.
    again, _, _ = callback._gather_weights(_wrapped(model))
    for chunk in again:
        for name, tensor in chunk.items():
            torch.testing.assert_close(tensor, exported[name], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        assert (id(parameter), parameter.requires_grad) == identities[name]
    assert {name: module.training for name, module in model.named_modules()} == modes


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@torch.no_grad()
def test_dense_export_preserves_weights_and_reload(tmp_path, dtype):
    model = _model(lora=False, dtype=getattr(torch, dtype))
    callback = _callback(dtype)
    chunks, manifest, total_size = callback._gather_weights(_wrapped(model))
    for chunk in chunks:
        for name, tensor in chunk.items():
            torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
    callback._do_save_and_upload(chunks, manifest, total_size, model.config, "", str(tmp_path), 1)
    reloaded = AutoModelForCausalLM.from_pretrained(
        tmp_path, local_files_only=True, dtype=getattr(torch, dtype), attn_implementation="eager"
    )
    tokens = torch.tensor([[1, 4, 9, 2]])
    torch.testing.assert_close(reloaded(tokens).logits, model(tokens).logits, rtol=0, atol=0)


@pytest.mark.parametrize("wrappers", [(), ("_orig_mod",), ("_checkpoint_wrapped_module", "_orig_mod")])
@torch.no_grad()
def test_root_lora_and_wrapper_names(wrappers):
    layer = LoraInjectedLinear(nn.Linear(4, 3), rank=2, alpha=4, use_rslora=True)
    network = layer
    for name in wrappers:
        network = nn.ModuleDict({name: network})
    chunks, manifest, _ = _callback()._gather_weights(_wrapped(network))
    exported = {name: tensor for chunk in chunks for name, tensor in chunk.items()}
    assert set(manifest) == {"weight", "bias"}
    dense = nn.Linear(4, 3)
    dense.load_state_dict(exported, strict=True)
    inputs = torch.randn(2, 4)
    torch.testing.assert_close(dense(inputs), layer(inputs))


@torch.no_grad()
def test_cpu_export_owns_snapshot_storage():
    model = nn.Linear(4, 3)
    chunks, _, _ = _callback()._gather_weights(_wrapped(model))
    expected = model.weight.clone()
    model.weight.add_(1)
    torch.testing.assert_close(chunks[0]["weight"], expected, rtol=0, atol=0)


def test_ambiguous_tied_lora_weight_is_rejected():
    base = nn.Linear(4, 3)
    network = nn.ModuleDict({"base": base, "adapted": LoraInjectedLinear(base, rank=2, alpha=4)})
    with pytest.raises(ValueError, match="unmerged LoRA weights=.*adapted.weight"):
        _callback()._gather_weights(_wrapped(network))


def test_wrapper_name_collision_is_rejected():
    network = nn.ModuleDict({"layer": nn.Linear(4, 3), "_orig_mod": nn.ModuleDict({"layer": nn.Linear(4, 3)})})
    with pytest.raises(ValueError, match="duplicate keys=.*layer.weight"):
        _callback()._gather_weights(_wrapped(network))


def test_similar_names_are_not_removed():
    network = nn.ModuleDict({"my_orig_mod": nn.Linear(4, 3), "lora_A": nn.Linear(4, 3)})
    _, manifest, _ = _callback()._gather_weights(_wrapped(network))
    assert set(manifest) == set(network.state_dict())


@pytest.fixture(scope="module")
def fsdp_group():
    """Keep the process group alive across cases, including DTensor's cached ops."""
    if os.environ.get("WORLD_SIZE") != "2":
        pytest.skip("requires torchrun --nproc_per_node=2 for actual FSDP/NCCL collectives")
    rank = int(os.environ["RANK"])
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=60))
    try:
        yield rank, device
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.GPU
@pytest.mark.parametrize("lora", [False, True])
def test_fsdp_export_gathers_once_in_rank_order_and_reloads(tmp_path, lora, fsdp_group):
    """Run with torchrun --nproc_per_node=2, like the other distributed CI cases."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    rank, device = fsdp_group
    output = str(tmp_path / "export")
    model = _model(lora=lora, use_rslora=True).cuda()
    tokens = torch.tensor([[1, 4, 9, 2]], device=device)
    with torch.no_grad():
        expected = model(tokens).logits.cpu()
    fully_shard(model, mesh=init_device_mesh("cuda", (2,)))
    parameters = dict(model.named_parameters())
    assert all(isinstance(parameter, DTensor) for parameter in parameters.values())
    before = {name: parameter.full_tensor().detach().clone() for name, parameter in parameters.items()}
    names = {id(parameter): name for name, parameter in parameters.items()}
    order = []
    full_tensor = DTensor.full_tensor

    def recording_full_tensor(tensor, *args, **kwargs):
        order.append(names[id(tensor)])
        return full_tensor(tensor, *args, **kwargs)

    callback = _callback()
    DTensor.full_tensor = recording_full_tensor
    try:
        chunks, manifest, total_size = callback._gather_weights(_wrapped(model))
    finally:
        DTensor.full_tensor = full_tensor
    orders = [None, None]
    torch.distributed.all_gather_object(orders, order)
    assert orders[0] == orders[1]
    assert len(order) == len(parameters) and set(order) == set(parameters)
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]
        torch.testing.assert_close(parameter.full_tensor(), before[name], rtol=0, atol=0)
    if rank == 0:
        callback._do_save_and_upload(chunks, manifest, total_size, model.config, "", output, 1)
        dense, info = AutoModelForCausalLM.from_pretrained(
            output,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        assert not info["unexpected_keys"] and not info["missing_keys"]
        with torch.no_grad():
            torch.testing.assert_close(dense(tokens.cpu()).logits, expected, atol=2e-6, rtol=2e-5)
    else:
        assert (chunks, manifest, total_size) == ([], {}, 0)
    torch.distributed.barrier()
