# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from cosmos_framework.posttrain.model.specs import LoadPlan
from cosmos_framework.posttrain.model.weight_loader import load_weights
from cosmos_framework.posttrain.tests.helpers.tiny_decoder import TinyDecoder


@pytest.mark.L0
@pytest.mark.CPU
def test_single_rank_load_exact(tmp_path):
    from safetensors.torch import save_file

    src = TinyDecoder()
    sd = {k: v.contiguous() for k, v in src.state_dict().items()}
    save_file(sd, str(tmp_path / "model.safetensors"))
    dst = TinyDecoder()  # different random init
    assert not torch.equal(next(iter(dst.state_dict().values())), next(iter(sd.values())))
    loaded = load_weights(dst, str(tmp_path), LoadPlan(), pdims=None)
    assert set(loaded) == set(sd)
    for k in sd:
        assert torch.equal(dst.state_dict()[k], sd[k])


@pytest.mark.L0
@pytest.mark.CPU
def test_skip_pattern_tolerated(tmp_path):
    from safetensors.torch import save_file

    src = TinyDecoder()
    sd = {k: v for k, v in src.state_dict().items() if not k.startswith("head")}  # head absent
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / "model.safetensors"))
    dst = TinyDecoder()
    loaded = load_weights(dst, str(tmp_path), LoadPlan(skip_patterns=(r"^head\.",)), pdims=None)
    assert not any(k.startswith("head") for k in loaded)  # head skipped, no error


@pytest.mark.L0
@pytest.mark.CPU
def test_skip_pattern_present_key_not_copied(tmp_path):
    """skip-copy meaning: a skipped key PRESENT in the ckpt must not overwrite the model."""
    from safetensors.torch import save_file

    src = TinyDecoder()
    sd = {k: v.contiguous() for k, v in src.state_dict().items()}  # head IS present in ckpt
    save_file(sd, str(tmp_path / "model.safetensors"))
    dst = TinyDecoder()
    sentinel = {k: v.clone() for k, v in dst.state_dict().items() if k.startswith("head")}
    loaded = load_weights(dst, str(tmp_path), LoadPlan(skip_patterns=(r"^head\.",)), pdims=None)
    assert not any(k.startswith("head") for k in loaded)  # head present but skip-copied
    for k, original in sentinel.items():
        assert torch.equal(dst.state_dict()[k], original)  # untouched, not the ckpt value
        assert not torch.equal(dst.state_dict()[k], sd[k])  # and definitely not overwritten


@pytest.mark.L0
@pytest.mark.CPU
def test_rename_maps_checkpoint_keys(tmp_path):
    """rename ckpt-key -> model-key: checkpoint stored under a prefixed namespace."""
    from safetensors.torch import save_file

    src = TinyDecoder()
    # Store the checkpoint with a "backbone." prefix on every key.
    sd = {f"backbone.{k}": v.contiguous() for k, v in src.state_dict().items()}
    save_file(sd, str(tmp_path / "model.safetensors"))
    dst = TinyDecoder()
    loaded = load_weights(
        dst,
        str(tmp_path),
        LoadPlan(rename=lambda k: k.replace("backbone.", "", 1)),
        pdims=None,
    )
    ref = src.state_dict()
    assert set(loaded) == set(ref)
    for k in ref:
        assert torch.equal(dst.state_dict()[k], ref[k])


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_required_key_raises(tmp_path):
    """A model key absent from the ckpt AND not skipped must raise (completeness)."""
    from safetensors.torch import save_file

    src = TinyDecoder()
    sd = {k: v for k, v in src.state_dict().items() if not k.startswith("head")}  # head absent
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / "model.safetensors"))
    dst = TinyDecoder()
    with pytest.raises(ValueError):
        load_weights(dst, str(tmp_path), LoadPlan(), pdims=None)  # no skip -> head is required


@pytest.mark.L1
@pytest.mark.GPU
def test_sharded_load_exact(tmp_path):
    """2-GPU FSDP striped load: each rank fills its dp_shard slice; gathered == source."""
    import torch.distributed as dist
    from safetensors.torch import save_file
    from torch.distributed.tensor import DTensor

    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.model.parallelize import parallelize
    from cosmos_framework.posttrain.model.specs import ShardPlan
    from cosmos_framework.posttrain.tests.helpers.tiny_decoder import Block

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("needs >=2 GPUs")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # Build the source on CPU, serialize to a shared safetensors file (rank 0 writes).
    # Seed so every rank builds the IDENTICAL source (each rank compares against its own
    # ref_sd below). Under torchrun each rank gets its own tmp_path, so broadcast rank 0's
    # dir string and have every rank use that one shared path (single node -> shared /tmp).
    import os

    torch.manual_seed(0)
    src = TinyDecoder()
    ref_sd = {k: v.contiguous() for k, v in src.state_dict().items()}
    _obj = [str(tmp_path / "ckpt")] if rank == 0 else [None]
    dist.broadcast_object_list(_obj, src=0)
    ckpt_dir = _obj[0]
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        save_file(ref_sd, os.path.join(ckpt_dir, "model.safetensors"))
    dist.barrier()

    # Build + FSDP-shard a fresh (differently-initialized) model on GPU.
    m = TinyDecoder().cuda()
    pd = ParallelDims(world_size=world, dp_shard=world)
    pd.build_device_mesh("cuda")
    parallelize(m, ShardPlan((Block,), torch.float32, torch.float32), pd)
    assert any(isinstance(p, DTensor) for p in m.parameters())

    loaded = load_weights(m, str(ckpt_dir), LoadPlan(), pdims=pd)
    assert set(loaded) == set(ref_sd)

    # Gather every (possibly sharded) parameter and compare against the source.
    for name, param in m.state_dict().items():
        full = param.full_tensor() if isinstance(param, DTensor) else param
        assert torch.equal(full.cpu(), ref_sd[name].to(full.dtype)), f"mismatch for {name}"
