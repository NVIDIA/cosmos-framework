# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for HFModel (LLM branch): single-device CPU forward + multi-GPU FSDP setup."""

from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_model_llm_single_device_forward(tmp_path):
    import torch.nn as nn
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.model.hf import HFModel

    # Save a tiny GPT2 to disk so HFModel can AutoConfig + load it back.
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=256, n_positions=32)
    GPT2LMHeadModel(cfg).save_pretrained(tmp_path)

    m = HFModel(str(tmp_path), dtype=torch.float32, attn_implementation="eager")
    # block_types resolved from GPT2's _no_split_modules == ["GPT2Block"].
    assert any(t.__name__ == "GPT2Block" for t in m.shard_plan.block_types)

    m.setup(None)  # single device (no ParallelDims)
    assert isinstance(m.net, nn.Module)

    dev = next(m.net.parameters()).device  # materialized to cpu or cuda per host
    out = m(input_ids=torch.zeros(1, 4, dtype=torch.long, device=dev))
    assert out.shape == (1, 4, cfg.vocab_size)


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_model_llm_load_matches_source_values(tmp_path):
    """HFModel.setup(None) must load the EXACT checkpoint VALUES, not merely run: compare every
    param/buffer in the loaded net against the source GPT2 state dict (bit-for-bit, fp32). Guards
    the real-HFModel load path (HF key space, meta materialize, tied lm_head<->wte) which the
    forward-shape and export round-trip tests do NOT value-check."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.model.hf import HFModel

    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=256, n_positions=32)
    src_model = GPT2LMHeadModel(cfg)  # random init -> the values must survive the round-trip
    ref_sd = {k: v.detach().clone() for k, v in src_model.state_dict().items()}
    src_model.save_pretrained(tmp_path)

    m = HFModel(str(tmp_path), dtype=torch.float32, attn_implementation="eager")
    m.setup(None)  # single device: materialize + tie + load
    got = m.net.state_dict()
    assert set(got) == set(ref_sd), set(got).symmetric_difference(ref_sd)
    for k in ref_sd:
        assert torch.equal(got[k].cpu(), ref_sd[k].cpu()), f"loaded value mismatch on {k}"
    # The tied output head must be re-tied to the LOADED embedding, not left at meta/init.
    assert torch.equal(got["lm_head.weight"].cpu(), got["transformer.wte.weight"].cpu())


@pytest.mark.L1
@pytest.mark.GPU
def test_hf_model_llm_fsdp_load_matches_source_values():
    """2-GPU: the FSDP2-sharded HFModel load reproduces the source checkpoint EXACTLY. Gather
    each (possibly DTensor) param via full_tensor() and compare to the GPT2 source — the sharded
    analogue of the single-device value check, through the REAL HF key space (incl. tied lm_head
    and dim-0 striping)."""
    import tempfile

    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.model.hf import HFModel
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("needs >=2 GPUs")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=256, n_positions=32)
    # Rank 0 writes the checkpoint; broadcast the shared dir (each rank's tmp differs under torchrun).
    ckpt = [tempfile.mkdtemp(prefix="hf_fsdp_val_") if rank == 0 else None]
    dist.broadcast_object_list(ckpt, src=0)
    ckpt_dir = ckpt[0]
    if rank == 0:
        GPT2LMHeadModel(cfg).save_pretrained(ckpt_dir)
    dist.barrier()

    # Every rank rebuilds the SAME reference from the shared checkpoint (identical bytes -> identical
    # ref on every rank), so the per-rank comparison against gathered full tensors is well-defined.
    ref_sd = {k: v.detach().cpu() for k, v in AutoModelForCausalLM.from_pretrained(ckpt_dir).state_dict().items()}

    m = HFModel(ckpt_dir, dtype=torch.float32, attn_implementation="eager")
    pd = ParallelDims(world_size=world, dp_shard=world)
    pd.build_device_mesh("cuda")
    m.setup(pd)
    assert any(isinstance(p, DTensor) for p in m.net.parameters())

    got = m.net.state_dict()
    assert set(got) == set(ref_sd), set(got).symmetric_difference(ref_sd)
    # full_tensor() is a collective -> every rank must iterate in the SAME (deterministic) order.
    for name in sorted(got):
        param = got[name]
        full = param.full_tensor() if isinstance(param, DTensor) else param
        assert torch.equal(full.cpu(), ref_sd[name].to(full.dtype)), f"sharded load mismatch on {name}"

    dist.barrier()
    if rank == 0:
        import shutil

        shutil.rmtree(ckpt_dir, ignore_errors=True)


@pytest.mark.L1
@pytest.mark.GPU
def test_hf_model_llm_fsdp_setup_shards_and_loads():
    """2-GPU: setup(pdims) FSDP-shards params to DTensor and loads pretrained weights."""
    import tempfile

    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.model.hf import HFModel
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("needs >=2 GPUs")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # n_embd / vocab_size divisible by the dp_shard size for even dim-0 striping.
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=256, n_positions=32)
    # Under torchrun each rank runs its own pytest, so a `tmp_path` fixture differs per
    # rank. Rank 0 writes the checkpoint and broadcasts the shared path to all ranks.
    ckpt = [tempfile.mkdtemp(prefix="hf_fsdp_test_") if rank == 0 else None]
    dist.broadcast_object_list(ckpt, src=0)
    ckpt_dir = ckpt[0]
    if rank == 0:
        GPT2LMHeadModel(cfg).save_pretrained(ckpt_dir)
    dist.barrier()

    m = HFModel(ckpt_dir, dtype=torch.bfloat16, attn_implementation="eager")
    pd = ParallelDims(world_size=world, dp_shard=world)
    pd.build_device_mesh("cuda")
    m.setup(pd)

    # Params are FSDP2 DTensors after setup, and weights are loaded (no leftover meta).
    assert any(isinstance(p, DTensor) for p in m.net.parameters())
    assert all(p.device.type != "meta" for p in m.net.parameters())

    ids = torch.zeros(1, 4, dtype=torch.long, device="cuda")
    out = m(input_ids=ids)
    assert out.shape[-1] == cfg.vocab_size
    assert torch.isfinite(out.float()).all()

    dist.barrier()
    if rank == 0:
        import shutil

        shutil.rmtree(ckpt_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# VLM parity (Task 9): ported helpers + AutoClass selection / plan construction.
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_tensor_names_to_skip_for():
    """Ported RADIO buffer skip table: NemotronH_Nano_VL_V2 -> 3 patterns, others -> []."""
    from cosmos_framework.posttrain.model.hf import _tensor_names_to_skip_for

    skips = _tensor_names_to_skip_for("NemotronH_Nano_VL_V2")
    assert skips == [
        r"vision_model\.radio_model\.summary_idxs",
        r"vision_model\.radio_model\.input_conditioner\.norm_mean",
        r"vision_model\.radio_model\.input_conditioner\.norm_std",
    ]
    assert _tensor_names_to_skip_for("qwen3_vl") == []
    assert _tensor_names_to_skip_for("gpt2") == []


@pytest.mark.L0
@pytest.mark.CPU
def test_is_moe_vlm_detection():
    """Ported MoE-VLM predicate: config num_experts OR a `.mlp.experts.` key -> True."""
    import torch.nn as nn

    from cosmos_framework.posttrain.model.hf import _is_moe_vlm

    class _Cfg:
        pass

    class _Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.lin = nn.Linear(4, 4)

    class _MoEByConfig(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.config.num_experts = 8
            self.lin = nn.Linear(4, 4)

    class _MoEByKey(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            # nested so the state-dict key is `layer.mlp.experts.weight`, which contains the
            # `.mlp.experts.` substring the predicate matches (a real MoE key is always nested,
            # e.g. model.layers.0.mlp.experts.0.gate_proj.weight).
            self.layer = nn.Module()
            self.layer.mlp = nn.Module()
            self.layer.mlp.experts = nn.Linear(4, 4)

    assert _is_moe_vlm(_Dense()) is False
    assert _is_moe_vlm(_MoEByConfig()) is True
    assert _is_moe_vlm(_MoEByKey()) is True


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_model_vlm_autoclass_selection(tmp_path):
    """Construct a tiny dense Qwen3-VL from_config (no weight download) and assert the VLM
    AutoClass path is taken: net is the ImageTextToText conditional-generation class and the
    shard/load plans build. Skips cleanly when the local transformers cannot build a tiny
    Qwen3-VL (needs qwen3_vl registered + a compatible monkey_patch import); a real dense VLM
    is version-sensitive and we must not download large weights on CI."""
    pytest.importorskip("transformers.models.qwen3_vl", reason="qwen3_vl not available in this transformers")

    from cosmos_framework.posttrain.model.hf import HFModel

    try:
        from transformers import Qwen3VLConfig
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig

        text_cfg = Qwen3VLTextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=256,
            head_dim=16,
        )
        vision_cfg = Qwen3VLVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            depth=2,
            num_heads=2,
            out_hidden_size=32,
        )
        cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)
        cfg.save_pretrained(tmp_path)
    except Exception as exc:  # noqa: BLE001 — intentional: any build failure -> documented skip
        pytest.skip(f"cannot build a tiny Qwen3-VL config on this transformers: {exc!r}")

    try:
        m = HFModel(str(tmp_path), dtype=torch.float32, attn_implementation="eager", trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001 — from_config / monkey_patch env-sensitive -> skip
        pytest.skip(f"cannot construct tiny Qwen3-VL HFModel on this transformers: {exc!r}")

    # VLM branch selected AutoModelForImageTextToText -> the full conditional-generation
    # class (exposes .logits), NOT the backbone-only Qwen3VLModel (which AutoModel returns).
    net_cls = type(m.net).__name__
    assert net_cls.endswith("ForConditionalGeneration"), net_cls
    assert net_cls != "Qwen3VLModel"
    assert m.is_vlm is True
    # Plans are declared (shard_plan may legitimately be empty if block types are unresolved,
    # but the attribute must exist and load_plan must carry the tied/VLM skip tuple).
    assert hasattr(m, "shard_plan")
    assert isinstance(m.load_plan.skip_patterns, tuple)


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_vlm_forward_preserves_processor_image_sizes() -> None:
    """Architecture-specific processor outputs must reach the corresponding HF model forward."""
    from types import SimpleNamespace

    import torch.nn as nn

    from cosmos_framework.posttrain.model.hf import HFModel

    class _CaptureModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight: nn.Parameter = nn.Parameter(torch.ones(()))  # []
            self.seen: dict[str, object] = {}

        def forward(self, **kwargs: object) -> SimpleNamespace:
            self.seen = dict(kwargs)
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            logits = torch.zeros((*input_ids.shape, 4)) + self.weight  # [B,L,V]
            return SimpleNamespace(logits=logits)

    model = HFModel.__new__(HFModel)
    nn.Module.__init__(model)
    model.compute_ctx = nullcontext()  # what `__init__` builds for mode "fsdp"
    capture = _CaptureModel()
    model.model = capture
    model.hf_config = SimpleNamespace(model_type="generic_vlm")
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)  # [B,L]
    image_sizes = torch.tensor([[8, 6]], dtype=torch.long)  # [N,2]

    logits = model.forward(input_ids=input_ids, image_sizes=image_sizes)  # [B,L,V]

    assert tuple(logits.shape) == (1, 2, 4)
    assert capture.seen["image_sizes"] is image_sizes


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_vlm_policy_preparation_moves_media_and_removes_objective_bookkeeping() -> None:
    """The converged preparation phase must produce a complete model-ready batch."""
    from types import SimpleNamespace

    import torch.nn as nn

    from cosmos_framework.posttrain.algorithm.primitives.policy_forward import prepare_policy_capability
    from cosmos_framework.posttrain.model.hf import HFModel

    class _MetaModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight: nn.Parameter = nn.Parameter(torch.empty((), device="meta"))  # []

        def forward(self, **_kwargs: object) -> None:
            raise AssertionError("preparation must not invoke model forward")

    model = HFModel.__new__(HFModel)
    nn.Module.__init__(model)
    model.compute_ctx = nullcontext()  # what `__init__` builds for mode "fsdp"
    model.model = _MetaModel()
    model.hf_config = SimpleNamespace(model_type="generic_vlm")
    batch = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),  # [B,L]
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),  # [B,L]
        "pixel_values": torch.ones((1, 3, 2, 2)),  # [N,C,H,W]
        "logprob_masks": torch.tensor([[False, True]]),  # [B,L]
    }

    prepared = prepare_policy_capability(model, batch)

    assert set(prepared) == {"input_ids", "attention_mask", "pixel_values"}
    assert all(value.device.type == "meta" for value in prepared.values())


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_vlm_policy_rejects_a_rank_local_batch_error_before_model_forward() -> None:
    """Malformed rank-local input must fail inside the converged preparation phase."""
    from types import SimpleNamespace

    import torch.nn as nn

    from cosmos_framework.posttrain.algorithm.primitives.policy_forward import prepare_policy_capability
    from cosmos_framework.posttrain.model.hf import HFModel

    class _CountingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight: nn.Parameter = nn.Parameter(torch.ones(()))  # []
            self.forward_calls: int = 0

        def forward(self, **_kwargs: object) -> None:
            self.forward_calls += 1

    model = HFModel.__new__(HFModel)
    nn.Module.__init__(model)
    model.compute_ctx = nullcontext()  # what `__init__` builds for mode "fsdp"
    capture = _CountingModel()
    model.model = capture
    model.hf_config = SimpleNamespace(model_type="generic_vlm")

    with pytest.raises(ValueError, match="missing input_ids"):
        prepare_policy_capability(model, {"pixel_values": torch.ones((1, 3, 2, 2))})

    assert capture.forward_calls == 0
