# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit and end-to-end tests for unified_mot.py.

The file has two tiers of tests:

1. CPU-only, pure-logic units -- the hand-rolled pieces that need neither a GPU,
   HF weights, nor a distributed process group:

     * ``ReasonerKVCache``     -- per-layer KV cache append/reset bookkeeping.
     * ``_sample_next_token``  -- greedy / multinomial sampling with penalties.
     * ``_all_ranks_finished`` -- the single-process (non-distributed) branch.
     * ``LayerTypes``          -- architecture-family dispatch table.
     * ``_MoTConfigBase``      -- config materialization (text/vision/full).

2. GPU end-to-end tests (``@pytest.mark.L1``, gated by ``skip_if_no_gpu_backend``
   -- they need a Hopper or Blackwell DC-class GPU, since cosmos_framework.model.attention has
   no CPU backend). All run a miniaturized random-weight Qwen3-VL-8B MoT model:

     * ``test_end_to_end_reasoner_generate_smoke`` /
       ``test_end_to_end_reasoner_generate_sampling_controls`` -- single-GPU AR
       decode through the reasoner tower's ``generate_reasoner_text`` loop.
     * ``test_end_to_end_training`` -- a short FSDP-sharded training loop on the
       generator tower (see below).

Run the CPU + single-GPU tests with:
    pytest cosmos_framework/model/generator/mot/unified_mot_test.py -v -s --all

``test_end_to_end_training`` is the exception: it shards the model with FSDP and
must be launched under ``torchrun`` so it can spread the (very long) packed
sequences across every available GPU (``skip_if_not_multigpu`` skips it otherwise).
Run it on all local GPUs with:

    torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m pytest \\
        cosmos_framework/model/generator/mot/unified_mot_test.py::test_end_to_end_training --L1 -s
"""

import json
import math
import os
import subprocess
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from cosmos_framework.model.attention.utils import is_blackwell_dc, is_hopper
from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.generator.mot.attention import build_packed_sequence
from cosmos_framework.model.generator.mot.parallelize_unified_mot import parallelize_unified_mot
from cosmos_framework.model.generator.mot.unified_mot import (
    LayerTypes,
    Nemotron3DenseVLMoTConfig,
    Qwen3VLMoTConfig,
    Qwen3VLTextForCausalLM,
    ReasonerKVCache,
    _all_ranks_finished,
    _MoTConfigBase,
    _sample_next_token,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    Nemotron3DenseVLMLP,
    Nemotron3DenseVLRMSNorm,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import (
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
    Qwen3VLMoeTextMLP,
    Qwen3VLMoeTextRMSNorm,
)
from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, get_und_seq
from cosmos_framework.utils.generator.parallelism import ParallelDims

# -----------------------------------------------------------------------------
# ReasonerKVCache
# -----------------------------------------------------------------------------


def _kv(batch: int, seqlen: int, num_kv_heads: int = 2, head_dim: int = 4) -> torch.Tensor:
    """Build a deterministic [B, T, num_kv_heads, head_dim] tensor."""
    n = batch * seqlen * num_kv_heads * head_dim
    return torch.arange(n, dtype=torch.float32).reshape(batch, seqlen, num_kv_heads, head_dim)


def test_reasoner_kv_cache_empty() -> None:
    cache = ReasonerKVCache.empty(num_layers=3)
    assert cache.num_layers == 3
    assert cache.keys == [None, None, None]
    assert cache.values == [None, None, None]
    # seq_len is 0 while the first layer is unpopulated.
    assert cache.seq_len == 0


def test_reasoner_kv_cache_update_populates_then_appends() -> None:
    cache = ReasonerKVCache.empty(num_layers=2)

    k0 = _kv(batch=1, seqlen=3)
    v0 = _kv(batch=1, seqlen=3) + 100.0
    out_k, out_v = cache.update(layer_idx=0, k=k0, v=v0)

    # First update populates the layer verbatim and returns the stored tensors.
    assert out_k is cache.keys[0]
    assert out_v is cache.values[0]
    torch.testing.assert_close(out_k, k0)
    torch.testing.assert_close(out_v, v0)
    assert cache.seq_len == 3

    # Second update concatenates along the sequence dim (dim=1).
    k1 = _kv(batch=1, seqlen=2) + 1000.0
    v1 = _kv(batch=1, seqlen=2) + 2000.0
    out_k, out_v = cache.update(layer_idx=0, k=k1, v=v1)
    assert out_k.shape[1] == 5
    assert out_v.shape[1] == 5
    torch.testing.assert_close(out_k, torch.cat([k0, k1], dim=1))
    torch.testing.assert_close(out_v, torch.cat([v0, v1], dim=1))
    assert cache.seq_len == 5

    # Other layers remain untouched by updates to layer 0.
    assert cache.keys[1] is None


def test_reasoner_kv_cache_reset() -> None:
    cache = ReasonerKVCache.empty(num_layers=2)
    cache.update(layer_idx=0, k=_kv(1, 3), v=_kv(1, 3))
    cache.update(layer_idx=1, k=_kv(1, 3), v=_kv(1, 3))
    assert cache.seq_len == 3

    cache.reset()
    assert cache.keys == [None, None]
    assert cache.values == [None, None]
    assert cache.seq_len == 0


# -----------------------------------------------------------------------------
# _sample_next_token
# -----------------------------------------------------------------------------


def test_sample_greedy_returns_argmax() -> None:
    logits = torch.tensor([[0.1, 5.0, -3.0, 2.0], [9.0, 0.0, 1.0, 0.5]])
    out = _sample_next_token(
        logits,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
    )
    assert out.shape == (2,)
    torch.testing.assert_close(out, torch.tensor([1, 0]))


def test_sample_top_k_then_argmax_keeps_top_token() -> None:
    # With do_sample but a degenerate distribution (one huge logit), the
    # multinomial draw should still land on the dominant token.
    logits = torch.tensor([[0.0, 0.0, 50.0, 0.0]])
    out = _sample_next_token(
        logits,
        do_sample=True,
        temperature=1.0,
        top_k=2,
        top_p=None,
    )
    assert out.item() == 2


def test_sample_top_p_keeps_at_least_one_token() -> None:
    # A very small top_p must still leave one selectable token (no all -inf).
    logits = torch.tensor([[0.0, 1.0, 2.0, 10.0]])
    out = _sample_next_token(
        logits,
        do_sample=True,
        temperature=1.0,
        top_k=None,
        top_p=0.01,
    )
    assert out.item() == 3


def test_sample_repetition_penalty_rescales_seen_logits() -> None:
    # CTRL/HF formula: positive seen logits are divided by the penalty,
    # negative seen logits are multiplied by it. Applied before argmax.
    logits = torch.tensor([[4.0, 3.0, -1.0, 0.0]])
    seen_mask = torch.tensor([[True, False, False, False]])
    # Without penalty, argmax is index 0.
    base = _sample_next_token(logits, do_sample=False, temperature=1.0, top_k=None, top_p=None)
    assert base.item() == 0
    # With a strong penalty on index 0, its logit (4.0/8.0=0.5) drops below
    # index 1 (3.0), shifting the greedy choice.
    penalized = _sample_next_token(
        logits.clone(),
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=8.0,
        seen_mask=seen_mask,
    )
    assert penalized.item() == 1


def test_sample_presence_penalty_shifts_seen_logits() -> None:
    # OpenAI semantics: subtract a constant from every output-seen logit.
    logits = torch.tensor([[2.0, 1.5, 0.0, 0.0]])
    output_seen_mask = torch.tensor([[True, False, False, False]])
    penalized = _sample_next_token(
        logits.clone(),
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        presence_penalty=1.0,
        output_seen_mask=output_seen_mask,
    )
    # index 0: 2.0 - 1.0 = 1.0 < index 1: 1.5 -> argmax flips to 1.
    assert penalized.item() == 1


def test_sample_generator_is_reproducible() -> None:
    logits = torch.randn(4, 32)
    gen_a = torch.Generator().manual_seed(1234)
    gen_b = torch.Generator().manual_seed(1234)
    out_a = _sample_next_token(logits, do_sample=True, temperature=1.0, top_k=None, top_p=None, generator=gen_a)
    out_b = _sample_next_token(logits, do_sample=True, temperature=1.0, top_k=None, top_p=None, generator=gen_b)
    torch.testing.assert_close(out_a, out_b)


# -----------------------------------------------------------------------------
# _all_ranks_finished (non-distributed branch)
# -----------------------------------------------------------------------------


def test_all_ranks_finished_local_all_true() -> None:
    assert _all_ranks_finished(torch.tensor([True, True, True])) is True


def test_all_ranks_finished_local_some_false() -> None:
    assert _all_ranks_finished(torch.tensor([True, False, True])) is False


# -----------------------------------------------------------------------------
# LayerTypes
# -----------------------------------------------------------------------------


def test_layer_types_qwen3_vl_dense() -> None:
    lt = LayerTypes("qwen3_vl_dense")
    assert lt.mlp is Qwen3VLTextMLP
    assert lt.rms_norm is Qwen3VLTextRMSNorm
    assert lt.is_moe is False


def test_layer_types_qwen3_vl_moe() -> None:
    lt = LayerTypes("qwen3_vl_moe")
    assert lt.mlp is Qwen3VLMoeTextMLP
    assert lt.rms_norm is Qwen3VLMoeTextRMSNorm
    assert lt.is_moe is True


def test_layer_types_nemotron_dense() -> None:
    lt = LayerTypes("nemotron_dense")
    assert lt.mlp is Nemotron3DenseVLMLP
    assert lt.rms_norm is Nemotron3DenseVLRMSNorm
    assert lt.is_moe is False


def test_layer_types_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LayerTypes variant"):
        LayerTypes("not_a_real_variant")


# -----------------------------------------------------------------------------
# _MoTConfigBase and subclasses
# -----------------------------------------------------------------------------


def test_mot_config_base_full_config_sentinel_raises() -> None:
    # The base class leaves _full_config_cls as type(None); accessing
    # full_config must raise ValueError before any HF instantiation.
    cfg = _MoTConfigBase({})
    with pytest.raises(ValueError, match="No _full_config_cls"):
        _ = cfg.full_config


def test_mot_config_base_text_config_sentinel_raises() -> None:
    cfg = _MoTConfigBase({})
    with pytest.raises(ValueError, match="No _text_config_cls"):
        _ = cfg.text_config


def test_mot_config_base_vision_config_sentinel_raises() -> None:
    # include_visual=True with a vision_config sub-section reaches the
    # _vision_config_cls sentinel check (still type(None) on the base).
    cfg = _MoTConfigBase({"vision_config": {"foo": 1}}, include_visual=True)
    with pytest.raises(ValueError, match="No _vision_config_cls"):
        _ = cfg.vision_config


def test_text_config_flat_vs_nested_agree() -> None:
    flat = Qwen3VLMoTConfig({"num_hidden_layers": 7})
    nested = Qwen3VLMoTConfig({"text_config": {"num_hidden_layers": 7}})
    assert flat.text_config.num_hidden_layers == 7
    assert nested.text_config.num_hidden_layers == 7


def test_text_config_overrides_win() -> None:
    # text_config_overrides must beat the JSON defaults (SMOKE-style shrink).
    cfg = Qwen3VLMoTConfig(
        {"num_hidden_layers": 99},
        text_config_overrides={"num_hidden_layers": 2},
    )
    assert cfg.text_config.num_hidden_layers == 2


def test_nemotron_transform_text_dict_folds_56_to_28() -> None:
    cfg = Nemotron3DenseVLMoTConfig({})
    folded = cfg._transform_text_dict({"num_hidden_layers": 56})
    assert folded["num_hidden_layers"] == 28


def test_nemotron_transform_text_dict_identity_otherwise() -> None:
    cfg = Nemotron3DenseVLMoTConfig({})
    untouched = cfg._transform_text_dict({"num_hidden_layers": 12})
    assert untouched["num_hidden_layers"] == 12


def test_vision_config_none_when_not_included() -> None:
    cfg = Qwen3VLMoTConfig({"text_config": {"num_hidden_layers": 2}})
    assert cfg.vision_config is None


def test_vision_config_missing_subsection_raises() -> None:
    cfg = Qwen3VLMoTConfig({"text_config": {"num_hidden_layers": 2}}, include_visual=True)
    with pytest.raises(ValueError, match="requires a vision_config sub-section"):
        _ = cfg.vision_config


def test_from_json_file_round_trips(tmp_path) -> None:
    payload = {"text_config": {"num_hidden_layers": 5}, "image_token_id": 42}
    json_path = tmp_path / "cfg.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = Qwen3VLMoTConfig.from_json_file(str(json_path))
    assert isinstance(cfg, Qwen3VLMoTConfig)
    assert cfg.config_dict == payload
    assert cfg.text_config.num_hidden_layers == 5


# -----------------------------------------------------------------------------
# End-to-end model (GPU required)
# -----------------------------------------------------------------------------
#
# The reasoner / training forward paths dispatch all attention through
# ``cosmos_framework.model.attention.attention``, whose backends (Flash3 / NATTEN / cuDNN)
# require a Hopper- or Blackwell-class CUDA device. There is no CPU attention
# backend, so these tests are skipped off-GPU rather than run on CPU.

skip_if_no_gpu_backend = pytest.mark.skipif(
    not torch.cuda.is_available() or (not is_blackwell_dc() and not is_hopper()),
    reason="End-to-end MoT model tests require a Hopper or Blackwell DC-class GPU "
    "(cosmos_framework.model.attention has no CPU backend).",
)


def _nvidia_smi_gpu_count() -> int:
    """Number of GPUs reported by ``nvidia-smi -L``; 0 if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return sum(1 for line in out.splitlines() if line.strip())


# ``test_end_to_end_training`` shards the model with FSDP across every GPU, so it
# only makes sense when launched under ``torchrun`` with one rank per GPU. Skip it
# unless WORLD_SIZE is set and matches the GPU count reported by nvidia-smi (i.e. a
# plain ``pytest`` run, or a launch that does not span all GPUs), rather than
# running it on a single / partial device set where the long packed sequences OOM.
skip_if_not_multigpu = pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", 0)) != _nvidia_smi_gpu_count(),
    reason="test_end_to_end_training requires a torchrun launch spanning all GPUs "
    "(WORLD_SIZE must equal the nvidia-smi GPU count). Run with `torchrun --standalone "
    "--nproc_per_node=$(nvidia-smi -L | wc -l) -m pytest "
    "cosmos_framework/model/generator/mot/unified_mot_test.py::test_end_to_end_training --L1 -s`.",
)


def _qwen3vl_8b_mot_config_miniaturized() -> Qwen3VLMoTConfig:
    """A miniaturized dense Qwen3-VL MoT config based on the Qwen3-VL-8B text tower.

    The per-layer text hyperparameters are borrowed verbatim from the published
    ``Qwen/Qwen3-VL-8B-Instruct`` ``text_config`` (hidden 4096, 32 / 8
    attention / KV heads, ``head_dim`` 128, intermediate 12288, ``rope_theta``
    5e6, interleaved mrope ``[24, 20, 20]``, 151936 vocab) so the end-to-end
    tests exercise production-shaped layers, but the depth is shrunk to
    ``num_hidden_layers=2`` (the real model has 36) to keep the model small
    enough for tests. The ``text_config`` is nested so the wrapper's
    ``full_config`` materializes a correctly-shaped ``Qwen3VLConfig``.

    NOTE: even miniaturized to 2 layers, the full-width hidden/intermediate
    sizes plus the 151936-row embedding make this a sizable model (the MoT
    dual-pathway layout roughly doubles the per-layer attention/MLP params), so
    the GPU-gated end-to-end tests still need a reasonably large-memory device.
    """
    return Qwen3VLMoTConfig(
        {
            "text_config": {
                "vocab_size": 151936,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 2,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000.0,
                "rope_scaling": {
                    "mrope_interleaved": True,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default",
                },
                "max_position_embeddings": 262144,
                "tie_word_embeddings": False,
            }
        }
    )


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the rank-local shard of a (possibly FSDP-sharded) ``DTensor``.

    After ``fully_shard`` parameters / gradients are ``DTensor`` shards; the
    per-rank assertions (finiteness, "did this weight move?") only need the
    local shard, so this avoids the collective that ``full_tensor()`` would
    incur. Plain tensors (e.g. the input ``packed_sequence.grad``) pass through.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _init_fsdp_parallel_dims() -> ParallelDims:
    """Initialize NCCL + an all-GPU pure-FSDP (``dp_shard``) mesh from the torchrun env.

    The training test shards the model across every rank, so it must be launched
    under ``torchrun`` (see the module docstring). ``dp_shard=-1`` auto-infers to
    ``world_size`` — one FSDP shard group spanning all available GPUs.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip(
            "test_end_to_end_training shards the model with FSDP and must be launched under "
            "torchrun, e.g. `torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) "
            "-m pytest cosmos_framework/model/generator/mot/unified_mot_test.py::test_end_to_end_training "
            "--L1 -s`."
        )
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", dist.get_rank())))
    parallel_dims = ParallelDims(world_size=dist.get_world_size(), dp_shard=-1)
    parallel_dims.build_meshes(device_type="cuda")
    return parallel_dims


def _parallelize_for_training(model: Qwen3VLTextForCausalLM, parallel_dims: ParallelDims) -> Qwen3VLTextForCausalLM:
    """Apply activation checkpointing + torch.compile + FSDP via the production pass.

    Reuses ``parallelize_unified_mot`` — the exact optimization pass the real
    training path runs — so this test exercises the full stack at once:

    * **Activation checkpointing** (``mode="full"``): recompute each decoder
      block in the backward pass, which (together with the per-rank batch split)
      keeps the very long packed sequences' activations within GPU memory.
    * **torch.compile** (``compiled_region="language"``): compile each decoder
      block. ``compile_dynamic=True`` (the default) tolerates varying sequence
      lengths without recompiling.
    * **FSDP2**: shard every decoder block across ``parallel_dims.dp_mesh``.

    ``parallelize_unified_mot`` only shards the decoder blocks; the root module
    (``embed_tokens`` / ``lm_head`` / ``norm``) is sharded here afterwards,
    mirroring ``parallelize_vfm_network``, which wraps the root once
    ``parallelize_unified_mot`` returns.
    """
    parallelize_unified_mot(
        model,
        parallel_dims=parallel_dims,
        # Disable torch.compile for the test.
        compile_config=CompileConfig(enabled=False),
        # Defaults to mode="full": checkpoint (recompute) each whole decoder block.
        ac_config=ActivationCheckpointingConfig(),
    )
    fully_shard(model, mesh=parallel_dims.dp_mesh)
    return model


@pytest.fixture(scope="module")
def reasoner_lm() -> Qwen3VLTextForCausalLM:
    """Build a random-weight miniaturized Qwen3-VL-8B ``Qwen3VLTextForCausalLM`` on CUDA in bf16.

    Built on the ``meta`` device, then cast to bf16, materialized on CUDA, and
    initialized — mirroring ``trainable_lm`` / the production ``build_model_unified``
    path. ``init_weights`` re-creates the rotary ``inv_freq`` buffers in float32 so
    the rotary embedding stays on its fast path. Kept in ``eval()`` and unsharded:
    the AR-decode tests it backs run on a single GPU.
    """
    config = _qwen3vl_8b_mot_config_miniaturized()
    with torch.device("meta"):
        model = Qwen3VLTextForCausalLM(config)
    model = model.to(torch.bfloat16)
    model.to_empty(device="cuda")
    model.init_weights(buffer_device="cuda")
    model.eval()
    return model


@skip_if_no_gpu_backend
@pytest.mark.L1
def test_end_to_end_reasoner_generate_smoke(reasoner_lm: Qwen3VLTextForCausalLM) -> None:
    """End-to-end smoke test of the reasoner-tower AR decode loop.

    Builds a miniaturized Qwen3-VL-8B dense MoT causal LM with random weights and runs
    the full prefill -> decode pipeline through ``generate_reasoner_text`` (which
    exercises ``_impl_generate_reasoner_text`` -> ``_impl_reasoner_forward`` ->
    ``MoTDecoderLayer.reasoner_forward`` -> ``cosmos_framework.model.attention``). Random
    weights are not semantically meaningful, but the pipeline must still produce
    a well-formed integer tensor of the documented shape and obey the API
    contract (vocab range, prompt-prefix preservation, greedy determinism).
    """
    model = reasoner_lm
    vocab_size = int(model.config.text_config.vocab_size)

    batch, prompt_len, new_tokens = 2, 6, 4
    prompt = torch.randint(0, vocab_size, (batch, prompt_len), dtype=torch.long, device="cuda")

    out = model.generate_reasoner_text(prompt, max_new_tokens=new_tokens, do_sample=False)

    assert out.dtype == torch.long
    assert out.shape == (batch, prompt_len + new_tokens)
    assert (out >= 0).all() and (out < vocab_size).all(), "Generated ids out of vocab range"
    assert torch.equal(out[:, :prompt_len], prompt), "Prompt prefix must be preserved"

    # return_only_new_tokens returns just the generated suffix.
    out_new = model.generate_reasoner_text(
        prompt, max_new_tokens=new_tokens, do_sample=False, return_only_new_tokens=True
    )
    assert out_new.shape == (batch, new_tokens)
    assert torch.equal(out_new, out[:, prompt_len:])

    # Greedy decoding is deterministic across calls.
    out_again = model.generate_reasoner_text(prompt, max_new_tokens=new_tokens, do_sample=False)
    assert torch.equal(out, out_again), "Greedy decoding must be deterministic"


@skip_if_no_gpu_backend
@pytest.mark.L1
def test_end_to_end_reasoner_generate_sampling_controls(reasoner_lm: Qwen3VLTextForCausalLM) -> None:
    """top-k / top-p / temperature / penalties reach the sampler without error.

    Also pins the seeded-generator reproducibility contract: two sampled decodes
    that share a ``seed`` must produce byte-identical tokens.
    """
    model = reasoner_lm
    vocab_size = int(model.config.text_config.vocab_size)

    prompt = torch.randint(0, vocab_size, (1, 4), dtype=torch.long, device="cuda")

    out = model.generate_reasoner_text(
        prompt,
        max_new_tokens=8,
        do_sample=True,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        presence_penalty=0.1,
        seed=1234,
    )
    assert out.shape == (1, 4 + 8)
    assert (out >= 0).all() and (out < vocab_size).all()

    out_same_seed = model.generate_reasoner_text(
        prompt,
        max_new_tokens=8,
        do_sample=True,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        presence_penalty=0.1,
        seed=1234,
    )
    assert torch.equal(out, out_same_seed), "Seeded sampling must be reproducible"


# -----------------------------------------------------------------------------
# End-to-end training step (forward + backward, GPU required)
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trainable_lm() -> Iterator[tuple[Qwen3VLTextForCausalLM, ParallelDims]]:
    """A miniaturized Qwen3-VL-8B ``Qwen3VLTextForCausalLM``, FSDP-sharded, in train mode.

    Separate from ``reasoner_lm`` (which is ``eval()`` and only ever run
    under ``@torch.no_grad`` via ``generate_reasoner_text``): the training path
    needs gradients flowing, so this fixture keeps the model in ``train()``.
    bfloat16 keeps the kernels on the Flash/cuDNN fast path while still
    accumulating fp32 gradients into the parameters.

    The model is built and optimized on the ``meta`` device (no real allocation),
    then materialized and initialized on CUDA — mirroring the production
    ``build_model_unified`` path. Optimization is the production
    ``parallelize_unified_mot`` pass (activation checkpointing + torch.compile +
    FSDP sharding across every available GPU) so the very long packed sequences
    this test builds fit in memory; this requires a ``torchrun`` launch (see the
    module docstring). Weights are seeded identically on every rank so
    ``init_weights`` produces consistent shards. Returns ``(model, parallel_dims)``;
    the process group is torn down on fixture teardown.
    """
    parallel_dims = _init_fsdp_parallel_dims()
    # Identical (fixed) seed on every rank -> identical init_weights -> consistent shards.
    # Do NOT seed with the rank: FSDP2/DTensor sharded init is SPMD and offsets the RNG
    # per shard internally, so every rank must start from the same seed.
    torch.manual_seed(0)
    config = _qwen3vl_8b_mot_config_miniaturized()
    # Build + parallelize on meta (cheap, no storage), then cast to bf16,
    # materialize the (now FSDP-sharded) params on CUDA, and initialize. Using
    # init_weights also re-creates the rotary ``inv_freq`` buffers in float32.
    with torch.device("meta"):
        model = Qwen3VLTextForCausalLM(config)
    model = _parallelize_for_training(model, parallel_dims)
    model = model.to(torch.bfloat16)
    model.to_empty(device="cuda")
    model.init_weights(buffer_device="cuda")
    model.train()
    yield model, parallel_dims
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _build_two_way_training_pack(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    und_lens: list[int],
    gen_lens: list[int],
    packed_sequence: torch.Tensor | None = None,
):
    """Hand-build the minimal ``two_way`` training inputs for a batch of samples.

    Mirrors what ``Cosmos3VFMNetwork`` feeds ``language_model.forward`` but
    without the full data pipeline: ``len(und_lens)`` samples are packed end to
    end, and within each sample the first ``und_lens[i]`` tokens form the
    understanding (causal) split and the next ``gen_lens[i]`` tokens form the
    generation (full) split. ``packed_sequence`` stands in for the already
    embedded hidden states (the text model consumes embeddings directly, it does
    not re-embed) and carries ``requires_grad`` so we can assert the gradient
    reaches the model input. Pass an explicit ``packed_sequence`` to reuse a
    fixed input across training steps (each step must build a fresh pack: the
    pack indexes ``packed_sequence`` into the autograd graph, and that graph is
    freed by every ``backward()`` call).

    Returns ``(input_pack, attention_meta, position_ids, packed_sequence)``.
    """
    assert len(und_lens) == len(gen_lens)
    n_total = sum(und_lens) + sum(gen_lens)
    if packed_sequence is None:
        packed_sequence = torch.randn(n_total, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Per sample: one "causal" (und) split followed by one "full" (gen) split.
    # attn_modes therefore alternate causal/full across the batch, split_lens
    # interleave und/gen lengths, and sample_lens are the per-sample totals.
    attn_modes: list[str] = []
    split_lens: list[int] = []
    sample_lens: list[int] = []
    und_indices: list[int] = []
    gen_indices: list[int] = []
    position_id_chunks: list[torch.Tensor] = []
    offset = 0
    for n_und, n_gen in zip(und_lens, gen_lens):
        attn_modes.extend(["causal", "full"])
        split_lens.extend([n_und, n_gen])
        sample_lens.append(n_und + n_gen)
        und_indices.extend(range(offset, offset + n_und))
        gen_indices.extend(range(offset + n_und, offset + n_und + n_gen))
        # Positions restart per sample (und occupies 0..n_und-1, gen n_und..L-1).
        position_id_chunks.append(torch.arange(n_und + n_gen, dtype=torch.long, device="cuda"))
        offset += n_und + n_gen

    input_pack, attention_meta, natten_metadata_list = build_packed_sequence(
        "two_way",
        packed_sequence=packed_sequence,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=torch.tensor(und_indices, dtype=torch.long, device="cuda"),
        packed_gen_token_indexes=torch.tensor(gen_indices, dtype=torch.long, device="cuda"),
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        is_image_batch=True,
    )
    assert natten_metadata_list is None, "two_way attention must not emit NATTEN metadata"
    # 1D per-sample absolute positions over the packed sequence -> standard-RoPE branch.
    position_ids = torch.cat(position_id_chunks)
    return input_pack, attention_meta, position_ids, packed_sequence


@skip_if_not_multigpu
@skip_if_no_gpu_backend
@pytest.mark.L1
def test_end_to_end_training(trainable_lm: tuple[Qwen3VLTextForCausalLM, ParallelDims]) -> None:
    """A short distributed training loop that trains only the generator (gen) tower.

    Exercises the real training entry point (``Qwen3VLTextForCausalLM.forward``
    -> ``Qwen3VLTextModel`` -> ``MoTDecoderLayer`` -> ``PackedAttentionMoT`` ->
    ``two_way_attention``) on a model optimized with the production
    ``parallelize_unified_mot`` pass (activation checkpointing + torch.compile +
    FSDP). The reasoner / understanding (causal) tower is frozen
    (``requires_grad=False``); only the generation (full) tower's ``*_moe_gen``
    parameters are optimized, against a loss computed on the gen-tower output.

    The global batch is split across ranks (data parallelism): each rank packs
    only its own slice of samples, which — together with FSDP parameter sharding
    and full-block activation checkpointing — keeps the very long packed sequences
    within a single GPU's memory. Parameters and gradients are ``DTensor`` shards,
    so the per-rank checks below inspect the local shard via ``_local``.

    We assert that:

    * the forward output pack is well-formed (correct und/gen token counts,
      finite values),
    * on the first step gradients reach the gen tower and the packed input but
      NOT the frozen reasoner tower, with no NaNs/Infs,
    * every step's loss is finite,
    * 10 steps reduce the loss (optimizer + autograd are wired up), and
    * the reasoner weights are unchanged while the generator weights move.
    """
    model, parallel_dims = trainable_lm
    rank = dist.get_rank()
    world_size = parallel_dims.world_size
    text_config = model.config.text_config
    hidden_size = int(text_config.hidden_size)
    num_heads = int(text_config.num_attention_heads)
    head_dim = int(text_config.head_dim)
    num_layers = int(text_config.num_hidden_layers)

    # Generator (gen) tower == every ``*_moe_gen`` parameter; everything else
    # (the understanding/reasoner tower plus embeddings / lm_head) is frozen so
    # only the generator is trained.
    def _is_generator(name: str) -> bool:
        return "moe_gen" in name

    generator_params = []
    for name, param in model.named_parameters():
        is_gen = _is_generator(name)
        param.requires_grad_(is_gen)
        if is_gen:
            generator_params.append(param)
    assert len(generator_params) > 0, "Found no generator (moe_gen) parameters to train"

    # Snapshot one reasoner and one generator weight to verify, after training,
    # that the reasoner is frozen while the generator actually moved. Parameters
    # are FSDP ``DTensor`` shards; compare the rank-local shard.
    reasoner_ref_name, reasoner_ref = next(
        (n, p) for n, p in model.named_parameters() if "q_proj.weight" in n and not _is_generator(n)
    )
    generator_ref_name, generator_ref = next(
        (n, p) for n, p in model.named_parameters() if "q_proj_moe_gen.weight" in n
    )
    reasoner_before = _local(reasoner_ref).detach().clone()
    generator_before = _local(generator_ref).detach().clone()

    # Explicit generators for reproducible tensors across runs. The CPU generator
    # uses the same seed on every rank so all ranks derive the identical global
    # batch before slicing; the CUDA generator is seeded per-rank so each rank's
    # local input / target differ but stay reproducible.
    cpu_gen = torch.Generator().manual_seed(0)
    cuda_gen = torch.Generator(device="cuda").manual_seed(rank)

    # Global batch of 10 packed samples whose total length (und + gen) is a random
    # value in [30000, 75000), stressing varlen packing across a wide length range.
    # The lengths are generated identically on every rank (shared ``cpu_gen`` seed);
    # each rank then trains on its own ``rank::world_size`` slice so the packed
    # sequence — and thus the per-GPU activation memory — is divided across ranks.
    # Each sample is split into an understanding (causal) length, a random value in
    # [512, total // 4), and a generation (full) remainder.
    global_batch_size = 10
    global_total_lens: list[int] = []
    global_und_lens: list[int] = []
    for _ in range(global_batch_size):
        total = int(torch.randint(30000, 75000, (1,), generator=cpu_gen).item())
        global_total_lens.append(total)
        global_und_lens.append(int(torch.randint(512, total // 4, (1,), generator=cpu_gen).item()))

    und_lens = global_und_lens[rank::world_size]
    sample_total_lens = global_total_lens[rank::world_size]
    gen_lens = [length - n_und for length, n_und in zip(sample_total_lens, und_lens)]
    assert len(und_lens) > 0, f"world_size={world_size} exceeds global batch size {global_batch_size}"
    n_total = sum(sample_total_lens)
    n_und_total = sum(und_lens)
    n_gen_total = sum(gen_lens)

    # Fixed input data across steps so the loss curve reflects parameter updates
    # only. The pack is rebuilt from a fresh leaf each step because every pack
    # indexes its source tensor into the autograd graph, which backward() frees.
    base_embeddings = torch.randn(n_total, hidden_size, device="cuda", dtype=torch.bfloat16, generator=cuda_gen)

    # Fixed regression target for the gen tower. Fitting an arbitrary target makes
    # the loss depend on the actual gen-tower output (hence on every trained weight
    # and the network depth), unlike a plain ``gen_out.pow(2)`` which the final
    # RMSNorm renders invariant to the upstream activations.
    target_gen = torch.randn(n_gen_total, hidden_size, device="cuda", dtype=torch.float32, generator=cuda_gen)

    optimizer = torch.optim.Adam(generator_params, lr=1e-2)

    num_steps = 10
    losses: list[float] = []
    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        packed_sequence = base_embeddings.clone().detach().requires_grad_(True)
        input_pack, attention_meta, position_ids, packed_sequence = _build_two_way_training_pack(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            und_lens=und_lens,
            gen_lens=gen_lens,
            packed_sequence=packed_sequence,
        )

        out_pack, lbl_metadata = model(
            input_pack,
            attention_mask=attention_meta,
            position_ids=position_ids,
        )

        und_out = get_und_seq(out_pack)  # [n_und_total, hidden_size]
        gen_out = get_gen_seq(out_pack)  # [n_gen_total, hidden_size]
        assert und_out.shape == (n_und_total, hidden_size)
        assert gen_out.shape == (n_gen_total, hidden_size)
        assert torch.isfinite(und_out.float()).all()
        assert torch.isfinite(gen_out.float()).all()
        # Dense (non-MoE) model: no load-balancing-loss metadata is produced.
        assert lbl_metadata == {}

        # Train the generator only: MSE between the gen-tower output and a fixed
        # target, so gradients flow through the generation (full) attention path
        # and the loss actually reflects the trained weights at every depth.
        loss = (gen_out.float() - target_gen).pow(2).mean()
        assert torch.isfinite(loss), f"Training loss must be finite (step {step})"

        loss.backward()

        if step == 0:
            # Gradient must reach the (embedded) input tokens.
            assert packed_sequence.grad is not None, "No gradient reached the packed input"
            assert torch.isfinite(packed_sequence.grad.float()).all()
            assert packed_sequence.grad.float().abs().sum() > 0, "Input gradient is identically zero"

            # The gen tower must receive finite gradients; the frozen reasoner
            # tower must receive none. Gradients are ``DTensor`` shards under FSDP.
            params_with_grad = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
            assert len(params_with_grad) > 0, "No parameter received a gradient"
            for name, p in params_with_grad:
                assert _is_generator(name), f"Frozen reasoner parameter received a gradient: {name}"
                assert torch.isfinite(_local(p.grad).float()).all(), f"Non-finite gradient in {name}"

            grad_names = {n for n, _ in params_with_grad}
            assert any("q_proj_moe_gen" in n for n in grad_names), "Generation tower got no grad"
            assert reasoner_ref.grad is None, "Reasoner tower must be frozen (no grad)"

        optimizer.step()
        losses.append(loss.item())

    assert all(math.isfinite(loss_value) for loss_value in losses), "All training losses must be finite"
    assert losses[-1] < losses[0], f"10 training steps should reduce the loss: {losses}"

    # The reasoner stays exactly frozen; the generator weights actually moved.
    # Compare the rank-local shard of each (FSDP ``DTensor``) parameter.
    assert torch.equal(_local(reasoner_ref), reasoner_before), f"Reasoner weight changed: {reasoner_ref_name}"
    assert not torch.equal(_local(generator_ref), generator_before), (
        f"Generator weight did not move: {generator_ref_name}"
    )

    if rank == 0:
        print(f"\ntraining losses over {num_steps} steps: {[f'{loss_value:.2f}' for loss_value in losses]}")
    # Keep all ranks in lockstep before the fixture tears down the process group.
    dist.barrier()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
