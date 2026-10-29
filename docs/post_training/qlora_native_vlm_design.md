# Design for native Qwen3 VL QLoRA training

This proposal adds adapter training on a frozen four-bit language backbone to the native Cosmos posttrain SFT path. The first qualification target is Qwen3-VL-8B-Instruct on one A100-SXM4-80GB. Success requires real image/video conditioning, measured lifecycle memory savings, useful held-out quality, and fresh-process adapter reload and resume.

**Status: historical design baseline.** This document records the original audit and proposal at commit `1a2f03ab6111b457633827736f94f67d55bd5202` on `design/qlora-vlm`. The audited table and proposed interfaces below describe that baseline. An experimental implementation now exists; see the [native QLoRA guide](native_qlora.md) for its current contracts, completed bounded training/reload/resume qualification and memory measurements, and remaining longer-run and held-out quality release gates. The historical proposal below does not establish those results independently.

## Audited starting point

The audit baseline is `NVIDIA/cosmos-framework` commit `1a2f03ab6111b457633827736f94f67d55bd5202`, fetched from `main` on 2026-10-28. The design branch is `design/qlora-vlm`. There was no active implementation diff on this branch. The earlier VLM adapter work is present through [PR 277](https://github.com/NVIDIA/cosmos-framework/pull/277), with dense adapter export changes through [PR 308](https://github.com/NVIDIA/cosmos-framework/pull/308). New implementation patches must recheck references if their base moves.

The audit followed [AGENTS.md](../../AGENTS.md), read both dependency locks, and inspected native adapters, posttrain model construction, distributed wrapping, optimizer construction, checkpoint publication/recovery, and export. The documentation destination follows [repository documentation rules](../README.md); `docs/posttrain/` is upstream-synced and must not be edited here.

| Area                | Actual starting point and remaining gap                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native adapters     | [utils/generator/lora.py:49](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/utils/generator/lora.py#L49) implements dense adapters with native `.lora_A.weight` and `.lora_B.weight` names. Reuse its ordinary LoRA contract. The selected posttrain native builder does not yet integrate that contract.                                                                                 |
| Native construction | [posttrain/model/qwen3_vl.py:302](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/model/qwen3_vl.py#L302) provides preparation, parallelization, materialization, and loading hooks. Its default materializer allocates dense destinations; the native loader retains dense CPU source tensors. A flag on this path would not solve loading peaks.                               |
| HF construction     | [posttrain/model/hf.py](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/model/hf.py) is a separate builder. Its generic streaming loader still uses dense destinations. Neither HF inference quantization nor an alternative standalone HF trainer establishes support on the selected native path.                                                                              |
| Optimizer           | [posttrain/config/build.py:547](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/config/build.py#L547) builds parameter groups; a nonempty `train_keys` can admit frozen parameters. Its native dtype guard assumes model dtype describes trainable storage. Both require a narrowly scoped adapter contract.                                                                     |
| Distribution        | [posttrain/model/qwen3_vl.py:313](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/model/qwen3_vl.py#L313) uses native FSDP2/DTensor machinery for data-parallel worlds, including HSDP; this is not FSDP1 or a DDP stack. The initial singleton path must avoid invoking FSDP wrapping.                                                                                          |
| Checkpointing       | [posttrain/roles/trainer/llm.py:1269](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/roles/trainer/llm.py#L1269) extracts state from the full model. An adapter-only codec must select a parameter view before extraction, while retaining the existing manager, commit, and recovery machinery.                                                                                |
| Export              | [callbacks/hf_export.py:62](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/callbacks/hf_export.py#L62) and the [native weight mapper](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/model/qwen3_vl.py#L971) assume compatible dense model weights. Quantized adapters must explicitly refuse these paths initially. |
| Quantization        | No completed frozen-four-bit training path was found. Existing quantization utilities/recipes do not establish this native training, loading, optimizer, multimodal, and checkpoint contract. Neither lock contains bitsandbytes.                                                                                                                                                                                                                    |

The root [environment](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/pyproject.toml) and [lock](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/uv.lock) are distinct from the selected [posttrain environment](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/docker/posttrain/environment/pyproject.toml) and [lock](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/docker/posttrain/environment/uv.lock). Select Python 3.12 with posttrain groups `sft` and `cu128-runtime`, excluding the default `transformers4` group. The candidate selection is Torch `2.11.0+cu128`, Transformers `5.8.1`, TorchCodec `0.11.0`, Accelerate `1.13.0`, and safetensors `0.8.0`. Add a separately locked optional `qlora` group pinning bitsandbytes `0.49.2`; its compatibility remains a kernel-test gate. The root PEFT pin is `0.18.1`; the posttrain `0.17.1` entry belongs to a separate aarch64 recipe and is not a requirement of this native design.

## First capability and exact coverage

Use `Qwen/Qwen3-VL-8B-Instruct` at immutable commit `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, with its processor/tokenizer/template from the same verified snapshot. The [pinned model configuration](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json) is dense, has 36 language layers, and is handled by the repository's native Qwen3-VL implementation. Its attention and MLP projections are called as modules, permitting replacement without changing attention, media fusion, or loss semantics: [native qwen3_vl.py:440](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/model/generator/reasoner/qwen3_vl/qwen3_vl.py#L440).

Qualify one Linux x86-64 process on one A100-SXM4-80GB, SM80. No claim is made for other accelerators or smaller memory capacities. Use the native `NativePolicyBuildSpec` and `cosmos_framework.posttrain.entrypoints.sft`; no new training script or distributed stack.

Resolve exactly these canonical module names before wrapping:

```text
model.language_model.layers.{0..35}.self_attn.{q_proj,k_proj,v_proj,o_proj}
model.language_model.layers.{0..35}.mlp.{gate_proj,up_proj,down_proj}
```

| Projection  | Logical weight shape | Total matrices |
| ----------- | -------------------- | -------------- |
| q and o     | `[4096,4096]`        | 72             |
| k and v     | `[1024,4096]`        | 72             |
| gate and up | `[12288,4096]`       | 72             |
| down        | `[4096,12288]`       | 36             |

Expected coverage is **252 matrices / 6,945,767,424 source weight elements**. Rank-16 LoRA on those same matrices has **504 A/B tensors / 43,646,976 trainable elements**. These are static calculations; runtime assertions must resolve and record every name, logical shape, alias, packed shape, and quantization state. A packed parameter's shape is not its logical weight shape.

Exclude the complete vision tower, projector/mergers/deepstack paths, embeddings, output head, normalization, biases, and buffers. Reject any quantization target aliasing an excluded or tied weight. Do not use `all-linear` or leaf-name matching across the entire VLM. Missing, duplicate, unexpected, or unmatched targets are errors.

## Typed configuration and precision contracts

The following declarations are proposed, dependency-light dataclasses, not existing accepted fields. Extend `Qwen3VLModelConfig` with independent `adapter` and `quantized_base` fields. The adapter contract must also serve ordinary BF16 LoRA; it must not import bitsandbytes.

```python
@dataclass(frozen=True)
class NativeLoRAConfig:
    method: Literal["lora"] = "lora"
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    parameter_dtype: Literal["float32"] = "float32"
    target_profile: Literal["qwen3_vl_8b_language_v1"] = "qwen3_vl_8b_language_v1"
    bias: Literal["none"] = "none"
    modules_to_save: tuple[str, ...] = ()
    initialization: Literal["kaiming_A_zero_B"] = "kaiming_A_zero_B"
    initialization_seed: int = 42

@dataclass(frozen=True)
class QuantizedBaseTrainingConfig:
    backend: Literal["bitsandbytes"] = "bitsandbytes"
    backend_version: Literal["0.49.2"] = "0.49.2"
    format: Literal["nf4"] = "nf4"
    block_size: Literal[64] = 64
    nested: Literal[True] = True
    source_quantization_dtype: Literal["bfloat16"] = "bfloat16"
    storage_dtype: Literal["uint8"] = "uint8"
    compute_dtype: Literal["bfloat16"] = "bfloat16"
    target_profile: Literal["qwen3_vl_8b_language_v1"] = "qwen3_vl_8b_language_v1"
    offload: Literal["none"] = "none"

@dataclass(frozen=True)
class ImmutableModelSource:
    repository: str
    revision: str                     # required full immutable commit
    snapshot_path: str                # verified local weights and processor
    weight_manifest_sha256: str
    processor_manifest_sha256: str
```

Validate these types and values at configuration preflight and validate realized tensors after loading. The first capability also fixes world size one, `model_backend="i4"` with `NativePolicyBuildSpec`, SDPA, full non-reentrant activation checkpointing, `use_cache=False`, and compilation disabled. Reject contradictory base trainability, additional trained modules, adapter method extensions, unsupported offload, quantizers, dtypes, hardware, or strategy combinations. Do not catch a failed backend/kernel check and continue with a dense base, CPU backend, or alternate quantizer.

| Quantity                                 | Initial policy                                                                                                             |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Quantized base storage                   | Packed UINT8 `Params4bit`, frozen                                                                                          |
| Quantizer                                | NF4, block 64, `compress_statistics=True`                                                                                  |
| Input to quantization                    | Deterministic BF16 normalization of each eligible floating source matrix before quantization; record original source dtype |
| Dequantized base compute                 | BF16                                                                                                                       |
| Adapter parameters and branch arithmetic | FP32; cast branch result to base output dtype                                                                              |
| Optimizer state                          | Torch fused AdamW FP32 moments for FP32 adapters; step counters follow the pinned implementation                           |
| Non-quantized parameters                 | Frozen BF16, including vision, projector, embedding, head, and normalization scales                                        |
| Buffers/internal arithmetic              | Preserve native dtypes, including initialized FP32 rotary buffers and FP32 normalization arithmetic                        |

Storage, compute, trainable parameter, and optimizer-state precision must be separate fields in the resolved manifest. A future BF16 storage carrier for packed quantized bits would not be a dense BF16 base.

`ImmutableModelSource` is the persisted result of preflight resolution, not an additional model-loading implementation. Preserve its original repository/revision and digests when replacing the model ID with the verified local path. For quantization, accept only the source floating dtypes explicitly checked in the pinned snapshot manifest, normalize each eligible matrix to BF16 on CPU, and then copy it to the owned CUDA device. Reject integer, prequantized, nonfinite, or unexpected source tensors. Include the normalization policy in effective-base identity and apply the same BF16 normalization to the dense control.

Keep `model.dtype`, `rl.master_dtype`, and `rl.param_dtype` at BF16 and `rl.mixed_precision_mode="fsdp"`, as required by [native precision validation](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/config/build.py#L1827). The name does not imply actual FSDP wrapping in a singleton world. Add an explicit adapter storage override to the optimizer preflight, then assert all realized trainables are FP32. Do not globally relax native precision validation or introduce global autocast.

Use `OptimizerConfig(name="adamw", impl="fused", train_keys=[])`. Here `impl="fused"` selects Torch AdamW, not Apex `fusedadam`. Build the groups only after final transformations and verify their parameter identities equal exactly the 504 trainable adapter tensors. Base parameters and quantization state must never enter optimizer groups.

Resolve immutable source identity once before constructing the packer or model. Merely setting the existing `revision` field is insufficient: native source/config and processor paths currently omit it. Reuse bounded content hashing in [artifact_identity.py](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/utils/artifact_identity.py), add processor/tokenizer/template byte hashes, and bind [packer semantic identity](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/data/packer/sft.py#L237). Reject a separately supplied checkpoint path whose complete identity differs.

## Memory-safe construction and training

Introduce a bitsandbytes `Linear4bit` implementation sharing a backend-neutral native LoRA mixin. Preserve `.weight`, `.lora_A.weight`, and `.lora_B.weight` naming. Do not call the dense `LoraInjectedLinear` constructor on a materialized quantized module: it allocates a dense superclass weight. Generalize adapter inventory/initialization helpers, while retaining a separate dense-merge capability.

The inspected [bitsandbytes 0.49.2 constructor](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.49.2/bitsandbytes/nn/modules.py#L421) accepts meta construction, compute dtype, quantization type, nested statistics, and storage dtype. Block size belongs to `Params4bit`, not the `Linear4bit` constructor; set and verify it explicitly during loading.

```python
def forward(self, x):
    base = super().forward(x)             # bnb base operation, BF16 compute
    delta = self.lora_B(self.lora_A(self.dropout(x).float()))
    return base + (self.alpha / self.rank) * delta.to(base.dtype)
```

Do not run the frozen base inside `no_grad`; gradients through its inputs connect earlier adapters to later losses. The first A gradient can legitimately be zero because B starts at zero. Require multiple-step adapter updates, not a nonzero gradient on every adapter tensor at the first step.

Fresh construction uses the existing native hooks plus a post-load `finalize_model` hook:

```python
validate_config_and_optional_resume_metadata()   # before large allocations
source = resolve_verified_immutable_snapshot()
init_native_torchrun_world_and_local_device()
kernel_preflight_preserving_rng_and_global_flags()

net = native_model_on_meta(source.config, include_buffers=False)
normalize_ordinary_meta_parameters_to_bf16(net)
prepare_existing_native_multimodal_forward(net)
freeze_all_existing_parameters(net)
plan = resolve_targets_shapes_aliases_and_source_keys(net)
replace_targets_with_quantized_lora_meta_modules(plan)
apply_native_nonreentrant_activation_checkpointing_once(net)

materialize_excluded_bf16_parameters_and_fp32_adapters(net)
move_all_initialized_buffers_without_dtype_changes(net)
# Leave every quantized weight slot unmaterialized.

for source_tensor in stream_one_safetensors_tensor_at_a_time(source):
    if plan.is_quantized(source_tensor.name):
        dense_bf16_cpu = normalize_qualified_floating_source_to_bf16(source_tensor)
        dense = bounded_copy_to_owned_cuda_device(dense_bf16_cpu)
        packed = quantize_once_nf4_nested_block64_uint8(dense)
        plan.attach_parameter_and_complete_quantization_state(packed)
    else:
        copy_to_final_dense_destination(source_tensor)
    release_source_and_temporary_storage()

initialize_adapters_in_canonical_order_with_dedicated_generator()
assert_no_meta_tensors_and_exact_coverage_precision_trainability()
record_base_quantization_and_initial_adapter_digests()
optimizer = build_final_adapter_optimizer()
codec = build_adapter_parameter_view_after_all_transformations()
assert_live_codec_and_optimizer_parameter_identities()
```

The loading plan stores direct module references and canonical source keys before AC wrapping. It must not resolve pre-wrapper names afterward; wrappers change module paths. Materialize all registered buffers, including nonpersistent rotary `inv_freq`, from their initialized values. They are absent from safetensors. Never replace missing rotary buffers with uninitialized memory, or call root `init_weights()` after quantization. The native construction uses `include_buffers=False`: [native lifecycle](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/model/qwen3_vl.py#L644), [rotary initialization](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/model/generator/reasoner/qwen3_vl/qwen3_vl.py#L331).

Bypass both [_default_materialize](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/model/qwen3_vl.py#L481) and the dense load hook for quantized slots. The [generic loader](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/model/weight_loader.py#L160) casts source to target dtype; a dense-to-UINT8 cast is not quantization. Keep at most one source matrix, its quantization temporary, and final resident storage live, rather than a full dense CPU state dictionary or full dense GPU destination. Measure host RSS/PSS and device-process peaks; memory-mapped file pages and media decoding remain material.

Do not use `device_map="auto"`. Respect the native local-rank owner. Do not use model-wide `.to(dtype)`, `.half()`, `_apply`, or `.to_empty()` after packed tensors exist. Repeated same-device placement must preserve packed bytes and quantization metadata without requantizing. CPU/NVMe offload and memory-saver offload are unsupported initially. Reject explicit offload and weight-sync requests before movement; ordinary resident `onload()` is already a no-op. See [residency.py:224](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/utils/residency.py#L224).

Do not call PEFT preparation in this native path. The [root-pinned helper](https://github.com/huggingface/peft/blob/v0.18.1/src/peft/utils/other.py#L121) freezes parameters and promotes non-quantized FP16/BF16 parameters broadly to FP32. Explicitly prepare the frozen base before inserting adapters and restore/assert the intended trainability policy. Non-reentrant AC must preserve adapter gradients without making embeddings trainable; test first-block adapters and frozen embedding identity.

The kernel preflight must actually quantize, execute BF16 multi-token forward and input backward, compare with the same dequantized effective base, and perform several adapter optimizer updates. Check finite outputs, frozen base/state fingerprints, and quantization metadata completeness. Use dedicated generators or restore CPU/CUDA RNG and global determinism/TF32 settings. Environment import and shared-library errors are fatal. A passing canary qualifies only the backend representation and kernels, not full-model quality or memory savings.

## Native recipe and preserved objective

Add paired native Python recipes, `sft_qwen3_vl_8b_lora.py` and `sft_qwen3_vl_8b_qlora.py`, sharing one configuration factory. The proposed fragment below depends on the new schema and is not runnable on the audit baseline:

```python
config.model_backend = "i4"
config.model = NativePolicyBuildSpec(
    builder="cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy",
    model_id=verified_snapshot_path,
    model_config=Qwen3VLModelConfig(
        name_or_path="", checkpoint_path="", trust_remote_code=False,
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        dtype="bfloat16", attn_impl="sdpa", activation_checkpointing="full",
        fsdp_reduce_dtype="float32",
        adapter=NativeLoRAConfig(initialization_seed=42),       # proposed field
        quantized_base=QuantizedBaseTrainingConfig(),          # None in control
    ),
    parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=1),
    requires_collated_policy_forward=True,
)
config.rl.seed = 42
config.rl.deterministic = True
config.rl.master_dtype = config.rl.param_dtype = "bfloat16"
config.rl.reduce_dtype = "float32"
config.rl.mixed_precision_mode = "fsdp"
config.rl.max_steps = 300
config.rl.micro_batch_size = 1
config.rl.grad_accum = 8
config.rl.max_length = 4096
config.rl.ce_exponent = 0.0
config.rl.mask_think_opener = False
config.rl.serve_contract = "tools_no_prefill"
config.rl.moe_lbl_coeff = 0.0
config.rl.optimizer = OptimizerConfig(
    name="adamw", impl="fused", train_keys=[], lr=1e-4,
    betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01,
    grad_clip_norm=1.0, warmup_steps=30,
)
config.rl.lr_schedule = SFTLRScheduleConfig(
    kind="cosine", f_start=0.01, f_min=0.1, cycle_steps=300,
)
config.rl.checkpoint_format = "dcp"
config.rl.save_async = False
config.rl.save_every = 50
config.rl.processor_kwargs = {
    "images_kwargs": {"size": {"shortest_edge": 65536, "longest_edge": 262144}},
    "videos_kwargs": {
        "size": {"shortest_edge": 4096, "longest_edge": 65536},
        "num_frames": 8, "fps": None,
    },
}
# Require explicit content-addressed training/validation manifests.
config.rl.data = SFTDatasetConfig(path=train_manifest_path)
config.rl.validation = SFTValidationConfig(
    enabled=True, before_training=True, every_steps=50, batch_size=1, seed=42,
    data=SFTDatasetConfig(path=heldout_manifest_path, shuffle=False),
)
```

Processor size fields above are pixel-area budgets, not square image dimensions. Record realized resolution, grids, media tokens, frame count, and sequence-length distributions. The existing packer rejects overflow rather than truncating. Require identical accepted/rejected row IDs, token IDs, attention masks, labels, supervised-token counts, and media tensors between arms. Preserve all/last assistant selection, assistant body/end-token supervision, expanded media masks, and accumulation reduction semantics. Do not inherit the native MoE example's auxiliary loss or Apex optimizer. See [packer masking](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/data/packer/sft.py#L531) and [SFT configuration](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/config/sft.py#L121).

Use the actual native CLI and process initialization:

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$SFT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m cosmos_framework.posttrain.entrypoints.sft \
  --config cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py
```

`SFT_PYTHON` denotes the explicitly qualified posttrain interpreter. The recipe is proposed; the `--config` flag and torchrun/NCCL initialization are existing: [entry.py:33](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/utils/entry.py#L33), [dist.py:22](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/utils/dist.py#L22). Existing `--validate-manifest` and `--validate-rows` modes should validate the new contract but cannot establish kernel support. The native determinism setting uses warning-only enforcement; capture warnings and measure reproducibility rather than assuming bitwise behavior.

## Distribution matrix and sharding boundary

No row below is declared qualified by this design. Existing higher-precision behavior must remain compatible.

| Model                               | Quantizer and precision                              | Native strategy                             | Persistence                                           | Qualification status                |
| ----------------------------------- | ---------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------- | ----------------------------------- |
| Pinned Qwen3-VL-8B                  | Dense BF16 base, FP32 LoRA                           | One rank, no FSDP wrapping                  | Local consolidated adapter artifact; local DCP resume | Required matched control            |
| Same                                | NF4 UINT8 storage, BF16 compute, FP32 LoRA           | One rank, no FSDP wrapping                  | Local consolidated adapter artifact; local DCP resume | First candidate                     |
| Same                                | Same                                                 | Replicated FSDP2/HSDP                       | Any                                                   | Explicit capability error initially |
| Same                                | NF4, including any floating packed carrier           | Sharded FSDP2/DTensor                       | Any                                                   | Explicit capability error initially |
| Same                                | Other quantizers, FP16 compute, quantized DoRA/PiSSA | Any                                         | Any                                                   | Independently qualified extensions  |
| Native MoE, generator/action models | Any four-bit training                                | Any                                         | Any                                                   | Out of scope                        |
| Any                                 | Any                                                  | TP, PP, CP, EP, alternate distributed stack | Any                                                   | Out of scope                        |

The follow-up sharding design must first prove that the selected bitsandbytes parameter subclass and every `QuantState` tensor survive native DTensor/FSDP2 wrapping. Define a floating packed storage dtype only if required and supported; do not cast packed bits numerically. Define which metadata is replicated, its device ownership, original logical dimensions, and reconstruction after all-gather. Partition frozen quantized bases and FP32 trainable adapters at explicit wrapping boundaries; ensure all-gather/release does not requantize or lose subclass metadata. Construct the optimizer after final wrapping. Qualify training, adapter and full checkpoint extraction, fresh-process reload, and resume for each exact mesh. Initially require the same mesh/topology; topology-changing resume needs separate evidence. An upstream FSDP1 QLoRA example does not satisfy these requirements. Do not add FSDP1 or another distributed stack to reproduce it.

## Adapter artifacts and exact resume

Introduce an optional `ModelCheckpointCodec`, outside the algorithm-facing policy protocol:

```python
class ModelCheckpointCodec(Protocol):
    def state_module(self) -> nn.Module: ...
    def metadata(self) -> ModelCheckpointMetadata: ...
    def validate_live_state(self, optimizer: Optimizer) -> None: ...
    def validate_saved_metadata(self, saved, *, purpose: Literal["reload", "resume"]) -> None: ...
    def validate_saved_inventory(self, inventory: Mapping[str, TensorSpec]) -> None: ...
```

The state module contains references to the live adapter Parameters under canonical names. It contains no quantized base, excluded parameter, or buffer, and is not registered as a child of the full network. Do not register the original adapted linear modules in the view. Build it after all transformations and validate `id(view_parameter) == id(live_adapter_parameter)` and exact optimizer membership on every save/load. Copy-based restoration must preserve those identities.

Use the view for both model and optimizer DCP state APIs, including restoration. Filtering after full-model extraction is too late. Preserve the existing never-stepped optimizer guard, saved group options, and scheduler restoration. Verify pinned Torch behavior with both never-stepped and stepped optimizer round trips. Ordinary models without a codec retain their existing behavior.

The versioned metadata document, `cosmos.adapter-model.v1`, contains:

- Base repository/revision, weight/config hashes, native model implementation commit and format.
- Quantizer/backend versions, block size, nested settings, packed storage, compute dtype, source quantization dtype, actual targets/exclusions and logical/packed shapes.
- Fingerprints of packed bytes and every quantization-state field, including nested `state2`, offsets, code maps, and scale tensors; excluded-base hashes for invariance/identity checks.
- Adapter method/rank/alpha/dropout/scaling/initialization/dtype, exact resolved tensor inventory, and additional trained modules (empty initially).
- Processor/tokenizer/template file hashes and existing semantic identity; dependency versions and hardware/strategy row.
- Ordered optimizer parameter names, optimizer type/static group configuration, scheduler configuration, and checkpoint purpose/topology requirements.

Use sorted canonical names, explicit dtype/shape descriptors, exact tensor bytes, and versioned canonical encoding. Hash one tensor at a time through plain raw byte views; never cast `Params4bit` to another dtype to hash it. Reject incomplete nested state and canonical-name collisions. Record original base identity and effective quantized-base identity separately. Do not compare the freshly initialized learning rate with the saved scheduled learning rate: static optimizer identity and restored progress are distinct.

Add physical component `model_metadata` to logical family `model` in [manager.py:45](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/checkpoint/manager.py#L45). Write `model_metadata.pt` whenever the model component is selected, including model-only artifacts. Do not put it solely in `trainer.pt`, and do not change the outer artifact protocol's exact top-level schema; the existing named-component map can carry it.

| Artifact                         | Required contents and semantics                                                                                                                                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Portable Cosmos adapter artifact | Consolidated adapter tensors plus metadata and existing file-SHA256-bound commit/manifest. It identifies/reconstructs the frozen base; it is not a full training resume or a generic PEFT/HF artifact.                                     |
| Resumable native checkpoint      | `model` adapter tensors, `model_metadata`, `optim`, `trainer`, and `data`. Trainer state includes applicable scheduler, Python/NumPy/Torch/CUDA RNG, clocks, accumulation/update-control and validation state; data contains exact cursor. |

Use local POSIX DCP for initial full resume qualification and local consolidated format for the portable adapter artifact. Preserve `model.pt = {"net": adapter_state}` with the metadata sidecar and the existing consolidated artifact file digests. Current DCP partial-artifact descriptors contain paths rather than per-file content digests; do not claim equivalent digest binding or advertise portable DCP adapters in v1. Object-store and remote archive round trips require separate qualification. Model-only artifacts remain in the existing partial-artifact namespace and must not advance the full-resume latest pointer.

Save only after a completed optimizer step and accumulation window; no partial-accumulation promise. Validate resident state and identities, extract adapter/optimizer state, capture controls/data, write bulk state then metadata/data/trainer controls, and publish the existing commit/latest last. Keep trainer control last because legacy completeness probing uses its presence. Still require all QLoRA components on restore. Reuse fsync/replace, archive/recovery and stable-provenance mechanisms.

Initial synchronous DCP qualification must explicitly pass `async_safe_snapshot=False`. `save_async=False` alone does not disable the independently scheduled DCP control tail: [manager.py:573](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/checkpoint/manager.py#L573).

```python
# Fresh-process adapter reload
metadata = read_committed_adapter_metadata()
validate_source_processor_dependency_strategy_contract(metadata)
net, optimizer, codec = construct_normal_frozen_base_and_adapters(metadata)
verify_reconstructed_effective_base_fingerprint(metadata)
verify_exact_saved_adapter_names_shapes_and_dtypes(metadata)
load_adapters_by_copy(codec.state_module(), strict=True)
evaluate_controlled_text_image_video_batches()

# Full resume adds:
require_full_model_optimizer_trainer_data_metadata_components()
validate_required_rng_topology_and_static_optimizer_identity()
restore_optimizer_and_scheduler_and_update_control()
restore_exact_data_cursor()
restore_checkpoint_rng_after_all_initialization_randomness()
continue_from_next_optimizer_update_without_reseeding()
```

Read small metadata before expensive construction to reject incompatible requests, then validate again within the authoritative stable-provenance load. Effective-base byte verification happens after reconstruction. Existing DCP inventory exposes names/shapes; extend it to include dtype and reject extra source tensors before DCP fills targets. Consolidated copies may cast, so validate saved dtypes before copying. Enforce required RNG explicitly because legacy restoration tolerates missing/world-size-mismatched RNG. If provenance changes after load has mutated live state, terminate or reconstruct before retrying; provenance checking is not rollback.

Initial reproducibility is numerical continuation on the same one-rank hardware/software/configuration/data row. Compare exact cursor, clocks, RNG payloads and metadata, and controlled losses/adapter/optimizer trajectories within empirically established bounds. Do not promise bitwise continuation or topology changes based on `deterministic=True` alone.

## Reload and export mathematics

Training uses `Wq = dequantize(Q(W))` and ordinary adapter update `Delta = (alpha/r) BA`, with the actual branch casts described above. Here `Q` includes the explicitly recorded BF16 source-normalization step before NF4 quantization.

| Operation                                                  | Numerical contract                                                                                                         |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Reload adapter on the same quantized base                  | Initial parity target: same effective base, adapter branch, processor, and dtype policy.                                   |
| Export dequantized quantized weights plus adapter          | Dense `Wq + Delta`; merge and matmul rounding differ from the training branches. Requires separate logits and task checks. |
| Apply or merge adapter onto original higher-precision base | `W + Delta`, a different model because generally `W != Wq`. Requires separate checks.                                      |
| Requantize a merged model                                  | Another quantization error term and potentially another backend. Requires separate qualification.                          |

Do not expose automatic in-place four-bit merging. Guard native serving weight publication before the mapper walks packed tensors/metadata, and guard dense automatic/offline export. No universal serving-backend compatibility is implied by a portable Cosmos adapter artifact.

## Correctness and failure tests

CPU/mocked tests may validate schema, resolution, lifecycle order and checkpoint selection. They do not qualify kernels, memory savings or task quality. Add real accelerator gates for every advertised matrix row.

| Test                    | Required assertion                                                                                                                                                                                 |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Coverage and exclusions | Exactly 252 expected quantized matrices and 504 adapter tensors; actual backend/format/storage/compute/state agrees with metadata; no excluded or tied alias quantized.                            |
| Frozen invariance       | Packed bytes, complete quantization state, and excluded frozen tensors remain unchanged across several updates; no frozen gradients or optimizer membership.                                       |
| Trainability            | Optimizer identities equal intended live adapter identities exactly; finite losses and real updates on text, image and video batches; permit first-step zero A gradients.                          |
| Conditioning            | Existing real media path runs; changed media changes controlled model outputs; media tensors/grids and loss masks agree between BF16/QLoRA arms.                                                   |
| Loading and movement    | No full dense GPU destination or retained full CPU source dictionary; initialized rotary buffers preserved; same-device movement does not requantize. Instrument actual peaks and quantizer calls. |
| AC and RNG              | First-block adapters train with frozen embeddings; wrapper resolution remains valid; adapter initialization hashes match between arms; canary leaves RNG/global flags unchanged.                   |
| Adapter reload          | Fresh process reconstructs identical effective-base fingerprint and exact adapter inventory; controlled logits/loss match justified tolerances on all modalities.                                  |
| Resume                  | Uninterrupted versus stop/save/fresh-process/resume matches exact controls/cursor and declared numerical trajectory; include never-stepped optimizer serialization and stepped moments.            |
| Checkpoint isolation    | Make full-model state extraction raise; adapter codec save/reload/resume must still work. Verify live/view/optimizer identities and no base-sized checkpoint snapshot.                             |
| Failure handling        | Wrong revision/processor, missing metadata, incompatible dtypes/strategy, unmatched targets, incomplete state, extra tensors, stale views and interrupted commits fail explicitly.                 |
| Compatibility           | Existing dense/full-parameter training, dense LoRA, checkpoint artifacts and export tests retain their behavior.                                                                                   |

Calibrate kernel forward/input-gradient tolerances against the same dequantized base using repeated controlled runs. Freeze tolerance values in the qualification manifest before full comparison; they must exceed observed kernel noise without masking wrong identities, absent media, or absent updates. Reload and resume have separate tolerances. Do not invent a single tolerance for all outputs or silently widen one after failure.

## Reproducible benchmark and release evidence

Run the paired BF16-LoRA and QLoRA recipes with identical immutable base, accepted training/held-out rows, processor/template, exact target list, rank/alpha, sequence/media limits, effective batch, optimizer, schedule and seed policy. Initial recipe: 300 optimizer steps, microbatch one, accumulation eight, maximum 4096 tokens, eight video frames, and full AC. Use paired seeds 17, 42 and 123, overriding `rl.seed`, `adapter.initialization_seed`, validation seed policy, and launch `PYTHONHASHSEED` identically in both arms. Dedicated generator initialization in sorted canonical order must produce bit-identical initial A/B tensors between arms; equal `rl.seed` alone is insufficient because construction consumes RNG.

Use VideoPhy-2-derived posttrain conversation rows as the candidate video adaptation workload, retaining the repository's real media conditioning. The [existing preparation utility](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/scripts/reasoner/prepare_videophy2_from_hf.py) and recipes are starting points, not a qualified posttrain dataset. Freeze source dataset revision, conversion code, media hashes, train/validation/test identities and leakage policy before execution. Add fixed image/text regression sets. The exact manifests and task evaluator are release dependencies; do not invent their hashes or call synthetic smoke rows useful held-out quality.

Native held-out validation currently reports supervised-token CE/counts/time, not task accuracy or correlation: [sft.py:349](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/cosmos_framework/posttrain/entrypoints/sft.py#L349). Add a deterministic, versioned task readout on the same native model; pin prompts, scoring/parsing, decoding and invalid-output policy. Specify the primary task metric before training. Compare against both frozen BF16 and frozen quantized bases to distinguish quantization damage from adaptation benefit. The baseline-capability regression set must cover text, image and video behavior relevant to the claimed support.

Measure synchronized phases independently: loading, first forward/backward/optimizer step, warmed steady state, validation, checkpoint extraction/write, fresh reload and resume. Record per-rank peak allocated and reserved GPU memory, NVML/device-process memory including allocations outside PyTorch, sampled host RSS/PSS, pinned host memory where available, elapsed time, tokens/second, examples/second, update time and artifact size. Record sampling cadence; external sampled peaks are lower bounds. Record page-cache/cold-cache policy and media decoder worker memory. Do not infer total training memory from four-bit weight size.

Archive hardware SKU/count/VRAM/driver, runtime versions and wheel/lock hashes, CUDA/kernel settings, repository SHA, resolved configuration, actual sequence and media distributions, accumulation, AC, dataset hashes and all per-attempt logs. Existing output uses a run-attempt directory and `metrics.jsonl`; resume trims metrics after the restored step, so preserve interruption evidence separately. Existing `timer/e2e_step_s` is insufficient for phase memory measurement.

Predeclare these workload-specific candidate release gates before launching the paired runs:

- For every paired seed, at least 6 GiB lower maximum device-process memory over the complete measured training lifecycle than the matched BF16-LoRA control, with both arms completing the same workload. Report allocator and host peaks as well; reject a loader that merely trades the GPU peak for an unbounded host copy.
- The one-sided 95% upper confidence bound on relative held-out supervised-token NLL increase versus the control must be at most 2%.
- For the preregistered task metrics, the one-sided 95% upper confidence bound on regression must be at most 1 percentage point absolute accuracy loss or 0.02 correlation loss, as applicable. The one-sided 95% lower bound on the primary task improvement over the frozen quantized baseline must exceed zero. Freeze which metric applies before observing results.
- Use 10,000 deterministic paired hierarchical bootstrap replicates (bootstrap seed 2026), resampling paired training seeds and independent evaluation units, clustering related rows by source image/video. Retain pairing across arms and evaluate each metric's declared aggregation. Freeze the evaluation units and aggregation in the manifest. If the three-seed data cannot resolve a margin, report inconclusive qualification rather than declaring success or changing the gate after observing results.
- All correctness/reload/resume gates pass on the advertised row. Report throughput and elapsed-time cost even if memory and quality pass.

These are proposed gates for this exact workload, not measured results or universal QLoRA percentages. A separately tuned best-fit QLoRA recipe may change batch/media/AC settings, but must be reported separately from the controlled comparison with every unavoidable difference disclosed.

## Ordered patches and acceptance criteria

Paths in this table are repository-relative. New modules and recipes are proposals. The posttrain mapping workflow must preserve the corresponding upstream source changes; a downstream-only patch that is overwritten by sync is incomplete.

| Stage                        | Exact files                                                                                                                                                                                                                                                                                  | Acceptance before dependent work                                                                                                                                                                                                                       |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A. Native ordinary LoRA      | `cosmos_framework/utils/generator/lora.py`; `cosmos_framework/posttrain/config/model.py`; `cosmos_framework/posttrain/model/qwen3_vl.py`; `cosmos_framework/posttrain/config/build.py`                                                                                                       | Reuse native naming/initialization; BF16 base frozen; FP32 adapter precision accepted; exact optimizer membership; real multimodal updates and dense compatibility.                                                                                    |
| B. Immutable source identity | `cosmos_framework/posttrain/model/sft_source.py`; `cosmos_framework/posttrain/utils/artifact_identity.py`; `cosmos_framework/posttrain/data/packer/sft.py`; `cosmos_framework/posttrain/model/qwen3_vl.py`; `cosmos_framework/posttrain/entrypoints/sft.py`                                  | Model and SFT processor consume the same verified snapshot; revision or processor mismatch fails before allocation. Independent of quantization.                                                                                                       |
| C. Capability boundaries     | Native builder and weight mapper; `cosmos_framework/posttrain/roles/trainer/llm.py`; `cosmos_framework/callbacks/hf_export.py`; dense adapter merge helpers                                                                                                                                  | Early errors for unsupported distribution/offload/method/export, before quantized construction becomes reachable; resident save remains safe; no fallback. Required by D.                                                                              |
| D. Quantized lifecycle       | New `cosmos_framework/posttrain/model/quantization/config.py`, `bnb.py`, `loader.py`; update native builder; `docker/posttrain/environment/pyproject.toml` and `docker/posttrain/environment/uv.lock`                                                                                        | Exact optional backend pin; real kernel canary; streamed loading; complete coverage/state; preserved buffers and gradients; bounded host/device loading peaks. Depends on A/B/C.                                                                       |
| E. Adapter persistence       | New `cosmos_framework/posttrain/model/checkpoint.py`, `cosmos_framework/posttrain/checkpoint/adapter.py`; `cosmos_framework/posttrain/roles/trainer/llm.py`; `cosmos_framework/posttrain/checkpoint/manager.py` and `store.py`; SFT preflight                                                | Adapter-only extraction and metadata family; same manager/commit/recovery; typed inventories; stale-view detection; pristine/stepped optimizer round trips; fresh-process reload/resume; fault injection. Codec can first land with A, then qualify D. |
| F. Qualification and recipes | New `cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_lora.py`, `sft_qwen3_vl_8b_qlora.py`; new `test_model_qwen3_vl_qlora.py`, `test_model_qwen3_vl_qlora_gpu.py`, `test_checkpoint_adapter.py` under `cosmos_framework/posttrain/tests/`; extend existing optimizer/SFT/checkpoint tests | Frozen benchmark manifests, CPU/accelerator correctness, reload/resume, lifecycle memory and held-out quality reports. No support claim before results.                                                                                                |

Do not start by adding a `load_in_4bit` flag to a dense loader. Do not fork the native trainer, introduce a second distributed stack, reuse inference quantization configuration, blanket-upcast the VLM, use broad target matching, filter a full checkpoint after extraction, or promise merge parity with original higher-precision weights. Each loses a required loading, precision, identity, or numerical contract.

Primary risks are bitsandbytes compatibility with the pinned Torch/CUDA combination, initialization/materialization preserving quantization metadata, FP32 adapter branch cost, media/activation memory dominating savings, strict reconstruction determinism, and quantization reducing useful task quality. FSDP2/DTensor compatibility is a separate risk and is excluded from the initial capability.

## Proposed pull request descriptions and merge evidence

Design-only PR title: **Design native Qwen3 VL QLoRA training and qualification**.

> Specify a native posttrain SFT path for pinned Qwen3-VL-8B with a frozen NF4 language base and ordinary FP32 LoRA. Define streaming construction, exact component precision and coverage, native adapter checkpoint/reload/resume contracts, unsupported strategy/export errors, and a matched BF16 benchmark. This PR changes documentation only and makes no runtime support claim.

Subsequent implementation PR title: **Add qualified native Qwen3 VL NF4 adapter training**. Use this title only after attaching the qualified row and evidence; before then mark the implementation draft/unqualified.

> Train ordinary adapters through native Cosmos SFT while keeping the selected language matrices frozen in NF4. Preserve actual image/video processing and existing loss masks. Stream source weights into final storage and publish adapter-only artifacts with exact base/processor/quantizer identity; resume through the existing checkpoint manager. Initial support is the documented one-A100 row. Other distributed strategies, offload, and merged/serving exports fail explicitly.

Required implementation merge evidence: exact environment/model/data manifests; resolved coverage and trainable inventory; real-kernel and multimodal update logs; frozen-state hashes; checkpoint fault-injection results; fresh-process reload and resume comparisons; matched BF16 memory/throughput/quality reports with uncertainty; and existing higher-precision regression results.

At the time of this design audit, the host had A100 hardware, but the selected quantized runtime, complete immutable model snapshot, benchmark data and task evaluator were not qualified. No accelerator kernels, training, reload/resume experiments, or memory/quality benchmark results were supplied by this design. The design PR was documentation only; see the [native QLoRA guide](native_qlora.md) for subsequent implementation evidence and remaining release gates.
