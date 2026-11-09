# Portable adapter inference

`CosmosFrameworkRuntime` accepts a local immutable base snapshot and
`portable_adapter={"directory": ..., "execution_profile": ...}`. The same
dictionary is `[model.portable_adapter]` in the reasoner evaluator TOML. A public
generation example is [portable_adapter.toml](../../examples/reasoner/portable_adapter.toml):

```bash
python -m cosmos_framework.inference.reasoner.generate \
  --config examples/reasoner/portable_adapter.toml --output answer.json
python -m cosmos_framework.evaluation.reasoner.evaluate --config evaluation.toml
```

In both commands `model.model_name` is the original base snapshot. The portable
directory must contain `adapter_manifest.json` and `adapter.safetensors`.
This path uses the shared artifact/adapter primitives directly; it does not use
a training builder, qualification factory or an evaluation-only loader override.
Existing merged-model configuration is unchanged: omit `portable_adapter` and
point `model_name` at the completed export.

| Explicit profile | Artifact / deployment precision | Computation |
| --- | --- | --- |
| `qwen3_vl_bf16_canonical_cache_v1` | v4 `merged_weight_bf16_v1`; BF16 frozen base and FP32 adapter storage | Existing canonical linear/attention arithmetic and rounded effective weights. Optional `cache_effective_weights=true`. |
| `qwen3_vl_factorized_fp64_branch_v1` | v3 `qwen3_vl_hf_stock_v1`, `factorized_v1`; BF16 base loaded then explicitly promoted to FP32 | FP32 frozen base projection, FP64 separate low-rank projections and final sum. PiSSA subtracts its saved initial branch; DoRA computes its direction norm and magnitude ratio in FP64. No dense effective-weight forward or cache. |

Both public profiles are inference-only: exact stock dense HF Qwen3-VL,
Transformers 5.8.1 / PyTorch 2.10.0, a single A100 SM80, SDPA, batch one, eager
`eval()` and `no_grad()`/`inference_mode()`, no autocast, compilation, CUDA graphs
or CPU/disk offload. Canonical additionally requires Triton 3.6.0. Stable
factorized requires FP32 matmul without TF32 (`torch.set_float32_matmul_precision("highest")`).
Ordinary non-offloaded `DynamicCache` and the existing runtime's video-prefix
reuse are supported; static/custom caches are rejected. These profiles do not
claim support for arbitrary checkpoints, other hardware, serving backends or
training/autograd. Qualification results are recorded separately from API availability.

The stable factorized profile is a deployment override recorded in output
provenance. It leaves the original artifact, tensors, ties and trainability
unchanged. It is not a new training execution mode and does not erase the
original final `factorized_v1` strict merge failure. Call
`runtime.restore_factorized_execution()` to restore the original factorized
forward with the same FP32 tensors. Installing a conflicting profile fails.
Its FP32 storage and transient FP64 branch/norm work cost more than BF16
canonical execution; consult the qualification measurements before deployment.

Loading checks the payload checksum and complete tensor schema, base weight
hashes, processor/tokenizer/template assets, dependency versions and execution
identity. It resolves concrete module placement and checks that identity before
injection, loads the saved PiSSA initial factors without decomposition, then
installs the profile. `processor_path` optionally selects an identical staged
processor snapshot, never different assets. The runtime records the training
media contract as well as effective generation/media preprocessing settings;
the evaluator saves `adapter-deployment-shard<N>.json`. Generation overrides do
not rewrite the portable training identity.

Derived-weight caches never enter model state or optimizer ownership. Normal
in-place updates, `load_state_dict`, replacement, dtype/device or adapter setting
changes invalidate derived weights and runtime-owned feature/KV caches at
generation entry. Call `runtime.clear_adapter_caches()` after unsafe `.data`
writes or externally executed fused optimizer updates, which may bypass tensor
version counters. Such modifications invalidate the claim that execution uses
the unchanged saved artifact; reload for an exact artifact-bound deployment.
Do not mutate a runtime concurrently with generation. Caller-owned caches from
direct `runtime.model` calls must be discarded by the caller after a mutation.
Construct runtimes outside `inference_mode()` so source tensors have version
counters. Direct model calls retain the profile's inference-only guards.

Plain Transformers loading of a portable directory is unsupported. Merged BF16
canonical artifacts also require the Cosmos loader to activate their declared
arithmetic; the existing FP32 stock merged model remains the comparison oracle.
