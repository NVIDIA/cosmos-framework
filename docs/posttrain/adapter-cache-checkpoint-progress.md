# Checkpoint extraction and canonical BF16 cache continuation

Starting revision: `95b00e9`. Historical results remain 12 named checkpoints,
48 fixed profiles and 47 fully passing legacy profiles. This continuation does
not replace their precision, checkpoint, backend or benchmark identities.

| Requirement | Implementation | Qualification |
|---|---|---|
| Selective DCP/consolidated adapter extraction | `checkpoint/adapter_extract.py`; existing checkpoint boundary/provenance | Public MoE DCP fresh/reuse under 16 GiB whole-job cap; public consolidated and DCP recovery under 8 GiB; exact 882/625 tensors; stock-loader check pending |
| Canonical BF16 cached inference | `utils/generator/bf16_cache*.py`, explicit portable API and tagged merged loader | Operator/ownership/metadata source tests and installed 6a1a36d Qwen3-VL 96/104-case reload/cache/merge/resource checks pass |
| Matched full WTS training | `configs/sft_wts_execution_comparison.py`, explicit factorized stock-HF5 control | Both lifecycle preflights pass; full one-epoch jobs running on immutable `4423805` |
| Independent review | Local review package | Extraction/control reviewed; cache profile review in progress |

## Explicit inference contract

`qwen3_vl_bf16_canonical_cache_v1` is an optional inference override for dense
Qwen3-VL v4 `merged_weight_bf16_v1` adapters. It preserves BF16 weights,
activations and KV storage. Language projections/head use fixed 32×64×64 tiles
with FP32 accumulation; attention uses fixed 64-key reductions in FP64 without
a full attention matrix. This is not whole-model FP32 transient execution.
The training execution and portable tensor format are unchanged.

It requires Torch 2.10.0, Transformers 5.8.1, Triton 3.6.0, one A100 (SM80),
batch one, eager eval under no_grad/inference_mode, SDPA architecture before
installation, and ordinary DynamicCache without offload. Training, autocast,
compile/graphs, static/quantized/offloaded caches and other serving backends are
not supported by this profile. FP32 equivalence remains a separate stock-loader
contract; it must not acquire this override.

Export explicitly with `--output-dtype bfloat16
--execution-profile qwen3_vl_bf16_canonical_cache_v1`. The config and transaction
marker bind this choice; `CosmosFrameworkRuntime` validates and installs it.
Plain Hugging Face `from_pretrained` alone does not install the declared custom
arithmetic. For portable inference, first use the strict portable loader, put
its net in eval mode, then call
`posttrain.model.adapter_bf16_execution.install_adapter_inference_profile(net,
"qwen3_vl_bf16_canonical_cache_v1")`. Parameter objects, ties, factor ownership,
trainability flags and optimizer membership remain unchanged; inference calls
cannot silently detach a training graph.

The unmodified legacy BF16 failures, original populations and all numerical gates
remain in the local evidence package. Qwen3.5's separate `qwen3_5_bf16_canonical_cache_v1` uses the same linear and
attention kernels plus one FP64 gated-delta recurrence for prefill, suffix and
decode, rounding recurrent state to FP32 after every token. It allocates request
state in ordinary DynamicCache without global Transformers patches; recurrent
state is FP32, convolution and full-attention KV remain BF16. Source operator and
multimodal cache tests pass; installed real-model qualification is pending.

Portable training builders derive generation defaults from the model config.
For cross-loader generation, explicitly use the same saved base generation
configuration, including both EOS IDs. The initial comparison retained different
EOS masks on closed-chat training probes: all first logits were exact, but
continuations differed. Its failed records remain; controlled decode v2 uses the
same EOS/pad/BOS settings for both loaders and retains every input and threshold.

## Export and evidence boundaries

Selective checkpoint reads retain full-checkpoint completeness and immutable
source checks. DCP extraction allocates requested adapter/extra/PiSSA initial
state only. Consolidated extraction reuses mmap and clones compact selected
state; it does not deserialize optimizer state. Existing transaction locking,
interrupted staging recovery and idempotent completed-export reuse remain.
Physical-format full-model exporters have not acquired bounded-memory
qualification from these HF results.

Local evidence: `/localhome/local-rarunachalam/cosmos-adapter-cache-checkpoint-20261105`.
Whole-WTS comparison: 5,555 training and 2,676 validation rows, then the reused
576-row scene-disjoint MCQ benchmark. Both modes use the same checkpoint,
components, data, budget and corrected preprocessing. This benchmark is not an
untouched final test; physical-model correctness does not establish downstream
physical task quality. No HPO, four-bit work, push or PR publication is included.
