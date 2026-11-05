# Final trained-artifact deployment

This continuation starts at `bf622ea` and evaluates the final update-1111
Qwen3-VL-2B-Instruct mixed-component WTS artifacts. It does not repeat training
or change the historical 12-checkpoint / 48-profile / 47-pass qualification.
The reused scene-disjoint 576-question benchmark is part of validation, not an
untouched final test. Physical-model correctness is not task-performance evidence.

Production changes are `62a25aa` (derived-weight cache and attention launch) and
`dc11a03` (portable inference modality assembly). The final package was built
from clean `dc11a0343b4e1f05e8ed41cbb18fbfa0a64bc0b4`; its wheel SHA256 is
`e416acee9b57b5a11973043b15df09c6d32b363c9ac3513230b5db70a45a5a61`.
All 1,914 packaged source files were checked against the clean archive and
installed environments. Later documentation does not relabel earlier runs.

## Requirement-to-code-and-test matrix

| Requirement | Implementation / evidence | Final status |
| --- | --- | --- |
| Final update-1111 BF16 portable and merged deployment | Existing strict portable loader plus `model/adapter_bf16_execution.py::install_adapter_inference_profile`; existing public export and `CosmosFrameworkRuntime` merged loader | Validated on this final mixed-component checkpoint: 108 inputs per loader, unchanged gates; portable/merged logits and generated sequences are bitwise equal |
| Full versus incremental decode and repeated prefix use | Existing canonical kernels/runtime; `qualify_final_cache.py`, `qualify_final_horizons.py` in local evidence | Validated: all 108 inputs, 162 prefix executions per loader, plus ten extended cases per loader covering 32 generated tokens and 2K/8K contexts; prefix ownership/growth gates pass |
| Preserve original canonical correctness reference | Unmodified `_attention_kernel`; bounded `_attention_warp_count` launch selection | Validated: all 108 old/new outputs bitwise equal; actual stock-loaded final model also equals the four-warp reference at 2K/8K, with and without caching |
| Portable weight reuse, invalidation and ownership | `model/adapter_inference_cache.py`; `PosttrainAdapterLinear._apply/train`; focused cache tests | Validated: source/alias updates, load/replacement, dtype/device, bound fused optimizers, partial step failure, explicit clearing, unchanged state and optimizer ownership; derived memory reported separately |
| Remove portable-only inference work | Inference installer restores pinned stock inner forward after validation | Validated: text/image/video/both and decode/prefix visual-call counts, outputs, ownership, cache behavior; ordered training and Qwen3.5 are unchanged |
| Additional deployment performance budget | Predeclared `inputs/protocol.json`; `performance-summary.json` | **Incomplete:** 8K prefill is 4.236 s against a 2.0 s budget. 2K/8K stock-BF16 ratios and the 8K FP32 ratio also miss their gates. Portable-overhead gates now pass |
| Final checkpoint task drift | Public evaluator, identical preprocessing/decoding and same 576 questions | Passed: portable and merged BF16 each 527/576, zero answer changes versus this checkpoint's FP32 reference; paired row interval [0, 0] percentage points |
| Original final factorized control | Unmodified `factorized_v1` portable model evaluated in FP32 | Task uncertainty resolved: 527/576 and zero changes versus its merged export. **Strict merge remains incomplete:** the preserved input-12 failure is not waived |
| Stable factorized accumulation investigation | Two separate, bounded FP64 diagnostics; no production algebra change | Both diagnostic modes pass the existing 32-input gate, retaining separate low-rank branches. These are changed execution semantics, not support for the unchanged legacy artifact |
| Installed validation and independent review | 91 unique focused tests on `62a25aa`; 40 affected executions on `dc11a03`; two independent reviews | Passed in their recorded execution profiles. Initial fused-optimizer invalidation finding fixed; no remaining confirmed defect in the reviewed production delta |

Model/checkpoint paths above are relative to `cosmos_framework/posttrain` unless
otherwise specified. These are repeated qualification executions of one final
artifact, not additions to the historical 48 fixed checkpoint/method profiles.

## Exact artifact and execution scope

Base: `Qwen/Qwen3-VL-2B-Instruct`, revision
`89644892e4d85e24eaac8bacfd4f463576704203`. Both final checkpoints are at update
1111, after the already completed matched one-epoch WTS experiment. The new-mode
portable weight SHA256 is
`dd098738120826d75c22f3a0c3188dd0d58b2de307f9776fa4f79f383a5902df`;
the factorized control's is
`a90b742cfff0d1fbcf97305d3b96bcf3d2efe0047f03cef6f954ffc945c46846`.
The canonical BF16 merged materialization identity is
`54908c5f59dbd88532c6b8767cd9258b1bb28b8853fe2fd2d92efff60b36c2e0`.
`artifact-catalog.json` records full paths, all canonical/stock export file hashes,
base/processor/template identities, original training receipts and source markers.

The recipe has 65 targets: 56 language q/v projections use ordinary LoRA
(rank 8, alpha 12, LR multiplier 1, weight decay .01); the merger/deepstack's
eight `linear_fc1/linear_fc2` targets use rsDoRA (rank 4, alpha 9, LR .3);
`model.visual.blocks.23.attn.qkv` uses rsPiSSA (rank 3, alpha 6, LR .1).
`model.language_model.norm` is separately trained (LR .07). Dropout is zero;
adapter/extra state is FP32, the base and canonical effective weights are BF16.
The portable format remains v4, with `merged_weight_bf16_v1` construction and
preserved PiSSA initial factors. No artifact tensors or training mathematics changed.

Deployment is the explicit `qwen3_vl_bf16_canonical_cache_v1` profile: one A100
80 GiB / SM80, batch one, eager inference, ordinary DynamicCache, Torch 2.10.0
CUDA 12.8, Transformers 5.8.1, Triton 3.6.0. The tested merged loader is Cosmos's
runtime, which validates and activates the declared profile on the stock HF model.
Plain Transformers consumers that ignore the profile, other serving backends,
CPU/offload, compile/CUDA graphs, other GPUs and larger contexts are not qualified
by these results. No new distributed training configuration was added.
Portable evaluation uses the strict HF builder/warm-start API and the explicit
inference installer; an evaluation-only factory bridge connects that model to
the unchanged public evaluator. This is not a new portable-directory consumer
for plain Transformers or the merged-model CLI.

## Portable inference weight reuse

After strict portable loading, the explicit canonical inference installer accepts
`cache_effective_weights=True`. It reuses the existing rounded effective-weight
construction for LoRA, rsLoRA, DoRA and PiSSA, without changing factors, initial
factors, magnitude, base parameters, optimizer membership or checkpoint state.
The default remains uncached. This cache is separate from the KV/prefix cache.

```python
from cosmos_framework.posttrain.model.adapter_bf16_execution import install_adapter_inference_profile
from cosmos_framework.posttrain.model.adapter_inference_cache import (
    adapter_inference_cache_info,
    bind_adapter_inference_cache_optimizer,
    clear_adapter_inference_cache,
)

# net is the strictly loaded portable model, on the supported GPU, in eval mode.
install_adapter_inference_profile(
    net, "qwen3_vl_bf16_canonical_cache_v1", cache_effective_weights=True
)
print(adapter_inference_cache_info(net))
# Optional: bind every optimizer that may update parameters between inference calls.
handle = bind_adapter_inference_cache_optimizer(net, optimizer)
# Use no_grad/inference_mode for inference; the canonical profile cannot train.
# Unsafe .data/custom writes that bypass tensor versions require explicit clearing:
clear_adapter_inference_cache(net)
# Remove the binding when no longer needed; reinstalling with caching disabled
# also removes bindings and releases all derived tensors.
handle.remove()
```

Version-tracked in-place writes, tensor replacement, dtype/device conversion,
state loading and entering training invalidate derived weights. Fused optimizers
can bypass tensor versions: bind them explicitly as above or clear after each
update. Bound optimizers clear before and after stepping, including partial
updates followed by an exception. Unbound fused/custom writes are not detected
automatically. Do not mutate weights concurrently with inference. Derived tensors
are neither Parameters nor buffers and are never exported or checkpointed.

The explicit single-GPU Qwen3-VL canonical inference profile also uses the pinned
stock modality forward, matching the merged loader. It skips the training
forward's dummy vision executions for absent media, including incremental decode.
Real per-modality batches, deepstack features and cache/position handling stay
stock. The training builder and ordered cross-rank forward are unchanged;
Qwen3.5 is unchanged. Installing the inference profile still forbids training.

## Canonical attention optimization

The original fixed-tile kernel remains the correctness reference. Long Qwen3-VL
prefill with head dimension 128 and at least 1024 query positions uses a one-warp
launch to reduce cross-warp exchanges. It retains the same 64-key FP64 reductions,
online recurrence and BF16 output. Other architectures, head dimensions, short
inputs and decoding retain the original four-warp launch. This does not introduce
whole-model FP32 execution or change the execution-profile identity.

The deployment matrix was fixed before candidate evaluation, including a
two-second 8K prefill budget and comparisons against stock BF16 and FP32. Passing
numerical checks does not imply that the additional performance criteria pass.
Final installed-package measurements and task results are recorded separately.

The following are warm medians in seconds; short inputs are a fixed
text/image/video group, decode is one token, and prefix reuse generates four
tokens. Measurements use the predeclared inputs/repetition counts. Stock columns
are performance comparators, not claims that stock BF16 passes canonical cache gates.

| Operation | Old canonical merged | Revised merged | Revised portable | Stock BF16 | Stock FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short multimodal group | .1982 | .2074 | .2093 | .1997 | .2325 |
| 2K prefill | .4488 | .3439 | .3458 | .0583 | .4664 |
| 8K prefill | 6.0715 | 4.2359 | 4.2576 | .2424 | 3.0491 |
| Decode after 2K | .0392 | .0404 | .0427 | .0410 | .0434 |
| Decode after 8K | .0443 | .0441 | .0443 | .0410 | .0440 |
| Reused video prefix | .1722 | .1714 | .1795 | .1762 | .1803 |

The revised merged prefill is 23.4% faster at 2K and 30.2% faster at 8K than the
original canonical implementation. It is still 5.90x / 17.47x stock BF16 at
2K/8K and 1.389x stock FP32 at 8K. The predeclared requirements were at most
4x / 8x stock BF16 and .75x FP32 at 8K; those failures remain explicit.
The absolute 2K .35-second gate passes, close to its limit. The 8K two-second
gate fails. Query grouping and transposed/preconverted K/V staging were measured
and excluded because they did not improve 8K performance.

Portable warm short inputs improved from .3967 to .2093 seconds; 8K prefill,
decode and prefix overhead are now within the declared portable gates. It holds
534 MiB of derived BF16 weights, below the .75-GiB gate. Cold first multimodal
execution was 1.385 s; model loading was 6.462 s, recorded separately.
This is a fresh-process model call with an empty derived-weight cache and an
existing Triton disk cache, not a measurement of cold compiler-cache creation.
Merged / portable Parameter bytes are 4,255,064,064 / 4,262,555,648; FP32 is
8,510,128,128. KV bytes are unchanged at 234,881,024 (2K) and 939,524,096 (8K),
half the FP32 reference. 8K prefill peak allocations are 5,670,855,168 bytes
merged and 6,238,335,488 portable, versus 20,834,372,608 FP32. Portable decode's
8K peak is 6,738,185,728 bytes. The derived cache is counted separately, not hidden
in model or KV byte totals. All predeclared peak-memory gates pass.

## Factorized control and review findings

The original final factorized portable FP32 evaluation used the same 576 rows,
corrected media settings and decoding as its existing merged export. All parsed
answers match: 527/576 (91.493%), no invalid answers. This does not repair its
strict merge failure: input 12 still has 13 of 1,215,488 values outside the
unchanged `atol=5e-4, rtol=2e-4` contract. Exported weights were already validated.

Two predeclared diagnostics retained factorized output algebra and parameter
ownership: FP64 adapter/PiSSA-initial projections plus DoRA normalization/output
sum, then also FP64 selected-base projection. Each passed all 32 fixed inputs;
maximum logit errors were 5.9414e-4 and 5.3978e-4 (the pointwise relative tolerance
still applies), with means 1.5491e-5 and 1.3582e-5. At the first divergent target,
`model.visual.deepstack_merger_list.0.linear_fc1`, their maximum differences from
an independent FP64 reference were 1.7166e-5 and zero after FP32 output rounding.
The original failed output was restored bit for bit afterward. Dense weights
were used only to calculate the independent reference, never as a replacement
forward. These are evidence for a possible explicitly versioned accumulation
mode; they are not enabled, have not received wider qualification, and do not
turn the original `factorized_v1` checkpoint into a passing strict merge result.

Independent review identified fused AdamW updates that bypass tensor `_version`.
The fix adds explicit weak-reference optimizer bindings with pre/post invalidation,
including partial failed steps, rebinding and cleanup. Unbound fused/custom/.data
writes still require explicit clearing. Both independent reviews found no other
confirmed defect in their final production snapshots; reviewer CPU/mocked checks
are distinguished from the separately executed real GPU qualification.

## Validation provenance and reproduction

Training remains attributed to `4423805`; portable/FP32 extraction and the
matched FP32 task references to `6a1a36d`; final FP32 oracles and the new BF16
exports/preoptimization checks to `cd60999`. Optimized merged qualification,
task/performance, and 91 focused tests ran on installed `62a25aa` (wheel
`2b491ac9928e2ecadc7534c5673a123560f138a401db6a9c153708d198f89bf2`).
Final portable qualification/task/performance and 40 affected test executions ran
on installed `dc11a03`. These are not 131 unique tests. Two existing CPU fixtures
were initially invoked with CUDA visible; they passed when run in their intended
CPU profile. Both initial logs and corrected runs are preserved.

`validation-provenance.json` hashes the merged runtime/kernel files that are
unchanged between these two new revisions, justifying reuse of merged results.
`task-summary.json`, `performance-summary.json`, `portable-final-oracle-summary.json`
and `tests/installed-test-summary.json` bind results to their actual installed
revisions. The local review package contains the exact scripts, reports, protocol,
artifact catalog, independent reviews, patches and reproduction commands. It
excludes runtime installations and duplicate checkpoint weights. No training,
HPO or four-bit work was run; PR #342 was not reopened and nothing was pushed or
published.

## Evidence location

Local evidence and reproduction scripts:
`/localhome/local-rarunachalam/cosmos-adapter-final-deployment-20261105`.
The original final factorized portable model and its strict merge failure remain
unchanged. Higher-precision factorized-branch diagnostics are separate experiments,
not a silent reinterpretation of `factorized_v1` artifacts.
