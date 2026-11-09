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

## Qualification of the final update-1111 artifacts

The implementation and independent review cover production `00714d6` on
`feat/posttrain-family-adapters`. The clean installed wheel has SHA256
`abfa5bf8e7971468331eb8692efca4d708792f037e26106488e19d5f43aa7c9a`.
All 1,918 packaged files were verified against that revision. Numerical runs on
`2bccfba`, resource runs on `a2bfca4`, and final public evaluation/affected tests
on `00714d6` remain attributed to those revisions. The later production changes
only fix provenance and reject live attention-profile changes; they do not alter
accepted forward arithmetic. Historical `dc11a03` results retain their original
revision and wheel identities.

This qualification uses **Qwen/Qwen3-VL-2B-Instruct**, immutable base snapshot
`89644892e4d85e24eaac8bacfd4f463576704203`, and the existing final update-1111
WTS adapters. The actual mixed recipe has 56 language q/v LoRA targets (rank 8,
alpha 12), eight merger/deepstack rsDoRA targets (rank 4, alpha 9), vision block 23 qkv
rsPiSSA (rank 3, alpha 6), and separately trained final language normalization.
Dropout is zero. This is not qualification of every checkpoint, component recipe
or backend accepted structurally by the adapter format.

| Requirement | Code / evidence | Result |
| --- | --- | --- |
| Public portable loading | `PortableAdapterConfig`, `load_portable_model`, `CosmosFrameworkRuntime`; evaluator TOML and generation CLI | Implemented and validated. Fresh CLI sessions for both profiles match controlled answers. Canonical public cached/full outputs match the preserved programmatic loader bitwise on all 32 inputs. |
| Strict compatibility and ownership | Captured `read_adapter_snapshot` + `load_adapter_state`; pre-injection resolved identity; tied tiny real-Qwen fixtures | Implemented and validated, including wrong base/profile, tensor state, tied embedding/head, optimizer/cache ownership regressions and unchanged frozen/adapter tensors. |
| Stable factorized execution | `factorized_inference.install`, `_forward`, `restore` | Implemented and validated on all 32 preserved plus 16 fixed additional text/image/video/both inputs. Strict FP32 stock-export comparison passes for forward and cached/full generation logits; greedy sequences agree. |
| Caching and inference-only guards | Stable root/direct-forward guards; existing DynamicCache fork; runtime cache signature and explicit clearing | Stable profile passes 48 short cache cases and 12 forced 32-token cases, including repeat-prefix immutability. Autograd, autocast, static cache, profile conflicts and changing SDPA to eager are rejected. Restoration reproduces original failing input 12 bitwise. |
| Task quality | Public evaluator with `[model.portable_adapter]`, final installed 00714d6 | 527/576 correct (91.493%), zero changed answers versus both the original factorized portable FP32 and its stock merged export. No retraining. |
| Production canonical long-prefill performance | Existing canonical kernel remains unchanged | Incomplete: public portable 2K .3469 s / 8K 4.2494 s; the 2K stock ratio,two-second 8K budget,8K stock ratio and 8K FP32 ratio still fail. |
| Experimental attention investigation | Two separately named diagnostic profiles in the review package | Completed as an investigation; neither is enabled in production. The second candidate meets the unchanged latency/memory matrix but fails the expanded cache population, identically to the canonical reference. |

The strict merge gates remain elementwise `atol=5e-4, rtol=2e-4`, mean absolute
logit error at most `5e-5`, and equal greedy positions. Cache-distribution gates,
exact generated IDs and 16 MiB prefix-growth limit are unchanged. The additional 16
inputs were fixed before testing; they reuse a real WTS clip with previously
unused prompts/frame selections and are numerical fixtures, not new task scenes.
The original final `factorized_v1` strict merge failure remains a failure. The
new inference override does not retroactively qualify that training execution.

The 576 rows are the **reused scene-disjoint benchmark**, already used in prior
validation/HPO, not an untouched final test. The paired comparison uses the
same corrected media pipeline, FP32 model precision, rows and decoding. The
TOML requests 32 tokens, but the existing answer-letter evaluator caps effective
`max_tokens` at 10 and temperature at 0; this applied to both historical controls
and the new run. The separate numerical horizon probes actually force 32 tokens.
The new evaluator provenance records the effective settings. Generated-token
NLL is .007896; it is not a training or validation loss. Physical-model runs
remain correctness evidence, without simulator/driving-quality claims.

## Performance and remaining numerical limits

Warm synchronized A100-SXM4-80GB, batch-one latencies in seconds:

| Case | Public canonical portable | Stable FP64-branch portable | Diagnostic tensor-core long-prefill |
| --- | ---: | ---: | ---: |
| Short text/image/video triplet | .2084 | .4577 | .2021 |
| 2K prefill | .3469 | .5567 | .1780 |
| 8K prefill | 4.2494 | 3.1890 | 1.7271 |
| Decode after 8K | .0443 | .1339 | .0443 |
| Four-token repeated video prefix | .1848 | .5480 | .1701 |

Stable factorized inference prioritizes controlled numerical equivalence and
costs more: 8,517,615,616 weight bytes, 1,879,048,192 bytes of 8K KV and
20,844,696,576 peak allocated bytes. It does not meet the BF16 interactive
latency targets. Canonical portable uses 4,262,555,648 weight bytes,
939,524,096 bytes of 8K KV, a separate 559,939,584-byte derived-weight cache, and
6,238,335,488 peak allocated bytes. Public portable overhead/cache gates pass.
The 576-row stable evaluation took 378.68 s in the attested worker including loading
and evaluation, with 9,015,198,208 peak CUDA allocated bytes. It does not imply a
speed comparison with training or a different model.

Nsight measured 248 registers/thread, no spills, 12.35% occupancy, 58.73% cycles
without an eligible warp, and 98.62% L2 hit rate for the current one-warp kernel.
It executes about 10.57 billion FP64 instructions for the captured 8K attention
launch. These instrumented counters are separate from warm latency measurements.
The mechanism tested was FP64 tensor-core dot products with explicit layouts,
retaining FP64 online softmax. Its reduction semantics have distinct experimental
version names; the canonical profile was not reinterpreted. Ordinary Triton
lowering hit its FP64 large-K layout assertion; the explicit Gluon layout ran
correctly. Register use 255 and 124 spills remain costs, despite faster arithmetic.

Candidate 1 reduced the captured operator from 136.3 ms to 47.8 ms and 8K model prefill
to 1.7322 s, but missed the decode ratio gate. Candidate 2 restricts this kernel to
query lengths at least 1024 and head dimension 128, keeping the reference kernel
for short/decode calls. Its 1.7271 s 8K prefill is 7.12x stock BF16 and .566x stock
FP32, within the unchanged 8x/.75x limits; all latency and allocator-memory gates
pass. Its merged weight/KV/allocator bytes equal the canonical merged reference.
Separate 10 ms whole-process NVML sampling observed the same 6,396,313,600-byte peak
for construction and 8K prefill in the reference and candidate, including memory
outside the PyTorch allocator.

**Neither experimental profile is a supported public loader option.** Both match
all 48 controlled full-forward outputs bitwise, and their 2K/8K cache checks pass,
but additional video fixture `video-961109` (index 8 in the new set, combined
index 40) fails full-versus-incremental decoding. The unchanged canonical reference
fails identically: the third generated-position maximum logit difference is
.1875 with a token change, followed by 16.875 after the trajectories diverge.
Both candidate and reference cached outputs are bitwise identical to each other,
as are their full-recomputation outputs. No input or threshold was removed.
This newly discovered expanded-input limitation is separate from the historical
final-artifact qualification, and prevents broadening cache qualification or
promoting the faster kernel. The original canonical implementation remains the
production reference; its long-prefill performance criterion is incomplete.

## Review and reproduction evidence

Independent review reproduced three P2 defects and verified their fixes:
explicit clearing after untracked writes now invalidates artifact provenance;
evaluation records effective decoding settings; and changing attention after
stable-profile installation fails before a caller's cache can grow. No confirmed
review findings remain open. Installed validation comprises 36 tests on 2bccfba,
11 affected tests on 00714d6, 40 shared artifact/math regressions and 26 existing
reasoner regressions on 00714d6: 103 unique tests across these executions. Independent
review additionally ran 9 CPU tests with 2 GPU tests skipped; those are not new
unique tests. Existing adapter-disabled merged loading passes the reasoner tests
and real FP32 stock-loader comparisons.

The local review package is
`/localhome/local-rarunachalam/cosmos-adapter-public-inference-20261109/review-package`.
Its `REPRODUCE.md`, `handoff.json`, `validation-provenance.json`, exact patches,
independent review, fixed-input definitions, per-row paired CSVs and experiment
reports are independently inspectable. Large weights, raw tensors, datasets and
runtimes stay at their catalogued immutable paths. The historical 12 checkpoints,
48 fixed training profiles and 47 fully passing legacy profiles are unchanged;
new inference executions do not increase those counts. PR #342 was not reopened;
nothing was pushed or published. No full training, checkpoint-export sweep,
family expansion, four-bit work or HPO was performed.
