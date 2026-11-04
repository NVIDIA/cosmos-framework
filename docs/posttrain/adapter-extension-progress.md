# Adapter execution and deployment extension

This is the preserved extension snapshot through `afbd4ab`. The subsequent
[deployment continuation](adapter-deployment-continuation.md) qualifies an explicit
ordinary-LoRA BF16 inference profile, fixes optional prefix-cache/loading defects
and records four more failed MoE accumulation probes. The results below retain
their original source revisions and qualification boundaries.

This extends clean `feat/posttrain-family-adapters` at
`d2627108da1282f2de7577f9e878acd269084fd5`. Production commits are `7b0d100`
(HSDP/selective checkpointing and independent recipes), `f6baa78` (bounded BF16
export), and `2d12fd5` (execution-bound checkpoint continuation and attention-aware
reload qualification). PR #342 remains closed; nothing is pushed or published.

The [previous qualification](adapter-final-review.md) and
[independent review](adapter-independent-review.md) remain historical evidence:
**12 named checkpoints, 48 fixed method profiles, 47 fully passing profiles**.
The five mixed-setting profiles below are additional execution/configuration
coverage on three of those checkpoints. They do not replace the fixed 48 profiles,
add checkpoints, or qualify every combination in a model family.

## Requirement-to-code-and-test status

Paths below are under `cosmos_framework/posttrain/` unless stated otherwise.

| Requirement | Code and actual validation | Status |
| --- | --- | --- |
| Independent component selection/settings | Existing `model/adapters.py::{resolve_targets,install_adapters,optimizer_groups}`; new `configs/sft_{qwen3_vl,qwen3,alpamayo}_independent.py`; five real-model profiles below | Implemented and validated. Methods, ranks, alpha/rs scaling, dropout, LR multipliers and decay differ within each run |
| Ownership and optimizer membership | Existing `assert_trainability`, selector/alias tests; per-rank placement, optimizer inventories and final-gradient observations in `tests/qualify_adapters.py` | Implemented and validated. Every intended logical trainable changes; no duplicate optimizer parameters or unintended base parameters |
| Shared/tied parameters | Existing shared-target/normalization-owner tests; real Qwen3-0.6B embedding/head identity checked before injection, portable reload and stock merged reload | Validated for those cases. Cross-FSDP-unit ties and custom tied-target stock-HF export remain refused |
| LoRA, rsLoRA, DoRA, PiSSA | Existing native LoRA math and `PosttrainAdapterLinear`; new mixed language/merger/vision/action runs include DoRA magnitudes and saved PiSSA initial factors | Validated within the listed profiles. No new adapter formula or training loop |
| Selective activation checkpointing | `model/qwen3_vl.py::build_qwen3_vl_model` adds actual PyTorch SDPA forward operations to the existing selective save policy, only for adapter-enabled construction | Implemented; dropout/RNG/gradient fixture and real native 2-rank/4-rank runs pass |
| HSDP and replicate-only execution | `model/adapters.py::validate_adapter_profile` admits valid DP meshes; existing native/shared FSDP2 machinery owns synchronization | Native 2×2 and HF 2×2 HSDP, and native 2×1 replication, pass multi-step/recovery/export checks; no multi-node claim |
| Fresh initialization and frozen-base invariants | Existing lifecycle; initialization outputs, finite gradients, four updates and unchanged frozen-weight hashes in each profile | Validated. PiSSA keeps W0 and A0/B0; initial A gradients need not be nonzero with zero B |
| Full checkpoint continuation | `entrypoints/sft.py::adapter_execution_identity`, existing `LLMTrainer` identity validation | Same-profile native/HF/Alpamayo GPU continuation is exact; attention/AC changes and unbound legacy checkpoints are refused in real CPU lifecycle tests |
| Portable warm start/reload | Existing portable reader/identity; new qualification trains four fresh updates after warm start and checks a pristine optimizer/clock/data cursor | All five profiles pass. Portable identities remain compatible; a deliberate new execution profile is a new run |
| FP32 merged export | Existing pure merge/materializer/native physical exporter; corrected `tests/reload_adapters.py::recorded_attention` and `tests/adapter_numerics.py::verify_promoted_merge` | Five mixed profiles pass the unchanged gate. Native MoE PiSSA remains incomplete after seven accumulation/execution comparisons |
| BF16 merged deployment | `checkpoint/tools/adapters.py::export_merged_adapter`, precision-bound materializer/config verification | Newly implemented and qualified for the three exact WTS-trained rsLoRA/DoRA/PiSSA artifacts below. Ordinary LoRA remains incomplete and refused |
| Adapter-disabled behavior | Existing installed native dense/MoE forward/backward and FSDP GPU tests; affected SFT lifecycle CPU tests | Four existing GPU tests pass on each of two ranks; source and installed regression suites pass |

## Exact additional training profiles

All use one host with A100 80 GB GPUs, Torch `2.10.0+cu128`, Python 3.12,
AdamW global LR `1e-4`, FP32 adapters/extra modules and FP32 gradient reduction.
Language/VLM storage is BF16. Alpamayo retains its native FP32 action/Fourier
boundaries. HF/native VLM uses Transformers `5.8.1`; physical uses its separate
`4.57.6` runtime. These are attested host environments, not container executions.

Every profile performs four updates, saves a complete checkpoint, compares the
next uninterrupted/resumed update, exports portable state, trains four new
warm-start updates, and separately reloads portable and merged artifacts. All
logical trainable tensors change in both update sequences, all observed final
optimizer gradients are finite, frozen base hashes are unchanged, portable and
warm-start controlled output error is zero, and next-update parameter error is
zero. The FP32 merge tolerance remains `atol=5e-4, rtol=2e-4`, mean error at most
`5e-5`, with all controlled greedy outputs equal. Physical checks compare
fixed-noise/time flow outputs and loss, not generated language tokens.

| Profile / exact checkpoint revision | Mesh; attention; AC | Trainable tensors / elements | FP32 merge max / mean absolute error | Training / reload source |
| --- | --- | ---: | ---: | --- |
| `native_a_fsdp`: Qwen/Qwen3-VL-2B-Instruct `89644892e4d85e24eaac8bacfd4f463576704203` | 1×2; SDPA; selective | 139 / 1,873,920 | 7.25e-5 / 6.68e-6 | `f6baa78` / `f6baa78` |
| `native_b_replicate`: same checkpoint | 2×1; eager; none | 132 / 1,160,192 | 1.26e-4 / 6.89e-6 | `f6baa78` / `2d12fd5` |
| `native_a_hsdp`: same checkpoint | 2×2; SDPA; selective | 139 / 1,873,920 | 9.16e-5 / 6.72e-6 | `2d12fd5` / `2d12fd5` |
| `text_hsdp`: Qwen/Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca` | 2×2; SDPA; full | 113 / 918,528 | 8.87e-5 / 7.12e-6 | `2d12fd5` / `2d12fd5` |
| `alpamayo_mixed`: nvidia/Alpamayo-R1-10B `dd4a24cacefc9a6477a6dfc7354de2443401409d` | 1×2; model-owned SDPA; vision | 398 / 4,449,138 | 1.91e-6 / 3.06e-7 | `f6baa78` / `f6baa78` |

The native eight-row fixture contains text, PNG, MP4 and WTS video, with different
modality presence on different ranks. Text HSDP uses eight text conversations and
is not VLM evidence. Alpamayo uses eight prepared real KITTI trajectory rows and
processor `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. It establishes physical-model
correctness and component connectivity, not driving quality or safety.

The two native rotations use the actual 56 language Q/V projections,
eight final/deepstack merger projections, vision block 23 QKV, and separately
trained `model.language_model.norm`. Settings are `(method, rank, alpha,
rs scaling, dropout, LR multiplier)`:

| Recipe group | Settings |
| --- | --- |
| Native A language | LoRA, 8, 12, no, .10, 1.0; weight decay .01 |
| Native A merger | DoRA, 4, 9, yes, .05, .3 |
| Native A vision | PiSSA, 3, 6, yes, 0, .1 |
| Native B language | PiSSA, 4, 7, yes, 0, .8 |
| Native B merger | LoRA, 6, 9, no, .15, .4 |
| Native B vision | DoRA, 2, 3, no, .05, .15 |
| Native A/B extra norm | FP32, LR multiplier .07, decay .01 |
| HF text Q / V | rsLoRA, 8, 12, yes, .10, 1.0, decay .01 / PiSSA, 4, 7, no, 0, .4 |
| HF text extra `model.norm` | FP32, LR multiplier .1 |
| Alpamayo language K/V | LoRA, 8, 12, no, .10, .9; decay .01 |
| Alpamayo merger / vision block 26 QKV | DoRA, 4, 7, yes, .05, .3 / PiSSA, 3, 6, no, 0, .1 |
| Alpamayo 75 expert/input-action projections | DoRA, 4, 9, yes, .10, .8 |
| Alpamayo `action_out_proj` | LoRA, 1, 2, no, 0, 1.2 |
| Alpamayo extra `vlm.model.language_model.layers.35.self_attn.k_norm` | FP32, LR multiplier .07; this final-prefix K-normalization has an action-loss gradient |

Unspecified decay is zero. PiSSA uses `svd_lowrank`, four iterations, and zero
dropout. Exact SVD remains fixture coverage. These rotations cover all component
paths, DoRA/rs scaling, PiSSA placement, nonzero dropout and heterogeneous
optimizer policy without claiming a Cartesian product of settings.

## BF16 deployment contract and measurements

BF16 export rounds each completed FP32 merged tensor once. All 625 serialized
floating tensors equal that rounding exactly, and stock `dtype="auto"` reloads
BF16. The commit marker retains base/adapter/processor/media identity and adds
`deployment_precision={dtype:bfloat16, weight_conversion:float32_merge_then_bfloat16,
equivalence:approximate_requires_task_qualification}`. FP32 remains the default.

This is an approximate deployment contract, separate from FP32 merge equivalence.
Before experiments, the acceptance file fixed all existing BF16 drift bounds plus
mean total variation ≤ .02, p99 TV ≤ .1, greedy agreement ≥ .95; accuracy drop ≤
one percentage point and paired 95% bootstrap lower delta ≥ −2 points; weight
bytes ≤ .51, peak allocated CUDA memory ≤ .75 and median latency ≤ 1.1 times FP32.
The existing confident-decision regret bound remains .02. Measurements use every
saved controlled input, three warm-ups and 20 timed repetitions, and both the
original BF16 portable and FP32 merged references.

The paired WTS evaluation uses the same corrected four-frame preprocessing,
50,176–81,920 pixels/frame, TorchCodec CPU decoding, SDPA, batch one, greedy
32-token generation, seed 42 and exactly the same 576 IDs/golds as the FP32
reference. This is a **reused scene-disjoint benchmark**, not an untouched final
test. The descriptive intervals use 10,000 paired-row bootstrap samples, seed 42;
they are not evidence from independent training seeds or independent scenes.

| Method / existing host WTS artifact | FP32 → BF16 correct / 576 | Paired delta and 95% interval (percentage points) | Controlled FP32 max logit error / mean TV | BF16/FP32 median latency | Result |
| --- | ---: | --- | --- | ---: | --- |
| LoRA | 528 → 527 | −.174 [−.868, .347] | 1.459 / .00504 | .900 | **Incomplete, refused**: decision regret .033916 > .02 |
| rsLoRA | 530 → 530 | 0 [−.521, .521] | .952 / .00198 | .885 | Qualified |
| DoRA | 528 → 527 | −.174 [−.868, .347] | 1.376 / .00309 | .885 | Qualified |
| PiSSA | 530 → 530 | 0 [−.521, .521] | .992 / .00231 | .888 | Qualified |

All four have zero invalid answers. Floating weight storage falls from 7.926 to
3.963 GiB, and controlled peak allocated CUDA memory from 8.172 to 4.260 GiB
(ratio .521). These measurements cover this input/device/loader profile, not
end-to-end serving throughput. Eager attention and disabling reduced-precision
BF16 reductions were also tried for LoRA, with the same inputs and bounds; neither
closes its failure. Passing task accuracy does not override its controlled gate.
BF16 export is bounded to dense Qwen3-VL and refuses ordinary non-rs LoRA,
physical formats and other families. Mixed recipes and other checkpoints are not
additional BF16 deployment qualifications.

## MoE PiSSA investigation

The preserved Qwen3-VL-30B-A3B checkpoint
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, portable factors and all eight failing
probe batches remain unchanged. The independent native portable FP32 computation
is the reference. No router semantics, thresholds, input population or reference
computation is changed, and no adapter branches remain in stock candidates.

| Tested effective-weight construction | Max / mean final-logit error | First changed expert choice |
| --- | --- | --- |
| Existing FP32 rank-2r conversion | .0295000 / .000388840 | Layer 37, token 7 |
| FP64 effective weight, rounded once to FP32 | .0295095 / .000388907 | Layer 37, token 7 |
| FP32 `(B−B0)A + B0(A−A0)` update | .0294924 / .000388730 | Layer 37, token 7 |
| FP32 explicit residual then trained update | .0295010 / .000388878 | Layer 37, token 7 |
| Serialized FP64-accumulated candidate, stock SDPA loader | .0295076 / .000388857 | Layer 37, token 7 |
| Original serialized rank-2r candidate, stock eager loader | .0294952 / .000389398 | Layer 37, token 7 |
| Serialized FP64-accumulated candidate, stock eager loader | .0330219 / .000732229 | Layer 37, token 7 |

Every row fails the original FP32 contract despite all 64 greedy tokens agreeing.
Router traces use the preserved failing batch (probe 3); output gates use all
eight batches. The table's routing location refers to that instrumented batch.
The stock candidate is an ordinary 882-tensor FP32 model loaded through
`AutoModelForImageTextToText.from_pretrained(dtype="auto", attn_implementation="sdpa")`.
The [earlier FP64 diagnostic](adapter-final-review.md#moe-pissa-residual-math-versus-routing-sensitivity)
and new traces compare residual reconstruction, effective weights, layer inputs,
raw router scores, local FP64 router scores, actual selected experts and downstream
block outputs. They expose rounding before a nearly tied hard top-k boundary,
not an observed residual/conversion error. The portable margin is 7.45e-9;
expert 5 is replaced by 68 and the block error reaches .097. Computing each router
at its captured input in FP64 preserves the opposite expert order.

The stock eager candidates also fail. All seven comparisons use every original
batch and preserve the independent reference. This does not prove that every
possible numerical implementation must fail, but no tested residual, conversion,
accumulation or stock attention change satisfies the contract. The public
rejection remains. Preserved portable support is working; it is not a completed
stock merged export.

Concrete alternatives require a different contract: new training with a consistent
dense effective-weight execution path followed by fresh qualification, or a
separately approved approximate deployment contract with routing/task-quality
validation. Neither can retroactively make this preserved checkpoint pass its
original oracle. Keeping adapter branches, changing that oracle to dense execution,
forcing the selected experts or widening the bounds would not complete this task.

## Execution and compatibility boundaries

`adapter_execution_identity` binds backend/builder, configured attention,
activation checkpointing and **resolved** DP dimensions in full checkpoints.
The existing trainer validates this before continuation. Portable weight identity
does not acquire these fields: warm start is allowed to select a new execution
profile with a fresh optimizer, clock and data cursor. Checkpoints written before
`2d12fd5` lack a trustworthy execution identity and are deliberately refused for
exact resume by the new runtime. Use their preserved original runtime/configuration
for exact continuation, or export their portable adapters and start a new run.
Do not edit old checkpoint identities to bypass this check. Adapter-disabled
checkpoint identities are unchanged.

The reviewer reproduced the pre-existing silent eager→SDPA resume: all 18 tiny
fixture trainables diverged on the next update, max error 8.47e-5. The correction
refuses that change, while eager→eager remains exact. The qualification helper
also previously hardcoded SDPA on reload; it now reads the target training run's
recorded backend, rejects conflicting/unknown rank records, and uses it for native,
private stock and freshly loaded merged comparisons. Historical qualified records
without this field default to their documented SDPA profile. The failed eager
check is preserved; only reload/export was repeated after the fix.

The new mesh support removes an adapter-only restriction over existing FSDP2.
Selective AC also reuses existing `apply_ac`; its new save regex covers the actual
PyTorch SDPA flash/efficient/cuDNN operations. Eager was already implemented and
is newly qualified here. External FlashAttention-2 is absent from these pinned
runtimes and remains unqualified. TP/EP/CP/PP/CFG axes are not wired into the
posttrain `ParallelDims` path even without adapters; their guards remain.
Physical HSDP, multi-node execution, arbitrary compile modes and custom fused
expert tensors are unqualified. Alpamayo keeps vision-only AC because generic
full replay does not preserve its mutable differentiable prefix cache. Four-bit
training, new model families, broad retraining and HPO are outside this extension.

## Provenance, review and reproduction

The installed clean-archive wheels contain 1,884 verified package files:

| Revision | Wheel SHA-256 | Evidence |
| --- | --- | --- |
| `f6baa78090813ed75bbd57122c7815012630162b` | `33792b5d8ec4f3307ad3b537d377b4388467994eb55406ca8792838f145681e6` | First native/Alpamayo mixed runs; actual BF16 export, fresh stock loading/resource checks and paired WTS evaluation |
| `2d12fd5a11c2e86a5e47808b5da17efb2200ab87` | `3df44383519c45a2f5eb8a6692af13f5af71899765e499645482164f1aa2f894` | Execution-bound native/HF HSDP, corrected eager reload, frozen-tie audit, adapter-disabled GPU gates |

The original accumulation diagnostics used the preserved `9f90524` wheel; their
original attestation is retained. The initial BF16 prototypes also retain that
revision. Actual production BF16 exports at `f6baa78` are independently verified
tensor-for-tensor equal to those prototypes, then reloaded and measured at
`f6baa78`. WTS inference/evaluation code is unchanged from reference evaluation
`2a8b48d`; only model precision/artifact differs. Training artifacts remain bound
to `076cfbb`, never relabeled as new training.

The final production delta after `f6baa78` changes checkpoint identity and reload
qualification, not adapter math, optimizer policy, media processing, exporter or
evaluation. Consequently completed BF16 evaluations and first mixed runs remain
valid for those unchanged paths. New-identity continuation is checked separately.
The Alpamayo follow-up at `2d12fd5` warm-started from the completed mixed adapter,
then saved a new bound checkpoint. Its external harness initially assumed a
scheduler existed in a constant-LR recipe and stopped after checkpoint commit.
That failed script/log are retained. The corrected check continued from the valid
checkpoint, performed two more updates, saved again and compared the next update
with uninterrupted execution. Both ranks have zero parameter error, matching
RNG/clock/data state, optimizer continuation verified at `atol=1e-7, rtol=1e-5`,
398 changed trainables and unchanged frozen hashes.
No completed baseline training was restarted to repair this harness issue.

Installed `f6baa78` suites: 274 HF/adapter/export/execution tests pass (two GPU
cases excluded from CPU invocation), and 60 physical tests pass with one
CUDA-only skip. Installed `2d12fd5` affected suites: 57 SFT/native/execution tests
and 60 physical tests pass, with the same physical skip. Four existing
adapter-disabled GPU tests pass on each of two ranks. Counts overlap and are not
summed. Ruff and diff checks pass. Independent review covers every production
change through `2d12fd5`, found and rechecked the gradient-observer and attention/
resume corrections, and has no unresolved correctness findings.

The machine-readable [extension results](assets/adapter-extension-validation.json)
and [resolved profile contracts](assets/adapter-extension-profile-contracts.json.gz)
retain concrete module names, settings, immutable identities, report hashes and
the source revision of each result. Local evidence and reproduction scripts are in
`/localhome/local-rarunachalam/cosmos-adapter-extension-20261104`.
The local `review-package/` contains patches, commit/provenance maps,
commands, manifests, per-rank results, paired predictions, numerical reports and
independent review. Large trained weights and isolated environments remain in
their original directories and are identified by hashes, not vendored into the
Framework patch.

For normal training, point the existing environment variables at complete pinned
snapshots and staged data, then use the installed entrypoint:

```bash
export SFT_MODEL_PATH=/models/Qwen3-VL-2B-Instruct-89644892e4d85e24eaac8bacfd4f463576704203
export SFT_DATA_ROOT=/data/multimodal-eight-row-fixture
export SFT_INDEPENDENT_VARIANT=vision_pissa
"$SFT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m cosmos_framework.posttrain.entrypoints.sft \
  --config=cosmos_framework/posttrain/configs/sft_qwen3_vl_independent.py -- \
  rl.seed=42 model.parallelism.dp_replicate=2 model.parallelism.dp_shard=2 \
  model.model_config.activation_checkpointing=selective

# Same native checkpoint: use language_pissa, two ranks, replicate=2/shard=1,
# attn_impl=eager and activation_checkpointing=none for the other measured profile.
# HF Qwen3-0.6B uses sft_qwen3_independent.py and model_hf.parallelism.*.
# Alpamayo uses sft_alpamayo_independent.py plus SFT_PROCESSOR_PATH, two pure-shard ranks.

# Only the qualified dense Qwen3-VL deployment methods/profiles described above:
"$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --adapter /exports/portable-rslora --base "$SFT_MODEL_PATH" \
  --merged --output-dtype bfloat16 --out /exports/merged-rslora-bf16
```

`run_component_case.py` records the exact training, portable, export and merged
commands, installed wheel and model/data hashes for each case. It uses fresh
destinations and must not be rerun over completed evidence.
