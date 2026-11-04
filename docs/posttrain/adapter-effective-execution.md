# Adapter effective-weight execution and physical HSDP

This continues `feat/posttrain-family-adapters` from `2d14b95`. The production
changes through `e5d975d` add explicitly versioned PiSSA execution, physical 2×2
HSDP, safe full Alpamayo replay, and independent recurrent prefix-cache ownership.
**The new whole-model MoE profile passes its 64-input stock-loader qualification.**
The public exporter now admits that explicit v3 profile; the final installed-wheel
public-CLI check is pending. Legacy v1/v2 MoE PiSSA exports remain refused.
PR #342 stays closed; no branch is pushed and no PR is published.

The historical [final review](adapter-final-review.md),
[execution extension](adapter-extension-progress.md) and
[deployment continuation](adapter-deployment-continuation.md) retain their original
revisions and outcomes: **12 named checkpoints, 48 fixed profiles, 47 fully passing
profiles**. The runs here are additional execution/configuration profiles. They do
not turn the failed legacy MoE profile into a pass or qualify arbitrary checkpoints.

## Requirement-to-code-and-test matrix

Code paths below are under `cosmos_framework/`. A passing tiny fixture does not
qualify a real checkpoint, precision or distributed backend.

| Requirement | Implementation and evidence | Status |
| --- | --- | --- |
| Consistent effective PiSSA weight | `posttrain/model/adapters.py::{pissa_dense_weight,PosttrainAdapterLinear.forward,merged_state_weight}`; shared FP64 construction followed by one FP32 rounding, low-rank trainables, saved initial decomposition | Implemented; focused gradient/ownership/resume/export fixtures and real dense-mode training pass |
| Whole-model stock-compatible MoE execution | `posttrain/model/adapter_moe_execution.py`; actual pinned stock grouped-expert helper and original native router; `qwen3_vl.py::_separate_vision_features` | Implemented and independently reviewed; new real training, exact continuation, warm-start updates and actual stock FP32 merge/reload pass |
| Legacy semantics and artifact identity | `posttrain/config/adapters.py`, `posttrain/checkpoint/adapters.py`; v1 factorized, v2 dense, v3 whole-model identity; conflicting identities rejected before mutation | Implemented and validated in fixtures. Legacy eleven failures and the subsequent default-native dense failure remain failed |
| Physical hybrid sharding | `posttrain/model/adapters.py::validate_adapter_parallelism`; GR00T/Pi05/Alpamayo builders use existing FSDP2 mesh and final optimizer parameters | Implemented and validated on three named 2×2 profiles below |
| Alpamayo full activation replay | `posttrain/model/alpamayo/policy.py::{forward,_flow_forward}`; new cache per whole prefix→expert→flow replay, non-reentrant checkpoint, preserved RNG; execution identity in `entrypoints/sft.py` | Implemented and validated: CPU replay/gradient/RNG comparisons, independent CPU 2×2 test, real mixed-component GPU lifecycle |
| Independent language/projector/vision/action settings | `configs/sft_{gr00t,pi05,alpamayo}_independent.py`; existing target resolution, alias ownership and optimizer inventory | Three additional physical profiles pass with different methods/ranks/scales/dropout/LR policies |
| Fresh initialization, updates, warm start, resume | Existing SFT trainer/checkpoint transaction; `tests/qualify_adapters.py` adds final-gradient/replica/resource checks | Physical profiles pass four updates, next-update recovery and four fresh warm-start updates; no optimizer/RNG/data state is inferred from portable exports |
| Portable and merged physical artifacts | Existing model-owned exporters and `tests/reload_flow_adapters.py` | Fresh-process portable errors zero; FP32 merged errors below unchanged gates |
| Broader BF16 deployment | Existing explicit residual profile and stock BF16 candidates tested on mixed recipes and Qwen3.5, new inputs/generation/cache/long contexts | **Incomplete**: output-error gates fail. Weight/memory/latency gates pass; those do not override numerical failures |
| Recurrent prefix reuse | `inference/reasoner/runtime.py::_fork_dynamic_cache`; clone mutable recurrent/conv state, retain safe append-only KV sharing, reject unknown cache types | Implemented; real Qwen3.5 FP32 repeated-prefix checks pass |
| Stock merged loading | `inference/reasoner/model_execution.py`, runtime `_load`, materializer/config transaction binding | Implemented; real v3 64-input stock-loader qualification passes; final public-CLI wheel check pending |

## PiSSA contracts and the new mechanism

`AdapterComponentConfig.execution="factorized_v1"` remains the default and is
omitted from historical identity serialization. Its forward, merge and original
oracle are unchanged. Explicit `execution="pissa_dense_v1"` is supported only for
PiSSA with zero dropout. It computes

```text
W_eff = round_fp32(W0 + s * ((B - B0) @ A + B0 @ (A - A0)))
```

All operands are promoted before the FP64 subtraction/products; the result is
cast to the live base compute dtype before one `F.linear`. The same helper builds
the merged weight. Algebraically this is `(W0 - s*B0@A0) + s*B@A`. `W0`, `A0` and
`B0` remain immutable; only `A` and `B` enter the optimizer. At initialization the
correction is exactly zero. There is no persistent second base, no trainable dense
weight, and no raw PiSSA adapter attached to an untouched base without its initial
decomposition. Resume and portable warm start restore the saved factors and never
repeat decomposition. This mode sacrifices low-rank forward efficiency for one
consistent effective-weight calculation.

Dense weight equality alone did **not** close the native MoE gap. The first new
real run uses the default native model execution and an independently trained v2
artifact. All 105 effective target weights agree bitwise between GPU execution and
CPU serialization; all 882 serialized model tensors verify. Nevertheless, stock
portable and stock merged execution agree bitwise with each other while both fail
against the native reference. On the 24 new inputs the maximum logit error is
`.686099052`; the original eight also fail. Matching 256/256 greedy decisions does
not override that failure.

The first divergence is earlier than the legacy near-tied layer-37 route:

* For 31 of the 32 inputs, first-layer router inputs, scores and selected experts
  agree bitwise. The first differing tensor is the expert block output. Native
  FP32 evaluates all tokens in every expert; the pinned stock loader uses sorted,
  selected grouped execution and applies route weights after the down projection.
  Native BF16 grouped execution applies them before that projection.
* The remaining original mixed image/video input first differs at the visual
  token boundary. Native execution batches image/video patches in one vision
  call, whereas stock execution runs separate image and video calls.
* Independent actual-helper, FP64-reference and full-forward probes establish
  these execution differences. A diagnostic on all 32 inputs matches stock
  portable and stock merged logits/routes bitwise when both execution differences
  are aligned. That diagnostic is evidence for a mechanism, **not** a passing
  lifecycle qualification or conversion of the v2 artifact.

The new opt-in `AdapterConfig.model_execution="qwen3_vl_moe_stock_v1"` therefore
binds the **whole model**, in addition to requiring dense PiSSA on every selected
target and no separately trained modules in this profile. It reuses the actual
Transformers 5.8.1 grouped-expert helper with differentiable temporary transposed
contiguous views/copies of existing native parameters. It registers no replacement
parameters and caches no expert weights. The original softmax/top-k router,
expert count, selection and routing thresholds are unchanged. Custom/cosine/noisy
routers, route biases, shared experts and top-k overrides are refused.

Native vision executes image then video in a fixed order on every rank, with a
zero-contribution dummy pass for missing modalities. That keeps distributed
registration and gradient collectives consistent even when ranks have different
media. Tiny two-rank full-AC tests compare three mixed-modality updates against an
unsharded reference with bitwise equal gradients/parameters and unchanged RNG.

The profile requires Torch 2.10.0, Transformers 5.8.1 and SDPA throughout, including
live text/vision attention configs. A v3 portable artifact and full checkpoint bind
this name; a v1/v2 artifact cannot acquire it by warm start or metadata rewriting.
New training is required. FP32 merged export must contain only stock model tensors,
with `cosmos_model_execution` bound in both `config.json` and `.materialized`.
The Framework loader validates that identity, pins, FP32, SDPA and stock
`grouped_mm`, then loads an **unmodified stock model**. It installs no adapter
branches or custom arithmetic. Other consumers must explicitly enforce this
contract; an arbitrary Transformers version/backend is not qualified. Direct HF
loads do not validate the Cosmos marker, and vLLM, TensorRT and alternate serving
backends are not qualified consumers of this profile.

Both new MoE runs use `Qwen/Qwen3-VL-30B-A3B-Instruct` at
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`: 96 language Q/V projections across
48 layers, eight final/deepstack merger projections, and
`model.visual.blocks.26.attn.qkv`. They use rank 8, alpha 16, ordinary alpha/r
scaling, zero dropout, FP32 factors, BF16 base storage, SDPA/full AC and a 1×2
replicate/shard mesh. Language/projector/vision LR multipliers are 1/.5/.1 with
base LR `1e-4` and zero decay. Fused expert tensors remain frozen and are not
adapter targets. The new v3 profile does not qualify an rs-scaled variant or
another checkpoint merely because small fixtures cover those settings.

The real qualification population is fixed before the new profile's results:
all eight original failing inputs, the first extension's 24 text/image/video
inputs, and a second set of 32 previously unused text/image/video/mixed inputs.
The second set has eight inputs per modality and 72–2,729 tokens. Every population
is gated independently and together; router margins/choices and intermediate
activations are recorded throughout. At least one new margin below `1e-5` is
required to substantiate routing-sensitive coverage. These are numerical fixtures
using new prompt/frame combinations from an existing WTS clip, not new held-out
scenes or a downstream task test. FP32 gates stay `atol=5e-4, rtol=2e-4`, mean error
at most `5e-5`, with every controlled greedy position equal.

The new v3 training run has completed four updates, exact next-update continuation
and a fresh four-update warm start. All 210 trainable tensors (3,887,104 elements)
change, with zero initialization/resume/warm-output error and unchanged frozen
base/initial factors. Its recorded eight-row correctness validation loss changes
from `.393474` to `.0000887833`; this is not downstream task-performance evidence.
Fresh-process BF16 portable reload also passes with zero controlled logit error.
Stock portable FP32 and actual Framework→stock merged FP32 loading both pass all
64 inputs with **zero** maximum/mean logit error, 512/512 equal controlled greedy
positions, and bitwise equal router inputs/scores, selected experts and downstream
expert outputs. The original 8, first unused 24 and second unused 32 inputs each
pass independently. Both unused populations include exact router-score ties; none
is filtered out. All 882 serialized tensors verify, with no retained adapter
branches or custom arithmetic in the stock merged model.

The FP32 comparisons also preserve the recorded numerical settings:
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms, highest FP32 matmul
precision, TF32 disabled and cuDNN benchmarking disabled. Ordinary model loading
does not silently change the caller's global numerical settings. Matching pins
and execution names alone does not qualify a different numerical configuration.

The matched four-update cost control measures **whole-profile** overhead,
including stock-layout expert copies and the two native vision calls, rather than
isolating dense-weight algebra. With identical retained snapshots and gradient
observation, the slowest-rank total is 19.676 s for factorized/native, 19.988 s for
dense/default-native, and 20.434 s for dense/stock-compatible execution. Throughput
is .407/.400/.392 examples per second. The new profile takes 3.85% more update
time and 0.72% more peak allocation than the control (34.22 versus 33.98 GiB per
rank); median steady updates are 5.313 versus 5.016 s. These are four-update,
one-seed measurements on short fixed fixtures, not a general efficiency bound.

Only 3,887,104 elements are optimized; the factorized control records 15,549,256
Adam-state bytes per shard rank, and the new mode preserves the same parameter
inventory and optimizer-state shapes. Dense effective weights and transposed
expert copies are temporary and never enter the optimizer. Initial construction
still materializes the full BF16 base per rank; per-update peaks alone do not
establish a smaller minimum device capacity. Initial output preservation is
measured against the unadapted base **under the same execution profile**. The
new profile's different BF16 expert arithmetic can change default-native outputs;
it must not be presented as converting a legacy run without changing its oracle.

A separate, exact-input-matched comparison of the new trained model's BF16 and
promoted FP32 native outputs changes **3 of 64** controlled greedy positions:
maximum logit error 10.8606, mean absolute error .324725, and maximum TV .840755.
The FP32 merge gate compares against the promoted FP32 native adapter, not those
BF16 outputs. A successful FP32 merge cannot establish unchanged training-precision
predictions or downstream quality. BF16 portable reload remains exact.

The CPU candidate export takes 1,535.67 s (25.6 min) and peaks at 329.60 GiB host
RSS. The final FP32 shards total 115.748 GiB; the portable artifact is 29.77 MiB
including initial factors and its manifest. Export uses the existing full-state
staging, expert-layout conversion, hashing and verification transaction; it is
not a streaming low-memory merge. Its wall measurement overlaps the independent
native GPU reference process and is not an exclusive-host throughput benchmark.

## Physical execution profiles that passed

All three use four A100 80 GB GPUs on one host, a 2×2 replicate/shard mesh,
Transformers 4.57.6 and Torch 2.10.0+cu128, seed 42, full activation checkpointing,
FP32 adapters/gradient reduction, and the existing AdamW/trainer/data path. Base
precision remains model owned: GR00T FP32 masters with BF16 autocast, Pi05's native
mixed storage, and Alpamayo BF16 language/vision with FP32 action/Fourier boundaries.
Results use installed clean-archive revision `e8d8621`; later changes are separately
reviewed/default-path regression checked, not relabeled as new GPU training.

| Exact checkpoint | Trainable tensors / elements | Portable / next-update error | FP32 merge max / mean absolute error | Slowest-rank median steady update / peak per-update allocation |
| --- | ---: | ---: | ---: | ---: |
| `nvidia/GR00T-N1.7-LIBERO`, `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21/libero_spatial`; backbone `9ce19a195e423419c349abfc86fd07178b230561` | 218 / 2,002,944 | 0 / 0 | 1.09673e-5 / 7.00378e-8 | .802 s / 11.65 GiB |
| `RLinf/RLinf-Pi05-LIBERO-SFT`, `45ccfcc4e28634f1576ebf78cab0fbe2fd82432d` | 155 / 1,014,016 | 0 / 0 | 7.15256e-6 / 2.65604e-7 | .790 s / 7.51 GiB |
| `nvidia/Alpamayo-R1-10B`, `dd4a24cacefc9a6477a6dfc7354de2443401409d`; processor `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 398 / 4,449,138 | 0 / 0 | 3.21865e-6 / 3.33378e-7 | 1.635 s / 50.31 GiB |

These timings include the same qualification gradient observer and use three
steady updates after the first warm update; they are descriptive measurements,
not general serving/training throughput guarantees. Every logical trainable
changes, all observed gradients are finite, intended/actual/optimizer inventories
agree, frozen base and initial factors are unchanged, and corresponding sharded
replicas agree after training and resume. Four fresh warm-start updates also pass
with exact loaded output/state and pristine optimizer/update clock/data cursor.

GR00T and Pi05 use 40 staged real LIBERO training rows and 55 validation rows.
Alpamayo uses the first 12 of 14 chronological KITTI training rows to make the
global four-rank batch complete; both omitted row IDs and selection hashes are
recorded before execution. Its 38 validation rows are unchanged. The initial
14-row launcher failed before model construction and remains preserved. Flow
validation loss moves `.399482→.397725`, `.000383522→.000380076`, and
`1.955215→1.953256`, respectively. These small correctness runs establish neither
simulator success nor driving quality/safety.

GR00T/Pi05 settings by component are language LoRA `(r=8, alpha=12, dropout=.1,
LR multiplier=.9, decay=.01)`, projector DoRA `(4,7,rs,.05,.3)`, selected last vision
block PiSSA `(3,6,0,.1)`, and action rsLoRA `(4,9,.1,.8)`. Alpamayo uses its existing
five-group independent recipe, including action DoRA, action-out rank-one LoRA and
an extra trained final-prefix K norm. Concrete selectors remain in the recipes;
there is no generic action/expert-tensor adapter assumption.

Full Alpamayo checkpointing replays a closure containing prefix creation, action
expert and flow loss, with a fresh internal differentiable cache on every call.
It rejects externally supplied caches, preserves stochastic noise/time/dropout
RNG, and leaves generic decoder checkpointing off. Full checkpoints bind
`alpamayo_fresh_prefix_flow_v1`; changing execution is a warm start, not exact
resume. The old vision-only path stays unchanged.

## BF16 extension results and precision limits

The predeclared gates compare each candidate to **both** its recorded BF16 portable
computation and supported FP32 merge. FP32 promotion changes computation and is
reported separately from FP32 adapter/merge equivalence. Existing fixed-artifact
qualification is preserved, but it cannot be generalized to new inputs or recipes.

The new corpus includes eight text, eight image and eight video inputs, differing
sequence lengths, cached four-token generation, three repeated prefix uses, and
2,048/8,192-token KV measurements. Original and new input populations are checked
separately. Broader ordinary LoRA, two mixed-component Qwen3-VL-2B rotations and a
previously covered Qwen3.5-0.8B DoRA checkpoint all fail the numerical gates. No
paired WTS task evaluation is run for candidates failing these prerequisites.

| Candidate | Representative failed bound | Weight ratio / short latency ratio to FP32 | Result |
| --- | --- | ---: | --- |
| WTS ordinary LoRA, existing residual-FP32 profile | New portable-reference maximum TV .1008448 > .1; original max logit error remains exactly 2.0 | .5 / 1.09181 | Broader qualification incomplete |
| Mixed vision PiSSA, Qwen3-VL-2B | Portable max TV .122729; FP32 max KL .071676 and TV .18307 | .5 / 1.06710 | Incomplete |
| Mixed language PiSSA, Qwen3-VL-2B | Multiple logit/distribution/prefix bounds fail; original eager reference retained | .5 / 1.07873 | Incomplete |
| Qwen3.5-0.8B DoRA | Portable max TV .298768; FP32 .195224, additional bounds fail | .5 / 1.01090 | Diagnostic only; production family guard remains |

The two mixed profiles have portable/FP32 reference TV separations `.234946` and
`.301570`, exceeding twice the required `.1` bound: no one candidate distribution
can meet both at those positions. Their admissible greedy choices under the `.02`
regret gate also conflict. Qwen3.5 has a disjoint admissible-choice case as well.
These are concrete incompatibilities under the existing dual-reference contract,
not justification for weakening it. Ordinary LoRA is not proven impossible: a
whole-model transient FP32-compute candidate fails its original max-logit bound,
and a subsequent FP32-head-only candidate fails both that bound (2.01626) and the
new portable TV bound (.109569). Neither becomes a production profile.

BF16 floating storage halves for every candidate; all serialized tensors equal
one BF16 rounding of their FP32 merge. Short peak ratios are .5025 (Qwen3-VL) and
.5068 (Qwen3.5). The residual-FP32 Qwen3-VL profile keeps **FP32 KV storage**, so KV
is 448 MiB at 2K and 1,792 MiB at 8K for both profiles. Total peaks change from
9.08/19.40 GiB to 5.12/15.44 GiB. The Qwen3.5 diagnostic halves KV from
67.7/211.7 MiB to 33.8/105.8 MiB. Cached/recomputed and repeated-prefix checks are
scored numerically, not only by matching token IDs.

The Qwen3.5 prefix fix itself passes in FP32: eight unused video inputs, three
reuses each, all cached/recomputed generated IDs equal, unchanged cached tensors,
and maximum logit error 8.78572e-5 under the original FP32 bounds. Only the whitelisted
non-offloaded DynamicCache layer implementations are supported. Unknown/static/
offloaded/subclass caches fail explicitly. BF16 broader support remains incomplete.

## Runtime provenance and reproduction

Evidence root: `/localhome/local-rarunachalam/cosmos-adapter-effective-20261104`.
Large artifacts live under its `large` symlink on RAID. Old review packages, model
weights, trained adapters, failure traces and environments are unchanged. There
is no container claim for these attested host-wheel executions.

| Source | Wheel SHA-256 | Validation |
| --- | --- | --- |
| `21f5cc821efd3f8c8be742c04f97f209e8a3a1f7` | `31b14e53db9ca6f935b027cce1fb8272ad8ed26b4891917fd67a3195369cfae9` | Dense PiSSA training and lifecycle; default-native merge subsequently fails |
| `e8d86214e844753da3aab238c850c7511b5bf7ca` | `069c6e78fd578b80f926e12e864be16fee3c842a6f843cc2c83730a35a243627` | Three physical profiles, BF16 numerical/resource checks, recurrent-cache GPU validation |
| `e5d975ddd17dd898c0937fe3102d486b584b4268` | `cdcdf340be832b43ce697eae1568693ff35f8886a031a0a8b512ea6eb6ed1cdc` | New stock-compatible MoE training/recovery/warm/portable and 64-input stock merged qualification pass; 1,896 packaged files verified |

New physical recipes run through the existing SFT qualification entrypoint, with
staged immutable sources described in [flow adapters](guides/flow-adapters.md):

```bash
"$SFT_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m cosmos_framework.posttrain.tests.qualify_adapters \
  --recipe cosmos_framework/posttrain/configs/sft_gr00t_independent.py \
  --method configured --activation-checkpointing full --updates 4 --warm-start-check \
  --out "$NEW_RUN" --wheel "$WHEEL" --build-record "$BUILD_RECORD"
```

Use `sft_pi05_independent.py` for Pi05. Alpamayo uses
`sft_alpamayo_independent.py` plus `--override model.parallelism.dp_replicate=2`
and `--override model.parallelism.dp_shard=2`. Use fresh output directories.
The new MoE recipe is `sft_qwen3_vl_moe_pissa_stock.py`, with two ranks, SDPA/full
AC, the pinned 30B checkpoint and `--method configured`. Public merged export is
restricted to the explicit v3 profile; the legacy guard remains. Use the recorded
`SFT_MODEL_PATH` and `SFT_DATA_ROOT`; physical recipes also require their staged
backbone/processor variables. The local review package supplies complete recorded
commands with fresh output destinations.
The local harness records complete argv/environment/source/wheel/data hashes for
every execution; independent review reports are under `review/`.

Multi-node, alternate Torch/Transformers pins, external FlashAttention, arbitrary
compile modes, expert adapters, TP/EP/CP/PP, four-bit training and HPO are not newly
supported. The 576-row WTS subset remains a **reused scene-disjoint benchmark**;
this work adds no downstream task-performance claim.
