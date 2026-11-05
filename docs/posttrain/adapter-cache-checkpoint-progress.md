# Checkpoint extraction, BF16 caches and matched WTS training

This continuation starts at `95b00e99f870bf0b11026e0984619e77ba9ca6a6` on
`feat/posttrain-family-adapters`. Production changes end at
`cd609990ee76d595f936ae930788ba2d73a1f9b5`; later documentation does not relabel
validation revisions. Historical results remain **12 named checkpoints, 48 fixed
profiles and 47 fully passing legacy profiles**. The versioned MoE execution,
physical HSDP/replay and earlier rounded-weight training qualifications remain
attached to their original artifacts and revisions.

| Requirement | Production code / focused tests | Evidence and status |
| --- | --- | --- |
| Selective consolidated/DCP extraction | `checkpoint/adapter_extract.py::load_adapter_tensors`; `test_adapter_checkpoint_extract.py` | Implemented and validated: exact selected tensors/aliases/PiSSA initial factors, independent two-rank DCP reader, public real-checkpoint exports under whole-job memory caps |
| Complete immutable checkpoint boundary | `checkpoint/adapters.py::checkpoint_adapter_state`, existing manifest/provenance; CLI destination validation | Implemented and validated: missing unselected storage, invalid coverage, source mutation, missing optimizer component and conflicting aliases fail; source/temporary/transaction path collisions fail before mutation |
| Transaction, recovery and reuse | Existing HF materializer and bounded tensor source | Implemented and validated: interrupted public DCP export recovers; completed export reuses its original marker; every 882 MoE / 625 dense output tensor matches its same-update reference |
| Qwen3-VL BF16 cache consistency | `utils/generator/bf16_cache*.py`; `test_adapter_canonical_cache.py` | Implemented and validated: 96/104 cases per portable and merged loader pass strict merge/cache/ownership checks; merged artifacts pass resource gates; extended horizons and paired task gates pass |
| Qwen3.5 BF16 cache consistency | `utils/generator/qwen3_5_cache*.py`; `test_adapter_qwen3_5_cache.py` | Implemented and validated: 92 cases per portable and merged loader pass numeric/cache/ownership checks; FP32 recurrent state and safe repeated forks; merged resource gates, extended horizons and paired task gates pass |
| Matched factorized control | `configs/sft_wts_execution_comparison.py`, `qwen3_vl_hf_stock_v1` | Implemented and validated: same HF5 modality/norm path, exact initial state; both lifecycle preflights pass, including exact controlled next-update resume |
| Full-WTS quality comparison | Existing SFT trainer, packer, objective and evaluator | Completed: one matched epoch each, 5,555 training / 2,676 validation rows; both FP32 exports score 527/576 on the reused benchmark with identical parsed choices |
| Final full-epoch artifact lifecycle | Existing exact-resume, portable reader and FP32 loader | Both final-update replays pass; new-mode strict portable/merged FP32 checks pass. **Factorized control strict FP32 merge is incomplete** on one image fixture; preserved arithmetic diagnostics below |
| Independent production review | Five separate local review reports, final snapshots/hashes | Completed; confirmed findings fixed before final installed qualification. No remaining confirmed defect in reviewed production delta |

Checkpoint, test and config paths are relative to `cosmos_framework/posttrain`;
`utils/generator` paths are relative to `cosmos_framework`.
No new trainer, optimizer, packer, checkpoint transaction or LoRA formula was added.

## Canonical cache execution contract

The new opt-in profiles are `qwen3_vl_bf16_canonical_cache_v1` and
`qwen3_5_bf16_canonical_cache_v1`. They apply only to corresponding dense v4
`merged_weight_bf16_v1` artifacts, with a recorded BF16 base and zero adapter
dropout. They are inference overrides; training mathematics and the portable
state format are unchanged. Legacy artifacts and failed stock BF16 oracles are
not converted into passing results.

Both profiles retain BF16 weights and activation outputs. Frozen language
projections, adapted language projections and the output head use fixed
32×64×64 linear tiling with FP32 accumulation. Attention uses bounded per-query
64-key FP64 reductions and returns BF16, without constructing a full attention
matrix. Existing mask semantics remain authoritative. This addresses the
preserved shape-dependent frozen-GEMM divergence without a whole-model FP32
transient copy or a global GEMM/SDPA switch.

Qwen3.5 also uses one FP64 gated-delta recurrence for prefill, suffix and decode,
rounding state to FP32 after **every token**. Stock chunked/recurrent execution
and an FP32-only recurrence produced split-dependent state despite matching
inputs. The new kernel matches an independent high-precision reference and
split execution. Request-local recurrent state is FP32; convolution state and
full-attention KV remain BF16. The ordinary `DynamicCache` owns the state;
Transformers classes/functions are not globally patched. Foreign, malformed or
incomplete caches fail before mutation. Repeated prefix branches clone mutable
convolution/recurrent buffers and preserve the original prefix.

These profiles require Torch 2.10.0, Transformers 5.8.1, Triton 3.6.0, one CUDA
SM80 device, batch one, eval with `no_grad`/`inference_mode`, and the stock SDPA
architecture before installation. Real hardware qualification is A100-SXM4-80GB.
Autocast, training, compile/graphs, static/quantized/offloaded caches, CPU and
other serving backends are unsupported. No claim is made for arbitrary
checkpoints, adapter settings, long contexts beyond those measured or GPUs.

Export an existing eligible portable artifact explicitly:

```bash
python -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --adapter /exports/portable --base /models/immutable-snapshot \
  --merged --output-dtype bfloat16 \
  --execution-profile qwen3_vl_bf16_canonical_cache_v1 \
  --out /exports/canonical-bf16
```

Use the Qwen3.5 profile name for its artifact. Config and transaction identity
bind the choice. `CosmosFrameworkRuntime` validates and installs it after strict
stock weight loading. **Plain HF `from_pretrained` alone does not activate this
custom arithmetic.** For portable inference, first strict-load the adapter,
put the model in eval, then call
`posttrain.model.adapter_bf16_execution.install_adapter_inference_profile(net,
profile)`. The installer preserves Parameter identities, tying, state-dict keys,
trainability flags and optimizer membership, and validates FP32 factors,
magnitudes, initial buffers and extra modules. Calls cannot silently detach a
training graph. The separate FP32 equivalence oracle keeps its stock execution.

Cross-loader generation must use identical settings, including the saved base
EOS/pad/BOS configuration. Training builders derive defaults from model config;
the HF merged loader reads `generation_config.json` when present. The first
comparison exposed different EOS masks on closed-chat training probes: first
logits were exact, continuations differed. Controlled decode v2 retains all
inputs and thresholds with identical settings. Its failed predecessor remains
in the evidence package.

## Exact qualified artifacts and populations

These are the three existing short-training artifacts from `df334fc`, not three
new training profiles. Inference is one GPU regardless of how they were trained.
All include an FP32 trained final language norm (LR×.07, weight decay .01),
FP32 adapters/reductions, zero dropout and preserved PiSSA initial factors.

| Artifact label | Checkpoint | Language | Merger/projector | Selected vision | Original training |
| --- | --- | --- | --- | --- | --- |
| `vl_language_fsdp` | Qwen/Qwen3-VL-2B-Instruct @ `89644892e4d85e24eaac8bacfd4f463576704203` | rsPiSSA r4/α7, LR×.8 | LoRA r6/α9, LR×.4 | rsDoRA r2/α3, LR×.15 | 1×2 FSDP2, full AC |
| `vl_vision_hsdp` | Same snapshot | LoRA r8/α12, LR×1 | rsDoRA r4/α9, LR×.3 | rsPiSSA r3/α6, LR×.1 | 2×2 HSDP, full AC |
| `q35_vision_single` | Qwen/Qwen3.5-0.8B @ `2fc06364715b967f1860aea9cf38778875588b17` | LoRA r8/α12, LR×1 | rsDoRA r4/α9, LR×.3 | rsPiSSA r3/α6, LR×.1 | one GPU, full AC |

Qwen3-VL targets 56 `model.language_model.layers.*.self_attn.{q_proj,v_proj}`
linears, eight `model.visual.{merger,deepstack_merger_list.*}.linear_fc{1,2}`
linears and `model.visual.blocks.23.attn.qkv`. Qwen3.5 targets 48 language MLP
`up_proj`/`down_proj` linears, two final merger linears and vision block 11's
`attn.qkv`. Complete concrete names, shapes, settings and ownership remain in
artifact manifests. These labels do not imply language-only training.

The 96/104/92-case populations contain every saved training probe, the preserved
24-input population, the earlier 32 inputs and a separately defined new set of
32 text/image/video/combined inputs. New inputs were fixed before candidate
measurements, using a different WTS clip and prompt/sequence lengths. Each
loader must pass cache/full recomputation, three video-prefix continuations,
prefix immutability and growth limits. Portable versus merged outputs must pass
`atol=5e-4`, `rtol=2e-4`, mean error ≤5e-5 and identical controlled greedy IDs.
Existing cache distribution/logit gates are unchanged, including max logit error
2.0, max TV .1 and exact generated IDs. Matching tokens alone is insufficient.

All three profiles pass. Maximum four-token cache logit differences are .375,
.25 and .1875 respectively; repeated-prefix differences are zero. Each profile
also passes a predeclared extension on the shortest/longest example of each
modality with 32 generated tokens, plus 2K/8K inputs with four tokens. Those ten
cases reuse the fixed populations and are not an untouched task test. They pass
through both declared loaders in separate installed `cd60999` processes. The
strict cross-loader merge comparison remains the complete 96/104/92-instance
population; the horizon extension checks each loader's cache consistency.

## Cache resource costs

Measurements use the same inputs and allocator accounting on an isolated A100;
other GPUs on the host may be busy. Short latency is a median over 20 repetitions
of text/image/video forwards. Long latency is median prefill over three runs.
Ratios compare with each artifact's separate FP32 export, not stock BF16.

| Artifact | Weight GiB | Short peak GiB / latency ratio | 2K cache GiB / peak GiB | 8K cache GiB / peak GiB | 8K latency / FP32 |
| --- | ---: | --- | --- | --- | ---: |
| Qwen3-VL language PiSSA | 3.963 | 4.047 / .863× | .219 / 4.317 | .875 / 5.281 | 1.98× |
| Qwen3-VL vision PiSSA | 3.963 | 4.047 / .862× | .219 / 4.317 | .875 / 5.281 | 1.98× |
| Qwen3.5 vision PiSSA | 1.589 | 1.707 / .259× | .042 / 1.751 | .112 / 2.082 | .777× |

All existing resource gates pass. Weight memory is .5× FP32. Qwen3-VL KV is .5×;
Qwen3.5 total cache is .633× at 2K and .543× at 8K because recurrent state remains
FP32. Canonical Qwen3-VL 8K prefill is slower despite lower memory. These are
fixture measurements, not universal throughput claims.

The separate **stock BF16** cost reference exposes an important limit:
Qwen3-VL 8K prefill is about **25× slower** with canonical arithmetic (6.08 versus
.243 seconds), despite similar weight, KV and peak allocation. At 2K it is
5.54–7.62× slower; short mixed-modality latency is .973–1.007×. These stock
BF16 references still fail numerical cache consistency. Qwen3.5 canonical 8K
prefill is .974× stock BF16 latency and .839× peak allocation, with 1.085× cache
bytes; at 2K its cache is 1.266×. FP32 recurrent state costs memory. Existing
resource gates compare against FP32 and do not impose a long-prefill latency
limit; passing them must not conceal this cost.

Portable inference has a separate measured overhead from constructing dense
effective weights on each call. Relative to the corresponding canonical merged
model, its short-input latency is 2.04× / 1.97× / 1.68× and peak allocation is
1.115× / 1.115× / 1.139×, respectively. At 8K, latency is 1.013× / 1.014× /
1.030×; cache bytes are equal. Parameter and buffer bytes are recorded
separately. These are portable-loading costs, not extra passing deployment
resource gates: the predeclared deployment gates apply to merged artifacts.

Task measurements use the same original fixed artifacts, corrected preprocessing,
decoding and reused 576-row benchmark. FP32 references remain attributed to
`df334fc`; new canonical evaluation uses attested `cd60999`. Base identity,
adapter identity and adapter-weight digests match before comparison. Source
descriptor hashes differ by declared output precision, as designed.

| Artifact | FP32 correct | Canonical BF16 correct | Delta pp | Paired row 95% interval pp |
| --- | ---: | ---: | ---: | --- |
| Qwen3-VL language PiSSA | 269/576 | 271/576 | +.347 | [0, .868] |
| Qwen3-VL vision PiSSA | 269/576 | 268/576 | −.174 | [−.868, .521] |
| Qwen3.5 vision PiSSA | 260/576 | 257/576 | −.521 | [−1.389, .347] |

All pass the existing task gates (point drop at most 1 pp, paired-row 95% lower
bound at least −2 pp; 10,000 resamples, seed 42). This is precision drift on
short-training correctness artifacts, not a training-quality or unadapted-model
improvement claim. Full-WTS quality remains the separate experiment below.

## Selective checkpoint export

`load_adapter_tensors` uses the existing consolidated mmap reader and clones only
compact selected tensors, releasing the full mapping. DCP reads metadata and
validates **all model storage extents and tensor coverage**, then asks the normal
DCP planner to read only the portable inventory. It never deserializes optimizer
payloads. Full-checkpoint completeness, stable source provenance, alias equality,
identity, initial PiSSA factors and existing transaction rules still apply.
No partial-load option can silently skip missing adapter tensors.

The consolidated reader was already mmap-based; the improvement is selection
and lifetime control, not a claim that it previously touched all base pages.
DCP no longer consolidates the full frozen model. Export continues through the
existing bounded `AdapterTensorSource` and public CLI:

```bash
python -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --checkpoint /runs/checkpoints/iter_000004 --base /models/immutable-snapshot \
  --merged --output-dtype float32 --out /exports/merged
```

The public MoE export uses the preserved **iter 4** versioned stock-compatible
PiSSA checkpoint for Qwen3-VL-30B-A3B-Instruct snapshot
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`. The historical portable artifact is
update 5 and is not used as the same-update oracle. An independent old full
reader extracted the iter-4 oracle; it was never an input to the new exporter.
All 882 output tensors are bitwise equal. Portable FP32 versus actual stock
merged loader outputs are exact on all 64 established inputs, with existing
expert routing semantics. That FP32 loader verification uses two A100s with a
recorded device map; the export itself uses no GPU. Legacy failed MoE PiSSA
artifacts remain guarded.

| Public export | Whole-job limit | Peak process-tree RSS | Wall seconds | GPU allocation |
| --- | ---: | ---: | ---: | ---: |
| Fresh MoE DCP → FP32 HF | 16 GiB | 4.049 GiB | 673.19 | 0 |
| Reuse completed same export | 16 GiB | 3.951 GiB | 406.94 | 0 |
| Dense consolidated → FP32 HF | 8 GiB | 3.201 GiB | 52.80 | 0 |

The cgroup includes the exporter and children, has swap disabled and records no
OOM. Charged page cache reaches the memory cap and is reclaimed; **4.049 GiB is
RSS, not whole-job peak usage**. Fresh MoE temporary disk peaks at
124,325,979,266 bytes, predominantly its staged output. No full-model allocation
was moved outside the measured job. Verification/reuse still reads large files
and is not cheap. Timing is observational under shared-host activity.

A real consolidated checkpoint saved by the existing trainer and a DCP recovery
export have all 625 tensors equal. SIGTERM after the first shard leaves an
uncommitted staging tree; the public command validates/rebuilds it and publishes
only a complete export under an 8 GiB cap. Completed reuse retains its marker.
Physical-format full-model exporters have **not** acquired bounded-memory
qualification from these HF results.

## Matched full-WTS experiment

Both runs use the same Qwen3-VL-2B snapshot, vision-PiSSA mixed recipe above,
seed 42, one A100, full AC, base LR 2e-5, microbatch one and gradient accumulation
five. Only component execution (`merged_weight_bf16_v1` versus `factorized_v1`)
and the corresponding explicit model/artifact tag change. The factorized
`qwen3_vl_hf_stock_v1` control uses the same HF5 image/video batching and norm
precision. It retains real low-rank branches, rather than simulating factorized
execution with dense weights.

Each consumes exactly 5,555 training rows in 1,111 updates, with initial/final
loss on all 2,676 validation rows. Four video frames, corrected media budgets,
16,384 maximum tokens, data order, component settings and trainable ownership
are matched. Both preflights verify all 139 trainable tensors update, frozen
state stays unchanged, resumed next-update state is exact and portable warm
start/reload plus FP32 merge pass. Both full epochs and the full validation passes completed without skipped
updates. The final training results are:

| Measurement | Factorized control | Rounded-weight new mode |
| --- | ---: | ---: |
| Token-weighted mean training loss | .246032 | .249170 |
| Initial validation loss (2,676 rows) | .843903 | .843903 |
| Final validation loss (2,676 rows) | .171627 | .170830 |
| Summed update time, seconds | 6,279.95 | 6,248.41 |
| Training rows / summed update second | .88456 | .88903 |
| Whole training job GPU-hours | 3.11151 | 3.10957 |
| Peak CUDA allocated GiB | 4.54391 | 6.15688 |
| Peak CUDA reserved GiB | 4.62695 | 6.50391 |
| Peak sampled device MiB | 5,919 | 7,841 |
| FP32 export task accuracy | 527/576 (91.493%) | 527/576 (91.493%) |

Whole-job time includes setup, checkpointing and both full validation passes;
update timers exclude validation. Evaluation and the scratch verification update
are separate costs. These are shared-host observations, not universal throughput
measurements. The new mode costs about 35.5% more peak CUDA allocation in this
matched full-AC recipe. The task result is a separate measurement from losses.

The final checkpoint continuation check restores update 1110, replays exactly
five rows through the existing trainer, then compares with uninterrupted update
1111. Both pass: rounded-weight parameters are exact; factorized maximum error
is 7.28e-12 within `atol=1e-7`, `rtol=1e-5`. Losses are exact. Optimizer tensors
pass the same tolerance; scheduler, RNG, data cursor, update clock, optimizer
group settings, frozen base and initial factors match exactly. Method
initialization/decomposition is forbidden during restoration. Source/wheel and
actual host-environment hashes equal the original run. The verification consumes
one scratch update per model and does not change original checkpoints.

Final portable identity, provenance and all 141 compact tensors are bound to
checkpoint 1111 before the merged-output comparison. The new-mode FP32 model
passes all fixed 32 text/image/video/combined fixtures through the actual Cosmos
loader: maximum logit difference 1.90735e-4, mean 5.75553e-6 and all 256 controlled
argmaxes equal. This does not extend BF16 cache qualification to the newly
trained full-epoch weights; the cache results above remain attached to their
three original fixed artifacts.

The **full-trained factorized control fails strict FP32 merge equivalence** on
image fixture 12: 13 of its 1,215,488 logits violate `atol=5e-4`, `rtol=2e-4`.
Overall maximum error is 7.80106e-4 and mean 1.54313e-5; all 256 argmaxes match.
The mean and token results do not waive the elementwise failure. Its preflight
and full-state resume passes remain valid, but this final merge criterion is
incomplete.

Read-only diagnostics reproduce the failed factorized output exactly and verify
all 65 exported target weights bitwise against the existing CPU merge formula.
The first target, `model.visual.deepstack_merger_list.0.linear_fc1`, differs by
2.09808e-5 between factorized and dense evaluation on identical inputs. Both
FP32 computations differ from the independent FP64 reference; subsequent
projections accumulate rounding differences. Substituting the actual exported
dense weights only inside the diagnostic reproduces the actual stock loader
exactly. This isolates execution association rather than a missing tensor,
wrong scale or loader conversion in the tested path. It is **not** a correction
to the factorized artifact and is not counted as a pass. No gates, original
states, export values or factorized semantics were changed. A general guarantee
of this strict tolerance for trained factorized arithmetic remains unqualified;
the separately versioned shared-weight mode passes the tested full-epoch case.

The task comparison uses both FP32 exports, greedy decoding up to 32 tokens,
and the same 576-row **reused scene-disjoint benchmark**, a subset of validation.
It is not an untouched final test. Parsed choices match on every row: no
wrong-to-right or right-to-wrong changes, no invalid answers. Paired row and
scene-cluster bootstrap intervals for the observed difference are both [0, 0]
percentage points because the paired differences are all zero; this is not a
guarantee of population equivalence. All 576 rows are environment/full-video
questions from ten scenes. One seed is descriptive; no comparison with a
historical different model/preprocessing or unadapted checkpoint is used to
claim improvement. These task results describe the FP32 exports and do not
waive the control's image-fixture merge failure. Physical-model correctness
runs do not establish downstream physical task quality.

## Revision and review provenance

| Installed source | New evidence |
| --- | --- |
| `4423805` | Both real lifecycle preflights, matched full-WTS training and final-update replays; fresh public MoE checkpoint export |
| `d2d63f3` | Reviewed source-path protection; completed-export reuse, exact all-tensor checks, interruption/recovery and real MoE stock loader |
| `6a1a36d` | Qwen3-VL canonical cache 96/104 cases, resources and strict portable/merge checks; full-WTS finalization/evaluation launcher |
| `cd60999` | Qwen3.5 canonical cache 92 cases/resources, both loaders' extended horizons, portable costs, final installed CPU/GPU regressions, canonical task evaluations and final WTS artifact checks/diagnostics |

Final wheel SHA256 is
`8d61c872bfea0a642f73166e5026c434a49d1dc48bb3e269f463d0ef6c1be55e`;
1,912 installed package files match its clean source archive. Torch is
2.10.0+cu128, Transformers 5.8.1 and Triton 3.6.0. These are attested host venvs,
not newly built Slurm containers. The final installed suites report 14 cache GPU
passes and 81 affected CPU passes / one skip; overlapping earlier checks are
not added to these counts.

Independent reviews cover extraction, the factorized control, transaction path
protection, canonical Qwen3-VL and Qwen3.5. Confirmed fixes include trailing-slash
transaction/source collisions, FP32 adapter-child ownership, pre-mutation cache
argument validation, direct-call gradient guards, explicit mask semantics,
actual-device kernel context, repeat-install revalidation, and Qwen3.5's explicit
`use_cache=None` and invalid-input cache ownership. Final snapshots and reviewer
probes are retained. No remaining confirmed production defect was reported.

Three additional evidence reviews cover the qualification records, portable
horizons/costs and final WTS results. Confirmed reporting/harness fixes include
single-letter invalid-answer counting, preservation of exact historical memory
measurement scripts, binding final portable/merged state to checkpoint 1111,
actual host-environment equality on replay and merged-only resource-gate wording.
The final reviewer independently recomputed all 65 control target weights,
checked both 141-tensor portable snapshots, raw logits and all task rows. No new
conversion or accounting defect was found. The factorized control's final strict
FP32 merge criterion remains incomplete; the package is reviewable with that
limitation, rather than an unconditional all-artifact qualification.

Local evidence and reproduction scripts are in
`/localhome/local-rarunachalam/cosmos-adapter-cache-checkpoint-20261105`.
Earlier failures, independent input definitions, per-process wheel attestation,
all-tensor comparisons, memory samples and task rows remain preserved. No push,
PR publication, HPO, four-bit work or extra model-family expansion is included;
PR #342 was not reopened or updated. Its last known state is closed/unmerged;
a read-only live recheck was unavailable because of organization SAML enforcement.
