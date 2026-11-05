# Versioned rounded-weight BF16 adapter execution

This records the original new-artifact qualification. The subsequent
[canonical cache profiles](adapter-cache-checkpoint-progress.md) address cache
consistency with explicit Cosmos loader activation. Those custom profiles do
not change the original stock-execution failures or the historical results here.

Status at `df334fc`: this opt-in new-artifact execution was approved on 2026-11-05 and is
implemented. Real GPU training, exact resume, portable warm start/reload, both
precision-specific stock exports and resource gates pass for the three profiles
below. **Broad BF16 deployment qualification is incomplete:** stock cached/full
and split-prefix execution exceed the existing BF16 drift gates. No threshold
was changed. This is not a repair or relabeling of any legacy BF16 artifact.

`merged_weight_bf16_v1` is selected before initialization, stored in portable
format `cosmos-posttrain-adapter-v4`, and required by exact resume, warm start and
merged identity. The existing HF SFT builder binds stock Transformers 5.8.1 /
PyTorch 2.10.0, SDPA, dense Qwen3-VL and Qwen3.5. The runnable recipe is
`cosmos_framework/posttrain/configs/sft_merged_weight_bf16.py`; environment
variables `SFT_MODEL_PATH`, `SFT_DATA_ROOT` and
`SFT_MERGED_VARIANT=vision_pissa|language_pissa` specify staged inputs and
independent component settings. Base/compute storage must be BF16.

Each adapted Linear constructs its differentiable dense effective weight using
FP64 low-rank accumulation and DoRA direction normalization, rounds once to
FP32, then casts to the input model's declared weight precision. LoRA reuses the
existing native helper/scaling. PiSSA keeps the untouched original base and
initial decomposition, subtracting the initial low-rank product from the current
product. Only FP32 factors, DoRA magnitudes and selected extra modules are
optimized. Frozen base ownership and parameter identities are preserved.
Dense construction is transient but can substantially increase training memory
and time compared with factorized execution; measurements must accompany support.

The selected extra norm executes the exact stock normalization with an effective
weight cast to the input dtype, retaining its optimized FP32 master. Dense
Qwen3-VL uses stock image/video visual batches with ordered dummy modality
passes to preserve distributed collectives and gradients. Qwen3.5 uses its
existing ordered stock forward. Zero adapter dropout is required for this mode;
legacy factorized dropout support is unchanged.

BF16 portable and stock merged models must pass the existing strict merge gates
(`atol=5e-4`, `rtol=2e-4`, mean absolute logit error ≤5e-5, all controlled greedy
positions equal). A separate FP32-promoted portable reference and FP32 export
must pass the same gates. Whole-model modality batching, caches and generation
are part of the contract; equal weights alone are insufficient. Cross-precision
drift is reported, not required to match both incompatible legacy references.
All legacy failed references and acceptance criteria retain their status.

Cosmos reasoner loading validates the execution tag, marker, precision and
pinned libraries, and selects stock SDPA arithmetic. Exported models contain no
adapter branches and can use the same pinned stock HF class. Other libraries,
attention kernels, quantizers and serving backends are unqualified; they cannot
be claimed equivalent merely because they read the weights.

| Requirement | Code | Current validation |
| --- | --- | --- |
| Methods, independent component selection | `config/adapters.py`, `model/adapters.py`, recipe | CPU all-method/rs gradients; three real mixed-component profiles preserve frozen state and update every intended parameter |
| Norm ownership / rounded computation | `model/adapter_bf16_execution.py` | CPU stock-value/ownership checks; trained final norm participates in real optimizer, checkpoints and both exports |
| Ordered real/dummy modalities | `model/qwen3_vl_hf.py`, existing Qwen3.5 path | CPU value/gradient equality and real full-AC training, including different modalities across FSDP2/HSDP ranks |
| Versioned resume/warm start | `checkpoint/adapters.py` | CPU incompatible-profile rejection; all three GPU profiles reproduce next update exactly and warm-start with fresh optimizer/clock/data |
| Portable / BF16 and FP32 stock exports | shared merge helper, streaming exporter, reasoner loader | Exact every-tensor and whole-forward reload on three real profiles; generation matches portable separately with cache enabled and disabled |
| Training / deployment resources and WTS quality | qualification tools | Measured separately below; BF16 cache gate remains failed regardless of resource or task results |

Historical qualification remains 12 named checkpoints, 48 fixed profiles and
47 fully passing legacy profiles. New-mode executions are separate. The
576-row scene-disjoint WTS benchmark is reused, not an untouched final test;
correctness fixtures and physical-model runs do not establish task performance.

## Exact real-model profiles

All runs use the installed clean `df334fc4a17925bf985b8fb8f2a64a93e6aa567e`
wheel (`71202b6003111df4ec314873f24070db9da4ad371fd84c0ba75f67fac7c2cbaa`),
Transformers 5.8.1 / Torch 2.10.0+cu128, A100-SXM4-80GB, SDPA and full activation
checkpointing. Frozen base storage/compute is BF16; factors, magnitudes and the
extra norm have FP32 masters. Adapter dropout is zero. Component rank/alpha,
scaling and LR multipliers are independent; this is representative coverage,
not every possible configuration.

| Profile | Model / immutable checkpoint | Language | Projector | Vision | Distribution / updates before save |
| --- | --- | --- | --- | --- | --- |
| `vl_language_fsdp` | Qwen/Qwen3-VL-2B-Instruct, `89644892e4d85e24eaac8bacfd4f463576704203` | rsPiSSA r4/α7, LR×0.8 | LoRA r6/α9, LR×0.4 | rsDoRA r2/α3, LR×0.15 | 1×2 FSDP2 / 4 |
| `vl_vision_hsdp` | Same exact checkpoint | LoRA r8/α12, LR×1 | rsDoRA r4/α9, LR×0.3 | rsPiSSA r3/α6, LR×0.1 | 2×2 HSDP / 4 |
| `q35_vision_single` | Qwen/Qwen3.5-0.8B, `2fc06364715b967f1860aea9cf38778875588b17` | LoRA r8/α12, LR×1 | rsDoRA r4/α9, LR×0.3 | rsPiSSA r3/α6, LR×0.1 | one GPU / 8 |

Base LR is 1e-4. The second/third profiles use language weight decay .01;
other adapter groups use zero. All profiles separately train
`model.language_model.norm.weight` (LR×.07, weight decay .01). PiSSA uses
`svd_lowrank` with four initialization iterations and retains initial factors.
Qwen3-VL selects 56 language q/v projections, eight final/deepstack merger
linears and `model.visual.blocks.23.attn.qkv`; Qwen3.5 selects 48 language MLP
up/down projections, two merger linears and `model.visual.blocks.11.attn.qkv`.
Resolved identity records every concrete module, shape, alias and setting.

The eight correctness rows contain text, PNG, synthetic video, combined media,
and four real WTS video questions. They are not a task-quality training corpus.
All 132 / 139 / 105 intended trainable tensors changed respectively; optimizer
membership is exact and duplicate-free. Initial BF16 output error and resumed
next-update parameter error are zero. Full optimizer/scheduler/data state checks
and portable warm-start checks pass; HSDP replica hashes match after updates and
resume. These are three new execution profiles, not additions to or retroactive
passes within the historical 48-profile matrix.

Fresh processes compare 64 / 72 / 60 prompt instances respectively: every saved
training probe (including rank repeats), the preserved 24-input population and
all 32 newly defined inputs. New inputs span text, image, video and combined
media, approximately 50–1942 tokens, plus separate 2K/8K cache cases. Both BF16
and FP32 stock-loader outputs match their precision-matched portable oracle
exactly. All 625 / 625 / 473 exported tensors match, including ties and the
trained norm. For each precision and profile, all 64 cache-on/cache-off
four-token generation comparisons against the corresponding portable model
also match exactly. This does not imply cache-on equals cache-off within BF16.

## Cache blocker, not an adapter merge pass

| Profile | BF16 cache vs full recomputation | BF16 prefix reuse | FP32 equivalents | Corrected prefix ownership |
| --- | --- | --- | --- | --- |
| `vl_language_fsdp` | 26/32 pass | 18/48 pass | 32/32 and 48/48 pass | 48/48 forks per precision pass |
| `vl_vision_hsdp` | 25/32 pass | 27/48 pass | 32/32 and 48/48 pass | 48/48 forks per precision pass |
| `q35_vision_single` | 32/32 pass | 36/48 pass | 32/32 and 48/48 pass | 48/48 forks per precision pass |

The original ownership helper snapshotted state after the first continuation.
Independent review identified this evidence gap; a separate real-GPU check now
snapshots inside `_fork_dynamic_cache` before the first and each subsequent
continuation. All 288 forks preserve every original cache tensor and length;
Qwen3.5 mutable recurrent states use distinct storage. Original numerical
failures remain failed. RoPE, masks and media-boundary audits also pass.

For Qwen3-VL, the first prefix divergence is frozen
`model.language_model.layers.0.mlp.down_proj`: identical `[1,78,6144]` shared
inputs and identical weights yield 70,350 different outputs (max .0625) when
computed as a 125-token versus 78-token GEMM. FP64-reference comparisons confirm
shape-dependent BF16 arithmetic, before any changed adapter weight can explain
the difference. Unadapted stock Qwen3-VL also fails (25/32 cache and 30/48 prefix).

Changing only GEMM reduced-precision reduction gives 27/32 and 27/48; selecting
SDPA math gives 28/32 and 27/48; both gives 28/32 and 33/48. These are diagnostics,
not production changes or qualified candidates. Every input is retained.
For Qwen3.5, first-layer convolution and gated-delta inputs q/k/v/g/beta match
exactly; the stock chunked gated-delta output first differs by 4.66e-10. An FP64
reference of that calculation has exact shared-prefix equality. The divergence
is upstream of the first selected MLP adapter target. No routing or thresholds
were changed, and no production integration defect was confirmed.

A future path would require explicitly versioned whole-model activation
arithmetic, not another adapter-weight accumulation rearrangement. One concrete
option is BF16 stored/rounded weights with FP32 transient linear, embedding,
convolution, normalization, attention and cache execution. That would need a
new training graph, new identity, explicit Cosmos loader activation and new
qualification. Plain stock BF16 consumers would be unsupported, KV memory would
be FP32, and runtime conversion could lose latency savings. It is **not enabled
or approved by the current stock-BF16 contract**. Existing gates and legacy
failures must remain unchanged when reviewing such an alternative.

## Measured resources and limits

Relative to each profile's FP32 export, BF16 weight bytes and 2K/8K KV bytes are
.5×. Short-input peak allocated memory is .503× / .503× / .507× and median
latency .886× / .867× / 1.001×, respectively. All predeclared resource gates pass.
These A100 fixture measurements are not universal serving-throughput claims.

A matched eight-update Qwen3.5 cost control uses the same checkpoint, data,
component settings, seed, full AC, optimizer and GPU. New dense execution takes
13.649 seconds total versus 13.199 for factorized execution (1.034×); median
update time is 1.918 versus 1.731 seconds (1.108×); peak allocated memory is
2.112 versus 1.851 GiB (1.141×). Both optimize 1,824,768 elements. Retained
low-rank state stays small, but transient dense construction loses some training
efficiency. This control is a cost measurement, not another qualified profile.

Public BF16 / FP32 exports take 41.79 / 55.31 seconds for the language-PiSSA
Qwen3-VL profile, 38.76 / 48.14 for the vision-PiSSA profile, and 27.24 / 28.65
for Qwen3.5. Artifacts occupy 3.974 / 7.937 GiB for Qwen3-VL and 1.610 / 3.199
GiB for Qwen3.5, including assets. Export does not mutate the portable artifact.

## Evidence and reproduction

Local evidence root: `cosmos-adapter-export-precision-20261105` under the shared
workspace. `new-mode-summary.json` maps each result to its source, wheel,
identity, input hashes and report; `new-mode-wts-summary.json` contains paired
WTS results. `reproduction-commands.md` and per-step `execution.json` preserve
exact commands. Review reports and independent probes are under `review/`.
No generated environments, weights or run logs are added to the source tree.

Independent review of `df334fc` found no confirmed production defect. The
installed bounded CPU suite passed 138 tests (four expected skips), and the
installed native LoRA CUDA suite passed 26 tests. Earlier collection attempts
and a GPU-visible invocation of CPU fixtures are preserved with their corrected
commands; they are not silently counted as passing runs. Final documentation
changes do not alter the installed production source used for these results.

## Paired WTS task measurements

All 576 evaluation rows, corrected media preprocessing, SDPA and greedy decoding
are identical within each precision pair. This is a reused scene-disjoint
benchmark, not an untouched final test. These small-fixture-trained adapters are
not compared with an unadapted checkpoint and no accuracy improvement is claimed.

| Profile | FP32 correct / 576 | BF16 correct / 576 | BF16 − FP32, percentage points | Paired 95% interval | Task gate |
| --- | --- | --- | --- | --- | --- |
| `vl_language_fsdp` | 269 | 267 | -0.347 | [-1.215, 0.521] | pass |
| `vl_vision_hsdp` | 269 | 265 | -0.694 | [-1.389, -0.174] | pass |
| `q35_vision_single` | 260 | 255 | -0.868 | [-1.736, -0.174] | pass |

The predeclared task gate is at most a one-percentage-point drop with paired
bootstrap lower bound at least −2 points (10,000 resamples, seed 42). All three
pass. This does not waive the failed BF16 cache criterion. FP32 equivalence,
cross-precision behavioral differences, task results and resource measurements
remain separate.

Readiness: the bounded exporter is qualified for the recorded artifacts. The
new mode implements and validates precision-specific adapter training/export;
its full BF16 deployment contract remains incomplete. Use the qualified FP32
export where the unchanged cache criteria are required. Broader checkpoint,
backend, nonzero-dropout and custom-consumer claims are not supported by these
three executions.
