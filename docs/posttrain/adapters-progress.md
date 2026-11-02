# Posttrain adapter implementation qualification

This is historical Cosmos3 validation evidence. The 576-row subset is a reused
scene-disjoint benchmark, not an untouched final test. These model/preprocessing
results are not a baseline for Qwen3-VL-2B; see the
[final family review](adapter-final-review.md) for matched comparisons.

This is the historical Cosmos3 qualification record. The subsequent implementation
and real-checkpoint results for other families are tracked in
[adapter-family-progress.md](adapter-family-progress.md). Results below retain the
source revisions and execution profiles that produced them.

Branch `feat/posttrain-vlm-adapters`; [PR #342](https://github.com/NVIDIA/cosmos-framework/pull/342)
was closed, unmerged, at the user’s request on 2026-10-30. The implementation
and qualification artifacts remain available on the branch; this report does
not claim that the new adapter extension has landed on main.
Implementation starts from main `7738e9e7969e30160163b9d7941782fefd6eaa5c`.
The audited baseline was `1a2f03ab6111b457633827736f94f67d55bd5202`;
intervening changes affect quantization/docs, not the SFT builder or adapters.
Native TOML LoRA/rsLoRA and pure LoRA merge already existed. This contribution
reuses them in the separate posttrain SFT builder, with the existing trainer,
packer, checkpoint transactions and HF materializer. Earlier WTS/AutoML runs
used unchanged framework code and are baseline context only.

## Requirement status

Paths below are relative to `cosmos_framework/posttrain/` unless stated otherwise.

| Requirement | Implementation and evidence | Status |
|---|---|---|
| Explicit components, independent settings, identity/ties | `config/adapters.py`, `model/adapters.py::resolve_targets`; `tests/test_adapters.py` checks each selector, aliases, placement/counts and optimizer IDs | Implemented and validated |
| LoRA / rsLoRA | Native `LoraInjectedLinear` reused; 20 method/world/AC GPU cases include both | Implemented and validated |
| DoRA magnitude and pure merge | `PosttrainAdapterLinear`, `optimizer_groups`; reference gradients, magnitude updates, full/portable state, real-model reload | Implemented and validated |
| PiSSA initialization/residual/artifact contract | `initial_A/initial_B`, untouched W0 and rank-2r conversion; principal-component, continuation and real-model reload tests | Implemented and validated |
| Fresh, warm start and exact resume | `model/qwen3_vl.py::Qwen3VLBuildHooks.after_load_before_parallelize`, SFT `build_trainer`; CPU full-recipe and GPU next-update comparisons | Implemented and validated |
| Vision/projector gradients, missing modalities and AC | Isolated components plus first/final vision layers; real text, PNG, MP4 and WTS; one/two GPUs, AC on/off | Implemented and validated |
| Portable artifacts and failure handling | `checkpoint/adapters.py`; immutable publication, base/media identity, strict state, incomplete/wrong artifacts; all 35 fresh-process GPU reloads bitwise exact | Implemented and validated |
| FP32 merged export | `checkpoint/tools/adapters.py`, existing HF materializer; 35 numerical cases and complete retained metadata; re-resolution rejects corrupt target manifests | Implemented and validated: complete numerical matrix plus final installed/GPU metadata/target validation |
| Installed package provenance | Wheel/source comparison and runtime byte attestation of all 1,838 package files | Implemented and validated: GPU runtime `480be5a`, merged-main CPU wheel `faa61c8` |
| Full WTS epoch/evaluation | Existing SFT/evaluation entrypoints; exact 5,555-row cursor, 2,676-row validation, reused scene-disjoint 576-row benchmark score | Implemented and validated for LoRA, rsLoRA, DoRA and PiSSA; full comparison and job receipts below |
| Recipes, documentation and review | `configs/sft_cosmos3_nano_adapters.py`, WTS converter, [adapter guide](guides/adapters.md), corrected general TOML/training guides, PR #342 | Implemented and validated; complete qualification and WTS evidence below |

## Qualified profile and exclusions

Dense native Qwen3-VL / staged Cosmos3-Nano-VLM (8.7B), BF16 frozen base,
FP32 adapters, AdamW, SDPA, full non-reentrant activation checkpointing or none,
one node with one or two A100 80 GB GPUs; two GPUs use pure FSDP2.
The recipe selects 72 language q/v projections, eight final/deepstack merger
projections and final vision block 26's fused QKV (81 targets; 4,444,160 LoRA
parameters). Independent mixed-method settings and an extra native RMSNorm are
also qualified. First/final vision block coverage is tested separately.

Generic HF wrappers, native MoE, HSDP, TP/EP/CP/PP, compilation, quantization,
stacked adapters, custom tied HF merged targets, larger worlds/multiple nodes
and BF16 merged inference are excluded. Full base loading precedes sharding,
so the base must fit on each GPU. CPU fixtures additionally cover FP32 bases,
eager attention, exact-SVD PiSSA, and rsLoRA with DoRA/PiSSA; these are not
additional real-model GPU claims. The general Torch 2.11 SFT environment remains
a separate qualification case.

## Final main synchronization

Main advanced to `b42e1a571dd7c48dced286c837be3d0f81d43f98` during
qualification (PRs #331 and #333: internal-reference cleanup in comments,
documentation and test fixtures). This branch merges those changes, preserving
the adapter lifecycle arguments and materializer format v5. Comparison with
qualified runtime `480be5a` confirms no production execution changes: all changed
Python ASTs outside tests are identical after removing documentation strings,
except one removed job ID inside a generated shell comment. Dependency pins are
unchanged. Affected checks pass again: **193 focused tests** (63.84 s) and
**205 adapter-disabled tests, two optional skips** (23.85 s). GPU records retain
the actual installed revision and digest; they are not relabeled as this merge.

The merged-main wheel `faa61c8ec897002f635e2a92d5c8f75ec6a67687` also passes
all **193 installed CPU tests**, with all 1,838 packaged files byte-matched to
source and installation. Its wheel SHA-256 is
`2213c96f129cb2ed19d9c4b35d27ec2e96a794279a8593e1bdf49e71beb37157`.
Subsequent main `86e1ee52f276c11d6cc66dfd17da2795dfbe937b` (PR #324) was
inspected: generation skills, CI and documentation only, with no overlapping
files, framework source or dependency changes.

## Correctness and numerical evidence

The current focused adapter/export suite passes **193 tests** locally (64.18 s).
The sealed `480be5a` wheel passed **193 installed tests** locally (65.24 s) and
in the CUDA container (104.32 s). The four newest tests reject malformed target
type/component/scaling/count metadata before export. The actual Cosmos3-Nano-VLM
architecture also resolves all 81 targets and the extra norm on meta exactly,
and the final real-model GPU export/reload check passes.
Adapter-disabled regression on `f29da1c`: **205 passed, two optional skips**
(22.47 s), covering configuration, entrypoint, objective, packer, trainer
validation, native Qwen3-VL loading and HF export.

CPU: Python 3.12.3, torch 2.10.0+cpu, Transformers 5.8.1, TorchCodec 0.10.0,
safetensors 0.8.0. GPU: Python 3.13, torch 2.10.0+cu128, torchvision 0.25.0+cu128,
Transformers 5.8.1, CPU TorchCodec 0.10.0, av 18.0.0. PEFT 0.21.2 is a test
reference only. The guide gives isolated installation and reproduction commands.

All **35 real-model cases** passed: 20 method/world/AC combinations; 12 isolated
component/world/AC combinations; one full-WTS media/accumulation profile; two
first/final vision cases. Their 53 rank reports contain **386 finite applied
updates**. Every intended trainable tensor changed over multiple updates;
frozen base and PiSSA initial factors stayed unchanged. Initialization error and
controlled resumed next-update parameter error are both zero. Portable reload
error is zero in all cases.

FP32 merged qualification checks all 750 serialized tensors per case exactly,
then native and stock-HF controlled forwards and fresh-process reload. Maximum
logit error is **0.0004294515**, maximum mean error **0.0000178695**; all
**1,696/1,696 argmax positions** agree. Declared bounds are `atol=5e-4,
rtol=2e-4`, mean at most `5e-5`, and exact argmax. Probes include the last eight
tokens after media, not only prompt-prefix tokens.

The BF16 merged path failed its fixed numerical gates (worst total variation
`.1115` for all-component LoRA and `.1356` for early vision; rsLoRA changed-token
regret `.0475`). Those failures are retained; limits were not widened to pass.
The supported export is FP32, compared with a portable adapter promoted from
its recorded training-base values. This does not claim BF16/FP32 generation
equivalence. Portable reload preserves the training precision exactly.

## Revision and job provenance

Training/portable runtime is unchanged since
`3a58c98e4cafe24c73c4392571953c855b69833c`. Subsequent runtime changes are confined
to export precision, export identity and export target validation. This permits
composing the training matrix with new export checks without relabeling old
results as a new training run.

| Stage | Revision | Slurm jobs / outcome |
|---|---|---|
| Training/continuation matrix | `3a58c98` | `37629559`, `37629560`, `37629971`: all training cases passed; original jobs ended failed at the then-unsupported BF16 export gate |
| Complete FP32 numerical matrix | `f29da1c` | Completed cases from `37633099`/`37633101` retained before redistribution; `37633104`, `37636040`, `37636045`, `37636047`, `37636052` completed the 35-case matrix |
| Installed metadata build | `52a92d9` | `37638174`: 189 tests passed; Lustre immutable-publication probe and package attestation passed |
| Real-model metadata contract | `52a92d9` | `37638175`: passed fresh reload, full adapter/media identity and 750 tensors; merged payload SHA identical to the numerical reference |
| Final installed/export target validation | `480be5a` | `37641221`: 193 installed tests; `37641231`: fresh reload, all 750 tensors and full metadata passed; merged payload SHA unchanged |
| Full WTS training | `52a92d9` | `37640623`: scheduled signal handled with full checkpoint 666; incomplete-epoch verifier exited 1 as expected. `37648509`: resumed at update 667 and completed `0:0`; all 1,111 updates and the final 2,676-row validation verified |
| Full WTS export/evaluation | `480be5a` | `37652713`: export, fresh reload and all 2,676 predictions completed; wrapper exited 1 because the reused-benchmark JSON was not mounted. CPU reconciliation `37653787` completed `0:0` with the unchanged verifier and input/prediction hashes |

Immutable artifact SHA-256 values:

| Revision | Wheel | Container |
|---|---|---|
| `3a58c98` | `88eeb16b59156577ade24b60f5a324bd489f3692f66a05445d3c1813521265fb` | `64269714b74a6ee720b081cb24be2e3dbb3911fee0acef675dcc03d88b66838e` |
| `f29da1c` | `66b66fad810a2b998b5c8347bb6c8528c5842a43aae5c10205a758776390f302` | `6861377c581a16585d6c5510d6e2064cb9709589c90d6d7efe130ef1691ab574` |
| `52a92d9` | `b764b457e629eb8a9a23054bc87b71c42a486f5e5cf16d6c2aa8fd2bb674848b` | `8cbc7ccaf2f40391a8c5eb9f6e7c13f889c0404af1dcb1284b9934347b5c055b` |
| `480be5a` | `e32dc80bdf6336a3fad54360751ec0d04d5291ccadc9ebd353fb82b594f9c7cb` | `02b38355a58313bfd33161db595fd7324e8bd26fb208a4ef19b0f21e1eb5f12f` |

The full epoch uses one GPU, microbatch one and accumulation five: 1,111 updates
consume exactly 5,555 rows with no padding/drop. It measures full teacher-forced
validation before and after training, then the existing evaluator reports MCQ
accuracy, invalid answers and generated-token NLL separately. The reused
576-row scene-disjoint benchmark is scored separately. This comparison uses a
fixed recipe; the subset has participated in earlier development and AutoML work.
The full evaluation split has no row/video overlap with training, but 44 rows
share four scene IDs; report both scores. Resource and artifact measurements
are retained with each installed revision. No new AutoML search was started.

## Original ordinary-LoRA full WTS result

![One-epoch training and validation losses with WTS answer accuracy](assets/wts-adapters-one-epoch.png)

[Machine-readable results and artifact identities](assets/wts-adapters-one-epoch.json)
record the measured revisions, package/container digests, job outcomes and evidence
hashes. Training uses `52a92d9`; final export and evaluation use `480be5a`.

The all-component ordinary-LoRA recipe completed exactly one epoch: 5,555 unique
rows, 1,111 applied optimizer updates, zero padding/drop, and cursor
`epoch=1, position=5555, world_size=1, seed=42`. Every recorded loss and gradient
norm is finite; optimizer clocks match every metric step with zero skipped updates.
The real Slurm continuation starts at 667 after committed checkpoint 666, retaining
the initial validation result and the continuous metric/data history.

| Measurement | Result |
|---|---:|
| Token-weighted mean training loss | 0.18750258 |
| Last training-batch loss | 0.02707118 |
| Full validation loss before / after | 0.65918256 / 0.14688044 |
| Validation examples per pass | 2,676 |
| Sum of training step durations | 6,358.95 s |
| Throughput over training steps | 0.87357 rows/s; 1.74714 supervised tokens/s |
| Wall time including setup, both validations, saves and restart gap | 11,620 s |
| Active attempt time / gap between attempts | 11,450 s / 170 s |
| Allocated GPU time across both one-GPU jobs | 11,605 GPU-s (3.224 GPU-h) |
| Peak sampled GPU memory (10 s sampling) | 20,439 MiB (19.96 GiB) |
| Final resumable checkpoint size | 17,590,388,729 bytes |
| Full WTS MCQ accuracy | 2,467 / 2,676 (92.1898%) |
| Reused scene-disjoint benchmark MCQ accuracy | 533 / 576 (92.5347%) |
| Invalid answers, full / reused benchmark | 0 / 0 |
| Generated-token NLL (separate from teacher-forced validation) | 0.01411575 |
| Two-GPU evaluation duration / throughput | 992 s / 2.69758 examples/s |
| GPU allocation for export, reload and evaluation | 1,491 s × 2 GPUs (0.8283 GPU-h) |
| Peak sampled GPU memory across that entire job, GPU 0 / 1 | 43,611 / 34,185 MiB |
| Portable adapter size / export with verification | 17,844,563 bytes / 71.33 s |
| FP32 merged model size / export | 35,084,709,565 bytes / 198.41 s |

All 750 frozen base tensors in the completed checkpoint equal their original
training-dtype values. The final merged export verifies all 750 serialized tensors
exactly. Fresh-process output checks pass the unchanged FP32 bounds: maximum logit
error `0.0001678467`, mean `0.0000121429`, and 32/32 argmax positions agree.
Evaluation memory sampling starts before export/reload, so the GPU 0 peak includes
that work rather than representing inference alone.

The GPU evaluator completed successfully and wrote all predictions, but its batch
wrapper returned `1:0` when the reused-benchmark annotation file was missing from the
container mounts. The file existed on shared storage. After correcting the mount,
CPU-only job `37653787` ran the original verifier unchanged, reconciled every gold
and prediction ID, and confirmed input/prediction hashes before and after. It
completed `0:0` in 118 s; inference was not repeated. The failed wrapper status is
retained alongside that recovery. The reused-benchmark annotation SHA-256 is
`3cab357a0ebb9bf7292525ae6f33dd43ffef916226c0e6d8171c3f45dce71e28`.

The earlier TOML WTS baseline uses a different workflow, effective batch and adapter
scope; it is not an equivalent performance comparison. All requirements above are
validated within the stated model/method/distributed profile; excluded profiles
remain separate qualification work.

## Full WTS method comparison (2026-10-30)

![Full-epoch losses and WTS accuracy for all four adapter methods](assets/wts-adapter-method-comparison.png)

[Machine-readable comparison and artifact provenance](assets/wts-adapter-method-comparison.json)
preserves the measured revisions, job outcomes, numerical checks and evidence hashes.
This extends the ordinary-LoRA baseline above with separate full rsLoRA, DoRA
and PiSSA runs. All new training and evaluation jobs completed `0:0`, without
training restarts or evaluation reconciliation.

Each method consumes the same 5,555 unique training rows in the same order,
with 1,111 applied updates, seed 42, rank 8, alpha 16, dropout zero, the same
component learning rates, media preprocessing and 81 targets. Training uses one
A100 80 GB GPU, BF16 base storage, FP32 adapters and full activation checkpointing;
merged-model evaluation uses two A100 80 GB GPUs and FP32 inference. rsLoRA's
effective scale is `16/sqrt(8) = 5.656854`; the other low-rank branches use
`16/8 = 2`. This is one fixed recipe and seed, with no method-specific tuning.
All initial validation losses equal **0.6591825619353532** exactly.

| Method | Mean training loss | Final validation loss | Full WTS accuracy (2,676) | Scene-disjoint accuracy (576) |
|---|---:|---:|---:|---:|
| LoRA | 0.18750258 | 0.14688044 | 2467/2676 (92.1898%) | 533/576 (92.5347%) |
| rsLoRA | 0.16758934 | 0.15466100 | 2474/2676 (92.4514%) | 533/576 (92.5347%) |
| DoRA | 0.18554666 | 0.14597262 | 2466/2676 (92.1525%) | 533/576 (92.5347%) |
| PiSSA | 0.15602595 | 0.17113492 | 2462/2676 (92.0030%) | 534/576 (92.7083%) |

Every loss and gradient norm is finite, every optimizer clock matches its metric
step, and no update was skipped. All four methods produced zero invalid answers.
The full evaluation split includes 44 rows sharing four training-scene IDs;
the reused benchmark is scene-disjoint. The differences are small: rsLoRA gains
seven full-set answers over LoRA, DoRA loses one, and PiSSA loses five while
gaining one reused-benchmark answer. These single-seed observations do not
establish a generally superior method. PiSSA's lower training loss is accompanied
by higher validation loss under this recipe.

The completed new checkpoints verify all 750 base tensors against their original
BF16 values. Every rsLoRA/PiSSA trainable tensor changed (162 each). All 243 DoRA
trainable tensors changed, including all 81 magnitude tensors. All 162 PiSSA
initial-factor buffers remain unchanged from the preserved step-111 checkpoint.
Portable reload in a fresh process reproduces controlled text, image and video
outputs exactly for each new method. All 750 serialized merged-model tensors
are checked exactly for each export.

| Method | FP32 merged maximum / mean logit error | FP32 argmax match | Final-update replay maximum parameter error |
|---|---:|---:|---:|
| rsLoRA | 0.0002398714 / 0.0000137575 | 32/32 | 2.274e-13 |
| DoRA | 0.0001764297 / 0.0000131782 | 32/32 | 0 |
| PiSSA | 0.0001778603 / 0.0000132460 | 32/32 | 7.276e-12 |

The FP32 comparison promotes the trained unmerged model to FP32 before comparing
it with the merged model. The existing bounds remain `atol=5e-4, rtol=2e-4`,
mean error at most `5e-5`, with exact argmax agreement. BF16-to-FP32 compute drift
is recorded separately: LoRA and rsLoRA each change one of the 32 probe argmaxes;
DoRA and PiSSA change none. This is not a claim of BF16/FP32 generation equivalence.
BF16 merged export remains outside the qualified profile.

A separate read-only job restores checkpoint 1,110 and replays update 1,111
through the existing SFT builder and `train_step`, then loads the uninterrupted
checkpoint 1,111 for comparison. All three losses match exactly. Adapter and
optimizer tensors meet the existing continuation tolerance (`atol=1e-7,
rtol=1e-5`); optimizer groups, scheduler, RNG, data cursor and update clocks match
exactly. Frozen weights and PiSSA initial factors remain unchanged. The harness
rejects any method initialization during either restore, so PiSSA decomposition
cannot silently repeat. The original checkpoints are mounted read-only.

| Method | Training job | Export/reload/evaluation job | Training GPU-h | Export/evaluation GPU-h | Sampled peak training GiB | Portable MB |
|---|---|---|---:|---:|---:|---:|
| rsLoRA | `37687160` | `37697603` | 2.9617 | 0.8844 | 19.960 | 17.846 |
| DoRA | `37687163` | `37698189` | 3.2331 | 0.9344 | 19.964 | 18.745 |
| PiSSA | `37687166` | `37697555` | 2.9206 | 0.8856 | 20.104 | 35.642 |

Continuation job `37698213` completed `0:0` in 785 seconds (0.2181 GPU-h shared
across the three methods). Memory is sampled every ten seconds; it is not an
allocator-level peak. Timings include each allocation's setup and verification
work, and are observations from different cluster nodes. All merged models are
about 35.08 GB. Training and continuation use byte-attested revision `52a92d9`;
export/reload/evaluation use `480be5a`. The wheel/container hashes are listed above
and retained in every result. No new hyperparameter search was run.

## Subsequent family extension

The original qualification above covered dense native Qwen3-VL / Cosmos3 only.
The subsequent branch adds architecture-owned component bindings in
`model/adapter_families.py`, an HF lifecycle, and native physical-action SFT
builders. It reuses the adapter mathematics, trainer and checkpoint/export
transactions described here. Public Alpamayo R1 SFT uses its pinned native model
source and does not require the separate AlpaGym runtime.

Consult [the family requirement and qualification table](adapter-family-progress.md)
for completed real-checkpoint tests, pending profiles and the GR00T metadata
access blocker. The Cosmos3 losses and WTS scores above are not results for those
other checkpoints. The closed PR remains closed.

The family-extension media audit later found that the existing decoded-video
evaluator lost source timestamps and resized frames differently from SFT. The
historical WTS scores in this report retain their original evaluator provenance;
they must not be compared directly with new scores from the corrected
source-metadata evaluator. See [family progress](adapter-family-progress.md).
