# Posttrain adapter implementation qualification

Branch `feat/posttrain-vlm-adapters`; [PR #342](https://github.com/NVIDIA/cosmos-framework/pull/342).
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
| Installed package provenance | Wheel/source comparison and runtime byte attestation of all 1,838 package files | Implemented and validated through final runtime revision `480be5a` |
| Full WTS epoch and held-out evaluation | Existing SFT/evaluation entrypoints; exact 5,555-row cursor, 2,676-row validation, reserved 576-row score | Implemented and validated: training loss 0.18750, validation 0.65918 → 0.14688; MCQ 92.19%, reserved subset 92.53%, zero invalid answers; operational recoveries recorded below |
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
| Full WTS export/evaluation | `480be5a` | `37652713`: export, fresh reload and all 2,676 predictions completed; wrapper exited 1 because the reserved-test JSON was not mounted. CPU reconciliation `37653787` completed `0:0` with the unchanged verifier and input/prediction hashes |

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
accuracy, invalid answers and generated-token NLL separately. A reserved
576-row scene-disjoint subset is scored without selecting hyperparameters on it.
The full evaluation split has no row/video overlap with training, but 44 rows
share four scene IDs; report both scores. Resource and artifact measurements
are retained with each installed revision. No new AutoML search was started.

## Full WTS result

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
| Reserved scene-disjoint MCQ accuracy | 533 / 576 (92.5347%) |
| Invalid answers, full / reserved | 0 / 0 |
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
wrapper returned `1:0` when the reserved-test annotation file was missing from the
container mounts. The file existed on shared storage. After correcting the mount,
CPU-only job `37653787` ran the original verifier unchanged, reconciled every gold
and prediction ID, and confirmed input/prediction hashes before and after. It
completed `0:0` in 118 s; inference was not repeated. The failed wrapper status is
retained alongside that recovery. The reserved-test annotation SHA-256 is
`3cab357a0ebb9bf7292525ae6f33dd43ffef916226c0e6d8171c3f45dce71e28`.

The earlier TOML WTS baseline uses a different workflow, effective batch and adapter
scope; it is not an equivalent performance comparison. All requirements above are
validated within the stated model/method/distributed profile; excluded profiles
remain separate qualification work.
