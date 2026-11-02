# Adapter implementation: final qualification review

This review starts from clean `feat/posttrain-family-adapters` at
`06f59f89fb64f55216bec271bb581c0878a0b4e4`, against audited main
`86e1ee52f276c11d6cc66dfd17da2795dfbe937b`. The final production correction is
`44913e6f93e0b4fbc55e005c5ee933057b1169c2`; `69d565b` also makes the reload
qualification use the same production configuration helper. Later changes are
review documentation and evidence packaging. PR #342 stays closed. No branch is pushed and no new PR
is published.

**47 of the 48 fixed checkpoint/method profiles pass every gate.** The remaining
profile, native Qwen3-VL-30B-A3B PiSSA, passes training, continuation and portable
reload but fails merged-export equivalence. Its public merged export is refused.
This refusal is not completion of that requirement. The 182 passing correctness
executions include repeated world/activation-checkpointing/host/Slurm profiles;
they are not 182 independent model/method combinations. Four canonical Slurm and
four additional host full-WTS epochs are separate from that count. The four old
GR00T host placeholders were superseded before execution by the recorded Slurm
qualification; the host MoE PiSSA run completed portable checks but failed
merge. Neither adds a fully passing execution.

## Requirement-to-code-and-test matrix

Implementation paths in this table are under `cosmos_framework/posttrain/` and
test files are under its `tests/`, except the explicitly named native LoRA utility. “Validated” means the exact profiles
listed below, plus the stated small fixtures; it does not qualify untested
checkpoints, attention kernels or distributed modes.

| Requirement | Implementation and evidence | Status |
| --- | --- | --- |
| Explicit component/layer selection | `model/adapter_families.py::AdapterFamily.locate`, `model/adapters.py::_selected`, `resolve_targets`; `test_adapters.py::test_resolution_refuses_before_mutation`, family/native tests; saved manifests in the profile archive | Implemented and validated. Every selector must match; wrong components/types, overlaps with different settings and missing layer indices fail before injection |
| Independent settings and extra trained modules | `config/adapters.py`, per-target `PosttrainAdapterLinear`, `optimizer_groups`; `test_independent_components_and_optimizer_membership`, `test_promoted_merge_control_bf16_independent_methods_and_extra_norm` | Implemented and validated in small mixed-method/dropout/norm fixtures. Real profiles use one method per run with distinct component LR multipliers; arbitrary mixed real-model settings are not GPU-qualified |
| Parameter ownership and optimizer membership | `install_adapters`, `assert_trainability`, `entrypoints/sft.py::build_trainer`; shared-module/weight tests and per-rank qualification inventories | Implemented and validated. All selected aliases must agree; optimizer membership equals intended trainables, without duplicates. Cross-FSDP-unit ties and custom tied-target stock-HF merge are refused |
| Ordinary LoRA / rsLoRA | Existing `cosmos_framework/utils/generator/lora.py::LoraInjectedLinear`; FP32 adapter accumulation in `PosttrainAdapterLinear`; initialization/update/export tests | Implemented and validated on all 12 checkpoints. Scaling is alpha/r or alpha/sqrt(r); default A is Kaiming, B zero. Multiple-step tests allow A's first gradient to be zero |
| DoRA | Row magnitudes in forward, trainable spec, artifact state and pure merged-state conversion; `test_dora_forward_and_gradients_match_peft_reference` | Implemented and validated on all 12 checkpoints, including magnitude updates, resume and merge. One adapter merged into its base; adapter composition/stacking is unsupported |
| PiSSA initialization and artifacts | `initialize_method`, immutable `initial_A/initial_B`, `merged_state_weight`; `test_pissa_exact_principal_components_and_residual` and all-method continuation/reload tests | Implemented and validated for training/portable artifacts on all 12 checkpoints; merged export validated on 11. Original base W0 stays unchanged; effective weight is W0 + sBA − sB0A0. Initial decomposition is saved and never repeated during resume/warm start |
| Loaded base, meta materialization and freezing | `HFModel.setup`, `Qwen3VLBuildHooks.after_load_before_parallelize`, physical builders; native meta-builder and hook-order regressions | Implemented and validated. Load/materialize/tie base, install/init adapters, apply trainability, apply AC/FSDP, build optimizer from final parameters, then restore checkpoint state. Adapter-disabled construction order remains intact |
| Vision/projector gradients and missing modalities | Native ordered vision path; `Pi05Model.training_condition`; Alpamayo/GR00T SFT policies; real image/video fixtures and per-rank training checks | Implemented and validated. No reused detached vision features. Native VLM ranks with different modality presence pass. Alpamayo supports vision-only AC because generic language AC drops the differentiable prefix cache |
| Warm start and portable reload | `checkpoint/adapters.py::{adapter_configuration,load_adapter_artifact}`; fresh-process reload, wrong-identity/incomplete-artifact tests | Implemented and validated. Final review fixes tied-normalization owner reconstruction without changing old artifact identities; four new shared-owner/invalid-owner regressions pass |
| Exact training resume | Existing checkpoint transaction/lifecycle, strict objective/adapter identity, `LLMTrainer` state restore; native and physical next-update tests | Implemented and validated. Restore optimizer, scheduler, RNG, clock and data cursor; portable export is not a resumable checkpoint. Fixed 48 profiles reproduce controlled next-update parameters exactly; full Slurm WTS rsLoRA differs by at most 7.28e-12 with zero loss error |
| Portable artifacts | `checkpoint/adapters.py`, commit-last publication and checksums | Implemented and validated. Immutable base, resolved targets, method/extra state, processor/template/media, versions and PiSSA factors are checked. Training precision is preserved; all 48 fixed profiles have exact portable controlled outputs |
| Merged model artifacts | `checkpoint/tools/adapters.py`, existing HF materializer and family-owned physical exporters | Implemented and validated for 47 fixed profiles; **MoE PiSSA acceptance incomplete**. Merge is non-mutating, identity-bound, FP32 and independently reloaded. Frozen native embedding/head ties are preserved |
| Recipes, runtime and regression behavior | `configs/sft_*_adapters.py`, preparation scripts, pinned `docker/posttrain/flow-environment`, adapter guides; installed-wheel and adapter-disabled suites | Implemented and validated within the profiles below. No new training loop or generic causal-LM/PEFT adapter wrapper |

## Exact qualification scope

Every row below represents four methods: LoRA, rsLoRA, DoRA and PiSSA. The fixed
comparison uses **two ranks, pure FSDP2, SDPA, AdamW, rank 8, alpha 16, dropout
zero, FP32 adapter storage/accumulation/reduction**, full AC except Alpamayo's
vision AC. All rows have portable artifacts, complete training checkpoints and
FP32 merged artifacts, except the explicitly blocked MoE PiSSA merge. Physical
merged artifacts reload through their native loaders, not a generic HF loader.

| Exact checkpoint (revision prefix; full immutable revision in CSV/archive) | Actual selected components | Base storage |
| --- | --- | --- |
| Qwen/Qwen3-0.6B `c1899de289a0` | 56 language Q/V projections | BF16 |
| Qwen/Qwen3-VL-2B-Instruct `89644892e4d8` | 56 language Q/V, 8 merger/deepstack projections, vision block 23 QKV | BF16 |
| Qwen/Qwen2.5-0.5B-Instruct `7ae557604adf` | 48 language Q/V | BF16 |
| HuggingFaceTB/SmolLM2-135M-Instruct `12fd25f77366` | 60 language Q/V; Llama architecture, not a Meta Llama checkpoint | BF16 |
| mistralai/Mistral-7B-Instruct-v0.3 `c170c708c41d` | 64 language Q/V | BF16 |
| unsloth/gemma-2b-it `5ae754dee0b5` | 36 language Q/V; no Unsloth runtime | BF16 |
| unsloth/gemma-2-2b-it `457f2e15bf55` | 52 language Q/V; no Unsloth runtime | BF16 |
| Qwen/Qwen3.5-0.8B `2fc06364715b` | 48 language MLP up/down projections, 2 merger projections, vision block 11 QKV | BF16 |
| Qwen/Qwen3-VL-30B-A3B-Instruct `9c4b90e1e4ba` | 96 language attention Q/V, 8 merger/deepstack projections, vision block 26 QKV; no fused expert targets | BF16 |
| RLinf/RLinf-Pi05-LIBERO-SFT `45ccfcc4e286` | 36 language K/V, 1 projector, vision block 26 Q/V, 38 action projections | 691 BF16 and 130 FP32 base tensors; exact per-parameter map in artifact |
| nvidia/Alpamayo-R1-10B `dd4a24cacefc` | 72 language K/V, 8 merger/deepstack projections, vision block 26 QKV, 76 action projections | BF16; native FP32 Fourier/action boundaries preserved |
| nvidia/GR00T-N1.7-LIBERO `2ea293aa20ba` (`libero_spatial`) | 32 language Q/V, 8 merger/deepstack projections, vision block 23 QKV, 64 action projections | FP32 master parameters, native BF16 autocast |

The [48-row results](assets/family-adapter-method-results.csv) retain each
source/wheel/container or host-environment identity and the actual losses.
[Saved profile contracts](assets/adapter-profile-contracts.json.gz) contain all
concrete module names, types, shapes, ranks/scales, ownership, trainable counts,
processor/template/preprocessing settings and data hashes. Selectors in recipes
are not used as a substitute for these resolved maps.

The additional qualified profiles are those recorded in
[the execution inventory](assets/family-adapter-qualification.json.gz): one/two
ranks × none/full AC for Qwen3, dense native Qwen3-VL, Qwen2.5, SmolLM2,
Qwen3.5, π0.5 and GR00T; one/two ranks with full AC for Mistral; two ranks/full
AC for both Gemmas and MoE; one/two ranks × none/vision AC for Alpamayo.
These are bounded to recorded checkpoints. Eager attention has fixture coverage,
not these GPU qualifications. No HSDP, TP/CP/PP/EP, larger worlds, multi-node,
generic compile, fused expert/category-specific tensors, stacked adapters,
four-bit training, additional model families or HPO is claimed.

## MoE PiSSA: residual math versus routing sensitivity

The final diagnostic reuses the original failed Slurm probe files and portable
weights byte-for-byte, with source `239c3e7` and the pinned host environment.
It checks the existing serialized merge against all 882 expected tensors, then
compares native portable and privately merged copies. Saved training/export
artifacts stay read-only; neither router decisions nor tolerances are overridden.

Across the 105 adapter targets, FP64 calculations give:

| Independent calculation | Maximum absolute error |
| --- | ---: |
| `(W0 − sB0A0) + sB0A0` versus W0 | 2.78e-17 |
| Residual-base versus original-base rank-2r conversion | 1.11e-16 |
| FP64 factorized versus dense linear outputs, same captured input | 1.21e-13 |
| Production FP32 effective weights versus FP64 reference | 4.63e-8 |
| Production factorized versus dense FP32 outputs, same captured input | 1.38e-5 |

The first observed adapter output differences are already small FP32 differences
in a visual deepstack merger (maximum 8.58e-6). Expert sets remain equal through
language layer 36. At layer 37/token 7, incoming hidden-state error reaches
9.06e-6 and raw router error 6.68e-6. The portable 8th/9th probability margin is
7.45e-9; expert 5 is replaced by expert 68. Recomputing each captured router
input in FP64 retains the opposite order: raw score(5)−score(68) is +2.57e-7
for portable and −9.07e-7 for merged. Thus this is not merely FP32 softmax tie
rounding. The MoE block output error jumps from 1.19e-6 at layer 36 to 0.0970 at
layer 37, and downstream final-logit maximum/mean errors are
`0.0295000 / 0.000388840`.

All 64 greedy tokens agree, but the existing contract still fails:
`atol=5e-4`, `rtol=2e-4`, mean error ≤ `5e-5`, plus greedy agreement.
FP64 local calculations are references at captured FP32 inputs, not a claim of
full-model FP64 inference. They reveal no residual-base/conversion defect in this
probe; ordinary arithmetic-order differences are amplified at an ill-conditioned
hard top-k boundary. Earlier stock-HF **portable** inference already diverges
similarly, before serialization. A native-only export does not fix the failure.
Changing routing, widening tolerances or replacing the independent portable
oracle with dense execution would change the contract. No such change is made.
See [the diagnostic summary](assets/moe-pissa-fp64-summary.json).

## Precision and artifact contracts

Portable reload preserves the recorded training computation and base storage.
Merged equivalence compares **FP32 portable to FP32 merged**, retaining the
already-rounded training-base values. It does not compare against untouched
source FP32 tensors or promise BF16-to-FP32 behavioral equivalence. The largest
passing fixed-profile merged maximum error is `2.289e-4` (Gemma DoRA); each
profile also passes its elementwise/mean gate. Physical outputs use the same
absolute/relative/mean bounds without a language-token criterion.

Cross-precision drift is reported separately: the MoE LoRA example changes 6/64
greedy positions when promoted from BF16 to FP32, despite passing FP32 merge
equivalence. Full Slurm WTS LoRA/DoRA change one of 32 controlled positions across
precision; each FP32 portable/merged pair agrees on all 32. No BF16 dense-merge
support is inferred. DoRA magnitudes and PiSSA initial factors are mandatory
artifact state. Exact resume does not tolerate missing adapter keys or rerun
decomposition. Portable artifacts cannot restore optimizer/data/RNG state.

## Validation provenance and final patch coverage

| Executed revision | Evidence and material differences to final code |
| --- | --- |
| `3a9552b` | Canonical Qwen3/native-VL subset runs and four full Slurm WTS epochs/recovery. Later shared forward changes preserve family-owned autocast; native BF16 probes reload exactly under later code. Text packing, physical bindings and Qwen3.5 source exclusions do not relabel these old results |
| `2f8d33f`, `db7c2ab`, `076cfbb` | Qwen3.5 source-exclusion, text-template and Gemma-constant fixes respectively; actual per-profile revisions remain in the CSV. Later 400-test installed suite covers these shared paths |
| `14c9e09`, `08e5ff5` | Corrected Alpamayo FP32 construction and GR00T contiguous Conv3d storage; corresponding real physical training/resume/reloads pass. No physical forward changes after `08e5ff5` |
| `239c3e7` | Native MoE FP32 reference and device-local mask handling. Three complete methods; PiSSA portable passes and merge fails. The new FP64 diagnostic deliberately uses this historical failing implementation |
| `2a8b48d` | Corrected source frame indices/fps and processor-owned video resizing. Four Slurm plus four host WTS export/evaluations; exact two-clip media/prompt preflight. Matched unadapted evaluation uses this same corrected runtime |
| `4dd3826` | Only subsequent production change before the handoff is the explicit MoE/PiSSA merge refusal: 400 installed tests (4 skips), 13 media tests, and adapter-disabled two-rank FSDP regression passed |
| `44913e6` | Final review changes only `checkpoint/adapters.py::adapter_configuration` and tests. Preserve extra-normalization alias owner order from existing manifests. 97 source + 97 installed adapter/export tests pass; 50 physical tests pass, one CUDA-only skip. These counts overlap and are not summed |
| `69d565b` | Qualification-only cleanup: `tests/reload_adapters.py::portable_policy` now reuses production configuration reconstruction, including tied extra-module aliases. Settings are unchanged for all 48 saved profiles. 97 source + 97 installed tests and eight fresh-process GPU reloads pass again; no production or physical forward change |

See [final installed validation](assets/final-installed-validation.json) for individual
report hashes and errors. The final wheel SHA-256 is
`e487b90c607e5344b71786a6fc5b833c56a2bc0e3f5970fd56fb82b670d21062`.
All 1,880 packaged source files match clean Git archive `69d565b`. Its copied,
isolated host environment is attested separately from a container; no fictitious
image identity is assigned. The review package records every revision's changed
production/test files, source-build records, actual installed environment, logs
and checksums. Eight additional fresh-process GPU checks at this wheel pass using the completed
host full-epoch checkpoints: all four portable reloads are exact, and all four
existing FP32 merged files pass (maximum errors 8.96e-5 / 7.63e-5 / 7.53e-5 /
1.10e-4 for LoRA / rsLoRA / DoRA / PiSSA). These are additional reload executions,
not new training profiles. The preceding eight `44913e6` reloads remain
separately preserved, rather than counted as additional model coverage.

The unchanged training/optimizer/checkpoint implementation does not require
another epoch or hyperparameter search for this alias-order correction. Existing
experiments, failures, canceled-before-execution launchers and old review files
are preserved. Newly queued baseline/diagnostic jobs were moved while still
pending to idle local A100s; unrelated jobs were not touched.

## WTS evaluation and interpretation

A new unadapted Qwen3-VL-2B evaluation completed all 2,676 rows in 946 seconds
(two A100 80 GB GPUs). All 625 unadapted tensors are verified as the exact original
base rounded to BF16 and promoted to FP32, matching the adapted export contract.
Base/processor identity, environment, preprocessing, decoding and every gold row
are checked equal before pairing. The baseline has no adapters and no training.

This table pairs the **completed host repeats** trained at `076cfbb` with that
baseline. All five evaluations use source `2a8b48d`, wheel
`832cc43cc7d933981b9308ab27595471aaebdd322714ee97db09edb5a5e761d9`, and host
environment `ee8bd063a246931f33b8497eebd50ff9f45d20c0c0b0626e4da49ce867a61667`.
The separate canonical Slurm scores and their `3a9552b` training provenance remain
in [the full-epoch report](adapter-family-progress.md#full-epoch-native-qwen3-vl-wts-results).

| Method | Mean training loss | Final validation loss | Full accuracy (2,676) | Reused scene-disjoint benchmark (576) | Full paired wrong→right / right→wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unadapted | — | — | 1269/2676 (47.42%) | 271/576 (47.05%) | — |
| LoRA | 0.227146 | 0.170965 | 2465/2676 (92.12%) | 528/576 (91.67%) | 1227 / 31 |
| rsLoRA | 0.192750 | 0.179041 | 2455/2676 (91.74%) | 530/576 (92.01%) | 1222 / 36 |
| DoRA | 0.224450 | 0.171388 | 2470/2676 (92.30%) | 528/576 (91.67%) | 1228 / 27 |
| PiSSA | 0.188904 | 0.184454 | 2467/2676 (92.19%) | 530/576 (92.01%) | 1227 / 29 |

All five evaluations produce zero invalid answers. Full-set paired gains are
44.69 / 44.32 / 44.88 / 44.77 percentage points for LoRA / rsLoRA / DoRA / PiSSA.
On the reused benchmark, wrong→right / right→wrong counts are 261/4, 262/3,
261/4 and 261/2. These are measured gains for this one-seed recipe and reused
benchmark, not evidence of a universally best method or untouched-test
generalization. Training/teacher-forced validation losses retain BF16 training
computation (common pre-training validation loss 0.843903); generated-answer
accuracy uses FP32 and generated-token NLL is a separate metric.

[Paired results and exact prediction hashes](assets/wts-matched-paired-results.json)
record the common four-frame TorchCodec CPU decoding, original timestamps,
50,176–81,920 pixels/frame, SDPA, length 16,384, greedy generation, 32 new tokens,
seed 42, batch size one and the identical answer-letter system prompt.

The 576-row subset is a **reused scene-disjoint benchmark**. It has participated
in earlier development and AutoML work; it is not an untouched final test. The
2,676-row full evaluation contains 44 rows sharing four training-scene IDs.
Only comparisons with identical Qwen3-VL-2B base, corrected media processing,
FP32 computation, greedy decoding and row IDs are paired. Historical Cosmos3
scores use a different model and preprocessing and are not an accuracy baseline.

Physical runs remain correctness qualifications: eight training rows plus one
controlled global update, staged LIBERO/KITTI validation, real images/actions,
adapter changes, recovery and reload. They do not establish simulator success,
driving quality/safety or improvement on a physical downstream task. No new
physical performance claim or additional family expansion is made.

## Reproduction and review artifacts

The local review package contains the cumulative binary patch from audited main,
a patch-application/tree-equality check, dependent commit list, PR description,
source/revision audit, immutable report copies/hashes and diagnostic/baseline
launchers. The copied original WTS prediction shards and a CPU-only verifier
reproduce all paired metrics without another evaluation job. It preserves the original handoff and generated vector figures.
The repository retains compact results, lossless raw evidence archives, runnable
recipes, pinned runtime construction and focused tests; no checkpoint weights,
datasets or third-party checkout is vendored.

Focused correctness (use the pinned HF and separate physical runtime):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$SFT_PYTHON" -m pytest \
  -o addopts='' --confcutdir=cosmos_framework/posttrain/tests -q \
  cosmos_framework/posttrain/tests/test_adapters.py \
  cosmos_framework/posttrain/tests/test_adapters_native.py \
  cosmos_framework/posttrain/tests/test_adapter_families.py \
  cosmos_framework/posttrain/tests/test_adapter_moe_export.py \
  cosmos_framework/posttrain/tests/test_adapter_qwen3_5_export.py
```

A new small native dense-VLM qualification, after installing/attesting the wheel
and staging the pinned model plus text/PNG/MP4/WTS fixtures (use a new output):

```bash
export SFT_MODEL_PATH=/models/Qwen3-VL-2B-Instruct/89644892e4d85e24eaac8bacfd4f463576704203
export SFT_DATA_ROOT=/data/adapter-fixtures
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$SFT_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m cosmos_framework.posttrain.tests.qualify_adapters \
  --recipe cosmos_framework/posttrain/configs/sft_family_adapters.py --backend i4 \
  --method pissa --activation-checkpointing full --out "$NEW_RUN" \
  --wheel "$WHEEL" --build-record "$BUILD_RECORD"
"$SFT_PYTHON" -m cosmos_framework.posttrain.tests.reload_adapters \
  --run "$NEW_RUN" --base "$SFT_MODEL_PATH" --wheel "$WHEEL" --build-record "$BUILD_RECORD"
"$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --adapter "$NEW_RUN/portable" --base "$SFT_MODEL_PATH" --merged --out "$NEW_RUN/merged"
"$SFT_PYTHON" -m cosmos_framework.posttrain.tests.reload_adapters \
  --run "$NEW_RUN" --base "$SFT_MODEL_PATH" --merged "$NEW_RUN/merged" \
  --wheel "$WHEEL" --build-record "$BUILD_RECORD"
```

The recipe's explicit seed is 42. Use `--method lora|rslora|dora|pissa` for the
four profiles. Physical recipes/loaders and exact source staging are documented
in [flow adapters](guides/flow-adapters.md). Full WTS commands, corrected
evaluation configuration, and read-only reproduction of the original MoE
diagnostic are in the local package; successful training is not repeated as
part of this final review.
