# Adapter family extension: implementation and evidence

Work branch: `feat/posttrain-family-adapters`. This extends the original Cosmos3
implementation; [PR #342](https://github.com/NVIDIA/cosmos-framework/pull/342)
remains closed and unmerged. The audited starting revision was
`53b2382e00e8487270a2d09fcf2a13d9597340ac`. Main
`86e1ee52f276c11d6cc66dfd17da2795dfbe937b` was inspected and merged; its
intervening changes were skills, documentation and CI.

This report distinguishes implemented bindings from completed real-checkpoint
qualification. The machine-readable [case evidence](assets/family-adapter-qualification.json)
records immutable model revisions, actual source/wheel/environment identities,
losses, trainable counts, numerical errors and report hashes. Historical Cosmos3
full-WTS results remain in [adapters-progress.md](adapters-progress.md); they are
not results for another checkpoint. Four-bit training remains separate.

There are **181 completed real-model qualification cases across 12 checkpoints**.
This includes repeated profiles in independently attested host and Slurm runtimes;
it counts only cases with completed training, continuation and both reload gates;
some parent matrices are still running. Repeated host/Slurm profiles remain
separate executions rather than distinct model/method combinations. The compact
[method results](assets/family-adapter-method-results.csv) select one fixed
world-size/AC profile per checkpoint: 47 of 48 method profiles pass every gate;
PiSSA/MoE training and portable reload pass but its merged export is blocked.
The CSV includes that failed merge and its observed losses rather than hiding it.

## Requirement status

Paths below are relative to `cosmos_framework/posttrain/` unless stated otherwise.

| Requirement | Code and focused tests | Status |
| --- | --- | --- |
| Shared LoRA/rsLoRA, DoRA and PiSSA | `model/adapters.py::PosttrainAdapterLinear`, existing native `utils/generator/lora.py`; `test_adapters.py`, `test_adapter_autocast.py` | Implemented and validated for the completed cases below. Native LoRA mathematics reused; DoRA magnitudes and PiSSA initial factors retained |
| Component/layer selection and independent settings | `model/adapter_families.py`, `resolve_targets`; `test_adapter_families.py`, `test_adapters_native.py` | Implemented and validated. Every selector must match its component; alias conflicts and unsupported target types fail before injection |
| Placement, trainability and optimizer ownership | `install_adapters`, `assert_trainability`, `optimizer_groups`, SFT `build_trainer` | Implemented and validated. Logical trainable inventory equals optimizer membership, without duplicates or frozen parameters |
| HF construction, tying, freezing and FSDP lifecycle | `model/hf.py::HFModel.setup`; `test_model_hf.py`, `test_text_sft_adapters.py` | Implemented and validated. Load/materialize base, install adapters, apply distributed transformation, then construct optimizer; adapter-disabled branch retained |
| Native VLM/MoE lifecycle | `model/qwen3_vl.py::Qwen3VLBuildHooks.after_load_before_parallelize`; `test_adapters_native.py`, `test_adapter_moe_export.py` | Dense VLM implemented and validated; MoE LoRA/rsLoRA/DoRA training, continuation and both reload gates passed. PiSSA training/continuation/portable reload pass; merged export is blocked by measured hard-routing sensitivity. FP32 conversion changes some BF16 predictions, as quantified below |
| Text SFT templates and masks | `data/packer/text_sft.py`, existing source-traced SFT masker; `test_data_packer_text_families.py`, `test_text_sft_adapters.py` | Implemented and validated for explicit ChatML, Llama3, Mistral and Gemma protocols. Unsupported or reordered renderings fail |
| Vision/projector/action gradient paths | Native Qwen ordered-vision forward, `model/pi05/modeling.py::training_condition`, Alpamayo and GR00T native SFT policies; family fixtures | Implemented and validated in completed pretrained cases, including all four GR00T methods across language/projector/vision/action. Differing rank modalities tested on native VLM |
| Fresh initialization, warm start, exact resume | Shared adapter preparation and checkpoint lifecycle; qualification next-update comparison | Implemented and validated in completed cases. Resume restores optimizer/scheduler/data/RNG clock and skips method initialization/decomposition |
| Portable artifact and merged export | `checkpoint/adapters.py`, `checkpoint/tools/adapters.py`, existing materializer; family-owned physical exporters | Implemented and validated in completed cases. Separate artifact kinds, exact base/processor/media identities, strict inventory, non-mutating FP32 merge |
| π0.5 offline SFT | `model/pi05/{sft,adapter_export}.py`, existing flow objective and `LLMTrainer`; `test_pi05_adapters.py`, `test_flow_sft.py` | Implemented and validated: all four methods, real LIBERO data, 1/2 GPUs × none/full AC, continuation and both reload types |
| Public Alpamayo R1 SFT | `model/alpamayo/{sft,policy,adapter_export}.py`; `test_alpamayo_adapters.py` | Implemented and validated: all four methods, real KITTI data, 1/2 GPUs × none/vision AC, continuation and both reload types; no AlpaGym runtime required |
| GR00T full-backbone/action SFT | `model/gr00t/{sft,sft_policy,adapter_export}.py`; `test_gr00t_adapters.py` | Implemented and validated: all four methods, real 3.144B checkpoint/LIBERO data, 105 resolved targets, 1/2 GPUs × none/full AC, exact continuation, portable and FP32 merged reload. Corrected parameter layout and pre-fix artifact compatibility are validated |
| Installed artifact provenance | `tests/qualify_adapters.py::attest`, `installed_environment`; `test_qualification_environment.py` | Implemented and validated. Installed package bytes match the recorded wheel; host environment identity is distinct from container identity |
| Gemma FP32 embedding constant | `model/hf_adapter_buffers.py`; `test_hf_adapter_buffers.py` | Implemented and validated. Canonical nonpersistent constant preserves BF16 forwards and stock FP32 merge; legacy missing-contract artifacts fail explicitly |
| Native Qwen3-VL full WTS epoch/evaluation | Existing SFT/evaluator, method-specific full-profile gates, final checkpoint/recovery/export verification | Implemented orchestration; all four Slurm full-profile gates passed and full epochs/final validation are running; local LoRA/rsLoRA epochs and validation are complete. Corrected installed evaluator preflight passed; full scores and recovery still pending. Existing Cosmos3 scores are not reused |

## Completed real-checkpoint profiles

All four methods means ordinary LoRA, rsLoRA, DoRA and PiSSA. Every passing case
includes multiple optimizer updates, complete target/update and frozen-base checks,
controlled checkpoint continuation, fresh-process portable reload, and independent
FP32 merged-file reload. These are correctness subsets, not complete task benchmarks.

| Checkpoint | Qualified GPU / AC profile | All four methods |
| --- | --- | --- |
| Qwen3-0.6B | 1/2 GPU × none/full | Passed (16/16 cases) |
| Native Qwen3-VL-2B-Instruct | 1/2 GPU × none/full | Passed (16/16 cases) |
| Qwen2.5-0.5B-Instruct | 1/2 GPU × none/full | Passed (16 Slurm + 4 host cases) |
| SmolLM2-135M-Instruct (Llama architecture) | 1/2 GPU × none/full | Passed (16 Slurm + 4 host cases) |
| Mistral-7B-Instruct-v0.3 | 1/2 GPU / full | Passed (8 Slurm + 4 host cases) |
| Gemma-2B-IT | 2 GPU / full | Passed (4/4 cases) |
| Gemma2-2B-IT | 2 GPU / full | Passed (4/4 cases) |
| Qwen3.5-0.8B | 1/2 GPU × none/full | Passed (16 Slurm + 4 host cases) |
| π0.5 LIBERO SFT | 1/2 GPU × none/full | Passed (16 Slurm + 4 host cases) |
| Alpamayo-R1-10B | 1/2 GPU × none/vision | Passed (16 Slurm + 4 host cases) |
| Native Qwen3-VL-30B-A3B-Instruct | 2 GPU / full | LoRA/rsLoRA/DoRA passed; PiSSA merged export blocked by measured hard-routing sensitivity. FP32 equivalence is separate from BF16 predictions |
| GR00T-N1.7 LIBERO-Spatial | 1/2 GPU × none/full | Passed (16 corrected cases plus 8 earlier one-GPU cases) |

Text fixtures use eight real GSM8K training rows and eight held-out test rows.
Conversational VLM fixtures combine real WTS videos with controlled text, PNG and
MP4 examples, including differing modality presence across ranks. SmolLM2 verifies
the Llama architecture, not Meta Llama checkpoints. Gemma fixtures use pinned public
BF16 checkpoints from `unsloth/gemma-2b-it` and `unsloth/gemma-2-2b-it`; no Unsloth
runtime or quantization is used.

The staged π0.5/GR00T dataset contains an explicit 40/41-window training subset
and 55 validation windows from a separate LIBERO episode. Alpamayo's staged
dataset contains 14/15 training windows and 38 validation windows from a separate
KITTI drive. Each correctness case consumes eight training rows before its
checkpoint and one more global update for continuation (nine rows on one GPU,
ten on two GPUs); it evaluates the complete staged validation split. Omitted
row IDs are recorded. These are not full LIBERO/PhysicalAI training runs.
The physical loss is a masked mean over
action elements. Report both weighted training mean and last-batch loss because
native flow noise/time sampling makes a single batch unrepresentative. These
losses do not establish simulator success or driving performance.

All completed local continuation comparisons have zero maximum parameter error;
portable reload comparisons are exact. Alpamayo FP32 merged errors are at most
`2.7e-6`; π0.5 at most `8.9e-6`. Conversational FP32 merge uses elementwise
`atol=5e-4, rtol=2e-4`, mean error at most `5e-5`, and identical controlled argmaxes.
BF16 training-versus-FP32 deployment drift is measured separately; it is not
conflated with the merge oracle. Physical reload compares fixed-noise/time velocity
and loss. The JSON artifact contains method-specific losses and allocator peaks;
shared CPU/disk timings are not isolated throughput measurements. Older Slurm
launchers recorded the node and allocated GPU count but not the GPU model; their
reports make no GPU-model throughput claim. Local runs use A100 80 GB devices.

## Runtime and observed fixes

Conversational SFT uses Torch `2.10.0+cu128`, Transformers `5.8.1`, safetensors
`0.8.0`, and CPU TorchCodec `0.10.0`. Slurm uses Python 3.13; the local isolated
host environment uses Python 3.12.13. Physical SFT has its own pinned x86 runtime
at [`docker/posttrain/flow-environment`](../../docker/posttrain/flow-environment),
with Transformers `4.57.6` and pinned Isaac/Alpamayo source hashes. Existing RL
and serving environments are not modified.

| Installed source | Validation | Artifact |
| --- | --- | --- |
| `3a9552b` | 223 CPU tests; two completed Slurm matrices, 16 cases each | Qwen3 job `37701630`, native Qwen3-VL job `37701632`; wheel/container hashes in case evidence |
| `5eecf10` HF host | 386 passed, 1 skipped | Original text/Qwen3.5/MoE host cases; byte-attested wheel and environment |
| `2556b4c` physical host | 138 passed, 1 skipped | π0.5 logical-shard update checks |
| `14c9e09` physical host/container | 138 passed, 1 skipped | Alpamayo FP32 merged construction; image job `37709361` completed |
| `076cfbb` HF host | 407 passed, 6 skipped | Corrected Gemma/Gemma2 and new full-WTS wheel; 1,876 installed package files verified |
| `076cfbb` HF container | 396 passed, 1 skipped | Image job `37710535` completed; SHA `8e9e8cf5591c77a2a6f4a143e432f5234d924292d6be68d7d257ff9aca64eeca` |
| `076cfbb` evaluator host | Historical execution preflight only; media mismatch found later | Superseded before full scoring by `2a8b48d` |
| `239c3e7` HF host | 411 passed, 9 skipped; 4 installed GPU numerical/device tests passed | MoE LoRA/rsLoRA/DoRA complete on Slurm; PiSSA merge limitation measured separately |
| `8a8ead7` physical host | 143 passed, 4 skipped; real GR00T CPU preflight passed | Exact metadata staging and corrected 24-block vision recipe; 1,879 installed package files verified |
| `08e5ff5` physical container | 143 passed, 5 skipped; CUDA layout and legacy-portable compatibility gates passed | All 16 corrected GR00T method/world/AC cases completed |
| `2a8b48d` evaluator container | 400 passed, 4 skipped plus 13 media tests; two real clips have exact SFT pixels/timestamps/prompt IDs; two-GPU MCQ preflight passed | Job `37718708`; image SHA `f50fa1157fbd4377c86ae0bad9d332c0aa82e10cc3a900a7f0081c692e961083` |
| `4dd3826` export guard | 36 focused source tests passed | Real MoE PiSSA merge is explicitly refused; rejection is not counted as feature completion |

Counts overlap and must not be added. The general Torch 2.11 environment is a
separate unqualified profile. Every GPU result retains its actual revision; a
later documentation commit does not relabel an earlier run.

Real runs exposed and verified these fixes:

- Qwen3.5's public checkpoint includes unused MTP source tensors. Its export
  exception is bounded to that exact architecture/namespace and recorded in the
  materialization marker; unexpected tensors elsewhere still fail.
- FSDP can place only padded action-output rows on one rank. Qualification now
  requires each logical trainable tensor to update somewhere, while recording
  rank-local unchanged shards and checking each rank's optimizer/frozen state.
- Alpamayo's nested BF16 text config rounded stock FP32 merged weights during
  construction. The merged loader constructs FP32 before loading; a realistic
  nested-dtype regression failed before the fix. LoRA's original training is
  retained, with its corrected merge tested at `14c9e09`; later methods trained
  from that revision.
- Gemma's BF16 nonpersistent embedding scale was `45.25` instead of
  `45.2548339959`. Keeping the config-derived constant in FP32 preserves its BF16
  forward and reduces real-model merge mismatch from `0.00805` to `0.000210`.
  Corrected qualification is a fresh run; the failed attempt remains preserved.

## Pending execution and explicit gaps

GR00T-N1.7 LIBERO-Spatial weights, complete processor assets and real LIBERO data
are staged. Direct Cosmos-Reason2 metadata downloads returned HTTP 401, but the
public immutable Hub tree exposed its Git blob identities. All nine config,
tokenizer, template and media-processor files exactly match the public pinned
Qwen3-VL-2B snapshot. `scripts/prepare_gr00t_backbone.py` verifies those identities
before publication; it neither infers configuration nor substitutes model weights.
The staging tool rejects missing or changed files, and records both source
revisions and SHA-256 hashes. All four methods have now passed pretrained
training, continuation, portable and FP32 merged reload on one A100 80 GB.
LoRA/rsLoRA/PiSSA changed all 210 trainable tensors; DoRA changed all 315,
including magnitudes. All 105 targets span language, merger, vision and action.
Initial controlled output error and continuation error were zero. Peak allocated
GPU memory was 15.94–16.67 GiB without AC. All one/two-GPU profiles with AC off/full now pass
for every method. The two-GPU full-AC profile uses 11.88–11.90 GiB peak allocated
memory on rank zero; its maximum controlled merged error is `2.34e-5`.

The first two-GPU attempts at `8a8ead7` failed before training: Isaac's
channels-last Conv3d weight has non-contiguous storage, which FSDP2 rejects.
The SFT constructor now keeps that parameter contiguous while retaining the
native channels-last input hook. The actual small-model constructor regression
failed before the change and now passes with exact native output agreement;
the focused suite has 23 passes and one CUDA-only skip. The `08e5ff5` rebuilt
physical image passed 143 installed CPU tests with five skips. A new CUDA
output/input-gradient regression and reload of pre-fix portable artifacts both
pass exactly in every method job. Rebuilt real-model jobs `37717289`, `37717290`,
`37717292`, `37717294` completed all 16 one/two-GPU × none/full-AC cases.

Native MoE implementation targets ordinary two-dimensional attention/vision/merger projections;
fused three-dimensional expert tensors remain unsupported targets. Its existing
native grouped-mm training forward stays BF16. The FP32 export oracle promotes a
private portable copy, reuses `MoEExperts`' existing `native.naive` FP32 dispatch,
and compares the native reference with stock HF portable/merged execution on two
GPUs. It checks every serialized tensor without introducing another expert kernel
or training loop. BF16 stock-HF and native merged probes exceeded the fixed drift
bounds (maximum logit errors 4.75 and 6.609375); those profiles remain unsupported.
The initial portable reload verifier omitted the deterministic kernel setting used
by training. All 882 frozen checkpoint tensors matched the original BF16 base;
three full-model reload probes with matching settings then matched training
exactly. `tests/adapter_numerics.py::configure_qualification_numerics` now aligns
and records the controlled profile across training/reload, with a real CUDA
grouped-expert forward/backward regression. A separate real-model FP32 diagnostic
matched native and stock HF outputs (maximum `4.2915e-5`, mean `3.6162e-6`, all
64 greedy decisions equal). The installed LoRA serialized-merge gate now passes: all 882 output tensors
match the expected merge; maximum/mean FP32 logit errors are `4.9591e-5`/
`3.8509e-6`, with all 64 greedy decisions unchanged against the FP32 adapter.
The same FP32 model differs from BF16 training computation at 6/64 greedy
positions (maximum logit error `6.6499`, mean `0.34860`). This is a precision
conversion limitation, not portable-reload or FP32 merge equivalence. BF16
portable reload remains exact. Deployments requiring unchanged BF16 predictions
should use the portable model; FP32 task quality still needs separate evaluation.
Slurm rsLoRA and DoRA also pass. PiSSA's merge gate remains failed; see the measured limitation below.

Other bound Qwen2-VL/2.5-VL small fixtures do not qualify their public checkpoints'
legacy processor/namespaces; the production SFT packer still refuses them.
Unregistered architectures, HSDP, TP/CP/PP/EP, compilation, larger worlds,
multiple nodes, stacked adapters and quantized training are not qualified here.
Physical merged artifacts have model-owned FP32 loaders; they are not stock HF,
Omni/PPO or closed-loop deployment claims. No universal model-family support is
inferred from a component binding.

Slurm text/Qwen3.5 profiles below completed and are included above. Physical
method jobs failed before training because media symlinks resolved to an unmounted
canonical `/lustre/fs11` namespace. Fresh attempts add both verified namespace
mounts and require an actual container data/processor preflight before launch;
old inputs and failures are preserved. After the real MoE LoRA merge gate
passed, only the held, never-started old MoE attempts were canceled and replaced
with sealed per-method jobs. Table labels distinguish these states:

| Family | LoRA | rsLoRA | DoRA | PiSSA | Profiles per job |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5 — completed | 37707970 | 37707974 | 37707976 | 37707978 | 1/2 GPU × AC none/full |
| π0.5 — failed mount preflight | 37710088 | 37710094 | 37710096 | 37710463 | 1/2 GPU × AC none/full |
| Alpamayo — failed mount preflight | 37710102 | 37710106 | 37711261 | 37711674 | 1/2 GPU × AC none/vision |
| Native Qwen3-VL-MoE — first three complete, PiSSA merge failed | 37716232 | 37716235 | 37716239 | 37716247 | 2 GPU, AC full |
| Qwen2.5 — completed | 37708301 | 37708306 | 37708308 | 37708309 | 1/2 GPU × AC none/full |
| SmolLM2 — completed | 37708310 | 37708311 | 37708312 | 37708313 | 1/2 GPU × AC none/full |
| Mistral — completed | 37708398 | 37708399 | 37708400 | 37708401 | 1/2 GPU, AC full |
| π0.5 — corrected jobs completed | 37715425 | 37715427 | 37715429 | 37715430 | 1/2 GPU × AC none/full |
| Alpamayo — corrected jobs completed | 37715434 | 37715435 | 37715436 | 37715441 | 1/2 GPU × AC none/vision |
| GR00T — earlier one-GPU cases completed | 37715443 | 37715444 | 37715445 | 37715446 | 1 GPU × AC none/full |
| GR00T — corrected complete matrix | 37717289 | 37717290 | 37717292 | 37717294 | 1/2 GPU × AC none/full |

## New native Qwen3-VL full WTS verification

The local source copy verifies all 517 files, including 512 videos, against
SHA-256 and preserves all 5,555 training and 2,676 validation rows. Each method
must first pass its own full-resolution/gradient-accumulation qualification on
the installed `076cfbb` wheel before the existing SFT entrypoint trains exactly
1,111 updates (microbatch one, accumulation five). Every recorded row ID and data
cursor is reconciled; no implicit padding or dropped training rows is accepted.

After each complete epoch, independent read-only recovery replays update
1110→1111 and compares parameters, optimizer, scheduler, RNG, clock and cursor.
Final export checks the frozen base and PiSSA initial factors, verifies all
trainable tensors changed in the final update, then performs portable and FP32
merged reload. Only then does the existing two-GPU evaluator score all 2,676
validation examples and the reserved 576-example scene-disjoint subset. Generated
NLL is reported separately from teacher-forced validation loss. No new AutoML
search substitutes for these checks.

Slurm full-training jobs `37706319`, `37706324`, `37706326`, `37706327`
are running after passing their method-specific full-profile checks. Their corrected dependent evaluation jobs are `37718903`, `37718931`,
`37718943`, `37718945`. Unstarted earlier evaluation launchers were replaced after
removing an unbound environment-variable read; old sealed inputs and cancellation
receipts remain available. Local runs use true host-environment identities and
local attempt IDs, never fabricated Slurm/container identities.

Slurm final-update recovery jobs `37716221`, `37716223`, `37716231`, and
`37716233` depend on their respective successful full epochs. They mount
training artifacts read-only, attest the exact training image, reject any method
initialization/decomposition during resume, and compare update 1110→1111
against the uninterrupted final checkpoint. Recovery retains the exact training
image; corrected inference/export uses its separately attested evaluator image.

The Slurm evaluator preflight `37716773` completed on two A100 80 GB GPUs
using the exact `3a9552b` WTS training wheel/image. It exercised actual video
decode, multimodal inference and the shared MCQ scorer on two examples. That historical preflight validated execution, but did not detect the media
contract defect described below; its scores are not used. The corrected
`2a8b48d` preflight `37718708` additionally verifies exact SFT media/prompt
equivalence before actual two-rank MCQ execution. Full evaluation remains gated
on completed training and artifact checks.

Source comparison after `076cfbb` finds no changes to dense Qwen3-VL's production
SFT/adapter/packer/checkpoint path through `08e5ff5`: later production changes
are GR00T-specific construction/metadata and the native MoE inference-device
bridge. Qualification helpers gained explicit deterministic profiles and the MoE
FP32 oracle. Local full-WTS results remain labeled `076cfbb`; Slurm results remain
labeled `3a9552b`, which predates the shared portable-publication refactor and
physical-autocast extensions. Results are never relabeled as a newer wheel.

The prior full WTS evaluation jobs `37711018`, `37711029`, `37711043`,
`37711056` were held before execution after a
real CPU probe found that decoded-frame evaluation lost source timestamps and
changed the resize grid. The new inference-only fix preserves original frame
metadata and delegates decoded-video preprocessing to Qwen3's processor. Small
regressions cover short clips, batched processing, cache isolation and invalid
metadata; two real WTS clips now have exact SFT/evaluation pixel and timestamp
agreement, including exact generation-prompt token IDs and cached repeats.
Installed host and Slurm checks passed; the held unstarted jobs were canceled
and replaced with the corrected jobs above. Old inputs and receipts remain
available. Historical scores using the earlier preprocessing are not comparable.

The local `076cfbb` LoRA and rsLoRA epochs are complete: all 5,555 rows,
1,111 applied updates, no omitted/padded rows, and complete 2,676-row validation.
Their token-weighted training means are `0.227146` / `0.192750`; validation
loss moves from `0.843903` to `0.170965` / `0.179041`. Their generation scores
and final checkpoint recovery/export remain pending; these partial results are
not substituted for the independently running Slurm verification.

MoE Slurm LoRA, rsLoRA and DoRA have completed training, continuation, portable
reload and FP32 merged qualification. PiSSA's training and exact BF16 portable
reload pass, and all 882 serialized merged tensors match; the FP32 stock-HF
portable/merged oracle differs from the native reference on one of eight probes
(maximum logit error about `0.02952`). The diagnostic job `37718577` also
reproduces the failure using a private native merged forward (`0.02950`),
so a native-only export loader does not resolve it. The first route change is
at language layer 37, token 7: the 8th/9th expert probability margin is only
`7.45058e-9`; experts 5 and 68 exchange the final selected slot. Later routing
and attention amplify the difference. All 64 controlled greedy positions agree,
but the fixed elementwise/mean output gates fail.

This is a concrete incomplete PiSSA/MoE merged-export requirement. The public
exporter now refuses that combination with an actionable portable-adapter
alternative; the rejection does not count as merged-export implementation.
It does not change training, checkpoint continuation, PiSSA decomposition, expert
routing or the numerical thresholds. Dense VLM/HF and physical PiSSA exports
remain supported in their completed profiles. The compact
[diagnostic evidence](assets/moe-pissa-export-limitation.json) retains the actual
source/wheel/container, artifact identity, errors and first routing divergence.
