# Native GR00T N1.7 backbone adaptation

This integration connects native LIBERO action flow matching to ordinary LoRA on a frozen
language backbone. The implementation is available; task quality, performance and device support
must be read from the accompanying qualification evidence. CPU tests and configuration acceptance
alone do not establish accelerator support or robotic task success.

The native integration is implemented, but the requested quality and substantial
total-memory outcomes remain unmet for the selected full-head recipe:

| Requirement | Current evidence |
|---|---|
| Native action training and recovery | Language LoRA and the full action head receive the native flow loss. Real A100 80GB checks cover exact trainable membership, frozen NF4 state, recovery, and fresh native policy reload. |
| NF4-only memory benefit | The matched selected run measured 30.412 GiB NF4 versus 31.516 GiB BF16 sampled GPU-process peaks: 1.104 GiB saved, about 3.5%. FP32 action-head parameters, gradients and optimizer state dominate; substantial total-memory savings are not established. |
| Offline quality | Across three paired training seeds, mean NF4 action MSE was 2.224% lower than matched BF16 LoRA, but the one-sided 95% upper regression bound was +9.532%, failing the predefined 5% guard. |
| Rollout quality | The selected update-200 development rollout scored 98/100 NF4 versus 97/100 BF16; the accuracy-loss upper bound was 3 percentage points, failing the 2-point guard. This does not establish real-robot behavior. |
| Execution improvement | Four paired cache trials, same-build mature recovery and current-build compatibility passed. Evaluation reuse is qualified as an opt-in for the tested full-head recipe; this is a shared execution improvement, not NF4 learning superiority. |
| Deployment artifacts | Native portable and standalone unmerged policies retain action components and reproduce task outputs. Merged exports failed parity checks; controlled deployment-latency improvement remains unproven. |

Detailed results below preserve the original builds, negative trials, uncertainty
procedure and resource accounting. Optimizer quantization is a separately identified
integration and is not included in these baseline memory or quality results.

The separate `integration/native-action-qlora-bnb8` branch now has an A100
four-cell base-precision × optimizer-precision canary on native build `1af7d04`
(evidence `517f240`, job `38612140`). All fourteen training, own-update-5-to-10
recovery, interruption and fresh-portable stages passed. Over the ten-update
training stages, sampled owned GPU peaks were 31.516/22.289 GiB for BF16 with
FP32/eight-bit AdamW and 30.412/21.184 GiB for NF4. The approximately 9.23 GiB
within-base reduction belongs to optimizer compression, not NF4. Serialized
FP32 masters add 6.111 GiB to the resumable payload; the complete diagnostic
process tree reached 70.383 GiB host RSS. This separate canary establishes
correctness and measured memory savings, not quality, speed, cache-enabled
operation or rollout qualification. Its fixed-300-update development screen
and mature recovery remain pending. The main recipe still uses fused AdamW
with FP32 state. The [bound evidence](evidence/native_qlora_family_engineering_4cba88f.json)
also preserves the earlier pre-model packaging failure (`38611005`).

The first controlled selected-policy response benchmark (`38616477`, 207
GPU-seconds) failed in its measurement wrapper after the first GR00T worker
completed 65 response calls and 20 untimed input/task proofs. Torch reported the
same physical GPU without NVML's `GPU-` prefix, and the wrapper compared the raw
strings. Frozen/model/RNG integrity passed in that worker, but the original
wrapper and allocation remain failed; no later case ran. A repair passing five
captured-identity CPU tests uses the existing physical-UUID normalizer and checks
ownership before loading.
All sixteen cases must be rerun together before a matched latency claim. This
does not change either native model or its historical quality conclusions.

The clean retry (`38618180`, 212 GPU-seconds) passed that first case completely,
including its wrapper and 100/100 strict memory samples, then failed before case
two: `srun` consumed the input stream carrying the remaining case schedule. The
original sixteen-line schedule and failed allocation are retained. The narrow
launcher repair isolates stdin; no partial case is spliced into a matched result.

The next full attempt (`38619742`, 4,681 GPU-seconds) passed thirteen native and
wrapper cases, then failed before case fourteen: Pyxis/enroot could not create
the canonical output mount directory beneath a read-only parent bind. All 226
raw files, 845 response calls, 260 untimed proofs and 3,009 strict memory samples
were independently replayed. Available own-policy output comparisons pass, but
the sixteen-case comparator never ran; this remains a failed study with no
matched latency claim. A later login-host write probe passed. It does not explain
the exact failed-node trigger or establish container writability. Narrow input
mounts and repeated compute-side container starts must be checked before a retry.

The narrower-mount canary (`38624272`, 91 GPU-seconds) passed its first actual
container probe: all forty mount flags, seventeen input roots at both aliases,
4,653 bounded file reads, mapped host identity, A100 80GB placement and durable
write/fsync/rename/readback checks. It then failed before start two because an
immediate scheduler query still listed step `.0` after successful `srun` return.
Later accounting records that step as completed. The original twenty-start test
remains failed; a bounded release wait is needed before the next container.
Single-container task diagnostics can proceed with their own compute preflight.
No repeated-start, model-performance or quality result follows from this probe.

The bounded-wait successor (`38634861`, **1,230 GPU-seconds**) subsequently passed
all twenty fresh container starts and their step-release checks. Its 109 retained
files cover 93,060 bounded one-byte file-access checks, all forty mount flags,
durable-write cleanup and unchanged GPU/CPU placement. Eighteen releases required
four recorded queries and two required five. No `srun` was retried. This qualifies
the repeated-container infrastructure for the unchanged sixteen-case benchmark;
no policies were loaded and no latency or quality result follows from this canary.
All preceding failed allocations retain their original outcomes.

The subsequent full response study (`38638146`, **5,210 GPU-seconds**) completed
fifteen of sixteen native cases, all with valid model/output/RNG and memory checks.
After case fifteen, three quick queries still listed the finishing numeric step;
two later scheduler queries each timed out at ten seconds. The thirty-second
release deadline stopped the allocation before the last GR00T NF4 portable case.
Its eventual step-clearance time is unknown. The allocation remains **failed**;
the full comparator rejects the missing case, and no matched latency result is
reported. All 307 files, 975 response calls, 300 input/task-output proofs and 3,234
raw memory samples were replayed. Thirteen short phase intervals lack samples.
The eight completed first-block cases remain valid artifact-output references;
they do not supply a completed timing comparison.

A prospective operational repair extends only the read-only step-release wait to
180 seconds and 90 attempts, retaining ten-second query limits, two-second
intervals, exact numeric-step absence, atomic failure records and no `srun` retry.
Its CPU refusal/timeout tests pass; actual repeated-container execution remains
to be checked. This does not change either model, a quality margin, or the outcome
of the failed thirty-second protocol.

The legacy-inference loader `374cee0` then passed its fresh-process qualification
(`38642797`, one A100 80GB, 53 min 31 s, **3,211 GPU-seconds**). All eight retained
selected policies (GR00T and Alpamayo × BF16/NF4 LoRA × portable/standalone
unmerged) loaded through the current native evaluation entrypoints and reproduced
their twenty reference task-score rows, with model, RNG and module modes unchanged.
All four GR00T policies also reloaded through the normal rollout-policy loader.
Every container step released within the 180-second bounded wait. This checks
current-build artifact compatibility against the eight first-block references of
`38638146`; it is not a latency, quality or training result, and the matched
sixteen-case latency study still requires a clean full rerun.

Native commands: [training and recovery](#native-training-and-recovery),
[fresh-process evaluation](#fresh-process-evaluation),
[standalone artifacts](#standalone-artifacts), and [LIBERO rollout](#libero-rollout).
The source-build and qualification limits in each section remain applicable.

The initial architecture is `nvidia/GR00T-N1.7-LIBERO` at
`2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`, LIBERO-spatial policy, with the checkpoint's
16 retained Cosmos-Reason2-2B language layers, 2048 hidden width and 40 × 132 action head.
The native processor supplies the `libero_sim` embodiment, state/action statistics, two RGB cameras
and 16 valid embodiment action slots. The selected prepared demonstration has ten supervised
7-dimensional environment-action steps. State is xyz, axis-angle and two finger positions;
training converts environment gripper `[-1,1]` to native `[0,1]`, then uses checkpoint normalization.
Action evaluation reverses normalization and gripper conversion. Native four-step flow sampling
is unchanged. Offline action error is not rollout success or a real-robot safety result.

Backbone processor/configuration metadata is pinned to
`nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561`; the public
`Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203` supplies
byte-identical metadata. The weights come from the complete pinned GR00T checkpoint
above, not replacement Qwen weights. These identities are enforced in
[`qlora_source.py`](../../cosmos_framework/posttrain/model/gr00t/qlora_source.py).

The language allowlist contains exactly 112 bias-free matrices: q/k/v/o and gate/up/down in each
retained layer. Vision, embeddings, norms, output head and all action components are excluded
from NF4. The source architecture is built on meta and official safetensors are read one tensor at
a time into their final owned CUDA representation. This avoids loading a dense base on the GPU.
Both precision arms store the remaining backbone in BF16; selected trainable action-head/projector
parameters and ordinary rank-32/alpha-64 LoRA are FP32, with FP32 fused AdamW moments. Task state/action inputs and training noise/time math retain native FP32 precision; the
backbone embedding dtype does not change the native action noise schedule. Native BF16
autocast also applies to adapter GEMMs; FP32 parameter storage does not imply FP32 GEMM execution.
TF32 controls are explicitly enabled as in the upstream head-training path and recorded in identity.

`model.backbone_adaptation.mode` selects four distinct controls:

| Mode | Language weights | Trainable language adapters | Native action projector/DiT |
|---|---|---|---|
| `nf4_lora` | frozen NF4, nested quantization, uint8 storage, BF16 compute | ordinary FP32 LoRA | trained FP32 by default |
| `bf16_lora` | frozen BF16 | identical LoRA | identical policy |
| `nf4_frozen` | frozen NF4 | none | trained FP32 |
| `bf16_frozen` | frozen BF16 | none | trained FP32 |

Job `38536188` exercised the two **head-only** modes at LR `1e-5` for 300 updates
on seed 17, with the same 160-row scorer and fixed target `0.049561327725898344`.
Both original final artifacts contain all 537 FP32 head tensors, no LoRA factors,
and exactly match their DCP tensor values. All 537 tensors changed between updates
200 and 300 and retained nonzero optimizer first moments. The repeated BF16 arm
is the existing seed-17 control, not an additional independent seed.

| Head-only mode | MSE at 300 | First target update | Native workflow (s) | Cold case including step release (s) | Sampled owned GPU peak (GiB) |
|---|---:|---:|---:|---:|---:|
| BF16 frozen backbone | 0.03983719 | 200 | 483.74 | 1002.93 | 31.154 |
| NF4 frozen backbone | 0.04042368 | 100 | 512.63 | 1049.78 | 30.045 |

NF4 used 1.109 GiB less sampled GPU memory and reached this offline target earlier
(318.37 versus 418.66 seconds including preparation), but its complete case was
slower and final MSE was 1.47% higher. Both arms' six decoded motion-axis MAEs
worsened relative to their own frozen source, while gripper MAE improved. These
are single-seed point results; no rollout preservation or general speed improvement
is established.

The allocation **failed final qualification acceptance** after all native stages
finished (3,643 GPU-seconds). Each own-200-to-300 continuation reproduced all 100
numerical update records, task history and full 160-row scores; fresh DCP and
portable scores also matched exactly. The on-node full model/optimizer/RNG/data
comparisons ran before these later checks, but their returned dictionaries were
not published when the final memory gate failed. The retained tensor audits and
task files are separate evidence, not a fabricated final acceptance receipt.
Both uninterrupted arms had complete host observations; one of 1,149 recovery
samples exceeded the fixed four-second v9 collection budget during process exit.
That sample and the failed whole-recovery host-PSS criterion remain unchanged.
Old `4cba88f` fresh evaluation emits no allocator phase records, so its sampled
GPU/host peaks are lower bounds. The bound raw receipts and precise failure are in
the [family evidence](evidence/native_qlora_family_engineering_4cba88f.json).

`train_action_head=false` is permitted only with adapters. It is a separate learning recipe,
not the selected jointly adapted recipe. The old cached-feature PPO path is rejected for backbone
adaptation because its no-grad features bypass language adapters. The flow-BC forward retains the
original noise distribution, time sampling, velocity objective and masked denominator. No language
SFT surrogate is used. The matched head-only controls above use fused AdamW and a single decay group;
the older native official recipe uses unfused AdamW and HF decay groups, and remains a distinct
comparison rather than an interchangeable control.

Build `df24559`, using shared artifact changes `6fbf332`, adds the explicit
`gr00t_n1d7_dit_attention_lora_v1` head profile. It trains ordinary rank-32/alpha-64
LoRA in the 128 q/k/v/output projections of the native DiT, alongside the existing
112 language targets. All 537 original head tensors remain frozen FP32; native
dropout, flow loss, state/action processing and sampling are unchanged. Category-specific
projectors, feed-forward blocks, normalization and VL self-attention receive no adapters.
The optimizer contains exactly 480 FP32 factor tensors. This is a new learning
recipe, not an arithmetic-preserving replacement for full-head training.

Select it through the same native training/evaluation/export commands with both
overrides:

```text
model.backbone_adaptation.train_action_head=false
model.backbone_adaptation.action_head_adapter_profile=gr00t_n1d7_dit_attention_lora_v1
```

Here `train_action_head=false` freezes original head parameters; it does not disable
the head adapters or put the head in evaluation mode. Portable artifacts retain all
480 factors. Standalone unmerged artifacts additionally retain every frozen FP32
head base and bias; merged export explicitly refuses this profile. The new source
identity requires its own compatible loader; historical artifacts retain their
recorded builds and are not rewritten.

The shared 132 CPU checks, 53 affected native checks and 21 existing configuration
checks passed; independent peers also passed the 24 shared and 28 native new cases.
These overlapping suites are contract evidence, not CUDA qualification. The
[prospective paired development screen](evidence/native_qlora_gr00t_dit_attention_protocol.json)
fixes seed 17, LR `1e-5`, 200 updates, the unchanged action-MSE target and all seven
source-relative decoded-axis guards before outcomes. A100 job `38559188` passed all
six initialized BF16/NF4 controls, including exact task outputs, RNG and existing
language-adapter gradients, then completed three native BF16 updates. Its enclosing
qualification failed because the checker omitted one `.model` level from the
canonical language-target paths; the saved policy contains all 112 intended targets.
The failed 634-GPU-second allocation and its complete 438 memory observations are
retained. Continuation `38561399` then passed the remaining nine model stages,
reusing those initialized controls and the completed BF16 run. Both arms verified
all 480 trainables and 960 FP32 moments, frozen original head/quantized state,
exact own-update-2-to-3 recovery, and fresh two-row portable and standalone
unmerged outputs. Unmerged artifacts retain all frozen head weights and biases
alongside the factors; both merged output dtypes are explicitly refused.
All 764 current host observations and the 438 original observations were complete,
with separate clocks and memory records. Some short phase intervals were unsampled.
The successful continuation cost 1,072 GPU-seconds; the failed 634-second attempt
and the earlier eight-second pre-runtime failure remain charged separately.

These three-update checks establish lifecycle correctness, not development quality,
long-run recovery or a matched performance result. Diagnostic full-head memory is
separate from the new recipe's training peak. The retained full-head and frozen-head
negative results remain unchanged; any head-state memory saving belongs to both
precision arms.

The fixed 200-update screen (`38562915`) subsequently **failed its quality rule**
and the head-adapter recipe is not promoted. Both precision arms applied every
update, preserved frozen state, and passed fresh 160-row portable parity. Their
initial full-development rows and metrics exactly reproduce their own frozen-source
calibrations, so this result does not identify an initialization or scoring defect.

| Head-adapter screen | Source MSE | MSE at 100 | MSE at 200 | Worsened decoded axes at 200 | Sampled training-process peak (GiB) |
|---|---:|---:|---:|---:|---:|
| BF16 | 0.05506814 | 0.05492163 | 0.05700561 | 5 | 13.049 |
| NF4 | 0.05565446 | 0.05588415 | 0.05705529 | 3 | 11.926 |

Neither arm reached the unchanged `0.049561327725898344` target at a scheduled
observation. NF4 is only 0.087% worse than BF16 at the endpoint, but both worsen
against their source and fail the required action-axis guards. Gripper error
worsens in both; the drawer and next-to-plate tasks are shared adverse slices.
All 480 factors changed between updates 100 and 200, so missing optimizer updates
do not explain the failure. This does not establish whether rank, target coverage
or learning rate caused it; another intervention requires that diagnosis first.

Native workflow clocks were 471.85 seconds BF16 and 505.36 seconds NF4, including
startup, training, scheduled evaluation and publication. Training components were
87.19/95.42 seconds; evaluation components were 304.22/329.44 seconds. Separate
enclosing fresh-reload stages took 188.74/197.49 seconds. The whole four-stage pair,
including intervening audits, took 1,406.99 seconds; allocated cost was 1,476
GPU-seconds. These are one fixed-order seed on one A100, not a sustained performance
qualification or a time-to-quality result. All 1,098 host observations were complete;
short phase gaps and sampled-peak limits remain recorded. The reduction relative
to earlier full-head runs is shared by BF16; those runs are a descriptive comparison
with a different build and horizon. The matched NF4 increment here is 1.123 GiB.
Memory savings do not override the failed learning gate. No mature continuation,
horizon extension, extra seed or rollout is promoted from this screen.

A read-only diagnostic (`38567435`, 222 GPU-seconds) then loaded the rejected
BF16 update-200 policy and temporarily disabled either learned adapter group.
It used the first two already-consumed training rows per task in the original
200-update cursor: 20 fixed rows, identical inputs/noise, and native four-step
sampling. These are interventions on one trained policy, **not retraining with
different trainable modules** or held-out quality estimates.

| Applied learned deltas | Native flow loss | Sampled normalized action MSE |
|---|---:|---:|
| Language and head | 0.02260408 | 0.01013668 |
| Language only | 0.02358116 | 0.01011890 |
| Head only | 0.03066952 | 0.00994220 |
| Neither | 0.03151182 | 0.00994368 |

With the learned head delta retained, adding the language delta reduced flow loss
by 26.30% but increased sampled action MSE by 1.96%; MSE worsened on 13/20 rows.
Thus lower native flow loss does not establish better sampled actions here.
The result identifies a checkpoint-specific effect, not a coding defect, an NF4
effect, or proof that another learning rate or training policy will help. The
original screen remains rejected; the selected full-head recipe is a different
policy and requires its own diagnosis before transferring this conclusion.

All input/RNG comparisons and restoration of scales, mixed modes, gradients,
storage and tensor values passed; independent replay checked the 68 retained
files and native metric reductions. Full tensor/RNG equality remains the on-device
attestation. The observer failed before its first host/NVML sample because absolute
script dispatch under `PYTHONSAFEPATH` omitted a sibling import. Those measurements
are missing, not zero. Native allocator records remain. A future-only package-module
dispatch correction passed five actual safe-path subprocess tests; the original
payload and failure are unchanged, and no GPU repeat was made solely for memory
paperwork. This diagnostic supplies no memory-saving or performance claim.

A separate diagnostic (`38573209`, 177 GPU-seconds) then tested the **selected
full-head BF16 update-200 policy**, using its original compatible `4cba88f` loader
and the same 20 consumed training rows. All 537 trained head tensors stayed fixed;
only the learned language-adapter scale changed temporarily.

| Selected full-head diagnostic | Language enabled | Language disabled |
|---|---:|---:|
| Native flow loss | 0.05072161 | 0.05048209 |
| Sampled normalized action MSE | 0.02207795 | 0.02185888 |

The prospective hypothesis condition **failed**. Language-on MSE increased by
1.0022%, with 8/10 task means worse, but flow loss also increased by 0.4745%.
The required lower-flow-loss condition was absent. All seven decoded-axis mean
errors increased on these calibration rows. This is a checkpoint-local inference
intervention, not a retraining result or held-out estimate. No lower-language-LR
trial or new recipe was selected from it; the earlier head-profile diagnosis is
not transferred to the full-head policy.

All 40 native row evaluations, six restoration checks and the 84-file independent
replay passed. The corrected observer collected 100 complete host observations
with valid owned-GPU attribution; these are descriptive sampled lower bounds,
not matched performance measurements. The original missing observations remain
missing. A preceding admission preflight timed out on a scheduler query before
any GPU launch. Its failure and successful write probe were preserved; a narrow
reuse of the existing bounded whole-census retry passed 41 CPU checks and the
fresh preflight before the single submission. No scientific setting changed.

The subsequent source-versus-update-200 component diagnostic (`38624956`,
361 GPU-seconds) completed all 84 native calls in both precision arms. Independent
replay verified 80 captured rows, 160 masked error arrays, native scalar reductions,
paired inputs/RNG and all four restoration receipts. The proposed explanation was
that gripper gains concealed worsening motion errors inside an improved objective.
Its prospective conjunction **failed**: all six motion-axis means worsened and
gripper means improved, but total error increased in both arms and both objectives.

| Fixed TRAIN diagnostic | BF16 source → selected200 | NF4 source → selected200 |
|---|---:|---:|
| Native flow loss | 0.031512 → 0.050722 | 0.026149 → 0.044059 |
| Sampled normalized action MSE | 0.009944 → 0.022078 | 0.009969 → 0.018253 |

The component-weighting hypothesis is terminated; no learning screen was selected.
These twenty consumed training rows use per-row common noise, unlike the retained
full-160 development cohort and scorer stream. They do not overturn its observed
MSE improvements or establish representative training harm, gradient conflict,
a code defect, quality preservation or performance. All raw captures and the
negative decision remain in the existing family evidence record.

The separate `gr00t_n1d7_libero_official_head.py` native action-SFT recipe makes that existing
control executable. It streams the same pinned base into **FP32 backbone/head storage with BF16
autocast**, trains only the native head, and uses `adamw_torch` with HF decay/no-decay groups.
The actual pinned finetune launcher overrides its generic training-config fused-AdamW default;
this control follows the launcher and the existing Cosmos optimizer recipe. Training uses the
checkpoint's random crop/color jitter and 0.2 processor state dropout, while evaluation uses the
native deterministic transforms. Albumentations 1.4.18 is required: its transform randomness uses
the checkpointed global Python/NumPy streams, verified with actual nonzero augmentations.

This control explicitly adjusts the generic upstream batch-64/10,000-update defaults to batch one and
300 updates, with 15 warmup updates and the pinned HF cosine scheduler (including zero endpoints).
The same prepared ten-action supervision prefix remains explicit. It is a separate practical
control, not a replacement for matched BF16 LoRA or the already frozen development protocol.
Use the normal training command below with `--config
cosmos_framework/posttrain/configs/gr00t_n1d7_libero_official_head.py`, replace the
`model.backbone_adaptation.*` overrides with `data.backbone_path=/staged/Cosmos-Reason2-2B-metadata`,
retain the dataset overrides, and replace the starter horizon/cadence with
`rl.max_steps=300 rl.evaluation_every=100 rl.save_every=100`.
Set `rl.keep_model_last=0 rl.keep_optim_last=0` to retain the intermediate checkpoints
used in the recorded comparison. Native checkpoint/recovery and complete head-artifact reload
use the same entrypoints; dense export is explicitly unavailable for this additional control.
CPU optimizer/scheduler, augmentation and alias contracts are covered. A100 job `38509993`
also passed ten real updates, exact fresh checkpoint/portable scores, update-5-to-10 recovery,
frozen-source and optimizer-membership checks, eight shared CUDA cases, and fresh reload of
the earlier NF4 artifact. These lifecycle checks are distinct from the completed
300-update development comparison below.

The [prospective selected-candidate comparison](evidence/native_qlora_gr00t_selected_official_comparison.json)
binds the retained LR `1e-5` NF4 candidate and its complete paired-seed study to
this separate official control, before any official development quality result.
The original practical-analysis plan named the failed LR `1e-4` study; that
complete comparison remains unperformed. Its identities and failures are retained.
The selected comparison requires all three official seeds, keeps the matched
BF16 LoRA comparison separate, and retains the original target and quality guards.
It does not establish equal-compute superiority or independent confirmation.

All three practical-control seeds completed 300 updates and three fresh 160-row
portable evaluations each. All updates applied, primary fresh scores exactly
matched resident endpoints, and every decoded action-axis MAE worsened against
its own frozen source. These are negative whole-recipe results, not an isolated
test of augmentation. The historical seed-17 result and all failed attempts remain
in the [family evidence](evidence/native_qlora_family_engineering_4cba88f.json).

| Official control seed / job | MSE at 300 | Fixed target | Complete case through owned-step release (s) | Complete host samples |
|---|---:|---|---:|---:|
| 17 / `38546671` | 0.09820884 | unreached | 1,096.554 | 768/768 |
| 42 / `38552199` | 0.10274918 | unreached | 1,109.778 | 782/782 |
| 123 / `38553000` | 0.11324241 | unreached | 1,095.880 | 779/779 |

The full-panel analysis (`38553548`) validated all nine selected cells and all
three official cells before computing the unchanged 10,000-draw paired
seed/task/episode analysis. NF4's equal-seed/task/episode mean MSE is 0.04412088.

| Reference | Reference mean MSE | NF4 relative change | One-sided 95% upper regression bound |
|---|---:|---:|---:|
| Matched BF16 LoRA + head | 0.04512450 | -2.224% | +9.532% |
| Matched BF16 head-only | 0.04406531 | +0.126% | +12.159% |
| Separate official head recipe | 0.10473348 | -57.873% | -47.785% |

The primary **5% preservation guard fails** against matched BF16 LoRA despite
the favorable mean. Improvement over the weaker practical control does not
replace this comparison. Three seeds and ten exposed task clusters do not
establish independent confirmation, and the three-seed rollout-preservation
claim remains unverified. Fixed 300-update endpoints are not equal training
compute. Additional sampling-noise seeds are diagnostics, not training seeds.

Independent replay of all 1,920 endpoint rows reproduced the three comparisons
exactly. Descriptive slices show variation in both seeds and tasks: NF4 changes
against BF16 LoRA are -2.12%, -8.11% and +3.26% for seeds 17, 42 and 123.
Five of ten task means worsen; the largest are the top-drawer task (+30.59%)
and the next-to-ramekin task (+18.12%). These slices neither identify a coding
defect nor justify removing examples or changing the acceptance procedure.

The original first target crossings remain: NF4/BF16 LoRA at updates 200/200
for seed 17, 100/100 for seed 42, and 300/100 for seed 123. Every selected arm
attained the target; every official arm remained unreached. Original clocks
and their placement/order limitations are retained separately; no pooled new
timing block or demonstrated NF4 time-to-quality advantage is inferred.

The missing matched head-only seed-42 case (`38547636`) completed all 300
updates and fresh 160-row evaluations, but **failed its enclosing completion
check**. Its final MSE was 0.04523234 and its first target crossing was update
100; all 738 observed host collections were complete. One Slurm status query
timed out after ten seconds, and the former release helper stopped immediately.
Its later verified release cannot recreate the original completion clock.

The [prospective measurement repair](evidence/native_qlora_family_engineering_4cba88f.json)
retries explicit query timeouts within the same 180-second deadline, retaining
every query and requiring final exact step absence plus matching parent ownership.
Partial output and ambiguous completed replies cannot establish release. The
clean head-42 repeat and remaining selected/official seeds retain their native
builds, objectives, settings, target and statistical procedure. Original failed
results and costs remain recorded; the repair itself supplies no quality or
performance result.

The clean repeat `38548973` completed and passed the repaired release check.
All 300 reported numerical update records, four resident scores and three fresh
scores match the failed attempt; the portable model component digest also matches.
This is not a full optimizer/RNG/data tensor-state comparison. The new complete
case took 1,014.985 seconds, including 5.460 seconds of owned-step cleanup, and
all 735 host collections were complete. Its first target publication was at
update 100, 309.034 seconds after case preparation began. These are new clocks;
the original failed clock remains unaccepted.

The remaining selected seed-123 block (`38549881`) completed all three 300-update
arms and three fresh 160-row evaluations per arm. Primary fresh scores matched
resident update-300 scores exactly, all 900 updates applied, and all 2,393 recorded
host collections were complete. The allocation consumed 3,384 GPU-seconds.

| Seed-123 arm | MSE at 300 | First target update | Preparation through target publication (s) | Complete case through step release (s) |
|---|---:|---:|---:|---:|
| NF4 LoRA + head | 0.04925060 | 300 | 708.684 | 1,218.746 |
| BF16 LoRA + head | 0.04769373 | 100 | 347.388 | 1,122.945 |
| Matched BF16 head-only | 0.04712640 | 200 | 444.210 | 1,029.949 |

These are point results with original clocks. The separately allocated correctness
gate below overlapped the final BF16 LoRA arm on a different node; this asymmetric
overlap remains a timing limitation. The completed grouped analysis above retains
this block without reconstructing clocks. It supplies development uncertainty,
not independent confirmation or recipe-level superiority.

The bounded GR00T profile `38549122` measured duplicate backbone work between
native flow-loss and sampled-action evaluation on two exposed rows per arm.
After warmup, that duplicate occupied 20.44% of NF4 and 17.33% of BF16 traced
serial evaluation time. Applying those fractions to retained evaluation costs
gives optimistic native-workflow reductions of 10.51% and 8.60%; **these are
projections, not achieved speedups**. The traced calls were 1.64×/1.70× slower
than their surrounding untraced calls. All five evaluations per arm preserved
the controlled scores, RNG, modes, frozen weights and prepared inputs.

Implementation `47587e0`, with ownership fix `cbc950f`, adds
`rl.reuse_gr00t_evaluation_backbone=true` for the pinned one-row NF4/BF16 LoRA path.
It reuses only the raw backbone output within one evaluation batch, with independent
inputs for both native head calls and a 16 MiB retained-tensor bound. The head
mutates its input mapping, so reusing already transformed features is incorrect.
The option remains off by default. Its matched workflow study, trained-boundary
recovery and current-build compatibility checks below qualify it as an opt-in for
the tested full-head recipe. Both precision arms receive the same option and must
preserve their numerical trajectories.
The first profile allocation `38548693` failed before model loading because a
launcher imported an executable monitor module; its 68 GPU-seconds remain
recorded separately from the successful diagnostic's 334 GPU-seconds.

The first correctness allocation (`38551713`, 996 GPU-seconds) remains **FAILED**.
Before its diagnostic comparator failed on a Transformers `BatchFeature` mapping,
NF4 OFF/ON states at updates 5 and 10 matched exactly; its own enabled 5-to-10
recovery matched model/optimizer, RNG, cursor, controls and subsequent updates.
Fresh DCP and portable evaluation matched all 20 engineering rows. Reuse retained
2,214,696 bytes and recorded 60 hits/60 misses over the uninterrupted validations.
All 702 recorded host collections were complete. Those subproofs alone did not
establish the arithmetic/failure diagnostic, BF16, standalone export, Alpamayo
regression or overall acceptance. The original failure remains unchanged.

Continuation `38554101` completed on the same `cbc950f` native build after a
qualification-only repair of the actual `BatchFeature` comparison. Both precision
arms now pass OFF/ON state comparisons at updates 5 and 10, enabled 5-to-10
recovery, fresh 20-row checkpoint/portable scoring, and strict source refusal.
Real-model diagnostics match all 761 unclipped gradients and native sampled
outputs, with unchanged inputs. Injected evaluation failures restore RNG and mixed
module modes and clear cached inputs/features. Standalone unmerged reload preserves all 761 FP32 trained
tensors, 112 effective BF16 weights, aliases and controlled task outputs. A
two-row Alpamayo portable regression also matches through the shared evaluator.

The new allocation consumed 2,100 GPU-seconds and passed all 1,557 observed host
collections with valid GPU attribution. Its observed workflow was 2,027.981 seconds;
startup outside the observer and unsampled short subintervals remain explicit.
The original NF4 training/recovery/fresh-load proofs are carried forward from
`38551713`; new NF4 diagnostics/export and all BF16 checks come from `38554101`.
The combined qualification cost is 3,096 GPU-seconds, with separate clocks and
no repaired historical acceptance. Independent replay verified all 193 new raw
files. This establishes `cbc950f` correctness, not a speed, quality or new
source-path-absence claim. Matched timing and any resulting mature enabled-mode
qualification remain separate gates. Evaluation reuse combined with the newer DiT
head-adapter profile has not been qualified; its retained `e64a735` lifecycle and
learning checks used reuse disabled. Configuration acceptance does not establish
that combination or transfer the older source-bound artifact evidence.

The first frozen timing pair (`38590055`, NF4 seed 17, OFF then ON) completed
on one A100 80GB with identical 16-CPU affinity and recorded NUMA constraints.
Memory-page residency was not measured. Both cases ran all 300 updates, full 160-row evaluations at 0/100/200/300, and three fresh
160-row portable evaluations. All numerical updates, trained/optimizer states
at 100/200/300, scheduler/RNG/cursor controls and task rows matched exactly.

Evaluation time fell from **328.91 to 259.96 seconds (20.96%)**, native-process
time from **630.50 to 560.84 seconds (11.05%)**, and complete cold-case time
through fresh scoring and owned-step release from **1,182.69 to 1,067.76 seconds
(9.72%)**. Both reached the unchanged target at update 200; the native run's own
target-publication clock was 435.11/381.35 seconds. That clock excludes the
separately recorded launcher prelude and is not reconstructed from cold-case time.

Sampled owned-GPU peak was unchanged at **30.412 GiB**. Host PSS rose by 2.31 MiB;
the cache retained 2.11 MiB with 640 hits and 640 misses in native evaluation.
All 1,604 observed host samples and eight model-stage summaries passed. Short
unsampled setup, publication and checkpoint intervals remain explicit; sampled
peaks are lower bounds. Independent replay verified 238 files and terminal
release. The allocation cost 2,622 GPU-seconds, including a separate 271.00-second
saved-state comparison.

The second frozen pair (`38592360`, BF16 seed 17, OFF then ON) also passes the
continuation rule. It repeats the same 300-update/full-160-row/fresh-three-seed
protocol on one A100 with identical within-pair CPU affinity and recorded NUMA
constraints. All saved states at 100/200/300, numerical updates, task rows and
fresh scores match exactly. Evaluation time fell **303.15 to 258.37 seconds
(14.77%)**, native-process time **606.95 to 560.82 seconds (7.60%)**, and complete
cold-case time **1,146.60 to 1,053.89 seconds (8.09%)**. Both reached the unchanged
target at update 200; their native target-publication clocks were 413.30/382.31
seconds, with the same separate-launcher scope described above.

Sampled owned-GPU peak remained **31.516 GiB**, and all memory gates passed.
The cache retained 2.11 MiB; the observed host-peak decrease does not establish a
causal memory benefit of caching. Independent replay verified all 242 files,
1,566 complete host samples, eight stage summaries and terminal release. Short
unsampled intervals and sampled lower bounds remain. The allocation cost 2,499
GPU-seconds, including a separate 251.50-second saved-state comparison.

The third pair (`38595290`, NF4 seed 42, ON then OFF) also passed the unchanged
continuation rule. Evaluation time fell **613.75 to 570.83 seconds (6.99%)**,
native-process time **1,332.80 to 1,214.67 seconds (8.86%)**, and complete cold-case
time **3,271.24 to 3,084.11 seconds (5.72%)**. Both reached the fixed target at
update 100; native target-publication clocks were 501.73/416.06 seconds, with the
same separate-launcher scope. All 300 updates, checkpoint states at 100/200/300,
resident rows and three fresh score sets matched exactly.

Sampled owned-GPU peak remained **30.414 GiB**; host PSS rose 25.55 MiB, within
the frozen bound, and the retained cache remained 2.11 MiB. Independent replay
verified 244 files, all 3,299 observed host samples and eight stage summaries.
The observed model-workflow spans exclude earlier launcher/container verification;
short unsampled phases and unmeasured filesystem cache remain explicit. Loading
was unusually slow, including 548.06 seconds for one fresh model load. The cause
was not established. All delays remain in the cold clocks. The allocation cost
8,473 GPU-seconds, including a separate 2,070.99-second saved-state comparison.

Supplemental Slurm task accounting for the second and third pairs starts
52–596 seconds before the application sampler. Its task MaxRSS values were
7.660/8.877 GiB for BF16 seed 17 (OFF/ON), and 8.892/8.887 GiB for NF4 seed 42
(ON/OFF). These are separate accounting observations, not replacements for
process-tree PSS: maxima lack timestamps, the currently reported sampling interval
is 30 seconds, and complete launcher coverage is unverified. They do not close
the initialization-memory gap. The raw query and interpretation are retained in
the family evidence; no acceptance rule changed.

The fourth pair (`38601377`, BF16 seed 42, ON then OFF) passed with exact saved
states, all 300 numerical updates, resident rows and three fresh score sets.
Evaluation time fell **305.53 to 255.96 seconds (16.22%)**, native-process time
**609.59 to 562.28 seconds (7.76%)**, and complete cold-case time **1,145.92 to
1,052.49 seconds (8.15%)**. Both first reached the target at update 100; native
publication clocks were 274.16/249.52 seconds, excluding separate launcher costs.
Sampled owned-GPU peak remained **31.516 GiB**. All 1,595 host observations,
eight stage summaries, 248 retained files and memory gates passed independent
replay. Short unsampled phases and pre-observer startup remain outside the bound.
The allocation cost 2,545 GPU-seconds, including a separate 285.36-second CPU
state comparison.

All four pairs pass the original aggregate timing rule on build `cbc950f`:

| Precision | Evaluation reduction | Native-process reduction | Complete cold-case reduction |
|---|---:|---:|---:|
| NF4 | 14.26% | 9.96% | 7.74% |
| BF16 | 15.50% | 7.68% | 8.12% |

These use the two-pair geometric mean in each precision, with reversed order for
seed 42. Within-pair GPU/CPU placement matched; physical placements differ across
allocations and filesystem-cache warmth was uncontrolled. The four allocations
cost **16,139 GPU-seconds**, including all fresh scoring and separate comparisons.
The original hash-bound timing-only decision retains **`promote=false`**. The
subsequent same-build recovery and current-`e64a735` compatibility below supply
separate evidence. These are shared execution gains, not improved learning,
NF4 superiority over BF16, new quality confidence intervals, or inference
acceleration. Exact receipts and clocks remain in the existing family evidence.

The first mature-recovery allocation (`38603421`) failed before starting a
resumed update, consuming **88 GPU-seconds**. Its launcher passed `fsw` reference
paths while the accepted decision recorded canonical `fs11` paths; separate
container mounts defeated literal resolved-path identity. A CPU reproduction
with identical bytes confirms the predicate failure. The correction dispatches
the exact recorded paths and leaves the 48-file payload, native source, recovery
comparisons and timing results unchanged. No recovery or memory pass is inferred
from this failed attempt.

Corrected allocation `38604682` completed on the same sealed `cbc950f` build,
consuming **1,345 GPU-seconds**. Both precision arms resumed their own enabled
update-200 checkpoint through update 300. All 100 numerical update records,
761 trainables, optimizer tensors, scheduler/update state, RNG, data cursor and
evaluation history matched the uninterrupted reference exactly. Fresh checkpoint
and portable-policy evaluations each reproduced all 160 resident task rows.
The 224 language factors, 537 trained head tensors and 112 language targets
remained unchanged in membership.

All **908 recorded host samples** and all six model-stage memory checks passed.
Sampled recovery-workflow owned-GPU peaks were 31.092 GiB NF4 and 32.178 GiB BF16;
host PSS peaks were 29.739/29.333 GiB. These are recovery-workflow lower bounds,
not replacements for the paired training peaks. Prerequisite/reference checks
preceded the observer, and tiny unsampled phase intervals remain explicit.
Cold case clocks through verified step release were 654.730/677.953 seconds.
The restored target crossing retains `seconds=null`; these recovery clocks do not
reconstruct uninterrupted time to quality. Independent replay verified all 265
retained files, raw memory records, small controls and fresh scores; the full
model/optimizer tensor comparison ran on the allocated node. The original
88-second failure and timing-only decision remain immutable. Current-`e64a735`
compatibility is established separately below.

Current-build job `38606005` then passed all six native stages on `e64a735`:
two updates, own-update-1-to-2 recovery and fresh two-row portable evaluation in
each precision. Both arms exactly matched the existing same-build disabled-cache
controls at updates 1 and 2, and their own enabled checkpoints resumed exactly.
All 761 model tensors, 1,522 FP32 moments, RNG/data state and resident/fresh rows
matched under the declared comparisons. Only the enabled flag is normalized for
the observational OFF/ON comparison; production resume remains strict.

The cache recorded 6/6 hits/misses in each full run and 2/2 in each recovery and
fresh process, retaining 2,214,696 bytes. All 885 recorded host samples and six
stage summaries passed. The qualification workflow took 1,197.212 seconds and
cost 1,269 GPU-seconds; sampled owned-GPU/host-PSS peaks were 32.648/33.874 GiB.
These are correctness-workflow clocks and lower-bound peaks, not another speed
comparison. Earlier startup and tiny unsampled intervals remain outside coverage.

For current `e64a735` full-head rank-32/alpha-64 NF4 or BF16 LoRA, add
`rl.reuse_gr00t_evaluation_backbone=true` to the native training command and retain
that setting when resuming. This opt-in is qualified on one A100 80GB with the
documented one-row, two-camera LIBERO geometry. Portable reload with reuse is
covered. The separate standalone-artifact checks used reuse disabled; combining
reuse with the rejected DiT head-adapter profile remains unqualified. The complete
300-update timing study and mature recovery retain their `cbc950f` identity;
this short current-build test does not create a new mature or quality result.

A source review also verified that all 29 update/recovery contract files are
byte-identical between `cbc950f` and `e64a735`. The added head-adapter path is
disabled for this full-head configuration; it creates no additional parameters,
optimizer entries or RNG stream. Both recorded comparisons explicitly use LR
`1e-5`, despite the changed starter default. This supports reusing the completed
`cbc950f` mature recovery as mechanism evidence alongside the current-build exact
checks. It does not establish an observed `e64a735` 200-to-300 run or permit
cross-build resume. The source comparison is retained in the existing family evidence.

For the official seed-17 control `38546671`, strict v11 memory acceptance passed
across the whole observed workflow and all four model stages: all 768 host samples were complete and owned-GPU attribution
was valid. Sampled peaks were 38.254 GiB owned GPU memory and 29.019 GiB host PSS;
these remain lower bounds and exclude disposable preflight peaks. The complete
case took 1,096.554 seconds, including 7.479 seconds of verified owned-step cleanup;
the allocation consumed 1,106 GPU-seconds. Native699 fresh scoring has no allocator
phase records, and no new 300-update mature recovery or rollout was measured.

The original attempt `38546021` failed before evaluation or training because its
qualification wrapper inserted a null adaptation field that the native official
identity intentionally omits. The narrow, actual-metadata-tested v7 correction
left native699, data, recipe and all other validation gates unchanged. The failed
attempt's 156 GPU-seconds and receipts remain separate. Exact successful and failed
receipts, including independent replay, are retained in the
[family evidence](evidence/native_qlora_family_engineering_4cba88f.json).

This is not a reproduction of the published LIBERO training benchmark. The pinned
[upstream Spatial recipe](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/examples/LIBERO/README.md#fine-tune-libero-spatial)
uses eight GPUs, global batch 640 and 20,000 updates from the generic
`nvidia/GR00T-N1.7-3B` base on the IPEC Spatial no-noops dataset. The present study
continues an already LIBERO-specialized checkpoint on the separately identified
HuggingFaceVLA export. Its outcome cannot be compared directly with the published
success rate or attributed to quantization without the matched BF16 LoRA arm.

## Native training and recovery

Use pinned Isaac-GR00T source `51d4c89f72fda44cbf77285c6a8114b52676b8a1`, Transformers 4.57.6,
and bitsandbytes 0.49.2 for NF4. Construction verifies the upstream source digest and content-bound
checkpoint/processor identity. No missing-kernel, dense-weight or alternate-backend fallback occurs.
The shared [native action source profile](../../docker/posttrain/native_action/README.md)
reproduces the retained CUDA 12.8 package versions for both GR00T and Alpamayo without
changing Qwen's runtime:

```bash
python docker/posttrain/native_action/install.py \
  --uv /path/to/uv-0.11.29 --venv /durable/venvs/cosmos-native-action
source /durable/venvs/cosmos-native-action/native-action-env.sh
```

The initial native strategy is one CUDA process, BF16-capable hardware, no FSDP/DTensor/TP/PP/CP,
no gradient accumulation and no compilation. Only configurations with completed accelerator and
task evidence can be described as qualified. Other embodiments, model revisions and online PPO
are outside this initial integration. The starter below uses 100 updates and LR `1e-5`;
the selected development comparison uses the explicit 300-update settings
in the first-target stopping section below. These are different recipes. Commit
`40dcf52` changes only the starter LR after the completed paired study; it does not
promote the failed quality guard or alter the separate official control. Reproduce
historical `1e-4` starter runs with `rl.optimizer.lr=1e-4`; resume must retain the
original resolved optimizer settings and compatible source build.

```bash
export IMAGINAIRE_OUTPUT_ROOT=/durable/path/gr00t-runs
export PYTHONHASHSEED=17 CUBLAS_WORKSPACE_CONFIG=:4096:8

torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_sft \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py -- \
  model.checkpoint_path=/staged/GR00T-N1.7-LIBERO/libero_spatial \
  model.backbone_adaptation.backbone_path=/staged/Cosmos-Reason2-2B-metadata \
  model.backbone_adaptation.mode=nf4_lora \
  data.train_path=/prepared/gr00t/train.jsonl \
  data.validation_path=/prepared/gr00t/validation.jsonl \
  rl.max_steps=100 rl.evaluation_every=25 rl.save_every=25 rl.seed=17
```

Prepared JSONL rows require `id`, independent `episode_id`, `observation`, `actions` and `media`.
The loader binds every row and image/packed-media payload to the data cursor identity, rejects
train/validation episode, row or media overlap, and detects source mutation. It does not establish
independence between semantically related episodes; provenance screening remains necessary.

The entrypoint uses the existing `LLMTrainer`, flow-BC objective, optimizer/scheduler and DCP
publication. Resident evaluation preserves all RNG streams and mixed module modes, uses a separate
validation traversal and leaves the training cursor unchanged, including when scoring raises.
Each claimed target is checked after checkpoint publication/readback. Set target metric/threshold
before measuring outcomes; no target is inferred from candidate performance. Resumed runs explicitly
report process-only timing and do not reconstruct missing uninterrupted clocks.
Outputs live at `IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/<run_ts>`;
the resolved run's `completion.json` records its committed `checkpoint` and `portable` paths.
Portable publication is part of native completion, not a separate adapter-only export step.

For a deliberate recovery check, keep `rl.max_steps` fixed and first set `rl.pause_after_step=50`.
Rerun against the same native run identity with `rl.pause_after_step=0` and `rl.resume=auto`.
Only committed optimizer boundaries are resumable; a mid-update interruption restarts from the last
committed checkpoint. Optimizer, scheduler, RNG, data cursor and evaluation history are restored.
No simulator state is captured, so this is not a claim of exact online rollout continuation.

Build `cbe368d` additionally binds 25 explicit training/recovery source files in the
checkpoint's training contract. A changed or missing source contract refuses resume
before data preparation and model construction; the checkpoint codec checks it again
before restoring state. Resume older checkpoints with their original qualified build.
Do not edit metadata to bypass this restriction. Portable inference reload remains a
separate contract and accepts compatible older artifacts. The guard has 67 passing
CPU lifecycle tests. GR00T full-model five-to-ten recovery subsequently passed on
`26bc60f` and `02f93ba` within the mode scopes recorded below, separately from the
retained `4cba88f` mature recovery evidence.
The current inventory has 29 files: it also binds shared flow-dataset row lookup,
media decoding and both opt-in evaluator helpers. Those execution paths can change
without changing dataset bytes; evaluator reuse remains off by default.
Earlier 25-file contracts still require their original build; they are not rewritten
or upgraded during resume. This extension does not bind portable inference to training code.

`cbc950f` changes the GR00T task implementation identity, including with reuse disabled.
Training resume and export of historical artifacts require their recorded compatible
builds. Native inference has only the narrow reviewed exception described below;
other historical configurations still require their original loaders. Never edit
source-identity metadata to force an older policy through a new loader.

Current `e64a735` default full-head checks passed in `38563905`: both BF16 and
NF4 completed two updates, exact own-update-1 recovery and fresh two-row portable
reload with evaluation reuse disabled. The optimizer contained all 224 language
factors and 537 FP32 auxiliary tensors, with 1,522 FP32 moment tensors and no
`head_adapters` record. Model and optimizer differences were zero; RNG, data
cursor, evaluation history and the resumed update matched exactly. Frozen-base
validation and fresh source/scorer identities passed. All six model stages and
726 observed host collections passed; short unsampled phases remain recorded.
The allocation cost 1,054 GPU-seconds. This closes current-source default training,
recovery and portable reload coverage; it does not repeat mature recovery or
establish learning/performance improvement.

Current-default standalone export subsequently passed in `38578617` on the same
`e64a735` implementation. Both own-update-2 BF16/NF4 portable policies reproduced
their two native action-score rows, controlled task outputs and processor inputs
after export and fresh loading without the original checkpoint paths or quantizer
imports. Each artifact retained all 761 trained tensors, 112 effective backbone
matrices and required native sidecars: 21 indexed shards, 10,247,208,358 bytes.
The 512-MiB payload limit permits three indivisible oversized tensors per artifact.
The shared Alpamayo/GR00T allocation completed all twelve stages and cost 2,806
GPU-seconds; all 1,818 observed host/GPU samples passed independent replay, while
22 short unsampled intervals remain explicit. Native scoring/publication clocks
and extra integrity-capture costs are recorded separately in the
[family evidence](evidence/native_qlora_family_engineering_4cba88f.json).
This closes current-default unmerged reconstruction coverage, not merged parity,
full-cohort quality, mature recovery or deployment-speed qualification. The earlier
35-second pre-model UID-check failure remains a separate failed allocation.

## Fresh-process evaluation

The native evaluation and rollout loaders now have a narrow inference-only path
for the selected full-head rank-32/alpha-64, zero-dropout native64 artifacts. It
accepts only the reviewed native64-to-e64 model-source closure, unchanged LIBERO
task settings and the matching adapter seed/runtime. Base, processor, tensor and
trainable identities remain strict. The same check covers standalone unmerged
GR00T artifacts; it does not enable legacy merged loading, export or cross-build
training resume. GR00T evaluation-backbone reuse must remain disabled for this
transition. Evaluation records the compatibility identifier without rewriting the
artifact identity. CPU contract tests pass; current-build accelerator comparison
against the retained selected policies is still required before claiming parity.

Every checkpoint/portable policy includes all trained action-head/projector tensors alongside LoRA,
with source, model, quantization, processor/statistics, embodiment, action settings, resolved targets,
implementation hashes and dtype identities. Portable reload reconstructs the same quantized base.
Use a new output directory for a fresh-process native action evaluation. This
example specifies the selected seed-17, rank-32/alpha-64 NF4 policy explicitly;
use the artifact's actual settings for another trained policy:

```bash
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_eval \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py \
  --checkpoint /run/checkpoints/iter_000200 \
  --output /durable/path/fresh-action-evaluation -- \
  model.checkpoint_path=/staged/GR00T-N1.7-LIBERO/libero_spatial \
  model.backbone_adaptation.backbone_path=/staged/Cosmos-Reason2-2B-metadata \
  model.backbone_adaptation.mode=nf4_lora \
  model.backbone_adaptation.rank=32 model.backbone_adaptation.alpha=64 \
  model.backbone_adaptation.dropout=0 model.backbone_adaptation.seed=17 \
  model.backbone_adaptation.train_action_head=true \
  model.denoising_steps=4 model.training_prefix=10 model.execution_horizon=8 \
  data.validation_path=/prepared/gr00t/validation.jsonl \
  rl.seed=17 rl.evaluation_seed=123 rl.evaluation_max_samples=0 \
  rl.reuse_evaluation_inputs=false rl.reuse_gr00t_evaluation_backbone=false
```

The same command accepts the portable path printed in `completion.json`. It verifies the commit,
parameter inventory and complete numerical identity before loading trained components. It reports
masked normalized-action MSE, per-component decoded environment-action MAE and native teacher-forced
flow loss separately. Prediction never receives demonstration actions as RTC conditioning. Generic
HF language-model loading and serving acceleration are not established by this portable loader.

## Standalone artifacts

The qualified standalone representation is **dequantized and unmerged**. It stores the actual
BF16 effective base, separate FP32 LoRA factors, all trained action components and native sidecars.
Native loading preserves the separate adapter GEMMs and cast order without reading the original
base or running a quantizer. Standard indexed shards have a configurable payload limit (512 MiB
by default; a larger individual tensor gets its own shard). The commit-last manifest binds tensor
and file digests, processor, embodiment, normalization statistics and backbone-tokenizer metadata.

```bash
# First create and activate the exact build provenance using the source-profile guide.
: "${COSMOS_TRAINING_CODE_REVISION:?Set this from the actual build provenance}"
: "${COSMOS_TRAINING_WHEEL_SHA256:?Set this from the actual wheel digest}"
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.checkpoint.tools.export_family_dense \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py \
  --checkpoint /run/checkpoint_artifacts/iter_000200 --output /durable/gr00t-unmerged \
  --unmerged --max-shard-size-bytes 536870912 \
  --override model.checkpoint_path=/staged/GR00T-N1.7-LIBERO/libero_spatial \
  --override model.backbone_adaptation.backbone_path=/staged/Cosmos-Reason2-2B-metadata \
  --override model.backbone_adaptation.mode=nf4_lora \
  --override model.backbone_adaptation.rank=32 --override model.backbone_adaptation.alpha=64 \
  --override model.backbone_adaptation.dropout=0 --override model.backbone_adaptation.seed=17

torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_eval \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py --unmerged \
  --checkpoint /durable/gr00t-unmerged --output /durable/unmerged-evaluation -- \
  model.backbone_adaptation.mode=nf4_lora \
  model.backbone_adaptation.rank=32 model.backbone_adaptation.alpha=64 \
  model.backbone_adaptation.dropout=0 model.backbone_adaptation.seed=17 \
  rl.evaluation_seed=123 data.validation_path=/prepared/gr00t/validation.jsonl
```

The [native action source-profile guide](../../docker/posttrain/native_action/README.md)
contains the wheel-build and provenance-export commands. The example uses the selected
seed-17 NF4 update-200 artifact and its recorded compatible exporter/loader; use `bf16_lora` in both
commands for its BF16 control. Preserve the original mode, rank/alpha/dropout/adapter
seed and task configuration.
Source training metadata is preserved; the exporter build is recorded separately.
This representation was implemented on `8c64b5a`. CPU checks cover exact factors/heads,
aliases, interrupted publication, corrupt artifacts and fresh loading without source
weights or a quantizer. It is dense-base deployment, not four-bit inference, and
cannot resume training. The full-model qualification below identifies its tested scope.

The separate **merged** representation remains unqualified: its controlled-output gates
failed in both precision arms. It reconstructs `dequantize(Q(W)) + (alpha/rank) B A` with
FP32 CPU accumulation and declared BF16 or FP32 merged-language storage, preserving other
native dtypes. It uses `--merged-backbone-dtype` for export and `--dense` for loading;
these flags do not select the qualified unmerged graph. The representation flags are
mutually exclusive. Merging and any additional BF16 cast change finite-precision execution.
The negative results below are retained. Merging onto the untouched original BF16 base,
requantization, generic serving and deployment-latency improvements are outside this contract.

Full-model job `38522604` passed on exporter/inference build `64ea823` for both
selected `4cba88f` LR `1e-5`, seed-17, update-300 policies. Fresh native loading
exactly reproduced all 160 primary evaluation rows and controlled first/last-row
flow losses and action outputs, including input identities. All 537 FP32 trained
head tensors, 224 FP32 adapter factors, 112 BF16 effective backbone matrices and
the tied embedding/output-head ownership were preserved. The fresh containers
verified both original `fsw`/`fs11` checkpoint/backbone paths unavailable and
blocked quantizer imports before loading. The installed runtime still contains
bitsandbytes; this is not qualification of a bitsandbytes-free installation.

Each artifact contains 21 weight shards and 10,247,208,358 bytes across its
manifest-listed files. Three tensors exceed the 512 MiB payload limit and each
occupies its own shard, as declared by the format. The complete artifact-only
allocation consumed 1,240 GPU-seconds. Fresh loading/scoring peaked at 9.043 GiB
allocated and 9.695 GiB sampled GPU-process memory in both arms. All observed host
samples were complete; sampled peaks remain lower bounds between observations.
The [raw receipts and phase measurements](evidence/native_qlora_family_engineering_4cba88f.json)
separate source capture, export, loading, evaluation and publication costs. This
establishes fidelity for these selected checkpoints, not rollout quality or an
inference-speed improvement. The failed merged conversion gates remain unchanged.

Build `64ea823` additionally emits opt-in allocator records from native
`action_eval` when `COSMOS_QLORA_METRICS_FILE` is set. Loading includes portable
inventory verification; scoring, frozen-state verification and result publication
have separate reset-delimited intervals. The existing external sampler is still
needed for device allocations outside PyTorch and process-tree host memory.
The historical JSON workflow clock ends before result publication; the new phase
records expose that publication cost without rewriting old measurements.

## LIBERO rollout

### Prospectively generated Spatial scenes

The evaluation-only `libero-generated-scenes-v1` input can bind fresh draws from the
same ten Spatial BDDL task initializers. It does not reinterpret published `init_state_id`
values. LIBERO's authors describe evaluation states as independent draws from the
[same initialization distribution](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/2#issuecomment-1596101904)
and describe [repeated seeded initialization](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/34#issuecomment-2341628132).
The original generation seeds, subset indices and generation-time settling recipe are
not recorded. Fresh draws therefore do not reconstruct the released finite inventory.
They are conditional confirmation on these tasks, not new tasks/embodiments, offline
expert-action labels, or evidence about an entire training recipe. The pinned
`rlinf-libero==0.1.3` source-file manifest is authoritative; no exact wheel-to-Git
attestation is claimed.

Freeze the candidate/control set and explicit task/construction/reset seed list before
preparation. One manifest is shared by every paired policy, including both precisions
and native frozen-base controls. Preparation imports no policy or trained weights.
The source description hashes all ten BDDLs, LIBERO assets, native generation/rollout
code, and installed robosuite/MuJoCo/NumPy files (including Panda assets and controller
configuration), in addition to the existing pinned LIBERO runtime/source checks.

```bash
SIM_PY=/verified/libero-venv/bin/python
export MUJOCO_GL=egl
"$SIM_PY" -m cosmos_framework.posttrain.embodied.sims.libero.generated \
  --describe --output /protocol/generated-source.json
```

Create `/protocol/generated-plan.json` with these exact fields. `source` is the complete
description above; `requests` is the prospectively frozen list of objects with integer
`task_id` in `[0,10)`, `construction_seed` and `reset_seed` in `[0,2**32)`.
There is no default sample count, task subset, seed search or replacement seed.
`policies` is a distinct list of typed entries:

- `{"kind":"artifact","manifest_sha256":"<SHA256>"}` binds the committed portable
  `.posttrain-checkpoint-artifact.json` or the native standalone export manifest selected
  by `--unmerged`/`--dense`. Existing loader support limits still apply.
- A native BF16/NF4 `--base` entry is produced by
  `entrypoints.gr00t_rollout.base_policy_binding(identity)` from its existing complete
  untrained codec identity, retained in a source-policy receipt. It records the exact
  source, adapter/quantization/config and full identity digest; it creates no fake artifact.
  The loaded base must match that entry before policy actions. This generated mode does
  not yet accept the separate official-head source factory.

The plan's `format` is `libero-generated-plan-v1`; its other keys are `source`, `policies`,
and `requests`. Candidate-set bindings do not enter scene IDs. The independently versioned
`libero-generated-scene-policy-v1` noise derivation uses scene ID, policy seed and decision,
so paired policies share scene/noise coordinates without impersonating public-state seeds.

```bash
"$SIM_PY" -m cosmos_framework.posttrain.embodied.sims.libero.generated \
  --plan /protocol/generated-plan.json --output /protocol/generated-scenes.json
SCENE_SHA=$(sha256sum /protocol/generated-scenes.json | cut -d' ' -f1)
"$SIM_PY" -m cosmos_framework.posttrain.embodied.sims.libero.generated \
  --verify /protocol/generated-scenes.json --sha256 "$SCENE_SHA" \
  --output /protocol/generated-scenes-verification.json
# In the native rollout command below, replace --cases with:
# --generated-scenes /protocol/generated-scenes.json --generated-scenes-sha256 "$SCENE_SHA"
```

Each preparation request runs in a fresh simulator process with a bounded timeout.
Python/NumPy are seeded before construction, then reseeded for one call to the underlying
native reset. This bypasses the upstream wrapper's hidden retry loop. Exactly fifteen
existing open-gripper settling controls follow. The manifest records constructor and
pre/post-settling physics, compiled model/fixture transforms, controller/buffers, RNG
and observations. Evaluation recreates each scene in a fresh process and verifies these
records exactly before exposing its observation to the policy. It never restores only
flattened `time/qpos/qvel`, which omit fixture and controller state. Native task predicates,
20 Hz controls, eight-action chunks, action decoding and the 256-control limit are unchanged.

Before workers start, `<output>.attempts/header.json` durably binds the complete plan and
its canonical JSON digest. Each request gets create-only `NNNNNN.started.json` and
`NNNNNN.result.json` records; a result is durable before the next request starts. A started
request without a result remains uncertain after interruption; requests without a start
record remain identifiable from the header. Any existing attempt directory refuses reuse:
there is no resume, replacement-seed or automatic retry path. Keep the complete journal.

All requested seeds and failures remain in that journal. Only a complete, validated manifest
is published to `--output`, after serialization and fsync, using an atomic no-overwrite link.
Interrupted staging never leaves partial JSON at the final path. A failed/incomplete attempt
or duplicate physical scene refuses final publication. No success-based filtering or resampling
occurs. Duplicate physical identities are not counted as independent episodes.
Preparation and regeneration seconds measure construction/reset/settling and evidence capture;
the rollout workflow also includes fresh-process startup and source verification. Neither is
policy response latency. CPU fixtures cover these contracts; real pinned-runtime preparation,
fresh-process regeneration and policy/simulator acceptance remain required before use as
rollout evidence. No generated-scene campaign is qualified by this implementation alone.

The first allocated check (`38629190`) failed after **57 GPU-seconds**, before
model loading, EGL initialization or scene generation. Compute-side storage probes
passed, but the qualification measurement package shadowed the native tools package
and omitted `storage.py`. The retained 86-file archive includes the failure and
terminal release. Its mechanical successor adds the existing pinned module while
preserving all 32 original packet files, the native build and the experiment
protocol. Isolated import tests validate packaging; they do not replace the pending
physical execution. Earlier path-alias and GLVND-schema preflight failures consumed
no GPU time and remain recorded separately.

The corrected run (`38630422`, **299 GPU-seconds**) passed those startup checks
and completed two old-build public rollouts, then failed their exact trace gate.
For the first case, their first observation and eight controls matched. At the next decision, only
the wrist-camera image differed among observations; processed pixels and policy
actions subsequently differed. Both runs succeeded on both cases, but one reward
arrived at a different control step. This is not a session-UUID mismatch or a
QLoRA-versus-BF16 comparison. No new-build or generated-scene stage ran. The 126
retained files and independent replay locate the earliest recorded divergence
upstream of prediction; raw pixels and hidden physics state are needed to identify
its cause. The exact gate and historical outcomes remain unchanged.

The model-free diagnostic `38633630` failed after **116 GPU-seconds** when its
512-MiB capture limit refused another array during constructor-boundary storage.
It retained 330 arrays (281,170,600 bytes), initial RNG and two constructor renders,
but no complete constructor boundary, settling controls or replayed policy controls.
Native construction can itself reset the environment before the observation hooks
attach; zero recorded reset attempts does not prove construction performed no reset.
All 458 retained files and the terminal release passed independent verification.
The failed array's size and field path were not journaled. Source ordering points to
the full MuJoCo model blob, whose assets overlap the separately captured model fields;
this attribution is an inference, not a recovered descriptor. Lossless chunk-packing
of the retained arrays measured 68,608,936 bytes, but the complete six-process
capture footprint remains unmeasured. This failure supplies no trajectory-parity
result and does not weaken the preceding equality gate.

The lossless chunked successor (`38636235`, **854 GPU-seconds**) completed all six
fresh-process fixed-control replays and exact analysis. Each captured fifteen
settling controls, eight policy controls and 62 native render calls; all 21,776
retained files passed integrity verification. From the first seeded reset render
onward, recorded physical, controller, camera-pose, non-image cache and clock state
matched across repeat pairs. Python RNG remained different because the original
native seed call seeds NumPy only. Image bytes and their cache copies differed by
at most one RGB level. At the final unchanged physical state, repeated wrist
renders differed in 16/18 image pairs and agent-view renders in 2/18 pairs.
This locates variability in image production beyond the captured identical state;
it does not establish a particular GPU-driver, rasterization or readback cause.
No model ran, no tolerance changed, and the original rollout equality failure
remains a failure. Generated-scene qualification remains blocked on this issue;
fixed-input policy artifact and response checks are separate.

The separate timeout-diagnostic fix `1e33f58` retains partial worker stdout and
stderr in failed-request journals. Its 60 CPU tests include actual subprocess
timeouts in preparation and verification; they do not establish simulator parity.

Historical selected-update-200 policies require their compatible native loader.
The evaluation-only backport `fe72f850d9f6e3922bb1a3b1332549e3ab96c82d` on
`integration/libero-generated-native64` adds the same five scene-evaluation files
to native `64ea823`; both policies' nine bound model/shared source hashes, codec,
configuration and trainer remain unchanged. Its 102 CPU tests passed, but real
selected-policy/EGL qualification is still pending. This is not an identity
override. Generate a fresh source description and manifest for that build; a
manifest from the current build binds a different source closure.

### Published initial states

An observation-only native rollout adapter and `entrypoints.gr00t_rollout` also connect a loaded
portable or dense policy to the existing isolated LIBERO simulator gateway. The separate
`gr00t_qlora_spatial_8_v1` profile executes eight supervised actions, allows 256 control steps and
keeps the native 15-step reset; legacy spatial's 16-action protocol is unchanged. Canonical simulator
images are already oriented correctly, and the simulator alone binarizes/inverts native gripper
commands. The policy adapter performs neither conversion a second time.

```bash
# cases.json is a prospectively fixed list of {suite, task_id, init_state_id} objects.
# The simulator interpreter must satisfy the existing pinned LIBERO runtime and asset preflight.
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.gr00t_rollout \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py \
  --checkpoint /run/checkpoints/iter_000100 \
  --sim-python /verified/libero-venv/bin/python --cases /protocol/cases.json \
  --seed 17 --output /durable/rollout-seed17 -- \
  model.checkpoint_path=/staged/GR00T-N1.7-LIBERO/libero_spatial \
  model.backbone_adaptation.backbone_path=/staged/Cosmos-Reason2-2B-metadata
```

For the qualified standalone representation, add `--unmerged` and point `--checkpoint`
to the unmerged artifact. The separate merged `--dense` conversion remains unqualified. The separate official-head control uses this same rollout entrypoint with its
`gr00t_n1d7_libero_official_head.py` config and `data.backbone_path` override; it accepts committed
native or portable head artifacts, but rejects `--dense`. Each loader checks the pinned source
and complete native parameter/processor identity before applying trained tensors.
For a source-policy control, replace `--checkpoint ...` with `--base`. The matched recipe
requires `model.backbone_adaptation.mode=bf16_frozen` or `nf4_frozen`; the separate official-head
recipe loads its original FP32 source directly. This path creates no adapter/checkpoint artifact
and refuses LoRA configurations or `--dense`. The rollout receipt distinguishes source from
trained artifacts and records the complete immutable model/processor identity. Actual source
rollout outcomes remain a separate measurement from training quality.
The training runtime does not itself contain the pinned simulator stack. With the isolated
pinned LIBERO/EGL runtime, job `38510600` completed both predeclared engineering episodes for
each retained ten-update portable policy. Both precision arms had zero successes out of two.
This verifies native policy/simulator execution only; it establishes neither task preservation
nor deployment-latency improvement. The artifact-output probe in
`model/gr00t/qualification.py` separately captures actual normalized and decoded actions, native
flow loss and processed-input digests; its comparison requires explicit predeclared tolerances.

Source-policy job `38514573` subsequently completed those same two exposed cases
with both frozen precision arms and the separate original FP32 source: all three
achieved 2/2 successes. This confirms that the observation/action bridge can execute
successful policies on these cases; it does not qualify adaptation or a success-rate
margin. NF4/BF16 sampled process peaks were 11.088/12.139 GiB. Mean response times
after five warmup calls were 0.286/0.262 s, including native preparation and action
sampling but excluding simulator rendering/control and gateway IPC. Policy-dependent
observations and unequal call counts prevent a matched-input latency claim.
The earlier attempt `38513829` remains failed: its wrapper incorrectly rejected a
valid early success. The corrected wrapper accepts the recorded successful control
count, checks sampler health even on exceptions, and verifies the allocated CPUs.

The prospective `gr00t-development-v1` study compares `nf4_lora`, `bf16_lora` and `bf16_frozen`
with the same FP32 action-head policy, fused AdamW, learning rate `1e-4`, constant schedule,
effective batch one and training seeds 17/42/123. It fixes 300 committed updates, full resident
validation at updates 0/100/200/300, and publication at updates 100/200/300 (also update zero if it
already meets the target); there is no quality-based early stop.
Arm order rotates across seeds. The outcome-blind cohort contains all ten spatial tasks:
eight training episodes per task with sixteen time-spaced windows (1,280 rows), and four validation
episodes per task with four windows (160 rows). Its manifest SHA256 is
`c726e37725c2d4be88efb17e8f96b956d018a07db9f72bde1f6825084392dd00`.
All 2,880 source images are RGB 256 × 256 with verified PNG payloads/content hashes.
The source is `HuggingFaceVLA/libero@86958911c0f959db2bbbdb107eb3e17c5f9c798e`;
its [immutable metadata](https://huggingface.co/datasets/HuggingFaceVLA/libero/blob/86958911c0f959db2bbbdb107eb3e17c5f9c798e/meta/info.json)
declares 10 Hz. Cosmos preserves consecutive source action rows without temporal
resampling, but no retained upstream conversion attestation establishes their
physical clock. The pinned official and Cosmos LIBERO simulators use 20 Hz control.
These are separate recorded facts: do not infer 20 Hz dataset equivalence or convert
the ten supervised action indices into a claimed physical horizon. Source conversion
provenance remains required for that broader temporal claim.

A separate CPU provenance probe compared the already-exposed training episode
1347 with `physical-intelligence/libero@a4336d589d589045d1c56423ffdf3b88a0e19b1f`.
All 92 source state/action rows, indices and timestamps matched exactly; all 16
prepared training windows retained the same FP32 states and consecutive ten-action
sequences. The export's episode metadata points to the wrong parquet file for this
episode, but Cosmos enumerates the actual rows and does not use that locator.
The [scoped receipts](evidence/native_qlora_family_engineering_4cba88f.json) retain
both observations. This probe found no reason to resample actions; it does not
attest the original simulator clock or establish whole-dataset provenance.

A bounded follow-up inspected staged source locations and pinned upstream conversion
metadata. It found no authenticated raw-demo locator/controller configuration for
these 92 rows. Current regeneration code and a 20 Hz simulator default do not
establish the historical dataset clock. The next check requires that exact locator
and generation/conversion configuration, then an exact raw-state/action comparison.
No trajectory downloads, model scoring, resampling or recipe changes were made.
The pinned conversion publisher identifies OpenVLA as its source. A 1.915 GB raw
record scan could recover the episode locator, but the RLDS schema supplies no
controller clock; this scan remains deferred pending a way to establish that link.

A read-only coverage audit confirms that staged data is **not exhausted**. The
prepared spatial population contains 342 training and 90 validation episodes.
After excluding the recorded development and engineering selections, 248 training
and 45 validation episodes remain outside those selections. Simulator assets also
contain 400 initial states outside the admitted 100 cases. These are still the same
ten spatial tasks, not new independent task clusters; demonstration episodes and
simulator initial states are distinct units. Historical and pretrained-model exposure
is unknown, so “outside recorded selections” does not mean globally untouched.

The staged metadata also contains 454 object, 428 goal and 379 LIBERO-10 episodes
across 30 other tasks. Using those suites would change the confirmation population.
Independent confirmation requires a prospectively defined population and exposure
ledger, plus a separate pilot to verify native evaluation, source-policy usefulness
and paired variability before planning power under the declared margins and
task/seed clustering. This inventory
justifies no fixed confirmation sample count. The hash-bound
[coverage audit](evidence/native_qlora_family_engineering_4cba88f.json) preserves the
counts, source identities and limits without changing the current study.

A processor-only census of one real row per instruction covered all ten fixed instructions
in both splits: 151–156 tokens, `pixel_values=[512,1536]`, two image grids `[1,16,16]`,
state `[1,1,132]`, and action/mask `[1,40,132]` with exactly 70 supervised elements.
This used the pinned deterministic native processor without model weights or CUDA.
The remaining rows share these instructions and admitted image geometry; this establishes
processor shape coverage, not additional GPU training or quality coverage. Actual GPU evidence
remains the traced updates described below and completed portions of the 300-update study.

The first complete seed-17 study (`38511256`, native `4cba88f`) retained all three
300-update runs and exact fresh portable scoring on the full 160-row development set:

| Mode | Source normalized action MSE | Update 300 MSE | Native workflow | Sampled GPU-process peak |
|---|---:|---:|---:|---:|
| BF16 head-only | 0.055068 | 0.225794 | 665.66 s | 31.154 GiB |
| BF16 LoRA + head | 0.055068 | 0.378767 | 665.15 s | 31.516 GiB |
| NF4 LoRA + head | 0.055654 | 0.286100 | 626.78 s | 30.412 GiB |

All targets were unreached. Every task slice and all seven action-axis MAEs
worsened from the source; favorable NF4-versus-trained-BF16 means therefore do
not establish useful adaptation. Training loss also rose, and 296/296/295 of 300
updates were clipped respectively, with no skipped/nonfinite updates. A checked
audit found no new loss-normalization, mask, source-action truncation or cursor
defect. The next prospectively bounded learning screen reduces LR from 1e-4 to 1e-5
for all three arms while preserving the data, loss, heads, horizon and fixed target.
It must reach that target and satisfy the 5% paired precision point guard before
additional seeds are considered; this screen cannot establish recipe noninferiority.

The allocation included separate fresh scoring after each native workflow; its
per-arm totals were 1344.40/1240.98/1205.89 s in the table's order. These are one-seed,
fixed-order observations on one GPU, not a time-to-quality improvement. Requested
CPU allocation was 16, but measured pre-native process affinity was only two logical
CPUs. Future paired launchers must explicitly request and verify the step CPUs;
cross-study timing changes cannot be attributed solely to LR. The measured NF4
memory reduction was only 1.104 GiB versus matched BF16 LoRA because full FP32 head
training dominates. Host PSS sampling was incomplete during process teardown in
2/861, 2/850 and 1/872 samples; retain its lower-bound peaks and incomplete flags.
The [terminal evidence](evidence/native_qlora_family_engineering_4cba88f.json)
contains clocks, phase distinctions, task diagnostics, identities and failed targets.
Episode, row and media-content splits are disjoint.
These previously exposed episodes provide development evidence, not independent confirmation.

The bounded **LR `1e-5` screen completed** in job `38515136` on the same sealed
native `4cba88f` implementation. Reducing LR was the only learning change for all
three arms. The launcher also explicitly requested and verified 16 logical CPUs;
therefore timing differences from the earlier two-CPU run are not attributed to LR.
All 900 updates applied, with zero nonfinite/skipped updates, exact fresh portable
primary scores, and 112/110/106 clipped updates for NF4/BF16-LoRA/BF16-head.
Each arm first reached the unchanged target at update 200 after verified publication.

| Mode | Update 300 action MSE | Time to target, including launch preflight | Native workflow | Workflow with three fresh scores | Sampled GPU-process peak |
|---|---:|---:|---:|---:|---:|
| NF4 LoRA + head | 0.04154305 | 519.39 s | 620.85 s | 1191.01 s | 30.412 GiB |
| BF16 LoRA + head | 0.04244151 | 499.52 s | 588.84 s | 1147.34 s | 31.516 GiB |
| BF16 head-only | 0.03983719 | 432.40 s | 488.65 s | 1005.88 s | 31.154 GiB |

Updates 101–300 averaged 0.4913/0.4527/0.3162 s in the table's order. Native
workflow includes preparation, all scheduled validation/checkpointing and portable
export; the next column adds the fixed fresh-process scores at noise seeds
123/456/789 and wrapper costs. Only seed 123 supplies the primary quality metric.
The complete allocation consumed 3,358 GPU-seconds, including shared setup/teardown.
Source calibration and the earlier failed learning trial remain additional costs.

NF4's MSE point estimate is 2.12% below matched BF16 LoRA at the same 300 updates,
but its preflight-inclusive time to target is 3.98% longer. The head-only control
is stronger on both metrics. These single-seed, fixed-order development results
pass the prospective screen for considering further qualification; they establish
neither recipe noninferiority nor superiority at equal training compute. The sampled
1.104 GiB saving is only about 3.5% of the matched BF16 peak. The 805,306,368 eligible
language weights permit at most 1.125 GiB of raw BF16-to-four-bit weight saving before
metadata; full FP32 head training dominates the footprint. Substantial total-memory
savings remain an unmet GR00T requirement for this policy.

All three arms retain incomplete host-PSS samples during native process teardown
(2/851, 1/818 and 1/712 respectively); measured peaks remain lower bounds.
NF4's decoded gripper MAE improves, while its decoded x/y/z MAEs increase relative
to its source. This decomposition of decoded MAE does not decompose normalized MSE
or establish rollout behavior. The separately completed 100-case simulator check
below rejects promotion of this 300-update recipe. The previous LR `1e-4` failures,
unchanged quality margins and independent-confirmation requirement remain intact.

Selected-build recovery job `38521659` completed on the same `4cba88f` source:
both LoRA arms resumed from their committed update-200 checkpoints through update
300 with exact model/optimizer tensors, RNG, data cursor and evaluation state.
Only the new publication destination, identifier and manifest fields differ.
Fresh native checkpoint and portable loads preserved all 537 trained head tensors
and 224 LoRA factors and exactly reproduced all 160 primary evaluation rows and
metrics. Restored target crossings keep the original update-200 checkpoint and
report no reconstructed elapsed time. This is recovery evidence for the same
trained policies, not another seed or a change to the failed rollout decision.
The allocation consumed 1,331 GPU-seconds. Both resume stages retain incomplete
host-PSS sampling; all four fresh scoring stages have complete sampled host
receipts. Old-build `action_eval` emitted no allocator phase records, so those
fresh-scoring GPU/host peaks remain sampled lower bounds. The
[terminal receipts](evidence/native_qlora_family_engineering_4cba88f.json) preserve
these limitations and the separate resume allocator measurements.

Job `38518798` completed all 400 prospectively fixed rollouts: ten Spatial tasks,
ten initial states per task, seed 17, four flow-sampling steps, eight executed
actions per chunk and at most 256 controls. The same native simulator/reset and
observation-only policy path was used for every arm.

| Policy | Successes / 100 |
|---|---:|
| Original frozen BF16 source | 99 |
| Original frozen NF4 source | 99 |
| Trained BF16 LoRA + head, update 300 | 92 |
| Trained NF4 LoRA + head, update 300 | 93 |

The original additional-seed screen **fails**: trained NF4 loses six percentage
points against its own source, above the predefined two-point allowance. Its
one-point advantage over trained BF16 does not establish preservation: the paired
whole-task one-sided 95% upper loss bound is **3 points**. Corresponding bounds
against each arm's frozen source are 9 points for NF4 and 13 for BF16. These are
fixed-checkpoint development bounds from ten tasks, not recipe-level intervals
across training seeds or independent confirmation. The earlier offline-MSE screen
passed its own criteria; that result is preserved and does not override this
failed behavior gate. No further learning seeds are promoted from this endpoint.

The allocation consumed 3,733 GPU-seconds. All 2,780 host samples were complete;
trained NF4/BF16 sampled compute-process GPU peaks were 11.203/12.252 GiB.
Total-device peaks were also recorded separately. The
[retained evidence](evidence/native_qlora_family_engineering_4cba88f.json) includes
every case outcome, task-level comparison and response-time distribution.
On-policy observations and response counts differ, so these response timings do
not establish an inference acceleration claim. Offline time to the action-MSE
target above is not time to preserved rollout success.

The separate first-target stopping screen (`38523598`, 2,901 GPU-seconds) loaded
existing update-200 DCPs without retraining. All three arms had first met the
unchanged offline target at that update. On the same 100 development cases,
NF4 LoRA + head scored **98/100**, BF16 LoRA + head **97/100**, and the matched
BF16 trained-head-only control **97/100**. The prospective joint point screen
passes; keep first-target stopping as a development candidate. The original
update-300 failure above remains unchanged.

The paired whole-task bootstrap (10,000 draws, seed 6809) gives NF4 upper loss
bounds of **3 points versus BF16 LoRA** and **4 points versus its frozen source**,
which do not meet the unchanged two-point preservation guard. Both LoRA arms
improve five points over their own update-300 policies; their upper loss bounds
for that comparison are −2 points (NF4) and −1 point (BF16). BF16 head-only
loses two points against its source, with an upper bound of four points. These
are fixed-checkpoint development comparisons, not independent or paired-training-seed
qualification, superiority, or evidence of improved time to rollout quality.
Original offline target clocks are retained; no rollout target clock is reconstructed.

All 2,149 host samples were complete. NF4/BF16 LoRA sampled owned-process peaks
through checkpoint loading and rollout were 11.646/12.697 GiB; they remain sampled
lower bounds. The source-relative comparisons and all case outcomes are retained
in the [terminal evidence](evidence/native_qlora_family_engineering_4cba88f.json).
On-policy response inputs differ, so these timings do not establish inference
acceleration. The previously verified update-300 portable and standalone artifacts
validated their own tensors; selected update-200 publication and fresh scoring were
separate pending checks at that rollout decision. Their subsequent successful
qualification is recorded below.

To reproduce this screen through the native training command above, use the
admitted `gr00t-development-v1` JSONL files and these overrides in **each** arm:

```text
rl.optimizer.lr=1e-5 rl.optimizer.weight_decay=1e-5 rl.optimizer.grad_clip_norm=1.0
rl.max_steps=300 rl.evaluation_every=100 rl.save_every=100 rl.seed=17
rl.keep_model_last=0 rl.keep_optim_last=0
model.backbone_adaptation.rank=32 model.backbone_adaptation.alpha=64
model.backbone_adaptation.seed=17
rl.target_metric=normalized_action_mse rl.target_threshold=0.049561327725898344
rl.target_direction=min rl.stop_on_target=false
```

The numeric target is bound to that cohort and its recorded frozen-source scoring;
it is not a universal target for other data. Select `nf4_lora`, `bf16_lora` or
`bf16_frozen` explicitly. No optimizer-precision change is included.
Keep both retention settings at zero to preserve the update-200 optimizer and
control state for mature recovery after update 300. The local default otherwise
retains only the latest optimizer checkpoint.

Before either LoRA candidate runs, use native `action_eval --base` with `mode=bf16_frozen` and
`rl.evaluation_seed=123` on the locked validation cohort. Freeze the offline action-quality target
as `0.90 * M0`, where `M0` is that original source model's `normalized_action_mse`; record the
reference evaluation, identities and threshold before admission. Training uses
`rl.target_metric=normalized_action_mse`, the frozen `rl.target_threshold`, and
`rl.stop_on_target=false`. Count target time through verified checkpoint publication, retain
nonattainment at update 300, and charge the reference scoring cost explicitly. This is time to an
offline action-error target, not time to closed-loop success. Equal update counts are a controlled
work comparison, not proof of equal-compute quality or faster learning.

Prospective preservation guards are a one-sided 95% upper bound of 5% relative normalized action-MSE
increase and two percentage points of closed-loop success loss against the matched BF16 LoRA arm.
The first permits at most 2.47% relative RMSE amplitude increase; the second permits at most one
additional failure per fifty episodes. Neither is inherited from WTS answer accuracy. Weight tasks
equally, then episodes/windows equally. Use a paired crossed bootstrap of training seeds and tasks,
with episode resamples shared across all seeds and precision arms; repeated frames or scoring seeds
are not independent tasks. Three training seeds and ten task clusters may be insufficient to establish
these bounds. The separate planned rollout comparison uses ten fixed initial states per task under
the eight-action profile and must pass actual simulator preflight first. Improvement, preservation,
and useful adaptation over the source are separate claims; retain all failed trials.

For the selected LR `1e-5` continuation, the existing
`python -m tools.native_qlora_qualification.action_quality` command accepts
`--continuation-contract <gr00t-lr-paired-seeds-4cba88f-v2/execution-contract.json>`
alongside all nine `--run` inputs, `--validation`, `--manifest` and `--output`.
It requires the sealed contract digest `a916046a…`, preserves the five reused
results' exact bytes and original protocol identities, and refuses an incomplete
panel or altered target. The grouping, bootstrap, margins and update-300 endpoint
are unchanged. Seed 17 remains selection-exposed; seed-42 timing came from
different execution blocks and cannot become a new matched timing comparison.
Forty CPU tests and the five actual retained inputs pass this ingestion check;
they do not supply the four missing training outcomes or a recipe-level result.

Before inspecting dense-conversion outputs, its numerical gates are fixed at maximum absolute
error 0.02 on the valid 16 × 7 normalized actions and decoded controller commands, plus both
0.002 absolute and 2% relative masked flow-loss error (relative denominator floor `1e-6`).
The full 40 × 132 latent output is shape-checked and its unused-slot error reported separately.
Every decoded gripper command must preserve `sign(2*g-1)` exactly, and trained auxiliary tensors
and processed-input identities must match exactly. At the pinned OSC_POSE scales, a 0.02 command
difference corresponds to at most 1 mm position-command or 0.01 rad orientation-command difference
per step; this is a numerical budget, not a safety or rollout-quality claim. Same-quantized-base
portable reload retains zero numerical tolerances. A failed conversion gate is retained as a failure.

### Three-seed rollout preservation (step 3a, 2026-11-19)

Prospective protocol `3f7bca60…` (packet `gr00t-seed3-rollout-3c96ce8-v1`), fixed before any
outcome: the selected LR `1e-5` recipe's update-200 NF4 and BF16 LoRA DCPs for training seeds 17,
42 and 123, plus both frozen sources, each rolled out once on the 400 previously unused published
LIBERO-Spatial cases (tasks 0-9 x initial states 10-49), rollout seed 17, inference build `3c96ce8`,
settings identical to job 38523598. Every job admitted all six checkpoints (inventory, adapter
seed, identical training contracts) and re-hashed them before loading. Primary guard: one-sided 95%
upper NF4-vs-BF16 LoRA success loss <= 2 points, paired crossed training-seed x task bootstrap
(10,000 draws, seed 6809).

| Policy (job) | Seed 17 | Seed 42 | Seed 123 | Mean success |
| --- | ---: | ---: | ---: | ---: |
| NF4 LoRA (38705842/46/48) | 391/400 | 382/400 | 392/400 | 97.08% |
| BF16 LoRA (38705843/45/49) | 391/400 | 387/400 | 392/400 | 97.50% |
| NF4 frozen source (38705851) | | | | 391/400 = 97.75% |
| BF16 frozen source (38705853) | | | | 395/400 = 98.75% |

| Comparison | Point loss | Upper one-sided 95% loss | Margin |
| --- | ---: | ---: | --- |
| NF4 vs BF16 LoRA (primary) | 0.42 pt | **2.08 pt** | <= 2.0: **not met** |
| NF4 LoRA vs own frozen source | 0.67 pt | 2.58 pt | reported |
| BF16 LoRA vs own frozen source | 1.25 pt | 3.42 pt | reported |
| Frozen NF4 vs BF16 source | 1.00 pt | 2.00 pt | reported |

**The rollout preservation guard is not met** and remains a failure. The point loss is small and
comes entirely from training seed 42 (seeds 17 and 123 tie exactly), but three seeds and ten tasks
leave the bound just over the margin. Neither LoRA arm improves closed-loop success over its own
frozen source on these near-saturated tasks. Scope: exposed development tasks, update-200
checkpoints, three training seeds; not independent confirmation or real-robot behaviour.

All eight rollouts passed acceptance. The in-job analysis (38705853) failed only because
`sbatch --export` splits on commas and delivered one of seven job ids; the unchanged
`run.analyze` was then run over all eight acceptance files (analysis SHA-256 `1ec98abf…`).

## Completed native engineering checks

The [sealed `4cba88f` engineering receipts](evidence/native_qlora_gr00t_engineering_4cba88f.json)
record ten real LIBERO action updates in both precision arms on A100 80GB. Job
`38507341` passed exact fresh-process checkpoint and portable-policy task scores,
and update-5-to-10 recovery of model, optimizer, RNG, data cursor and evaluation
state. BF16's uninterrupted reference comes from job `38506697`; its original
outer qualification checker failed after training completed. The later checks
verify those retained artifacts and do not rewrite that job's failed status.

The small engineering set's normalized action MSE increased from 0.04976 to
0.17405 for NF4 and from 0.04870 to 0.15470 for BF16. This adverse ten-update
result establishes neither useful adaptation nor recipe quality. Warm update
means over steps 3–10 were 0.495/0.445 seconds, and native workflows took
142.45/143.80 seconds. Different placement and one short trial prevent a speed
claim; startup, scoring and checkpoint I/O dominate these workflows.

The trainable FP32 action head dominates total memory. The measured allocator
reduction is approximately 1.1 GiB, much smaller than Qwen's savings. Historical
external GPU-process samples are missing: Torch 2.10 returned a bare UUID and
the sampler expected NVML's `GPU-` prefix. Native allocator and process-RSS
records remain valid. UUID normalization and sampler-health checks must pass
before the larger comparison's memory claims; missing historical samples cannot
be recreated. Full development, dense conversion and rollout qualification remain
separate gates.

The subsequent [family engineering receipts](evidence/native_qlora_family_engineering_4cba88f.json)
preserve an actual process loss during update-six backward in both arms. Partial publication
was rejected, and restart from committed update five reproduced model, optimizer, RNG, cursor,
evaluation state and task scores at update ten exactly. Job `38507907` later failed before its
profile began because the launcher precreated the profiler's output directory; that job remains
failed and its completed recovery checks remain valid.

Dense BF16 conversion **failed** the unchanged controlled-output gates in job `38508290`.
NF4 had a 0.10645 maximum normalized-action difference on one probe and a 0.002462 flow-loss
difference on another; the BF16 control's two action maxima were 0.03418 and 0.03711.
The limits remain 0.02 action error and both 0.002/2% flow-loss error. All auxiliary tensors
and tied-weight aliases were preserved, and fresh native evaluation completed, but small mean
task-metric differences do not override failed probe gates. Both approximately 10.17 GB dense
artifacts and negative decisions are retained. Native portable reload and dense
conversion are separate capabilities.

Follow-up `38514034` isolated the conversion difference: dequantized but unmerged
execution matched the portable policy exactly on both probes and common-input
layer replays; all 112 merged matrices exactly matched the declared FP32 accumulation
followed by BF16 cast. Auxiliary tensors and standalone reconstruction were also exact.
This implicates changed finite-precision arithmetic in the observed drift, rather
than missing task state or a different quantized base. A bounded FP32 merged-compute
prototype (`38514818`) also failed the unchanged gates in both arms and increased
the sampled diagnostic process peak from 12.316 to 13.826 GiB. It is rejected;
no FP32 execution option is promoted. These two diagnostic allocations cost
1,037 GPU-seconds and do not constitute a training or serving benchmark. The
[retained decisions](evidence/native_qlora_family_engineering_4cba88f.json) preserve
every failed probe. GR00T merged-artifact fidelity remains unresolved.

Paired profile job `38509185` completed 24 native updates per arm on the same A100
and CPU allocation, tracing updates 21–24 after warmup. Actual inputs contained
151–156 tokens, two images packed as `[512,1536]`, and 70 supervised action elements
in `[1,40,132]` tensors. Gradient-finiteness validation consumed approximately
104.7/106.4 ms of traced NF4/BF16 update wall time; clipping was another distinct
region. The unused vocabulary projection took only about 0.59 ms of GPU time per
update. CPU and GPU regions overlap, and trace overhead prevents treating these
numbers as production throughput. This motivated a read-only batched gradient
check, shared by both precision arms and subject to correctness and unprofiled
timing gates. Its final rejection is recorded below.

That profile measured sampled owned-process GPU peaks of 30.404/31.490 GiB and
native training allocator peaks of 27.221/28.334 GiB. These new samples are valid;
they do not recreate the missing earlier samples. The 1.62-billion-parameter FP32
action head and its gradients/moments dominate memory. The approximately 1.1 GiB
saving remains modest, so substantial GR00T memory savings and task-quality/time-to-target
benefits remain open requirements.

The source-policy calibration scored all 160 development rows before candidate training.
Frozen BF16/NF4 normalized action MSE was 0.05506814/0.05565446, respectively; these are
point estimates without a preservation claim. The unchanged target rule fixes the common
threshold at **0.049561327725898344**. Calibration jobs `38509654`/`38509843`, target identity
and exact metrics are retained in the family evidence. The seed-17 three-arm study uses that
target; later outcomes cannot redefine it.

The batched gradient-check candidate `3c96ce8` passed 22 real CUDA cases on each of the
Torch 2.10 family and Torch 2.11 Qwen runtimes in job `38510926`, with no skips. Those tests
cover finite/NaN/Inf detection, unmodified gradient bytes/RNG and tiny native updates/recovery.
Full-model GR00T gate `38517306` subsequently passed exact model, optimizer,
RNG/data/control and task-score comparisons with `4cba88f` at updates five and ten
in both precision arms. Each candidate arm also passed actual process loss during
backward six, rejection of a partial save, and exact own-build recovery from five
to ten. Compatible fresh-artifact evidence was reused only through exact trained
tensors and unchanged native model/scorer/codec hashes; this gate does not claim
a new fresh-process reload. It consumed 1,403 GPU-seconds, with incomplete host-PSS
sampling retained. Alpamayo full-model parity and sustained unprofiled workflow
benefit remain promotion gates; selected family training still uses `4cba88f`.


The first sustained timing attempt (`38527266`) failed before model loading, using
66 GPU-seconds. Python's native-source working directory shadowed the separately
hashed measurement package; the observer identity guard correctly refused it.
The versioned dispatch correction sets Python safe-path mode and explicit observer
sibling imports for both native builds and the final comparison. All scientific,
training and observer code, target and acceptance gates remain unchanged. Sixty-one
CPU checks reproduced the failure and verified corrected parent, child and monitor
imports. Replacement `38528318` completed as described below; the original failed allocation remains retained.


The seed-17 replacement completed all four 300-update cases and exact 100/200/300
model, optimizer, RNG/data and task-score comparisons. Every case had complete
owned-GPU and host-PSS evidence. The candidate reduced mean steady update time
from 0.4890 to 0.4382 seconds for NF4 (10.38%) and 0.4627 to 0.4104 seconds for BF16
(11.31%). Native workflow time fell 2.93%/3.58%; both precisions first met the same
offline target at update 200, with native-clock NF4 time 435.72→421.76 seconds and
BF16 time 406.12→398.85 seconds. These are one-seed engineering observations;
NF4 still took longer than optimized BF16 and did not improve its fixed-step
quality. Candidate GPU-process peaks still differed by only about 1.10 GiB.

The complete-case timing claim is **unqualified**: Slurm returned from one
candidate `srun` about 64 seconds before releasing its step, shifting that cleanup
wait into the next reference case. The original NF4 1226.58→1149.48-second and
BF16 1129.17→1108.43-second case clocks are retained, without retrospective
reassignment or promotion. The allocation used 5,149 GPU-seconds; its separate exact-state comparison step
recorded 522 seconds. A prospectively corrected launcher binds
the actual owned step and waits for its verified release inside the producing
case clock, with bounded failure handling. Forty-nine focused CPU checks,
including the actual Enroot UID map and controller responses, supplement the
unchanged 61 import/measurement checks. Fresh 17/42 blocks under that measurement
contract are required; no old/new timing blocks may be combined. Candidate-owned
mature recovery and the remaining family gates are still separate requirements.

The first-target update-200 policies subsequently passed native portable publication
and fresh full-160-row scoring in job `38531405` (686 GPU-seconds). NF4 and BF16
LoRA artifacts preserved all 224 adapter factors and 537 trained action-head tensors;
the separate BF16 head-only artifact preserved its 537 trained tensors. Every value
matched the original checkpoint, all original control files remained unchanged,
and all three fresh task-score rows/metrics matched their retained update-200
references. This closes selected-checkpoint portable coverage without creating new
quality evidence or reconstructing the earlier target clock. The complete nine-stage
workflow had valid owned-GPU sampling and 489 complete host samples; its sampled
process/host peaks are lower bounds, since the original `4cba88f` evaluator does not
emit allocator-phase peaks.

The selected NF4 and BF16 LoRA update-200 artifacts then passed standalone unmerged
qualification in job `38532260` on build `64ea823` (1,222 GPU-seconds). Fresh native
loading exactly preserved all 761 trained tensors, 112 effective BF16 matrices,
tied ownership, controlled first/last-row outputs and all 160 primary score rows.
The original base paths were absent under both filesystem spellings and quantizer
imports were blocked. Each artifact has 21 weight shards and 10,247,208,358
manifest-listed bytes, including three oversized singleton tensors. Native export
took 145.70/149.65 seconds for NF4/BF16; fresh full-160 subprocesses took
145.18/143.93 seconds. Both fresh reloads peaked at 9.043 GiB allocated, 9.174 GiB
reserved and 9.695 GiB sampled GPU-process memory. All 795 observed host samples
were complete; sampled peaks remain lower bounds.

This closes standalone coverage for the actual selected update-200 LoRA checkpoints.
It preserves native separate adapter arithmetic and is dense-base inference, not a
merged or four-bit serving result. The historical merged failures remain unchanged;
later GR00T current-build lifecycle checks are recorded separately below. The
[raw hashes, exact source bindings and phase measurements](evidence/native_qlora_family_engineering_4cba88f.json)
do not establish new task quality, rollout success or inference acceleration.

The repaired timing block `38532930` completed four fresh 300-update cases on
seed 17, with verified release of each producing Slurm step included in its own
clock. NF4 complete-case time changed from 1173.81 to 1163.55 seconds (0.87% lower)
and BF16 from 1116.96 to 1104.33 seconds (1.13% lower). Steady-update time improved
7.39%/10.41%, and native workflow time improved 1.34%/3.53%, respectively.
Both arms still first met the unchanged target at update 200: preflight-inclusive
times were 514.77→509.01 seconds for NF4 and 480.89→476.08 seconds for BF16.

All six 100/200/300 checkpoint comparisons were exact for model, optimizer,
RNG/data and task outputs. Independent replay verified 201 receipt files, all
3,225 raw memory samples and four owned-step release proofs. Allocator increases
were 5.5/4 MiB and sampled owned-process increases 48/44 MiB for NF4/BF16, within
the fixed 256 MiB limits. The allocation consumed 5,084 GPU-seconds, including
513.92 seconds of separate post-workflow CPU comparisons.

The first reference BF16 case shared its host with an Alpamayo allocation on
different GPUs and disjoint CPU sets; the later cases had no known such overlap.
This asymmetric placement limitation remains, without clock correction or an
isolation claim. All mandatory first-block gates passed; promotion then remained
conditional on the reversed seed-42 block, joint geometric thresholds and further
qualification. That block used identical settings and retained its native outputs
for development analysis. Old `38528318` clocks remain separate. None of these
arithmetic-preserving results repairs the unmet quality guards or establishes
quantization superiority; NF4 still takes longer than optimized BF16 and saves
only about 1.10 GiB of training GPU memory in this head-training recipe.

The reversed seed-42 block `38535185` completed all four cases and six exact
100/200/300 model, optimizer, RNG/data and task comparisons, consuming 5,126
GPU-seconds. Independent replay verified all 204 receipt files, 3,323 raw memory
samples and four producing-step release proofs. Its separate CPU comparison
step took 459.42 seconds. Each case had complete owned-GPU and host-PSS evidence;
the sampled peaks remain lower bounds.

| Observed reduction across the two fixed blocks | NF4 | BF16 |
| --- | ---: | ---: |
| Geometric mean steady-update time | 8.682% | 8.908% |
| Geometric mean complete-case workflow | **0.798%** | 1.754% |
| Geometric mean native workflow | 2.623% | 2.262% |

The NF4 complete-case result **fails the unchanged 1% requirement**, so the
batched finite-check intervention is discarded despite exact numerical results
and faster updates. Commit `26bc60f` restored the original elementwise predicate,
removed the grouped helper and its runtime allowlist, and retained general
finite-check regressions. No further timing repeat or candidate-owned mature
recovery was run. Historical CUDA and full-model candidate evidence remains
available; it does not describe the selected implementation.

Both precisions first reached the offline target at update 100 in seed 42,
versus update 200 in seed 17. Observed preflight-inclusive target times decreased
within each matched precision pair. The producer's
`time_to_quality_improvement_established` fields describe only those two fixed
offline-clock observations; they do not establish quantization superiority,
useful rollout time-to-quality, statistical recipe improvement or promotion.
BF16 seed-42 native target time alone increased from 272.108 to 273.974 seconds
(0.686%); its preflight-inclusive time decreased from 356.238 to 353.192 seconds.

The first two seed-42 cases shared their host with the Alpamayo head control;
later cases shared it with the GR00T frozen-backbone qualification. Actual GPU
and CPU ownership was distinct, but shared-host and filesystem effects remain
uncontrolled. All cleanup costs and original clocks are retained without
adjustment. The joint decision and exact source/result hashes are in the
[family evidence](evidence/native_qlora_family_engineering_4cba88f.json).

The selected native build `26bc60f` passed GR00T lifecycle job `38537970` in both
NF4 and BF16 LoRA modes. Fresh ten-update runs matched the retained `4cba88f`
model, optimizer, RNG, data cursor and task results at updates five and ten.
Each current-build five-to-ten recovery was exact. This comparison permits only
the declared training-identity metadata differences; it does not enable
cross-build resume or modify old artifacts. The original elementwise finite
check remains selected.

Changing the bound `finite.py` source caused actual native resume to refuse
before data/model preparation or checkpoint mutation in both modes. All 28
bound implementation files matched the qualified source. Each artifact retained
all 224 FP32 LoRA factors and 537 FP32 head parameters, with exact optimizer
membership. Fresh current DCP, current portable and old `4cba88f` portable loads
all reproduced the corresponding 20 engineering rows exactly. Native official
head-policy `--base` evaluation also matched the original first two rows; this
is a two-row dispatch check, not a new quality result.

The gate consumed 1,674 GPU-seconds. Independent review verified 132 receipt
files and all 1,215 raw memory samples. One BF16 sample exceeded the original
v9 collection deadline, so **whole-host PSS coverage remains incomplete**; the
required per-stage observations passed and sampled peaks remain lower bounds.
The terminal controller and owned steps were released before the next admission.
This small current-build gate complements the separately qualified selected
`4cba88f` mature recovery and artifacts; it does not repeat or improve their
quality, timing or rollout conclusions. Exact hashes and artifact paths are in
the [family evidence](evidence/native_qlora_family_engineering_4cba88f.json).

A separate frozen-action-head screen (`38539345`, original `4cba88f`, seed 17)
kept all 537 head tensors in FP32 and trained only the 224 language LoRA factors.
Both modes first passed exact initialized forward/RNG and unclipped adapter-gradient
comparisons against their corresponding joint-head policy. Native three-update
runs, own two-to-three recovery, unchanged head/base fingerprints, complete
optimizer membership and fresh portable reloads passed before either learning
arm started. Both then completed the fixed 100 updates and fresh full-160
portable scoring with exact rows and metrics. This establishes the exercised
native path; the short recovery does not establish mature 100-update recovery.

The development candidate **failed the unchanged screen**. BF16 action MSE moved
from 0.0550681 to 0.0550214; NF4 moved from 0.0556545 to 0.0558056. Both missed the
fixed 0.0495613 target. BF16 worsened five decoded action-axis MAEs and NF4 worsened
three. The NF4/BF16 MSE ratio of 1.01425 passes the 1.05 point margin, but cannot
override the target and per-axis failures. Both time-to-target results remain
unreached. No extra updates, seed extension or rollout was admitted.

Measured native allocated peaks were 10.623 GiB for NF4 and 11.735 GiB for BF16;
owned-process sampled peaks across training and fresh scoring were 11.723 and
12.846 GiB. The approximately 1.1 GiB quantization saving remains modest. Freezing
the large action head changes both recipes; its lower footprint is not an
NF4-specific improvement over joint-head training. Native workflows took 278.281
and 258.392 seconds respectively, with median updates of 0.3733 and 0.3273 seconds.
NF4 was slower in this fixed-order observation. All 100 updates applied in each
arm; clipping occurred on 13 NF4 and nine BF16 updates.

The complete qualification consumed 1,801 GPU-seconds. All 1,322 v10 host samples
were complete, and owned-GPU evidence was valid; sampled peaks remain lower bounds.
The old native fresh scorer has no allocator-phase records. Independent receipt
replay retains the failed decision, native artifacts and exact source bindings in
the [family evidence](evidence/native_qlora_family_engineering_4cba88f.json).

Build `02f93ba` passed the affected GR00T checkpoint-load gate (`38541673`) in all
four modes: NF4/BF16 LoRA and NF4/BF16 frozen-backbone head training. Each mode
completed ten updates and exact own-build five-to-ten recovery, including model,
optimizer, RNG, controls, data cursor and all 20 engineering evaluation rows.
Both LoRA modes also matched the retained `4cba88f` five/ten-update numerical
references under the explicit observational metadata bridge. No cross-build
resume or checkpoint identity rewrite was performed. The frozen-backbone modes
train 537 FP32 head parameters with zero LoRA factors; they are distinct from QLoRA.

The gate observed each real native load twice, first with a pristine optimizer
and then with materialized moments. Before live restore, no validation-only
moment tensor remained alive in the pristine case: 1,522 FP32 CUDA moment tensors
(12.222 GiB) for each LoRA mode, or 1,074 tensors (12.074 GiB) for each head-only
mode. The second load retained exactly the optimizer-owned moments. Both restored
states and the subsequent updates were exact. The observer retained only weak
references and scalars, preserved native allocator peak counters and did not
force garbage collection. This establishes the tested lifetime correction, not
a matched old/new workflow speedup or global peak-memory reduction.

Independent replay verified 197 receipt files and all 1,500 v10 memory samples.
Observed host-PSS coverage was complete and owned-GPU attribution was valid for
the monitored main lifecycle/audit interval. Sampled peaks were 32.178 GiB for
owned GPU processes and 29.433 GiB host PSS; these remain lower bounds and exclude
earlier preflight and the later optional CPU audit. The instrumented main workflow
took 1,985.385 seconds; the allocation consumed 2,236 GPU-seconds.

A separate 180.799-second CPU audit on 2026-11-14 compared the existing
`38536188` head-only 200-to-300 recovery outputs with their original 300-update
checkpoints. Both modes had exact model, optimizer, controls, RNG, cursor and
full-160 evaluation results. These newly persisted retrospective proofs do not
change that allocation's **FAILED** status, its incomplete v9 sample 819, missing
historical comparison JSON or original clocks. They do not establish current-build
200-to-300 recovery or new quality/rollout results.

The source-02 staging correction is retained with the evidence: four campaign-root
objects were mistakenly written before the correctly located native bundle was
verified. Prior absence or overwrite of those root objects is unknown; they were
left in place. The sealed active `4cba88f`/native bundles were verified unaffected.
Exact proof hashes, load allocator observations and these limits are in the
[family evidence](evidence/native_qlora_family_engineering_4cba88f.json).
