# Alpamayo-R1 native trajectory QLoRA

This integration connects Alpamayo's supervised trajectory flow loss to the native
Cosmos `action_sft` trainer. It does not replace the separate AlpaGym replay-GRPO
entrypoint, and it does not establish closed-loop driving or real-world safety.
Accelerator and real-data qualification results must accompany a supported recipe;
the CPU tests below establish the integration contract only.

The first configuration binds `nvidia/Alpamayo-R1-10B` revision
`dd4a24cacefc9a6477a6dfc7354de2443401409d`, its public model source
`NVlabs/alpamayo@11a0e01c13a5622377c45ee37d653351453ec43b`, Transformers 4.57.6,
and the Qwen3-VL-8B processor at `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
Use the family runtime, not the Qwen qualification runtime using Transformers 5.8.
The shared [native action source profile](../../docker/posttrain/native_action/README.md)
pins the retained CUDA 12.8 dependency closure for both Alpamayo and GR00T:

```bash
python docker/posttrain/native_action/install.py \
  --uv /path/to/uv-0.11.29 --venv /durable/venvs/cosmos-native-action
source /durable/venvs/cosmos-native-action/native-action-env.sh
```

NF4 additionally requires bitsandbytes 0.49.2 and CUDA BF16 kernels. Checkpoint and
processor bytes, source implementation, task configuration, and resolved trainables
are recorded in native artifact metadata.

The explicit allowlist contains 247 ordinary language Linear weights: seven
attention/MLP projections in layers 0–34, and only K/V projections in layer 35.
The action expert consumes prefix K/V; the final language layer's Q/O/MLP output
and final normalization do not reach this loss. Those weights, vision/projectors,
embeddings, output head, and language normalization stay frozen BF16. NF4 uses
nested quantization, uint8 packed storage and BF16 computation. LoRA uses rank 32,
alpha 64, FP32 parameters; native BF16 autocast applies to adapter GEMMs. The native expert and
input/output action projections are explicitly trained in FP32 by default. Their
parameters and optimizer state are material memory costs, and all are included
in checkpoint and portable artifacts.

`model.backbone_adaptation.mode=bf16_lora` is the matched precision control.
`nf4_frozen`/`bf16_frozen` train only the explicitly selected native task head;
they are distinct controls and are not backbone QLoRA. `train_action_head=false`
is permitted only with trainable language adapters, but leaves the frozen expert
and action projections in their source BF16 dtype. It is not a trainability-only
ablation of the selected FP32 expert and has not been qualified. No mode changes
the task loss.

The model loads its complete approximately 22 GB BF16 source on the host, then
moves/quantizes one matrix at a time; no complete dense base is placed on a GPU
before quantization. Host loading peaks must be measured. No `device_map=auto`,
blanket post-quantization cast, distributed sharding, or CPU offload is supported.
Use one process under `torchrun`. Full activation checkpointing replays the whole
prefix/expert flow graph with a fresh mutable KV cache. Generic per-language-block
checkpointing is deliberately absent because its mutable-cache gradient path is
not equivalent. The configured alternatives are `none` and vision-only checkpointing;
only measured geometries/checkpointing policies may be advertised as qualified.

PhysicalAI input rows preserve four cameras in native order `[0,1,2,6]`, four
frames per camera, 16 history poses, and 64 future poses at 0.1-second intervals.
Positions are meters in the vehicle frame at the current time. Camera calibration,
frame times, media contents, and episode identities are checked. The official
processor renders the driving prefix with empty reasoning; no text-answer loss is
substituted. The native action space transforms future poses into normalized
acceleration/curvature and owns the flow target and noise/time schedule.

Evaluation reports native flow loss and open-loop planar ADE/FDE in meters at
1, 3, and 6.4 seconds. Trajectories use the checkpoint's action denormalization and
ten Euler flow steps. These are open-loop metrics, not simulator success. The
resident evaluator restores training RNG and mixed module modes, does not consume
the training cursor, and associates target crossings with committed checkpoints.

Stage immutable checkpoints and content-bound JSONL rows first, then run:

```bash
export IMAGINAIRE_OUTPUT_ROOT=/durable/path/alpamayo-runs
export PYTHONHASHSEED=17 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_sft \
  --config cosmos_framework/posttrain/configs/sft_alpamayo_qlora.py -- \
  model.checkpoint_path=/staged/Alpamayo-R1-10B-dd4a24c \
  model.processor_path=/staged/Qwen3-VL-8B-processor-0c351dd \
  data.train_path=/data/train.jsonl data.validation_path=/data/validation.jsonl
```

### Exact selected 512-update development command

The starter command above uses recipe defaults; it is not the selected 512-update
study. The following reproduces the recorded seed-17 settings on the pinned
`4cba88f` training build and immutable cohort (manifest digest
`2272c9fae7cd50eb13adaa8fd3bdaaa8467b91621c4983441b201b1faa97c810`).
Use that original build for its checkpoint recovery; current source-bound resume
correctly refuses legacy checkpoints. The matched BF16 control changes only
`model.backbone_adaptation.mode=bf16_lora` and its output name.

```bash
export IMAGINAIRE_OUTPUT_ROOT=/durable/path/alpamayo-development
export PYTHONHASHSEED=17 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_sft \
  --config cosmos_framework/posttrain/configs/sft_alpamayo_qlora.py -- \
  model.checkpoint_path=/staged/Alpamayo-R1-10B-dd4a24c \
  model.processor_path=/staged/Qwen3-VL-8B-processor-0c351dd \
  data.train_path=/data/alpamayo-development-v1/alpamayo-train.jsonl \
  data.validation_path=/data/alpamayo-development-v1/alpamayo-validation.jsonl \
  model.backbone_adaptation.mode=nf4_lora \
  model.backbone_adaptation.seed=17 \
  model.backbone_adaptation.rank=32 \
  model.backbone_adaptation.alpha=64.0 \
  model.backbone_adaptation.dropout=0.0 \
  model.train_action_head=true \
  model.activation_checkpointing=full \
  model.inference_steps=10 \
  rl.seed=17 \
  rl.max_steps=512 \
  rl.micro_batch_size=1 \
  rl.grad_accum=1 \
  rl.evaluation_every=128 \
  rl.evaluation_seed=123 \
  rl.evaluation_max_samples=0 \
  rl.save_every=128 \
  rl.keep_model_last=0 \
  rl.keep_optim_last=0 \
  rl.save_async=false \
  rl.optimizer.name=adamw \
  rl.optimizer.impl=fused \
  rl.optimizer.lr=1e-4 \
  rl.optimizer.warmup_steps=100 \
  rl.optimizer.grad_clip_norm=1.0 \
  rl.optimizer.weight_decay=0.0 \
  rl.optimizer.betas=[0.9,0.999] \
  rl.optimizer.eps=1e-8 \
  rl.lr_schedule.kind=cosine \
  rl.lr_schedule.f_start=0.01 \
  rl.lr_schedule.f_min=0.01 \
  rl.lr_schedule.cycle_steps=512 \
  rl.target_metric=open_loop_ade_6p4s_meters \
  rl.target_threshold=1.8582104004141 \
  rl.target_direction=min \
  rl.stop_on_target=false \
  rl.pause_after_step=0 \
  rl.resume=off \
  rl.run_ts=development \
  job.project=qualification \
  job.group=alpamayo-development-v1 \
  job.name=nf4_lora-seed17
```

The fixed target was derived from the frozen BF16 calibration before adaptation.
Evaluation uses every admitted development row; target nonattainment is retained
through update 512. Set `rl.resume` to an explicit original committed boundary
and a new output name for recovery while retaining every other training setting.
Do not shorten the 512-step cosine schedule to the remaining number of updates.

Use the existing native output/resume settings; resumable DCP includes the full
trainable expert/projections/adapters, AdamW state, scheduler/update clock, RNG,
data cursor and evaluation history. Portable export contains every trained
parameter and requires the same recorded quantized base. It is not a complete
standalone BF16 policy export. Applying adapters to the original BF16 backbone is
a different numerical model. A fresh native portable/checkpoint evaluation uses
`posttrain.entrypoints.action_eval` with `--config`, `--checkpoint`, `--output`
and the same model/data overrides.

From `cbe368d`, exact resume also binds 25 explicit training/recovery source files.
Changed sources or a legacy checkpoint without this contract fail before data/model
construction and before state restoration. Use the original qualified build for
older resumable checkpoints; do not rewrite their identities. Compatible portable
inference reload remains independent of this training-only check. CPU lifecycle
coverage passed; full-model checks of the new build remain separate from the
retained `4cba88f` recovery results below.
The current inventory extends this to 28 files by binding shared flow-dataset
row lookup, media decoding, and the evaluation-input reuse implementation.
Dataset-content identity alone does not identify that execution. Earlier 25-file
contracts require the original build and remain
unchanged; compatible portable inference is still separate.

The separate `posttrain.checkpoint.tools.export_family_dense` command reconstructs
`dequantize(Q(W)) + (alpha/rank) BA` with FP32 accumulation. Specify
`--merged-backbone-dtype=bfloat16` or `float32`; all other parameters and buffers
retain their native dtypes, including the complete trained FP32 expert/projections.
It copies pinned config/processor sidecars, writes deterministic size-bounded
safetensors shards, and commits the verified manifest last. The native Alpamayo
meta constructor streams those values through `action_eval --dense`; original
base-weight files are unnecessary for dense reload. Supply the exact export code
revision/wheel digest through `COSMOS_TRAINING_CODE_REVISION` and
`COSMOS_TRAINING_WHEEL_SHA256`. This dense route is inference-only and does not
replace resumable checkpoints. BF16 casting changes the model's numerical
arithmetic; implementation and tiny-graph checks do not establish full-model task
parity or deployment latency.

The distinct `--unmerged` export option on `8c64b5a` retains BF16 effective base
weights and exact FP32 LoRA factors/expert tensors with separate native adapter
arithmetic. Use it instead of `--merged-backbone-dtype`, and load through
`action_eval --unmerged` instead of `--dense`. It is a standalone dense-base
inference artifact, not four-bit inference or a resumable training checkpoint.
Its commit manifest, strict native loader and original training identity remain
separate from merged export. Tiny native Alpamayo CPU flow tests preserve exact
outputs after unmerged reload. Full-model CUDA qualification now also preserves
both selected NF4/BF16 update-512 policies exactly on all 256 development rows and
two controlled trajectory probes (`38530040`/`38531953`, export/inference build
`64ea823`). Current-build observation-only and evaluation-cache checks remain
pending. The failed BF16 merged-conversion result below remains unchanged.

On the qualified export build, use the original model/task settings, including
adapter seed 17 for the selected policy (the recipe default is 42):

```bash
export COSMOS_TRAINING_CODE_REVISION=64ea823db4a7978c387d195bb464879b4f4733d0
export COSMOS_TRAINING_WHEEL_SHA256=f1a3cc7332007881d41e6daef3379d3d08985bc883cebbe21cdc491d18acdaf7
recipe=cosmos_framework/posttrain/configs/sft_alpamayo_qlora.py
common=(
  model.checkpoint_path=/staged/Alpamayo-R1-10B-dd4a24c
  model.processor_path=/staged/Qwen3-VL-8B-processor-0c351dd
  model.backbone_adaptation.mode=nf4_lora
  model.backbone_adaptation.rank=32
  model.backbone_adaptation.alpha=64
  model.backbone_adaptation.seed=17
  rl.seed=17 rl.evaluation_seed=123
)
export_args=()
for setting in "${common[@]}"; do export_args+=(--override "$setting"); done
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.checkpoint.tools.export_family_dense \
  --config "$recipe" --checkpoint /durable/portable/iter_000512 \
  --output /durable/new-unmerged-policy --unmerged \
  --max-shard-size-bytes 536870912 "${export_args[@]}"
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_eval \
  --config "$recipe" --checkpoint /durable/new-unmerged-policy \
  --output /durable/new-unmerged-evaluation --unmerged -- "${common[@]}" \
  data.validation_path=/data/validation.jsonl
```

The inference loader uses the artifact's verified processor/config sidecars and
effective weights; original source directories are needed for export only.

For observation-only prediction, use
`model.alpamayo.observation.prepare_observation(policy, sample)` with the loaded
native policy. `sample` contains the same ordered four-camera `media` and
`observation` calibration, timestamps, coordinate frame and 16 history poses as
the training inputs. It needs neither `actions` nor `ego_future_rot`; do not
fabricate future labels. Preparation preserves the selected empty-reasoning
prompt and returns only `tokenized` and `history`. The existing ten-step Euler
sampler returns `[1,64,3]` trajectory positions in vehicle-at-t0 meters, at 0.1 s
intervals. This is the selected open-loop policy, not the upstream reasoning
generation workflow or a closed-loop controller.

Within an initialized single-rank native process, using the same resolved model
configuration and device as the training/evaluation recipe:

```python
import torch

from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from cosmos_framework.posttrain.model.alpamayo.observation import prepare_observation
from cosmos_framework.posttrain.model.alpamayo.policy import build_policy, sample_trajectories

policy = build_policy(config, device=device, output=output)
load_adapter_weights(
    artifact, policy.net._cosmos_adapter_checkpoint_codec,
    manager=CheckpointManager(checkpoint_format="dcp"),
)
policy.net.eval()
inputs = prepare_observation(policy, observation_sample)
with torch.random.fork_rng(devices=[device.index]):
    torch.manual_seed(123)
    trajectory_xyz_meters = sample_trajectories(policy, inputs)
```

For a qualified standalone unmerged artifact, replace the construction/reload
lines with `model.alpamayo.dense.build_dense_policy(config, artifact,
device=device, output=output, unmerged=True)`; preparation and sampling stay the
same. CPU tests use the actual pinned processor, history tokenizer and native
sampler: deleting future labels preserves prepared tensors and controlled
trajectories exactly, without mutating inputs or caller RNG. Full-model
observation-only artifact checks remain part of the final-build qualification;
this API does not establish closed-loop quality or deployment latency.

For the first full-model conversion check, the prospectively fixed budgets are:
controlled normalized flow velocity RMSE ≤0.01 and maximum absolute error ≤0.05;
flow-loss difference ≤max(1e-5, 1% of reference); paired-noise ten-step trajectory
deviation ≤0.10 m ADE and ≤0.25 m final displacement at 6.4 seconds; and each
actual open-loop ADE/FDE change ≤max(0.02 m, 1% of reference). Processed inputs
must match exactly, and all values must be finite. The optional
`model.alpamayo.qualification.capture_outputs`/`compare_outputs` helpers record
these outputs and every gate. Two development rows can validate conversion
behavior only; they cannot establish recipe quality, closed-loop driving, or
inference speed. A failed budget remains a failed conversion decision.

The implementation reuses reviewed task forward/data contracts from Cosmos
`48f6aa082b38a598a1fe59735e36a543f8834a02` (wheel SHA256
`fea15c5db109d63b1184eb90327b746260c5192e31ce1d4f7b79fc85c5b8389b`),
without importing its separate adapter framework. CPU contract tests cover exact
allowlist exclusions, units/media validation and trajectory metrics. The actual
tiny public Alpamayo tests run three native flow updates with full checkpointing
both enabled and disabled, require real adapter/head gradients and updates,
verify optimizer membership and frozen-base integrity, and reconstruct trained
state. The same test file includes NF4 cases that require real CUDA kernels under
a one-rank `torchrun`; skipped NF4 tests provide no quantization evidence.

## Prospective development comparison

The `alpamayo-development-v1` study is fixed before candidate outcomes. Its
selection digest is
`78c6643ea8bfb3ff36e30e7e635a1c167052abf973123a90fa27411b3b2f3080`.
It admits only the official TRAIN partition of PhysicalAI revision
`2ac2e9c056e0c8426face5bfad64b0961a295b28`: 1,024 training clips from 128
coarse blocks and 256 development clips from 32 different blocks, eight clips
per block. A block is `(platform_class, country, month)`. All seven blocks
containing the ten engineering-v2 clips are excluded. A fixed SHA256 rule
assigns whole blocks to splits and orders blocks/clips without model outcomes;
the selection and cohort manifests retain the exact rule, metadata hashes,
source receipts, and verified row/media hashes. TEST and ALPASIM are excluded.
Admission manifest
`2272c9fae7cd50eb13adaa8fd3bdaaa8467b91621c4983441b201b1faa97c810`
binds 21,760 verified files (32.51 GB), including 20,480 source images at
1920×1080. All 1,280 rows pass the native camera/timestamp/calibration contract
and finite 16-history/64-future pose checks; cross-split media-hash overlap is zero.
This is data admission, not measured processed shapes or task-quality evidence.

These are development blocks, not verified independent recording families.
The pinned metadata has no year, session ID, or parent-event mapping. Combining
years makes blocks coarser, but a session can cross a month or country boundary.
Thus block-disjoint data cannot establish recording independence. The selected
training/development sets cover 22/17 countries and both platform classes, with
eight examples per block; they do not estimate the full dataset's natural clip
distribution. Official TEST remains reserved until the candidate and confirmation
protocol are frozen and recording/event provenance is available.

Compare `nf4_lora`, `bf16_lora`, and `bf16_frozen` with the same native trajectory
objective, FP32 expert/action projections, fused AdamW state, four cameras by
four frames, rank 32/alpha 64 where applicable, batch one, and full activation
checkpointing. The first pair isolates backbone quantization; the third measures
adaptation relative to the native expert-only policy. It is not numerically
identical to public Stage-2 training: that recipe loads BF16 expert parameters,
whereas these matched controls retain FP32 trainables. No optimizer-precision
change or hyperparameter search is included in this comparison.

Use paired training seeds **17, 42, 123**, evaluation seed **123**, **512 updates**,
and full development evaluation/checkpoint publication every **128 updates**.
All arms use LR **1e-4**, **100** warmup steps and a cosine floor of **1e-6**;
these values follow the pinned public
[Stage-2 base configuration](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_sft/configs/sft_base.yaml).
Set `rl.optimizer.grad_clip_norm=1.0`, `weight_decay=0.0`,
`betas=[0.9,0.999]`, and `eps=1e-8` explicitly. These match the inherited pinned
Transformers 4.57.6 `TrainingArguments` defaults; its Torch ≥2.8 optimizer default
is fused AdamW. The earlier ten-update engineering runs disabled clipping and
remain distinct evidence. No production default or historical run is rewritten.
The native schedule uses `rl.lr_schedule.kind=cosine`, `f_start=0.01`,
`f_min=0.01`, `cycle_steps=512`; its nonzero initial LR is an explicit difference
from Hugging Face's warmup. The bounded 512-update horizon is not the public
recipe's three epochs. Counterbalance arm order across seeds, record placement,
and give every arm identical resources and shared execution optimizations.

The primary task endpoint is single-sample, ten-Euler-step **planar ADE at 6.4s
in meters**; FDE at 6.4s is a required guard. Retain flow loss and ADE/FDE at
1s/3s as diagnostics. Before candidate training, measure the frozen BF16 policy's
development ADE, `A0`, and freeze the common time-to-quality target at
`0.95 * A0`. Count a crossing only after the associated checkpoint is published
and verified. A target crossing alone does not pass the quality guards. Continue
all runs to update 512 for the matched-work comparison; nonattainment remains
unreached, never omitted. For matched elapsed compute, use the earlier final
checkpoint-publication time of each NF4/BF16 seed pair, then compare each arm's
last published evaluation at or before that budget. This rule does not inspect
quality or interpolate unmeasured checkpoints; report the checkpoint lag.

Prospective noninferiority requires the one-sided 95% upper bounds for ADE
degradation to be at most **both 5% and 0.10m**, and for FDE degradation to be at
most **both 5% and 0.25m**, versus matched BF16 LoRA. These proposed development
budgets limit relative error while capping absolute displacement; they are not
official project safety requirements, WTS margins, or the artifact-conversion
tolerances above. Quality improvement additionally requires the ADE-difference
upper bound below zero. Apply these checks separately to the final update-512
comparison and the matched elapsed-compute comparison, never to a retrospectively
chosen best checkpoint. All guards must pass; do not choose a favorable endpoint.
Carry these margins unchanged into independent confirmation.

Average clips within each block, then weight blocks and paired seeds equally.
For development uncertainty, use 10,000 paired crossed-bootstrap replicates
(bootstrap seed 70241): resample paired seeds and whole blocks, using the same
block draw across seed pairs. Report per-seed results and the 95th percentile
upper bounds for relative and absolute differences. Three training seeds and
unverified coarse blocks do not establish a population recipe claim; these
intervals remain exploratory. Final confirmation needs genuine recording/event
families and a variance pilot to determine adequate seed/group counts.

Record startup, preparation, training, evaluation, checkpoint, export/reload,
GPU allocated/reserved/process peaks, and host memory separately. Time to target
includes startup through verified checkpoint publication. Target calibration and
search costs are additional reported costs, not hidden speedups. The nine-run
study contains 4,608 updates and 11,520 evaluation rows before separate baseline
and artifact checks. Admit runs only after the engineering measurements establish
their duration and durable capacity; if correctness, finite loss, or storage
checks fail, stop the affected attempt and preserve it. Do not change the target,
margin, seed list, or one arm's horizon in response to quality outcomes.

## Completed native engineering checks

The [completed ten-update engineering run](evidence/native_qlora_family_engineering_4cba88f.json)
(`38507817`, native `4cba88f`) passed exact historical-state checks, update-5-to-10
recovery, and fresh checkpoint/portable task scores in both arms. It used
`PYTORCH_ALLOC_CONF=expandable_segments:True` for both arms after the earlier BF16
attempt exhausted memory at update seven. There are 247 eligible matrices and
903 trainable tensors: 494 LoRA factors plus all 409 expert/action-projection tensors.

NF4/BF16 peak allocated training memory was **64.664/74.009 GiB**, a **9.345 GiB**
reduction. Native-process host high-water marks reached approximately 42.67/42.65 GiB
during portable export. Historical NVML sampling failed, so no device-process or
complete process-tree peak is claimed for that run. Mean updates 3–10 took
2.966/3.092 seconds in a single short BF16-then-NF4 engineering block; this does not
establish sustained speed or time-to-quality. Open-loop ADE on its two validation
clips worsened in both arms. Process-loss recovery, full development, dense conversion
and processed-shape profiling remain separate qualification checks.

Frozen-source job `38509182` then scored all 256 development clips in both arms.
BF16/NF4 ADE at 6.4 seconds was **1.95601/1.97252 m**, and FDE was
**6.07471/6.10283 m**. These are calibration point estimates, not adaptation or
noninferiority results. The prospective rule fixes the common ADE target at
**1.8582104004141 m**; its digest and source metrics are retained in the same
family evidence. Both external memory samplers completed successfully.
Actual first/last development probes had 3,008 input tokens, 16 image-grid entries,
packed pixels `[11520,1536]`, 16 history poses and `[1,64,2]` action targets/masks.
These new observations do not recreate missing shape records from the earlier
ten-update run or establish all possible input geometries.

Job `38511234` subsequently passed actual process-loss recovery in both arms:
exit during backward attempt six after committed update five, explicit rejection
of a partial checkpoint, then exact model/optimizer/RNG/data and task-score
agreement at update ten. Its NF4 effective-base dense export passed every fixed
conversion gate on both real driving clips, including exact auxiliary tensors.
Maximum paired trajectory displacement was 0.015942 m ADE and 0.048649 m FDE.
The BF16 control failed one clip's task-ADE/FDE tolerances; the overall allocation
therefore remains **FAILED**, independently of the passed recovery and NF4 checks.
This is ten-update artifact evidence, not qualification of a selected 512-update
policy, task-quality preservation or closed-loop driving.

Both dense policies occupied 26.73 GB on disk. NF4/BF16 export allocated peaks were
16.023/25.306 GiB; fresh dense capture reached 25.782 GiB in both arms. Complete
artifact-workflow sampled GPU-process peaks were 26.482/31.162 GiB, including dense
loading. Fresh evaluator subprocesses emitted no allocator phase records, so
their allocator peaks remain unknown despite valid external sampling. The
[retained receipts](evidence/native_qlora_family_engineering_4cba88f.json) include
these distinctions and the failed BF16 measurements. The selected 512-update
checkpoint still needs its own mature-recovery and artifact checks.

The first full NF4 seed-17 study (`38513808`) completed all 512 updates and
published every scheduled checkpoint plus the complete portable policy. On all
256 development clips, final 6.4-second ADE/FDE were **2.10761/6.38240 m** versus
the same quantized frozen source's **1.97252/6.10283 m**. The fixed ADE target was
**unreached**. ADE worsened on 144/256 clips and 21/32 coarse blocks; no rows were
removed. These are adverse development observations, not a quantization-specific
diagnosis. The subsequently completed matched BF16 arm is reported below.

The complete measured workflow took **114.57 minutes**, including **26.91 minutes
training, 80.09 minutes scheduled evaluation, 2.27 minutes checkpoint publication**
and 23.07 seconds portable export. Native startup took 241.54 seconds. The allocation
consumed 6,937 GPU-seconds; wrapper and native clocks have distinct boundaries in
the retained evidence. Allocated/reserved peaks were **64.664/65.971 GiB**, sampled
owned GPU-process peak **67.104 GiB**, and sampled process-tree RSS peak **44.000 GiB**
at portable export. Loading, training, evaluation, checkpoint and export intervals
had valid GPU and complete host samples; startup/exit gaps keep the whole-run PSS
flag incomplete. These sampled peaks are lower bounds.

The 903-parameter-tensor inventory and optimizer/update/data/evaluation controls
passed verification at updates 128/256/384/512. This completed uninterrupted run
does not replace selected-checkpoint fresh reconstruction or the prepared
384-to-512 recovery check. Earlier ten-update recovery evidence retains its scope.

The matched BF16 LoRA seed-17 arm (`38518404`) also completed all 512 updates,
five full-development evaluations, four committed checkpoints and portable export.
Final NF4/BF16 6.4-second ADE was **2.10761/2.08567 m** and FDE was
**6.38240/6.35045 m**. Both missed the fixed **1.8582104004141 m** ADE target;
ADE was **6.85%/6.63% worse** than each arm's frozen source. Both arms initially
deteriorated and then partly recovered: ADE at updates 128/256/384 was
**2.69740/2.24563/2.12797 m** for NF4 and **2.57306/2.18913/2.11739 m** for BF16.
At update 512, per-clip ADE changes from each frozen source correlated 0.923, and
125 of 256 clips worsened in both arms. This points to a shared learning-recipe
problem; it does not identify its cause or establish a quantization defect.

NF4/BF16 mean updates 31–512 took **3.15289/3.05444 seconds**; complete workflows
were **114.57/112.64 minutes**. NF4 was 3.22% slower per update and 1.72% slower
for these workflows. The runs used different A100-80GB nodes, the same declared
16-CPU allocation and measured 16-CPU affinity, and one seed. They do not establish
a causal speed difference or a recipe-level confidence interval. Neither reached
the fixed target, so no time-to-quality improvement was demonstrated. The frozen
matched-compute rule selects NF4 update 384 and BF16 update 512 at 6,702.81 seconds;
its sparse checkpoint cadence leaves a 1,292-second NF4 publication lag. This
endpoint selection is retained without substituting a more favorable checkpoint.

The BF16 workflow included **26.08 minutes training, 78.43 minutes evaluation,
2.72 minutes checkpoint publication**, 243.86 seconds startup and 21.98 seconds
portable export. Its allocation consumed **6,816 GPU-seconds**. NF4/BF16 sampled
owned GPU-process peaks were **67.104/76.240 GiB**, saving **9.137 GiB**; allocated
peaks were **64.664/74.009 GiB** and reserved peaks **65.971/75.107 GiB**. Process-tree
RSS peaked at **44.000/43.783 GiB** during portable export. Major loading, training,
evaluation, checkpoint and export intervals have complete host samples; brief
startup/exit gaps keep the whole-run PSS completeness flag false. Sampling reports
lower bounds and does not replace allocator measurements.

Both runs preserved the same 903-tensor trainable inventory and verified their
native optimizer, update, RNG, cursor and evaluation controls at all four committed
boundaries. Their trainer/RNG control hashes match at each boundary. The fixed
training media layout remains four cameras with four frames each and 64 future
poses; the separate first/last development probes recorded 3,008 processed tokens
and 16 image-grid entries. These records are not an all-row token census.
Fresh reconstruction of the selected 512-update policies and 384-to-512 recovery
required separate 128-GiB-host allocations. An additional login-only
portable inspection exceeded the login host-memory limit; it is retained as a
failed auxiliary check, not relabeled as a training or artifact failure. The
completed native jobs verified their published artifacts; small metadata checks
do not establish fresh full-policy task parity.

The selected NF4 policy subsequently passed mature recovery in allocation
`38525895`: all 128 updates from committed update 384 to 512, model/optimizer
tensors, scheduler/update controls, RNG, cursor and evaluation history matched
the uninterrupted run exactly. All 256 final development rows were identical.
Fresh native DCP and portable loads also matched controlled flow/trajectory
outputs on indices 0 and 255 and preserved every trained auxiliary tensor.
The resumed clock is explicitly current-process-only; the quality target remains
unreached. Matched BF16 mature recovery subsequently passed in `38528271` with
the same exact model/optimizer/control, 128-update, full-256-row and fresh
checkpoint/portable checks. Its scheduler completed normally in 2,304 GPU-seconds;
the controller's brief completing state and subsequent record purge were
reconciled against the successful full-user queue and terminal accounting. No
cancellation was needed. Its 42 hash-listed receipts plus inventory are preserved,
with the later allocation-release proof separate from that archive.

BF16 recovery's sampled owned-GPU peak was 76.543 GiB and host RSS was 43.740 GiB.
One membership-transition sample leaves whole-run host PSS incomplete. These are
recovery measurements, distinct from the original 512-update training peaks;
they are not a new paired performance comparison. Both selected standalone
artifact results follow below; final-build cache/observation checks remain
separate work.

The earlier NF4 qualification (`38525895`) finished successfully, but Slurm
retained its allocation after reporting completion. Only that allocation was
cancelled after its 42 hash-listed receipt files had been verified. Final scheduler
status is
**CANCELLED after completed scientific work**, with **2,602 GPU-seconds**, including
379 seconds beyond the preliminary 2,223-second accounting record. The archive
has those 42 receipts plus their inventory; the later cleanup receipt preserves
the controller/accounting discrepancy. Sampled owned GPU memory and host RSS are
valid lower bounds, while whole-run host PSS coverage is incomplete. This does
not establish a complete-memory performance gate or a new quality result.

The selected NF4 update-512 policy now passes standalone unmerged qualification
(`38530040`, **3,143 GPU-seconds**, normal `COMPLETED` release). Fresh portable
evaluation first reproduced every original resident-development row and metric;
the standalone policy then reproduced all 256 rows exactly. Controlled rows 0/255
also matched every processed input, flow output and 64-waypoint trajectory,
all 494 LoRA factors, 409 trained expert/action tensors, 247 effective BF16 base
matrices, and native rotary aliases. Reload ran in a fresh namespace without
either original base/processor path spelling and with quantizer imports blocked.
The installed runtime still contains bitsandbytes; this proves import independence,
not installation without that package.

The artifact has **51 weight shards**, 1,663 tensor records and 63 indexed files,
totalling **27,070,952,449 bytes** excluding its commit manifest. Native export took
332.12 seconds; separate fresh portable and standalone full-development scoring
took 1,086.63 and 1,079.61 seconds. These qualification subprocess clocks include
loading and integrity work and are not inference-latency benchmarks. Sampled
owned GPU peaks were **21.879 GiB** for the combined export/portable-check phase
and **26.939 GiB** for standalone reload/evaluation; sampled host RSS peaked at
24.431/3.865 GiB. Both monitored phase scopes have complete sampled host PSS,
but exclude container startup and scheduler release. These are artifact-workflow
measurements, not training peaks. All 51 retained receipt files were hash-verified;
the later controller-release proof is separate from their archive.

This deployment representation preserves the existing checkpoint's ADE/FDE
**2.10761/6.38240 m**, including its unmet fixed quality target. It does not repair
that learning outcome, establish merged-export parity, qualify closed-loop driving
or demonstrate inference acceleration.

The matched BF16 update-512 policy also passed the same standalone qualification
(`38531953`, **3,185 GPU-seconds**, normal `COMPLETED` release). Its fresh portable
and standalone evaluations reproduced all 256 original development rows and
metrics exactly; controlled inputs, trajectory/flow outputs, all 903 trained
tensors, 247 effective BF16 matrices and native aliases also matched. Its complete
51-shard artifact has the same 27,070,952,449-byte indexed size, with the original
BF16 base and separate LoRA operations. The base and processor paths were absent
and quantizer imports blocked during fresh standalone loading, as in the NF4 check.

BF16 native export took 354.70 seconds; fresh portable and standalone full scoring
took 1,094.43 and 1,080.49 seconds. Sampled owned GPU peaks were **31.162/26.939 GiB**
for export/portable and standalone phases, with sampled host RSS **23.155/3.827 GiB**.
Both monitored scopes had complete sampled PSS; startup and scheduler release are
outside those scopes. These are qualification clocks and memory measurements,
not training or serving benchmarks. A GR00T allocation overlapped on the same
node with a different GPU and disjoint CPU affinity; node/filesystem interference
remains possible. All 51 receipt hashes, normal controller release and UTC step
boundaries are retained separately. This artifact preserves BF16's adverse
**2.08567/6.35045 m** ADE/FDE and unreached target; it does not turn that result into
a quality pass or establish merged-export parity.

The paired diagnostic profile (`38523526`, native `4cba88f`, 1,284 GPU-seconds)
completed 24 updates per arm and traced the last four updates plus the final
two-row resident evaluation. Each row was prepared twice: once for flow loss,
then again for trajectory prediction. Raw events place the second preparation
serially on the evaluator thread after the loss, with negligible recorded GPU
overlap. NF4/BF16 duplicate preparation consumed **2.549/2.245 seconds** of the
instrumented **9.759/8.826-second** evaluations: **26.12%/25.44%**. Image decoding
and processor ranges are inside those totals and must not be added again.
These fractions bound the removable evaluator work before reuse costs; they are
not achieved speedups or a projection for the complete 512-update workflow.

All 24 training preparations and the four final evaluator preparations per arm
had 3,008 tokens, 16 image slots and 128 supervised action elements. One prepared
CPU tensor batch occupied **70,829,184 bytes (67.55 MiB)**, supporting investigation
of bounded reuse within one evaluation row. The profile used constant LR and
no gradient clipping; its update timings do not represent the selected 512-update
cosine/clip-1.0 recipe. Profiler serialization also inflated stage clocks.
The [profile receipts and independent raw-event review](evidence/native_qlora_family_engineering_4cba88f.json)
retain allocator measurements, sampled memory bounds and incomplete whole-run
host-PSS flags. Commit `81c9913` implements default-off, one-row CPU preparation
reuse with a fixed 128-MiB retained limit, local-media/settings mutation checks,
separate tensor ownership and cleanup on failure. Thirty focused CPU cases
include native evaluation, committed recovery and the actual pinned processor.
Full-model accelerator checks and a matched uninstrumented workflow comparison
in both precision arms remain required before promotion.

The original third seed-17 control, frozen BF16 backbone with **409 FP32
expert/action trainables and no LoRA**, completed all 512 updates in `38533726`
(6,214 GPU-seconds, normal controller release). This is the native matched head
policy, not the public Stage2 recipe's BF16 expert precision. All four committed
boundaries preserved exact optimizer membership, scheduler/update/RNG controls,
training cursor and evaluation history. Native frozen-state and publication checks
passed; separate fresh-task reload and mature recovery of this head-only policy
remain untested. LoRA recovery evidence does not fill that mode-specific gap.

Head-only ADE/FDE at update 512 was **2.06870/6.30633 m**, with flow loss
**0.284094**. These are smaller errors than BF16 LoRA's 2.08567/6.35045 m, but
still **5.76%/3.81% above its frozen source**; flow loss is 6.56% higher. None of
the scheduled 0/128/256/384/512 evaluations reached the unchanged 1.8582104-m
ADE target. All three original arms therefore failed useful adaptation in this
seed. The head-only arm clipped 505/512 updates, versus 506/508 for BF16/NF4
LoRA, and all three had their largest training loss at update 279. Head-only
and BF16 LoRA per-update loss correlation was 0.99895. These retained diagnostics
support shared adaptation drift, not an identified NF4 correctness defect; no
difficult clips or release metrics were removed.

Head-only mean update time over steps 31–512 was 2.3359 seconds; the complete
workflow took 102.68 minutes, including 74.72 minutes of evaluation. Sampled owned
GPU memory peaked at **56.934 GiB**, with host RSS **42.849 GiB**; whole-run host
PSS coverage remains incomplete. These memory and update costs belong to the
head-only training graph, separate from the matched NF4/BF16 LoRA comparison.
All three runs had actual 16-CPU affinity, but placements differed; this control
also overlapped another allocation on the node and lacks a NUMA-topology receipt.
These clocks are descriptive, not an isolated runtime or time-to-quality gain.
The 50 hash-listed terminal receipts plus inventory preserve all failed targets
and raw release/accounting observations. Further paired seeds and independent
recording families remain necessary for recipe-quality claims.
