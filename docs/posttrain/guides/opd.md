# Qwen3-VL on-policy distillation

Qwen3-VL OPD adds a frozen vLLM teacher to the ordinary posttrain online-RL loop. The student
generates a realized completion, the teacher scores that same completion, and the policy trains on
the resulting token-level signal before publishing its next weight version to rollout.

The original architecture, failure semantics, and G0–G8 verification ladder are recorded in
the internal design record.
The older internal implementation record
is the M1 implementation and Cosmos-RL alignment record; its future-M2 choices are historical.

## When to choose it

Use OPD when you have a **stronger model to imitate** rather than a reward to maximize. The student
samples on-policy, so it learns to correct the states it actually reaches — unlike offline
distillation on a fixed corpus, which teaches it about states it may never visit.

Compared with the alternatives:

- [`grpo.md`](grpo.md) needs a reward that discriminates between samples of one prompt. OPD needs no
  reward at all: the teacher's own distribution supplies the credit. Choose OPD when a good teacher
  exists but a good scorer does not.
- [`dpo.md`](dpo.md) needs preference pairs collected in advance and no teacher at serving time.

OPD is GRPO's topology plus one extra role, so it reuses the GRPO loop rather than forking it: the
entry adds a ranked teacher ActorGroup and swaps the objective. The teacher is frozen and has **no
weight-sync edge** — it never receives updated policy weights.

Two credit forms are available: same-token direct OPD, and cross-token DPCA. See *Teacher modes*
below.

> **Checkpoint safety for every non-smoke OPD run:** `rl.save_every=0` writes neither cadence
> checkpoints nor a normal final checkpoint. Several commands below deliberately use zero for short
> evidence smokes; do not increase their `max_steps` and treat them as training recipes unchanged.
> Before spending cluster time, set a positive `rl.save_every`, keep `job.name` and
> `IMAGINAIRE_OUTPUT_ROOT` stable across relaunches, prefer `rl.checkpoint_format=dcp` for a large
> sharded student, and confirm that `checkpoints/latest_checkpoint.txt` appears early in the run.

## Quick start: four-GPU Countdown

Set up the container and Posttrain environment from
[sections 1 and 2 of the run guide](../howto/run-posttrain.md#1-start-the-container). The public
profile uses a two-GPU Qwen2.5-1.5B policy, a one-GPU rollout, and a one-GPU Qwen2.5-3B teacher,
and publishes the policy to the rollout over NIXL. It runs direct top-k OPD on the public Countdown
dataset for 10 functional steps, not as a quality recipe. Countdown reward is reported as a
diagnostic in direct mode; the direct objective itself minimizes teacher divergence.

```bash
case "$(uname -m)" in
  aarch64)
    uv sync --project docker/posttrain/environment \
      --frozen --group nixl-cu130-aarch64 --active --inexact
    ;;
  x86_64)
    uv pip install --python "$POSTTRAIN_PY" --no-deps \
      "nixl-cu13==1.3.2" "nixl==1.3.2"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

export POSTTRAIN_NIXL_CUDA_FLAVOR=cu130
export UCX_TLS=cuda_copy,cuda_ipc,sm,tcp,self
export UCX_NET_DEVICES=lo
export NCCL_NET_PLUGIN=none
export UCX_MODULE_DIR="$($POSTTRAIN_PY -c \
  'import sysconfig; print(sysconfig.get_path("purelib") + "/nixl_cu13.libs/ucx")')"

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.utils.nixl_preflight
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.opd \
  --config cosmos_framework/posttrain/configs/opd_qwen1p5b_countdown_public_4gpu.py -- \
  job.name=opd_public_4gpu
```

Override any typed field after `--`, for example `rl.max_steps=2 rl.save_every=0` for a shorter
smoke.

## Data format

`rl.data.path` reads one JSON object per line. `rl.data.prompt_column` names either a plain string or
a chat-message list; the environment renames that value to the canonical `prompt` field before the
model's `DataPacker` sees it. The JSONL reader retains only that prompt and the optional configured
response column, so put all reward ground truth inside the response value.

Text-only, pure-distillation row (`rl.reward=constant` needs no reference):

```json
{"messages":[{"role":"user","content":"Explain why the sky appears blue."}]}
```

Image-conditioned row:

```json
{"messages":[{"role":"user","content":[{"type":"image","image":"/shared/images/example.jpg"},{"type":"text","text":"Where is the red object?"}]}]}
```

All media paths must be absolute or resolve identically on every Ray node. The final
`cross_token_multimodal` teacher path supports ordered images; video teacher input is rejected.
For a rule reward, include the configured response column, for example:

```json
{"messages":[{"role":"user","content":"Use 2, 3, and 4 to make 14."}],"reference":{"nums":[2,3,4],"target":14}}
```

```text
rl.data.path=/shared/data/train.jsonl
rl.data.prompt_column=messages
rl.data.response_column=reference  # omit for rl.reward=constant
```

Named Hugging Face datasets use `rl.data.name/subset/split` plus `prompt_column`,
`response_column`, and optionally a pure row transformer at `items_fn`; mapped HF rows may preserve
other fields. The Countdown demo is the smallest concrete example.

## 397B → 235B resource topology and launch

The target is synchronous Qwen3.5-397B-A17B → Qwen3-VL-235B-A22B distillation:

| role                         | model                            |          shape | GB200 nodes |
| ---------------------------- | -------------------------------- | -------------: | ----------: |
| policy                       | native Qwen3-VL-235B-A22B        | FSDP2 width 64 |          16 |
| rollout + data/reward/replay | Qwen3-VL-235B-A22B in vLLM       |            TP4 |           1 |
| teacher                      | frozen Qwen3.5-397B-A17B in vLLM |            TP8 |           2 |
| total                        |                                  |        76 GPUs |          19 |

The three model-bearing groups occupy independent pools. Teacher scoring, policy training, and
policy→rollout weight publication are ordered barriers; this increment has no asynchronous overlap
or stale experience. The teacher is frozen and therefore has no incoming weight edge.

The following is the validated three-step launch shape. Substitute accessible model/data/image,
container, output, and Slurm values; keep the role widths and NIXL/UCX environment together.

```bash
STUDENT=/path/to/shared-storage
TEACHER=/path/to/shared-storage
DATA=/path/to/shared-storage
ACCOUNT=your_slurm_account
RUNTIME=/path/to/shared-storage
REPO=/path/to/shared-storage
OUT=/path/to/shared-storage
LOGDIR=/path/to/shared-storage

python3 cosmos_framework/posttrain/utils/launch_multinode.py \
  --nodes 19 --gpus-per-node 4 --partition batch_long \
  --account $ACCOUNT --wckey p0 --time 12:00:00 --timeout 46800 \
  --container-image $RUNTIME --container-mounts /path/to/shared-storage:/path/to/shared-storage \
  --repo-root $REPO --venv /opt/venv/cosmos_rl/bin/activate --no-pip \
  --log-dir $LOGDIR \
  --ray-runtime-env "export IMAGINAIRE_OUTPUT_ROOT=$OUT; \
    export WANDB_MODE=disabled; \
    export UCX_TLS=rc_mlx5,dc_mlx5,cuda_copy,cuda_ipc,sm,self,tcp; \
    export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,eth0; \
    export UCX_MODULE_DIR=/opt/venv/cosmos_rl/lib/python3.12/site-packages/nixl_cu13.libs/ucx; \
    export UCX_MAX_RMA_RAILS=4; export UCX_CUDA_IPC_ENABLE_MNNVL=yes; export UCX_LOG_LEVEL=error" \
  --run "python -m cosmos_framework.posttrain.entrypoints.opd \
    --config cosmos_framework/posttrain/config/config_opd.py -- \
    model_backend=qwen3_vl \
    model_qwen3_vl.name_or_path=$STUDENT model_qwen3_vl.checkpoint_path=$STUDENT \
    rollout.name_or_path=$STUDENT rollout.tensor_parallel_size=4 \
    rollout.gpu_memory_utilization=0.75 rollout.max_model_len=16384 rollout.max_tokens=16 \
    rollout.temperature=1 rollout.top_p=1 rollout.top_k=-1 rollout.repetition_penalty=1 \
    rl.teacher.name_or_path=$TEACHER rl.teacher.tensor_parallel_size=8 \
    rl.teacher.max_model_len=16384 rl.teacher.input_mode=cross_token_multimodal \
    rl.teacher.signal_mode=dpca rl.teacher.top_k=0 rl.teacher.scoring_temperature=1 \
    rl.data.path=$DATA rl.data.prompt_column=messages rl.reward=constant \
    rl.distillation_mode=advantage rl.jsd_beta=null rl.kl_beta=0 rl.kl_penalty_coef=1 \
    rl.optimizer.name=fusedadam rl.optimizer.lr=1e-6 rl.optimizer.weight_decay=0 \
    rl.optimizer.betas=[0.9,0.999] rl.optimizer.eps=1e-8 \
    rl.optimizer.grad_clip_norm=1 rl.optimizer.warmup_steps=0 \
    rl.epsilon_low=0.2 rl.epsilon_high=0.2 rl.mu_iterations=1 \
    rl.use_rollout_logprobs_for_loss=false rl.save_every=0 rl.resume=off \
    rl.temperature=1 rl.group_size=2 rl.batch_per_replica=64 \
    rl.micro_batch_per_replica=64 rl.max_steps=3 rl.weight_sync_transport=nixl \
    grid.backend=ray grid.gpu_placement=disaggregated \
    grid.shapes.policy=[1,64] grid.shapes.rollout=[1,4] \
    job.name=opd_397b_235b_smoke"
```

The command above is a validated **three-step smoke** and intentionally sets
`rl.save_every=0 rl.resume=off`; it produces no cadence or normal-final checkpoint. Before turning
that shape into a real training run, change at least the lifecycle fields below and retain the same
values on every requeue:

```text
rl.save_every=<positive collected-batch cadence>
rl.checkpoint_format=dcp
rl.resume=auto
job.name=<stable experiment name>
```

Run the same payload with `--validate-manifest` before requesting 19 nodes. That GPU-less path checks
manifest wiring plus teacher/objective/model/source compatibility. It also checks that the declared
GPU placement and weight transport agree; capacity, role-shape, and TP-width checks still occur when
`run()` starts. The GPU-less check
does verify that a local JSONL is readable on the driver, but it cannot prove cross-node data/media
visibility, model availability, or GPU capacity; validate those values in the launch review as well.

## Teacher modes

`rl.teacher.input_mode` controls modality ownership. `rl.teacher.signal_mode` controls the
mathematical signal. Valid production combinations are:

| input mode                | signal mode | objective   | use                                                                   |
| ------------------------- | ----------- | ----------- | --------------------------------------------------------------------- |
| `shared_vocab_multimodal` | `topk`      | `direct`    | same-token path; teacher sees the original media                      |
| `cross_token_multimodal`  | `dpca`      | `advantage` | final 397B VLM; each tokenizer/processor sees the same original image |
| `cross_token_text`        | `dpca`      | `advantage` | explicit text-only control; teacher never pulls media bulk            |

The historical same-token `advantage` mode remains available for compatibility. It uses the legacy
raw top-k-slice quantity; new same-token runs should use direct OPD's normalized
top-k-plus-residual partition.

### Same-token direct OPD

The teacher returns top-k token IDs and full-vocabulary logprobs for every scored position. The
objective builds a `(K+1)`-way categorical distribution from those K entries plus all remaining
probability mass. Duplicate support IDs, non-finite values, out-of-range IDs, and support mass above
one fail the batch. A typical direct setup is `distillation_mode=direct`, `jsd_beta=1`, and `top_k=8`.

The native and HF Qwen3-VL packers share one normalization: the teacher request carries the
student's prompt with each image placeholder run collapsed to one
`<|image_pad|>` and the completion verbatim, because vLLM re-applies the Qwen3-VL placeholder
replacement to token prompts: an already-expanded prompt comes back `N - 1` tokens longer per image.
The engine's expanded prompt must equal the student's token ids exactly before any position is
scored; a width or boundary drift fails the batch.

The teacher and policy scoring temperatures must both remain one. The current forced-sequence vLLM
path returns raw prompt logprobs, so setting equal non-unit fields would not actually temperature-
scale the teacher distribution. In direct mode, all rollout sampling controls remain data-generation
choices because the fixed-trajectory divergence forms no policy-gradient importance ratio.
Advantage mode additionally requires rollout temperature one, top-p one, top-k disabled, and neutral
repetition penalty. Bind also proves tokenizer, processor, chat-template, multimodal placeholder,
and media-cardinality identity for this mode.

### Cross-token DPCA

Qwen3.5 and Qwen3-VL do not share a tokenizer. The production path gives both roles the same ordered
raw image bytes but lets each role use its own tokenizer, processor, chat template, placeholders,
and resize defaults. The teacher jointly tokenizes its rendered multimodal prefix plus the student's
realized completion, attaches its own vLLM multimodal request, scores the realized teacher tokens,
and aligns only the completion tokenizations where their decoded text matches exactly. An alignment
guard rejects a long unmatched chunk rather than force-closing it. Video teacher input is not part
of this increment. The teacher identity consumes the packer's `PackerProcessorIdentity`
declaration ([`../howto/custom-datapacker.md`](../howto/custom-datapacker.md)); on Qwen3-VL it binds
the video processor, so OPD identities persisted before that binding refuse to resume. This is
intended: a changed teacher processor must never be mistaken for the recorded one.

Each submitted image block contains exactly one logical image marker between the teacher's vision
start/end tokens. Before scoring, the same teacher processor supplies the ordered image grids and
therefore each expected realized marker width. The vLLM response must preserve the complete prefix
after collapsing only those exact-width runs; a second expansion or any boundary drift fails the
batch before a teacher signal is published.

Exactly one think-conditioning prefix is allowed. A real Qwen3.5 template may emit the empty closed-
think block even with `enable_thinking=False`; the renderer detects and removes that suffix before
inserting the configured prefix exactly once. Missing, duplicate, or misplaced conditioning fails
before scoring.

For one aligned chunk `c`:

```text
L_S(c) = sum of student rollout logprobs in the chunk
L_T(c) = sum of teacher realized-token logprobs in the chunk
A_DPCA(t) = (L_T(c) / L_S(c) - 1) * log p_rollout(t)
```

The algorithm constructs credit over the full student completion, then applies
`n_ignore_prefix_tokens`. Policy credit is additive:

```text
A_policy(t) = A_reward(t) + kl_penalty_coef * A_DPCA(t)
```

DPCA requires `distillation_mode=advantage`, `teacher.top_k=0`, `jsd_beta=null`, `kl_beta=0`, collected
rollout logprobs, and neutral likelihood controls: temperature 1, top-p 1, top-k disabled, and no
sampling penalties. Configuration preflight rejects any other combination before model loading.

In both distillation modes `OPDConfig` rejects `rl.importance_sampling_level` other than `token`,
because its advantage objective scores a token-level ratio and direct mode forms none.

## Configuration examples

OPD uses the shared online-loop data plane. Set `rl.data_pool.data_plane=ray_nixl_pool` to pool large
trajectory and teacher fields in every Ray actor; this is independent of `rl.weight_sync_transport`.
Pool sizing and lifecycle details are in [`data-plane.md`](../howto/data-plane.md).

Same-token direct:

```text
rl.distillation_mode=direct
rl.jsd_beta=1.0
rl.kl_beta=0.0
rl.teacher.input_mode=shared_vocab_multimodal
rl.teacher.signal_mode=topk
rl.teacher.top_k=8
rl.teacher.scoring_temperature=1.0
```

Cross-token DPCA:

```text
model_backend=qwen3_vl
model_qwen3_vl.name_or_path=<student>
model_qwen3_vl.checkpoint_path=<student>
rollout.name_or_path=<student>
rl.teacher.name_or_path=<teacher>
rl.distillation_mode=advantage
rl.jsd_beta=null
rl.kl_beta=0.0
rl.teacher.input_mode=cross_token_multimodal
rl.teacher.signal_mode=dpca
rl.teacher.top_k=0
rl.teacher.scoring_temperature=1.0
rl.teacher.conditional_think_prefix='<think>\n\n</think>\n\n'
rollout.temperature=1.0
rollout.top_p=1.0
rollout.top_k=-1
rollout.repetition_penalty=1.0
```

`rl.teacher.tensor_parallel_size` is the teacher WorkerGroup rank width. A ranked teacher requires
`grid.backend=ray`; the local backend supports only TP1. The teacher lives in its own ActorGroup and
GPU DeviceGroup, so its TP group and physical pool are independent of rollout and policy shapes.
`grid.gpu_placement=colocated` shares only the student rollout/experience and policy DeviceGroup;
`disaggregated` separates those student pools too.

`rl.teacher.moe_backend` independently selects the vLLM MoE kernel for the teacher and defaults to
`auto`. It does not inherit `rollout.moe_backend`: set both explicitly when the student and teacher
need the same non-default backend. For example, `triton` avoids an automatically selected
architecture-specific kernel whose tile constraints do not support a particular teacher TP shard
shape; the selected backend must still be supported by the pinned vLLM runtime.

GPU capacity is therefore structural rather than fractional. Let `K` be the common student actor
product (`POLICY_RANKS`, equal to rollout/experience cells for colocated IPC) and `T` the teacher
actor product (`teacher.tensor_parallel_size` in the current one-replica recipe). Colocated OPD
requires `K + T` physical GPUs because the teacher DeviceGroup is disjoint from the shared student
pool; for example, `K=8` and `T=1` require 9 GPUs, not one 8-GPU node. Disaggregated OPD requires
`policy cells + experience cells + teacher cells`. The binder rejects an undersized allocation before
starting Ray, comparing against `NNODES x GPUS_PER_NODE` when the launcher declares them. A cluster
Ray started itself is also checked directly; a cluster it joined is not, because worker registration
is still in progress there and a shortfall would prove nothing.

## Metrics and checkpoints

The driver emits one committed metric frame per collected-batch boundary, after that batch's
optimizer step(s), to the console and, unless disabled, W&B. The main OPD keys are:

| metric                                                    | meaning                                                                                            |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `train/loss_avg`                                          | objective loss after the declared token/sequence reduction                                         |
| `train/valid_sequence_count`, `train/valid_token_count`   | local work folded into the DP-global reduction                                                     |
| `train/opd_divergence`                                    | mean same-token divergence; zero on DPCA credit-only batches                                       |
| `train/opd_advantage`                                     | mean final policy advantage in `advantage` mode                                                    |
| `train/opd_teacher_credit`                                | mean additive DPCA credit, before `kl_penalty_coef` scaling                                        |
| `train/opd_teacher_score_{request,success,failure}_count` | bounded teacher attempt diagnostics                                                                |
| `train/opd_teacher_scored_{sequences,tokens}`             | teacher workload accepted for the step                                                             |
| `train/opd_dpca_*`                                        | chunk count/length, alignment coverage, and teacher/student chunk logprob means                    |
| `train/reward_{mean,std,min,max}`                         | task reward distribution; all zero for `rl.reward=constant`                                        |
| `rollout/completion_length_*`, `rollout/advantage_*`      | generated-length and token-weighted broadcast group-advantage distributions                        |
| `timer/e2e_step_s`                                        | training batch through policy→rollout weight publication; excludes later validation/checkpoint I/O |

OPD does not publish separate model checkpoints for teacher or rollout. The policy checkpoint is the
source of truth; a resume restores it and performs a complete policy→rollout weight sync before the
next generation. Outputs are written under:

```text
$IMAGINAIRE_OUTPUT_ROOT/<job.project>/<job.group>/<job.name>/<RUN_TS>/
  config_resolved.yaml
  checkpoints/
    iter_*/
    latest_checkpoint.txt
```

`rl.save_every=0` disables normal cadence/final saves. Positive values save at committed collected-
batch boundaries; `rl.resume=auto` resumes the newest complete checkpoint, and
`rl.checkpoint_format` selects `consolidated` or `dcp`. OPD/GRPO share DPO's `save_async`,
`keep_optim_last`, `keep_model_last`, and `upload.*` lifecycle through the same checkpoint manager.
Consolidated FSDP CPU snapshots may write asynchronously; DCP model/optimizer writes stay synchronous
while their small publication tail may run in the background. Emergency saves use the same manager.

For a real OPD run, verify all of the following before scaling past a smoke:

1. `rl.save_every` is positive and affordable relative to one collected-batch duration;
2. `job.name` and `IMAGINAIRE_OUTPUT_ROOT` resolve to the same shared output namespace on relaunch;
3. `checkpoints/latest_checkpoint.txt` points to a complete `iter_*` directory after the first
   expected save;
4. when `rl.upload.enabled=true`, completed archive copies carry `.uploaded` (the destination-bound
   archive marker) locally before retention; the teacher's HF directory itself is produced by
   `python -m cosmos_framework.posttrain.checkpoint.tools.materialize_hf`
   ([`checkpoint-artifacts.md`](../howto/checkpoint-artifacts.md));
5. the relaunch logs a nonzero restored optimizer step before rollout resumes generation.

The checked-in countdown recipe now uses the stable job name `qwen1p5b_countdown`. Attempts created
under its former timestamp-expanded name remain valid but live in a different output namespace;
resume them with an explicit `rl.resume=<.../checkpoints/iter_*>` path or by restoring that exact old
`job.name`.

Only student policy training state is resumable. The frozen teacher and rollout engines are rebuilt,
then rollout receives the restored student weights. As with GRPO, the JSONL prompt cursor, rollout
RNG, replay contents, and next group identity are not checkpointed, so OPD resume preserves optimizer
progress but does not promise the exact next online trajectory. An explicit checkpoint path is
required if the new invocation changes the output namespace.

## Failure and publication semantics

Teacher scoring is all-or-nothing for one synchronous batch. Every TP lane must agree on the
ordered request digest, enter the same vLLM call, validate the same result, and exhaust the same
bounded retry attempt. Only the replica leader publishes the tensor-bearing trajectories; other
lanes publish small status metadata. Replay receives nothing until the complete teacher batch is
valid.

Policy→rollout weight publication happens after optimizer commit. A partial transfer never stamps a
new rollout version, but it may have physically changed part of the engine; the run is poisoned and
must recover from the policy checkpoint followed by a complete sync.

The landed asymmetric NIXL path is consumed through `OPDConfig`'s transport, bucket, probe, and
bandwidth fields. Native semantic mapping must drive the same streaming plan and NIXL sizing, and
the rollout validates mapped live-engine coverage before bucket zero mutates it. The mapper addresses
routed-expert parameters (`w13_weight`/`w2_weight`) on the checkpoint's `experts` container and names
no vLLM nesting; the rollout resolves those destinations onto its engine's live layout
(`VllmMoeLayout`, decided from the live `RoutedExperts` owner modules: the container itself on vLLM
0.23, its `routed_experts` child on vLLM 0.24) before that coverage check, and brackets the stream
with vLLM's layerwise checkpoint reload scoped to exactly those modules. During every mapped
stream, each active vLLM TP worker compares the policy's full language `q_proj` source slice with the
q segment of its live fused `qkv_proj`; after all q/k/v contributors load, it re-reads and hashes the
final segment before readiness/version publication. The final 235B run still requires a real mapped
live-weight witness. Tied embeddings and compiled native Qwen execution are not supported by this
increment.

`roles/rollout/vllm_weights.py` is an explicitly temporary compatibility layer for this reload and
witness behavior. It should disappear once vLLM consumes canonical checkpoint-format updates
directly; rollout generation and transaction ordering remain in `roles/rollout/vllm.py`.

## Verification

The implementation was closed in this order:

1. G0: fixed-tensor top-k-plus-other and DPCA arithmetic, codec, and config failures.
2. G1–G3: native dense/tiny-MoE, same-model fixed point, multimodal conditioning, and small
   cross-token alignment.
3. One typical Cosmos-RL image case and a few optimizer steps establish the implementation-level
   alignment. Bitwise replay and evidence-only exporters are not production requirements.
4. G7 completed one real 235B multimodal policy forward/backward at FSDP64 without OOM.
5. G8 completed three image-conditioned 397B→235B optimizer steps. Initial and post-update
   289-bucket NIXL streams each repacked the live MoE weights, verified the mapped Q projection,
   and published rollout versions 0 through 3. The finite losses were 0.1979, 0.2330, and 0.0962.

G7 and G8 consumed the landed native Qwen model and asymmetric NIXL paths. Their temporary recipes,
capacity launcher, replay artifacts, and evidence validators were removed after the runs; the merged
surface is the typed OPD configuration, entrypoint, model/packer/teacher implementations, and normal
multinode launcher. The immutable release image can therefore be built and published separately.

## Explicit limits

This increment does not implement a reference role or `kl_beta>0`, policy TP/EP/PP, CP greater than
one, asynchronous teacher/rollout execution, teacher replicas/failover, full-vocabulary teacher
logits, multi-turn agents, or shard-aware weight movement.
