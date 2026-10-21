# cosmos_framework.posttrain — Cosmos post-training framework

Post-training for Cosmos models on a decoupled model / algorithm / trainer design: typed config
with inherited experiment defaults; runnable SFT, DPO, GRPO, PPO, OPD, and NFT entries; and an orchestration
layer that scales one imperative loop from a single GPU to a multinode Ray cluster.

> **Documentation: start at [`docs/index.md`](index.md).**
> New here, go to [`docs/howto/run-posttrain.md`](howto/run-posttrain.md).

The package is `cosmos_framework.posttrain` (all internal imports are absolute). Run from the
cosmos-framework repository root. Tests live under `tests/` — `pytest cosmos_framework/posttrain/tests -m L0`.

## Quick start

For the public single-host workflow, follow the
[general run guide](howto/run-posttrain.md). It is the sole
release-facing setup and launch path for GRPO, DPO, PPO, OPD, NFT, and Joint PPO.

## Orchestration: backend × actor/device grouping

One `run()` in the entry drives every mode. Config selects the backend and recipe grouping; the
entry lowers those choices into typed placement:

| axis                 | entrypoints              | values                           | meaning                                                                                                                                                                                           |
| -------------------- | ------------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grid.backend`       | DPO, GRPO, PPO, OPD, NFT | `local` (default) \| `ray`       | WHO hosts the role bodies: this process (torchrun carries single-GPU/FSDP) vs a mesh of Ray worker processes (multinode via `RAY_ADDRESS`); PPO and production NFT require `ray`                  |
| `grid.grouping`      | DPO, PPO, NFT            | `single` (default) \| `per_role` | HOW roles become ActorGroups: all roles of a cell share one process vs one process per role (`per_role` is ray-only); PPO requires `per_role`, while production NFT requires `single`             |
| `grid.gpu_placement` | GRPO, PPO, OPD, NFT      | `colocated` \| `disaggregated`   | whether rollout/experience and policy ActorGroups share one GPU DeviceGroup or own disjoint pools; reference/critic (PPO) and teacher (OPD) stay independent; production NFT requires `colocated` |

```
                grid.backend=local                     grid.backend=ray (grid.ranks=K)
                (launch under torchrun, W ranks)
grouping=       ┌─ torchrun rank w ────────┐           ┌─ Ray worker k ───────────┐
"single"        │  data(env) + policy      │ ×W        │  data(env) + policy      │ ×K
                │  (FSDP shard w of W)     │           │  (FSDP shard k of K)     │
                └──────────────────────────┘           └──────────────────────────┘

grouping=       ✗ rejected                             ┌─ data worker k ──┐ ┌─ policy worker k ──┐
"per_role"        (single process,                     │  data(env) only  │ │  policy (8B shard) │ ×K
                   nothing to split)                   └──────────────────┘ └────────────────────┘
                                                       DeviceGroup decides shared vs disjoint GPUs
```

The `per_role` column is schematic. PPO applies it to data, rollout, reward, replay, reference,
critic, and policy ActorGroups; placement may colocate only policy and rollout, while preserving
independent role processes.

All modes bind through `bind_device_groups` and run the SAME imperative loop over
WorkerGroups — local calls return eager `Handle`s, ray returns `CollectiveHandle`s;
`resolve()` unifies them, so the loop never branches on the mode.

Production online NFT uses the Ray column exclusively: one ActorGroup contains data,
rollout, reward, replay, and policy in each rank process, and one GPU DeviceGroup gives
that actor mesh a single distributed FSDP world. Its algorithm-scoped dispatch profile
routes rank-owned WFM samples exactly by lane; the loop itself does not inspect ranks or
wrap handles at call sites.

Under colocated placement the loop hands each card back and forth within a step: the trainer
parks its state on the host while the rollout generates and the vLLM engine sleeps while the
trainer updates ([`batching.md`](howto/batching.md#colocated-offload-what-the-trainer-holds-while-the-rollout-runs));
`rl.colocated_offload=false` keeps both resident instead when the card has the room and the
hand-off time matters.

`ActorGroup` is the process axis; `DeviceGroup` is the CPU/GPU axis. ActorGroups in one GPU
DeviceGroup share cards by flat actor index, while different GPU DeviceGroups are disjoint. Ray
fractional requests are derived privately and never select topology.

## Layout

Top-level map — one row per concept; details in the sub-tables below.

| area                   | what it is                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `api/`                 | lightweight experimental import boundary for selected data and topology contract types ([`compatibility.md`](reference/compatibility.md))                                                                                                                                                                                                                                                   |
| `schema/`              | unified trajectories, observations, sample provenance and raw environment feedback; no training dependencies                                                                                                                                                                                                                                                                                |
| `embodied/`            | the simulator-side contracts and their plugin registries: `Simulator`/`SimSession`, `EmbodimentMap`, `WorkUnit`, the rollout driver, and the `sims/`/`rollouts/`/`sinks/`/`sources/` surface. Imports stdlib + numpy + `schema` only, so it loads in the simulator venv as well as the trainer's ([`embodied.md`](reference/embodied.md), [`sim-integration.md`](howto/sim-integration.md)) |
| `workers/`             | WorkerGroup substrate: WG/Handle, Ray meshes, and actor-owned NIXL data pools ([`data-plane.md`](howto/data-plane.md))                                                                                                                                                                                                                                                                      |
| `roles/`               | independent simulator/collector/rollout Role contracts and  role factories for env / rollout / reward / buffer / teacher / reference / critic / trainer, plus concrete bodies under the matching role subpackages                                                                                                                                                                           |
| `algorithm/`           | objective implementations, algorithm-local preparation, and reward providers                                                                                                                                                                                                                                                                                                                |
| `comm/`                | data and weight planes: channel, bulk, codec, transport, and versioned weight transfer                                                                                                                                                                                                                                                                                                      |
| `control/`             | control plane: typed ActorGroup/DeviceGroup placement and binding, loop-level checkpoint and validation transactions, experience buffer, and GPU-less manifest checks ([model validation](howto/train-time-validation.md), [manifest validation](howto/preflight.md))                                                                                                                       |
| `checkpoint/`          | shared `CheckpointManager`, local/direct-object-store primary backends, provenance and retention ([checkpointing](reference/checkpoint.md)); `archive/` shares publication and reads across trainer and CLI, `conversion/` owns HF conversion, and `tools/` exposes archive, merge and materialization commands ([checkpoint-artifacts](howto/checkpoint-artifacts.md))                     |
| `callbacks/`           | control-plane observers, including optional telemetry observers                                                                                                                                                                                                                                                                                                                             |
| `config/`              | typed config system with experiment inheritance                                                                                                                                                                                                                                                                                                                                             |
| `model/gr00t/runtime/` | GR00T training image assembly, in-image environment and allocation probes                                                                                                                                                                                                                                                                                                                   |
| `model/pi05/`          | Stateless RLinf OpenPI π0.5 model, processor, selected Flow-SDE and pinned Omni adapter for LIBERO-Spatial                                                                                                                                                                                                                                                                                  |
| `configs/`             | the runnable recipes launched as `--config` — each pins model, rollout engine, algorithm, and topology, so it doubles as the worked example of its algorithm                                                                                                                                                                                                                                |
| `entrypoints/`         | stable per-algorithm run entries, including model-independent Joint PPO ([public Joint PPO guides](howto/run-posttrain.md#8-embodied-ppo))                                                                                                                                                                                                                                                  |
| `launcher/`            | dependency-free run reservation and Slurm parsing shared by operator launchers                                                                                                                                                                                                                                                                                                              |
| `evaluation/`          | result identity/coverage and accepted-attempt state shared by checkpoint-driven evaluation tools                                                                                                                                                                                                                                                                                            |
| `data/datasets/`       | dataset sources (preference JSONL / tar / synthetic; local or S3/GCS SFT JSONL) behind the `roles/env` bodies                                                                                                                                                                                                                                                                               |
| `data/packer/`         | Two shaping seams: DataPacker (token sequences — GRPO/PPO/OPD rollouts + objectives) and preference batch assembly; model-specific GR00T assembly lives under `model/gr00t/` ([`docs/howto/custom-datapacker.md`](howto/custom-datapacker.md))                                                                                                                                              |
| `tests/`               | the whole test suite, `helpers/` fixtures                                                                                                                                                                                                                                                                                                                                                   |
| `utils/`               | the shared multinode training launcher (`launch_multinode.py`); `mesh.py` reads the `NNODES`/`GPUS_PER_NODE` contract                                                                                                                                                                                                                                                                       |
| `docs/`                | architecture, configuration, checkpoint, launch, validation, compatibility, and evaluation guides                                                                                                                                                                                                                                                                                           |
| `pyproject.toml`       | virtual uv project with CUDA runtime groups and an opt-in SFT environment; not an installable package                                                                                                                                                                                                                                                                                       |
| `uv.lock`              | shared frozen resolution for Linux x86_64/aarch64 and the CUDA runtime groups                                                                                                                                                                                                                                                                                                               |

DataPacker (the raw→tensor shaping seam) — usage, registry, adding a packer for a new arch:
[`docs/howto/custom-datapacker.md`](howto/custom-datapacker.md); design rationale/history:
internal design record.

## Checkpoints & resume

Every trainer and model backend uses the shared `CheckpointManager`. Regular cadence and final
saves default to the full exact-resume plan—`model`, `optimizer`, and `training_state`.
Validation does not save a checkpoint by default (`rl.validation.save_checkpoint=false`); when
enabled, its component plan also defaults to full. Configured partial plans are non-resumable
artifacts under the sibling `checkpoint_artifacts/iter_*` namespace and do not advance
`latest_checkpoint.txt`. Preemption and timeout saves always use the full plan.

When `keep_optim_last` is unset, a local primary keeps optimizer state for the newest checkpoint,
while a direct object-store primary resolves to `0` because remote retention is unsupported;
explicit local `0` keeps all optimizer states. Formats, component selection, resume, and storage
modes are documented in [`docs/reference/checkpoint.md`](reference/checkpoint.md).

## Design discipline

1. **Simplifications live in implementations, never in interfaces** — don't bake
   "single process" / "shared model" / "sync" into any signature.
2. **rollout receives weights *through* the weight-transfer edge** (no-op today); it never
   reaches into `policy.model`. Keeps the seam for real resharding + a separate inference
   engine (§8.2 #4).
3. **Control is always imperative** — a plain `train()` loop; topology is explicit wiring
   (WGs + groups), never a graph DSL.
