# Code Structure

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Repository Layout](#repository-layout)
- [The `cosmos_framework/` Package](#the-cosmos_framework-package)
  - [`cosmos_framework/algorithm/`](#cosmos_frameworkalgorithm)
  - [`cosmos_framework/callbacks/`](#cosmos_frameworkcallbacks)
  - [`cosmos_framework/checkpoint/`](#cosmos_frameworkcheckpoint)
  - [`cosmos_framework/communicator/`](#cosmos_frameworkcommunicator)
  - [`cosmos_framework/configs/`](#cosmos_frameworkconfigs)
  - [`cosmos_framework/controller/`](#cosmos_frameworkcontroller)
  - [`cosmos_framework/data/`](#cosmos_frameworkdata)
  - [`cosmos_framework/evaluation/`](#cosmos_frameworkevaluation)
  - [`cosmos_framework/inference/`](#cosmos_frameworkinference)
  - [`cosmos_framework/launcher/`](#cosmos_frameworklauncher)
  - [`cosmos_framework/model/`](#cosmos_frameworkmodel)
  - [`cosmos_framework/tools/`](#cosmos_frameworktools)
  - [`cosmos_framework/trainer/`](#cosmos_frameworktrainer)
  - [`cosmos_framework/utils/`](#cosmos_frameworkutils)
  - [`cosmos_framework/workers/`](#cosmos_frameworkworkers)
- [Supporting Directories](#supporting-directories)
- [Where to Add New Code](#where-to-add-new-code)

______________________________________________________________________

<!--TOC-->

## Repository Layout

```text
Cosmos/
├── cosmos_framework/             # Main package (training infra + inference subpackage)
│   ├── inference/      # Inference subpackage (model, args, defaults, Ray serving, common helpers, SFT experiment configs)
│   └── ...             # Training-infra subpackages: data, model, trainer, callbacks, checkpoint, …
│   └── scripts/        # CLI entry-point scripts: train.py, _train.py, inference.py, eval.py, export_model.py, …
├── packages/           # Backend shim packages: diffusers-cosmos3, transformers-cosmos3, vllm-cosmos3
├── docs/               # User documentation (you are here)
├── docker/             # Dockerfiles for reproducible environments
├── examples/           # Runnable training / fine-tuning / inference examples
├── tests/              # Unit and integration tests
├── tools/              # Standalone CLI utilities (e.g. checkpoint conversion)
├── pyproject.toml      # uv-managed dependency manifest
├── uv.lock             # Pinned dependency graph (do not edit by hand)
└── .python-version     # Python version pin (used by uv)
```

`cosmos_framework/` is the single Python package. Training infrastructure (data, model, trainer, callbacks, checkpoint, utils, …) lives in top-level subpackages; inference (Diffusers / Transformers / vLLM-friendly inference core, Ray serving, per-modality defaults, training-side experiment YAMLs) lives under `cosmos_framework/inference/`. The library-style backend shims that load Cosmos3 checkpoints into upstream ecosystems live under `packages/{diffusers,transformers,vllm}-cosmos3/`.

## The `cosmos_framework/` Package

The `cosmos_framework/` package is organized around the workflow of a large-scale, distributed training run — particularly post-training and reinforcement-learning regimes — with each subpackage owning one concern.

```text
cosmos_framework/
├── algorithm/      # Loss functions, reward models, RL algorithms        [planned]
│   ├── loss/
│   ├── reward/
│   └── rl/
├── callbacks/      # Lifecycle hooks (logging, profiling, eval triggers, checkpoint cadence)
├── checkpoint/     # Saving, loading, conversion (DCP ↔ HF safetensors)
├── communicator/   # Inter-process / inter-worker communication primitives  [planned]
├── configs/        # Pydantic-validated TOML schema + LazyConfig experiment SKUs
│   ├── base/       # config.py, experiment/{action,sft,posttrain_video}/, vlm/, defaults/
│   └── toml_config/# sft_config.py (pydantic schema), toml_config_helper.py
├── controller/     # Top-level orchestration of multi-worker training jobs  [planned]
├── data/           # Dataset loading, batching, augmentation, sharding
├── evaluation/     # Eval harness for trained checkpoints                [planned]
├── inference/      # Inference engine + modality defaults + Ray serving + shared helpers
├── launcher/       # Job launching (Slurm, torchrun, k8s)                [planned]
├── model/          # Model definitions and parallelism wrappers
├── tools/          # In-package CLI utilities (flops/, visualize/)
├── trainer/        # Training loop, optimizer step, gradient accumulation
├── utils/          # Shared low-level utilities (logging, config, distributed helpers, vfm/, vlm/)
└── workers/        # Specialized roles in a distributed RL job           [planned]
    ├── reference/  # Reference / frozen-policy worker (KL anchor)
    ├── reward/     # Reward-model worker
    ├── rollout/    # On-policy rollout generation worker
    └── simulations/# Simulator-driven environment worker
```

> **Note — some subpackages above don't exist yet.** Entries tagged **`[planned]`** (`algorithm/`, `communicator/`, `controller/`, `evaluation/`, `launcher/`, `workers/`) describe the intended home for functionality that has not yet landed in this release. The directories are intentionally absent on disk — they will be created when their respective feature areas (RL training, multi-worker orchestration, the distributed evaluation harness, job launchers, RL worker roles) come online. The "Where to Add New Code" table at the bottom of this document still applies: when you build one of these features, create the matching subpackage and put the code there. Subpackages without the tag (`callbacks/`, `checkpoint/`, `configs/`, `data/`, `inference/`, `model/`, `tools/`, `trainer/`, `utils/`, plus `scripts/` covered separately) exist today.

### `cosmos_framework/algorithm/`

*Planned — not yet present in this release; the subpackage will be created when RL / loss work lands.*

Algorithmic primitives that are independent of the model and trainer.

- `loss/` — supervised and distillation losses (cross-entropy, flow-matching, KL, etc.).
- `reward/` — reward functions and learned reward heads.
- `rl/` — RL update rules (PPO, GRPO, DPO-family) that consume losses and rewards.

Add new objectives here, not inside the trainer.

### `cosmos_framework/callbacks/`

Pluggable lifecycle hooks invoked by the trainer at well-defined points (step begin/end, epoch boundary, eval, save, exception). Use callbacks for cross-cutting concerns such as wandb/W&B logging, gradient clipping, MoE stability monitoring, dataloader-state checkpointing, and learning-rate logging.

### `cosmos_framework/checkpoint/`

All checkpoint I/O lives here:

- DCP (PyTorch Distributed Checkpoint) save/load
- HuggingFace `safetensors` import/export
- Schema migration and resume-from-step logic

### `cosmos_framework/communicator/`

*Planned — not yet present in this release; the subpackage will be created when multi-worker comms land.*

Communication primitives between processes — point-to-point send/recv, broadcast helpers, and any RPC-style channels used between the controller and workers. Keep raw `torch.distributed` / NCCL calls out of business logic; route them through this layer.

### `cosmos_framework/configs/`

Configuration system for training runs:

- `configs/toml_config/` — the user-facing pydantic schema for the structured TOML interface consumed by `cosmos_framework.scripts.train --sft-toml=…`. `sft_config.py` defines `SFTExperimentConfig` (with `extra="forbid"` on every sub-model). `toml_config_helper.py` handles VFM↔VLM path remaps and OmegaConf env interpolation.
- `configs/base/` — internal LazyConfig-based experiment SKUs. `base/experiment/sft/*.py` registers Cosmos3 SFT experiments (e.g. `vision_sft_nano.py`, `vision_sft_super.py`). `base/vlm/` is the VLM-side analogue with its own `config.py`, `experiment/`, `defaults/`, and `freeze_config.py`.

See [`docs/sft_config.md`](./sft_config.md) for the full field-by-field TOML reference.

### `cosmos_framework/controller/`

*Planned — not yet present in this release; the subpackage will be created when multi-worker orchestration lands.*

The orchestrator for a multi-worker job. The controller drives the training loop, hands batches to rollout/reward workers, collects gradients, and decides when to checkpoint or evaluate. Think "head node logic" — there is one controller per job.

### `cosmos_framework/data/`

Datasets, samplers, collators, augmentations, and data-side parallelism (e.g. sequence packing, multi-aspect batching). New dataset formats and new augmentations both live here.

### `cosmos_framework/evaluation/`

*Planned — not yet present in this release; the subpackage will be created when the offline eval harness lands.*

Evaluation harnesses run against trained checkpoints — metrics, dataset-driven eval loops, and reporting. Distinct from `inference/`: evaluation is offline and metric-oriented.

### `cosmos_framework/inference/`

The full inference subpackage:

- `args.py` — sampling/setup args (`SamplingArgs`, `SamplingOverrides`, `OmniSetupArgs`, `OmniSetupOverrides`, `OmniSampleOverrides`), plus the modality-defaults loader and the `_RESOLUTION_SHIFT_DEFAULTS` table.
- `model.py`, `inference.py` — model + inference engine entry points used by `cosmos_framework/scripts/inference.py`.
- `common/` — shared helpers for args, init, config, checkpoints (used by both training and inference scripts).
- `defaults/<mode>/sample_args.json` — per-modality default sample arguments (text2image, text2video, image2video, image2image, video2video, forward_dynamics, inverse_dynamics, policy, reasoner) plus `prompt_upsampler.txt` and `video_captioner.txt` system prompts.
- `ray/` — Ray Serve / Submit / Gradio entry points (`cosmos_framework.inference.ray.serve`, `cosmos_framework.inference.ray.submit`, `cosmos_framework.inference.ray.gradio`) and their YAML configs under `ray/configs/`.
- `configs/{checkpoint,model}/` — per-checkpoint and per-model inference configs.
- Modality entry points: `vision.py`, `action.py`, `sound.py`, `transfer.py`, `interactive.py`, `prompt_upsampling.py`, `dataset.py`.

Training-side experiment SKUs live separately at `cosmos_framework/configs/base/experiment/sft/*.py` (see [`cosmos_framework/configs/`](#cosmos_frameworkconfigs)) — not under `inference/`.

Library-style backend shims that adapt Cosmos3 checkpoints to the Diffusers / Transformers / vLLM ecosystems live separately under `packages/{diffusers,transformers,vllm}-cosmos3/`.

### `cosmos_framework/launcher/`

*Planned — not yet present in this release; the subpackage will be created when launcher back-ends land.*

Job launching back-ends: Slurm, `torchrun`, and Kubernetes adapters. Selects the launch path based on the environment and forwards process rank/world-size to the controller.

### `cosmos_framework/model/`

Model architectures and the parallelism wrappers around them (FSDP, tensor parallel, context parallel, pipeline parallel). The trainer is model-agnostic; everything the trainer touches goes through this layer.

### `cosmos_framework/tools/`

CLI entry points surfaced from the package (as opposed to standalone scripts in the top-level `tools/`). Use this for utilities that need to import `cosmos_framework.*` internals.

### `cosmos_framework/trainer/`

The training loop itself — gradient accumulation, optimizer step, scheduler step, mixed-precision policy, and the dispatcher that fires callbacks. Stays narrow on purpose: model details live in `model/`, algorithm details in `algorithm/`.

### `cosmos_framework/utils/`

Shared low-level helpers (logging, config loading, distributed setup, profiling). Keep this folder *thin* — anything substantial should grow into its own subpackage.

### `cosmos_framework/workers/`

*Planned — not yet present in this release; the subpackage will be created when RL worker roles land.*

Specialized worker roles for distributed RL jobs. Each worker is a long-running process the controller talks to:

- `reference/` — frozen reference policy (for KL anchoring in PPO/GRPO/DPO).
- `reward/` — reward-model worker; computes scalar rewards for rollouts.
- `rollout/` — on-policy generation worker; samples trajectories from the current policy.
- `simulations/` — simulator-backed environment worker (used when reward comes from a sim rather than a learned model).

Add new worker types as sibling subpackages — each owns its own startup, message loop, and shutdown.

## Supporting Directories

- `tests/` — pytest tests, mirroring the `cosmos_framework/` package layout.
- `examples/` — runnable end-to-end examples; see `examples/README.md`.
- `docker/` — Dockerfiles and image build helpers; see `docker/README.md`.
- `cosmos_framework/scripts/` — CLI entry-point scripts (`train.py`, `inference.py`, `eval.py`, `export_model.py`, …); invoke as `python -m cosmos_framework.scripts.<name>`. Primary training entry point: `cosmos_framework.scripts.train` driven by a structured, pydantic-validated TOML interface (`--sft-toml=<recipe-toml>`); recipe TOMLs live under [`examples/toml/sft_config/`](../examples/toml/sft_config/) and the schema is defined in [`cosmos_framework/configs/toml_config/sft_config.py`](../cosmos_framework/configs/toml_config/sft_config.py) — see [`examples/README.md`](../examples/README.md) and [`docs/training.md`](./training.md).
- `packages/` — library-style backend shims: `packages/{diffusers,transformers,vllm}-cosmos3/`, each installable independently.

## Where to Add New Code

| You want to add…                            | Put it in…                                                                       |
| ------------------------------------------- | -------------------------------------------------------------------------------- |
| A new loss function                         | `cosmos_framework/algorithm/loss/`                                               |
| A new RL update rule                        | `cosmos_framework/algorithm/rl/`                                                 |
| A new reward function or head               | `cosmos_framework/algorithm/reward/`                                             |
| A new model architecture                    | `cosmos_framework/model/`                                                        |
| A new dataset format / augmentation         | `cosmos_framework/data/`                                                         |
| A new training callback                     | `cosmos_framework/callbacks/`                                                    |
| A new checkpoint format or converter        | `cosmos_framework/checkpoint/`                                                   |
| A new launcher back-end (Slurm flavor, k8s) | `cosmos_framework/launcher/`                                                     |
| A new RL worker role                        | `cosmos_framework/workers/<new_role>/`                                           |
| A new evaluation suite                      | `cosmos_framework/evaluation/`                                                   |
| A new runnable example                      | `examples/`                                                                      |
| A new standalone CLI tool                   | `tools/` (repo root) for non-cosmos imports, otherwise `cosmos_framework/tools/` |
| A new test                                  | `tests/` mirroring the package path                                              |
