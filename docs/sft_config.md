# SFT Structured-TOML Config Reference

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Overview](#overview)
- [TOML at a glance](#toml-at-a-glance)
- [`[job]`](#job)
- [`[model]`](#model)
  - [`[model.ema]`](#modelema)
  - [`[model.parallelism]`](#modelparallelism)
  - [`[model.compile]`](#modelcompile)
  - [`[model.activation_checkpointing]`](#modelactivation_checkpointing)
  - [`[model.tokenizer]` (VFM only)](#modeltokenizer-vfm-only)
  - [`[model.backbone]` (VLM only)](#modelbackbone-vlm-only)
- [`[optimizer]`](#optimizer)
- [`[scheduler]`](#scheduler)
- [`[trainer]`](#trainer)
  - [`[trainer.callbacks.compile_tokenizer]` (VFM only)](#trainercallbackscompile_tokenizer-vfm-only)
  - [`[trainer.callbacks.grad_clip]`](#trainercallbacksgrad_clip)
- [`[checkpoint]`](#checkpoint)
- [`[dataloader_train]`](#dataloader_train)
- [`[custom]` (free-form escape hatch)](#custom-free-form-escape-hatch)
- [Cross-cutting behaviors](#cross-cutting-behaviors)
  - [`"???"` (MISSING) sentinel](#-missing-sentinel)
  - [Env interpolation](#env-interpolation)
  - [VFM ↔ VLM path remaps](#vfm--vlm-path-remaps)
  - [Out-of-schema knobs (Hydra tail overrides)](#out-of-schema-knobs-hydra-tail-overrides)
  - [Loading flow](#loading-flow)
- [Extending the schema](#extending-the-schema)

______________________________________________________________________

<!--TOC-->

## Overview

Every SFT recipe under `examples/toml/sft_config/<recipe>.toml` is parsed against the pydantic schema [`SFTExperimentConfig`](../cosmos_framework/configs/toml_config/sft_config.py). Each top-level TOML section (`[job]`, `[model]`, …) maps to one sub-model in that file. The schema is strict — every sub-model sets `extra="forbid"`, so an unknown key raises `ValidationError` before training starts (typo guard).

After validation, the TOML dict is converted to a Hydra override list by [`build_hydra_overrides`](../cosmos_framework/configs/toml_config/toml_config_helper.py) (see [VFM ↔ VLM path remaps](#vfm--vlm-path-remaps)), and `load_experiment_from_toml(...)` loads the base config (chosen by `[job].task`) and applies the overrides via Hydra. Trailing CLI overrides passed after `--` to `cosmos_framework.scripts.train` are appended last, so they win over TOML values.

## TOML at a glance

```toml
[job]                                # run identity + base-config / experiment selector
[model]                              # top-level model knobs
[model.ema]                          # EMA tracking of generation-pathway weights
[model.parallelism]                  # FSDP / context-parallel / CFG-parallel topology
[model.compile]                      # torch.compile knobs
[model.activation_checkpointing]     # AC mode + recompute knobs
[model.tokenizer]                    # VFM only: Wan VAE
[model.backbone]                     # VLM only: backbone HF id + optional safetensors path
[optimizer]                          # AdamW
[optimizer.lr_multipliers]           # optional per-substring lr multipliers
[scheduler]                          # LambdaLinear / LambdaCosine
[trainer]                            # max_iter, grad_accum, logging
[trainer.callbacks.compile_tokenizer]  # VFM only
[trainer.callbacks.grad_clip]        # clip_norm + force_finite
[checkpoint]                         # load_path, save_iter, key-skip blocklist
[dataloader_train]                   # top-level scalars only
[custom]                             # free-form, project-owned escape hatch (opaque to the framework)
```

The full pipeline (dataloader class, dataset wiring, model_instance LazyCall, etc.) lives in the experiment SKU Python file under `cosmos_framework/configs/base/experiment/sft/<recipe>.py`. The TOML only surfaces values the recipe author wants users to tune.

## `[job]`

Run identity + meta-fields that pick the Hydra config tree to load.

| field        | default      | description                                                                                                                                                                                                                               |
| ------------ | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`       | `"vfm"`      | **META** — chooses which `make_config()` to call: `"vfm"` → `cosmos_framework/configs/base/config.py`, `"vlm"` → `cosmos_framework/configs/base/reasoner/config.py`. Also picks the path-remap rules in `toml_config_helper.PATH_REMAPS`. |
| `experiment` | `""`         | **META** — names the Hydra experiment LazyDict registered in `ConfigStore` under `experiment/<name>`. Resolved at load time via `experiment=<name>` (e.g. `vision_sft_nano`).                                                             |
| `project`    | `""`         | W&B project (team-level bucket). Flows to `config.job.project`.                                                                                                                                                                           |
| `group`      | `""`         | W&B sub-label for clustering related runs (e.g. `"sft"`). Flows to `config.job.group`.                                                                                                                                                    |
| `name`       | `""`         | W&B run name; forms part of the output dir `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/`. Leave empty (or use `${now:%Y-%m-%d}_${now:%H-%M-%S}`) for auto-timestamped subdir.                                                       |
| `wandb_mode` | `"disabled"` | `"online"` (real-time, needs `WANDB_API_KEY`), `"offline"` (log locally, sync later via `wandb sync`), or `"disabled"`.                                                                                                                   |

## `[model]`

Top-level model knobs. Lands at `model.config.*` on VFM and on VLM; sub-tree paths are remapped per the [VFM ↔ VLM path remaps](#vfm--vlm-path-remaps).

| field                          | default                                                         | description                                                                                                                                                                                                                                      |
| ------------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `max_num_tokens_after_packing` | `13312`                                                         | Token-packing target: max tokens after sequence packing. `-1` disables the cap. **VFM only** — VLM uses `data_setting.max_tokens` (tail override).                                                                                               |
| `joint_attn_implementation`    | `"two_way"`                                                     | VFM attention layout: `"two_way"` (U/G blocks with cross-attention), `"three_way"` (adds sparsity-aware third block — NATTEN), or `"flex"` (legacy). **VFM only.**                                                                               |
| `attn_implementation`          | `"cosmos"`                                                      | VLM HF attention impl: `"cosmos"` (NATTEN/Blackwell-FMHA wrapper), `"flash_attention_2"`, `"sdpa"`, or `"eager"`. **VLM only.**                                                                                                                  |
| `lora_enabled`                 | `false`                                                         | Inject LoRA adapters into the generation pathway BEFORE FSDP wraps the network. Pair with `optimizer.keys_to_select=["lora_"]` and `checkpoint.keys_to_skip_loading=[…, "lora_"]`. Used by SUPER-tier recipes; NANO leaves it off. **VFM only.** |
| `lora_rank`                    | `16`                                                            | LoRA rank `r`. Adapter shape is (rank × hidden_dim) per target module. Typical: 4 / 8 / 16 / 32.                                                                                                                                                 |
| `lora_alpha`                   | `32`                                                            | LoRA scaling factor. Effective magnitude is `alpha / rank`; rank=16 alpha=32 → 2× scale.                                                                                                                                                         |
| `lora_target_modules`          | `"q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"` | Comma-separated substrings of param names that receive an adapter. Default targets the four MoE-gen projection matrices.                                                                                                                         |
| `precision`                    | `"bfloat16"`                                                    | Compute dtype for forward/backward (`MixedPrecisionPolicy.param_dtype`). `"bfloat16"` is standard for Hopper/Blackwell. (Was `[model.parallelism].precision` before the `ParallelismConfig` split.)                                              |

### `[model.ema]`

Exponential Moving Average of generation-pathway weights. Lands at `model.config.ema.*` on both VFM and VLM. When enabled, the trainer keeps a second fp32 copy of trainable params updated as `ema_w = (1 - rate^k) · w_curr + rate^k · ema_w_prev`. EMA weights are used for inference; live weights keep training.

| field             | default | description                                                                                                                                                    |
| ----------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`         | `true`  | Turn EMA tracking on/off. Full fine-tunes typically enable it; LoRA recipes leave it off because the adapter weights are tiny.                                 |
| `rate`            | `0.1`   | Base EMA decay. Lower = slower decay = EMA tracks live weights more tightly. Per-step rate is ramped by the iteration counter so the EMA "warms up" from init. |
| `iteration_shift` | `0`     | Step offset added before computing the warmup ramp. Use a positive value when resuming so the EMA doesn't reset to "early-iter" decay strength.                |

### `[model.parallelism]`

FSDP / context-parallel / classifier-free-guidance topology. Lands at `model.config.parallelism.*` on both VFM and VLM.

| field                            | default | description                                                                                                                                       |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_parallel_shard_degree`     | `-1`    | FSDP shard degree. `-1` = auto-fit `WORLD_SIZE` from torchrun. Set explicitly to make the run fail loudly on the wrong GPU count.                 |
| `data_parallel_replicate_degree` | `1`     | HSDP outer replicate degree. `>1` runs the same shard topology N times in parallel; usually only needed at very large cluster scale.              |
| `context_parallel_shard_degree`  | `1`     | Splits the sequence dimension across this many ranks so long-context models fit in memory. Used by super-tier configs (e.g. DP=4, CP=2 → 8 GPUs). |
| `cfg_parallel_shard_degree`      | `1`     | Splits the duplicated conditional/unconditional CFG forward across ranks. Almost always `1` for SFT.                                              |

The product `data_parallel_shard_degree × data_parallel_replicate_degree × context_parallel_shard_degree × cfg_parallel_shard_degree` must equal `WORLD_SIZE`.

### `[model.compile]`

`torch.compile` knobs. Lands at `model.config.compile.*` on both VFM and VLM. Both fields used to live on `[model.parallelism]` — the rename is the only behavior change.

| field             | default | description                                                                                                                                                                |
| ----------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`         | `false` | `torch.compile` the network. (Was `[model.parallelism].use_torch_compile`.) Big speedup on stable shapes; conflicts with some custom CUDA kernels and deterministic modes. |
| `compile_dynamic` | `true`  | When `enabled=true`, recompile per-shape rather than specializing for a single static shape. Required for the `compile_tokenizer` callback's progressive warmup.           |

### `[model.activation_checkpointing]`

Recompute activations during backward to trade FLOPs for memory. Lands at `model.config.activation_checkpointing.*` on both VFM and VLM.

| field                | default     | description                                                                                                                                                                                      |
| -------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mode`               | `"full"`    | `"selective"` (per-op SAC — keep matmuls/FMHA, recompute the rest; MoT path only), `"full"` (checkpoint each whole transformer block), or `"none"` (no checkpointing — fastest, highest memory). |
| `save_ops_regex`     | `["fmha"]`  | Regex patterns for ops to KEEP saved under `mode="selective"`. Ignored in `"full"`/`"none"`. Default keeps flash/multi-head-attention outputs.                                                   |
| `preserve_rng_state` | `true`      | Stash + restore CUDA RNG across recompute boundaries. Required for deterministic equivalence with the non-checkpointed path; small slowdown.                                                     |
| `determinism_check`  | `"default"` | Forwarded to `torch.utils.checkpoint`. `"default"` disables the extra determinism check; `"match"` cross-checks recomputed activations against the original (debug-only, very slow).             |

### `[model.tokenizer]` (VFM only)

Video tokenizer (VAE) settings. **VLM skips this sub-tree** (path-remap blocks it).

| field      | default                                                | description                                                                                                             |
| ---------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `vae_path` | `"pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"` | Path to `Wan2.2_VAE.pth`. SFT recipes typically pass this via env interpolation: `vae_path = "${oc.env:WAN_VAE_PATH}"`. |

### `[model.backbone]` (VLM only)

Foundation backbone settings. **VFM skips this sub-tree** — VFM keeps its backbone wiring inline in the experiment Python (`vlm_config.model_instance`).

| field              | default           | description                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------ | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`       | `"???"` (MISSING) | HF repo ID or local snapshot path of the VLM backbone (e.g. `"Qwen/Qwen3-VL-8B-Instruct"`). Drives `AutoConfig` + `AutoModel` selection (architecture). Remapped to `model.config.policy.backbone.model_name`.                                                                                                                                                                                                  |
| `safetensors_path` | `"???"` (MISSING) | Optional local path to a `.safetensors` file (or directory) used for weight loading. When set, overrides the auto-downloaded snapshot under `model_name`; the architecture is still driven by `model_name`. Useful for pointing at a converted/finetuned checkpoint while keeping the public HF `model_name` for tokenizer/architecture discovery. Remapped to `model.config.policy.backbone.safetensors_path`. |

## `[optimizer]`

AdamW-family optimizer parameters. Same shape on VFM and VLM (`eps` skipped on VLM).

| field            | default       | description                                                                                                                                                                           |
| ---------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `betas`          | `[0.9, 0.99]` | Adam β1, β2 — gradient and squared-gradient EMAs. Standard pair is `(0.9, 0.999)`; SFT recipes commonly use `(0.9, 0.99)` or `(0.9, 0.95)` for tighter tracking of recent gradients.  |
| `eps`            | `1.0e-8`      | Adam numerical-stability epsilon. `1e-8` is PyTorch default; `1e-6` is sometimes used in bf16 to avoid underflow in the squared-gradient denominator. **VFM only.**                   |
| `fused`          | `true`        | Use the fused AdamW kernel. Faster on modern GPUs; slightly different numerical behavior vs the foreach implementation.                                                               |
| `keys_to_select` | `[]`          | Substring allowlist for params that the optimizer trains. Empty = train everything. `["lora_"]` = LoRA-only fine-tune (freezes everything except adapters).                           |
| `lr`             | `2.0e-4`      | Base learning rate.                                                                                                                                                                   |
| `lr_multipliers` | `{}`          | Per-param-group LR multipliers (`<substring> = <multiplier>`). E.g. `action_modality_embed = 5.0` gives that param group 5× the base lr. Substrings not in the dict default to `1.0`. |
| `weight_decay`   | `0.0`         | AdamW decoupled weight decay. `0` disables.                                                                                                                                           |

## `[scheduler]`

LambdaLinear / LambdaCosine LR scheduler. All four `f_*` values are **ratios of `optimizer.lr`** — effective LR at the corresponding milestone = `lr × f_x`. Each list has one entry per scheduler cycle.

| field                | default    | description                                                                                                                                   |
| -------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `cycle_lengths`      | `[20000]`  | Length of each cycle in optimizer steps. With one entry, the scheduler completes one full warmup→peak→trough cycle over that many iterations. |
| `f_max`              | `[1.0]`    | Peak LR multiplier reached at the end of warmup.                                                                                              |
| `f_min`              | `[0.0]`    | Final LR multiplier at the end of each cycle (the "floor"). For LambdaCosine the LR decays toward `lr × f_min`.                               |
| `f_start`            | `[1.0e-6]` | Initial LR multiplier at step 0, before warmup ramps up.                                                                                      |
| `verbosity_interval` | `0`        | How often the scheduler logs current LR (in optimizer steps). `0` = silent. **VFM only.**                                                     |
| `warm_up_steps`      | `[100]`    | Linear warmup duration in optimizer steps. LR ramps from `lr × f_start` to `lr × f_max` linearly before cosine/linear decay begins.           |

## `[trainer]`

| field                     | default  | description                                                                                                                         |
| ------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `distributed_parallelism` | `"fsdp"` | Distributed strategy. `"fsdp"` is the only supported value today.                                                                   |
| `grad_accum_iter`         | `1`      | Micro-batches accumulated before each `optimizer.step()`. Effective global batch = `grad_accum_iter × per-rank batch × world_size`. |
| `logging_iter`            | `50`     | Console / W&B log frequency (in optimizer steps).                                                                                   |
| `max_iter`                | `500`    | Total number of optimizer steps the run will execute.                                                                               |

### `[trainer.callbacks.compile_tokenizer]` (VFM only)

Lazy `torch.compile` of the VAE tokenizer once shapes stabilize. **VLM skips this** — no tokenizer to compile.

| field                      | default | description                                                                                                                                                                                                                                               |
| -------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`                  | `true`  | Master switch for the callback.                                                                                                                                                                                                                           |
| `compile_after_iterations` | `3`     | Wait this many training iterations after start before triggering the compile (lets one-shot init / dataloader settle).                                                                                                                                    |
| `warmup_resolutions`       | `null`  | Resolutions to "prime" the compile cache with. The callback runs the tokenizer once per listed resolution so the compiled graph for each is ready before training hits it. `null` = use whatever resolutions the tokenizer's `encode_chunk_frames` knows. |

### `[trainer.callbacks.grad_clip]`

| field          | default | description                                                                                                                                        |
| -------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clip_norm`    | `1.0`   | Maximum global L2 norm of the gradient. Steps with a larger norm are rescaled so `‖grad‖ ≤ clip_norm`.                                             |
| `force_finite` | `true`  | When `true`, replace NaN/Inf grads with zero before the step (treats them as no-op rather than crashing). VFM default `true`; VLM default `false`. |

## `[checkpoint]`

Resume + save policy. Lands at `config.checkpoint.*`.

| field                  | default           | description                                                                                                                                                                                                                                                 |
| ---------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `keys_to_skip_loading` | `[]`              | Substring blocklist applied at load time. Any tensor whose FQN contains one of these substrings is skipped (kept at fresh init). Used to mask EMA / LoRA / action tensors when warm-starting from a base checkpoint that doesn't have them.                 |
| `load_path`            | `"???"` (MISSING) | Path to the checkpoint directory to load. The MISSING sentinel is skipped from the override list, so the user must provide a real path at runtime — typically via env interpolation `"${oc.env:BASE_CHECKPOINT_PATH}"` in the TOML, or a CLI tail override. |
| `save_iter`            | `100`             | Save a new checkpoint every N optimizer steps.                                                                                                                                                                                                              |

## `[dataloader_train]`

Top-level dataloader scalars only. The dataloader's class (LazyCall) and full pipeline wiring (datasets, packers, …) stay in the experiment Python — they vary too much between VFM `IterativeJointDataLoader`, `PackingDataLoader`, and VLM `DataPackerDataLoader` to model uniformly.

| field                   | default | description                                                                                                                                                                      |
| ----------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_samples_per_batch` | `null`  | Cap on samples per micro-batch. Remapped to `max_batch_size` on the VLM `DataPackerDataLoader`. `null` = no per-count cap (the packer's token budget is what limits batch size). |
| `max_sequence_length`   | `null`  | Cap on tokens per packed sequence. Remapped to `max_tokens` on the VLM `DataPackerDataLoader`. `null` = no per-token cap.                                                        |
| `seed`                  | `42`    | Dataloader RNG seed. **VFM only** — skipped on VLM (DataPackerDataLoader has no `seed` ctor kwarg).                                                                              |

## `[custom]` (free-form escape hatch)

`[custom]` lets a project carry its own config (dataset paths, sampling ratios, …) in the **same** TOML as the framework knobs. The framework never looks inside it — it's the one section exempt from the `extra="forbid"` typo guard (every other section still rejects unknown keys).

How it works:

- **Arbitrary nested content** passes through verbatim — scalars, sub-tables (`[custom.a.b]`), arrays-of-tables (`[[custom.items]]`).
- It does **not** go through Hydra. After `load_config` finishes, the table is attached as a plain `dict` via `config.custom = raw.get("custom", {})` (or `{}` when absent — reading `config.custom` is always safe).
- So values must be **concrete**: `${custom}` interpolation is **not** supported, and `config.custom` is **not** part of `config.to_dict()` / serialized config dumps.

```toml
[custom]
your_custom_files = "custom_value"
```

Read it directly to wire your own pipeline:

```python
project_cfg = TrainingDatasetConfig.model_validate(config.custom)
```

## Cross-cutting behaviors

### `"???"` (MISSING) sentinel

A handful of fields default to the literal string `"???"` — the OmegaConf MISSING sentinel. `build_hydra_overrides` recognizes this value and emits **no** override for the corresponding key (see `_emit_with_remap` in `toml_config_helper.py`). The effect: if the TOML doesn't explicitly set the field, the value falls through to whatever the experiment Python (or its Hydra base config) sets — instead of emitting `key=''` which would overwrite the inherited value with empty string.

Fields with this pattern today:

- `[checkpoint].load_path`
- `[model.backbone].model_name`
- `[model.backbone].safetensors_path`

### Env interpolation

Recipe TOMLs typically interpolate paths from the environment so the same TOML works across filesystems:

```toml
[checkpoint]
load_path = "${oc.env:BASE_CHECKPOINT_PATH}"

[model.tokenizer]
vae_path = "${oc.env:WAN_VAE_PATH}"
```

`DATASET_PATH` follows the same convention but is consumed inside the experiment-SKU Python (`cosmos_framework/configs/base/experiment/sft/<recipe>.py`), not in the TOML.

### VFM ↔ VLM path remaps

The same TOML key lands at different Hydra paths depending on `[job].task`:

| TOML path                                                                                | VFM (`task="vfm"`) Hydra path             | VLM (`task="vlm"`) Hydra path             |
| ---------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| `model.<X>`                                                                              | `model.config.<X>`                        | `model.config.<X>`                        |
| `model.parallelism.*`                                                                    | `model.config.parallelism.*`              | `model.config.parallelism.*`              |
| `model.compile.*`                                                                        | `model.config.compile.*`                  | `model.config.compile.*`                  |
| `model.activation_checkpointing.*`                                                       | `model.config.activation_checkpointing.*` | `model.config.activation_checkpointing.*` |
| `model.precision`                                                                        | `model.config.precision`                  | `model.config.precision`                  |
| `model.attn_implementation`                                                              | *(skipped — VLM-only)*                    | `model.config.policy.attn_implementation` |
| `model.backbone.*`                                                                       | *(skipped — VLM-only)*                    | `model.config.policy.backbone.*`          |
| `model.ema.*`                                                                            | `model.config.ema.*`                      | `model.config.ema.*`                      |
| `model.tokenizer.*`                                                                      | `model.config.tokenizer.*`                | *(skipped — VFM-only)*                    |
| `model.{max_num_tokens_after_packing, joint_attn_implementation, lora_*}`                | passes through                            | *(skipped — VFM-only)*                    |
| `dataloader_train.max_samples_per_batch`                                                 | passes through                            | `dataloader_train.max_batch_size`         |
| `dataloader_train.max_sequence_length`                                                   | passes through                            | `dataloader_train.max_tokens`             |
| `dataloader_train.seed`                                                                  | passes through                            | *(skipped — VLM has no seed kwarg)*       |
| `optimizer.eps`, `scheduler.verbosity_interval`, `trainer.callbacks.compile_tokenizer.*` | passes through                            | *(skipped — VLM has no analog)*           |

Authoritative source: `PATH_REMAPS` in [`toml_config_helper.py`](../cosmos_framework/configs/toml_config/toml_config_helper.py).

### Out-of-schema knobs (Hydra tail overrides)

A few useful knobs aren't currently modeled by `SFTExperimentConfig` because they're either niche or experiment-specific. Pass them as trailing `key.path=value` positionals after `--`:

| key                                                              | purpose                                                                                                     | used by                                                            |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `data_setting.max_tokens`                                        | VLM token-packing cap (the VLM analogue of `[model].max_num_tokens_after_packing`).                         | `launch_sft_llava_ov.sh` (when the launcher overrides the default) |

### Loading flow

`load_experiment_from_toml(toml_path, extra_overrides)` (in `sft_config.py`) is the end-to-end loader. It:

1. Reads the TOML with `tomllib`.
2. Validates the parsed dict against `SFTExperimentConfig` (raises `ValidationError` on unknown keys).
3. Picks the base config from `[job].task`: `TASK_TO_BASE_CONFIG["vfm"|"vlm"]`.
4. Calls `build_hydra_overrides(raw)` to produce a `["--", "experiment=<name>", "k.p=v", …]` list with per-task remaps applied and MISSING values filtered. `[custom]` is skipped here (it is injected verbatim in step 7, not per-leaf-remapped).
5. Appends `extra_overrides` (CLI tail) so they take precedence over the TOML.
6. Calls `cosmos_framework.utils.config.load_config(base_config_path, overrides)`, which imports the base config module and runs `make_config()` (registers every config group and imports every experiment SKU's `cs.store(group="experiment", …)`), then `override(config, overrides)` has Hydra `compose` resolve the `experiment=<name>` selector against `ConfigStore` and apply the dotted-path overrides.
7. Injects `[custom]` after loading: `config.custom = raw.get("custom", {})`. This runs **after** Hydra resolution, so it lands as a plain `dict` (no `${custom}` interpolation; not part of serialized config dumps).

The returned `Config` is ready for `launch()`.

## Extending the schema

To surface a new knob in the TOML:

1. **Add a `Field(default=…, description="…")` line** to the relevant sub-model in `cosmos_framework/configs/toml_config/sft_config.py`. Pick a sensible default; if the field should fall through to the experiment Python's value when omitted, use `"???"`.
2. **(Per-task wiring only)** If the new key needs to land at a different Hydra path on VFM vs VLM, or should be skipped on one task, add an entry to `PATH_REMAPS` in `cosmos_framework/configs/toml_config/toml_config_helper.py`. Plain pass-through doesn't need a remap.
3. **(Optional)** Add the field to one of the example TOMLs under `examples/toml/sft_config/` so users have a working reference.

`extra="forbid"` on every sub-model means **forgetting step 1 will make any TOML that uses the new key fail validation with a clear error**, so the schema can't silently diverge from real usage.
