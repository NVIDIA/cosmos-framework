---
name: cosmos3-codebase-nav
description: >
  Navigate the Cosmos3 package codebase to find where parameters, configs, defaults,
  scripts, and documentation live. Use when the user asks "where is X in cosmos3",
  "how do I find the config for Y", "where are the defaults", "where do I change a
  parameter", or any question about locating files, modules, or settings. Also use
  when the user opens or edits files and needs orientation.
---

# Cosmos3 Codebase Navigation

## When to use this skill

- Use this skill when an agent is navigating the Cosmos3 package
- Use this skill to answer "where is X", "how do I find the config for Y", or any file-location question
- Use this skill when the user opens or edits cosmos3 files and needs orientation

## Path convention

All paths below are relative to this file's location (`.claude/skills/cosmos3-codebase-nav/`). The repo is laid out as:

- `cosmos_framework/` — main training package (data, model, trainer, callbacks, checkpoint, utils, …).
- `cosmos_framework/configs/base/experiment/` — vfm (generator) experiment SKUs referenced by `[train.train_policy].experiment` in the recipe TOMLs.
- `cosmos_framework/configs/base/reasoner/experiment/` — vlm (reasoner) experiment SKUs.
- `cosmos_framework/inference/` — inference subpackage (args, model, inference engine, defaults, Ray serving, common helpers).
- `cosmos_framework/scripts/` — top-level entry-point scripts (train, inference, eval, export_model, convert_model_to_dcp, upsample_prompts, caption_from_video, captions_to_sft_jsonl, action_policy_server, …). Invoked as `python -m cosmos_framework.scripts.<name>`.
- `examples/toml/sft_config/<recipe>.toml` + `examples/launch_sft_<recipe>.sh` — paired SFT recipes (training entry-point input). The shell sources `examples/_sft_launcher_common.sh`, which forwards into `cosmos_framework.scripts.train --sft-toml=...`.
- `cosmos_framework/configs/toml_config/` — pydantic schemas (`sft_config.py`) and helpers that validate the recipe TOML at load time.

## Quick Reference

### Where parameters and defaults live

| What you're looking for                                   | File                                                                                                                              |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Sampling params (num_steps, guidance, shift, fps, etc.)   | `../../../cosmos_framework/inference/args.py` → `SamplingArgs`, `SamplingOverrides`                                               |
| Per-modality default values                               | `../../../cosmos_framework/inference/defaults/<mode>/sample_args.json`                                                            |
| Setup params (parallelism, checkpoints, model path)       | `../../../cosmos_framework/inference/args.py` → `OmniSetupArgs`, `OmniSetupOverrides`                                             |
| Common args base classes                                  | `../../../cosmos_framework/inference/common/args.py` → `ArgsBase`, `OverridesBase`                                                |
| Ray serving parallelism presets                           | `../../../cosmos_framework/inference/ray/configs/latency.yaml`, `../../../cosmos_framework/inference/ray/configs/throughput.yaml` |
| Feature flags                                             | `../../../cosmos_framework/utils/flags.py`                                                                                        |
| Prompt upsampler system prompt                            | `../../../cosmos_framework/inference/defaults/prompt_upsampler.txt`                                                               |
| Video captioner system prompt                             | `../../../cosmos_framework/inference/defaults/video_captioner.txt`                                                                |
| SFT recipe TOMLs (paired with `examples/launch_sft_*.sh`) | `../../../examples/toml/sft_config/<recipe>.toml`                                                                                 |
| SFT pydantic schema (validates the recipe TOML)           | `../../../cosmos_framework/configs/toml_config/sft_config.py`                                                                     |
| Training experiment SKUs (vfm)                            | `../../../cosmos_framework/configs/base/experiment/`                                                                              |
| Training experiment SKUs (vlm / reasoner)                 | `../../../cosmos_framework/configs/base/reasoner/experiment/`                                                                     |
| Example inputs                                            | `../../../inputs/omni/t2i.json`, `../../../inputs/omni/t2v.json`, `../../../inputs/omni/i2v.json`, …                              |

Available modality modes for defaults: `text2image`, `text2video`, `image2video`, `image2image`, `video2video`, `forward_dynamics`, `inverse_dynamics`, `policy`.

### Config defaults resolution chain

When a user runs inference, default parameter values are resolved in this order:

```
cosmos_framework/inference/defaults/<mode>/sample_args.json     # 1. Per-modality JSON defaults (num_steps, guidance, shift, fps, etc.)
        ↓
_load_modality_defaults() in cosmos_framework/inference/args.py # 2. Loaded and cached at import time
        ↓
SamplingArgs / SamplingOverrides                      # 3. Pydantic models with field-level validation
        ↓
OmniSampleOverrides.build_sample()                    # 4. Merges user overrides → final resolved args
        ↓
_RESOLUTION_SHIFT_DEFAULTS[model_size, resolution]    # 5. Model+resolution shift override (if user didn't set shift)
        ↓
CLI flags (--guidance, --shift, etc.)                 # 6. User overrides from command line
```

The `_RESOLUTION_SHIFT_DEFAULTS` table in `../../../cosmos_framework/inference/args.py` (on `OmniSampleOverrides`) overrides the default `shift` based on model size and resolution, unless the user explicitly specified `--shift`.

| Mode          | Default file                                                                | Key defaults                                   |
| ------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| `text2image`  | `../../../cosmos_framework/inference/defaults/text2image/sample_args.json`  | `num_frames=1`, `guidance=6.0`, `shift=10.0`   |
| `text2video`  | `../../../cosmos_framework/inference/defaults/text2video/sample_args.json`  | `num_frames=189`, `guidance=6.0`, `shift=10.0` |
| `image2video` | `../../../cosmos_framework/inference/defaults/image2video/sample_args.json` | `num_frames=189`, `guidance=6.0`, `shift=10.0` |

Action and video2video modes also have defaults under `cosmos_framework/inference/defaults/{image2image,video2video,forward_dynamics,inverse_dynamics,policy}/sample_args.json`.

Users can also supply a custom defaults file per-request via the `defaults_file` field in sample arguments (see `../../../docs/inference.md`).

### Where to make changes

| Task                            | Edit                                                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Change a built-in default value | `../../../cosmos_framework/inference/defaults/<mode>/sample_args.json`                                                     |
| Add a new CLI parameter         | `SamplingArgs` + `SamplingOverrides` in `../../../cosmos_framework/inference/args.py`, then add to each `sample_args.json` |
| Change parallelism presets      | `../../../cosmos_framework/inference/ray/configs/latency.yaml` or `throughput.yaml`                                        |
| Add a new script                | `../../../cosmos_framework/scripts/` — follow `inference.py` as the pattern                                                |

### Key entry points

| Entry point                        | How to run                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| Batch inference                    | `python -m cosmos_framework.scripts.inference`                                               |
| Training                           | `python -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml` |
| Online serving (Ray)               | `python -m cosmos_framework.inference.ray.serve`                                             |
| Submit to Ray server               | `python -m cosmos_framework.inference.ray.submit`                                            |
| Gradio UI                          | `python -m cosmos_framework.inference.ray.gradio`                                            |
| Prompt upsampling                  | `python -m cosmos_framework.scripts.upsample_prompts`                                        |
| Model export (HF)                  | `python -m cosmos_framework.scripts.export_model`                                            |
| DCP conversion                     | `python -m cosmos_framework.scripts.convert_model_to_dcp`                                    |
| Diffusers conversion               | `python -m cosmos_framework.scripts.convert_model_to_diffusers`                              |
| Video captioning                   | `python -m cosmos_framework.scripts.caption_from_video`                                      |
| Captions → SFT JSONL               | `python -m cosmos_framework.scripts.captions_to_sft_jsonl`                                   |
| Action policy server (LIBERO HTTP) | `python -m cosmos_framework.scripts.action_policy_server_libero`                             |
| Action policy server (RoboLab WS)  | `python -m cosmos_framework.scripts.action_policy_server_robolab`                            |

### Documentation

| Doc                                 | Covers                                                     |
| ----------------------------------- | ---------------------------------------------------------- |
| `../../../AGENTS.md`                | Commands, rules, key file locations (read this first)      |
| `../../../README.md`                | Overview, quickstart, examples                             |
| `../../../docs/setup.md`            | Installation, environment, checkpoints                     |
| `../../../docs/code_structure.md`   | Repo layout and per-subpackage tour of `cosmos_framework/` |
| `../../../docs/inference.md`        | Sample args, default values, custom defaults               |
| `../../../docs/training.md`         | SFT / post-training workflow                               |
| `../../../docs/faq.md`              | FAQ, tips, and troubleshooting                             |
