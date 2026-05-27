---
name: cosmos3-post-training
description: >
  Guide users through Cosmos3 supervised fine-tuning (SFT) post-training:
  preparing the example dataset and Wan2.2 VAE, converting the base
  checkpoint to DCP, launching distributed training (paired launch shell
  recommended, raw `torchrun` as an alternative), running T2V/I2V/V2V
  inference with the trained DCP checkpoint, optionally exporting it to
  Hugging Face safetensors, and running **action evaluation**
  (`cosmos_framework.scripts.eval`) on action checkpoints (forward / inverse
  dynamics, policy) for PSNR / action MSE. Use when the user asks how to
  post-train Cosmos3, fine-tune on a custom video dataset, export a
  trained checkpoint, evaluate an action checkpoint, run `eval.py`, or
  invoke one of the recipe launch shells (`launch_sft_action_fdm_nano.sh`,
  `launch_sft_action_policy_nano.sh`, `launch_sft_vision_nano.sh`,
  `launch_sft_llava_ov.sh`, plus their `_super` LoRA variants)
  — or any question about `cu130-train` / `cu128-train`,
  `convert_model_to_dcp` / `export_model` / `train` / `eval`, action-eval
  metrics, or SFT output paths. `eval.py` is action-only; for T2V/I2V/V2V
  use inference. For dataset captioning / JSONL assembly, see
  `docs/dataset_jsonl.md`.
---

# Cosmos3 Post-Training (SFT)

## When to use this skill

- User wants to fine-tune Cosmos3-Nano (or Cosmos3-Super via LoRA) on a Bridge / LIBERO / custom video dataset (SFT)
- User asks which fields in a recipe TOML to override (`[model.parallelism].data_parallel_shard_degree`, `[dataloader_train].max_samples_per_batch`, `[optimizer].lr`, `[trainer].max_iter`, `[checkpoint].load_path`, ...) or which experiment SKU to pick
- User wants to convert a base Hugging Face checkpoint to DCP, or convert a trained DCP back to safetensors
- User wants to score an **action** checkpoint (forward / inverse dynamics, policy) against a held-out dataset with `cosmos_framework.scripts.eval` — per-sample PSNR / action MSE plus an aggregate. Eval is action-only; do not invoke this skill's eval guidance for T2V/I2V/V2V checkpoints
- For installation, `--group=cu130-train` / `cu128-train`, or LD_LIBRARY_PATH issues, hand off to **cosmos3-setup**
- For inference parameters, parallelism presets, or online serving, hand off to **cosmos3-inference**
- For raw-video captioning or assembling a SFT JSONL, see `docs/dataset_jsonl.md` (the captioning flow has moved out of `docs/training.md`)
- For LIBERO closed-loop policy evaluation (simulator-in-the-loop), see `docs/action_policy_closed_loop_eval.md` — separate from `cosmos_framework.scripts.eval`

## Path convention

All paths below are relative to the cosmos3 package root (`../../../` from this skill file). All `uv run` / `python` / `torchrun` / `bash` commands should also be run from there.

## Where to find answers

The canonical reference is `docs/training.md`. Use this table to route questions:

| User question                                                               | Go to                                                                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Full step-by-step SFT workflow                                              | `docs/training.md`                                                              |
| Which install group? (`cu130-train` vs `cu128-train`)                       | `docs/setup.md` § CUDA Variants                                                 |
| Which recipes exist? (action FDM / Policy / Vision SFT / Reasoner)          | `docs/training.md` § Step 1 - Prepare data and config                           |
| How do I download the example dataset / Wan VAE?                            | `docs/training.md` § Step 1 - Prepare data and config                           |
| How do I convert a base HF checkpoint to DCP?                               | `docs/training.md` § Step 2 — Prepare checkpoint                                |
| How do I launch training (paired shell, recommended)?                       | `docs/training.md` § Step 3 → Generator Post-Training → Option A                |
| How do I launch training with raw `torchrun`?                               | `docs/training.md` § Step 3 → Generator Post-Training → Option B                |
| How do I override `DATASET_PATH` / `BASE_CHECKPOINT_PATH` / `WAN_VAE_PATH`? | `docs/training.md` § Step 3 → Generator Post-Training → Overriding the defaults |
| Which TOML keys are commonly tuned?                                         | `docs/training.md` § Config                                                     |
| LoRA knobs (`lora_enabled` / `lora_rank` / `lora_alpha`)                    | `docs/training.md` § Config — `[model]` block (VFM only)                        |
| How do I validate the config without actually training?                     | `cosmos_framework/scripts/train.py` `--dryrun` flag (not in docs)               |
| How do I export the trained DCP back to safetensors?                        | `docs/training.md` § Export checkpoint to Hugging Face safetensors              |
| How do I run inference with the trained checkpoint?                         | `cosmos3-inference` skill (point at `$RUN_DIR/checkpoints/iter_<N>`)            |
| How do I evaluate an action checkpoint (forward/inverse/policy)?            | `docs/training.md` § Evaluation                                                 |
| How do I run `cosmos_framework.scripts.eval` / `eval.py`?                   | `docs/training.md` § Evaluation                                                 |
| What metrics does action eval report (PSNR, MSE)?                           | `docs/training.md` § Evaluation                                                 |
| Where do training and action-eval artifacts land?                           | `docs/training.md` § Outputs                                                    |
| How do I caption raw videos / build a SFT JSONL?                            | `docs/dataset_jsonl.md`                                                         |
| LIBERO closed-loop policy eval                                              | `docs/action_policy_closed_loop_eval.md`                                        |

## Workflow at a glance

1. **Setup** — install the training extras: `uv sync --all-extras --group=cu130-train` (or `cu128-train` on older drivers), then `source .venv/bin/activate && export LD_LIBRARY_PATH=`.
2. **Step 1 - Prepare data and config** — for the recipe you're running, download the HF dataset to `examples/data/<dataset>/` and the Wan2.2 VAE to `examples/checkpoints/wan22_vae/Wan2.2_VAE.pth` via `uvx hf@latest download …`. The Reasoner recipe streams its dataset from HF Hub at startup — no Step 1 download needed.
3. **Step 2 — Prepare checkpoint** — set `BASE_CHECKPOINT_NAME` (`Cosmos3-Nano` or `Cosmos3-Super`, matching the recipe) and run `python -m cosmos_framework.scripts.convert_model_to_dcp -o examples/checkpoints/$BASE_CHECKPOINT_NAME --checkpoint-path $BASE_CHECKPOINT_NAME`. Skip for the Reasoner recipe (the Qwen3-VL backbone is fetched from HF Hub at startup).
4. **Step 3 — Run training (Option A, recommended)** — from the repo root, `bash examples/launch_sft_<recipe>.sh` (e.g. `launch_sft_action_fdm_nano.sh`). The launcher resolves `DATASET_PATH`, `BASE_CHECKPOINT_PATH`, `WAN_VAE_PATH` from the default `examples/` locations populated by Steps 1+2; export any of them in the shell first to override.
5. **Step 3 — Run training (Option B, raw `torchrun`)** — export the env vars yourself, then `IMAGINAIRE_OUTPUT_ROOT=outputs/train PYTHONPATH=. torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml`. Unlike Option A, raw `torchrun` does NOT auto-resolve the env-var trio from `examples/` — they must come from the shell, or you must hand-edit the TOML to inline literal paths.
6. **Outputs** — `$RUN_DIR = $IMAGINAIRE_OUTPUT_ROOT/<job.project>/<job.group>/<job.name>`. DCP checkpoints land under `$RUN_DIR/checkpoints/iter_<N>/`; the latest iter name is in `$RUN_DIR/checkpoints/latest_checkpoint.txt`. `$RUN_DIR/config.yaml` next to the checkpoints is what inference + eval consume.
7. **Inference** — point `cosmos_framework.scripts.inference` at `$RUN_DIR/checkpoints/iter_<N>` together with `--config-file $RUN_DIR/config.yaml` (see `cosmos3-inference` skill for presets / input formats).
8. **Action evaluation (action checkpoints only)** — `torchrun --nproc-per-node=8 -m cosmos_framework.scripts.eval -o outputs/train_eval --checkpoint-path $RUN_DIR/model --dataset.config-file $RUN_DIR/config.yaml`. Generates each held-out sample through the inference engine and scores against GT — PSNR for vision modes, MSE for action modes. Per-sample `metrics.json` lives next to each `vision.mp4`; rank-0 aggregate is `outputs/train_eval/metrics_aggregate.json`. Skip for T2V/I2V/V2V checkpoints.
9. **Export (optional)** — `python -m cosmos_framework.scripts.export_model --checkpoint-path $RUN_DIR/checkpoints/$(cat $RUN_DIR/checkpoints/latest_checkpoint.txt) --config-file $RUN_DIR/config.yaml -o $RUN_DIR/model` writes a portable HF safetensors checkpoint to `$RUN_DIR/model`.

## Things not obvious from the docs

- **Training extras are a separate group**: SFT requires the `cu130-train` / `cu128-train` install group, not the inference-only `cu130` / `cu128`. Re-running `uv sync` with the wrong group silently leaves training deps uninstalled.
- **Recipe = paired `examples/launch_sft_<r>.sh` + `examples/toml/sft_config/<r>.toml`**: the `.sh` declares `TOML_FILE` directly (full repo-relative path) plus `: "${DATASET_PATH:=…}"` / `: "${BASE_CHECKPOINT_PATH:=…}"` defaults that line up with where Steps 1+2 land, then sources `examples/_sft_launcher_common.sh`. `export`ed values in the user's shell win over the defaults. The helper forwards into `cosmos_framework.scripts.train --sft-toml=$TOML_FILE`, with any `TAIL_OVERRIDES` bash-array entries appended after `--` as Hydra-style `key.path=value` overrides (applied last on top of the pydantic-validated TOML schema in `cosmos_framework/configs/toml_config/sft_config.py`). `MASTER_PORT` defaults to `50012` in the helper; set it in the launcher (or `export`) only if you need to co-launch multiple jobs on one node.
- **Option A (paired launch shell) vs Option B (raw `torchrun`)**: Option A resolves the env-var trio (`DATASET_PATH` / `BASE_CHECKPOINT_PATH` / `WAN_VAE_PATH`) from `examples/` defaults so unset env runs out of the box. Option B requires you to either export those env vars yourself or hand-edit the TOML to inline the paths; the TOMLs use `${ENV:DATASET_PATH}` interpolation that's resolved at TOML load time.
- **`IMAGINAIRE_OUTPUT_ROOT` controls the entire output tree**: setting `IMAGINAIRE_OUTPUT_ROOT=outputs/train` makes everything land under `outputs/train/<job.project>/<job.group>/<job.name>/` (logs, `config.yaml`, `checkpoints/iter_<N>`, callback outputs). Unset, training falls back to `/tmp/imaginaire4-output/...`.
- **W&B is disabled by default**: every recipe TOML sets `[job].wandb_mode = "disabled"`. To log to W&B, flip it to `"online"` in the TOML and export `WANDB_API_KEY` before launching.
- **Inference uses the DCP checkpoint directly**: the standard flow points `cosmos_framework.scripts.inference` at `$RUN_DIR/checkpoints/iter_<N>` together with `--config-file $RUN_DIR/config.yaml`. The Hugging Face safetensors export (`$RUN_DIR/model`) is optional — only needed if you want a portable single-file checkpoint.
- **Parallelism degree must match topology**: in the TOML `[model.parallelism]` block, `data_parallel_shard_degree × data_parallel_replicate_degree × context_parallel_shard_degree` must equal `WORLD_SIZE`. `-1` autoselects `data_parallel_shard_degree` from torchrun world size. Mismatch → FSDP init failure.
- **`--dryrun`**: `cosmos_framework.scripts.train` accepts `--dryrun` to validate the config end-to-end without launching training. Use it whenever iterating on TOML keys or Hydra overrides.
- **`eval.py` is action-only**: `cosmos_framework.scripts.eval` only scores action-mode generations (PSNR for predicted video, MSE for predicted action). It does *not* score T2V/I2V/V2V (vision SFT) checkpoints — those use `cosmos_framework.scripts.inference` (no GT scoring). Pointing `eval.py` at a non-action dataloader fails with "mode requires GT video/action but data_batch had none".
- **Eval flag names are nested under `--dataset.*`**: `--dataset.config-file`, `--dataset.model-mode`, `--dataset.num-samples`, `--dataset.sample-stride`, `--dataset.root-override`, `--dataset.gcs-root-override`, `--dataset.gcs-path-map`, `--dataset.dataset` (key-or-name override). The training docs only call out `--dataset.config-file`, `--dataset.model-mode`, `--dataset.num-samples`; the rest still work.
- **`--dataset.model-mode` defaults to `joint`**: every dataset entry is evaluated under all three action modes — total generation count = `len(modes) × ceil(len(val_split) / sample_stride)`, capped by `--dataset.num-samples`. Restrict during development with `--dataset.num-samples N`, `--dataset.sample-stride K`, or `--dataset.model-mode <single mode>`. Mode is also encoded in each sample's name (`<dataset>/<mode>/<id>`) and is what the metric dispatcher reads back when scoring.
- **Action eval reuses the training dataloader via `--dataset.config-file`**: pointing `eval.py` at `$RUN_DIR/config.yaml` resolves the same dataloader the model was trained against (`val` split by default; falls back to `dataloader_train` when there is no `dataloader_val`). Use `--dataset.root-override /path/to/eval/dataset` to swap in held-out data without editing the config; alternatives are `--dataset.gcs-root-override <s3-uri>` (downloads via `--dataset.cache-dir` / `-c`) and `--dataset.gcs-path-map`.
- **Throughput preset for full-dataset action eval**: `--parallelism-preset` defaults to `latency` (model sharded across all ranks, one sample at a time — required when the checkpoint is too large to fit on a single GPU). When the model fits on one GPU, pass `--parallelism-preset=throughput` so wall-clock scales as `N / num_gpus × per_sample_time` instead of `N × per_sample_time`. Not promoted in `docs/training.md` but still supported.
- **Eval `--checkpoint-path` expects the exported safetensors dir, not the DCP iter**: the command in `docs/training.md` (`--checkpoint-path outputs/train/model`) assumes you ran the export step first. To eval directly off a DCP iter, point `--checkpoint-path` at `$RUN_DIR/checkpoints/iter_<N>` instead.

## Related skills

| Skill                                  | When to use                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| `../cosmos3-setup/SKILL.md`            | Initial install, CUDA variant selection, container/`LD_LIBRARY_PATH` setup   |
| `../cosmos3-inference/SKILL.md`        | Inference parameters, parallelism presets, input JSON format, online serving |
| `../cosmos3-codebase-nav/SKILL.md`     | Locating configs, scripts, and defaults inside the package                   |
| `../cosmos3-env-troubleshoot/SKILL.md` | Debugging environment / runtime errors during training                       |
